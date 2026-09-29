// SPDX-License-Identifier: Apache-2.0
//! Host token cache, policy v2 (RocketKV tok_dynamo.md §1.1, mode B/Bf/C).
//!
//! Lookup/insert/extend rules are dynamo-tokenizers 1.3.2 `cache/l1.rs` (policy v1: deepest hit,
//! a miss inserts at every boundary, a hit inserts the deepest boundary). v2 changes, as in the
//! builders' ARM cache (RocketKV `tok/rust/src/prefix.rs`, `25f6a7f`):
//! - the pipeline is proven composable once (`PrefixCachePolicy`); otherwise every request is one
//!   complete encode, never a cached split, and is counted as neither hit nor miss;
//! - boundaries come from the library's own added-token matcher on every request;
//! - keys are the exact prefix bytes (`Arc<str>`): no hash-collision reuse;
//! - the budget charges key bytes + 4 B per ID.
//!
//! Values live in host DRAM. Moka defers its maintenance (no inline `run_pending_tasks` and no
//! insert lock, unlike the ARM arena cache: tok_dynamo.md M7).
//!
//! Shared verbatim by the Dynamo fork (`lib/llm/src`) and RocketKV `tok/dyn/toksvc` (mode C's
//! service) through `#[path]`, so it depends on std, moka and tokenizers only.

use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

use moka::{notification::RemovalCause, sync::Cache};

use crate::tokenizer_cache_policy::PrefixCachePolicy;

/// What one request did in the cache.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Lookup {
    /// Policy rejected the pipeline: one complete encode (neither hit nor miss).
    #[default]
    Bypass,
    Hit,
    Miss,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Outcome {
    pub lookup: Lookup,
    /// IDs taken from the cache.
    pub reused_tokens: usize,
    /// Input bytes that went through the encoder.
    pub encoded_bytes: usize,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Stats {
    pub hits: u64,
    pub misses: u64,
    pub bypassed: u64,
    pub reused_tokens: u64,
    pub encoded_tokens: u64,
    pub entries: u64,
    pub weighted_bytes: u64,
    pub capacity_bytes: u64,
    pub evictions: u64,
    pub admission_drops: u64,
}

pub struct PrefixCacheV2 {
    policy: Option<PrefixCachePolicy>,
    disabled: Option<String>,
    index: Cache<Arc<str>, Arc<[u32]>>,
    capacity: u64,
    hits: AtomicU64,
    misses: AtomicU64,
    bypassed: AtomicU64,
    reused: AtomicU64,
    encoded: AtomicU64,
    drops: AtomicU64,
    evictions: Arc<AtomicU64>,
}

impl PrefixCacheV2 {
    /// `tokenizer` is the exact instance the segments are encoded with (after Dynamo's
    /// `tokenizer_config.json` special-token merge). `capacity` = key bytes + ID bytes.
    pub fn new(tokenizer: &tokenizers::Tokenizer, capacity: u64) -> Self {
        let (policy, disabled) = match PrefixCachePolicy::new(tokenizer) {
            Ok(policy) => (Some(policy), None),
            Err(reason) => (None, Some(reason)),
        };
        let evictions = Arc::new(AtomicU64::new(0));
        let count = evictions.clone();
        let index = Cache::builder()
            .max_capacity(capacity)
            .weigher(|key: &Arc<str>, ids: &Arc<[u32]>| {
                (key.len() + ids.len() * 4).min(u32::MAX as usize) as u32
            })
            .eviction_listener(move |_, _, cause| {
                if cause == RemovalCause::Size {
                    count.fetch_add(1, Ordering::Relaxed);
                }
            })
            .build();
        Self {
            policy,
            disabled,
            index,
            capacity,
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            bypassed: AtomicU64::new(0),
            reused: AtomicU64::new(0),
            encoded: AtomicU64::new(0),
            drops: AtomicU64::new(0),
            evictions,
        }
    }

    /// `Some(reason)` when prefix reuse is disabled for this tokenizer.
    pub fn disabled_reason(&self) -> Option<&str> {
        self.disabled.as_deref()
    }

    /// Encode `text`; `segment` encodes one chunk with add_special_tokens = false.
    pub fn encode<E>(
        &self,
        text: &str,
        mut segment: impl FnMut(&str) -> Result<Vec<u32>, E>,
    ) -> Result<(Vec<u32>, Outcome), E> {
        let Some(policy) = &self.policy else {
            let ids = segment(text)?;
            self.bypassed.fetch_add(1, Ordering::Relaxed);
            self.encoded.fetch_add(ids.len() as u64, Ordering::Relaxed);
            let outcome = Outcome {
                lookup: Lookup::Bypass,
                reused_tokens: 0,
                encoded_bytes: text.len(),
            };
            return Ok((ids, outcome));
        };
        let boundaries = policy.boundaries(text);
        let mut found = None;
        for &end in boundaries.iter().rev() {
            // Arc<str> equality checks every prefix byte: a hash collision cannot reuse IDs.
            if let Some(ids) = self.index.get(&text[..end]) {
                found = Some((end, ids));
                break;
            }
        }
        let (ids, outcome) = if let Some((prefix, cached)) = found {
            let deepest = *boundaries.last().expect("a hit implies a boundary");
            let mut ids = Vec::with_capacity(cached.len() + (text.len() - prefix) / 2 + 16);
            ids.extend_from_slice(&cached);
            if deepest > prefix {
                ids.extend(segment(&text[prefix..deepest])?);
                self.insert(&text[..deepest], &ids);
                ids.extend(segment(&text[deepest..])?);
            } else {
                ids.extend(segment(&text[prefix..])?);
            }
            self.hits.fetch_add(1, Ordering::Relaxed);
            let outcome = Outcome {
                lookup: Lookup::Hit,
                reused_tokens: cached.len(),
                encoded_bytes: text.len() - prefix,
            };
            (ids, outcome)
        } else {
            let mut ids = Vec::new();
            let mut previous = 0;
            for &end in &boundaries {
                ids.extend(segment(&text[previous..end])?);
                self.insert(&text[..end], &ids);
                previous = end;
            }
            ids.extend(segment(&text[previous..])?);
            self.misses.fetch_add(1, Ordering::Relaxed);
            let outcome = Outcome {
                lookup: Lookup::Miss,
                reused_tokens: 0,
                encoded_bytes: text.len(),
            };
            (ids, outcome)
        };
        self.reused
            .fetch_add(outcome.reused_tokens as u64, Ordering::Relaxed);
        self.encoded.fetch_add(
            (ids.len() - outcome.reused_tokens) as u64,
            Ordering::Relaxed,
        );
        Ok((ids, outcome))
    }

    fn insert(&self, key: &str, ids: &[u32]) {
        let weight = key.len() as u64 + 4 * ids.len() as u64;
        if weight > self.capacity || weight > u32::MAX as u64 {
            self.drops.fetch_add(1, Ordering::Relaxed);
            return;
        }
        self.index.insert(Arc::from(key), Arc::from(ids));
    }

    /// Cumulative counters. `sync` runs Moka's pending maintenance first (exact entries/bytes);
    /// never call it on the request path.
    pub fn stats(&self, sync: bool) -> Stats {
        if sync {
            self.index.run_pending_tasks();
        }
        Stats {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            bypassed: self.bypassed.load(Ordering::Relaxed),
            reused_tokens: self.reused.load(Ordering::Relaxed),
            encoded_tokens: self.encoded.load(Ordering::Relaxed),
            entries: self.index.entry_count(),
            weighted_bytes: self.index.weighted_size(),
            capacity_bytes: self.capacity,
            evictions: self.evictions.load(Ordering::Relaxed),
            admission_drops: self.drops.load(Ordering::Relaxed),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn tokenizer() -> tokenizers::Tokenizer {
        let vocab: HashMap<String, u32> = [
            ("[UNK]".into(), 0),
            ("shared".into(), 1),
            ("private".into(), 2),
            ("one".into(), 3),
            ("two".into(), 4),
        ]
        .into();
        let model = tokenizers::models::wordlevel::WordLevel::builder()
            .vocab(vocab.into_iter().collect())
            .unk_token("[UNK]".into())
            .build()
            .unwrap();
        let mut t = tokenizers::Tokenizer::new(model);
        t.with_pre_tokenizer(Some(
            tokenizers::pre_tokenizers::whitespace::WhitespaceSplit,
        ));
        t.add_special_tokens(&[tokenizers::AddedToken::from("<s>", true)]);
        t
    }

    fn full(t: &tokenizers::Tokenizer, s: &str) -> Result<Vec<u32>, String> {
        t.encode(s, false)
            .map(|e| e.get_ids().to_vec())
            .map_err(|e| e.to_string())
    }

    #[test]
    fn hit_miss_extend_and_parity() {
        let t = tokenizer();
        let c = PrefixCacheV2::new(&t, 4096);
        assert!(c.disabled_reason().is_none());
        let turns = [
            "shared <s>private one",
            "shared <s>private one <s>two",
            "shared <s>private one <s>two <s>one",
            "shared <s>two",
        ];
        for (i, text) in turns.iter().enumerate() {
            let (ids, out) = c.encode(text, |s| full(&t, s)).unwrap();
            assert_eq!(ids, full(&t, text).unwrap(), "{text}");
            assert_eq!(out.lookup, if i == 0 { Lookup::Miss } else { Lookup::Hit });
        }
        let s = c.stats(true);
        assert_eq!((s.hits, s.misses, s.bypassed), (3, 1, 0));
        assert!(s.weighted_bytes <= 4096 && s.entries >= 2);
    }

    #[test]
    fn budget_charges_key_and_ids() {
        let t = tokenizer();
        let c = PrefixCacheV2::new(&t, 4096);
        c.encode("shared <s>private", |s| full(&t, s)).unwrap();
        let s = c.stats(true);
        assert_eq!((s.entries, s.weighted_bytes), (1, "shared <s>".len() as u64 + 2 * 4));
        let c = PrefixCacheV2::new(&t, "shared <s>".len() as u64 + 2 * 4 - 1);
        c.encode("shared <s>private", |s| full(&t, s)).unwrap();
        let s = c.stats(true);
        assert_eq!((s.entries, s.admission_drops), (0, 1));
    }

    #[test]
    fn unsupported_pipeline_bypasses() {
        let mut t = tokenizer();
        t.add_special_tokens(&[tokenizers::AddedToken::from("<x>", true).lstrip(true)]);
        let c = PrefixCacheV2::new(&t, 4096);
        assert!(c.disabled_reason().is_some());
        let text = "shared <s>  <x>private";
        let (ids, out) = c.encode(text, |s| full(&t, s)).unwrap();
        assert_eq!(ids, full(&t, text).unwrap());
        assert_eq!(out.lookup, Lookup::Bypass);
        let s = c.stats(true);
        assert_eq!((s.hits, s.misses, s.bypassed, s.entries), (0, 0, 1, 0));
    }

    #[test]
    fn longer_added_token_wins_over_cached_boundary() {
        let mut t = tokenizer();
        t.add_special_tokens(&[tokenizers::AddedToken::from("<s>x", true)]);
        let c = PrefixCacheV2::new(&t, 4096);
        for text in ["shared <s>private", "shared <s>xprivate", "shared <s>xprivate two", "<s><s>xone"] {
            for _ in 0..2 {
                assert_eq!(c.encode(text, |s| full(&t, s)).unwrap().0, full(&t, text).unwrap());
            }
        }
    }

    #[test]
    fn eviction_keeps_parity_and_bound() {
        let t = tokenizer();
        let c = PrefixCacheV2::new(&t, 64);
        for i in 0..400 {
            let text = format!("shared {i} <s>private one <s>two");
            assert_eq!(c.encode(&text, |s| full(&t, s)).unwrap().0, full(&t, &text).unwrap());
        }
        assert!(c.stats(true).weighted_bytes <= 64);
    }
}
