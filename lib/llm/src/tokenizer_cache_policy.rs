// SPDX-License-Identifier: Apache-2.0
//! Conservative proof of composable HF encodes, shared with the CMM cache.
//!
//! This file must remain identical to RocketKV-upstream/tok/rust/src/cache_policy.rs.
//! A recognized, unconditional added token separates the library's normalization
//! chunks. Only chunk-local pipeline stages and ID-preserving postprocessing are
//! accepted. Unknown configurations use one complete encode instead of caching.
//!
//! Cuts come from the library's own added-token matcher, but not over the whole prompt on
//! every request (tok_dynamo.md Q16: that pass cost as much as encoding the new suffix). A
//! cached prefix carries a restart point; a hit rescans only from there. Why that is exact,
//! for every tokenizer this policy accepts (every added token unnormalized, no lstrip/rstrip/
//! single_word, so the matcher is leftmost-longest, non-overlapping Aho-Corasick on raw bytes):
//!   (1) Scanning resumes at each match end and takes the leftmost-starting, then longest,
//!       candidate. From any position r strictly inside no match of the full scan, a scan of
//!       text[r..] reports exactly the full scan's matches that start at or after r.
//!   (2) Whether a candidate starts at s depends only on text[s..s + L], L = the longest added
//!       token. Two texts sharing their first c bytes therefore have identical matches up to
//!       the first match that starts after c - L.
//!   Hence a restart point of a key K that is <= |K| - L is a restart point of every text that
//!   starts with K. `restart` picks one; a hit rescans from it and confirms that |K| is still a
//!   cut (appended bytes can make a longer token win), and finds the cuts after it.
//! Candidate keys come from a plain scan for every special-token occurrence (a superset of the
//! cuts); only confirmed cuts are used, so hits and IDs equal the full scan's.

use std::collections::HashMap;

use aho_corasick::AhoCorasick;
use serde_json::Value;
use tokenizers::{AddedVocabulary, OffsetReferential, OffsetType, Tokenizer};

pub struct PrefixCachePolicy {
    added: AddedVocabulary,
    specials: HashMap<u32, String>,
    /// Every special token's text, for the candidate scan (overlapping: all occurrences).
    special_finder: AhoCorasick,
    /// Bytes of the longest added token (special or not): L in (2).
    longest: usize,
}

/// The library's matches from one restart point to the end of a text.
pub struct Scan {
    from: usize,
    /// Every added-token match (start, end), absolute byte offsets, in order.
    matches: Vec<(usize, usize)>,
    /// Ends of special-token matches before the end of the text, ascending: the cache cuts.
    pub cuts: Vec<usize>,
}

impl PrefixCachePolicy {
    pub fn new(tokenizer: &Tokenizer) -> Result<Self, String> {
        if tokenizer.get_padding().is_some() || tokenizer.get_truncation().is_some() {
            return Err("padding/truncation applies to the complete input".into());
        }
        if tokenizer.get_encode_special_tokens() {
            return Err("special-token extraction is disabled".into());
        }
        let config: Value = serde_json::from_str(
            &tokenizer
                .to_string(false)
                .map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
        let model = &config["model"];
        let deterministic = match model["type"].as_str() {
            Some("BPE") => model["dropout"].is_null() || model["dropout"].as_f64() == Some(0.0),
            Some("WordLevel" | "WordPiece" | "Unigram") => true,
            _ => false,
        };
        if !deterministic {
            return Err("model is not validated as deterministic".into());
        }
        if !normalizer_safe(&config["normalizer"]) {
            return Err("normalizer is not validated for prefix caching".into());
        }
        if !pretokenizer_safe(&config["pre_tokenizer"]) {
            return Err("pre-tokenizer depends on input position or is not validated".into());
        }
        if !postprocessor_safe(&config["post_processor"]) {
            return Err("postprocessor is not ID-preserving with add_special_tokens=false".into());
        }
        let added = tokenizer.get_added_tokens_decoder();
        // Include ordinary added tokens: they participate in the same matcher and
        // can otherwise hide a shorter special token or depend on adjacent text.
        if added.values().any(|token| {
            token.content.is_empty()
                || token.normalized
                || token.single_word
                || token.lstrip
                || token.rstrip
        }) {
            return Err("added-token normalization/context flags are not validated".into());
        }
        let longest = added.values().map(|token| token.content.len()).max().unwrap_or(0);
        let specials: HashMap<_, _> = added
            .into_iter()
            .filter(|(_, token)| token.special)
            .map(|(id, token)| (id, token.content))
            .collect();
        if specials.is_empty() {
            return Err("no atomic special-token boundaries".into());
        }
        let special_finder =
            AhoCorasick::new(specials.values()).map_err(|error| error.to_string())?;
        Ok(Self {
            added: tokenizer.get_added_vocabulary().clone(),
            specials,
            special_finder,
            longest,
        })
    }

    /// Every cut of the whole text (the library's matcher from position 0).
    pub fn boundaries(&self, text: &str) -> Vec<usize> {
        self.scan(text, 0).cuts
    }

    /// Ends of every special-token occurrence before the end of the text, ascending: a superset
    /// of the cuts, found without the library's matcher. Candidates for a cache lookup only.
    pub fn candidates(&self, text: &str) -> Vec<usize> {
        let mut ends: Vec<usize> = self
            .special_finder
            .find_overlapping_iter(text)
            .map(|found| found.end())
            .filter(|&end| end < text.len())
            .collect();
        ends.sort_unstable();
        ends.dedup();
        ends
    }

    /// The library's matcher on `text[from..]`. `from` must be 0 or a restart point from
    /// `restart` for a prefix of `text`; then the result is the full scan's, from `from` on (1).
    pub fn scan(&self, text: &str, from: usize) -> Scan {
        // Every added token is unnormalized, so normalization is irrelevant to
        // matching. Keeping the HF matcher avoids duplicating its tie breaking.
        let split = self.added.extract_and_normalize(
            None::<&tokenizers::normalizers::NormalizerWrapper>,
            &text[from..],
        );
        let mut matches = Vec::new();
        let mut cuts = Vec::new();
        for (_, (start, end), tokens) in split.get_splits(OffsetReferential::Original, OffsetType::Byte) {
            let Some(tokens) = tokens.as_ref() else { continue };
            let (start, end) = (start + from, end + from);
            matches.push((start, end));
            if tokens.len() != 1 || end >= text.len() {
                continue;
            }
            if let Some(special) = self.specials.get(&tokens[0].id) {
                if text.get(start..end) == Some(special.as_str()) {
                    cuts.push(end);
                }
            }
        }
        Scan { from, matches, cuts }
    }

    /// A restart point for the key `text[..key_len]`, from a scan of `text` that covers it:
    /// at most `key_len - L` and strictly inside no match, so it holds for every text that
    /// starts with the key (module comment). 0 is always one.
    pub fn restart(&self, text: &str, scan: &Scan, key_len: usize) -> usize {
        let Some(mut r) = key_len.checked_sub(self.longest) else { return 0 };
        if r < scan.from {
            return 0;
        }
        while !text.is_char_boundary(r) {
            r -= 1;
        }
        match scan.matches.iter().find(|&&(start, end)| start < r && r < end) {
            Some(&(start, _)) => start,
            None => r,
        }
    }
}

fn all_children(value: &Value, key: &str, check: fn(&Value) -> bool) -> bool {
    value[key]
        .as_array()
        .is_some_and(|children| children.iter().all(check))
}

fn normalizer_safe(value: &Value) -> bool {
    if value.is_null() {
        return true;
    }
    match value["type"].as_str() {
        Some("Sequence") => all_children(value, "normalizers", normalizer_safe),
        // These built-ins transform one NormalizedString, independently of its
        // original offset and of the other added-vocabulary chunks.
        Some(
            "BertNormalizer" | "Bert" | "Strip" | "StripAccents" | "NFC" | "NFD" | "NFKC" | "NFKD"
            | "Lowercase" | "Nmt" | "Precompiled" | "Replace" | "Prepend" | "ByteLevel",
        ) => true,
        _ => false,
    }
}

fn pretokenizer_safe(value: &Value) -> bool {
    if value.is_null() {
        return true;
    }
    match value["type"].as_str() {
        Some("Sequence") => all_children(value, "pretokenizers", pretokenizer_safe),
        Some("Metaspace") => matches!(value["prepend_scheme"].as_str(), Some("always" | "never")),
        // Each of these transforms/splits the existing non-tokenized chunks; it
        // does not branch on their original input offsets or their global index.
        Some(
            "BertPreTokenizer" | "ByteLevel" | "Delimiter" | "Whitespace" | "WhitespaceSplit"
            | "Split" | "Punctuation" | "Digits" | "UnicodeScripts" | "FixedLength",
        ) => true,
        _ => false,
    }
}

fn postprocessor_safe(value: &Value) -> bool {
    if value.is_null() {
        return true;
    }
    match value["type"].as_str() {
        Some("Sequence") => all_children(value, "processors", postprocessor_safe),
        Some("ByteLevel" | "BertProcessing" | "RobertaProcessing") => true,
        Some("TemplateProcessing") => {
            let Some(pieces) = value["single"].as_array() else {
                return false;
            };
            let mut sequences = 0;
            for piece in pieces {
                if let Some(sequence) = piece.get("Sequence") {
                    if sequence["id"] != "A" {
                        return false;
                    }
                    sequences += 1;
                } else if piece.get("SpecialToken").is_none() {
                    return false;
                }
            }
            sequences == 1
        }
        _ => false,
    }
}
