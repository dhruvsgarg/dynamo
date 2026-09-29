// SPDX-License-Identifier: Apache-2.0
//! Conservative proof of composable HF encodes, shared with the CMM cache.
//!
//! This file must remain identical to RocketKV-upstream/tok/rust/src/cache_policy.rs.
//! A recognized, unconditional added token separates the library's normalization
//! chunks. Only chunk-local pipeline stages and ID-preserving postprocessing are
//! accepted. Unknown configurations use one complete encode instead of caching.

use std::collections::HashMap;

use serde_json::Value;
use tokenizers::{AddedVocabulary, OffsetReferential, OffsetType, Tokenizer};

pub struct PrefixCachePolicy {
    added: AddedVocabulary,
    specials: HashMap<u32, String>,
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
        let specials: HashMap<_, _> = added
            .into_iter()
            .filter(|(_, token)| token.special)
            .map(|(id, token)| (id, token.content))
            .collect();
        if specials.is_empty() {
            return Err("no atomic special-token boundaries".into());
        }
        Ok(Self {
            added: tokenizer.get_added_vocabulary().clone(),
            specials,
        })
    }

    /// Recompute the library's actual full-input matches on every request. A
    /// formerly recognized short token is not reused when appended bytes make a
    /// longer token win. Ordinary added tokens also participate in this decision.
    pub fn boundaries(&self, text: &str) -> Vec<usize> {
        // Every added token is unnormalized, so normalization is irrelevant to
        // matching. Keeping the HF matcher avoids duplicating its tie breaking.
        let split = self
            .added
            .extract_and_normalize(None::<&tokenizers::normalizers::NormalizerWrapper>, text);
        split
            .get_splits(OffsetReferential::Original, OffsetType::Byte)
            .into_iter()
            .filter_map(|(_, (start, end), tokens)| {
                let tokens = tokens.as_ref()?;
                if tokens.len() != 1 || end >= text.len() {
                    return None;
                }
                let special = self.specials.get(&tokens[0].id)?;
                (text.get(start..end) == Some(special.as_str())).then_some(end)
            })
            .collect()
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
