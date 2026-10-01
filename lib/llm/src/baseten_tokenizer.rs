// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! RocketKV backport of Dynamo v1.5's Baseten tokenizer backend (`DYN_TOKENIZER=basetenkenizer`).
//!
//! Ported from `dynamo-tokenizers` 1.8.0 `src/basetenkenizer.rs` onto the 1.3.2 traits this v1.3.1 fork uses: the
//! same loading (tokenizer.json + special tokens from tokenizer_config.json), encode without added special tokens
//! (as the HF and fastokens backends here), decode by the crate. 1.8.0's `encode_segments` and `TokenizerOptions` do
//! not exist in 1.3.2 and are left out.

use std::path::Path;

use crate::tokenizers::{
    Encoding, Error, Result, TokenIdType,
    traits::{DecodeResult, Decoder, Encoder, Tokenizer},
};

/// Tokenizer backed by the `basetenkenizer` crate.
pub struct BasetenTokenizer {
    tokenizer: basetenkenizer::Tokenizer,
}

impl BasetenTokenizer {
    /// Load a tokenizer from a Hugging Face `tokenizer.json` file.
    pub fn from_file(path: &str) -> Result<Self> {
        let path = Path::new(path);
        let raw = std::fs::read_to_string(path)
            .map_err(|e| Error::msg(format!("Error reading Baseten tokenizer: {e}")))?;
        let mut json: serde_json::Value = serde_json::from_str(&raw)
            .map_err(|e| Error::msg(format!("Error parsing Baseten tokenizer: {e}")))?;
        if let Some(parent) = path.parent() {
            merge_special_tokens_from_config(&mut json, parent);
        }
        let tokenizer = basetenkenizer::Tokenizer::from_json(json)
            .map_err(|e| Error::msg(format!("Error loading Baseten tokenizer: {e}")))?;
        Ok(Self { tokenizer })
    }
}

/// Special tokens declared only in tokenizer_config.json's `added_tokens_decoder` (as 1.8.0 does).
fn merge_special_tokens_from_config(json: &mut serde_json::Value, model_dir: &Path) {
    let Ok(raw) = std::fs::read_to_string(model_dir.join("tokenizer_config.json")) else {
        return;
    };
    let Ok(config) = serde_json::from_str::<serde_json::Value>(&raw) else {
        return;
    };
    let Some(decoder) = config.get("added_tokens_decoder").and_then(serde_json::Value::as_object) else {
        return;
    };
    if json.get("added_tokens").is_none() {
        json["added_tokens"] = serde_json::json!([]);
    }
    let Some(added_tokens) = json.get_mut("added_tokens").and_then(serde_json::Value::as_array_mut) else {
        return;
    };
    for (id, spec) in decoder {
        let (Ok(id), Some(spec)) = (id.parse::<u32>(), spec.as_object()) else {
            continue;
        };
        if spec.get("special").and_then(serde_json::Value::as_bool) != Some(true) {
            continue;
        }
        let Some(content) = spec
            .get("content")
            .and_then(serde_json::Value::as_str)
            .filter(|c| !c.is_empty())
        else {
            continue;
        };
        if let Some(existing) = added_tokens
            .iter_mut()
            .find(|t| t.get("content").and_then(serde_json::Value::as_str) == Some(content))
        {
            existing["special"] = serde_json::Value::Bool(true);
            continue;
        }
        let mut token = serde_json::Map::from_iter([
            ("id".to_string(), serde_json::json!(id)),
            ("content".to_string(), serde_json::json!(content)),
            ("special".to_string(), serde_json::Value::Bool(true)),
        ]);
        for field in ["single_word", "lstrip", "rstrip", "normalized"] {
            let v = spec.get(field).and_then(serde_json::Value::as_bool).unwrap_or(false);
            token.insert(field.to_string(), serde_json::Value::Bool(v));
        }
        added_tokens.push(serde_json::Value::Object(token));
    }
}

impl Encoder for BasetenTokenizer {
    fn encode(&self, input: &str) -> Result<Encoding> {
        let ids = self
            .tokenizer
            .encode_with_special_tokens(input, false)
            .map_err(|e| Error::msg(format!("Baseten tokenizer encode error: {e}")))?;
        Ok(Encoding::Sp(ids))
    }

    fn encode_batch(&self, inputs: &[&str]) -> Result<Vec<Encoding>> {
        self.tokenizer
            .encode_batch(inputs, false)
            .map(|ids| ids.into_iter().map(Encoding::Sp).collect())
            .map_err(|e| Error::msg(format!("Baseten tokenizer batch encode error: {e}")))
    }
}

impl Decoder for BasetenTokenizer {
    fn decode(&self, token_ids: &[TokenIdType], skip_special_tokens: bool) -> Result<DecodeResult> {
        self.tokenizer
            .decode(token_ids, skip_special_tokens)
            .map(DecodeResult::from)
            .map_err(|e| Error::msg(format!("Baseten tokenizer decode error: {e}")))
    }
}

impl Tokenizer for BasetenTokenizer {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizers::HuggingFaceTokenizer;

    /// RocketKV's served model (DeepSeek-R1-Distill-Llama-8B): RK_MODEL_DIR=<RocketKV>/tok/hf, else skipped
    fn model_dir() -> Option<String> {
        std::env::var("RK_MODEL_DIR").ok()
    }

    #[test]
    fn baseten_ids_match_hf_on_the_served_model() {
        let Some(dir) = model_dir() else { return };
        let path = format!("{dir}/tokenizer.json");
        let bt = BasetenTokenizer::from_file(&path).unwrap();
        let hf = HuggingFaceTokenizer::from_file(&path).unwrap();
        let long: String = (0..20_000).map(|i| format!("w{} ", i * 7919 % 50_021)).collect();
        for text in [
            "Hello, world!",
            "<｜User｜>hi<｜end▁of▁sentence｜><｜Assistant｜>I have read the task context.<｜end▁of▁sentence｜>",
            "fn main() {\n    println!(\"{}\", 42);\n}\n\t\t  trailing  ",
            long.as_str(),
        ] {
            assert_eq!(bt.encode(text).unwrap().token_ids(), hf.encode(text).unwrap().token_ids(), "{:.40}", text);
        }
        let ids = hf.encode("Hello, world!").unwrap().token_ids().to_vec();
        assert_eq!(bt.decode(&ids, true).unwrap().as_str(), "Hello, world!");
        let batch = bt.encode_batch(&["Hello", " world"]).unwrap();
        assert_eq!(batch[1].token_ids(), bt.encode(" world").unwrap().token_ids());
    }
}
