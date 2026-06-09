// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! HuggingFace tokenizer wrapper for BERT-style text tokenization.
//!
//! Handles loading tokenizer files, configuring padding/truncation,
//! and batch encoding of text inputs.

use std::path::Path;

use tokenizers::{AddedToken, PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};

/// Output of batch tokenization: token IDs, attention masks, and token type IDs.
pub struct TokenizedBatch {
    pub input_ids: Vec<Vec<u32>>,
    pub attention_mask: Vec<Vec<u32>>,
    pub token_type_ids: Vec<Vec<u32>>,
    /// Sequence length (max length in this batch after padding).
    pub seq_len: usize,
}

/// Wrapper around HuggingFace tokenizer configured for BERT-style encoding.
pub struct BertTokenizer {
    inner: Tokenizer,
}

impl BertTokenizer {
    /// Load tokenizer from a model directory containing:
    /// - `tokenizer.json`
    /// - `config.json`
    /// - `special_tokens_map.json`
    /// - `tokenizer_config.json`
    pub fn from_dir(dir: &Path, max_length: usize) -> anyhow::Result<Self> {
        let tokenizer_json = std::fs::read(dir.join("tokenizer.json"))?;
        let config_json = std::fs::read(dir.join("config.json"))?;
        let special_tokens_map = std::fs::read(dir.join("special_tokens_map.json"))?;
        let tokenizer_config = std::fs::read(dir.join("tokenizer_config.json"))?;

        Self::from_bytes(
            &tokenizer_json,
            &config_json,
            &special_tokens_map,
            &tokenizer_config,
            max_length,
        )
    }

    /// Load tokenizer from raw file bytes.
    pub fn from_bytes(
        tokenizer_json: &[u8],
        config_json: &[u8],
        special_tokens_map_json: &[u8],
        tokenizer_config_json: &[u8],
        max_length: usize,
    ) -> anyhow::Result<Self> {
        let mut tokenizer = Tokenizer::from_bytes(tokenizer_json)
            .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {e}"))?;

        // Parse config files
        let config: serde_json::Value = serde_json::from_slice(config_json)?;
        let tokenizer_config: serde_json::Value = serde_json::from_slice(tokenizer_config_json)?;
        let special_tokens_map: serde_json::Value =
            serde_json::from_slice(special_tokens_map_json)?;

        // Determine max length from tokenizer_config
        let model_max_length = tokenizer_config
            .get("model_max_length")
            .and_then(|v| v.as_f64())
            .map(|v| v.min(1e9) as usize)
            .unwrap_or(512);
        let effective_max_length = max_length.min(model_max_length);

        // Determine pad token and id
        let pad_token = tokenizer_config
            .get("pad_token")
            .and_then(|v| v.as_str())
            .unwrap_or("[PAD]")
            .to_string();
        let pad_token_id = config
            .get("pad_token_id")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32;

        // Configure padding: pad to longest in batch
        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            pad_token: pad_token.clone(),
            pad_id: pad_token_id,
            ..PaddingParams::default()
        }));

        // Configure truncation
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: effective_max_length,
                ..TruncationParams::default()
            }))
            .map_err(|e| anyhow::anyhow!("failed to set truncation: {e}"))?;

        // Add special tokens from special_tokens_map
        let mut special_tokens = Vec::new();
        if let Some(map) = special_tokens_map.as_object() {
            for (_key, value) in map {
                match value {
                    serde_json::Value::String(s) => {
                        special_tokens.push(AddedToken::from(s.clone(), true));
                    }
                    serde_json::Value::Object(obj) => {
                        if let Some(content) = obj.get("content").and_then(|v| v.as_str()) {
                            special_tokens.push(AddedToken::from(content.to_string(), true));
                        }
                    }
                    serde_json::Value::Array(arr) => {
                        for item in arr {
                            match item {
                                serde_json::Value::String(s) => {
                                    special_tokens.push(AddedToken::from(s.clone(), true));
                                }
                                serde_json::Value::Object(obj) => {
                                    if let Some(content) =
                                        obj.get("content").and_then(|v| v.as_str())
                                    {
                                        special_tokens
                                            .push(AddedToken::from(content.to_string(), true));
                                    }
                                }
                                _ => {}
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        if !special_tokens.is_empty() {
            tokenizer.add_special_tokens(&special_tokens);
        }

        Ok(Self { inner: tokenizer })
    }

    /// Tokenize a batch of texts.
    ///
    /// Returns input_ids, attention_mask, and token_type_ids for each text,
    /// all padded to the same length (longest in batch).
    pub fn encode_batch(&self, texts: &[&str]) -> anyhow::Result<TokenizedBatch> {
        let encodings = self
            .inner
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| anyhow::anyhow!("tokenization failed: {e}"))?;

        let seq_len = encodings
            .first()
            .ok_or_else(|| anyhow::anyhow!("empty batch"))?
            .len();

        let mut input_ids = Vec::with_capacity(texts.len());
        let mut attention_mask = Vec::with_capacity(texts.len());
        let mut token_type_ids = Vec::with_capacity(texts.len());

        for enc in &encodings {
            input_ids.push(enc.get_ids().to_vec());
            attention_mask.push(enc.get_attention_mask().to_vec());
            token_type_ids.push(enc.get_type_ids().to_vec());
        }

        Ok(TokenizedBatch {
            input_ids,
            attention_mask,
            token_type_ids,
            seq_len,
        })
    }
}
