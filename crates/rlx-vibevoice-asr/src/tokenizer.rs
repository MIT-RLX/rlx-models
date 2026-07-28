// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Qwen2 byte-level BPE tokenizer loaded from the shipped `tokenizer.json`.
// Special tokens are spliced by id in `prompt.rs`, so plain-text chunks are
// encoded with `add_special_tokens = false`.

use anyhow::{Result, anyhow};
use std::path::Path;
use tokenizers::Tokenizer;

pub struct VibeTokenizer {
    inner: Tokenizer,
}

impl VibeTokenizer {
    /// Load from a `tokenizer.json` file.
    pub fn from_file(path: &Path) -> Result<Self> {
        let inner = Tokenizer::from_file(path).map_err(|e| anyhow!("load tokenizer.json: {e}"))?;
        Ok(Self { inner })
    }

    /// Encode a plain-text chunk (no special tokens) → token ids.
    pub fn encode_plain(&self, text: &str) -> Vec<i64> {
        match self.inner.encode(text, false) {
            Ok(enc) => enc.get_ids().iter().map(|&i| i as i64).collect(),
            Err(_) => Vec::new(),
        }
    }

    /// Decode token ids → string. `skip_special` drops `<|…|>` markers.
    pub fn decode(&self, ids: &[i64], skip_special: bool) -> String {
        let u: Vec<u32> = ids.iter().map(|&i| i.max(0) as u32).collect();
        self.inner.decode(&u, skip_special).unwrap_or_default()
    }
}
