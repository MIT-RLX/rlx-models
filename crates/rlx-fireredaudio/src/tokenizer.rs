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

//! HuggingFace `tokenizer.json` wrapper for FireRedAudio ChatML prompts.

use anyhow::{Context, Result, anyhow};
use std::path::Path;
use tokenizers::Tokenizer;

pub struct FireRedTokenizer {
    inner: Tokenizer,
}

impl FireRedTokenizer {
    pub fn from_model_dir(dir: &Path) -> Result<Self> {
        let path = dir.join("tokenizer.json");
        anyhow::ensure!(path.is_file(), "missing tokenizer.json under {dir:?}");
        let inner = Tokenizer::from_file(&path)
            .map_err(|e| anyhow!("load tokenizer.json from {path:?}: {e}"))?;
        Ok(Self { inner })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let enc = self
            .inner
            .encode(text, false)
            .map_err(|e| anyhow!("tokenize: {e}"))?;
        Ok(enc.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.inner
            .decode(ids, true)
            .map_err(|e| anyhow!("detokenize: {e}"))
            .context("FireRedAudio decode")
    }

    /// Token id for a special string, or `None` if missing.
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    pub fn eos_id(&self) -> u32 {
        self.token_to_id("<|im_end|>").unwrap_or(248_044)
    }

    pub fn pad_id(&self) -> u32 {
        self.token_to_id("<|endoftext|>").unwrap_or(248_044)
    }
}
