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

//! SigLIP 2 text tokenizer — the multilingual Gemma tokenizer.
//!
//! Loading the shipped `tokenizer.json` with the pure-Rust `tokenizers`
//! crate reproduces the HF *fast* tokenizer exactly (same file, same
//! backend): `encode(text, add_special = true)` applies the
//! `TemplateProcessing` post-processor that appends `<eos>` (id 1) and adds
//! **no** `<bos>`. We then right-pad with `<pad>` (id 0) to `context_length`
//! (64) — matching `SiglipProcessor(padding="max_length")`. There is no
//! attention mask (`model_input_names = [input_ids]`); the model attends to
//! every position and pools the **last** one.

use anyhow::{Result, anyhow};
use std::path::Path;
use tokenizers::Tokenizer;

/// `<pad>` token id (right-padding fill).
pub const PAD_TOKEN: u32 = 0;
/// `<eos>` token id (appended by the post-processor).
pub const EOS_TOKEN: u32 = 1;

/// SigLIP text tokenizer: the Gemma fast tokenizer plus fixed-length
/// (`ctx`) right-padding.
pub struct SiglipTokenizer {
    tk: Tokenizer,
    ctx: usize,
}

impl SiglipTokenizer {
    /// Load `tokenizer.json` from a model directory (or a direct file path).
    pub fn from_path(path: &Path, context_length: usize) -> Result<Self> {
        let file = if path.is_dir() {
            path.join("tokenizer.json")
        } else {
            path.to_path_buf()
        };
        let tk =
            Tokenizer::from_file(&file).map_err(|e| anyhow!("loading tokenizer {file:?}: {e}"))?;
        Ok(Self {
            tk,
            ctx: context_length,
        })
    }

    /// Tokenize one string into a `<pad>`-padded id sequence of length `ctx`.
    /// Adds `<eos>` (via the post-processor), then right-pads / right-truncates
    /// to `ctx`.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let enc = self
            .tk
            .encode(text, true)
            .map_err(|e| anyhow!("tokenizing {text:?}: {e}"))?;
        let ids = enc.get_ids();
        let mut out = vec![PAD_TOKEN; self.ctx];
        let n = ids.len().min(self.ctx);
        out[..n].copy_from_slice(&ids[..n]);
        Ok(out)
    }

    /// Pooling index for SigLIP text: always the last position.
    pub fn pool_index(&self) -> usize {
        self.ctx - 1
    }
}
