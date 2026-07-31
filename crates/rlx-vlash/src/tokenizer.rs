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

//! PaliGemma (`google/paligemma-3b-pt-224`) prompt tokenization.
//!
//! VLASH tokenizes only the *task text* (with a trailing `\n`, matching the
//! PaliGemma prefix convention); images enter the prefix as SigLIP patch
//! tokens, not via `<image>` placeholders. `add_special_tokens=True` prepends
//! `<bos>` via the tokenizer's post-processor. The reference uses right
//! padding — `padding="longest"` for single-shot inference (no padding at
//! batch=1), `padding="max_length"` (to `tokenizer_max_length`) for the padded
//! training path.

use anyhow::{Context, Result};
use std::path::Path;
use tokenizers::Tokenizer;

/// PaliGemma `<pad>` token id.
pub const PAD_ID: i64 = 0;

/// Tokenized prompt: right-padded token ids + a boolean attention mask
/// (`1.0` real, `0.0` pad).
pub struct TokenizedPrompt {
    pub ids: Vec<i64>,
    pub mask: Vec<f32>,
}

/// Thin wrapper around the PaliGemma `tokenizer.json`.
pub struct PaligemmaTokenizer {
    inner: Tokenizer,
}

impl PaligemmaTokenizer {
    /// Load `tokenizer.json` from a file path.
    pub fn from_file(path: &Path) -> Result<Self> {
        let inner = Tokenizer::from_file(path)
            .map_err(|e| anyhow::anyhow!("load tokenizer {}: {e}", path.display()))?;
        Ok(Self { inner })
    }

    /// Load `tokenizer.json` from a model directory.
    pub fn from_dir(dir: &Path) -> Result<Self> {
        let p = dir.join("tokenizer.json");
        Self::from_file(&p).with_context(|| format!("tokenizer.json in {}", dir.display()))
    }

    /// Encode `task` (a trailing `\n` is appended if absent) with `<bos>`.
    ///
    /// If `pad_to` is `Some(n)` the output is right-padded/truncated to `n`;
    /// otherwise the natural length is returned (single-shot inference).
    pub fn encode(&self, task: &str, pad_to: Option<usize>) -> Result<TokenizedPrompt> {
        let text = if task.ends_with('\n') {
            task.to_string()
        } else {
            format!("{task}\n")
        };
        let enc = self
            .inner
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
        let mut ids: Vec<i64> = enc.get_ids().iter().map(|&i| i as i64).collect();
        let mut mask: Vec<f32> = vec![1.0; ids.len()];
        if let Some(n) = pad_to {
            if ids.len() > n {
                ids.truncate(n);
                mask.truncate(n);
            } else {
                while ids.len() < n {
                    ids.push(PAD_ID);
                    mask.push(0.0);
                }
            }
        }
        Ok(TokenizedPrompt { ids, mask })
    }
}
