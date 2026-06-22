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

//! CLIP BPE tokenizer wrapper, reproducing OpenCLIP's `tokenize`:
//! `[sot] bpe(text)[:ctx-2] [eot]`, zero-padded to `context_length`.

use anyhow::{Result, anyhow};
use std::path::Path;
use tokenizers::Tokenizer;

/// `<|startoftext|>` token id.
pub const SOT_TOKEN: u32 = 49406;
/// `<|endoftext|>` token id (also the pooling/EOT marker).
pub const EOT_TOKEN: u32 = 49407;

pub struct ClipTokenizer {
    tk: Tokenizer,
    ctx: usize,
}

impl ClipTokenizer {
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

    /// Tokenize one string into a zero-padded id sequence of length `ctx`.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let enc = self
            .tk
            .encode(text, false)
            .map_err(|e| anyhow!("tokenizing {text:?}: {e}"))?;
        let bpe = enc.get_ids();
        let mut out = vec![0u32; self.ctx];
        out[0] = SOT_TOKEN;
        let n = bpe.len().min(self.ctx.saturating_sub(2));
        out[1..1 + n].copy_from_slice(&bpe[..n]);
        out[1 + n] = EOT_TOKEN;
        Ok(out)
    }
}

/// Index of the EOT token = `argmax(ids)` (OpenCLIP text pooling). Works
/// because EOT (49407) is the largest id and padding is 0.
pub fn eot_index(ids: &[u32]) -> usize {
    let mut best = 0usize;
    let mut best_v = 0u32;
    for (i, &v) in ids.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best
}
