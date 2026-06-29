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

//! HuggingFace `tokenizer.json` loader for Qwen2.5-VL chat prompts.

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

pub fn load_tokenizer(path: impl AsRef<Path>) -> Result<Tokenizer> {
    Tokenizer::from_file(path.as_ref()).map_err(|e| anyhow::anyhow!("load tokenizer: {e}"))
}

/// Resolve `tokenizer.json` from an HF model directory or GGUF sibling layout.
pub fn resolve_tokenizer_path(model_or_weights: &Path) -> Option<PathBuf> {
    let direct = model_or_weights.join("tokenizer.json");
    if direct.is_file() {
        return Some(direct);
    }
    if model_or_weights.extension().and_then(|s| s.to_str()) == Some("gguf") {
        if let Some(parent) = model_or_weights.parent() {
            let sibling = parent.join("tokenizer.json");
            if sibling.is_file() {
                return Some(sibling);
            }
        }
    }
    None
}

pub fn encode_prompt(tokenizer: &Tokenizer, text: &str) -> Result<Vec<u32>> {
    let enc = tokenizer
        .encode(text, false)
        .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
    Ok(enc.get_ids().to_vec())
}

pub fn decode_token(tokenizer: &Tokenizer, id: u32) -> Result<String> {
    tokenizer
        .decode(&[id], false)
        .map_err(|e| anyhow::anyhow!("decode: {e}"))
        .context("detokenize")
}
