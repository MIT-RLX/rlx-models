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

//! HuggingFace `tokenizer.json` bridge for FLUX.2 (Qwen2 tokenizer files).

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

/// Resolve tokenizer path: explicit, `tokenizer/tokenizer.json`, or sibling `tokenizer.json`.
pub fn resolve_tokenizer_path(model_root: &Path, explicit: Option<&Path>) -> Option<PathBuf> {
    if let Some(p) = explicit {
        return Some(p.to_path_buf());
    }
    crate::paths::find_tokenizer_json(model_root)
}

#[cfg(feature = "flux2-tokenizer")]
pub fn encode_prompt(tokenizer_path: &Path, text: &str) -> Result<Vec<u32>> {
    let data = std::fs::read_to_string(tokenizer_path)
        .with_context(|| format!("read tokenizer {}", tokenizer_path.display()))?;
    let tok: tokenizers::Tokenizer = tokenizers::Tokenizer::from_bytes(data.as_bytes())
        .map_err(|e| anyhow::anyhow!("parse tokenizer.json: {e}"))?;
    let enc = tok
        .encode(text, false)
        .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
    Ok(enc.get_ids().to_vec())
}

#[cfg(not(feature = "flux2-tokenizer"))]
pub fn encode_prompt(_tokenizer_path: &Path, _text: &str) -> Result<Vec<u32>> {
    bail!("tokenizer support not compiled in — rebuild with feature `flux2-tokenizer`")
}

/// Encode and pad/truncate to fixed `seq_len` (pad token id 0).
#[cfg(feature = "flux2-tokenizer")]
pub fn encode_prompt_padded(tokenizer_path: &Path, text: &str, seq_len: usize) -> Result<Vec<u32>> {
    let mut ids = encode_prompt(tokenizer_path, text)?;
    ids.truncate(seq_len);
    if ids.len() < seq_len {
        ids.resize(seq_len, 0);
    }
    Ok(ids)
}

#[cfg(not(feature = "flux2-tokenizer"))]
pub fn encode_prompt_padded(
    _tokenizer_path: &Path,
    _text: &str,
    _seq_len: usize,
) -> Result<Vec<u32>> {
    bail!("tokenizer support not compiled in — rebuild with feature `flux2-tokenizer`")
}
