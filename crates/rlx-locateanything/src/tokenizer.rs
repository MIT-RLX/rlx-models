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

//! HuggingFace tokenizer + Qwen-style chat prompt assembly.

use crate::config::LocateAnythingConfig;
use anyhow::Result;
use std::path::Path;

#[cfg(feature = "tokenizer")]
use tokenizers::Tokenizer;

/// Load BPE tokenizer from `vocab.json` + `merges.txt` (or `tokenizer.json` if present).
#[cfg(feature = "tokenizer")]
pub fn load_tokenizer(model_dir: &Path) -> Result<Tokenizer> {
    let json = model_dir.join("tokenizer.json");
    if json.is_file() {
        return Tokenizer::from_file(&json).map_err(|e| anyhow::anyhow!("load {json:?}: {e}"));
    }
    let vocab = model_dir.join("vocab.json");
    let merges = model_dir.join("merges.txt");
    let bpe = tokenizers::models::bpe::BPE::from_file(
        vocab
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("vocab path"))?,
        merges
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("merges path"))?,
    )
    .build()
    .map_err(|e| anyhow::anyhow!("BPE from {vocab:?}+{merges:?}: {e}"))?;
    Ok(Tokenizer::new(bpe))
}

#[cfg(feature = "tokenizer")]
pub fn encode(tokenizer: &Tokenizer, text: &str) -> Result<Vec<u32>> {
    let enc = tokenizer
        .encode(text, false)
        .map_err(|e| anyhow::anyhow!("encode: {e}"))?;
    Ok(enc.get_ids().to_vec())
}

#[cfg(feature = "tokenizer")]
pub fn decode(tokenizer: &Tokenizer, ids: &[u32]) -> Result<String> {
    tokenizer
        .decode(ids, true)
        .map_err(|e| anyhow::anyhow!("decode: {e}"))
}

/// Build user+assistant prompt token ids with `n_image_tokens` vision placeholders.
#[cfg(feature = "tokenizer")]
pub fn build_user_prompt_ids(
    cfg: &LocateAnythingConfig,
    tokenizer: &Tokenizer,
    user_text: &str,
    n_image_tokens: usize,
) -> Result<Vec<u32>> {
    let mut ids = encode(tokenizer, "<|im_start|>user\n")?;
    ids.extend(std::iter::repeat_n(cfg.image_token_index, n_image_tokens));
    ids.extend(encode(tokenizer, user_text)?);
    ids.extend(encode(tokenizer, "\n<|im_start|>assistant\n")?);
    Ok(ids)
}

#[cfg(not(feature = "tokenizer"))]
pub fn load_tokenizer(_model_dir: &Path) -> Result<()> {
    anyhow::bail!("rebuild with --features tokenizer")
}
