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

//! Tokenizer + HF plain-prompt assembly.

use crate::preprocess::PreprocessedImage;
use anyhow::{Context, Result};
use std::path::Path;
use tokenizers::Tokenizer;

pub fn load_tokenizer(model_dir: &Path) -> Result<Tokenizer> {
    let json = model_dir.join("tokenizer.json");
    Tokenizer::from_file(&json).map_err(|e| anyhow::anyhow!("load {json:?}: {e}"))
}

pub fn encode(model_dir: &Path, text: &str) -> Result<Vec<u32>> {
    let tok = load_tokenizer(model_dir)?;
    let enc = tok
        .encode(text, false)
        .map_err(|e| anyhow::anyhow!("encode: {e}"))?;
    Ok(enc.get_ids().to_vec())
}

pub fn decode(model_dir: &Path, ids: &[u32]) -> Result<String> {
    let tok = load_tokenizer(model_dir)?;
    tok.decode(ids, false)
        .map_err(|e| anyhow::anyhow!("decode: {e}"))
        .with_context(|| format!("decode {} ids", ids.len()))
}

/// HF plain prompt: split on `<image>`, insert placeholder spans, prepend BOS.
pub fn build_prompt_ids(
    model_dir: &Path,
    prompt: &str,
    images: &[PreprocessedImage],
    bos_id: u32,
    image_token_id: u32,
) -> Result<Vec<u32>> {
    let marker = "<image>";
    if !prompt.contains(marker) {
        let mut ids = vec![bos_id];
        ids.extend(image_token_span(images, image_token_id));
        ids.extend(encode(model_dir, prompt)?);
        return Ok(ids);
    }
    let parts: Vec<&str> = prompt.split(marker).collect();
    let mut ids = vec![bos_id];
    for (i, part) in parts.iter().enumerate() {
        if !part.is_empty() {
            ids.extend(encode(model_dir, part)?);
        }
        if i + 1 < parts.len() {
            // Single marker holds all pages' spans (HF infer_multi).
            if parts.len() == 2 || images.len() == 1 {
                ids.extend(image_token_span(images, image_token_id));
            } else if let Some(img) = images.get(i) {
                ids.extend(img.image_token_ids(
                    image_token_id,
                    crate::config::PATCH_SIZE,
                    crate::config::DOWNSAMPLE_RATIO,
                ));
            }
        }
    }
    Ok(ids)
}

fn image_token_span(images: &[PreprocessedImage], image_token_id: u32) -> Vec<u32> {
    let mut out = Vec::new();
    for img in images {
        out.extend(img.image_token_ids(
            image_token_id,
            crate::config::PATCH_SIZE,
            crate::config::DOWNSAMPLE_RATIO,
        ));
    }
    out
}
