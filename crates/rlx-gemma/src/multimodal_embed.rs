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

//! Build LM input embeddings for multimodal prefill.

use crate::config::GemmaConfig;
use crate::multimodal::{GemmaMultimodalConfig, fuse_multimodal_embeddings};
use anyhow::{Context, Result, bail};
use std::collections::HashMap;

/// Lookup + Gemma embed-scale (`sqrt(hidden)`) for a token stream.
pub fn embed_token_ids_scaled(
    weights: &HashMap<String, (Vec<f32>, Vec<usize>)>,
    cfg: &GemmaConfig,
    token_ids: &[u32],
) -> Result<Vec<f32>> {
    let key = "model.embed_tokens.weight";
    let (data, shape) = weights
        .get(key)
        .with_context(|| format!("missing {key} in weight cache"))?;
    if shape.len() != 2 {
        bail!("{key}: expected rank-2, got {shape:?}");
    }
    let vocab = shape[0];
    let hidden = shape[1];
    if hidden != cfg.hidden_size {
        bail!("embed hidden {hidden} != config {}", cfg.hidden_size);
    }
    let scale = (hidden as f32).sqrt();
    let mut out = vec![0f32; token_ids.len() * hidden];
    for (i, &tok) in token_ids.iter().enumerate() {
        let t = tok as usize;
        if t >= vocab {
            bail!("token id {tok} out of vocab range {vocab}");
        }
        let src = t * hidden;
        let dst = i * hidden;
        for d in 0..hidden {
            out[dst + d] = data[src + d] * scale;
        }
    }
    Ok(out)
}

/// Text embeddings + splice vision/audio rows at placeholder token ids.
pub fn build_multimodal_inputs_embeds(
    weights: &HashMap<String, (Vec<f32>, Vec<usize>)>,
    cfg: &GemmaConfig,
    mm_cfg: &GemmaMultimodalConfig,
    token_ids: &[u32],
    image_embeds: &[f32],
    audio_embeds: &[f32],
    video_embeds: &[f32],
) -> Result<Vec<f32>> {
    let mut embeds = embed_token_ids_scaled(weights, cfg, token_ids)?;
    fuse_multimodal_embeddings(
        &mut embeds,
        token_ids,
        cfg.hidden_size,
        mm_cfg,
        image_embeds,
        audio_embeds,
        video_embeds,
    )?;
    Ok(embeds)
}
