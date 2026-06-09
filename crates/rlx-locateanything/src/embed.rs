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

//! Fuse vision embeddings into the text embedding stream at `image_token_index`.

use crate::config::LocateAnythingConfig;
use crate::load::LocateAnythingWeightStore;
use crate::weights::LocateAnythingWeightPrefix;
use anyhow::{Result, ensure};
use rlx_core::weight_map::WeightMap;

pub fn fuse_inputs_embeds(
    cfg: &LocateAnythingConfig,
    weights: &WeightMap,
    token_ids: &[u32],
    vision_embeds: &[f32],
) -> Result<Vec<f32>> {
    let h = cfg.text_config.hidden_size;
    let vocab = cfg.text_config.vocab_size;
    let embed_key = LocateAnythingWeightPrefix::lm_embed_tokens();
    let (embed, shape) = weights
        .get(embed_key)
        .ok_or_else(|| anyhow::anyhow!("missing {embed_key}"))?;
    ensure!(
        shape == [vocab, h],
        "unexpected embed shape {shape:?}, expected [{vocab}, {h}]"
    );
    fuse_inputs_embeds_inner(cfg, token_ids, vision_embeds, h, vocab, |tok| {
        let row = &embed[tok as usize * h..(tok as usize + 1) * h];
        Ok(row.to_vec())
    })
}

/// Fuse using mmap row slices for text tokens (no full embedding table in RAM).
pub fn fuse_inputs_embeds_from_store(
    cfg: &LocateAnythingConfig,
    store: &LocateAnythingWeightStore,
    token_ids: &[u32],
    vision_embeds: &[f32],
) -> Result<Vec<f32>> {
    let h = cfg.text_config.hidden_size;
    let vocab = cfg.text_config.vocab_size;
    let rows = store.load_lm_embed_rows_for_tokens(token_ids, vocab, h)?;
    fuse_inputs_embeds_inner(cfg, token_ids, vision_embeds, h, vocab, |tok| {
        rows.get(&tok)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("missing embed row for token {tok}"))
    })
}

fn fuse_inputs_embeds_inner(
    cfg: &LocateAnythingConfig,
    token_ids: &[u32],
    vision_embeds: &[f32],
    h: usize,
    vocab: usize,
    mut row: impl FnMut(u32) -> Result<Vec<f32>>,
) -> Result<Vec<f32>> {
    let seq = token_ids.len();
    let n_image_slots = token_ids
        .iter()
        .filter(|&&id| id == cfg.image_token_index)
        .count();
    let n_image_vecs = vision_embeds.len() / h;
    ensure!(
        n_image_slots == n_image_vecs,
        "image token placeholders ({n_image_slots}) != vision vectors ({n_image_vecs})"
    );

    let mut out = vec![0f32; seq * h];
    let mut img_idx = 0usize;
    for (pos, &tok) in token_ids.iter().enumerate() {
        if tok == cfg.image_token_index {
            let src = &vision_embeds[img_idx * h..(img_idx + 1) * h];
            out[pos * h..(pos + 1) * h].copy_from_slice(src);
            img_idx += 1;
            continue;
        }
        ensure!((tok as usize) < vocab, "token {tok} >= vocab {vocab}");
        let vec = row(tok)?;
        out[pos * h..(pos + 1) * h].copy_from_slice(&vec);
    }
    Ok(out)
}

pub fn argmax_token(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}
