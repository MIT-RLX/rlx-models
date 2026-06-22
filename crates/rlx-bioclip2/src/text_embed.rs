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

//! Host-side text token embedding.
//!
//! The token-embedding lookup is performed on the CPU (like the vision
//! conv1 patch stem) rather than as an in-graph `Gather`. This keeps the
//! text graph a pure float pipeline — no integer index tensor — which
//! both matches the vision tower's structure and avoids backends (MLX)
//! that cannot host-eval the index array inside a compiled function.

use crate::config::BioClip2Config;
use anyhow::{Result, ensure};
use rlx_core::weight_map::WeightMap;

/// Text-embed weights extracted from the checkpoint (consumed on host).
pub struct TextEmbedWeights {
    /// `token_embedding.weight` `[vocab · width]`.
    pub token_embedding: Vec<f32>,
    /// `positional_embedding` `[ctx · width]`.
    pub positional: Vec<f32>,
    pub vocab: usize,
    pub width: usize,
    pub ctx: usize,
}

pub(crate) fn extract_text_embed_weights(
    weights: &mut WeightMap,
    cfg: &BioClip2Config,
) -> Result<TextEmbedWeights> {
    let width = cfg.text.width;
    let ctx = cfg.text.context_length;

    let (token_embedding, tok_shape) = weights.take("token_embedding.weight")?;
    ensure!(
        tok_shape.len() == 2 && tok_shape[1] == width,
        "token_embedding.weight expected [vocab, {width}], got {tok_shape:?}"
    );
    let vocab = tok_shape[0];

    let (positional, pos_shape) = weights.take("positional_embedding")?;
    ensure!(
        positional.len() == ctx * width,
        "positional_embedding length {} != ctx*width ({ctx}*{width}); shape={pos_shape:?}",
        positional.len()
    );

    Ok(TextEmbedWeights {
        token_embedding,
        positional,
        vocab,
        width,
        ctx,
    })
}

/// Assemble the text hidden tensor `[ctx · width]` for one token sequence:
/// `hidden[i] = token_embedding[ids[i]] + positional[i]`.
pub fn assemble_text_hidden(pre: &TextEmbedWeights, ids: &[u32]) -> Result<Vec<f32>> {
    let w = pre.width;
    let ctx = pre.ctx;
    ensure!(ids.len() == ctx, "expected {ctx} ids, got {}", ids.len());
    let mut hidden = vec![0f32; ctx * w];
    for (i, &id) in ids.iter().enumerate() {
        let t = id as usize;
        ensure!(
            t < pre.vocab,
            "token id {t} out of range (vocab {})",
            pre.vocab
        );
        let src = &pre.token_embedding[t * w..(t + 1) * w];
        let pos = &pre.positional[i * w..(i + 1) * w];
        let dst = &mut hidden[i * w..(i + 1) * w];
        for k in 0..w {
            dst[k] = src[k] + pos[k];
        }
    }
    Ok(hidden)
}
