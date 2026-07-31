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

//! Host-side prefix assembly + full-sequence attention bookkeeping.
//!
//! The prefix is `[image tokens ++ text tokens]`, all in one bidirectional
//! attention block (`att = 0`). Matching the reference prefix embedder against
//! the pinned `transformers` (`get_image_features` returns the raw projector
//! output — no scaling — at commit `dcddb97`):
//! - **image tokens** = SigLIP projector output, used as-is,
//! - **text tokens**  = `embed_tokens[ids] · √hidden`.
//!
//! The suffix (state/action tokens) is appended by the caller; this module
//! also builds the full-sequence `pad` / `att` masks, `position_ids`, and the
//! block-causal additive bias used by every joint layer (batch 1).

use crate::config::{VlashConfig, VlashVariant};
use crate::util::{block_causal_bias, position_ids_from_pad, rope_tables};

/// Assembled prefix (batch 1) + its per-token pad mask.
pub struct Prefix {
    /// `[prefix_len · hidden]` row-major embeddings.
    pub emb: Vec<f32>,
    /// `prefix_len` pad flags (image tokens all true; text per attention mask).
    pub pad: Vec<bool>,
    pub len: usize,
    pub hidden: usize,
}

/// Gather + scale text token embeddings: `embed[ids] · √hidden`.
fn embed_text(embed_weight: &[f32], vocab: usize, hidden: usize, ids: &[i64]) -> Vec<f32> {
    let scale = (hidden as f32).sqrt();
    let mut out = vec![0f32; ids.len() * hidden];
    for (t, &id) in ids.iter().enumerate() {
        let id = id.max(0) as usize;
        debug_assert!(id < vocab, "token id {id} out of vocab {vocab}");
        let row = &embed_weight[id * hidden..(id + 1) * hidden];
        for (d, &w) in row.iter().enumerate() {
            out[t * hidden + d] = w * scale;
        }
    }
    out
}

/// Build the prefix embeddings + pad mask for batch 1.
///
/// - `image_features`: `[num_images · patches · hidden]` raw projector output.
/// - `embed_weight`:   `[vocab · hidden]` Gemma token-embedding table.
/// - `text_ids` / `text_mask`: tokenized prompt + its attention mask.
#[allow(clippy::too_many_arguments)]
pub fn assemble_prefix(
    image_features: &[f32],
    num_images: usize,
    patches: usize,
    hidden: usize,
    embed_weight: &[f32],
    vocab: usize,
    text_ids: &[i64],
    text_mask: &[f32],
) -> Prefix {
    let img_tokens = num_images * patches;
    let text_tokens = text_ids.len();
    let len = img_tokens + text_tokens;

    let mut emb = vec![0f32; len * hidden];
    // Image tokens: raw projector output (get_image_features applies no scaling
    // at the pinned transformers commit).
    emb[..img_tokens * hidden].copy_from_slice(&image_features[..img_tokens * hidden]);
    // Text tokens (scaled by √hidden).
    let text = embed_text(embed_weight, vocab, hidden, text_ids);
    emb[img_tokens * hidden..].copy_from_slice(&text);

    let mut pad = vec![true; img_tokens];
    pad.extend(text_mask.iter().map(|&m| m > 0.5));

    Prefix {
        emb,
        pad,
        len,
        hidden,
    }
}

/// Suffix attention bookkeeping (the `att` block pattern + pad, per variant).
///
/// - π₀:   `att = [1(state), 1(action0), 0…0]`, all real → length `1 + chunk`.
/// - π₀.₅: `att = [1(action0), 0…0]`,           all real → length `chunk`.
pub fn suffix_att_pad(cfg: &VlashConfig) -> (Vec<i32>, Vec<bool>) {
    let chunk = cfg.chunk_size;
    match cfg.variant {
        VlashVariant::Pi0 => {
            let mut att = Vec::with_capacity(1 + chunk);
            att.push(1); // state token opens a block
            att.push(1); // first action token opens a block
            att.extend(std::iter::repeat(0).take(chunk - 1));
            let pad = vec![true; 1 + chunk];
            (att, pad)
        }
        VlashVariant::Pi05 => {
            let mut att = Vec::with_capacity(chunk);
            att.push(1);
            att.extend(std::iter::repeat(0).take(chunk - 1));
            let pad = vec![true; chunk];
            (att, pad)
        }
    }
}

/// Full-sequence attention inputs for the joint stack (batch 1).
pub struct AttnInputs {
    /// RoPE cos table `[seq · head_dim/2]`.
    pub cos: Vec<f32>,
    /// RoPE sin table `[seq · head_dim/2]`.
    pub sin: Vec<f32>,
    /// Block-causal additive bias `[heads · seq · seq]`.
    pub bias: Vec<f32>,
    pub seq: usize,
}

/// Build cos/sin, block-causal bias, and position ids for the concatenated
/// `[prefix ++ suffix]` sequence (prefix `att` is all-zero / bidirectional).
pub fn build_attn_inputs(cfg: &VlashConfig, prefix_pad: &[bool]) -> AttnInputs {
    let (suffix_att, suffix_pad) = suffix_att_pad(cfg);
    let p = prefix_pad.len();

    let mut pad = prefix_pad.to_vec();
    pad.extend(suffix_pad);
    let mut att = vec![0i32; p]; // prefix all bidirectional
    att.extend(suffix_att);

    let position_ids = position_ids_from_pad(&pad);
    let (cos, sin) = rope_tables(cfg.vlm.rope_theta, cfg.head_dim(), &position_ids);
    let bias = block_causal_bias(&pad, &att, cfg.heads());
    AttnInputs {
        cos,
        sin,
        bias,
        seq: pad.len(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn suffix_masks_match_variants() {
        let (att0, pad0) = suffix_att_pad(&VlashConfig::pi0());
        assert_eq!(att0.len(), 51);
        assert_eq!(&att0[0..3], &[1, 1, 0]);
        assert!(pad0.iter().all(|&p| p));

        let (att5, pad5) = suffix_att_pad(&VlashConfig::pi05());
        assert_eq!(att5.len(), 50);
        assert_eq!(&att5[0..2], &[1, 0]);
        assert!(pad5.iter().all(|&p| p));
    }

    #[test]
    fn attn_inputs_have_expected_shapes() {
        let cfg = VlashConfig::pi05();
        let prefix_pad = vec![true; 8]; // tiny prefix
        let a = build_attn_inputs(&cfg, &prefix_pad);
        let seq = 8 + 50;
        assert_eq!(a.seq, seq);
        assert_eq!(a.cos.len(), seq * cfg.head_dim() / 2);
        assert_eq!(a.bias.len(), cfg.heads() * seq * seq);
    }
}
