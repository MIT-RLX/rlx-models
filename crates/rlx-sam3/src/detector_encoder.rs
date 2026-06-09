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

//! Native SAM3 detector encoder fusion (pre-norm, 6 layers, d_model=256).
//!
//! Mirrors `sam3.model.encoder.TransformerEncoderFusion` configured by
//! `model_builder._create_transformer_encoder`. Each layer runs:
//!
//!   `tgt2 = norm1(tgt); q=k=tgt2 + pos`
//!   `tgt += self_attn(q, k, v=tgt2, key_padding_mask=src_kpm)`
//!   `tgt2 = norm2(tgt)`
//!   `tgt += cross_attn(q=tgt2, k=v=prompt, key_padding_mask=prompt_kpm)`
//!   `tgt2 = norm3(tgt)`
//!   `tgt += linear2(relu(linear1(tgt2)))`
//!
//! Builder flags (encoder fusion): `pre_norm=True`, `pos_enc_at_attn=True`,
//! `pos_enc_at_cross_attn_keys=False`, `pos_enc_at_cross_attn_queries=False`,
//! `num_feature_levels=1`, `add_pooled_text_to_img_feat=False`.
//! Hence no `level_embed` or `text_pooling_proj` weights are loaded.

use super::detector_decoder::mha_with_bias_maybe_gguf;
use super::tensor::layer_norm;
use rlx_core::weight_map::WeightMap;
use rlx_flow::GgufPackedParams;

use crate::packed_gguf::{linear_maybe_gguf, take_or_gguf, take_transposed_with_gguf_key};
use anyhow::{Result, ensure};

const D_MODEL: usize = 256;
const DIM_FF: usize = 2048;
const N_HEADS: usize = 8;
pub const N_LAYERS: usize = 6;

#[derive(Clone)]
pub struct Sam3EncoderLayerWeights {
    pub self_attn_in_w_t: Vec<f32>,
    pub self_attn_in_b: Vec<f32>,
    pub self_attn_in_gguf_key: Option<String>,
    pub self_attn_out_w_t: Vec<f32>,
    pub self_attn_out_b: Vec<f32>,
    pub self_attn_out_gguf_key: Option<String>,
    pub cross_attn_in_w_t: Vec<f32>,
    pub cross_attn_in_b: Vec<f32>,
    pub cross_attn_in_gguf_key: Option<String>,
    pub cross_attn_out_w_t: Vec<f32>,
    pub cross_attn_out_b: Vec<f32>,
    pub cross_attn_out_gguf_key: Option<String>,
    pub linear1_w_t: Vec<f32>,
    pub linear1_b: Vec<f32>,
    pub linear1_gguf_key: Option<String>,
    pub linear2_w_t: Vec<f32>,
    pub linear2_b: Vec<f32>,
    pub linear2_gguf_key: Option<String>,
    pub norm1_w: Vec<f32>,
    pub norm1_b: Vec<f32>,
    pub norm2_w: Vec<f32>,
    pub norm2_b: Vec<f32>,
    pub norm3_w: Vec<f32>,
    pub norm3_b: Vec<f32>,
}

#[derive(Clone, Default)]
pub struct Sam3EncoderWeights {
    pub loaded: bool,
    /// Checkpoint prefix (`detector.transformer.encoder` or `transformer.encoder`).
    pub prefix: String,
    pub layers: Vec<Sam3EncoderLayerWeights>,
}

pub fn extract_encoder_weights(
    weights: &mut WeightMap,
    gguf_packed: Option<&GgufPackedParams>,
) -> Result<Sam3EncoderWeights> {
    let prefixes = ["detector.transformer.encoder", "transformer.encoder"];
    let base = {
        let mut found = None;
        for p in prefixes {
            let k = format!("{p}.layers.0.self_attn.in_proj_weight");
            if weights.has(&k) {
                found = Some(p);
                break;
            }
        }
        found.ok_or_else(|| anyhow::anyhow!("SAM3 detector encoder not found"))?
    };

    let mut layers = Vec::with_capacity(N_LAYERS);
    for i in 0..N_LAYERS {
        let p = format!("{base}.layers.{i}");
        let (self_attn_in_w_t, self_attn_in_gguf_key) = take_transposed_with_gguf_key(
            weights,
            gguf_packed,
            &format!("{p}.self_attn.in_proj_weight"),
        )?;
        let (self_attn_in_b, _) =
            take_or_gguf(weights, gguf_packed, &format!("{p}.self_attn.in_proj_bias"))?;
        let (self_attn_out_w_t, self_attn_out_gguf_key) = take_transposed_with_gguf_key(
            weights,
            gguf_packed,
            &format!("{p}.self_attn.out_proj.weight"),
        )?;
        let (self_attn_out_b, _) = take_or_gguf(
            weights,
            gguf_packed,
            &format!("{p}.self_attn.out_proj.bias"),
        )?;
        let (cross_attn_in_w_t, cross_attn_in_gguf_key) = take_transposed_with_gguf_key(
            weights,
            gguf_packed,
            &format!("{p}.cross_attn_image.in_proj_weight"),
        )?;
        let (cross_attn_in_b, _) = take_or_gguf(
            weights,
            gguf_packed,
            &format!("{p}.cross_attn_image.in_proj_bias"),
        )?;
        let (cross_attn_out_w_t, cross_attn_out_gguf_key) = take_transposed_with_gguf_key(
            weights,
            gguf_packed,
            &format!("{p}.cross_attn_image.out_proj.weight"),
        )?;
        let (cross_attn_out_b, _) = take_or_gguf(
            weights,
            gguf_packed,
            &format!("{p}.cross_attn_image.out_proj.bias"),
        )?;
        let (linear1_w_t, linear1_gguf_key) =
            take_transposed_with_gguf_key(weights, gguf_packed, &format!("{p}.linear1.weight"))?;
        let (linear1_b, _) = take_or_gguf(weights, gguf_packed, &format!("{p}.linear1.bias"))?;
        let (linear2_w_t, linear2_gguf_key) =
            take_transposed_with_gguf_key(weights, gguf_packed, &format!("{p}.linear2.weight"))?;
        let (linear2_b, _) = take_or_gguf(weights, gguf_packed, &format!("{p}.linear2.bias"))?;
        let (norm1_w, _) = weights.take(&format!("{p}.norm1.weight"))?;
        let (norm1_b, _) = weights.take(&format!("{p}.norm1.bias"))?;
        let (norm2_w, _) = weights.take(&format!("{p}.norm2.weight"))?;
        let (norm2_b, _) = weights.take(&format!("{p}.norm2.bias"))?;
        let (norm3_w, _) = weights.take(&format!("{p}.norm3.weight"))?;
        let (norm3_b, _) = weights.take(&format!("{p}.norm3.bias"))?;
        layers.push(Sam3EncoderLayerWeights {
            self_attn_in_w_t,
            self_attn_in_b,
            self_attn_in_gguf_key,
            self_attn_out_w_t,
            self_attn_out_b,
            self_attn_out_gguf_key,
            cross_attn_in_w_t,
            cross_attn_in_b,
            cross_attn_in_gguf_key,
            cross_attn_out_w_t,
            cross_attn_out_b,
            cross_attn_out_gguf_key,
            linear1_w_t,
            linear1_b,
            linear1_gguf_key,
            linear2_w_t,
            linear2_b,
            linear2_gguf_key,
            norm1_w,
            norm1_b,
            norm2_w,
            norm2_b,
            norm3_w,
            norm3_b,
        });
    }
    Ok(Sam3EncoderWeights {
        loaded: true,
        prefix: base.to_string(),
        layers,
    })
}

/// Run the encoder fusion. `src` is the FPN feature flat in NCHW
/// `[B, C, H, W]`. `src_pos` matches. `prompt` is sequence-first
/// `[L_p, B, C]`. Returns the encoded memory in batch-first flat
/// `[B, H*W, C]`.
#[allow(clippy::too_many_arguments)]
pub fn forward_encoder(
    weights: &Sam3EncoderWeights,
    src_bchw: &[f32],
    src_pos_bchw: &[f32],
    prompt_seq_first: &[f32],
    prompt_kpm: &[u8],
    batch: usize,
    src_h: usize,
    src_w: usize,
    prompt_len: usize,
    gguf_packed: Option<&GgufPackedParams>,
) -> Result<Vec<f32>> {
    ensure!(weights.loaded, "SAM3 detector encoder not loaded");
    ensure!(
        src_bchw.len() == batch * D_MODEL * src_h * src_w,
        "encoder src shape mismatch"
    );
    ensure!(
        prompt_seq_first.len() == prompt_len * batch * D_MODEL,
        "encoder prompt shape mismatch"
    );
    ensure!(
        prompt_kpm.len() == batch * prompt_len,
        "encoder prompt mask shape mismatch"
    );

    let hw = src_h * src_w;

    // Flatten src and pos from NCHW → [B, H*W, C] (batch-first), matching
    // `src.flatten(2).transpose(1, 2)` upstream.
    let mut tgt = vec![0f32; batch * hw * D_MODEL];
    let mut pos = vec![0f32; batch * hw * D_MODEL];
    for b in 0..batch {
        for s in 0..hw {
            for c in 0..D_MODEL {
                tgt[(b * hw + s) * D_MODEL + c] = src_bchw[((b * D_MODEL + c) * hw) + s];
                pos[(b * hw + s) * D_MODEL + c] = src_pos_bchw[((b * D_MODEL + c) * hw) + s];
            }
        }
    }

    // Reorder prompt from [L, B, C] to [B, L, C] for batch-first attention.
    let mut prompt_bf = vec![0f32; batch * prompt_len * D_MODEL];
    for b in 0..batch {
        for l in 0..prompt_len {
            let src = (l * batch + b) * D_MODEL;
            let dst = (b * prompt_len + l) * D_MODEL;
            prompt_bf[dst..dst + D_MODEL].copy_from_slice(&prompt_seq_first[src..src + D_MODEL]);
        }
    }

    for layer in &weights.layers {
        // Pre-norm self-attention with pos added to Q and K.
        let n1 = layer_norm(&tgt, &layer.norm1_w, &layer.norm1_b, D_MODEL, 1e-5)?;
        let mut q = vec![0f32; n1.len()];
        for i in 0..n1.len() {
            q[i] = n1[i] + pos[i];
        }
        let sa = mha_with_bias_maybe_gguf(
            &q,
            &q,
            &n1,
            &layer.self_attn_in_w_t,
            &layer.self_attn_in_b,
            layer.self_attn_in_gguf_key.as_deref(),
            &layer.self_attn_out_w_t,
            &layer.self_attn_out_b,
            layer.self_attn_out_gguf_key.as_deref(),
            gguf_packed,
            batch,
            hw,
            hw,
            D_MODEL,
            N_HEADS,
            None,
            None,
        )?;
        for i in 0..tgt.len() {
            tgt[i] += sa[i];
        }

        // Pre-norm cross-attention to prompt (text). No pos added to Q/K.
        let n2 = layer_norm(&tgt, &layer.norm2_w, &layer.norm2_b, D_MODEL, 1e-5)?;
        let ca = mha_with_bias_maybe_gguf(
            &n2,
            &prompt_bf,
            &prompt_bf,
            &layer.cross_attn_in_w_t,
            &layer.cross_attn_in_b,
            layer.cross_attn_in_gguf_key.as_deref(),
            &layer.cross_attn_out_w_t,
            &layer.cross_attn_out_b,
            layer.cross_attn_out_gguf_key.as_deref(),
            gguf_packed,
            batch,
            hw,
            prompt_len,
            D_MODEL,
            N_HEADS,
            None,
            Some(prompt_kpm),
        )?;
        for i in 0..tgt.len() {
            tgt[i] += ca[i];
        }

        // Pre-norm FFN with ReLU.
        let n3 = layer_norm(&tgt, &layer.norm3_w, &layer.norm3_b, D_MODEL, 1e-5)?;
        let mut ff = linear_maybe_gguf(
            &n3,
            batch * hw,
            D_MODEL,
            &layer.linear1_w_t,
            layer.linear1_gguf_key.as_deref(),
            gguf_packed,
            DIM_FF,
            &layer.linear1_b,
        )?;
        for v in ff.iter_mut() {
            if *v < 0.0 {
                *v = 0.0;
            }
        }
        let ffn = linear_maybe_gguf(
            &ff,
            batch * hw,
            DIM_FF,
            &layer.linear2_w_t,
            layer.linear2_gguf_key.as_deref(),
            gguf_packed,
            D_MODEL,
            &layer.linear2_b,
        )?;
        for i in 0..tgt.len() {
            tgt[i] += ffn[i];
        }
    }

    Ok(tgt)
}
