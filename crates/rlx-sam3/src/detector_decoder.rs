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

//! Native SAM3 detector decoder (6 layers, 200 queries + presence token,
//! box refinement, log boxRPB, text + image cross-attention).
//!
//! Mirrors `sam3.model.decoder.TransformerDecoder` configured by
//! `model_builder._create_transformer_decoder`. Inference-time settings:
//!
//!   * `apply_dac=False` (DAC is training-only)
//!   * `presence_token=True` (extra +1 query prepended in self-attn)
//!   * `box_refine=True`, `boxRPB="log"`, `return_intermediate=True`
//!   * `use_text_cross_attention=True`
//!   * `num_queries=200`, `d_model=256`, `n_heads=8`, `dim_ff=2048`

use super::tensor::{layer_norm, linear};
use anyhow::{Result, ensure};
use rlx_core::weight_map::WeightMap;
use rlx_flow::GgufPackedParams;

use crate::packed_gguf::{linear_maybe_gguf, take_or_gguf, take_transposed_with_gguf_key};

const D_MODEL: usize = 256;
const DIM_FF: usize = 2048;
const N_HEADS: usize = 8;
const N_LAYERS: usize = 6;
const NUM_QUERIES: usize = 200;

#[derive(Clone)]
pub struct Sam3DecoderLayerWeights {
    pub self_attn_in_w_t: Vec<f32>,
    pub self_attn_in_b: Vec<f32>,
    pub self_attn_in_gguf_key: Option<String>,
    pub self_attn_out_w_t: Vec<f32>,
    pub self_attn_out_b: Vec<f32>,
    pub self_attn_out_gguf_key: Option<String>,
    pub ca_text_in_w_t: Vec<f32>,
    pub ca_text_in_b: Vec<f32>,
    pub ca_text_in_gguf_key: Option<String>,
    pub ca_text_out_w_t: Vec<f32>,
    pub ca_text_out_b: Vec<f32>,
    pub ca_text_out_gguf_key: Option<String>,
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
    pub norm1_w: Vec<f32>, // post image cross-attn
    pub norm1_b: Vec<f32>,
    pub norm2_w: Vec<f32>, // post self-attn
    pub norm2_b: Vec<f32>,
    pub norm3_w: Vec<f32>, // post FFN
    pub norm3_b: Vec<f32>,
    pub catext_norm_w: Vec<f32>, // post text cross-attn
    pub catext_norm_b: Vec<f32>,
}

#[derive(Clone, Default)]
pub struct Sam3DecoderWeights {
    pub loaded: bool,
    /// Checkpoint prefix (`detector.transformer.decoder`).
    pub prefix: String,
    pub layers: Vec<Sam3DecoderLayerWeights>,
    pub query_embed: Vec<f32>,      // [num_queries, D]
    pub reference_points: Vec<f32>, // [num_queries, 4]
    pub norm_w: Vec<f32>,
    pub norm_b: Vec<f32>,
    pub bbox_embed: Mlp3,          // 256→256→256→4
    pub ref_point_head: Mlp2,      // 512→256→256
    pub boxrpb_x: Mlp2,            // 2→256→n_heads
    pub boxrpb_y: Mlp2,            // 2→256→n_heads
    pub presence_token: Vec<f32>,  // [1, D]
    pub presence_token_head: Mlp3, // 256→256→256→1
    pub presence_token_out_norm_w: Vec<f32>,
    pub presence_token_out_norm_b: Vec<f32>,
}

#[derive(Clone, Default)]
pub struct Mlp2 {
    pub w0_t: Vec<f32>,
    pub b0: Vec<f32>,
    pub w1_t: Vec<f32>,
    pub b1: Vec<f32>,
    pub in_dim: usize,
    pub hidden: usize,
    pub out_dim: usize,
    pub w0_gguf_key: Option<String>,
    pub w1_gguf_key: Option<String>,
}

#[derive(Clone, Default)]
pub struct Mlp3 {
    pub w0_t: Vec<f32>,
    pub b0: Vec<f32>,
    pub w1_t: Vec<f32>,
    pub b1: Vec<f32>,
    pub w2_t: Vec<f32>,
    pub b2: Vec<f32>,
    pub in_dim: usize,
    pub hidden: usize,
    pub out_dim: usize,
    pub w0_gguf_key: Option<String>,
    pub w1_gguf_key: Option<String>,
    pub w2_gguf_key: Option<String>,
}

pub fn take_mlp2(
    weights: &mut WeightMap,
    gguf_packed: Option<&GgufPackedParams>,
    base: &str,
    in_dim: usize,
    hidden: usize,
    out_dim: usize,
) -> Result<Mlp2> {
    let (w0_t, w0_gguf_key) =
        take_transposed_with_gguf_key(weights, gguf_packed, &format!("{base}.layers.0.weight"))?;
    let (b0, _) = take_or_gguf(weights, gguf_packed, &format!("{base}.layers.0.bias"))?;
    let (w1_t, w1_gguf_key) =
        take_transposed_with_gguf_key(weights, gguf_packed, &format!("{base}.layers.1.weight"))?;
    let (b1, _) = take_or_gguf(weights, gguf_packed, &format!("{base}.layers.1.bias"))?;
    Ok(Mlp2 {
        w0_t,
        b0,
        w1_t,
        b1,
        in_dim,
        hidden,
        out_dim,
        w0_gguf_key,
        w1_gguf_key,
    })
}

pub fn take_mlp3(
    weights: &mut WeightMap,
    gguf_packed: Option<&GgufPackedParams>,
    base: &str,
    in_dim: usize,
    hidden: usize,
    out_dim: usize,
) -> Result<Mlp3> {
    let (w0_t, w0_gguf_key) =
        take_transposed_with_gguf_key(weights, gguf_packed, &format!("{base}.layers.0.weight"))?;
    let (b0, _) = take_or_gguf(weights, gguf_packed, &format!("{base}.layers.0.bias"))?;
    let (w1_t, w1_gguf_key) =
        take_transposed_with_gguf_key(weights, gguf_packed, &format!("{base}.layers.1.weight"))?;
    let (b1, _) = take_or_gguf(weights, gguf_packed, &format!("{base}.layers.1.bias"))?;
    let (w2_t, w2_gguf_key) =
        take_transposed_with_gguf_key(weights, gguf_packed, &format!("{base}.layers.2.weight"))?;
    let (b2, _) = take_or_gguf(weights, gguf_packed, &format!("{base}.layers.2.bias"))?;
    Ok(Mlp3 {
        w0_t,
        b0,
        w1_t,
        b1,
        w2_t,
        b2,
        in_dim,
        hidden,
        out_dim,
        w0_gguf_key,
        w1_gguf_key,
        w2_gguf_key,
    })
}

pub fn mlp2_forward(
    mlp: &Mlp2,
    x: &[f32],
    rows: usize,
    gguf_packed: Option<&GgufPackedParams>,
) -> Result<Vec<f32>> {
    let mut h = linear_maybe_gguf(
        x,
        rows,
        mlp.in_dim,
        &mlp.w0_t,
        mlp.w0_gguf_key.as_deref(),
        gguf_packed,
        mlp.hidden,
        &mlp.b0,
    )?;
    for v in h.iter_mut() {
        if *v < 0.0 {
            *v = 0.0;
        }
    }
    linear_maybe_gguf(
        &h,
        rows,
        mlp.hidden,
        &mlp.w1_t,
        mlp.w1_gguf_key.as_deref(),
        gguf_packed,
        mlp.out_dim,
        &mlp.b1,
    )
}

/// Apply the decoder's `bbox_embed` MLP to `[rows, D]` and return `[rows, 4]` deltas.
pub fn bbox_embed_forward(
    weights: &Sam3DecoderWeights,
    x: &[f32],
    rows: usize,
    gguf_packed: Option<&GgufPackedParams>,
) -> Result<Vec<f32>> {
    mlp3_forward(&weights.bbox_embed, x, rows, gguf_packed)
}

pub fn mlp3_forward(
    mlp: &Mlp3,
    x: &[f32],
    rows: usize,
    gguf_packed: Option<&GgufPackedParams>,
) -> Result<Vec<f32>> {
    let mut h = linear_maybe_gguf(
        x,
        rows,
        mlp.in_dim,
        &mlp.w0_t,
        mlp.w0_gguf_key.as_deref(),
        gguf_packed,
        mlp.hidden,
        &mlp.b0,
    )?;
    for v in h.iter_mut() {
        if *v < 0.0 {
            *v = 0.0;
        }
    }
    h = linear_maybe_gguf(
        &h,
        rows,
        mlp.hidden,
        &mlp.w1_t,
        mlp.w1_gguf_key.as_deref(),
        gguf_packed,
        mlp.hidden,
        &mlp.b1,
    )?;
    for v in h.iter_mut() {
        if *v < 0.0 {
            *v = 0.0;
        }
    }
    linear_maybe_gguf(
        &h,
        rows,
        mlp.hidden,
        &mlp.w2_t,
        mlp.w2_gguf_key.as_deref(),
        gguf_packed,
        mlp.out_dim,
        &mlp.b2,
    )
}

/// In-place mlp2: `out = w1·relu(w0·x + b0) + b1`. Uses BLAS when weights are F32.
pub fn mlp2_forward_into(
    mlp: &Mlp2,
    x: &[f32],
    rows: usize,
    hidden: &mut [f32],
    out: &mut [f32],
    gguf_packed: Option<&GgufPackedParams>,
) -> Result<()> {
    if mlp.w0_gguf_key.is_none() && !mlp.w0_t.is_empty() {
        rlx_cpu::blas::sgemm_bias_epilogue(
            x,
            &mlp.w0_t,
            &mlp.b0,
            hidden,
            rows,
            mlp.in_dim,
            mlp.hidden,
            |v| if v < 0.0 { 0.0 } else { v },
        );
        rlx_cpu::blas::sgemm_bias(
            hidden,
            &mlp.w1_t,
            &mlp.b1,
            out,
            rows,
            mlp.hidden,
            mlp.out_dim,
        );
        return Ok(());
    }
    let mut h = linear_maybe_gguf(
        x,
        rows,
        mlp.in_dim,
        &mlp.w0_t,
        mlp.w0_gguf_key.as_deref(),
        gguf_packed,
        mlp.hidden,
        &mlp.b0,
    )?;
    for v in h.iter_mut() {
        if *v < 0.0 {
            *v = 0.0;
        }
    }
    hidden.copy_from_slice(&h);
    let h2 = linear_maybe_gguf(
        hidden,
        rows,
        mlp.hidden,
        &mlp.w1_t,
        mlp.w1_gguf_key.as_deref(),
        gguf_packed,
        mlp.out_dim,
        &mlp.b1,
    )?;
    out.copy_from_slice(&h2);
    Ok(())
}

/// In-place mlp3 into caller buffers (no alloc when F32 + BLAS).
pub fn mlp3_forward_into(
    mlp: &Mlp3,
    x: &[f32],
    rows: usize,
    h0: &mut [f32],
    h1: &mut [f32],
    out: &mut [f32],
    gguf_packed: Option<&GgufPackedParams>,
) -> Result<()> {
    if mlp.w0_gguf_key.is_none() && !mlp.w0_t.is_empty() {
        let relu = |v: f32| if v < 0.0 { 0.0 } else { v };
        rlx_cpu::blas::sgemm_bias_epilogue(
            x, &mlp.w0_t, &mlp.b0, h0, rows, mlp.in_dim, mlp.hidden, relu,
        );
        rlx_cpu::blas::sgemm_bias_epilogue(
            h0, &mlp.w1_t, &mlp.b1, h1, rows, mlp.hidden, mlp.hidden, relu,
        );
        rlx_cpu::blas::sgemm_bias(h1, &mlp.w2_t, &mlp.b2, out, rows, mlp.hidden, mlp.out_dim);
        return Ok(());
    }
    let o = mlp3_forward(mlp, x, rows, gguf_packed)?;
    out.copy_from_slice(&o);
    Ok(())
}

pub fn extract_decoder_weights(
    weights: &mut WeightMap,
    gguf_packed: Option<&GgufPackedParams>,
) -> Result<Sam3DecoderWeights> {
    let base = "detector.transformer.decoder";
    ensure!(
        weights.has(&format!("{base}.query_embed.weight")),
        "SAM3 detector decoder not found"
    );

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
        let (ca_text_in_w_t, ca_text_in_gguf_key) = take_transposed_with_gguf_key(
            weights,
            gguf_packed,
            &format!("{p}.ca_text.in_proj_weight"),
        )?;
        let (ca_text_in_b, _) =
            take_or_gguf(weights, gguf_packed, &format!("{p}.ca_text.in_proj_bias"))?;
        let (ca_text_out_w_t, ca_text_out_gguf_key) = take_transposed_with_gguf_key(
            weights,
            gguf_packed,
            &format!("{p}.ca_text.out_proj.weight"),
        )?;
        let (ca_text_out_b, _) =
            take_or_gguf(weights, gguf_packed, &format!("{p}.ca_text.out_proj.bias"))?;
        let (cross_attn_in_w_t, cross_attn_in_gguf_key) = take_transposed_with_gguf_key(
            weights,
            gguf_packed,
            &format!("{p}.cross_attn.in_proj_weight"),
        )?;
        let (cross_attn_in_b, _) = take_or_gguf(
            weights,
            gguf_packed,
            &format!("{p}.cross_attn.in_proj.bias"),
        )?;
        let (cross_attn_out_w_t, cross_attn_out_gguf_key) = take_transposed_with_gguf_key(
            weights,
            gguf_packed,
            &format!("{p}.cross_attn.out_proj.weight"),
        )?;
        let (cross_attn_out_b, _) = take_or_gguf(
            weights,
            gguf_packed,
            &format!("{p}.cross_attn.out_proj.bias"),
        )?;
        let (linear1_w_t, linear1_gguf_key) =
            take_transposed_with_gguf_key(weights, gguf_packed, &format!("{p}.linear1.weight"))?;
        let (linear1_b, _) = take_or_gguf(weights, gguf_packed, &format!("{p}.linear1.bias"))?;
        let (linear2_w_t, linear2_gguf_key) =
            take_transposed_with_gguf_key(weights, gguf_packed, &format!("{p}.linear2.weight"))?;
        let (linear2_b, _) = take_or_gguf(weights, gguf_packed, &format!("{p}.linear2.bias"))?;
        let (norm1_w, _) = take_or_gguf(weights, gguf_packed, &format!("{p}.norm1.weight"))?;
        let (norm1_b, _) = take_or_gguf(weights, gguf_packed, &format!("{p}.norm1.bias"))?;
        let (norm2_w, _) = take_or_gguf(weights, gguf_packed, &format!("{p}.norm2.weight"))?;
        let (norm2_b, _) = take_or_gguf(weights, gguf_packed, &format!("{p}.norm2.bias"))?;
        let (norm3_w, _) = take_or_gguf(weights, gguf_packed, &format!("{p}.norm3.weight"))?;
        let (norm3_b, _) = take_or_gguf(weights, gguf_packed, &format!("{p}.norm3.bias"))?;
        let (catext_norm_w, _) =
            take_or_gguf(weights, gguf_packed, &format!("{p}.catext_norm.weight"))?;
        let (catext_norm_b, _) =
            take_or_gguf(weights, gguf_packed, &format!("{p}.catext_norm.bias"))?;
        layers.push(Sam3DecoderLayerWeights {
            self_attn_in_w_t,
            self_attn_in_b,
            self_attn_in_gguf_key,
            self_attn_out_w_t,
            self_attn_out_b,
            self_attn_out_gguf_key,
            ca_text_in_w_t,
            ca_text_in_b,
            ca_text_in_gguf_key,
            ca_text_out_w_t,
            ca_text_out_b,
            ca_text_out_gguf_key,
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
            catext_norm_w,
            catext_norm_b,
        });
    }

    let (query_embed, qs) =
        take_or_gguf(weights, gguf_packed, &format!("{base}.query_embed.weight"))?;
    ensure!(qs == vec![NUM_QUERIES, D_MODEL], "query_embed shape {qs:?}");
    let (reference_points, rs) = take_or_gguf(
        weights,
        gguf_packed,
        &format!("{base}.reference_points.weight"),
    )?;
    ensure!(rs == vec![NUM_QUERIES, 4], "reference_points shape {rs:?}");
    let (norm_w, _) = take_or_gguf(weights, gguf_packed, &format!("{base}.norm.weight"))?;
    let (norm_b, _) = take_or_gguf(weights, gguf_packed, &format!("{base}.norm.bias"))?;
    let bbox_embed = take_mlp3(
        weights,
        gguf_packed,
        &format!("{base}.bbox_embed"),
        D_MODEL,
        D_MODEL,
        4,
    )?;
    let ref_point_head = take_mlp2(
        weights,
        gguf_packed,
        &format!("{base}.ref_point_head"),
        2 * D_MODEL,
        D_MODEL,
        D_MODEL,
    )?;
    let boxrpb_x = take_mlp2(
        weights,
        gguf_packed,
        &format!("{base}.boxRPB_embed_x"),
        2,
        D_MODEL,
        N_HEADS,
    )?;
    let boxrpb_y = take_mlp2(
        weights,
        gguf_packed,
        &format!("{base}.boxRPB_embed_y"),
        2,
        D_MODEL,
        N_HEADS,
    )?;
    let (presence_token, ps) = take_or_gguf(
        weights,
        gguf_packed,
        &format!("{base}.presence_token.weight"),
    )?;
    ensure!(ps == vec![1, D_MODEL], "presence_token shape {ps:?}");
    let presence_token_head = take_mlp3(
        weights,
        gguf_packed,
        &format!("{base}.presence_token_head"),
        D_MODEL,
        D_MODEL,
        1,
    )?;
    let (presence_token_out_norm_w, _) = take_or_gguf(
        weights,
        gguf_packed,
        &format!("{base}.presence_token_out_norm.weight"),
    )?;
    let (presence_token_out_norm_b, _) = take_or_gguf(
        weights,
        gguf_packed,
        &format!("{base}.presence_token_out_norm.bias"),
    )?;

    Ok(Sam3DecoderWeights {
        loaded: true,
        prefix: base.to_string(),
        layers,
        query_embed,
        reference_points,
        norm_w,
        norm_b,
        bbox_embed,
        ref_point_head,
        boxrpb_x,
        boxrpb_y,
        presence_token,
        presence_token_head,
        presence_token_out_norm_w,
        presence_token_out_norm_b,
    })
}

#[derive(Debug, Clone, Default)]
pub struct Sam3DecoderOutput {
    /// `[num_layers, num_queries, batch, d_model]` post-norm.
    pub intermediate: Vec<f32>,
    /// `[num_layers, num_queries, batch, 4]` refined reference boxes
    /// (sigmoid scale). The first entry is the *initial* boxes; the last
    /// is layer 5's output (per upstream convention).
    pub intermediate_ref_boxes: Vec<f32>,
    /// `[num_layers, batch, 1]` per-layer presence logits.
    pub presence_logits: Vec<f32>,
    /// `[1, batch, d_model]` final presence features.
    pub presence_feats: Vec<f32>,
    pub num_layers: usize,
    pub num_queries: usize,
    pub batch: usize,
    pub d_model: usize,
}

/// Sine/cos encoding of a 4D position tensor `[nq, bs, 4]` → `[nq, bs, 2*D]`
/// matching `sam3.model.model_misc.gen_sineembed_for_position` with
/// `num_feats=256`.
fn sineembed_for_position_4d(pos: &[f32], nq: usize, bs: usize, d_model: usize) -> Vec<f32> {
    let half = d_model / 2;
    let scale = 2.0 * std::f32::consts::PI;
    let mut dim_t = vec![0.0f32; half];
    for i in 0..half {
        let exp = 2.0 * ((i / 2) as f32) / half as f32;
        dim_t[i] = 10000.0f32.powf(exp);
    }
    let mut out = vec![0.0f32; nq * bs * 2 * d_model];
    for q in 0..nq {
        for b in 0..bs {
            let p = &pos[(q * bs + b) * 4..(q * bs + b + 1) * 4];
            let x_e = p[0] * scale;
            let y_e = p[1] * scale;
            let w_e = p[2] * scale;
            let h_e = p[3] * scale;
            // Layout: cat([pos_y, pos_x, pos_w, pos_h], dim=-1).
            let base = (q * bs + b) * 2 * d_model;
            for axis in 0..4 {
                let val = [y_e, x_e, w_e, h_e][axis];
                let slot = base + axis * half;
                for i in 0..half {
                    let theta = val / dim_t[i];
                    out[slot + i] = if i % 2 == 0 { theta.sin() } else { theta.cos() };
                }
            }
        }
    }
    out
}

fn inverse_sigmoid(x: f32) -> f32 {
    let eps = 1e-3f32;
    let x = x.clamp(0.0, 1.0);
    let x1 = x.max(eps);
    let x2 = (1.0 - x).max(eps);
    (x1 / x2).ln()
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Compute the log-scale boxRPB attention mask for one batch element.
/// Returns flat `[n_heads * num_queries, H * W]` to be broadcast-added to
/// the attention scores per head.
fn boxrpb_log_mask(
    weights: &Sam3DecoderWeights,
    reference_boxes: &[f32], // [nq, 1, 4] cxcywh in [0, 1]
    nq: usize,
    h: usize,
    w: usize,
    gguf_packed: Option<&GgufPackedParams>,
) -> Result<Vec<f32>> {
    // coords_h[y] = y / H, coords_w[x] = x / W.
    let coords_h: Vec<f32> = (0..h).map(|y| y as f32 / h as f32).collect();
    let coords_w: Vec<f32> = (0..w).map(|x| x as f32 / w as f32).collect();

    // For each query, compute boxes_xyxy = (cx-w/2, cy-h/2, cx+w/2, cy+h/2).
    let mut deltas_x = vec![0f32; nq * w * 2];
    let mut deltas_y = vec![0f32; nq * h * 2];
    for q in 0..nq {
        let p = &reference_boxes[q * 4..(q + 1) * 4];
        let (cx, cy, bw, bh) = (p[0], p[1], p[2], p[3]);
        let x0 = cx - 0.5 * bw;
        let x1 = cx + 0.5 * bw;
        let y0 = cy - 0.5 * bh;
        let y1 = cy + 0.5 * bh;
        for xi in 0..w {
            let dx0 = (coords_w[xi] - x0) * 8.0;
            let dx1 = (coords_w[xi] - x1) * 8.0;
            deltas_x[(q * w + xi) * 2] = log_norm(dx0);
            deltas_x[(q * w + xi) * 2 + 1] = log_norm(dx1);
        }
        for yi in 0..h {
            let dy0 = (coords_h[yi] - y0) * 8.0;
            let dy1 = (coords_h[yi] - y1) * 8.0;
            deltas_y[(q * h + yi) * 2] = log_norm(dy0);
            deltas_y[(q * h + yi) * 2 + 1] = log_norm(dy1);
        }
    }
    // MLPs: [nq*W, 2] → [nq*W, n_heads].
    let dx_feats = mlp2_forward(&weights.boxrpb_x, &deltas_x, nq * w, gguf_packed)?;
    let dy_feats = mlp2_forward(&weights.boxrpb_y, &deltas_y, nq * h, gguf_packed)?;

    // B[q, y, x, head] = dy_feats[q, y, head] + dx_feats[q, x, head].
    // Repack to [n_heads, nq, h*w] for use as additive attention mask.
    let mut out = vec![0f32; N_HEADS * nq * h * w];
    for q in 0..nq {
        for y in 0..h {
            for x in 0..w {
                for head in 0..N_HEADS {
                    let dy = dy_feats[(q * h + y) * N_HEADS + head];
                    let dx = dx_feats[(q * w + x) * N_HEADS + head];
                    out[(head * nq + q) * h * w + y * w + x] = dy + dx;
                }
            }
        }
    }
    Ok(out)
}

fn log_norm(v: f32) -> f32 {
    // sign(v) * log2(|v| + 1) / log2(8)
    let s = if v < 0.0 { -1.0 } else { 1.0 };
    s * (v.abs() + 1.0).log2() / 8.0f32.log2()
}

fn narrow_last(row: &[f32], rows: usize, width: usize, start: usize, len: usize) -> Vec<f32> {
    let mut out = vec![0f32; rows * len];
    for r in 0..rows {
        out[r * len..(r + 1) * len]
            .copy_from_slice(&row[r * width + start..r * width + start + len]);
    }
    out
}

/// Multi-head attention with optional GGUF packed linears, per-head bias, and key mask.
#[allow(clippy::too_many_arguments)]
pub(crate) fn mha_with_bias_maybe_gguf(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    in_proj_w_t: &[f32],
    in_proj_b: &[f32],
    in_gguf_key: Option<&str>,
    out_proj_w_t: &[f32],
    out_proj_b: &[f32],
    out_gguf_key: Option<&str>,
    gguf_packed: Option<&GgufPackedParams>,
    batch: usize,
    l_q: usize,
    l_k: usize,
    embed_dim: usize,
    num_heads: usize,
    attn_bias_h_lq_lk: Option<&[f32]>,
    key_padding_mask: Option<&[u8]>,
) -> Result<Vec<f32>> {
    if in_gguf_key.is_none() && out_gguf_key.is_none() {
        return mha_with_bias_f32(
            q,
            k,
            v,
            in_proj_w_t,
            in_proj_b,
            out_proj_w_t,
            out_proj_b,
            batch,
            l_q,
            l_k,
            embed_dim,
            num_heads,
            attn_bias_h_lq_lk,
            key_padding_mask,
        );
    }

    use super::tensor::{matmul, matmul_bt, softmax_rows};
    let head_dim = embed_dim / num_heads;
    let rows_q = batch * l_q;
    let rows_k = batch * l_k;

    let (q_proj, k_proj, v_proj) = if let Some(in_key) = in_gguf_key {
        let qkv_q = linear_maybe_gguf(
            q,
            rows_q,
            embed_dim,
            in_proj_w_t,
            Some(in_key),
            gguf_packed,
            3 * embed_dim,
            in_proj_b,
        )?;
        let qkv_k = linear_maybe_gguf(
            k,
            rows_k,
            embed_dim,
            in_proj_w_t,
            Some(in_key),
            gguf_packed,
            3 * embed_dim,
            in_proj_b,
        )?;
        let qkv_v = linear_maybe_gguf(
            v,
            rows_k,
            embed_dim,
            in_proj_w_t,
            Some(in_key),
            gguf_packed,
            3 * embed_dim,
            in_proj_b,
        )?;
        (
            narrow_last(&qkv_q, rows_q, 3 * embed_dim, 0, embed_dim),
            narrow_last(&qkv_k, rows_k, 3 * embed_dim, embed_dim, embed_dim),
            narrow_last(&qkv_v, rows_k, 3 * embed_dim, 2 * embed_dim, embed_dim),
        )
    } else {
        let (wq, wk, wv) = split3(in_proj_w_t, embed_dim);
        let bq = &in_proj_b[0..embed_dim];
        let bk = &in_proj_b[embed_dim..2 * embed_dim];
        let bv = &in_proj_b[2 * embed_dim..3 * embed_dim];
        (
            linear_maybe_gguf(q, rows_q, embed_dim, &wq, None, gguf_packed, embed_dim, bq)?,
            linear_maybe_gguf(k, rows_k, embed_dim, &wk, None, gguf_packed, embed_dim, bk)?,
            linear_maybe_gguf(v, rows_k, embed_dim, &wv, None, gguf_packed, embed_dim, bv)?,
        )
    };

    let bh = batch * num_heads;
    let mut qh = vec![0f32; bh * l_q * head_dim];
    let mut kh = vec![0f32; bh * l_k * head_dim];
    let mut vh = vec![0f32; bh * l_k * head_dim];
    repack(&q_proj, &mut qh, batch, l_q, num_heads, head_dim);
    repack(&k_proj, &mut kh, batch, l_k, num_heads, head_dim);
    repack(&v_proj, &mut vh, batch, l_k, num_heads, head_dim);

    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let mut scores = vec![0f32; l_q * l_k];
    let mut attn_out = vec![0f32; bh * l_q * head_dim];
    for bi in 0..batch {
        for h in 0..num_heads {
            let bhi = bi * num_heads + h;
            let q_h = &qh[bhi * l_q * head_dim..(bhi + 1) * l_q * head_dim];
            let k_h = &kh[bhi * l_k * head_dim..(bhi + 1) * l_k * head_dim];
            let v_h = &vh[bhi * l_k * head_dim..(bhi + 1) * l_k * head_dim];
            matmul_bt(q_h, k_h, &mut scores, l_q, head_dim, l_k, scale);
            if let Some(bias) = attn_bias_h_lq_lk {
                let bias_h = &bias[h * l_q * l_k..(h + 1) * l_q * l_k];
                for i in 0..scores.len() {
                    scores[i] += bias_h[i];
                }
            }
            if let Some(mask) = key_padding_mask {
                let mask_b = &mask[bi * l_k..(bi + 1) * l_k];
                for r in 0..l_q {
                    let row = &mut scores[r * l_k..(r + 1) * l_k];
                    for (c, m) in mask_b.iter().enumerate() {
                        if *m != 0 {
                            row[c] = f32::NEG_INFINITY;
                        }
                    }
                }
            }
            softmax_rows(&mut scores, l_q, l_k);
            let out_h = &mut attn_out[bhi * l_q * head_dim..(bhi + 1) * l_q * head_dim];
            matmul(&scores, v_h, out_h, l_q, l_k, head_dim);
        }
    }

    let mut packed = vec![0f32; batch * l_q * embed_dim];
    for bi in 0..batch {
        for l in 0..l_q {
            for h in 0..num_heads {
                let src = ((bi * num_heads + h) * l_q + l) * head_dim;
                let dst = (bi * l_q + l) * embed_dim + h * head_dim;
                packed[dst..dst + head_dim].copy_from_slice(&attn_out[src..src + head_dim]);
            }
        }
    }
    linear_maybe_gguf(
        &packed,
        batch * l_q,
        embed_dim,
        out_proj_w_t,
        out_gguf_key,
        gguf_packed,
        embed_dim,
        out_proj_b,
    )
}

/// F32-only MHA (used when weights are materialized at extract).
#[allow(clippy::too_many_arguments)]
fn mha_with_bias_f32(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    in_proj_w_t: &[f32],
    in_proj_b: &[f32],
    out_proj_w_t: &[f32],
    out_proj_b: &[f32],
    batch: usize,
    l_q: usize,
    l_k: usize,
    embed_dim: usize,
    num_heads: usize,
    attn_bias_h_lq_lk: Option<&[f32]>,
    key_padding_mask: Option<&[u8]>,
) -> Result<Vec<f32>> {
    use super::tensor::{matmul, matmul_bt, softmax_rows};
    let head_dim = embed_dim / num_heads;
    let (wq, wk, wv) = split3(in_proj_w_t, embed_dim);
    let bq = &in_proj_b[0..embed_dim];
    let bk = &in_proj_b[embed_dim..2 * embed_dim];
    let bv = &in_proj_b[2 * embed_dim..3 * embed_dim];

    let q_proj = linear(q, batch * l_q, embed_dim, &wq, embed_dim, bq)?;
    let k_proj = linear(k, batch * l_k, embed_dim, &wk, embed_dim, bk)?;
    let v_proj = linear(v, batch * l_k, embed_dim, &wv, embed_dim, bv)?;

    let bh = batch * num_heads;
    let mut qh = vec![0f32; bh * l_q * head_dim];
    let mut kh = vec![0f32; bh * l_k * head_dim];
    let mut vh = vec![0f32; bh * l_k * head_dim];
    repack(&q_proj, &mut qh, batch, l_q, num_heads, head_dim);
    repack(&k_proj, &mut kh, batch, l_k, num_heads, head_dim);
    repack(&v_proj, &mut vh, batch, l_k, num_heads, head_dim);

    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let mut scores = vec![0f32; l_q * l_k];
    let mut attn_out = vec![0f32; bh * l_q * head_dim];
    for bi in 0..batch {
        for h in 0..num_heads {
            let bhi = bi * num_heads + h;
            let q_h = &qh[bhi * l_q * head_dim..(bhi + 1) * l_q * head_dim];
            let k_h = &kh[bhi * l_k * head_dim..(bhi + 1) * l_k * head_dim];
            let v_h = &vh[bhi * l_k * head_dim..(bhi + 1) * l_k * head_dim];
            matmul_bt(q_h, k_h, &mut scores, l_q, head_dim, l_k, scale);
            if let Some(bias) = attn_bias_h_lq_lk {
                let bias_h = &bias[h * l_q * l_k..(h + 1) * l_q * l_k];
                for i in 0..scores.len() {
                    scores[i] += bias_h[i];
                }
            }
            if let Some(mask) = key_padding_mask {
                let mask_b = &mask[bi * l_k..(bi + 1) * l_k];
                for r in 0..l_q {
                    let row = &mut scores[r * l_k..(r + 1) * l_k];
                    for (c, m) in mask_b.iter().enumerate() {
                        if *m != 0 {
                            row[c] = f32::NEG_INFINITY;
                        }
                    }
                }
            }
            softmax_rows(&mut scores, l_q, l_k);
            let out_h = &mut attn_out[bhi * l_q * head_dim..(bhi + 1) * l_q * head_dim];
            matmul(&scores, v_h, out_h, l_q, l_k, head_dim);
        }
    }

    let mut packed = vec![0f32; batch * l_q * embed_dim];
    for bi in 0..batch {
        for l in 0..l_q {
            for h in 0..num_heads {
                let src = ((bi * num_heads + h) * l_q + l) * head_dim;
                let dst = (bi * l_q + l) * embed_dim + h * head_dim;
                packed[dst..dst + head_dim].copy_from_slice(&attn_out[src..src + head_dim]);
            }
        }
    }
    linear(
        &packed,
        batch * l_q,
        embed_dim,
        out_proj_w_t,
        embed_dim,
        out_proj_b,
    )
}

fn split3(w_t: &[f32], e: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut wq = vec![0f32; e * e];
    let mut wk = vec![0f32; e * e];
    let mut wv = vec![0f32; e * e];
    for i in 0..e {
        for j in 0..e {
            wq[i * e + j] = w_t[i * 3 * e + j];
            wk[i * e + j] = w_t[i * 3 * e + e + j];
            wv[i * e + j] = w_t[i * 3 * e + 2 * e + j];
        }
    }
    (wq, wk, wv)
}

fn repack(src: &[f32], dst: &mut [f32], batch: usize, l: usize, num_heads: usize, head_dim: usize) {
    let e = num_heads * head_dim;
    for bi in 0..batch {
        for li in 0..l {
            for h in 0..num_heads {
                let s = (bi * l + li) * e + h * head_dim;
                let d = ((bi * num_heads + h) * l + li) * head_dim;
                dst[d..d + head_dim].copy_from_slice(&src[s..s + head_dim]);
            }
        }
    }
}

/// Run the full decoder (batch must be 1 for the boxRPB path).
#[allow(clippy::too_many_arguments)]
pub fn forward_decoder(
    weights: &Sam3DecoderWeights,
    memory: &[f32],      // [batch, h*w, D] batch-first (matches our encoder output)
    memory_pos: &[f32],  // same layout
    memory_text: &[f32], // [seq, batch, D] seq-first
    text_attention_mask: &[u8], // [batch, seq]
    batch: usize,
    h: usize,
    w: usize,
    seq_len: usize,
    gguf_packed: Option<&GgufPackedParams>,
) -> Result<Sam3DecoderOutput> {
    ensure!(weights.loaded, "SAM3 detector decoder not loaded");
    ensure!(batch == 1, "decoder forward requires batch=1 for boxRPB");
    let hw = h * w;
    let nq = NUM_QUERIES;

    // Build initial tgt and reference boxes.
    let mut tgt = vec![0f32; nq * batch * D_MODEL]; // seq-first [nq, bs, D]
    for q in 0..nq {
        let src = &weights.query_embed[q * D_MODEL..(q + 1) * D_MODEL];
        for b in 0..batch {
            tgt[(q * batch + b) * D_MODEL..(q * batch + b + 1) * D_MODEL].copy_from_slice(src);
        }
    }
    let mut reference_boxes = vec![0f32; nq * batch * 4];
    for q in 0..nq {
        let src = &weights.reference_points[q * 4..(q + 1) * 4];
        for b in 0..batch {
            let dst = &mut reference_boxes[(q * batch + b) * 4..(q * batch + b + 1) * 4];
            for k in 0..4 {
                dst[k] = sigmoid(src[k]);
            }
        }
    }

    let mut presence_out = vec![0f32; batch * D_MODEL];
    for b in 0..batch {
        presence_out[b * D_MODEL..(b + 1) * D_MODEL].copy_from_slice(&weights.presence_token);
    }

    let mut intermediate = Vec::with_capacity(N_LAYERS);
    let mut intermediate_ref_boxes = Vec::with_capacity(N_LAYERS);
    let mut presence_logits = Vec::with_capacity(N_LAYERS);

    // First entry of intermediate_ref_boxes is the initial reference boxes.
    intermediate_ref_boxes.push(reference_boxes.clone());

    // Reorder memory and memory_text into batch-first [bs, len, D] for MHA.
    // memory is already [bs, hw, D]. memory_text is [seq, bs, D] → [bs, seq, D].
    let mut memory_text_bf = vec![0f32; batch * seq_len * D_MODEL];
    for b in 0..batch {
        for l in 0..seq_len {
            let src = (l * batch + b) * D_MODEL;
            let dst = (b * seq_len + l) * D_MODEL;
            memory_text_bf[dst..dst + D_MODEL].copy_from_slice(&memory_text[src..src + D_MODEL]);
        }
    }

    for (layer_idx, layer) in weights.layers.iter().enumerate() {
        // Compute query_pos = ref_point_head(sineembed(ref_boxes)).
        let sine = sineembed_for_position_4d(&reference_boxes, nq, batch, D_MODEL);
        let query_pos = mlp2_forward(&weights.ref_point_head, &sine, nq * batch, gguf_packed)?;
        if let Some(dir) = rlx_ir::env::var("RLX_SAM3_DECODER_DUMP_DIR") {
            use std::io::Write as _;
            let path = format!("{dir}/host_layer{layer_idx}_query_pos.f32");
            let mut f = std::fs::File::create(&path).unwrap();
            for v in &query_pos {
                f.write_all(&v.to_le_bytes()).unwrap();
            }
        }

        // Build self-attn input: prepend presence_token to tgt. The
        // sequence becomes [presence, q0, ..., q199] = 201 tokens with
        // pos = [0, qp0, ..., qp199] (zeros for presence).
        let sa_len = 1 + nq;
        let mut sa_x = vec![0f32; sa_len * batch * D_MODEL];
        let mut sa_pos = vec![0f32; sa_len * batch * D_MODEL];
        for b in 0..batch {
            sa_x[b * D_MODEL..(b + 1) * D_MODEL]
                .copy_from_slice(&presence_out[b * D_MODEL..(b + 1) * D_MODEL]);
        }
        for q in 0..nq {
            for b in 0..batch {
                let src = &tgt[(q * batch + b) * D_MODEL..(q * batch + b + 1) * D_MODEL];
                sa_x[((1 + q) * batch + b) * D_MODEL..((1 + q) * batch + b + 1) * D_MODEL]
                    .copy_from_slice(src);
                let qp = &query_pos[(q * batch + b) * D_MODEL..(q * batch + b + 1) * D_MODEL];
                sa_pos[((1 + q) * batch + b) * D_MODEL..((1 + q) * batch + b + 1) * D_MODEL]
                    .copy_from_slice(qp);
            }
        }
        // Reorder to batch-first for MHA helper.
        let mut sa_x_bf = vec![0f32; batch * sa_len * D_MODEL];
        let mut sa_pos_bf = vec![0f32; batch * sa_len * D_MODEL];
        for b in 0..batch {
            for l in 0..sa_len {
                let s = (l * batch + b) * D_MODEL;
                let d = (b * sa_len + l) * D_MODEL;
                sa_x_bf[d..d + D_MODEL].copy_from_slice(&sa_x[s..s + D_MODEL]);
                sa_pos_bf[d..d + D_MODEL].copy_from_slice(&sa_pos[s..s + D_MODEL]);
            }
        }
        // q=k=sa_x+sa_pos, v=sa_x.
        let mut qk = vec![0f32; sa_x_bf.len()];
        for i in 0..qk.len() {
            qk[i] = sa_x_bf[i] + sa_pos_bf[i];
        }
        let sa = mha_with_bias_maybe_gguf(
            &qk,
            &qk,
            &sa_x_bf,
            &layer.self_attn_in_w_t,
            &layer.self_attn_in_b,
            layer.self_attn_in_gguf_key.as_deref(),
            &layer.self_attn_out_w_t,
            &layer.self_attn_out_b,
            layer.self_attn_out_gguf_key.as_deref(),
            gguf_packed,
            batch,
            sa_len,
            sa_len,
            D_MODEL,
            N_HEADS,
            None,
            None,
        )?;
        for i in 0..sa_x_bf.len() {
            sa_x_bf[i] += sa[i];
        }
        // Post-norm.
        let sa_x_bf = layer_norm(&sa_x_bf, &layer.norm2_w, &layer.norm2_b, D_MODEL, 1e-5)?;
        // Split presence + tgt back out (seq-first ordering).
        let mut new_presence = vec![0f32; batch * D_MODEL];
        for b in 0..batch {
            let src = &sa_x_bf[(b * sa_len) * D_MODEL..(b * sa_len + 1) * D_MODEL];
            new_presence[b * D_MODEL..(b + 1) * D_MODEL].copy_from_slice(src);
        }
        let mut after_sa = vec![0f32; batch * nq * D_MODEL];
        for b in 0..batch {
            for q in 0..nq {
                let src = (b * sa_len + 1 + q) * D_MODEL;
                let dst = (b * nq + q) * D_MODEL;
                after_sa[dst..dst + D_MODEL].copy_from_slice(&sa_x_bf[src..src + D_MODEL]);
            }
        }
        if let Some(dir) = rlx_ir::env::var("RLX_SAM3_DECODER_DUMP_DIR") {
            use std::io::Write as _;
            let path = format!("{dir}/host_layer{layer_idx}_sa_queries.f32");
            let mut f = std::fs::File::create(&path).unwrap();
            for v in &after_sa {
                f.write_all(&v.to_le_bytes()).unwrap();
            }
        }

        // Text cross-attention. q = after_sa + query_pos (batch-first
        // ordering: [bs, nq, D]).
        let mut q_text = vec![0f32; batch * nq * D_MODEL];
        for b in 0..batch {
            for q in 0..nq {
                let dst = (b * nq + q) * D_MODEL;
                let qp = &query_pos[(q * batch + b) * D_MODEL..(q * batch + b + 1) * D_MODEL];
                for c in 0..D_MODEL {
                    q_text[dst + c] = after_sa[dst + c] + qp[c];
                }
            }
        }
        let text_attn = mha_with_bias_maybe_gguf(
            &q_text,
            &memory_text_bf,
            &memory_text_bf,
            &layer.ca_text_in_w_t,
            &layer.ca_text_in_b,
            layer.ca_text_in_gguf_key.as_deref(),
            &layer.ca_text_out_w_t,
            &layer.ca_text_out_b,
            layer.ca_text_out_gguf_key.as_deref(),
            gguf_packed,
            batch,
            nq,
            seq_len,
            D_MODEL,
            N_HEADS,
            None,
            Some(text_attention_mask),
        )?;
        let mut tgt_after_ca_text = vec![0f32; batch * nq * D_MODEL];
        for i in 0..tgt_after_ca_text.len() {
            tgt_after_ca_text[i] = after_sa[i] + text_attn[i];
        }
        let tgt_after_ca_text = layer_norm(
            &tgt_after_ca_text,
            &layer.catext_norm_w,
            &layer.catext_norm_b,
            D_MODEL,
            1e-5,
        )?;
        if let Some(dir) = rlx_ir::env::var("RLX_SAM3_DECODER_DUMP_DIR") {
            use std::io::Write as _;
            let path = format!("{dir}/host_layer{layer_idx}_after_ca_text_q.f32");
            let mut f = std::fs::File::create(&path).unwrap();
            for v in &tgt_after_ca_text {
                f.write_all(&v.to_le_bytes()).unwrap();
            }
        }

        // Image cross-attention with boxRPB log mask.
        let rpb = boxrpb_log_mask(weights, &reference_boxes, nq, h, w, gguf_packed)?;
        // Need to prepend a row of zeros for the presence token mask, but
        // since we don't include presence in the cross-attn for our path
        // (presence was consumed during self-attn only and cross-attn
        // doesn't gain a presence row in the layer forward — let me
        // re-check upstream): upstream concatenates a zero row in mask
        // before calling cross_attn so that cross_attn's mask has shape
        // (bs*nheads, nq+1, hw). The presence token IS included in cross-
        // attn. We follow the same path.
        let cross_len_q = 1 + nq;
        // Build a [n_heads, cross_len_q, hw] mask with presence row of 0.
        let mut full_mask = vec![0f32; N_HEADS * cross_len_q * hw];
        for head in 0..N_HEADS {
            // presence row: 0s already.
            for q in 0..nq {
                let src = (head * nq + q) * hw;
                let dst = (head * cross_len_q + 1 + q) * hw;
                full_mask[dst..dst + hw].copy_from_slice(&rpb[src..src + hw]);
            }
        }
        // Cross-attention input: prepend the new presence to the tgt.
        let mut ca_in_seq_first = vec![0f32; cross_len_q * batch * D_MODEL];
        for b in 0..batch {
            // Presence at position 0.
            ca_in_seq_first[b * D_MODEL..(b + 1) * D_MODEL]
                .copy_from_slice(&new_presence[b * D_MODEL..(b + 1) * D_MODEL]);
            for q in 0..nq {
                let src = &tgt_after_ca_text[(b * nq + q) * D_MODEL..(b * nq + q + 1) * D_MODEL];
                ca_in_seq_first
                    [((1 + q) * batch + b) * D_MODEL..((1 + q) * batch + b + 1) * D_MODEL]
                    .copy_from_slice(src);
            }
        }
        // Reorder to batch-first.
        let mut ca_in_bf = vec![0f32; batch * cross_len_q * D_MODEL];
        let mut ca_pos_bf = vec![0f32; batch * cross_len_q * D_MODEL];
        for b in 0..batch {
            for l in 0..cross_len_q {
                let s = (l * batch + b) * D_MODEL;
                let d = (b * cross_len_q + l) * D_MODEL;
                ca_in_bf[d..d + D_MODEL].copy_from_slice(&ca_in_seq_first[s..s + D_MODEL]);
                if l == 0 {
                    // presence pos = 0
                } else {
                    let qp = &query_pos
                        [((l - 1) * batch + b) * D_MODEL..((l - 1) * batch + b + 1) * D_MODEL];
                    ca_pos_bf[d..d + D_MODEL].copy_from_slice(qp);
                }
            }
        }
        // Q = ca_in + ca_pos; K = memory + memory_pos; V = memory.
        let mut q_img = vec![0f32; ca_in_bf.len()];
        for i in 0..q_img.len() {
            q_img[i] = ca_in_bf[i] + ca_pos_bf[i];
        }
        let mut k_img = vec![0f32; memory.len()];
        for i in 0..k_img.len() {
            k_img[i] = memory[i] + memory_pos[i];
        }
        let ca_out = mha_with_bias_maybe_gguf(
            &q_img,
            &k_img,
            memory,
            &layer.cross_attn_in_w_t,
            &layer.cross_attn_in_b,
            layer.cross_attn_in_gguf_key.as_deref(),
            &layer.cross_attn_out_w_t,
            &layer.cross_attn_out_b,
            layer.cross_attn_out_gguf_key.as_deref(),
            gguf_packed,
            batch,
            cross_len_q,
            hw,
            D_MODEL,
            N_HEADS,
            Some(&full_mask),
            None,
        )?;
        if let Some(dir) = rlx_ir::env::var("RLX_SAM3_DECODER_DUMP_DIR") {
            use std::io::Write as _;
            let path = format!("{dir}/host_layer{layer_idx}_ca_img_proj.f32");
            let mut f = std::fs::File::create(&path).unwrap();
            for v in &ca_out {
                f.write_all(&v.to_le_bytes()).unwrap();
            }
        }
        for i in 0..ca_in_bf.len() {
            ca_in_bf[i] += ca_out[i];
        }
        // Post-norm1.
        let ca_in_bf = layer_norm(&ca_in_bf, &layer.norm1_w, &layer.norm1_b, D_MODEL, 1e-5)?;
        if let Some(dir) = rlx_ir::env::var("RLX_SAM3_DECODER_DUMP_DIR") {
            use std::io::Write as _;
            // Extract queries only (rows 1..lq) of [B, lq, D].
            let mut q_only = vec![0f32; batch * nq * D_MODEL];
            for b in 0..batch {
                for q in 0..nq {
                    let src = (b * cross_len_q + 1 + q) * D_MODEL;
                    let dst = (b * nq + q) * D_MODEL;
                    q_only[dst..dst + D_MODEL].copy_from_slice(&ca_in_bf[src..src + D_MODEL]);
                }
            }
            let path = format!("{dir}/host_layer{layer_idx}_after_ca_img_q.f32");
            let mut f = std::fs::File::create(&path).unwrap();
            for v in &q_only {
                f.write_all(&v.to_le_bytes()).unwrap();
            }
        }

        // FFN.
        let mut ff = linear_maybe_gguf(
            &ca_in_bf,
            batch * cross_len_q,
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
            batch * cross_len_q,
            DIM_FF,
            &layer.linear2_w_t,
            layer.linear2_gguf_key.as_deref(),
            gguf_packed,
            D_MODEL,
            &layer.linear2_b,
        )?;
        let mut after_ffn = ca_in_bf.clone();
        for i in 0..after_ffn.len() {
            after_ffn[i] += ffn[i];
        }
        // Post-norm3.
        let after_ffn = layer_norm(&after_ffn, &layer.norm3_w, &layer.norm3_b, D_MODEL, 1e-5)?;

        // Split off presence and tgt.
        let mut layer_presence = vec![0f32; batch * D_MODEL];
        let mut layer_tgt = vec![0f32; batch * nq * D_MODEL];
        for b in 0..batch {
            let src_p = &after_ffn[(b * cross_len_q) * D_MODEL..(b * cross_len_q + 1) * D_MODEL];
            layer_presence[b * D_MODEL..(b + 1) * D_MODEL].copy_from_slice(src_p);
            for q in 0..nq {
                let src = (b * cross_len_q + 1 + q) * D_MODEL;
                let dst = (b * nq + q) * D_MODEL;
                layer_tgt[dst..dst + D_MODEL].copy_from_slice(&after_ffn[src..src + D_MODEL]);
            }
        }
        if let Some(dir) = rlx_ir::env::var("RLX_SAM3_DECODER_DUMP_DIR") {
            use std::io::Write as _;
            for (vals, name) in [(&layer_tgt, "new_tgt"), (&layer_presence, "new_presence")] {
                let path = format!("{dir}/host_layer{layer_idx}_{name}.f32");
                let mut f = std::fs::File::create(&path).unwrap();
                for v in vals {
                    f.write_all(&v.to_le_bytes()).unwrap();
                }
            }
        }
        // tgt becomes the layer_tgt (in seq-first ordering).
        for q in 0..nq {
            for b in 0..batch {
                let src = (b * nq + q) * D_MODEL;
                let dst = (q * batch + b) * D_MODEL;
                tgt[dst..dst + D_MODEL].copy_from_slice(&layer_tgt[src..src + D_MODEL]);
            }
        }
        // presence_out is the new layer presence (in batch ordering).
        presence_out.copy_from_slice(&layer_presence);

        // Box refinement: delta = bbox_embed(out_norm(layer_tgt)); ref = sigmoid(inv_sig(ref) + delta).
        // out_norm in batch-first order.
        let out_norm = layer_norm(&layer_tgt, &weights.norm_w, &weights.norm_b, D_MODEL, 1e-5)?;
        if let Some(dir) = rlx_ir::env::var("RLX_SAM3_DECODER_DUMP_DIR") {
            use std::io::Write as _;
            let path = format!("{dir}/host_layer{layer_idx}_out_norm.f32");
            let mut f = std::fs::File::create(&path).unwrap();
            for v in &out_norm {
                f.write_all(&v.to_le_bytes()).unwrap();
            }
        }
        let delta = mlp3_forward(&weights.bbox_embed, &out_norm, batch * nq, gguf_packed)?;
        let mut new_ref = vec![0f32; nq * batch * 4];
        for q in 0..nq {
            for b in 0..batch {
                let cur = &reference_boxes[(q * batch + b) * 4..(q * batch + b + 1) * 4];
                let d = &delta[(b * nq + q) * 4..(b * nq + q + 1) * 4];
                for k in 0..4 {
                    new_ref[(q * batch + b) * 4 + k] = sigmoid(inverse_sigmoid(cur[k]) + d[k]);
                }
            }
        }
        reference_boxes = new_ref;
        if layer_idx != N_LAYERS - 1 {
            intermediate_ref_boxes.push(reference_boxes.clone());
        }

        // Intermediate output post-norm in seq-first.
        let mut out_seq_first = vec![0f32; nq * batch * D_MODEL];
        for q in 0..nq {
            for b in 0..batch {
                let src = (b * nq + q) * D_MODEL;
                let dst = (q * batch + b) * D_MODEL;
                out_seq_first[dst..dst + D_MODEL].copy_from_slice(&out_norm[src..src + D_MODEL]);
            }
        }
        intermediate.push(out_seq_first);

        // Presence logits per layer.
        let p_norm = layer_norm(
            &layer_presence,
            &weights.presence_token_out_norm_w,
            &weights.presence_token_out_norm_b,
            D_MODEL,
            1e-5,
        )?;
        let p_logit = mlp3_forward(&weights.presence_token_head, &p_norm, batch, gguf_packed)?;
        presence_logits.push(p_logit);
    }

    // Stack.
    let mut int_stack = vec![0f32; N_LAYERS * nq * batch * D_MODEL];
    for (li, layer_out) in intermediate.iter().enumerate() {
        int_stack[li * nq * batch * D_MODEL..(li + 1) * nq * batch * D_MODEL]
            .copy_from_slice(layer_out);
    }
    let mut ref_stack = vec![0f32; N_LAYERS * nq * batch * 4];
    for (li, ref_l) in intermediate_ref_boxes.iter().enumerate() {
        ref_stack[li * nq * batch * 4..(li + 1) * nq * batch * 4].copy_from_slice(ref_l);
    }
    let mut presence_stack = vec![0f32; N_LAYERS * batch];
    for (li, p) in presence_logits.iter().enumerate() {
        for b in 0..batch {
            presence_stack[li * batch + b] = p[b];
        }
    }

    Ok(Sam3DecoderOutput {
        intermediate: int_stack,
        intermediate_ref_boxes: ref_stack,
        presence_logits: presence_stack,
        presence_feats: presence_out,
        num_layers: N_LAYERS,
        num_queries: nq,
        batch,
        d_model: D_MODEL,
    })
}
