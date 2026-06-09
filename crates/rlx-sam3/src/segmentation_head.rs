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

//! Native SAM3 segmentation head + dot-product scoring.
//!
//! Mirrors `sam3.model.maskformer_segmentation.UniversalSegmentationHead`
//! and `sam3.model.model_misc.DotProductScoring` as configured in
//! `model_builder._create_segmentation_head` / `_create_dot_product_scoring`.

use super::detector::Sam3DetectorOutput;
use super::detector_decoder::{Mlp2, Mlp3};
use super::sam3::Sam3ImagePrediction;
use super::segmentation_pixel_ir::{
    Sam3Conv1x1Compiled, Sam3PixelDecoderStepCompiled, compile_pixel_decoder_steps,
};
use super::tensor::{layer_norm, matmul, matmul_bt, multihead_attention, softmax_rows};
use rlx_core::weight_map::WeightMap;
use rlx_flow::GgufPackedParams;

use crate::packed_gguf::{
    conv2d_3x3_nchw_gguf, conv2d_3x3_nchw_pad1, gguf_packed_conv1_to_nchw,
    gguf_packed_conv3_to_f32, linear_maybe_gguf, packed_linear, take_conv1x1_with_gguf_key,
    take_conv3x3_with_gguf_key, take_or_gguf, take_transposed_with_gguf_key,
};
use anyhow::{Result, ensure};
use rlx_runtime::Device;

const D_MODEL: usize = 256;
const N_HEADS: usize = 8;

#[derive(Default)]
pub struct Sam3SegmentationHeadWeights {
    pub loaded: bool,
    pub cross_attn_norm_w: Vec<f32>,
    pub cross_attn_norm_b: Vec<f32>,
    pub cross_attend_in_w_t: Vec<f32>,
    pub cross_attend_in_b: Vec<f32>,
    pub cross_attend_out_w_t: Vec<f32>,
    pub cross_attend_out_b: Vec<f32>,
    pub cross_attend_in_gguf_key: Option<String>,
    pub cross_attend_out_gguf_key: Option<String>,
    pub mask_embed_w0_gguf_key: Option<String>,
    pub mask_embed_w1_gguf_key: Option<String>,
    pub mask_embed_w2_gguf_key: Option<String>,
    pub pixel_conv_w: Vec<Vec<f32>>,
    pub pixel_conv_b: Vec<Vec<f32>>,
    pub pixel_conv_gguf_keys: Vec<Option<String>>,
    /// Lazy NCHW F32 `[c,c,3,3]` for packed 3×3 conv (host path when IR not compiled).
    pub pixel_conv_nchw_cache: Vec<Option<Vec<f32>>>,
    pub pixel_gn_w: Vec<Vec<f32>>,
    pub pixel_gn_b: Vec<Vec<f32>>,
    pub inst_w: Vec<f32>,
    pub inst_b: Vec<f32>,
    pub inst_gguf_key: Option<String>,
    pub sem_w: Vec<f32>,
    pub sem_b: Vec<f32>,
    pub sem_gguf_key: Option<String>,
    pub mask_embed: Mlp3,
    pub pixel_steps: Vec<Sam3PixelDecoderStepCompiled>,
    pub inst_head: Option<Sam3Conv1x1Compiled>,
    pub sem_head: Option<Sam3Conv1x1Compiled>,
}

#[derive(Clone, Default)]
pub struct Sam3DotProductScoringWeights {
    pub loaded: bool,
    pub prompt_mlp: Mlp2,
    pub prompt_mlp_out_norm_w: Vec<f32>,
    pub prompt_mlp_out_norm_b: Vec<f32>,
    pub prompt_proj_w_t: Vec<f32>,
    pub prompt_proj_b: Vec<f32>,
    pub hs_proj_w_t: Vec<f32>,
    pub hs_proj_b: Vec<f32>,
    pub prompt_mlp_w0_gguf_key: Option<String>,
    pub prompt_mlp_w1_gguf_key: Option<String>,
    pub prompt_proj_gguf_key: Option<String>,
    pub hs_proj_gguf_key: Option<String>,
}

pub fn extract_segmentation_head_weights(
    weights: &mut WeightMap,
    gguf_packed: Option<&GgufPackedParams>,
) -> Result<Sam3SegmentationHeadWeights> {
    let base = "detector.segmentation_head";

    let (cross_attn_norm_w, _) = take_or_gguf(
        weights,
        gguf_packed,
        &format!("{base}.cross_attn_norm.weight"),
    )?;
    let (cross_attn_norm_b, _) = take_or_gguf(
        weights,
        gguf_packed,
        &format!("{base}.cross_attn_norm.bias"),
    )?;
    let (cross_attend_in_w_t, cross_attend_in_gguf_key) = take_transposed_with_gguf_key(
        weights,
        gguf_packed,
        &format!("{base}.cross_attend_prompt.in_proj_weight"),
    )?;
    let (cross_attend_in_b, _) = take_or_gguf(
        weights,
        gguf_packed,
        &format!("{base}.cross_attend_prompt.in_proj_bias"),
    )?;
    let (cross_attend_out_w_t, cross_attend_out_gguf_key) = take_transposed_with_gguf_key(
        weights,
        gguf_packed,
        &format!("{base}.cross_attend_prompt.out_proj.weight"),
    )?;
    let (cross_attend_out_b, _) = take_or_gguf(
        weights,
        gguf_packed,
        &format!("{base}.cross_attend_prompt.out_proj.bias"),
    )?;

    let mut pixel_conv_w = Vec::new();
    let mut pixel_conv_b = Vec::new();
    let mut pixel_conv_gguf_keys = Vec::new();
    let mut pixel_gn_w = Vec::new();
    let mut pixel_gn_b = Vec::new();
    for i in 0..3 {
        let (cw, cs, ck) = take_conv3x3_with_gguf_key(
            weights,
            gguf_packed,
            &format!("{base}.pixel_decoder.conv_layers.{i}.weight"),
        )?;
        ensure!(
            cs == vec![D_MODEL, D_MODEL, 3, 3],
            "pixel_decoder conv {i} shape {cs:?}"
        );
        let (cb, _) = take_or_gguf(
            weights,
            gguf_packed,
            &format!("{base}.pixel_decoder.conv_layers.{i}.bias"),
        )?;
        let (nw, _) = take_or_gguf(
            weights,
            gguf_packed,
            &format!("{base}.pixel_decoder.norms.{i}.weight"),
        )?;
        let (nb, _) = take_or_gguf(
            weights,
            gguf_packed,
            &format!("{base}.pixel_decoder.norms.{i}.bias"),
        )?;
        pixel_conv_w.push(cw);
        pixel_conv_b.push(cb);
        pixel_conv_gguf_keys.push(ck);
        pixel_gn_w.push(nw);
        pixel_gn_b.push(nb);
    }

    let (inst_w, ins, inst_gguf_key) = take_conv1x1_with_gguf_key(
        weights,
        gguf_packed,
        &format!("{base}.instance_seg_head.weight"),
    )?;
    ensure!(
        ins == vec![D_MODEL, D_MODEL, 1, 1],
        "instance_seg_head shape {ins:?}"
    );
    let (inst_b, _) = take_or_gguf(
        weights,
        gguf_packed,
        &format!("{base}.instance_seg_head.bias"),
    )?;
    let (sem_w, ss, sem_gguf_key) = take_conv1x1_with_gguf_key(
        weights,
        gguf_packed,
        &format!("{base}.semantic_seg_head.weight"),
    )?;
    ensure!(
        ss == vec![1, D_MODEL, 1, 1],
        "semantic_seg_head shape {ss:?}"
    );
    let (sem_b, _) = take_or_gguf(
        weights,
        gguf_packed,
        &format!("{base}.semantic_seg_head.bias"),
    )?;

    let (m0_t, mask_embed_w0_gguf_key) = take_transposed_with_gguf_key(
        weights,
        gguf_packed,
        &format!("{base}.mask_predictor.mask_embed.layers.0.weight"),
    )?;
    let (m0_b, _) = take_or_gguf(
        weights,
        gguf_packed,
        &format!("{base}.mask_predictor.mask_embed.layers.0.bias"),
    )?;
    let (m1_t, mask_embed_w1_gguf_key) = take_transposed_with_gguf_key(
        weights,
        gguf_packed,
        &format!("{base}.mask_predictor.mask_embed.layers.1.weight"),
    )?;
    let (m1_b, _) = take_or_gguf(
        weights,
        gguf_packed,
        &format!("{base}.mask_predictor.mask_embed.layers.1.bias"),
    )?;
    let (m2_t, mask_embed_w2_gguf_key) = take_transposed_with_gguf_key(
        weights,
        gguf_packed,
        &format!("{base}.mask_predictor.mask_embed.layers.2.weight"),
    )?;
    let (m2_b, _) = take_or_gguf(
        weights,
        gguf_packed,
        &format!("{base}.mask_predictor.mask_embed.layers.2.bias"),
    )?;
    let mask_embed = Mlp3 {
        w0_t: m0_t,
        b0: m0_b,
        w1_t: m1_t,
        b1: m1_b,
        w2_t: m2_t,
        b2: m2_b,
        in_dim: D_MODEL,
        hidden: D_MODEL,
        out_dim: D_MODEL,
        w0_gguf_key: mask_embed_w0_gguf_key.clone(),
        w1_gguf_key: mask_embed_w1_gguf_key.clone(),
        w2_gguf_key: mask_embed_w2_gguf_key.clone(),
    };

    Ok(Sam3SegmentationHeadWeights {
        loaded: true,
        cross_attn_norm_w,
        cross_attn_norm_b,
        cross_attend_in_w_t,
        cross_attend_in_b,
        cross_attend_out_w_t,
        cross_attend_out_b,
        cross_attend_in_gguf_key,
        cross_attend_out_gguf_key,
        mask_embed_w0_gguf_key,
        mask_embed_w1_gguf_key,
        mask_embed_w2_gguf_key,
        pixel_conv_w,
        pixel_conv_b,
        pixel_conv_gguf_keys,
        pixel_conv_nchw_cache: vec![None; 3],
        pixel_gn_w,
        pixel_gn_b,
        inst_w,
        inst_b,
        inst_gguf_key,
        sem_w,
        sem_b,
        sem_gguf_key,
        mask_embed,
        pixel_steps: Vec::new(),
        inst_head: None,
        sem_head: None,
    })
}

/// Fill empty F32 conv tensors from packed GGUF so IR compile can run (one-time dequant).
pub fn materialize_segmentation_gguf_weights(
    weights: &mut Sam3SegmentationHeadWeights,
    gguf_packed: Option<&GgufPackedParams>,
) -> Result<()> {
    let Some(gguf) = gguf_packed else {
        return Ok(());
    };
    for i in 0..weights.pixel_conv_gguf_keys.len() {
        if weights.pixel_conv_w[i].is_empty() {
            if let Some(key) = &weights.pixel_conv_gguf_keys[i] {
                let p = packed_linear(gguf, key)
                    .ok_or_else(|| anyhow::anyhow!("missing packed pixel conv: {key}"))?;
                weights.pixel_conv_w[i] = gguf_packed_conv3_to_f32(p, D_MODEL, D_MODEL)?;
            }
        }
    }
    if weights.inst_w.is_empty() {
        if let Some(key) = &weights.inst_gguf_key {
            weights.inst_w = gguf_packed_conv1_to_nchw(gguf, key, D_MODEL, D_MODEL)?;
        }
    }
    if weights.sem_w.is_empty() {
        if let Some(key) = &weights.sem_gguf_key {
            weights.sem_w = gguf_packed_conv1_to_nchw(gguf, key, 1, D_MODEL)?;
        }
    }
    Ok(())
}

/// Compile pixel-decoder IR graphs for SAM3 base (72×72 trunk grid).
pub fn compile_segmentation_ir(
    weights: &mut Sam3SegmentationHeadWeights,
    gguf_packed: Option<&GgufPackedParams>,
    trunk_grid: usize,
    device: Device,
    profile: &rlx_flow::CompileProfile,
) -> Result<()> {
    if !weights.loaded {
        return Ok(());
    }
    materialize_segmentation_gguf_weights(weights, gguf_packed)?;

    if !weights.pixel_conv_w[0].is_empty() {
        weights.pixel_steps = compile_pixel_decoder_steps(
            &weights.pixel_conv_w,
            &weights.pixel_conv_b,
            &weights.pixel_gn_w,
            &weights.pixel_gn_b,
            trunk_grid,
            device,
            profile,
        )?;
    }

    let g2 = trunk_grid * 4;
    if let Some(gguf) = gguf_packed {
        if weights.inst_gguf_key.is_some() || !weights.inst_w.is_empty() {
            weights.inst_head = Some(Sam3Conv1x1Compiled::compile_with_gguf(
                D_MODEL,
                D_MODEL,
                g2,
                g2,
                &weights.inst_w,
                &weights.inst_b,
                weights.inst_gguf_key.as_deref(),
                gguf,
                device,
                profile,
            )?);
        }
        if weights.sem_gguf_key.is_some() || !weights.sem_w.is_empty() {
            weights.sem_head = Some(Sam3Conv1x1Compiled::compile_with_gguf(
                D_MODEL,
                1,
                g2,
                g2,
                &weights.sem_w,
                &weights.sem_b,
                weights.sem_gguf_key.as_deref(),
                gguf,
                device,
                profile,
            )?);
        }
    } else {
        if !weights.inst_w.is_empty() {
            weights.inst_head = Some(Sam3Conv1x1Compiled::compile_with_profile(
                D_MODEL,
                D_MODEL,
                g2,
                g2,
                &weights.inst_w,
                &weights.inst_b,
                device,
                profile,
            )?);
        }
        if !weights.sem_w.is_empty() {
            weights.sem_head = Some(Sam3Conv1x1Compiled::compile_with_profile(
                D_MODEL,
                1,
                g2,
                g2,
                &weights.sem_w,
                &weights.sem_b,
                device,
                profile,
            )?);
        }
    }
    Ok(())
}

pub fn extract_dot_product_scoring_weights(
    weights: &mut WeightMap,
    gguf_packed: Option<&GgufPackedParams>,
) -> Result<Sam3DotProductScoringWeights> {
    let base = "detector.dot_prod_scoring";
    let (pm0_t, prompt_mlp_w0_gguf_key) = take_transposed_with_gguf_key(
        weights,
        gguf_packed,
        &format!("{base}.prompt_mlp.layers.0.weight"),
    )?;
    let (pm0_b, _) = take_or_gguf(
        weights,
        gguf_packed,
        &format!("{base}.prompt_mlp.layers.0.bias"),
    )?;
    let (pm1_t, prompt_mlp_w1_gguf_key) = take_transposed_with_gguf_key(
        weights,
        gguf_packed,
        &format!("{base}.prompt_mlp.layers.1.weight"),
    )?;
    let (pm1_b, _) = take_or_gguf(
        weights,
        gguf_packed,
        &format!("{base}.prompt_mlp.layers.1.bias"),
    )?;
    let prompt_mlp = Mlp2 {
        w0_t: pm0_t,
        b0: pm0_b,
        w1_t: pm1_t,
        b1: pm1_b,
        in_dim: D_MODEL,
        hidden: 2048,
        out_dim: D_MODEL,
        w0_gguf_key: prompt_mlp_w0_gguf_key.clone(),
        w1_gguf_key: prompt_mlp_w1_gguf_key.clone(),
    };
    let (pm_norm_w, _) = take_or_gguf(
        weights,
        gguf_packed,
        &format!("{base}.prompt_mlp.out_norm.weight"),
    )?;
    let (pm_norm_b, _) = take_or_gguf(
        weights,
        gguf_packed,
        &format!("{base}.prompt_mlp.out_norm.bias"),
    )?;
    let (pp_t, prompt_proj_gguf_key) =
        take_transposed_with_gguf_key(weights, gguf_packed, &format!("{base}.prompt_proj.weight"))?;
    let (pp_b, _) = take_or_gguf(weights, gguf_packed, &format!("{base}.prompt_proj.bias"))?;
    let (hs_t, hs_proj_gguf_key) =
        take_transposed_with_gguf_key(weights, gguf_packed, &format!("{base}.hs_proj.weight"))?;
    let (hs_b, _) = take_or_gguf(weights, gguf_packed, &format!("{base}.hs_proj.bias"))?;
    Ok(Sam3DotProductScoringWeights {
        loaded: true,
        prompt_mlp,
        prompt_mlp_out_norm_w: pm_norm_w,
        prompt_mlp_out_norm_b: pm_norm_b,
        prompt_proj_w_t: pp_t,
        prompt_proj_b: pp_b,
        hs_proj_w_t: hs_t,
        hs_proj_b: hs_b,
        prompt_mlp_w0_gguf_key,
        prompt_mlp_w1_gguf_key,
        prompt_proj_gguf_key,
        hs_proj_gguf_key,
    })
}

#[derive(Debug, Clone, Default)]
pub struct Sam3SegmentationOutput {
    pub mask_pred: Vec<f32>,
    pub semantic_seg: Vec<f32>,
    pub h_out: usize,
    pub w_out: usize,
    pub num_queries: usize,
}

fn split_in_proj_w(in_proj_w_t: &[f32], embed_dim: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let e = embed_dim;
    let mut wq = vec![0f32; e * e];
    let mut wk = vec![0f32; e * e];
    let mut wv = vec![0f32; e * e];
    for i in 0..e {
        for j in 0..e {
            wq[i * e + j] = in_proj_w_t[i * 3 * e + j];
            wk[i * e + j] = in_proj_w_t[i * 3 * e + e + j];
            wv[i * e + j] = in_proj_w_t[i * 3 * e + 2 * e + j];
        }
    }
    (wq, wk, wv)
}

fn repack_heads(
    flat: &[f32],
    out: &mut [f32],
    batch: usize,
    seq: usize,
    num_heads: usize,
    head_dim: usize,
) {
    for bi in 0..batch {
        for l in 0..seq {
            for h in 0..num_heads {
                let src = (bi * seq + l) * num_heads * head_dim + h * head_dim;
                let dst = (bi * num_heads + h) * seq * head_dim + l * head_dim;
                out[dst..dst + head_dim].copy_from_slice(&flat[src..src + head_dim]);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn cross_attend_prompt(
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
    key_padding_mask: Option<&[u8]>,
) -> Result<Vec<f32>> {
    if in_gguf_key.is_none() && out_gguf_key.is_none() {
        return multihead_attention(
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
            key_padding_mask,
        );
    }
    ensure!(
        embed_dim.is_multiple_of(num_heads),
        "embed_dim {embed_dim} not divisible by num_heads {num_heads}"
    );
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
            narrow_last(qkv_q, rows_q, embed_dim, 0, embed_dim),
            narrow_last(qkv_k, rows_k, embed_dim, embed_dim, embed_dim),
            narrow_last(qkv_v, rows_k, embed_dim, 2 * embed_dim, embed_dim),
        )
    } else {
        let (wq, wk, wv) = split_in_proj_w(in_proj_w_t, embed_dim);
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
    repack_heads(&q_proj, &mut qh, batch, l_q, num_heads, head_dim);
    repack_heads(&k_proj, &mut kh, batch, l_k, num_heads, head_dim);
    repack_heads(&v_proj, &mut vh, batch, l_k, num_heads, head_dim);

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

fn narrow_last(qkv: Vec<f32>, rows: usize, width: usize, start: usize, len: usize) -> Vec<f32> {
    let mut out = vec![0f32; rows * len];
    for r in 0..rows {
        for i in 0..len {
            out[r * len + i] = qkv[r * width + start + i];
        }
    }
    out
}

fn mlp3_forward_gguf(
    mlp: &Mlp3,
    w0_key: Option<&str>,
    w1_key: Option<&str>,
    w2_key: Option<&str>,
    gguf_packed: Option<&GgufPackedParams>,
    x: &[f32],
    rows: usize,
) -> Result<Vec<f32>> {
    let mut h = linear_maybe_gguf(
        x,
        rows,
        mlp.in_dim,
        &mlp.w0_t,
        w0_key,
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
        w1_key,
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
        w2_key,
        gguf_packed,
        mlp.out_dim,
        &mlp.b2,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn forward_segmentation(
    weights: &mut Sam3SegmentationHeadWeights,
    enc_memory_bf: &[f32],
    backbone_fpn: &[Vec<f32>],
    backbone_shapes: &[(usize, usize)],
    obj_queries_last_bf: &[f32],
    prompt_seq_first: &[f32],
    prompt_kpm: &[u8],
    batch: usize,
    enc_h: usize,
    enc_w: usize,
    num_queries: usize,
    seq_len: usize,
    gguf_packed: Option<&GgufPackedParams>,
) -> Result<Sam3SegmentationOutput> {
    ensure!(weights.loaded, "SAM3 segmentation head not loaded");
    ensure!(batch == 1, "batch > 1 not supported yet");
    ensure!(
        backbone_fpn.len() == 3,
        "expected 3 FPN levels (after scalp)"
    );

    let hw = enc_h * enc_w;
    let norm_mem = layer_norm(
        enc_memory_bf,
        &weights.cross_attn_norm_w,
        &weights.cross_attn_norm_b,
        D_MODEL,
        1e-5,
    )?;
    let mut prompt_bf = vec![0f32; batch * seq_len * D_MODEL];
    for b in 0..batch {
        for l in 0..seq_len {
            let s = (l * batch + b) * D_MODEL;
            let d = (b * seq_len + l) * D_MODEL;
            prompt_bf[d..d + D_MODEL].copy_from_slice(&prompt_seq_first[s..s + D_MODEL]);
        }
    }
    let ca = cross_attend_prompt(
        &norm_mem,
        &prompt_bf,
        &prompt_bf,
        &weights.cross_attend_in_w_t,
        &weights.cross_attend_in_b,
        weights.cross_attend_in_gguf_key.as_deref(),
        &weights.cross_attend_out_w_t,
        &weights.cross_attend_out_b,
        weights.cross_attend_out_gguf_key.as_deref(),
        gguf_packed,
        batch,
        hw,
        seq_len,
        D_MODEL,
        N_HEADS,
        Some(prompt_kpm),
    )?;
    let mut enc_refined = enc_memory_bf.to_vec();
    for i in 0..enc_refined.len() {
        enc_refined[i] += ca[i];
    }
    let mut enc_visual = vec![0f32; batch * D_MODEL * hw];
    for b in 0..batch {
        for y in 0..enc_h {
            for xc in 0..enc_w {
                for c in 0..D_MODEL {
                    enc_visual[((b * D_MODEL + c) * enc_h + y) * enc_w + xc] =
                        enc_refined[(b * hw + y * enc_w + xc) * D_MODEL + c];
                }
            }
        }
    }

    let mut levels = backbone_fpn.to_vec();
    levels[2] = enc_visual;
    let mut shapes = backbone_shapes.to_vec();
    shapes[2] = (enc_h, enc_w);

    let mut prev = levels.pop().unwrap();
    let (mut ph, mut pw) = shapes.pop().unwrap();

    if weights.pixel_steps.len() == 2 {
        for (i, (curr, (ch, cw))) in levels.iter().rev().zip(shapes.iter().rev()).enumerate() {
            prev = weights.pixel_steps[i].run(&prev, curr)?;
            ph = *ch;
            pw = *cw;
        }
    } else {
        for (i, (curr, (ch, cw))) in levels.iter().rev().zip(shapes.iter().rev()).enumerate() {
            let up = nearest_upsample_nchw(&prev, D_MODEL, ph, pw, *ch, *cw);
            let mut combined = vec![0f32; curr.len()];
            for j in 0..combined.len() {
                combined[j] = curr[j] + up[j];
            }
            let conv = conv2d_3x3_pad1_maybe_gguf(
                &combined,
                D_MODEL,
                *ch,
                *cw,
                &weights.pixel_conv_w[i],
                weights.pixel_conv_gguf_keys[i].as_deref(),
                gguf_packed,
                &weights.pixel_conv_b[i],
                &mut weights.pixel_conv_nchw_cache[i],
            )?;
            let mut relud = group_norm(
                &conv,
                batch,
                D_MODEL,
                *ch,
                *cw,
                8,
                &weights.pixel_gn_w[i],
                &weights.pixel_gn_b[i],
            );
            for v in relud.iter_mut() {
                if *v < 0.0 {
                    *v = 0.0;
                }
            }
            prev = relud;
            ph = *ch;
            pw = *cw;
        }
    }
    let pixel_embed = prev;

    let inst = if let Some(ref mut head) = weights.inst_head {
        head.run(&pixel_embed)?
    } else {
        conv2d_1x1_maybe_gguf(
            &pixel_embed,
            D_MODEL,
            D_MODEL,
            ph,
            pw,
            &weights.inst_w,
            weights.inst_gguf_key.as_deref(),
            gguf_packed,
            &weights.inst_b,
        )?
    };

    let mask_embed_out = mlp3_forward_gguf(
        &weights.mask_embed,
        weights.mask_embed_w0_gguf_key.as_deref(),
        weights.mask_embed_w1_gguf_key.as_deref(),
        weights.mask_embed_w2_gguf_key.as_deref(),
        gguf_packed,
        obj_queries_last_bf,
        batch * num_queries,
    )?;
    let mut mask_pred = vec![0f32; batch * num_queries * ph * pw];
    for b in 0..batch {
        for q in 0..num_queries {
            for c in 0..D_MODEL {
                let qcoeff = mask_embed_out[(b * num_queries + q) * D_MODEL + c];
                if qcoeff == 0.0 {
                    continue;
                }
                let plane =
                    &inst[((b * D_MODEL + c) * ph * pw)..((b * D_MODEL + c) * ph * pw + ph * pw)];
                let dst = &mut mask_pred
                    [(b * num_queries + q) * ph * pw..(b * num_queries + q + 1) * ph * pw];
                for p in 0..ph * pw {
                    dst[p] += qcoeff * plane[p];
                }
            }
        }
    }

    let semantic_seg = if let Some(ref mut head) = weights.sem_head {
        head.run(&pixel_embed)?
    } else {
        conv2d_1x1_maybe_gguf(
            &pixel_embed,
            D_MODEL,
            1,
            ph,
            pw,
            &weights.sem_w,
            weights.sem_gguf_key.as_deref(),
            gguf_packed,
            &weights.sem_b,
        )?
    };

    Ok(Sam3SegmentationOutput {
        mask_pred,
        semantic_seg,
        h_out: ph,
        w_out: pw,
        num_queries,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn forward_dot_prod_scoring(
    weights: &Sam3DotProductScoringWeights,
    hs_bf: &[f32],
    prompt_seq_first: &[f32],
    prompt_kpm: &[u8],
    num_layers: usize,
    batch: usize,
    num_queries: usize,
    seq_len: usize,
    gguf_packed: Option<&GgufPackedParams>,
) -> Result<Vec<f32>> {
    ensure!(weights.loaded, "SAM3 dot product scoring not loaded");
    let rows = seq_len * batch;
    let pm = &weights.prompt_mlp;
    let mut h = linear_maybe_gguf(
        prompt_seq_first,
        rows,
        pm.in_dim,
        &pm.w0_t,
        weights.prompt_mlp_w0_gguf_key.as_deref(),
        gguf_packed,
        pm.hidden,
        &pm.b0,
    )?;
    for v in h.iter_mut() {
        if *v < 0.0 {
            *v = 0.0;
        }
    }
    h = linear_maybe_gguf(
        &h,
        rows,
        pm.hidden,
        &pm.w1_t,
        weights.prompt_mlp_w1_gguf_key.as_deref(),
        gguf_packed,
        pm.out_dim,
        &pm.b1,
    )?;
    for i in 0..h.len() {
        h[i] += prompt_seq_first[i];
    }
    let h = layer_norm(
        &h,
        &weights.prompt_mlp_out_norm_w,
        &weights.prompt_mlp_out_norm_b,
        D_MODEL,
        1e-5,
    )?;

    let mut pooled = vec![0f32; batch * D_MODEL];
    let mut counts = vec![0.0f32; batch];
    for b in 0..batch {
        for l in 0..seq_len {
            if prompt_kpm[b * seq_len + l] == 0 {
                let src = (l * batch + b) * D_MODEL;
                let dst = b * D_MODEL;
                for c in 0..D_MODEL {
                    pooled[dst + c] += h[src + c];
                }
                counts[b] += 1.0;
            }
        }
    }
    for b in 0..batch {
        let denom = counts[b].max(1.0);
        for c in 0..D_MODEL {
            pooled[b * D_MODEL + c] /= denom;
        }
    }

    let proj_pooled = linear_maybe_gguf(
        &pooled,
        batch,
        D_MODEL,
        &weights.prompt_proj_w_t,
        weights.prompt_proj_gguf_key.as_deref(),
        gguf_packed,
        D_MODEL,
        &weights.prompt_proj_b,
    )?;
    let proj_hs = linear_maybe_gguf(
        hs_bf,
        num_layers * batch * num_queries,
        D_MODEL,
        &weights.hs_proj_w_t,
        weights.hs_proj_gguf_key.as_deref(),
        gguf_packed,
        D_MODEL,
        &weights.hs_proj_b,
    )?;

    let scale = 1.0f32 / (D_MODEL as f32).sqrt();
    let clamp = 12.0f32;
    let mut scores = vec![0f32; num_layers * batch * num_queries];
    for l in 0..num_layers {
        for b in 0..batch {
            let pp = &proj_pooled[b * D_MODEL..(b + 1) * D_MODEL];
            for q in 0..num_queries {
                let row = &proj_hs[((l * batch + b) * num_queries + q) * D_MODEL
                    ..((l * batch + b) * num_queries + q + 1) * D_MODEL];
                let mut acc = 0.0f32;
                for c in 0..D_MODEL {
                    acc += row[c] * pp[c];
                }
                let s = (acc * scale).clamp(-clamp, clamp);
                scores[(l * batch + b) * num_queries + q] = s;
            }
        }
    }
    Ok(scores)
}

fn nearest_upsample_nchw(
    x: &[f32],
    c: usize,
    src_h: usize,
    src_w: usize,
    dst_h: usize,
    dst_w: usize,
) -> Vec<f32> {
    let mut out = vec![0f32; c * dst_h * dst_w];
    for cc in 0..c {
        let inp = &x[cc * src_h * src_w..(cc + 1) * src_h * src_w];
        let oup = &mut out[cc * dst_h * dst_w..(cc + 1) * dst_h * dst_w];
        for y in 0..dst_h {
            let sy = y * src_h / dst_h;
            for x in 0..dst_w {
                let sx = x * src_w / dst_w;
                oup[y * dst_w + x] = inp[sy * src_w + sx];
            }
        }
    }
    out
}

fn conv2d_3x3_pad1_maybe_gguf(
    input: &[f32],
    c: usize,
    h: usize,
    w: usize,
    weight: &[f32],
    weight_gguf_key: Option<&str>,
    gguf_packed: Option<&GgufPackedParams>,
    bias: &[f32],
    nchw_cache: &mut Option<Vec<f32>>,
) -> Result<Vec<f32>> {
    if !weight.is_empty() {
        return Ok(conv2d_3x3_nchw_pad1(input, c, h, w, weight, bias));
    }
    let key = weight_gguf_key
        .ok_or_else(|| anyhow::anyhow!("conv3: missing F32 weights and GGUF key"))?;
    let p = gguf_packed
        .and_then(|m| packed_linear(m, key))
        .ok_or_else(|| anyhow::anyhow!("missing packed conv3 weight: {key}"))?;
    conv2d_3x3_nchw_gguf(input, c, h, w, p, bias, nchw_cache)
}

fn conv2d_1x1(
    input: &[f32],
    in_c: usize,
    out_c: usize,
    h: usize,
    w: usize,
    weight: &[f32],
    bias: &[f32],
) -> Vec<f32> {
    let n = h * w;
    let mut out = vec![0f32; out_c * n];
    rlx_cpu::blas::sgemm(weight, input, &mut out, out_c, in_c, n);
    for oc in 0..out_c {
        let b = bias[oc];
        let row = &mut out[oc * n..(oc + 1) * n];
        for v in row {
            *v += b;
        }
    }
    out
}

fn conv2d_1x1_maybe_gguf(
    input: &[f32],
    in_c: usize,
    out_c: usize,
    h: usize,
    w: usize,
    weight: &[f32],
    weight_gguf_key: Option<&str>,
    gguf_packed: Option<&GgufPackedParams>,
    bias: &[f32],
) -> Result<Vec<f32>> {
    if weight_gguf_key.is_none() {
        return Ok(conv2d_1x1(input, in_c, out_c, h, w, weight, bias));
    }
    let n = h * w;
    let mut rows = vec![0f32; n * in_c];
    for ic in 0..in_c {
        for p in 0..n {
            rows[p * in_c + ic] = input[ic * n + p];
        }
    }
    let flat = linear_maybe_gguf(
        &rows,
        n,
        in_c,
        weight,
        weight_gguf_key,
        gguf_packed,
        out_c,
        bias,
    )?;
    let mut out = vec![0f32; out_c * n];
    for oc in 0..out_c {
        for p in 0..n {
            out[oc * n + p] = flat[p * out_c + oc];
        }
    }
    Ok(out)
}

fn group_norm(
    x: &[f32],
    batch: usize,
    channels: usize,
    h: usize,
    w: usize,
    num_groups: usize,
    gamma: &[f32],
    beta: &[f32],
) -> Vec<f32> {
    assert!(channels.is_multiple_of(num_groups));
    let cpg = channels / num_groups;
    let spatial = h * w;
    let mut out = vec![0f32; batch * channels * spatial];
    for b in 0..batch {
        for g in 0..num_groups {
            let c0 = g * cpg;
            let n = (cpg * spatial) as f32;
            let mut mean = 0.0f32;
            for c in 0..cpg {
                let plane = &x
                    [((b * channels + c0 + c) * spatial)..((b * channels + c0 + c + 1) * spatial)];
                for v in plane {
                    mean += *v;
                }
            }
            mean /= n;
            let mut var = 0.0f32;
            for c in 0..cpg {
                let plane = &x
                    [((b * channels + c0 + c) * spatial)..((b * channels + c0 + c + 1) * spatial)];
                for v in plane {
                    let d = *v - mean;
                    var += d * d;
                }
            }
            var /= n;
            let inv = 1.0 / (var + 1e-5).sqrt();
            for c in 0..cpg {
                let src = &x
                    [((b * channels + c0 + c) * spatial)..((b * channels + c0 + c + 1) * spatial)];
                let dst = &mut out
                    [((b * channels + c0 + c) * spatial)..((b * channels + c0 + c + 1) * spatial)];
                let g_ = gamma[c0 + c];
                let bias = beta[c0 + c];
                for (s, d) in src.iter().zip(dst.iter_mut()) {
                    *d = (*s - mean) * inv * g_ + bias;
                }
            }
        }
    }
    out
}

/// Legacy stub used by the not-yet-finished `Sam3::predict_image` path.
pub fn segmentation_forward_native(
    _weights: &Sam3SegmentationHeadWeights,
    detector: &Sam3DetectorOutput,
    h_out: usize,
    w_out: usize,
) -> Sam3ImagePrediction {
    Sam3ImagePrediction {
        masks: vec![0.0; detector.num_queries * h_out * w_out],
        mask_shape: vec![detector.num_queries, h_out, w_out],
        boxes: vec![0.0; detector.num_queries * 4],
        boxes_shape: vec![detector.num_queries, 4],
        scores: vec![0.0; detector.num_queries],
        scores_shape: vec![detector.num_queries],
        num_instances: detector.num_queries,
        h_out,
        w_out,
    }
}
