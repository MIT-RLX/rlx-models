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

//! SAM 2 prompt encoder — Fourier/point embeddings host-side; mask stack via IR.
//!
//! Mirrors `sam2/modeling/sam/prompt_encoder.py::PromptEncoder` exactly.
//! Structurally identical to SAM v1 (random-Fourier positional
//! encoding + per-label embeddings + 2-stage Conv mask downscale +
//! `no_mask_embed` broadcast). Two integration differences vs v1:
//!
//!   1. Weight key prefix is `sam_prompt_encoder.*` instead of
//!      `prompt_encoder.*` — SAM 2 nests the prompt encoder under
//!      `sam_prompt_encoder` inside the published checkpoints.
//!   2. The embedding grid resolution comes from the *finest* FPN
//!      level (stride 16, 64×64 for 1024 input) — the reference's
//!      `image_embedding_size = (64, 64)`. Constant for every Hiera
//!      variant.
//!
//! The prompt encoder is < 1 % of total compute, so keeping it on the
//! CPU keeps Phase 2 self-contained (no IR-surface growth for
//! Gaussian-PE / Conv2d k=2 s=2 / etc.).

use super::config::SAM2_IMG_SIZE;
use super::prompt_mask_ir::Sam2PromptMaskCompiled;
use anyhow::{Result, ensure};
use rlx_core::weight_map::WeightMap;

/// Spatial resolution of the dense image embedding fed to the mask
/// decoder. SAM 2 hardcodes 64×64 (stride 16 on a 1024 input). This
/// matches `image_embedding_size=(64, 64)` in every published
/// `sam2_hiera_*.yaml`.
pub const SAM2_PROMPT_GRID: usize = 64;

/// `mask_in_chans` per the reference YAML — fixed at 16 across all
/// Hiera variants, same as SAM v1.
pub const SAM2_MASK_IN_CHANS: usize = 16;

/// All weights consumed by [`prompt_encoder_forward`]. Loaded once
/// from the safetensors file and then reused per prompt.
pub struct Sam2PromptEncoderWeights {
    /// `[2, embed_dim/2]` Gaussian random projection used by the
    /// Fourier positional encoder.
    pub pe_gaussian: Vec<f32>,
    /// `[embed_dim]` learned token for "not a point" padding label
    /// (used when there are points but no boxes — labels of -1).
    pub not_a_point_embed: Vec<f32>,
    /// `[4, embed_dim]` learned per-label embeddings:
    /// 0 → background point, 1 → foreground point,
    /// 2 → box top-left corner, 3 → box bottom-right corner.
    pub point_embeddings: Vec<f32>,
    /// Mask downscaling stack (Conv2d → LN2d → GELU → Conv2d → LN2d
    /// → GELU → Conv2d).
    pub mask_conv1_w: Vec<f32>,
    pub mask_conv1_b: Vec<f32>,
    pub mask_ln1_g: Vec<f32>,
    pub mask_ln1_b: Vec<f32>,
    pub mask_conv2_w: Vec<f32>,
    pub mask_conv2_b: Vec<f32>,
    pub mask_ln2_g: Vec<f32>,
    pub mask_ln2_b: Vec<f32>,
    pub mask_conv3_w: Vec<f32>,
    pub mask_conv3_b: Vec<f32>,
    /// `[embed_dim]` learned token broadcast over the image grid when
    /// no mask prompt is supplied.
    pub no_mask_embed: Vec<f32>,
    pub embed_dim: usize,
    /// `mask_in_chans` (16 for all SAM 2 variants).
    pub mask_in_chans: usize,
    /// Grid edge length (64 for SAM 2's stride-16 path).
    pub grid: usize,
}

/// Drain the prompt-encoder weights from the safetensors map. Returns
/// `Sam2PromptEncoderWeights` and consumes the corresponding keys.
pub fn extract_prompt_encoder_weights(
    weights: &mut WeightMap,
    embed_dim: usize,
    mask_in_chans: usize,
) -> Result<Sam2PromptEncoderWeights> {
    let half = embed_dim / 2;
    let (pe_gaussian, sh) =
        weights.take("sam_prompt_encoder.pe_layer.positional_encoding_gaussian_matrix")?;
    ensure!(
        sh == vec![2, half],
        "pe_gaussian expected [2, {half}], got {sh:?}"
    );

    let (not_a_point_embed, _) = weights.take("sam_prompt_encoder.not_a_point_embed.weight")?;
    let (no_mask_embed, _) = weights.take("sam_prompt_encoder.no_mask_embed.weight")?;

    let mut point_embeddings = vec![0f32; 4 * embed_dim];
    for i in 0..4 {
        let (data, _) = weights.take(&format!("sam_prompt_encoder.point_embeddings.{i}.weight"))?;
        point_embeddings[i * embed_dim..(i + 1) * embed_dim].copy_from_slice(&data);
    }

    let q = mask_in_chans / 4;
    let (mask_conv1_w, sh1) = weights.take("sam_prompt_encoder.mask_downscaling.0.weight")?;
    ensure!(
        sh1 == vec![q, 1, 2, 2],
        "mask_downscaling.0.weight expected [{q}, 1, 2, 2], got {sh1:?}"
    );
    let (mask_conv1_b, _) = weights.take("sam_prompt_encoder.mask_downscaling.0.bias")?;
    let (mask_ln1_g, _) = weights.take("sam_prompt_encoder.mask_downscaling.1.weight")?;
    let (mask_ln1_b, _) = weights.take("sam_prompt_encoder.mask_downscaling.1.bias")?;

    let (mask_conv2_w, sh2) = weights.take("sam_prompt_encoder.mask_downscaling.3.weight")?;
    ensure!(
        sh2 == vec![mask_in_chans, q, 2, 2],
        "mask_downscaling.3.weight expected [{mask_in_chans}, {q}, 2, 2], got {sh2:?}"
    );
    let (mask_conv2_b, _) = weights.take("sam_prompt_encoder.mask_downscaling.3.bias")?;
    let (mask_ln2_g, _) = weights.take("sam_prompt_encoder.mask_downscaling.4.weight")?;
    let (mask_ln2_b, _) = weights.take("sam_prompt_encoder.mask_downscaling.4.bias")?;

    let (mask_conv3_w, sh3) = weights.take("sam_prompt_encoder.mask_downscaling.6.weight")?;
    ensure!(
        sh3 == vec![embed_dim, mask_in_chans, 1, 1],
        "mask_downscaling.6.weight expected [{embed_dim}, {mask_in_chans}, 1, 1], got {sh3:?}"
    );
    let (mask_conv3_b, _) = weights.take("sam_prompt_encoder.mask_downscaling.6.bias")?;

    Ok(Sam2PromptEncoderWeights {
        pe_gaussian,
        not_a_point_embed,
        point_embeddings,
        mask_conv1_w,
        mask_conv1_b,
        mask_ln1_g,
        mask_ln1_b,
        mask_conv2_w,
        mask_conv2_b,
        mask_ln2_g,
        mask_ln2_b,
        mask_conv3_w,
        mask_conv3_b,
        no_mask_embed,
        embed_dim,
        mask_in_chans,
        grid: SAM2_PROMPT_GRID,
    })
}

/// Output of [`prompt_encoder_forward`] — fed straight into the mask
/// decoder. All host-side `Vec<f32>`.
pub struct Sam2PromptEncoderOutput {
    /// `[num_tokens, embed_dim]` — concatenation of point and box
    /// embeddings. `num_tokens = 0` for the "no prompt" case.
    pub sparse_embeddings: Vec<f32>,
    pub num_sparse_tokens: usize,
    /// `[embed_dim, grid, grid]` — dense pixel-wise embedding.
    pub dense_embeddings: Vec<f32>,
    /// `[embed_dim, grid, grid]` — image positional encoding (the
    /// dense PE fed into the mask decoder).
    pub image_pe: Vec<f32>,
}

/// Run the SAM 2 prompt encoder. Mirrors
/// `sam2.modeling.sam.prompt_encoder.PromptEncoder.forward`.
///
/// `points`: optional `(coords, labels)` where coords is `[N, 2]`
///   (x, y in input-image pixels, 0..`SAM2_IMG_SIZE`) and labels is
///   `[N]` (1 = foreground, 0 = background, -1 = padding).
/// `boxes`: optional `[M, 4]` boxes (x0, y0, x1, y1).
/// `masks`: optional `[1, 4·grid, 4·grid]` mask prompt (a logits map
///   at 4× the embedding resolution); 256×256 for SAM 2's 64×64 grid.
pub fn prompt_encoder_forward(
    w: &Sam2PromptEncoderWeights,
    mask_stack: &mut Sam2PromptMaskCompiled,
    points: Option<(&[f32], &[f32])>,
    boxes: Option<&[f32]>,
    masks: Option<&[f32]>,
) -> Result<Sam2PromptEncoderOutput> {
    let e = w.embed_dim;
    let g = w.grid;

    // ── Sparse embeddings ──
    let pad_points = boxes.is_none();
    let mut sparse = Vec::new();

    if let Some((coords, labels)) = points {
        let n = labels.len();
        ensure!(
            coords.len() == n * 2,
            "points coords len {} ≠ N·2 ({}·2)",
            coords.len(),
            n
        );
        let mut pts: Vec<f32> = coords.iter().map(|c| c + 0.5).collect();
        let mut lbls = labels.to_vec();
        if pad_points {
            pts.push(0.0);
            pts.push(0.0);
            lbls.push(-1.0);
        }
        let n_padded = lbls.len();
        let emb = embed_points_and_boxes(w, &pts, n_padded, /*is_box=*/ false, Some(&lbls))?;
        sparse.extend_from_slice(&emb);
    }
    if let Some(box_coords) = boxes {
        let m = box_coords.len() / 4;
        ensure!(box_coords.len() == m * 4, "boxes len must be multiple of 4");
        let coords_with_half: Vec<f32> = box_coords.iter().map(|c| c + 0.5).collect();
        let emb = embed_points_and_boxes(w, &coords_with_half, m * 2, /*is_box=*/ true, None)?;
        sparse.extend_from_slice(&emb);
    }
    let num_sparse_tokens = if sparse.is_empty() {
        0
    } else {
        sparse.len() / e
    };

    // ── Dense embeddings ──
    let dense_embeddings = match masks {
        Some(m) => mask_stack.run(m)?,
        None => {
            // Broadcast no_mask_embed [E] to [E, g, g].
            let mut out = vec![0f32; e * g * g];
            for c in 0..e {
                let v = w.no_mask_embed[c];
                out[c * g * g..(c + 1) * g * g].fill(v);
            }
            out
        }
    };

    // ── Image PE: random-Fourier encoding of a g·g normalised grid ──
    let image_pe = compute_image_pe(w, g, g);

    Ok(Sam2PromptEncoderOutput {
        sparse_embeddings: sparse,
        num_sparse_tokens,
        dense_embeddings,
        image_pe,
    })
}

/// Random-Fourier positional encoding for a `(h, w)` grid.
/// Output shape `[embed_dim, h, w]`.
pub fn compute_image_pe(w: &Sam2PromptEncoderWeights, h: usize, ww: usize) -> Vec<f32> {
    let e = w.embed_dim;
    let half = e / 2;
    let mut out = vec![0f32; e * h * ww];
    for y in 0..h {
        let fy = (y as f32 + 0.5) / h as f32;
        for x in 0..ww {
            let fx = (x as f32 + 0.5) / ww as f32;
            let cx = fx * 2.0 - 1.0;
            let cy = fy * 2.0 - 1.0;
            for k in 0..half {
                let mut acc = cx * w.pe_gaussian[k] + cy * w.pe_gaussian[half + k];
                acc *= 2.0 * std::f32::consts::PI;
                out[k * h * ww + y * ww + x] = acc.sin();
                out[(half + k) * h * ww + y * ww + x] = acc.cos();
            }
        }
    }
    out
}

/// Apply the Gaussian + sin/cos PE to coords already in `[0, 1]`.
/// Returns `[N, embed_dim]`.
fn pe_encode_normalized(w: &Sam2PromptEncoderWeights, coords: &[f32], n: usize) -> Vec<f32> {
    let e = w.embed_dim;
    let half = e / 2;
    let mut out = vec![0f32; n * e];
    for i in 0..n {
        let cx = coords[i * 2] * 2.0 - 1.0;
        let cy = coords[i * 2 + 1] * 2.0 - 1.0;
        for k in 0..half {
            let mut acc = cx * w.pe_gaussian[k] + cy * w.pe_gaussian[half + k];
            acc *= 2.0 * std::f32::consts::PI;
            out[i * e + k] = acc.sin();
            out[i * e + half + k] = acc.cos();
        }
    }
    out
}

/// Embed N points or N boxes (each box becomes 2 corner points).
fn embed_points_and_boxes(
    w: &Sam2PromptEncoderWeights,
    coords_in_pixels: &[f32],
    n: usize,
    is_box: bool,
    labels: Option<&[f32]>,
) -> Result<Vec<f32>> {
    let e = w.embed_dim;
    let img = SAM2_IMG_SIZE as f32;
    let normed: Vec<f32> = coords_in_pixels.iter().map(|c| c / img).collect();
    let mut emb = pe_encode_normalized(w, &normed, n);

    if is_box {
        for i in 0..n {
            let pe_idx = if i % 2 == 0 { 2 } else { 3 };
            for k in 0..e {
                emb[i * e + k] += w.point_embeddings[pe_idx * e + k];
            }
        }
    } else if let Some(lbls) = labels {
        ensure!(lbls.len() == n, "labels len {} ≠ n {n}", lbls.len());
        for i in 0..n {
            let label = lbls[i];
            if label < 0.0 {
                for k in 0..e {
                    emb[i * e + k] = w.not_a_point_embed[k];
                }
            } else if label == 0.0 {
                for k in 0..e {
                    emb[i * e + k] += w.point_embeddings[k];
                }
            } else {
                for k in 0..e {
                    emb[i * e + k] += w.point_embeddings[e + k];
                }
            }
        }
    }
    Ok(emb)
}

// ─── Shared host-side helpers (also re-used by mask_decoder.rs / memory_encoder.rs) ────────

/// 2-D conv with kernel=2 stride=2 padding=0, NCHW.
#[allow(dead_code)]
pub(super) fn conv2d_stride2_k2_pad0(
    input: &[f32],
    in_c: usize,
    out_c: usize,
    in_h: usize,
    in_w: usize,
    weight: &[f32], // [out_c, in_c, 2, 2]
    bias: &[f32],   // [out_c]
) -> Vec<f32> {
    let out_h = in_h / 2;
    let out_w = in_w / 2;
    let mut out = vec![0f32; out_c * out_h * out_w];
    for oc in 0..out_c {
        for oy in 0..out_h {
            for ox in 0..out_w {
                let mut acc = bias[oc];
                for ic in 0..in_c {
                    for ky in 0..2 {
                        let iy = oy * 2 + ky;
                        for kx in 0..2 {
                            let ix = ox * 2 + kx;
                            let v = input[ic * in_h * in_w + iy * in_w + ix];
                            let w_idx = ((oc * in_c + ic) * 2 + ky) * 2 + kx;
                            acc += v * weight[w_idx];
                        }
                    }
                }
                out[oc * out_h * out_w + oy * out_w + ox] = acc;
            }
        }
    }
    out
}

/// 1×1 Conv2d = per-pixel matmul.
pub(super) fn conv2d_1x1(
    input: &[f32],
    in_c: usize,
    out_c: usize,
    h: usize,
    w: usize,
    weight: &[f32], // [out_c, in_c, 1, 1]
    bias: &[f32],   // [out_c]
) -> Vec<f32> {
    let mut out = vec![0f32; out_c * h * w];
    for oc in 0..out_c {
        let b = bias[oc];
        for y in 0..h {
            for x in 0..w {
                let mut acc = b;
                for ic in 0..in_c {
                    acc += input[ic * h * w + y * w + x] * weight[oc * in_c + ic];
                }
                out[oc * h * w + y * w + x] = acc;
            }
        }
    }
    out
}

/// LayerNorm over the channel axis of NCHW (per spatial pos).
pub(super) fn layernorm2d_nchw(
    data: &mut [f32],
    c: usize,
    h: usize,
    w: usize,
    gamma: &[f32],
    beta: &[f32],
    eps: f32,
) {
    let n = h * w;
    for i in 0..n {
        let mut mean = 0f32;
        for k in 0..c {
            mean += data[k * n + i];
        }
        mean /= c as f32;
        let mut var = 0f32;
        for k in 0..c {
            let d = data[k * n + i] - mean;
            var += d * d;
        }
        var /= c as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for k in 0..c {
            let v = (data[k * n + i] - mean) * inv;
            data[k * n + i] = v * gamma[k] + beta[k];
        }
    }
}

/// Exact erf-based GELU (matches `nn.GELU()` default).
pub(super) fn gelu_erf_inplace(data: &mut [f32]) {
    const INV_SQRT2: f32 = std::f32::consts::FRAC_1_SQRT_2;
    for v in data.iter_mut() {
        let x = *v;
        let s = (x * INV_SQRT2).abs();
        let p = 0.327_591_1;
        let a1 = 0.254_829_6;
        let a2 = -0.284_496_7;
        let a3 = 1.421_413_8;
        let a4 = -1.453_152_1;
        let a5 = 1.061_405_4;
        let t = 1.0 / (1.0 + p * s);
        let y = ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t;
        let erf_abs = 1.0 - y * (-s * s).exp();
        let erf = if x >= 0.0 { erf_abs } else { -erf_abs };
        *v = 0.5 * x * (1.0 + erf);
    }
}

/// Sigmoid in place.
pub(super) fn sigmoid_inplace(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = 1.0 / (1.0 + (-*v).exp());
    }
}
