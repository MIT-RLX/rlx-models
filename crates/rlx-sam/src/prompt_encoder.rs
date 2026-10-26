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

//! SAM v1 prompt encoder — Fourier/point embeddings host-side; mask
//! downscaling stack compiled via [`super::prompt_mask_ir`].
//!
//! Mirrors `candle-transformers/src/models/segment_anything/prompt_encoder.rs`
//! exactly so the same `lmz/candle-sam` safetensors checkpoint loads
//! without remapping.

use super::config::{SAM_EMBED_HW, SAM_PROMPT_EMBED_DIM};
use super::prompt_mask_ir::SamPromptMaskCompiled;
use anyhow::{Result, ensure};
use rlx_core::weight_map::WeightMap;

/// All weights consumed by `PromptEncoder::forward`. Loaded once
/// from the safetensors file and then reused per prompt.
pub struct PromptEncoderWeights {
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
    /// `mask_in_chans` from candle's `Sam::new` (= 16 for ViT-B).
    pub mask_in_chans: usize,
}

impl PromptEncoderWeights {
    /// What the point and box embeddings read ([`super::points`]).
    pub fn points(&self) -> super::points::PointWeights<'_> {
        super::points::PointWeights {
            embed_dim: self.embed_dim,
            pe_gaussian: &self.pe_gaussian,
            not_a_point_embed: &self.not_a_point_embed,
            point_embeddings: &self.point_embeddings,
        }
    }
}

pub(super) fn extract_prompt_encoder_weights(
    weights: &mut WeightMap,
    embed_dim: usize,
    mask_in_chans: usize,
) -> Result<PromptEncoderWeights> {
    let half = embed_dim / 2;
    let (pe_gaussian, sh) =
        weights.take("prompt_encoder.pe_layer.positional_encoding_gaussian_matrix")?;
    ensure!(
        sh == vec![2, half],
        "pe_gaussian expected [2, {half}], got {sh:?}"
    );

    let (not_a_point_embed, _) = weights.take("prompt_encoder.not_a_point_embed.weight")?;
    let (no_mask_embed, _) = weights.take("prompt_encoder.no_mask_embed.weight")?;

    // 4 separate point embeddings: indices 0..4. Each `[1, embed_dim]`.
    let mut point_embeddings = vec![0f32; 4 * embed_dim];
    for i in 0..4 {
        let (data, _) = weights.take(&format!("prompt_encoder.point_embeddings.{i}.weight"))?;
        point_embeddings[i * embed_dim..(i + 1) * embed_dim].copy_from_slice(&data);
    }

    let q = mask_in_chans / 4;
    let (mask_conv1_w, sh1) = weights.take("prompt_encoder.mask_downscaling.0.weight")?;
    ensure!(
        sh1 == vec![q, 1, 2, 2],
        "mask_downscaling.0.weight expected [{q}, 1, 2, 2], got {sh1:?}"
    );
    let (mask_conv1_b, _) = weights.take("prompt_encoder.mask_downscaling.0.bias")?;
    let (mask_ln1_g, _) = weights.take("prompt_encoder.mask_downscaling.1.weight")?;
    let (mask_ln1_b, _) = weights.take("prompt_encoder.mask_downscaling.1.bias")?;

    let (mask_conv2_w, sh2) = weights.take("prompt_encoder.mask_downscaling.3.weight")?;
    ensure!(
        sh2 == vec![mask_in_chans, q, 2, 2],
        "mask_downscaling.3.weight expected [{mask_in_chans}, {q}, 2, 2], got {sh2:?}"
    );
    let (mask_conv2_b, _) = weights.take("prompt_encoder.mask_downscaling.3.bias")?;
    let (mask_ln2_g, _) = weights.take("prompt_encoder.mask_downscaling.4.weight")?;
    let (mask_ln2_b, _) = weights.take("prompt_encoder.mask_downscaling.4.bias")?;

    let (mask_conv3_w, sh3) = weights.take("prompt_encoder.mask_downscaling.6.weight")?;
    ensure!(
        sh3 == vec![embed_dim, mask_in_chans, 1, 1],
        "mask_downscaling.6.weight expected [{embed_dim}, {mask_in_chans}, 1, 1], got {sh3:?}"
    );
    let (mask_conv3_b, _) = weights.take("prompt_encoder.mask_downscaling.6.bias")?;

    Ok(PromptEncoderWeights {
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
    })
}

/// Output of `PromptEncoder::forward` — fed straight into the mask
/// decoder. All host-side `Vec<f32>`.
pub struct PromptEncoderOutput {
    /// `[num_tokens, embed_dim]` — concatenation of point and box
    /// embeddings. `num_tokens = 0` for the "no prompt" case.
    pub sparse_embeddings: Vec<f32>,
    pub num_sparse_tokens: usize,
    /// `[embed_dim, hw, hw]` — dense pixel-wise embedding. Either the
    /// mask-downscaled signal or the broadcast `no_mask_embed`.
    pub dense_embeddings: Vec<f32>,
    /// `[embed_dim, hw, hw]` — image positional encoding (the dense PE
    /// fed into the mask decoder).
    pub image_pe: Vec<f32>,
}

/// Run the prompt encoder. Mirrors candle's `PromptEncoder::forward`.
///
/// `points`: optional `(coords, labels)` where coords is `[N, 2]`
///   (x, y in input-image pixels, 0..`SAM_IMG_SIZE`) and labels is
///   `[N]` (1 = foreground, 0 = background, -1 = padding).
/// `boxes`: optional `[M, 4]` boxes (x0, y0, x1, y1).
/// `masks`: optional `[1, 4·hw, 4·hw]` mask prompt (a logits map
///   at 4× the embedding resolution); pre-resized to 256×256 for
///   ViT-B (where hw=64).
pub fn prompt_encoder_forward(
    w: &PromptEncoderWeights,
    mask_stack: &mut SamPromptMaskCompiled,
    points: Option<(&[f32], &[f32])>,
    boxes: Option<&[f32]>,
    masks: Option<&[f32]>,
) -> Result<PromptEncoderOutput> {
    let e = w.embed_dim;
    let hw = SAM_EMBED_HW;

    // ── Sparse embeddings ─────────────────────────────────────────
    let sparse = super::points::sparse_embeddings(&w.points(), points, boxes)?;
    let num_sparse_tokens = sparse.len() / e;

    // ── Dense embeddings ──────────────────────────────────────────
    let dense_embeddings = match masks {
        Some(m) => embed_mask(mask_stack, m, hw)?,
        None => {
            // Broadcast no_mask_embed [E] to [E, hw, hw].
            let mut out = vec![0f32; e * hw * hw];
            for c in 0..e {
                let v = w.no_mask_embed[c];
                let plane = &mut out[c * hw * hw..(c + 1) * hw * hw];
                plane.fill(v);
            }
            out
        }
    };

    // ── Image PE: random-Fourier encoding of a hw·hw normalized grid ──
    let image_pe = super::points::image_pe(&w.points(), hw, hw);

    Ok(PromptEncoderOutput {
        sparse_embeddings: sparse,
        num_sparse_tokens,
        dense_embeddings,
        image_pe,
    })
}

/// Mask downscaling: Conv(k=2, s=2) → LN2d → GELU → Conv(k=2, s=2)
/// → LN2d → GELU → Conv(k=1) → `[embed_dim, hw, hw]`.
///
/// Input `mask`: `[1, 4·hw, 4·hw]` (256×256 for ViT-B).
fn embed_mask(stack: &mut SamPromptMaskCompiled, mask: &[f32], hw: usize) -> Result<Vec<f32>> {
    let in_h = 4 * hw;
    let in_w = 4 * hw;
    ensure!(
        mask.len() == in_h * in_w,
        "mask must be [1, {in_h}, {in_w}], got len {}",
        mask.len()
    );
    // NCHW `[1, 1, H, W]` for the compiled mask-downscaling graph.
    stack.run(mask, in_h, in_w)
}

// ─── Tiny CPU kernels (host-side) ────────────────────────────────

/// 2-D conv with kernel=2 stride=2 padding=0, NCHW.
/// Each input pixel produces a 2×2 patch contribution that doesn't
/// overlap with neighbours. Equivalent to im2col + matmul but kept
/// inline for clarity.
#[allow(dead_code)]
fn conv2d_stride2_k2_pad0(
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
#[allow(dead_code)]
fn conv2d_1x1(
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
/// Matches candle's `LayerNorm2d`.
#[allow(dead_code)]
fn layernorm2d_nchw(
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

/// Exact erf-based GELU (candle's `Activation::Gelu` → `gelu_erf`).
#[allow(dead_code)]
pub(super) fn gelu_erf_inplace(data: &mut [f32]) {
    const INV_SQRT2: f32 = std::f32::consts::FRAC_1_SQRT_2;
    for v in data.iter_mut() {
        // Abramowitz & Stegun erf approximation — same constants as
        // the rlx-cpu NEON kernel, so numerical agreement is exact.
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

#[cfg(test)]
#[allow(dead_code)]
pub(super) fn assert_shape(label: &str, actual: usize, expected: usize) {
    assert_eq!(actual, expected, "{label}: {actual} ≠ {expected}");
}

#[allow(dead_code)]
fn _silence_constant() {
    let _ = SAM_PROMPT_EMBED_DIM;
}
