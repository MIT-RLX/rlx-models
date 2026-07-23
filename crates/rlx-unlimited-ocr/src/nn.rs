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

//! Small host-side f32 kernels for the eager forward path: norms,
//! activations, RoPE, a generic im2col `Conv2d`, and bicubic / linear
//! resampling for absolute position embeddings.
//!
//! Heavy matmuls delegate to [`rlx_core::host_kernels`] (BLAS-backed);
//! everything here is the glue those kernels don't provide.

use anyhow::{Result, ensure};
use rlx_core::host_kernels;

/// `y = x @ w^T (+ b)`, where `w` is `[out_dim, in_dim]` (PyTorch `nn.Linear`
/// layout — no transpose needed at load time).
pub fn linear_wt(
    x: &[f32],
    rows: usize,
    in_dim: usize,
    w: &[f32],
    out_dim: usize,
    bias: Option<&[f32]>,
) -> Result<Vec<f32>> {
    ensure!(x.len() == rows * in_dim, "linear_wt: input shape mismatch");
    ensure!(
        w.len() == out_dim * in_dim,
        "linear_wt: weight shape mismatch ({} != {out_dim}*{in_dim})",
        w.len()
    );
    let mut out = vec![0f32; rows * out_dim];
    host_kernels::matmul_bt(x, w, &mut out, rows, in_dim, out_dim, 1.0);
    if let Some(b) = bias {
        ensure!(b.len() == out_dim, "linear_wt: bias shape mismatch");
        add_bias_rows(&mut out, rows, out_dim, b);
    }
    Ok(out)
}

pub fn add_bias_rows(x: &mut [f32], rows: usize, dim: usize, bias: &[f32]) {
    for r in 0..rows {
        let row = &mut x[r * dim..(r + 1) * dim];
        for (v, b) in row.iter_mut().zip(bias.iter()) {
            *v += *b;
        }
    }
}

pub fn add_inplace(x: &mut [f32], y: &[f32]) {
    for (a, b) in x.iter_mut().zip(y.iter()) {
        *a += *b;
    }
}

/// RMSNorm over the last axis (`[rows, dim]`).
pub fn rms_norm(x: &[f32], rows: usize, dim: usize, weight: &[f32], eps: f32) -> Vec<f32> {
    let mut out = vec![0f32; rows * dim];
    for r in 0..rows {
        let row = &x[r * dim..(r + 1) * dim];
        let ms = row.iter().map(|v| v * v).sum::<f32>() / dim as f32;
        let inv = 1.0 / (ms + eps).sqrt();
        let dst = &mut out[r * dim..(r + 1) * dim];
        for c in 0..dim {
            dst[c] = row[c] * inv * weight[c];
        }
    }
    out
}

/// LayerNorm over the last axis (`[rows, dim]`), f64 accumulation for parity.
pub fn layer_norm(
    x: &[f32],
    rows: usize,
    dim: usize,
    gamma: &[f32],
    beta: &[f32],
    eps: f32,
) -> Vec<f32> {
    let mut out = vec![0f32; rows * dim];
    for r in 0..rows {
        let row = &x[r * dim..(r + 1) * dim];
        let mean = row.iter().map(|v| *v as f64).sum::<f64>() / dim as f64;
        let var = row
            .iter()
            .map(|v| {
                let d = *v as f64 - mean;
                d * d
            })
            .sum::<f64>()
            / dim as f64;
        let inv = 1.0 / (var + eps as f64).sqrt();
        let dst = &mut out[r * dim..(r + 1) * dim];
        for c in 0..dim {
            dst[c] = (((row[c] as f64 - mean) * inv) as f32) * gamma[c] + beta[c];
        }
    }
    out
}

pub fn softmax_rows(x: &mut [f32], rows: usize, cols: usize) {
    host_kernels::softmax_rows(x, rows, cols);
}

pub fn silu(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v *= 1.0 / (1.0 + (-*v).exp());
    }
}

/// `x * sigmoid(1.702 * x)` — CLIP's activation.
pub fn quick_gelu(x: &mut [f32]) {
    for v in x.iter_mut() {
        let s = 1.0 / (1.0 + (-1.702 * *v).exp());
        *v *= s;
    }
}

const INV_SQRT2: f32 = std::f32::consts::FRAC_1_SQRT_2;

/// Exact (erf-based) GELU — matches `torch.nn.GELU()` default, used by SAM's MLP.
pub fn gelu_erf(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = 0.5 * *v * (1.0 + erf(*v * INV_SQRT2));
    }
}

/// Abramowitz & Stegun 7.1.26 rational approximation, max error ~1.5e-7.
fn erf(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    // f32 constants truncated to representable precision (clippy excessive_precision).
    const A1: f32 = 0.254_829_6;
    const A2: f32 = -0.284_496_75;
    const A3: f32 = 1.421_413_8;
    const A4: f32 = -1.453_152_1;
    const A5: f32 = 1.061_405_4;
    const P: f32 = 0.327_591_1;
    let t = 1.0 / (1.0 + P * x);
    let y = 1.0 - (((((A5 * t + A4) * t) + A3) * t + A2) * t + A1) * t * (-x * x).exp();
    sign * y
}

/// Precompute `[max_pos, head_dim/2]` cos/sin RoPE tables (θ = `theta`).
pub fn rope_tables(max_pos: usize, head_dim: usize, theta: f64) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;
    let mut cos = vec![0f32; max_pos * half];
    let mut sin = vec![0f32; max_pos * half];
    for pos in 0..max_pos {
        for i in 0..half {
            let freq = 1.0 / theta.powf((2 * i) as f64 / head_dim as f64);
            let angle = pos as f64 * freq;
            let (s, c) = angle.sin_cos();
            cos[pos * half + i] = c as f32;
            sin[pos * half + i] = s as f32;
        }
    }
    (cos, sin)
}

/// Apply rotate-half RoPE in place to `[n_tokens, n_heads, head_dim]` data,
/// one absolute position per token (`positions.len() == n_tokens`).
pub fn apply_rope(
    x: &mut [f32],
    n_tokens: usize,
    n_heads: usize,
    head_dim: usize,
    positions: &[usize],
    cos_table: &[f32],
    sin_table: &[f32],
) {
    let half = head_dim / 2;
    for t in 0..n_tokens {
        let pos = positions[t];
        let cos = &cos_table[pos * half..(pos + 1) * half];
        let sin = &sin_table[pos * half..(pos + 1) * half];
        for h in 0..n_heads {
            let base = (t * n_heads + h) * head_dim;
            let head = &mut x[base..base + head_dim];
            for i in 0..half {
                let x1 = head[i];
                let x2 = head[i + half];
                head[i] = x1 * cos[i] - x2 * sin[i];
                head[i + half] = x2 * cos[i] + x1 * sin[i];
            }
        }
    }
}

/// Cubic convolution kernel weight (Keys, a = -0.75 — matches PyTorch/PIL bicubic).
fn cubic_kernel(x: f32) -> f32 {
    const A: f32 = -0.75;
    let x = x.abs();
    if x <= 1.0 {
        (A + 2.0) * x * x * x - (A + 3.0) * x * x + 1.0
    } else if x < 2.0 {
        A * x * x * x - 5.0 * A * x * x + 8.0 * A * x - 4.0 * A
    } else {
        0.0
    }
}

fn cubic_weights(t: f32) -> [f32; 4] {
    [
        cubic_kernel(1.0 + t),
        cubic_kernel(t),
        cubic_kernel(1.0 - t),
        cubic_kernel(2.0 - t),
    ]
}

fn clamp_idx(i: isize, len: usize) -> usize {
    i.clamp(0, len as isize - 1) as usize
}

/// Bicubic-resize a `[sh, sw, c]` (row-major, channel-last) grid to
/// `[dh, dw, c]`. `align_corners = false` (matches `F.interpolate(..., mode="bicubic")`).
pub fn bicubic_resize_hwc(
    src: &[f32],
    sh: usize,
    sw: usize,
    c: usize,
    dh: usize,
    dw: usize,
) -> Vec<f32> {
    if sh == dh && sw == dw {
        return src.to_vec();
    }
    let scale_h = sh as f32 / dh as f32;
    let scale_w = sw as f32 / dw as f32;
    let mut out = vec![0f32; dh * dw * c];
    for oy in 0..dh {
        let src_y = (oy as f32 + 0.5) * scale_h - 0.5;
        let y0 = src_y.floor();
        let ty = src_y - y0;
        let wy = cubic_weights(ty);
        let y0 = y0 as isize;
        for ox in 0..dw {
            let src_x = (ox as f32 + 0.5) * scale_w - 0.5;
            let x0 = src_x.floor();
            let tx = src_x - x0;
            let wx = cubic_weights(tx);
            let x0 = x0 as isize;
            let dst = &mut out[(oy * dw + ox) * c..(oy * dw + ox + 1) * c];
            for (dy, &wyi) in wy.iter().enumerate() {
                let sy = clamp_idx(y0 - 1 + dy as isize, sh);
                for (dx, &wxi) in wx.iter().enumerate() {
                    let sx = clamp_idx(x0 - 1 + dx as isize, sw);
                    let w = wyi * wxi;
                    if w == 0.0 {
                        continue;
                    }
                    let src_px = &src[(sy * sw + sx) * c..(sy * sw + sx + 1) * c];
                    for ch in 0..c {
                        dst[ch] += w * src_px[ch];
                    }
                }
            }
        }
    }
    out
}

/// Linear-resample `[old_len, c]` rows to `[new_len, c]` (`align_corners = false`,
/// matches `F.interpolate(mode="linear")` used by SAM's `get_rel_pos`).
pub fn linear_resize_1d(src: &[f32], old_len: usize, c: usize, new_len: usize) -> Vec<f32> {
    if old_len == new_len {
        return src.to_vec();
    }
    let scale = old_len as f32 / new_len as f32;
    let mut out = vec![0f32; new_len * c];
    for oi in 0..new_len {
        let src_pos = (oi as f32 + 0.5) * scale - 0.5;
        let i0f = src_pos.floor();
        let t = src_pos - i0f;
        let i0 = clamp_idx(i0f as isize, old_len);
        let i1 = clamp_idx(i0f as isize + 1, old_len);
        let dst = &mut out[oi * c..(oi + 1) * c];
        let a = &src[i0 * c..(i0 + 1) * c];
        let b = &src[i1 * c..(i1 + 1) * c];
        for ch in 0..c {
            dst[ch] = a[ch] * (1.0 - t) + b[ch] * t;
        }
    }
    out
}

/// Generic im2col `Conv2d`: `input` is `[h, w, in_c]` (channel-last),
/// `weight` is `[out_c, in_c, kh, kw]` (PyTorch layout). Returns
/// `([out_h*out_w, out_c], out_h, out_w)`.
#[allow(clippy::too_many_arguments)]
pub fn conv2d_hwc(
    input: &[f32],
    h: usize,
    w: usize,
    in_c: usize,
    weight: &[f32],
    out_c: usize,
    kh: usize,
    kw: usize,
    stride: usize,
    pad: usize,
    bias: Option<&[f32]>,
) -> Result<(Vec<f32>, usize, usize)> {
    ensure!(
        input.len() == h * w * in_c,
        "conv2d_hwc: input shape mismatch"
    );
    ensure!(
        weight.len() == out_c * in_c * kh * kw,
        "conv2d_hwc: weight shape mismatch"
    );
    let out_h = (h + 2 * pad - kh) / stride + 1;
    let out_w = (w + 2 * pad - kw) / stride + 1;
    let patch_len = in_c * kh * kw;
    let mut patches = vec![0f32; out_h * out_w * patch_len];
    for oy in 0..out_h {
        for ox in 0..out_w {
            let dst =
                &mut patches[(oy * out_w + ox) * patch_len..(oy * out_w + ox + 1) * patch_len];
            for ky in 0..kh {
                let iy = oy as isize * stride as isize + ky as isize - pad as isize;
                if iy < 0 || iy >= h as isize {
                    continue;
                }
                for kx in 0..kw {
                    let ix = ox as isize * stride as isize + kx as isize - pad as isize;
                    if ix < 0 || ix >= w as isize {
                        continue;
                    }
                    let src = &input[(iy as usize * w + ix as usize) * in_c
                        ..(iy as usize * w + ix as usize + 1) * in_c];
                    // Patch layout must match weight's [in_c, kh, kw] flattening
                    // (in_c slowest): scatter this pixel's channels with stride kh*kw.
                    for c in 0..in_c {
                        dst[c * kh * kw + ky * kw + kx] = src[c];
                    }
                }
            }
        }
    }
    let mut out = vec![0f32; out_h * out_w * out_c];
    host_kernels::matmul_bt(
        &patches,
        weight,
        &mut out,
        out_h * out_w,
        patch_len,
        out_c,
        1.0,
    );
    if let Some(b) = bias {
        add_bias_rows(&mut out, out_h * out_w, out_c, b);
    }
    Ok((out, out_h, out_w))
}

/// Top-k softmax gate over `logits[experts]`: returns `(index, weight)`
/// pairs. `weight` is the raw softmax probability (caller applies
/// `norm_topk_prob` / `routed_scaling_factor`).
pub fn topk_softmax(logits: &[f32], k: usize) -> Vec<(usize, f32)> {
    let n = logits.len();
    let mut probs = logits.to_vec();
    softmax_rows(&mut probs, 1, n);
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_unstable_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
    idx.truncate(k.min(n));
    idx.into_iter().map(|i| (i, probs[i])).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rms_norm_unit_weight_matches_manual() {
        let x = [3.0f32, 4.0];
        let w = [1.0f32, 1.0];
        let out = rms_norm(&x, 1, 2, &w, 0.0);
        let ms: f32 = (9.0 + 16.0) / 2.0;
        let inv = 1.0 / ms.sqrt();
        assert!((out[0] - 3.0 * inv).abs() < 1e-6);
        assert!((out[1] - 4.0 * inv).abs() < 1e-6);
    }

    #[test]
    fn silu_zero_is_zero() {
        let mut x = [0.0f32];
        silu(&mut x);
        assert_eq!(x[0], 0.0);
    }

    #[test]
    fn quick_gelu_matches_definition() {
        let mut x = [2.0f32];
        quick_gelu(&mut x);
        let expected = 2.0 * (1.0 / (1.0 + (-1.702f32 * 2.0).exp()));
        assert!((x[0] - expected).abs() < 1e-6);
    }

    #[test]
    fn gelu_erf_zero_is_zero_and_odd_ish() {
        let mut x = [0.0f32, 1.0, -1.0];
        gelu_erf(&mut x);
        assert!(x[0].abs() < 1e-6);
        assert!(x[1] > 0.8 && x[1] < 0.9); // gelu(1) ≈ 0.8413
        assert!(x[2] < 0.0);
    }

    #[test]
    fn rope_preserves_norm() {
        let (cos, sin) = rope_tables(4, 4, 10_000.0);
        let mut x = [1.0f32, 2.0, 3.0, 4.0];
        let norm_before = x.iter().map(|v| v * v).sum::<f32>();
        apply_rope(&mut x, 1, 1, 4, &[2], &cos, &sin);
        let norm_after = x.iter().map(|v| v * v).sum::<f32>();
        assert!((norm_before - norm_after).abs() < 1e-4);
    }

    #[test]
    fn bicubic_identity_when_same_size() {
        let src = [1.0f32, 2.0, 3.0, 4.0]; // 2x2x1
        let out = bicubic_resize_hwc(&src, 2, 2, 1, 2, 2);
        assert_eq!(out, src);
    }

    #[test]
    fn linear_resize_identity_when_same_size() {
        let src = [1.0f32, 2.0, 3.0, 4.0]; // 4x1
        let out = linear_resize_1d(&src, 4, 1, 4);
        assert_eq!(out, src);
    }

    #[test]
    fn topk_softmax_picks_largest() {
        let logits = [0.0f32, 5.0, 1.0, -2.0];
        let top = topk_softmax(&logits, 2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].0, 1);
        assert_eq!(top[1].0, 2);
        assert!(top[0].1 > top[1].1);
    }

    #[test]
    fn conv2d_hwc_1x1_matches_linear() {
        // 2x2 spatial, 2 in-channels, 1x1 conv -> 3 out-channels == per-pixel linear.
        let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // [2,2,2]
        let weight = [1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0]; // [3,2,1,1]
        let (out, oh, ow) = conv2d_hwc(&input, 2, 2, 2, &weight, 3, 1, 1, 1, 0, None).unwrap();
        assert_eq!((oh, ow), (2, 2));
        // pixel0 = [1,2] -> [1, 2, 3]
        assert_eq!(&out[0..3], &[1.0, 2.0, 3.0]);
    }
}
