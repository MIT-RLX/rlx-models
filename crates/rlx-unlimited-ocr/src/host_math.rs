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

//! NCHW-flavored host-side f32 kernels used by [`crate::sam_tower`] (which
//! keeps its spatial tensors channel-first, matching the checkpoint's
//! `Conv2d`/`LayerNorm2d` layout, unlike [`crate::nn`]'s channel-last
//! helpers used by [`crate::clip_tower`]).
//!
//! Heavy matmuls delegate to [`rlx_core::host_kernels`] (BLAS-backed).

use anyhow::{Result, ensure};
use rlx_core::host_kernels;

/// `y = x @ w` (`w` already `[in_dim, out_dim]` — i.e. pre-transposed at load
/// time via `WeightMap::take_transposed`, unlike [`crate::nn::linear_wt`]
/// which takes untransposed PyTorch `[out, in]` weights).
pub fn linear(
    x: &[f32],
    rows: usize,
    in_dim: usize,
    w: &[f32],
    out_dim: usize,
    bias: Option<&[f32]>,
) -> Result<Vec<f32>> {
    ensure!(x.len() == rows * in_dim, "linear: input shape mismatch");
    ensure!(w.len() == in_dim * out_dim, "linear: weight shape mismatch");
    let mut out = vec![0f32; rows * out_dim];
    host_kernels::matmul(x, w, &mut out, rows, in_dim, out_dim);
    if let Some(b) = bias {
        ensure!(b.len() == out_dim, "linear: bias shape mismatch");
        for row in out.chunks_mut(out_dim) {
            for (v, bi) in row.iter_mut().zip(b.iter()) {
                *v += *bi;
            }
        }
    }
    Ok(out)
}

/// LayerNorm over the last axis of `[rows, dim]` (`rows` inferred from
/// `x.len() / dim`), f64 accumulation for parity.
pub fn layer_norm(
    x: &[f32],
    dim: usize,
    gamma: &[f32],
    beta: &[f32],
    eps: f32,
) -> Result<Vec<f32>> {
    ensure!(
        x.len().is_multiple_of(dim),
        "layer_norm: x.len() not a multiple of dim"
    );
    let rows = x.len() / dim;
    let mut out = vec![0f32; x.len()];
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
    Ok(out)
}

/// `LayerNorm2d` (SAM's neck norm): normalizes each pixel's `channels`
/// values (NCHW layout — the norm axis is the *outer* stride, not
/// contiguous), f64 accumulation for parity.
pub fn layer_norm2d_nchw(
    x: &[f32],
    channels: usize,
    hw: usize,
    gamma: &[f32],
    beta: &[f32],
    eps: f32,
) -> Vec<f32> {
    let mut out = vec![0f32; channels * hw];
    for p in 0..hw {
        let mut mean = 0f64;
        for c in 0..channels {
            mean += x[c * hw + p] as f64;
        }
        mean /= channels as f64;
        let mut var = 0f64;
        for c in 0..channels {
            let d = x[c * hw + p] as f64 - mean;
            var += d * d;
        }
        var /= channels as f64;
        let inv = 1.0 / (var + eps as f64).sqrt();
        for c in 0..channels {
            out[c * hw + p] = (((x[c * hw + p] as f64 - mean) * inv) as f32) * gamma[c] + beta[c];
        }
    }
    out
}

pub fn softmax_rows(x: &mut [f32], rows: usize, cols: usize) {
    host_kernels::softmax_rows(x, rows, cols);
}

const INV_SQRT2: f32 = std::f32::consts::FRAC_1_SQRT_2;

/// Exact (erf-based) GELU — matches `torch.nn.GELU()` default, used by SAM's MLP.
pub fn gelu_erf_inplace(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = 0.5 * *v * (1.0 + erf(*v * INV_SQRT2));
    }
}

/// Abramowitz & Stegun 7.1.26 rational approximation, max error ~1.5e-7.
fn erf(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    const A1: f32 = 0.254_829_6;
    const A2: f32 = -0.284_496_72;
    const A3: f32 = 1.421_413_8;
    const A4: f32 = -1.453_152_1;
    const A5: f32 = 1.061_405_4;
    const P: f32 = 0.327_591_1;
    let t = 1.0 / (1.0 + P * x);
    let y = 1.0 - (((((A5 * t + A4) * t) + A3) * t + A2) * t + A1) * t * (-x * x).exp();
    sign * y
}

/// Generic im2col `Conv2d`, NCHW throughout: `input` is `[in_c, h, w]`,
/// `weight` is `[out_c, in_c, k, k]` (PyTorch layout, square kernel, no
/// dilation). Returns `([out_c, out_h, out_w], out_h, out_w)`.
#[allow(clippy::too_many_arguments)]
pub fn conv2d(
    input: &[f32],
    in_c: usize,
    h: usize,
    w: usize,
    weight: &[f32],
    out_c: usize,
    k: usize,
    stride: usize,
    pad: usize,
    bias: Option<&[f32]>,
) -> (Vec<f32>, usize, usize) {
    assert_eq!(input.len(), in_c * h * w, "conv2d: input shape mismatch");
    assert_eq!(
        weight.len(),
        out_c * in_c * k * k,
        "conv2d: weight shape mismatch"
    );
    let out_h = (h + 2 * pad - k) / stride + 1;
    let out_w = (w + 2 * pad - k) / stride + 1;
    let patch_len = in_c * k * k;
    let mut patches = vec![0f32; out_h * out_w * patch_len];
    for oy in 0..out_h {
        for ox in 0..out_w {
            let dst =
                &mut patches[(oy * out_w + ox) * patch_len..(oy * out_w + ox + 1) * patch_len];
            for c in 0..in_c {
                for ky in 0..k {
                    let iy = oy as isize * stride as isize + ky as isize - pad as isize;
                    if iy < 0 || iy >= h as isize {
                        continue;
                    }
                    for kx in 0..k {
                        let ix = ox as isize * stride as isize + kx as isize - pad as isize;
                        if ix < 0 || ix >= w as isize {
                            continue;
                        }
                        dst[c * k * k + ky * k + kx] =
                            input[c * h * w + iy as usize * w + ix as usize];
                    }
                }
            }
        }
    }
    // `patches @ weight^T` -> [out_h*out_w, out_c] (token-major); transpose to NCHW.
    let mut hwc = vec![0f32; out_h * out_w * out_c];
    host_kernels::matmul_bt(
        &patches,
        weight,
        &mut hwc,
        out_h * out_w,
        patch_len,
        out_c,
        1.0,
    );
    let mut out = vec![0f32; out_c * out_h * out_w];
    let hw = out_h * out_w;
    for p in 0..hw {
        for c in 0..out_c {
            out[c * hw + p] = hwc[p * out_c + c] + bias.map(|b| b[c]).unwrap_or(0.0);
        }
    }
    (out, out_h, out_w)
}

/// Bicubic-resize a `[c, sh, sw]` (NCHW) grid to `[c, dh, dw]`
/// (`align_corners = false`, matches `F.interpolate(..., mode="bicubic")`).
pub fn bicubic_resize_chw(
    src: &[f32],
    c: usize,
    sh: usize,
    sw: usize,
    dh: usize,
    dw: usize,
) -> Vec<f32> {
    if sh == dh && sw == dw {
        return src.to_vec();
    }
    let scale_h = sh as f32 / dh as f32;
    let scale_w = sw as f32 / dw as f32;
    let mut out = vec![0f32; c * dh * dw];
    for oy in 0..dh {
        let src_y = (oy as f32 + 0.5) * scale_h - 0.5;
        let y0f = src_y.floor();
        let ty = src_y - y0f;
        let wy = cubic_weights(ty);
        let y0 = y0f as isize;
        for ox in 0..dw {
            let src_x = (ox as f32 + 0.5) * scale_w - 0.5;
            let x0f = src_x.floor();
            let tx = src_x - x0f;
            let wx = cubic_weights(tx);
            let x0 = x0f as isize;
            for (dy, &wyi) in wy.iter().enumerate() {
                let sy = clamp_idx(y0 - 1 + dy as isize, sh);
                for (dx, &wxi) in wx.iter().enumerate() {
                    let sx = clamp_idx(x0 - 1 + dx as isize, sw);
                    let weight = wyi * wxi;
                    if weight == 0.0 {
                        continue;
                    }
                    for ch in 0..c {
                        out[ch * dh * dw + oy * dw + ox] +=
                            weight * src[ch * sh * sw + sy * sw + sx];
                    }
                }
            }
        }
    }
    out
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

/// Multi-head attention over token-major `[s, heads*head_dim]` `q`/`k`/`v`
/// (`q` has `sq` rows, `k`/`v` have `sk` rows), with an optional dense
/// additive `[sq, sk]` mask (`0.0` visible, `-inf` masked). Returns the
/// merged `[sq, heads*head_dim]` output (no output projection).
#[allow(clippy::too_many_arguments)]
pub fn mha_with_mask(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    sq: usize,
    sk: usize,
    heads: usize,
    head_dim: usize,
    scale: f32,
    mask: Option<&[f32]>,
) -> Result<Vec<f32>> {
    let hidden = heads * head_dim;
    ensure!(q.len() == sq * hidden, "mha_with_mask: q shape mismatch");
    ensure!(k.len() == sk * hidden, "mha_with_mask: k shape mismatch");
    ensure!(v.len() == sk * hidden, "mha_with_mask: v shape mismatch");
    if let Some(m) = mask {
        ensure!(m.len() == sq * sk, "mha_with_mask: mask shape mismatch");
    }

    let mut merged = vec![0f32; sq * hidden];
    let mut scores = vec![0f32; sq * sk];
    for h in 0..heads {
        let off = h * head_dim;
        for qi in 0..sq {
            let qvec = &q[qi * hidden + off..qi * hidden + off + head_dim];
            for ki in 0..sk {
                let kvec = &k[ki * hidden + off..ki * hidden + off + head_dim];
                let dot: f32 = qvec.iter().zip(kvec.iter()).map(|(a, b)| a * b).sum();
                let masked = mask.map(|m| m[qi * sk + ki]).unwrap_or(0.0);
                scores[qi * sk + ki] = dot * scale + masked;
            }
        }
        softmax_rows(&mut scores, sq, sk);
        for qi in 0..sq {
            let dst = &mut merged[qi * hidden + off..qi * hidden + off + head_dim];
            for ki in 0..sk {
                let w = scores[qi * sk + ki];
                if w == 0.0 {
                    continue;
                }
                let vvec = &v[ki * hidden + off..ki * hidden + off + head_dim];
                for d in 0..head_dim {
                    dst[d] += w * vvec[d];
                }
            }
        }
    }
    Ok(merged)
}

/// Linear-resample `[old_len, c]` rows to `[new_len, c]` (`align_corners = false`,
/// matches `F.interpolate(mode="linear")` used by SAM's `get_rel_pos`).
pub fn linear_resize_1d(src: &[f32], old_len: usize, new_len: usize, c: usize) -> Vec<f32> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_matches_manual_matmul() {
        // x: [2,2], w (pre-transposed [in,out]=[2,2]): identity -> y == x.
        let x = [1.0f32, 2.0, 3.0, 4.0];
        let w = [1.0f32, 0.0, 0.0, 1.0];
        let y = linear(&x, 2, 2, &w, 2, None).unwrap();
        assert_eq!(y, x);
    }

    #[test]
    fn layer_norm_zero_mean_unit_var_row() {
        let x = [1.0f32, -1.0];
        let g = [1.0f32, 1.0];
        let b = [0.0f32, 0.0];
        let out = layer_norm(&x, 2, &g, &b, 0.0).unwrap();
        assert!((out[0] - 1.0).abs() < 1e-4);
        assert!((out[1] + 1.0).abs() < 1e-4);
    }

    #[test]
    fn conv2d_1x1_matches_per_pixel_linear() {
        // 2 in-channels, 2x2 spatial, 1x1 conv -> 3 out-channels.
        let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // [2,2,2] NCHW
        let weight = [1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0]; // [3,2,1,1]
        let (out, oh, ow) = conv2d(&input, 2, 2, 2, &weight, 3, 1, 1, 0, None);
        assert_eq!((oh, ow), (2, 2));
        // pixel (0,0): in = [1, 5] -> out channels [1*1, 1*5, 1+5] = [1, 5, 6].
        assert_eq!(out[0], 1.0);
        assert_eq!(out[4], 5.0);
        assert_eq!(out[8], 6.0);
    }

    #[test]
    fn bicubic_resize_chw_identity_when_same_size() {
        let src = [1.0f32, 2.0, 3.0, 4.0]; // [1,2,2]
        let out = bicubic_resize_chw(&src, 1, 2, 2, 2, 2);
        assert_eq!(out, src);
    }

    #[test]
    fn linear_resize_1d_identity_when_same_size() {
        let src = [1.0f32, 2.0, 3.0, 4.0];
        let out = linear_resize_1d(&src, 4, 4, 1);
        assert_eq!(out, src);
    }
}
