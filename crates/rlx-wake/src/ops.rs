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

//! Shared CPU ops for wake models (RLX BLAS backend).

use rlx_cpu::blas::{sgemm, sgemm_accumulate};

#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
pub fn relu(x: f32) -> f32 {
    x.max(0.0)
}

/// `y = A @ x` with A row-major `[m, n]`.
pub fn gemv(m: usize, n: usize, a: &[f32], x: &[f32], y: &mut [f32]) {
    if m == 0 || n == 0 {
        return;
    }
    y[..m].fill(0.0);
    sgemm(a, x, y, m, n, 1);
}

pub fn gemv_bias(m: usize, n: usize, a: &[f32], x: &[f32], bias: &[f32], y: &mut [f32]) {
    gemv(m, n, a, x, y);
    for i in 0..m {
        y[i] += bias[i];
    }
}

/// Accumulate `y += A @ x`.
pub fn gemv_add(m: usize, n: usize, a: &[f32], x: &[f32], y: &mut [f32]) {
    if m == 0 || n == 0 {
        return;
    }
    sgemm_accumulate(a, x, y, m, n, 1);
}

/// PyTorch Conv1d on channel-major `[in_ch, t]` → `[out_ch, t_out]`.
pub fn conv1d_nchw(
    x: &[f32],
    in_ch: usize,
    t_in: usize,
    w: &[f32],
    out_ch: usize,
    k: usize,
    stride: usize,
    pad: usize,
    bias: Option<&[f32]>,
    out: &mut [f32],
) -> usize {
    let t_out = if t_in + 2 * pad >= k {
        (t_in + 2 * pad - k) / stride + 1
    } else {
        0
    };
    out.fill(0.0);
    for oc in 0..out_ch {
        for ot in 0..t_out {
            let mut sum = bias.map(|b| b[oc]).unwrap_or(0.0);
            for ic in 0..in_ch {
                for ki in 0..k {
                    let ti = ot * stride + ki;
                    let ti = ti as isize - pad as isize;
                    if ti < 0 || ti >= t_in as isize {
                        continue;
                    }
                    let x_idx = ic * t_in + ti as usize;
                    let w_idx = oc * (in_ch * k) + ic * k + ki;
                    sum += x[x_idx] * w[w_idx];
                }
            }
            out[oc * t_out + ot] = sum;
        }
    }
    t_out
}

/// Conv2d NCHW: `x [in_ch, h, w]` with filter `[out_ch, in_ch, kh, kw]`.
pub fn conv2d_nchw(
    x: &[f32],
    in_ch: usize,
    h: usize,
    w: usize,
    weight: &[f32],
    out_ch: usize,
    kh: usize,
    kw: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
    bias: Option<&[f32]>,
    out: &mut [f32],
) -> (usize, usize) {
    let h_out = if h + 2 * pad_h >= kh {
        (h + 2 * pad_h - kh) / stride_h + 1
    } else {
        0
    };
    let w_out = if w + 2 * pad_w >= kw {
        (w + 2 * pad_w - kw) / stride_w + 1
    } else {
        0
    };
    out.fill(0.0);
    for oc in 0..out_ch {
        for oh in 0..h_out {
            for ow in 0..w_out {
                let mut sum = bias.map(|b| b[oc]).unwrap_or(0.0);
                for ic in 0..in_ch {
                    for kh_i in 0..kh {
                        for kw_i in 0..kw {
                            let ih = oh * stride_h + kh_i;
                            let iw = ow * stride_w + kw_i;
                            let ih = ih as isize - pad_h as isize;
                            let iw = iw as isize - pad_w as isize;
                            if ih < 0 || iw < 0 || ih >= h as isize || iw >= w as isize {
                                continue;
                            }
                            let x_idx = ic * h * w + ih as usize * w + iw as usize;
                            let w_idx = oc * (in_ch * kh * kw) + ic * kh * kw + kh_i * kw + kw_i;
                            sum += x[x_idx] * weight[w_idx];
                        }
                    }
                }
                out[oc * h_out * w_out + oh * w_out + ow] = sum;
            }
        }
    }
    (h_out, w_out)
}

pub fn global_mean_pool_chw(x: &[f32], ch: usize, spatial: usize, out: &mut [f32]) {
    debug_assert_eq!(out.len(), ch);
    let inv = if spatial == 0 {
        0.0
    } else {
        1.0 / spatial as f32
    };
    for c in 0..ch {
        let mut s = 0.0f32;
        let base = c * spatial;
        for i in 0..spatial {
            s += x[base + i];
        }
        out[c] = s * inv;
    }
}
