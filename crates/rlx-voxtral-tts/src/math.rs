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

//! Small ndarray helpers for eager CPU inference.

use ndarray::{Array2, ArrayView1, ArrayView2, ArrayView3, Axis, s};

pub fn rms_norm(x: ArrayView2<f32>, weight: ArrayView1<f32>, eps: f32) -> Array2<f32> {
    let (t, d) = x.dim();
    let mut out = Array2::<f32>::zeros((t, d));
    for i in 0..t {
        let row = x.row(i);
        let mut sum = 0f32;
        for v in row.iter() {
            sum += v * v;
        }
        let inv = 1.0 / (sum / d as f32 + eps).sqrt();
        for j in 0..d {
            out[[i, j]] = row[j] * inv * weight[j];
        }
    }
    out
}

pub fn silu(x: ArrayView2<f32>) -> Array2<f32> {
    x.mapv(|v| v / (1.0 + (-v).exp()))
}

pub fn swiglu(w1: ArrayView2<f32>, w3: ArrayView2<f32>, w2: &Array2<f32>) -> Array2<f32> {
    let h = silu(w1) * w3.to_owned();
    linear2(h.view(), w2.view(), None)
}

pub fn matmul2(a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32> {
    a.dot(b)
}

pub fn linear2(
    x: ArrayView2<f32>,
    w: ArrayView2<f32>,
    bias: Option<ArrayView1<f32>>,
) -> Array2<f32> {
    let mut out = x.dot(&w.t());
    if let Some(b) = bias {
        for mut row in out.rows_mut() {
            row += &b;
        }
    }
    out
}

pub fn pad1d_reflect(x: ArrayView2<f32>, pad_left: usize, pad_right: usize) -> Array2<f32> {
    let (c, t) = x.dim();
    let out_len = t + pad_left + pad_right;
    let mut out = Array2::<f32>::zeros((c, out_len));
    for ci in 0..c {
        for ti in 0..t {
            out[[ci, ti + pad_left]] = x[[ci, ti]];
        }
        for pi in 0..pad_left {
            let src = pi.min(t.saturating_sub(1));
            out[[ci, pad_left - 1 - pi]] = x[[ci, src]];
        }
        for pi in 0..pad_right {
            let src = (t.saturating_sub(1)).saturating_sub(pi);
            out[[ci, pad_left + t + pi]] = x[[ci, src]];
        }
    }
    out
}

pub fn conv1d(
    x: ArrayView2<f32>,
    weight: ArrayView3<f32>,
    stride: usize,
    pad_left: usize,
) -> Array2<f32> {
    let (out_ch, in_ch, k) = weight.dim();
    let padded = pad1d_reflect(x, pad_left, 0);
    let t_pad = padded.dim().1;
    let t_out = (t_pad - k) / stride + 1;
    let mut out = Array2::<f32>::zeros((out_ch, t_out));
    for oc in 0..out_ch {
        for ti in 0..t_out {
            let mut sum = 0f32;
            for ic in 0..in_ch {
                for ki in 0..k {
                    let x_val = padded[[ic, ti * stride + ki]];
                    sum += x_val * weight[[oc, ic, ki]];
                }
            }
            out[[oc, ti]] = sum;
        }
    }
    out
}

pub fn conv_transpose1d(
    x: ArrayView2<f32>,
    weight: ArrayView3<f32>,
    stride: usize,
    trim_left: usize,
    trim_right: usize,
) -> Array2<f32> {
    let (in_ch, out_ch, k) = weight.dim();
    let (_, t_in) = x.dim();
    let t_raw = (t_in - 1) * stride + k;
    let mut out = Array2::<f32>::zeros((out_ch, t_raw));
    for ti in 0..t_in {
        for ic in 0..in_ch {
            for ki in 0..k {
                let src = x[[ic, ti]];
                for oc in 0..out_ch {
                    out[[oc, ti * stride + ki]] += src * weight[[ic, oc, ki]];
                }
            }
        }
    }
    let end = out.dim().1 - trim_right;
    out.slice(s![.., trim_left..end]).to_owned()
}

pub fn sum_axis1(x: ArrayView3<f32>) -> Array2<f32> {
    x.sum_axis(Axis(1))
}
