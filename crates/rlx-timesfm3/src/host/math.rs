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

//! Tensor math helpers for the host reference forward.

use ndarray::{Array2, Array3, Array4, ArrayView1, ArrayView2, ArrayView4};

pub const RMS_EPS: f32 = 1e-6;
pub const RECIPROCAL_OF_SOFTPLUS_0: f32 = std::f32::consts::LOG2_E;
pub const REVIN_TOLERANCE: f32 = 1e-6;

pub fn relu(x: f32) -> f32 {
    x.max(0.0)
}

pub fn relu_array(x: &mut Array2<f32>) {
    x.mapv_inplace(relu);
}

pub fn linear2(x: ArrayView2<f32>, w: ArrayView2<f32>, b: Option<ArrayView1<f32>>) -> Array2<f32> {
    let mut y = x.dot(&w.t());
    if let Some(b) = b {
        for (mut row, &bias) in y.rows_mut().into_iter().zip(b.iter()) {
            row += bias;
        }
    }
    y
}

pub fn rms_norm(x: ArrayView2<f32>, weight: ArrayView1<f32>) -> Array2<f32> {
    let mut out = Array2::zeros(x.raw_dim());
    for (mut row, in_row) in out.rows_mut().into_iter().zip(x.rows()) {
        let mean_sq = in_row.iter().map(|v| v * v).sum::<f32>() / in_row.len() as f32;
        let inv = 1.0 / (mean_sq + RMS_EPS).sqrt();
        for (o, (&v, &w)) in row.iter_mut().zip(in_row.iter().zip(weight.iter())) {
            *o = v * inv * w;
        }
    }
    out
}

pub fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { (1.0 + x.exp()).ln() }
}

pub fn per_dim_factors(scale: ArrayView1<f32>, head_dim: usize) -> Vec<f32> {
    let factor = RECIPROCAL_OF_SOFTPLUS_0 / (head_dim as f32).sqrt();
    scale.iter().map(|&s| factor * softplus(s)).collect()
}

pub fn apply_per_dim_factors(q: ArrayView4<f32>, factors: &[f32]) -> Array4<f32> {
    let mut out = q.to_owned();
    let (b, n, h, hd) = out.dim();
    for bi in 0..b {
        for ni in 0..n {
            for hi in 0..h {
                for di in 0..hd {
                    out[[bi, ni, hi, di]] *= factors[di];
                }
            }
        }
    }
    out
}

pub fn per_dim_scale(q: ArrayView4<f32>, scale: ArrayView1<f32>, head_dim: usize) -> Array4<f32> {
    apply_per_dim_factors(q, &per_dim_factors(scale, head_dim))
}

pub fn apply_rope4(x: ArrayView4<f32>, positions: ArrayView2<i32>) -> Array4<f32> {
    let (b, n, h, hd) = x.dim();
    let half = hd / 2;
    let mut out = x.to_owned();
    let min_timescale = 1.0f32;
    let max_timescale = 10_000.0f32;
    let mut timescale = vec![0.0f32; half];
    for i in 0..half {
        let fraction = 2.0 * i as f32 / hd as f32;
        timescale[i] = min_timescale * (max_timescale / min_timescale).powf(fraction);
    }
    for bi in 0..b {
        for ni in 0..n {
            let pos = positions[[bi, ni]] as f32;
            for hi in 0..h {
                for d in 0..half {
                    let angle = pos / timescale[d];
                    let sin = angle.sin();
                    let cos = angle.cos();
                    let a = out[[bi, ni, hi, d]];
                    let b_ = out[[bi, ni, hi, d + half]];
                    out[[bi, ni, hi, d]] = a * cos - b_ * sin;
                    out[[bi, ni, hi, d + half]] = b_ * cos + a * sin;
                }
            }
        }
    }
    out
}

pub fn safe_div_sigma(sigma: f32) -> f32 {
    if sigma < REVIN_TOLERANCE { 1.0 } else { sigma }
}

pub fn cumprod_bool_mask(mask: &Array3<bool>) -> Array3<bool> {
    let (b, v, n) = mask.dim();
    let mut out = Array3::<bool>::from_elem((b, v, n), false);
    for bi in 0..b {
        for vi in 0..v {
            let mut prod = true;
            for ni in 0..n {
                prod = prod && mask[[bi, vi, ni]];
                out[[bi, vi, ni]] = prod;
            }
        }
    }
    out
}

pub fn merge_heads(x: ArrayView4<f32>, model_dims: usize) -> Array2<f32> {
    let (b, n, h, hd) = x.dim();
    let mut out = Array2::zeros((b * n, model_dims));
    for bi in 0..b {
        for ni in 0..n {
            for hi in 0..h {
                for di in 0..hd {
                    out[[bi * n + ni, hi * hd + di]] = x[[bi, ni, hi, di]];
                }
            }
        }
    }
    out
}

pub fn split_heads(x: ArrayView2<f32>, b: usize, n: usize, h: usize, hd: usize) -> Array4<f32> {
    let mut out = Array4::zeros((b, n, h, hd));
    for bi in 0..b {
        for ni in 0..n {
            for hi in 0..h {
                for di in 0..hd {
                    out[[bi, ni, hi, di]] = x[[bi * n + ni, hi * hd + di]];
                }
            }
        }
    }
    out
}

pub fn matmul_attn_scores(q: ArrayView4<f32>, k: ArrayView4<f32>, scale: f32) -> Array4<f32> {
    let (b, h, ql, d) = q.dim();
    let kl = k.shape()[2];
    let mut scores = Array4::<f32>::zeros((b, h, ql, kl));
    for bi in 0..b {
        for hi in 0..h {
            for qi in 0..ql {
                for ki in 0..kl {
                    let mut s = 0.0f32;
                    for di in 0..d {
                        s += q[[bi, hi, qi, di]] * k[[bi, hi, ki, di]];
                    }
                    scores[[bi, hi, qi, ki]] = s * scale;
                }
            }
        }
    }
    scores
}

/// Additive attention bias `[batch, seq, seq]` (matches compiled `MaskKind::Bias`).
pub fn add_attn_bias(scores: &mut Array4<f32>, bias: &[f32], batch: usize, seq: usize) {
    let (b, h, ql, kl) = scores.dim();
    assert_eq!(b, batch);
    assert_eq!(ql, seq);
    assert_eq!(kl, seq);
    for bi in 0..b {
        for hi in 0..h {
            for qi in 0..ql {
                for ki in 0..kl {
                    scores[[bi, hi, qi, ki]] += bias[bi * seq * seq + qi * seq + ki];
                }
            }
        }
    }
}

/// Last-axis softmax on `[batch, heads, query, key]` (matches compiled SDPA).
pub fn softmax_last_axis4(scores: &mut Array4<f32>) {
    let (b, h, ql, kl) = scores.dim();
    for bi in 0..b {
        for hi in 0..h {
            for qi in 0..ql {
                let mut max = f32::NEG_INFINITY;
                for ki in 0..kl {
                    max = max.max(scores[[bi, hi, qi, ki]]);
                }
                let mut sum = 0.0f32;
                for ki in 0..kl {
                    let e = (scores[[bi, hi, qi, ki]] - max).exp();
                    scores[[bi, hi, qi, ki]] = e;
                    sum += e;
                }
                if sum > 0.0 {
                    for ki in 0..kl {
                        scores[[bi, hi, qi, ki]] /= sum;
                    }
                } else {
                    for ki in 0..kl {
                        scores[[bi, hi, qi, ki]] = 0.0;
                    }
                }
            }
        }
    }
}

pub fn attn_weighted(v: ArrayView4<f32>, weights: ArrayView4<f32>) -> Array4<f32> {
    let (b, h, ql, kl) = weights.dim();
    let d = v.shape()[3];
    let mut out = Array4::<f32>::zeros((b, h, ql, d));
    for bi in 0..b {
        for hi in 0..h {
            for qi in 0..ql {
                for di in 0..d {
                    let mut acc = 0.0f32;
                    for ki in 0..kl {
                        acc += weights[[bi, hi, qi, ki]] * v[[bi, hi, ki, di]];
                    }
                    out[[bi, hi, qi, di]] = acc;
                }
            }
        }
    }
    out
}
