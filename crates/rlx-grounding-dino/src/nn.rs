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

//! Host-side neural-net primitives for the CPU-native reference path.
//!
//! Conventions: weights are in PyTorch `[out, in]` layout (exactly as stored
//! in the checkpoint — no transpose at load time), activations are row-major
//! `[rows, dim]`. These mirror the math the IR graphs compile to, so the
//! native path is the parity anchor for the GPU backends.

/// `y = x @ w^T + b`. `x` is `[rows, in_dim]`, `w` is `[out_dim, in_dim]`,
/// `b` is `[out_dim]` (or empty for no bias).
pub fn linear(
    x: &[f32],
    rows: usize,
    in_dim: usize,
    w: &[f32],
    out_dim: usize,
    b: &[f32],
) -> Vec<f32> {
    debug_assert_eq!(x.len(), rows * in_dim);
    debug_assert_eq!(w.len(), out_dim * in_dim);
    let mut out = vec![0f32; rows * out_dim];
    rlx_cpu::blas::sgemm_bt(x, w, &mut out, rows, in_dim, out_dim, 1.0);
    if !b.is_empty() {
        debug_assert_eq!(b.len(), out_dim);
        for r in 0..rows {
            let row = &mut out[r * out_dim..(r + 1) * out_dim];
            for (o, bv) in row.iter_mut().zip(b.iter()) {
                *o += *bv;
            }
        }
    }
    out
}

/// LayerNorm over the last `dim` elements of each row.
pub fn layer_norm(x: &[f32], gamma: &[f32], beta: &[f32], dim: usize, eps: f32) -> Vec<f32> {
    debug_assert_eq!(x.len() % dim, 0, "layer_norm input not divisible by dim");
    debug_assert_eq!(gamma.len(), dim);
    debug_assert_eq!(beta.len(), dim);
    let rows = x.len() / dim;
    let mut out = vec![0f32; x.len()];
    for r in 0..rows {
        let row = &x[r * dim..(r + 1) * dim];
        let mean = row.iter().sum::<f32>() / dim as f32;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / dim as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for c in 0..dim {
            out[r * dim + c] = (row[c] - mean) * inv * gamma[c] + beta[c];
        }
    }
    out
}

/// In-place ReLU.
#[allow(dead_code)] // used by enhancer/decoder FFNs (later phases)
pub fn relu(x: &mut [f32]) {
    for v in x {
        if *v < 0.0 {
            *v = 0.0;
        }
    }
}

/// In-place exact (erf) GELU — matches PyTorch `nn.GELU()` / HF BERT `gelu`.
pub fn gelu_erf(x: &mut [f32]) {
    const INV_SQRT2: f32 = std::f32::consts::FRAC_1_SQRT_2;
    for v in x {
        *v = 0.5 * *v * (1.0 + erf(*v * INV_SQRT2));
    }
}

/// Abramowitz–Stegun 7.1.26 erf approximation (max abs err ~1.5e-7).
#[allow(clippy::excessive_precision)]
fn erf(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-x * x).exp();
    sign * y
}

#[allow(dead_code)] // used by box decoding / query selection (later phases)
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Numerically-stable in-place row-wise softmax over `cols`.
pub fn softmax_rows(x: &mut [f32], rows: usize, cols: usize) {
    for r in 0..rows {
        let row = &mut x[r * cols..(r + 1) * cols];
        let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for v in row.iter_mut() {
            *v = (*v - m).exp();
            sum += *v;
        }
        let inv = 1.0 / sum;
        for v in row.iter_mut() {
            *v *= inv;
        }
    }
}

/// Additive attention bias selector. All variants are added to the raw
/// `[heads, lq, lk]` attention scores before softmax.
#[allow(dead_code)] // `None` used by deformable / cross attention (later phases)
pub enum AttnBias<'a> {
    None,
    /// `[lq, lk]` shared across heads (e.g. a phrase self-attention mask, with
    /// `-inf`/0 entries).
    Shared(&'a [f32]),
    /// `[heads, lq, lk]` per-head bias (e.g. Swin relative-position bias).
    PerHead(&'a [f32]),
}

/// Generic multi-head attention with explicit Q/K/V/out projections (PyTorch
/// `[out,in]` weights) and an optional additive bias. Single batch element.
///
/// `q_in` is `[lq, dim]`, `k_in`/`v_in` are `[lk, dim]`. Returns `[lq, dim]`.
#[allow(clippy::too_many_arguments)]
pub fn mha(
    q_in: &[f32],
    k_in: &[f32],
    v_in: &[f32],
    lq: usize,
    lk: usize,
    dim: usize,
    num_heads: usize,
    qw: &[f32],
    qb: &[f32],
    kw: &[f32],
    kb: &[f32],
    vw: &[f32],
    vb: &[f32],
    ow: &[f32],
    ob: &[f32],
    bias: AttnBias<'_>,
) -> Vec<f32> {
    let hd = dim / num_heads;
    let scale = 1.0 / (hd as f32).sqrt();
    let q = linear(q_in, lq, dim, qw, dim, qb);
    let k = linear(k_in, lk, dim, kw, dim, kb);
    let v = linear(v_in, lk, dim, vw, dim, vb);

    let mut ctx = vec![0f32; lq * dim];
    let mut scores = vec![0f32; lq * lk];
    for h in 0..num_heads {
        // scores[i,j] = scale * <q_i, k_j>
        for i in 0..lq {
            let qrow = &q[i * dim + h * hd..i * dim + h * hd + hd];
            for j in 0..lk {
                let krow = &k[j * dim + h * hd..j * dim + h * hd + hd];
                let mut s = 0f32;
                for d in 0..hd {
                    s += qrow[d] * krow[d];
                }
                s *= scale;
                s += match bias {
                    AttnBias::None => 0.0,
                    AttnBias::Shared(b) => b[i * lk + j],
                    AttnBias::PerHead(b) => b[(h * lq + i) * lk + j],
                };
                scores[i * lk + j] = s;
            }
        }
        softmax_rows(&mut scores, lq, lk);
        // ctx_i = sum_j p_ij * v_j
        for i in 0..lq {
            let crow = &mut ctx[i * dim + h * hd..i * dim + h * hd + hd];
            for j in 0..lk {
                let p = scores[i * lk + j];
                if p == 0.0 {
                    continue;
                }
                let vrow = &v[j * dim + h * hd..j * dim + h * hd + hd];
                for d in 0..hd {
                    crow[d] += p * vrow[d];
                }
            }
        }
    }
    linear(&ctx, lq, dim, ow, dim, ob)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_matches_manual() {
        // x [2,3], w [2,3] (out=2,in=3), b [2]
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let w = vec![1.0, 0.0, 0.0, 0.0, 1.0, 1.0];
        let b = vec![0.5, -0.5];
        let y = linear(&x, 2, 3, &w, 2, &b);
        // row0: [1*1, 2+3] + b = [1.5, 4.5]; row1: [4, 5+6] + b = [4.5, 10.5]
        assert_eq!(y, vec![1.5, 4.5, 4.5, 10.5]);
    }

    #[test]
    fn gelu_erf_near_zero_and_large() {
        let mut x = vec![0.0, 6.0, -6.0];
        gelu_erf(&mut x);
        assert!(x[0].abs() < 1e-6);
        assert!((x[1] - 6.0).abs() < 1e-3);
        assert!(x[2].abs() < 1e-3);
    }

    #[test]
    fn mha_uniform_when_scores_equal() {
        // identity projections, zero bias → attention averages V uniformly
        // when all keys are identical.
        let dim = 4;
        let (lq, lk, nh) = (1, 3, 2);
        let id: Vec<f32> = (0..dim * dim)
            .map(|i| if i / dim == i % dim { 1.0 } else { 0.0 })
            .collect();
        let zb = vec![0f32; dim];
        let q = vec![0f32; lq * dim]; // zero query → equal scores
        let k = vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0];
        let v = vec![2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0];
        let out = mha(
            &q,
            &k,
            &v,
            lq,
            lk,
            dim,
            nh,
            &id,
            &zb,
            &id,
            &zb,
            &id,
            &zb,
            &id,
            &zb,
            AttnBias::None,
        );
        // average of v rows = [2,0,0,0]
        assert!((out[0] - 2.0).abs() < 1e-5);
        assert!(out[1].abs() < 1e-5);
    }
}
