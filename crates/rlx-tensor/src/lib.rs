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

//! Small host-side tensor kernels used by the native SAM3 bring-up.
//!
//! Compute-heavy ops go through `rlx_cpu::blas::sgemm_auto` so we get
//! Accelerate/OpenBLAS instead of naive Rust loops. The pure-Rust path
//! falls back to a scalar gemm with the same signature (see
//! `rlx-cpu/src/blas.rs`).

use anyhow::{Result, ensure};
use rlx_cpu::blas;

pub fn layer_norm(
    x: &[f32],
    gamma: &[f32],
    beta: &[f32],
    dim: usize,
    eps: f32,
) -> Result<Vec<f32>> {
    ensure!(
        x.len().is_multiple_of(dim),
        "layer_norm input len must be divisible by dim"
    );
    ensure!(gamma.len() == dim, "layer_norm gamma len mismatch");
    ensure!(beta.len() == dim, "layer_norm beta len mismatch");
    let rows = x.len() / dim;
    let mut out = vec![0.0; x.len()];
    for r in 0..rows {
        let row = &x[r * dim..(r + 1) * dim];
        let mean = row.iter().sum::<f32>() / dim as f32;
        let var = row
            .iter()
            .map(|v| {
                let d = *v - mean;
                d * d
            })
            .sum::<f32>()
            / dim as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for c in 0..dim {
            out[r * dim + c] = (row[c] - mean) * inv * gamma[c] + beta[c];
        }
    }
    Ok(out)
}

/// `out = x @ w_t + b` where `w_t` is `[in_dim, out_dim]` (already
/// transposed at load time by `WeightMap::take_transposed`).
pub fn linear(
    x: &[f32],
    rows: usize,
    in_dim: usize,
    w_t: &[f32],
    out_dim: usize,
    b: &[f32],
) -> Result<Vec<f32>> {
    ensure!(x.len() == rows * in_dim, "linear input shape mismatch");
    ensure!(
        w_t.len() == in_dim * out_dim,
        "linear weight shape mismatch"
    );
    ensure!(b.len() == out_dim, "linear bias shape mismatch");
    let mut out = vec![0.0f32; rows * out_dim];
    blas::sgemm_bias(x, w_t, b, &mut out, rows, in_dim, out_dim);
    Ok(out)
}

/// `out = a @ b` with default row-major no-transpose strides.
pub fn matmul(a: &[f32], b: &[f32], out: &mut [f32], m: usize, k: usize, n: usize) {
    blas::sgemm(a, b, out, m, k, n);
}

/// `out = alpha * a @ b^T` (no bias). `b` is `[n, k]`.
pub fn matmul_bt(a: &[f32], b: &[f32], out: &mut [f32], m: usize, k: usize, n: usize, alpha: f32) {
    blas::sgemm_bt(a, b, out, m, k, n, alpha);
}

pub fn gelu_tanh(x: &mut [f32]) {
    const K: f32 = 0.797_884_6;
    for v in x {
        let x3 = *v * *v * *v;
        *v = 0.5 * *v * (1.0 + (K * (*v + 0.044715 * x3)).tanh());
    }
}

// Kept for the eventual mask-decoder sigmoid epilogue (referenced
// in SAM3's prediction-head spec but not yet wired into a caller).
#[allow(dead_code)]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Multi-head attention forward for an upstream `nn.MultiheadAttention`
/// with the standard `in_proj_weight [3*E, E]` / `out_proj.weight [E, E]`
/// parameterisation. Inputs are batch-first `[B, L_q, E]` / `[B, L_k, E]`.
/// `key_padding_mask` (when present) is `[B, L_k]` with `true` meaning the
/// position must be ignored.
#[allow(clippy::too_many_arguments)]
pub fn multihead_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    in_proj_w_t: &[f32], // [E, 3*E]
    in_proj_b: &[f32],   // [3*E]
    out_proj_w_t: &[f32],
    out_proj_b: &[f32],
    batch: usize,
    l_q: usize,
    l_k: usize,
    embed_dim: usize,
    num_heads: usize,
    key_padding_mask: Option<&[u8]>,
) -> Result<Vec<f32>> {
    ensure!(
        embed_dim.is_multiple_of(num_heads),
        "embed_dim {embed_dim} not divisible by num_heads {num_heads}"
    );
    let head_dim = embed_dim / num_heads;

    // PyTorch's nn.MultiheadAttention concatenates Q,K,V projections into
    // in_proj_weight = [W_q; W_k; W_v]. For self-attention (q==k==v) we
    // could fuse them; for safety we always do three linears.
    let (wq, wk, wv) = split_in_proj_w(in_proj_w_t, embed_dim);
    let bq = &in_proj_b[0..embed_dim];
    let bk = &in_proj_b[embed_dim..2 * embed_dim];
    let bv = &in_proj_b[2 * embed_dim..3 * embed_dim];

    let q_proj = linear(q, batch * l_q, embed_dim, &wq, embed_dim, bq)?;
    let k_proj = linear(k, batch * l_k, embed_dim, &wk, embed_dim, bk)?;
    let v_proj = linear(v, batch * l_k, embed_dim, &wv, embed_dim, bv)?;

    // Reshape to [B, H, L, D].
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

    // Repack [B, H, L_q, D] → [B, L_q, E].
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

fn split_in_proj_w(in_proj_w_t: &[f32], embed_dim: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    // in_proj_w_t is stored as the transposed [E, 3*E] (row-major). The
    // upstream nn.MultiheadAttention has [3*E, E] in source order. After
    // `take_transposed`, output shape is [E, 3*E]; we want to extract the
    // three [E, E] slabs at output stride 3*E.
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
    src: &[f32],
    dst: &mut [f32],
    batch: usize,
    l: usize,
    num_heads: usize,
    head_dim: usize,
) {
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

/// In-place row-wise softmax: each `cols`-long row is normalised
/// independently. Numerically stable.
pub fn softmax_rows(x: &mut [f32], rows: usize, cols: usize) {
    for r in 0..rows {
        let row = &mut x[r * cols..(r + 1) * cols];
        let mut m = row[0];
        for &v in row.iter().skip(1) {
            if v > m {
                m = v;
            }
        }
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
