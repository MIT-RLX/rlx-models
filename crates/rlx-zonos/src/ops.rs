// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0

//! Eager float32 primitives for Zonos transformer.

use matrixmultiply::sgemm;

#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// `y = x @ W^T` (+ optional bias). `W` is `[out, in]` row-major.
pub fn linear(
    x: &[f32],
    w: &[f32],
    bias: Option<&[f32]>,
    seq: usize,
    out_dim: usize,
    in_dim: usize,
) -> Vec<f32> {
    debug_assert_eq!(x.len(), seq * in_dim);
    debug_assert_eq!(w.len(), out_dim * in_dim);
    let mut y = vec![0.0; seq * out_dim];
    if seq == 0 || out_dim == 0 || in_dim == 0 {
        return y;
    }
    // C[seq,out] = X[seq,in] * W[out,in]^T
    unsafe {
        sgemm(
            seq,
            in_dim,
            out_dim,
            1.0,
            x.as_ptr(),
            in_dim as isize,
            1,
            w.as_ptr(),
            1,
            in_dim as isize,
            0.0,
            y.as_mut_ptr(),
            out_dim as isize,
            1,
        );
    }
    if let Some(b) = bias {
        debug_assert_eq!(b.len(), out_dim);
        for t in 0..seq {
            let row = &mut y[t * out_dim..(t + 1) * out_dim];
            for (o, &bv) in b.iter().enumerate() {
                row[o] += bv;
            }
        }
    }
    y
}

/// LayerNorm over last dim (`affine=True`).
pub fn layer_norm(
    x: &[f32],
    weight: &[f32],
    bias: &[f32],
    seq: usize,
    dim: usize,
    eps: f32,
) -> Vec<f32> {
    debug_assert_eq!(x.len(), seq * dim);
    debug_assert_eq!(weight.len(), dim);
    debug_assert_eq!(bias.len(), dim);
    let mut out = vec![0.0; x.len()];
    for t in 0..seq {
        let row = &x[t * dim..(t + 1) * dim];
        let mean = row.iter().sum::<f32>() / dim as f32;
        let mut var = 0.0f32;
        for &v in row {
            let d = v - mean;
            var += d * d;
        }
        var /= dim as f32;
        let inv = 1.0 / (var + eps).sqrt();
        let o = &mut out[t * dim..(t + 1) * dim];
        for j in 0..dim {
            o[j] = (row[j] - mean) * inv * weight[j] + bias[j];
        }
    }
    out
}

/// gpt-fast interleaved RoPE cache: `[seq, head_dim/2, 2]` as flat `[seq * (hd/2) * 2]`.
pub fn precompute_freqs_cis(seq_len: usize, n_elem: usize, base: f32) -> Vec<f32> {
    let half = n_elem / 2;
    let mut freqs = Vec::with_capacity(half);
    for i in 0..half {
        let exp = (2 * i) as f32 / n_elem as f32;
        freqs.push(1.0 / base.powf(exp));
    }
    let mut cache = vec![0.0; seq_len * half * 2];
    for t in 0..seq_len {
        for i in 0..half {
            let angle = t as f32 * freqs[i];
            let idx = (t * half + i) * 2;
            cache[idx] = angle.cos();
            cache[idx + 1] = angle.sin();
        }
    }
    cache
}

/// Apply interleaved RoPE to `[seq, n_heads, head_dim]` (packed as seq*heads*hd).
pub fn apply_rotary_emb(
    x: &mut [f32],
    freqs_cis: &[f32],
    seq: usize,
    n_heads: usize,
    head_dim: usize,
    pos0: usize,
) {
    let half = head_dim / 2;
    for t in 0..seq {
        let pos = pos0 + t;
        let fc = &freqs_cis[pos * half * 2..(pos + 1) * half * 2];
        for h in 0..n_heads {
            let base = (t * n_heads + h) * head_dim;
            for i in 0..half {
                let i0 = base + 2 * i;
                let xr = x[i0];
                let xi = x[i0 + 1];
                let cos = fc[i * 2];
                let sin = fc[i * 2 + 1];
                x[i0] = xr * cos - xi * sin;
                x[i0 + 1] = xi * cos + xr * sin;
            }
        }
    }
}

pub fn softmax_inplace(logits: &mut [f32]) {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in logits.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    let inv = 1.0 / sum.max(1e-12);
    for v in logits.iter_mut() {
        *v *= inv;
    }
}

pub fn argmax(logits: &[f32]) -> usize {
    let mut best_i = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best_i = i;
        }
    }
    best_i
}
