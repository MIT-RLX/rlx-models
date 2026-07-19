// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0

//! Eager TorchZonosBackbone (gpt-fast GQA + silu-gated MLP).

use anyhow::{Result, bail};

use crate::config::ZonosFileConfig;
use crate::ops::{apply_rotary_emb, layer_norm, linear, precompute_freqs_cis, silu};
use crate::weights::WeightMap;

pub struct KvCache {
    /// Per layer: flat `[batch * max_seq * 2 * n_kv * head_dim]` layout
    /// index `(b, t, kv, h, d)` → `((((b*max_seq+t)*2+kv)*n_kv+h)*head_dim+d)`.
    layers: Vec<Vec<f32>>,
    pub batch: usize,
    pub max_seq: usize,
    pub n_kv: usize,
    pub head_dim: usize,
}

impl KvCache {
    pub fn new(n_layer: usize, batch: usize, max_seq: usize, n_kv: usize, head_dim: usize) -> Self {
        let n = batch * max_seq * 2 * n_kv * head_dim;
        Self {
            layers: (0..n_layer).map(|_| vec![0.0; n]).collect(),
            batch,
            max_seq,
            n_kv,
            head_dim,
        }
    }
}

#[inline]
fn kv_idx(
    max_seq: usize,
    n_kv: usize,
    head_dim: usize,
    b: usize,
    t: usize,
    kv: usize,
    h: usize,
    d: usize,
) -> usize {
    ((((b * max_seq + t) * 2 + kv) * n_kv + h) * head_dim) + d
}

pub struct BackboneState {
    pub freqs_cis: Vec<f32>,
    pub seqlen_offset: usize,
}

impl BackboneState {
    pub fn new(head_dim: usize) -> Self {
        Self {
            freqs_cis: precompute_freqs_cis(16_384, head_dim, 10_000.0),
            seqlen_offset: 0,
        }
    }
}

/// Forward through all layers. `x` is `[batch * seq * d_model]`.
/// Returns last-token hidden for each batch: `[batch * d_model]`.
pub fn forward_last(
    cfg: &ZonosFileConfig,
    w: &WeightMap,
    x: &[f32],
    batch: usize,
    seq: usize,
    cache: &mut KvCache,
    state: &BackboneState,
) -> Result<Vec<f32>> {
    let d = cfg.backbone.d_model;
    let n_layer = cfg.backbone.n_layer;
    let n_heads = cfg.backbone.attn_cfg.num_heads;
    let n_kv = cfg.backbone.attn_cfg.num_heads_kv;
    let head_dim = cfg.head_dim();
    let eps = cfg.backbone.norm_epsilon;
    let mlp_inter = cfg.backbone.attn_mlp_d_intermediate;
    anyhow::ensure!(x.len() == batch * seq * d, "hidden size mismatch");
    anyhow::ensure!(cache.batch == batch, "cache batch mismatch");

    let mut h = x.to_vec();
    for layer in 0..n_layer {
        h = transformer_block(
            w, layer, &h, batch, seq, d, n_heads, n_kv, head_dim, mlp_inter, eps, cache, state,
        )?;
    }
    let nw = w.get("backbone.norm_f.weight")?;
    let nb = w.get("backbone.norm_f.bias")?;
    let normed = layer_norm(&h, nw, nb, batch * seq, d, eps);
    // Last token per batch.
    let mut out = vec![0.0; batch * d];
    for b in 0..batch {
        let src = ((b * seq) + (seq - 1)) * d;
        out[b * d..(b + 1) * d].copy_from_slice(&normed[src..src + d]);
    }
    Ok(out)
}

fn transformer_block(
    w: &WeightMap,
    layer: usize,
    x: &[f32],
    batch: usize,
    seq: usize,
    d: usize,
    n_heads: usize,
    n_kv: usize,
    head_dim: usize,
    mlp_inter: usize,
    eps: f32,
    cache: &mut KvCache,
    state: &BackboneState,
) -> Result<Vec<f32>> {
    let pref = format!("backbone.layers.{layer}");
    let nw = w.get(&format!("{pref}.norm.weight"))?;
    let nb = w.get(&format!("{pref}.norm.bias"))?;
    let xn = layer_norm(x, nw, nb, batch * seq, d, eps);
    let attn = attention(
        w, &pref, &xn, batch, seq, d, n_heads, n_kv, head_dim, layer, cache, state,
    )?;
    let mut h = vec![0.0; x.len()];
    for i in 0..x.len() {
        h[i] = x[i] + attn[i];
    }

    let nw2 = w.get(&format!("{pref}.norm2.weight"))?;
    let nb2 = w.get(&format!("{pref}.norm2.bias"))?;
    let xn2 = layer_norm(&h, nw2, nb2, batch * seq, d, eps);
    let ff = feed_forward(w, &pref, &xn2, batch * seq, d, mlp_inter)?;
    for i in 0..h.len() {
        h[i] += ff[i];
    }
    Ok(h)
}

fn feed_forward(
    w: &WeightMap,
    pref: &str,
    x: &[f32],
    seq: usize,
    d: usize,
    inter: usize,
) -> Result<Vec<f32>> {
    let fc1 = w.get(&format!("{pref}.mlp.fc1.weight"))?;
    let fc2 = w.get(&format!("{pref}.mlp.fc2.weight"))?;
    // fc1: [2*inter, d] → y, gate
    let fused = linear(x, fc1, None, seq, 2 * inter, d);
    let mut y = vec![0.0; seq * inter];
    for t in 0..seq {
        let base = t * 2 * inter;
        for i in 0..inter {
            y[t * inter + i] = fused[base + i] * silu(fused[base + inter + i]);
        }
    }
    Ok(linear(&y, fc2, None, seq, d, inter))
}

fn attention(
    w: &WeightMap,
    pref: &str,
    x: &[f32],
    batch: usize,
    seq: usize,
    d: usize,
    n_heads: usize,
    n_kv: usize,
    head_dim: usize,
    layer: usize,
    cache: &mut KvCache,
    state: &BackboneState,
) -> Result<Vec<f32>> {
    let in_proj = w.get(&format!("{pref}.mixer.in_proj.weight"))?;
    let out_proj = w.get(&format!("{pref}.mixer.out_proj.weight"))?;
    let q_size = n_heads * head_dim;
    let kv_size = n_kv * head_dim;
    let total = q_size + 2 * kv_size;
    if in_proj.len() != total * d {
        bail!("in_proj size {} != {}*{d}", in_proj.len(), total);
    }
    let qkv = linear(x, in_proj, None, batch * seq, total, d);

    let mut q = vec![0.0; batch * seq * q_size];
    let mut k = vec![0.0; batch * seq * kv_size];
    let mut v = vec![0.0; batch * seq * kv_size];
    for b in 0..batch {
        for t in 0..seq {
            let src = (b * seq + t) * total;
            let dst = (b * seq + t) * q_size;
            q[dst..dst + q_size].copy_from_slice(&qkv[src..src + q_size]);
            let dst_k = (b * seq + t) * kv_size;
            k[dst_k..dst_k + kv_size].copy_from_slice(&qkv[src + q_size..src + q_size + kv_size]);
            v[dst_k..dst_k + kv_size].copy_from_slice(&qkv[src + q_size + kv_size..src + total]);
        }
    }

    // Reshape helper views as [batch*seq, n_heads, head_dim] for RoPE.
    // apply_rotary expects packed [seq_total, n_heads, hd] with contiguous heads —
    // apply per-batch.
    let pos0 = state.seqlen_offset;
    for b in 0..batch {
        let q_slice = &mut q[b * seq * q_size..(b + 1) * seq * q_size];
        apply_rotary_emb(q_slice, &state.freqs_cis, seq, n_heads, head_dim, pos0);
        let k_slice = &mut k[b * seq * kv_size..(b + 1) * seq * kv_size];
        apply_rotary_emb(k_slice, &state.freqs_cis, seq, n_kv, head_dim, pos0);
    }

    // Write K/V into cache; sequence positions [pos0, pos0+seq).
    let max_seq = cache.max_seq;
    let layer_cache = &mut cache.layers[layer];
    for b in 0..batch {
        for t in 0..seq {
            let abs_t = pos0 + t;
            if abs_t >= max_seq {
                bail!("KV cache overflow at {abs_t}");
            }
            for h in 0..n_kv {
                for d_i in 0..head_dim {
                    let src_k = ((b * seq + t) * n_kv + h) * head_dim + d_i;
                    let ik = kv_idx(max_seq, n_kv, head_dim, b, abs_t, 0, h, d_i);
                    let iv = kv_idx(max_seq, n_kv, head_dim, b, abs_t, 1, h, d_i);
                    layer_cache[ik] = k[src_k];
                    layer_cache[iv] = v[((b * seq + t) * n_kv + h) * head_dim + d_i];
                }
            }
        }
    }

    let kv_len = pos0 + seq;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let causal = seq > 1;
    let groups = n_heads / n_kv;
    let mut y = vec![0.0; batch * seq * q_size];

    for b in 0..batch {
        for hq in 0..n_heads {
            let hk = hq / groups;
            for tq in 0..seq {
                let abs_q = pos0 + tq;
                let mut scores = vec![0.0f32; kv_len];
                let q_base = ((b * seq + tq) * n_heads + hq) * head_dim;
                for tk in 0..kv_len {
                    if causal && tk > abs_q {
                        scores[tk] = f32::NEG_INFINITY;
                        continue;
                    }
                    let mut dot = 0.0f32;
                    for d_i in 0..head_dim {
                        let ik = kv_idx(max_seq, n_kv, head_dim, b, tk, 0, hk, d_i);
                        dot += q[q_base + d_i] * layer_cache[ik];
                    }
                    scores[tk] = dot * scale;
                }
                let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for s in &mut scores {
                    *s = (*s - max).exp();
                    sum += *s;
                }
                let inv = 1.0 / sum.max(1e-12);
                for s in &mut scores {
                    *s *= inv;
                }
                let y_base = ((b * seq + tq) * n_heads + hq) * head_dim;
                for d_i in 0..head_dim {
                    let mut acc = 0.0f32;
                    for tk in 0..kv_len {
                        let iv = kv_idx(max_seq, n_kv, head_dim, b, tk, 1, hk, d_i);
                        acc += scores[tk] * layer_cache[iv];
                    }
                    y[y_base + d_i] = acc;
                }
            }
        }
    }

    Ok(linear(&y, out_proj, None, batch * seq, d, q_size))
}
