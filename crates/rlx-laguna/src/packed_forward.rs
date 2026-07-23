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

//! Packed mmap Laguna forward — fused GGUF dequant+matmul, no full F32 expand.
//!
//! Uses [`rlx_cpu::gguf_matmul`] / [`rlx_cpu::lm_head`] against bytes borrowed
//! from the retained [`GgufLoader`] mmap (experts use fused blocks — never the
//! process-wide F32 dequant cache). Optional [`DeviceMatmul`] accelerates packed
//! mats on Metal / MLX. [`generate`] prefills once then KV-cached decode steps.

use crate::config::{AttnGating, LagunaConfig, RopeLayerParams};
use crate::device_matmul::DeviceMatmul;
use crate::packed::{LagunaPackedFfn, LagunaPackedWeights, MatWeight};
use anyhow::{Result, anyhow, bail};
use rlx_core::GgufLoader;
use rlx_cpu::gguf_matmul::{gguf_matmul_bt, gguf_matmul_bt_dispatch};
use rlx_cpu::lm_head::gguf_tied_lm_argmax;
use rlx_flow::rope::{YarnScaling, default_inv_freq, yarn_scaled_inv_freq};
use rlx_gguf::QK_K;
use rlx_ir::quant::QuantScheme;

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}

fn rms_norm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let h = weight.len();
    assert_eq!(x.len() % h, 0);
    let t = x.len() / h;
    let mut out = vec![0.0; x.len()];
    for ti in 0..t {
        let row = &x[ti * h..(ti + 1) * h];
        let mean_sq = row.iter().map(|v| v * v).sum::<f32>() / h as f32;
        let inv = 1.0 / (mean_sq + eps).sqrt();
        for j in 0..h {
            out[ti * h + j] = row[j] * inv * weight[j];
        }
    }
    out
}

fn softmax_row(logits: &mut [f32]) {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for v in logits.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    let inv = 1.0 / sum.max(1e-12);
    for v in logits.iter_mut() {
        *v *= inv;
    }
}

pub(crate) fn rope_inv_freq(rope: &RopeLayerParams, rot_dim: usize) -> Vec<f64> {
    let use_yarn = rope.yarn_factor > 1.0
        || rope.rope_type.eq_ignore_ascii_case("yarn");
    if use_yarn {
        yarn_scaled_inv_freq(
            rope.rope_theta as f64,
            rot_dim,
            &YarnScaling {
                factor: rope.yarn_factor,
                beta_fast: rope.beta_fast,
                beta_slow: rope.beta_slow,
                original_max_position_embeddings: rope.original_max_position_embeddings
                    .max(1) as u32,
            },
        )
    } else {
        default_inv_freq(rope.rope_theta as f64, rot_dim)
    }
}

pub(crate) fn rotary_freqs(
    pos_start: usize,
    seq: usize,
    rot_dim: usize,
    inv_freq: &[f64],
) -> (Vec<f32>, Vec<f32>) {
    let half = rot_dim / 2;
    debug_assert_eq!(inv_freq.len(), half);
    let mut cos = vec![0.0; seq * rot_dim];
    let mut sin = vec![0.0; seq * rot_dim];
    for t in 0..seq {
        let pos = (pos_start + t) as f64;
        for i in 0..half {
            let angle = pos * inv_freq[i];
            let c = angle.cos() as f32;
            let s = angle.sin() as f32;
            cos[t * rot_dim + i] = c;
            cos[t * rot_dim + half + i] = c;
            sin[t * rot_dim + i] = s;
            sin[t * rot_dim + half + i] = s;
        }
    }
    (cos, sin)
}

#[cfg(test)]
mod yarn_tests {
    use super::*;
    use crate::config::RopeLayerParams;

    #[test]
    fn yarn_inv_freq_differs_from_default() {
        let plain = RopeLayerParams {
            rope_type: "default".into(),
            yarn_factor: 1.0,
            ..RopeLayerParams::default()
        };
        let yarn = RopeLayerParams {
            rope_type: "yarn".into(),
            yarn_factor: 128.0,
            original_max_position_embeddings: 8192,
            beta_fast: 32.0,
            beta_slow: 1.0,
            ..RopeLayerParams::default()
        };
        let a = rope_inv_freq(&plain, 64);
        let b = rope_inv_freq(&yarn, 64);
        assert_eq!(a.len(), b.len());
        let diff = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f64, f64::max);
        assert!(diff > 1e-6, "expected YaRN to change inv_freq, max_diff={diff}");
    }
}

fn apply_rope_inplace(
    x: &mut [f32],
    cos: &[f32],
    sin: &[f32],
    seq: usize,
    n_heads: usize,
    hd: usize,
    rot_dim: usize,
) {
    let half = rot_dim / 2;
    for t in 0..seq {
        for h in 0..n_heads {
            let base = (t * n_heads + h) * hd;
            let head = &mut x[base..base + hd];
            for i in 0..half {
                let a = head[i];
                let b = head[half + i];
                let c = cos[t * rot_dim + i];
                let s = sin[t * rot_dim + i];
                head[i] = a * c - b * s;
                head[half + i] = b * c + a * s;
            }
        }
    }
}

fn linear_f32(x: &[f32], w: &[f32], seq: usize, out_dim: usize, in_dim: usize) -> Vec<f32> {
    debug_assert_eq!(x.len(), seq * in_dim);
    debug_assert_eq!(w.len(), out_dim * in_dim);
    let mut y = vec![0.0; seq * out_dim];
    for t in 0..seq {
        for o in 0..out_dim {
            let mut acc = 0.0;
            let wr = &w[o * in_dim..(o + 1) * in_dim];
            let xr = &x[t * in_dim..(t + 1) * in_dim];
            for i in 0..in_dim {
                acc += xr[i] * wr[i];
            }
            y[t * out_dim + o] = acc;
        }
    }
    y
}

fn mat_bytes<'a>(loader: &'a GgufLoader, m: &MatWeight) -> Result<(&'a [u8], QuantScheme, usize, usize)> {
    match m {
        MatWeight::Packed { key, scheme, shape } => {
            if shape.len() != 2 {
                bail!("expected rank-2 packed mat for {key}, got {shape:?}");
            }
            let n = shape[0];
            let k = shape[1];
            let bytes = loader
                .tensor_bytes_borrowed(key)
                .ok_or_else(|| anyhow!("mmap bytes missing for {key}"))?;
            Ok((bytes, *scheme, n, k))
        }
        MatWeight::F32(_) => bail!("mat_bytes: F32 weight has no packed bytes"),
    }
}

fn linear_mat(
    loader: &GgufLoader,
    m: &MatWeight,
    x: &[f32],
    seq: usize,
    out_dim: usize,
    in_dim: usize,
    accel: Option<&mut DeviceMatmul>,
) -> Result<Vec<f32>> {
    match m {
        MatWeight::F32(w) => Ok(linear_f32(x, w, seq, out_dim, in_dim)),
        MatWeight::Packed { .. } => {
            let (bytes, scheme, n, k) = mat_bytes(loader, m)?;
            if n != out_dim || k != in_dim {
                bail!(
                    "packed mat shape [{n},{k}] != expected out={out_dim} in={in_dim}"
                );
            }
            if let Some(dev) = accel {
                return dev.matmul(x, bytes, seq, in_dim, out_dim, scheme);
            }
            let mut y = vec![0.0; seq * out_dim];
            // Shared/attn mats: cached BLAS is fine (few unique weights).
            gguf_matmul_bt_dispatch(x, bytes, &mut y, seq, in_dim, out_dim, scheme);
            Ok(y)
        }
    }
}

fn packed_gemm(
    x: &[f32],
    bytes: &[u8],
    m: usize,
    k: usize,
    n: usize,
    scheme: QuantScheme,
    accel: Option<&mut DeviceMatmul>,
    // When true, never use the CPU F32 dequant cache (MoE experts).
    expert: bool,
) -> Result<Vec<f32>> {
    if let Some(dev) = accel {
        return dev.matmul(x, bytes, m, k, n, scheme);
    }
    let mut y = vec![0.0; m * n];
    if expert {
        gguf_matmul_bt(x, bytes, &mut y, m, k, n, scheme);
    } else {
        gguf_matmul_bt_dispatch(x, bytes, &mut y, m, k, n, scheme);
    }
    Ok(y)
}

fn dequant_one_block(scheme: QuantScheme, block: &[u8], out: &mut [f32; QK_K]) {
    match scheme {
        QuantScheme::GgufQ4K => rlx_gguf::dequant_q4_k_block(block, out),
        QuantScheme::GgufQ5K => rlx_gguf::dequant_q5_k_block(block, out),
        QuantScheme::GgufQ6K => rlx_gguf::dequant_q6_k_block(block, out),
        QuantScheme::GgufQ8K => rlx_gguf::dequant_q8_k_block(block, out),
        QuantScheme::GgufQ2K => rlx_gguf::dequant_q2_k_block(block, out),
        QuantScheme::GgufQ3K => rlx_gguf::dequant_q3_k_block(block, out),
        other => panic!("laguna embed gather: unsupported scheme {other:?}"),
    }
}

fn block_bytes(scheme: QuantScheme) -> usize {
    scheme.gguf_block_bytes() as usize
}

fn block_elems(scheme: QuantScheme) -> usize {
    scheme.gguf_block_size() as usize
}

/// Gather embedding rows from packed `[vocab, hidden]` (GGML `[hidden, vocab]`).
fn gather_embed(
    loader: &GgufLoader,
    emb: &MatWeight,
    ids: &[u32],
    hidden: usize,
    vocab: usize,
) -> Result<Vec<f32>> {
    match emb {
        MatWeight::F32(w) => {
            let mut x = vec![0.0; ids.len() * hidden];
            for (t, &id) in ids.iter().enumerate() {
                let id = id as usize;
                if id >= vocab {
                    bail!("token id {id} >= vocab {vocab}");
                }
                x[t * hidden..(t + 1) * hidden]
                    .copy_from_slice(&w[id * hidden..(id + 1) * hidden]);
            }
            Ok(x)
        }
        MatWeight::Packed { key, scheme, shape } => {
            // After reverse: [vocab, hidden] or keep ggml — metadata stores reversed.
            let (n_vocab, n_embd) = if shape.len() == 2 {
                (shape[0], shape[1])
            } else {
                bail!("embed shape {shape:?}");
            };
            if n_embd != hidden || n_vocab < vocab {
                // vocab_size from config may be tokenizer size; table may be wider.
            }
            if n_embd != hidden {
                bail!("embed hidden {n_embd} != cfg {hidden}");
            }
            let bytes = loader
                .tensor_bytes_borrowed(key)
                .ok_or_else(|| anyhow!("embed bytes missing"))?;
            let be = block_elems(*scheme);
            let bb = block_bytes(*scheme);
            let mut x = vec![0.0; ids.len() * n_embd];
            if n_embd.is_multiple_of(be) {
                let blocks_per_row = n_embd / be;
                let row_bytes = blocks_per_row * bb;
                let mut block = [0f32; QK_K];
                for (t, &id) in ids.iter().enumerate() {
                    let id = id as usize;
                    if id >= n_vocab {
                        bail!("token id {id} >= embed rows {n_vocab}");
                    }
                    let off = id * row_bytes;
                    let row = &mut x[t * n_embd..(t + 1) * n_embd];
                    for b in 0..blocks_per_row {
                        let boff = off + b * bb;
                        dequant_one_block(*scheme, &bytes[boff..boff + bb], &mut block);
                        row[b * be..(b + 1) * be].copy_from_slice(&block[..be]);
                    }
                }
            } else {
                // Tiny / non-aligned rows: temporary full-table dequant (not used for XS).
                let (data, ggml_shape) = loader
                    .file()
                    .dequant_f32(key)
                    .map_err(|e| anyhow!("embed dequant {key}: {e:#}"))?;
                let ne0 = ggml_shape.first().copied().unwrap_or(n_embd);
                if ne0 != n_embd {
                    bail!("embed ggml ne0={ne0} != {n_embd}");
                }
                for (t, &id) in ids.iter().enumerate() {
                    let id = id as usize;
                    if id >= n_vocab {
                        bail!("token id {id} >= embed rows {n_vocab}");
                    }
                    let base = id * ne0;
                    x[t * n_embd..(t + 1) * n_embd].copy_from_slice(&data[base..base + ne0]);
                }
            }
            Ok(x)
        }
    }
}

fn dense_mlp_mat(
    loader: &GgufLoader,
    x: &[f32],
    gate: &MatWeight,
    up: &MatWeight,
    down: &MatWeight,
    seq: usize,
    h: usize,
    inter: usize,
    mut accel: Option<&mut DeviceMatmul>,
) -> Result<Vec<f32>> {
    let g = linear_mat(loader, gate, x, seq, inter, h, accel.as_deref_mut())?;
    let u = linear_mat(loader, up, x, seq, inter, h, accel.as_deref_mut())?;
    let mut mid = vec![0.0; seq * inter];
    for i in 0..mid.len() {
        mid[i] = silu(g[i]) * u[i];
    }
    linear_mat(loader, down, &mid, seq, h, inter, accel.as_deref_mut())
}

fn expert_slab<'a>(
    bytes: &'a [u8],
    scheme: QuantScheme,
    n_expert: usize,
    n: usize,
    k: usize,
    e: usize,
) -> Result<&'a [u8]> {
    let be = block_elems(scheme);
    let bb = block_bytes(scheme);
    let slab_bytes = (k * n) / be * bb;
    if bytes.len() != n_expert * slab_bytes {
        bail!(
            "expert pack bytes {} != {n_expert} * slab {slab_bytes} (n={n} k={k})",
            bytes.len()
        );
    }
    if e >= n_expert {
        bail!("expert {e} >= {n_expert}");
    }
    Ok(&bytes[e * slab_bytes..(e + 1) * slab_bytes])
}

fn moe_mlp(
    cfg: &LagunaConfig,
    loader: &GgufLoader,
    x: &[f32],
    ffn: &LagunaPackedFfn,
    seq: usize,
    mut accel: Option<&mut DeviceMatmul>,
) -> Result<Vec<f32>> {
    let LagunaPackedFfn::Moe {
        router,
        gate_bias,
        gate_exps,
        up_exps,
        down_exps,
        shared_gate,
        shared_up,
        shared_down,
    } = ffn
    else {
        bail!("moe_mlp: expected MoE ffn");
    };

    let h = cfg.hidden_size;
    let ne = cfg.num_experts;
    let top_k = cfg.num_experts_per_tok.min(ne).max(1);
    let inter = cfg.moe_intermediate_size;
    let shared_inter = cfg.shared_expert_intermediate_size;

    let shared = dense_mlp_mat(
        loader,
        x,
        shared_gate,
        shared_up,
        shared_down,
        seq,
        h,
        shared_inter,
        accel.as_deref_mut(),
    )?;

    // Router is often native F32 [n_expert, hidden] after reverse of [hidden, n_expert].
    let logits = match router {
        MatWeight::F32(w) => {
            // shape may be [ne, h] or we stored as F32 from [2048,256] → reverse [256,2048]
            linear_f32(x, w, seq, ne, h)
        }
        MatWeight::Packed { .. } => {
            linear_mat(loader, router, x, seq, ne, h, accel.as_deref_mut())?
        }
    };
    let mut scores = vec![0.0; seq * ne];
    for i in 0..logits.len() {
        let mut z = logits[i];
        if cfg.moe_router_logit_softcapping > 0.0 {
            let c = cfg.moe_router_logit_softcapping;
            z = (z / c).tanh() * c;
        }
        scores[i] = sigmoid(z);
    }

    let (g_bytes, g_scheme, g_shape) = match gate_exps {
        MatWeight::Packed { key, scheme, shape } => {
            let b = loader
                .tensor_bytes_borrowed(key)
                .ok_or_else(|| anyhow!("missing {key}"))?;
            (b, *scheme, shape.as_slice())
        }
        MatWeight::F32(_) => bail!("gate_exps must stay packed"),
    };
    let (u_bytes, u_scheme, u_shape) = match up_exps {
        MatWeight::Packed { key, scheme, shape } => {
            let b = loader
                .tensor_bytes_borrowed(key)
                .ok_or_else(|| anyhow!("missing {key}"))?;
            (b, *scheme, shape.as_slice())
        }
        MatWeight::F32(_) => bail!("up_exps must stay packed"),
    };
    let (d_bytes, d_scheme, d_shape) = match down_exps {
        MatWeight::Packed { key, scheme, shape } => {
            let b = loader
                .tensor_bytes_borrowed(key)
                .ok_or_else(|| anyhow!("missing {key}"))?;
            (b, *scheme, shape.as_slice())
        }
        MatWeight::F32(_) => bail!("down_exps must stay packed"),
    };
    // [E, n, k] with n=inter, k=hidden for gate/up; down is [E, hidden, inter]
    if g_shape.len() != 3 || g_shape[0] != ne {
        bail!("gate_exps shape {g_shape:?} expected E={ne}");
    }

    let mut out = vec![0.0; seq * h];
    for t in 0..seq {
        let row = &scores[t * ne..(t + 1) * ne];
        let mut order: Vec<(usize, f32)> = (0..ne)
            .map(|e| {
                let b = gate_bias
                    .as_ref()
                    .and_then(|b| b.get(e).copied())
                    .unwrap_or(0.0);
                (e, row[e] + b)
            })
            .collect();
        order.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut picks: Vec<(usize, f32)> = order
            .into_iter()
            .take(top_k)
            .map(|(e, _)| (e, row[e]))
            .collect();
        if cfg.norm_topk_prob {
            let sum: f32 = picks.iter().map(|(_, w)| *w).sum::<f32>().max(1e-12);
            for p in &mut picks {
                p.1 /= sum;
            }
        }
        let xt = &x[t * h..(t + 1) * h];
        for &(e, rw) in &picks {
            // gate/up: [E, inter, hidden] → n=inter, k=hidden
            let gn = g_shape[1];
            let gk = g_shape[2];
            let un = u_shape[1];
            let uk = u_shape[2];
            let dn = d_shape[1];
            let dk = d_shape[2];
            if gn != inter || gk != h || un != inter || uk != h || dn != h || dk != inter {
                bail!(
                    "expert dims gate[{gn},{gk}] up[{un},{uk}] down[{dn},{dk}] \
                     vs inter={inter} h={h}"
                );
            }
            let g_slab = expert_slab(g_bytes, g_scheme, ne, gn, gk, e)?;
            let u_slab = expert_slab(u_bytes, u_scheme, ne, un, uk, e)?;
            let d_slab = expert_slab(d_bytes, d_scheme, ne, dn, dk, e)?;
            // Expert mats: Metal/MLX via accel, else fused host (never CPU F32 cache).
            let gate = packed_gemm(
                xt,
                g_slab,
                1,
                h,
                inter,
                g_scheme,
                accel.as_deref_mut(),
                true,
            )?;
            let up = packed_gemm(
                xt,
                u_slab,
                1,
                h,
                inter,
                u_scheme,
                accel.as_deref_mut(),
                true,
            )?;
            let mut mid = vec![0.0; inter];
            for i in 0..inter {
                mid[i] = silu(gate[i]) * up[i];
            }
            let down = packed_gemm(
                &mid,
                d_slab,
                1,
                inter,
                h,
                d_scheme,
                accel.as_deref_mut(),
                true,
            )?;
            for o in 0..h {
                out[t * h + o] += down[o] * rw * cfg.moe_routed_scaling_factor;
            }
        }
        for o in 0..h {
            out[t * h + o] += shared[t * h + o];
        }
    }
    Ok(out)
}

/// Per-layer K/V cache for packed incremental decode (repeated-head layout).
pub struct LayerKvCache {
    /// `[seq, n_heads, head_dim]` flattened.
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub n_heads: usize,
    pub head_dim: usize,
}

impl LayerKvCache {
    fn empty(n_heads: usize, head_dim: usize) -> Self {
        Self {
            k: Vec::new(),
            v: Vec::new(),
            n_heads,
            head_dim,
        }
    }

    fn len(&self) -> usize {
        let row = self.n_heads * self.head_dim;
        if row == 0 {
            0
        } else {
            self.k.len() / row
        }
    }
}

/// KV caches for all Laguna layers — enables O(layers) decode vs full recompute.
pub struct PackedKvCache {
    pub layers: Vec<LayerKvCache>,
    pub seq_len: usize,
}

impl PackedKvCache {
    pub fn new(cfg: &LagunaConfig) -> Self {
        let layers = (0..cfg.num_hidden_layers)
            .map(|i| LayerKvCache::empty(cfg.n_heads(i), cfg.head_dim))
            .collect();
        Self {
            layers,
            seq_len: 0,
        }
    }
}

fn attention(
    cfg: &LagunaConfig,
    loader: &GgufLoader,
    layer: &crate::packed::LagunaPackedLayer,
    layer_idx: usize,
    x: &[f32],
    seq: usize,
    pos_start: usize,
    kv: &mut LayerKvCache,
    mut accel: Option<&mut DeviceMatmul>,
) -> Result<Vec<f32>> {
    let h = cfg.hidden_size;
    let n_heads = cfg.n_heads(layer_idx);
    let n_kv = cfg.num_key_value_heads;
    let hd = cfg.head_dim;
    let groups = n_heads / n_kv;
    let q_dim = n_heads * hd;
    let kv_dim = n_kv * hd;
    let rope = cfg.rope_for_layer(layer_idx);
    let scale = (hd as f32).sqrt().recip() * rope.attention_factor.max(1e-6);
    let rot_dim = ((hd as f32) * rope.partial_rotary_factor).round() as usize;
    let rot_dim = rot_dim.max(2) & !1;

    debug_assert_eq!(kv.n_heads, n_heads);
    debug_assert_eq!(kv.head_dim, hd);
    // Decode appends one row; prefill replaces the cache for this layer.
    if pos_start == 0 {
        kv.k.clear();
        kv.v.clear();
    } else if kv.len() != pos_start {
        bail!(
            "KV len {} != pos_start {pos_start} (layer {layer_idx})",
            kv.len()
        );
    }

    let q = linear_mat(loader, &layer.wq, x, seq, q_dim, h, accel.as_deref_mut())?;
    let k = linear_mat(loader, &layer.wk, x, seq, kv_dim, h, accel.as_deref_mut())?;
    let v = linear_mat(loader, &layer.wv, x, seq, kv_dim, h, accel.as_deref_mut())?;

    let qn_w = layer
        .q_norm
        .as_deref()
        .ok_or_else(|| anyhow!("missing q_norm layer {layer_idx}"))?;
    let kn_w = layer
        .k_norm
        .as_deref()
        .ok_or_else(|| anyhow!("missing k_norm layer {layer_idx}"))?;
    let mut qn = vec![0.0; q.len()];
    let mut kn = vec![0.0; k.len()];
    for t in 0..seq {
        for head in 0..n_heads {
            let base = (t * n_heads + head) * hd;
            let row = &q[base..base + hd];
            let mean_sq = row.iter().map(|v| v * v).sum::<f32>() / hd as f32;
            let inv = 1.0 / (mean_sq + cfg.rms_norm_eps).sqrt();
            for j in 0..hd {
                qn[base + j] = row[j] * inv * qn_w[j];
            }
        }
        for head in 0..n_kv {
            let base = (t * n_kv + head) * hd;
            let row = &k[base..base + hd];
            let mean_sq = row.iter().map(|v| v * v).sum::<f32>() / hd as f32;
            let inv = 1.0 / (mean_sq + cfg.rms_norm_eps).sqrt();
            for j in 0..hd {
                kn[base + j] = row[j] * inv * kn_w[j];
            }
        }
    }

    let (cos, sin) = rotary_freqs(pos_start, seq, rot_dim, &rope_inv_freq(&rope, rot_dim));
    apply_rope_inplace(&mut qn, &cos, &sin, seq, n_heads, hd, rot_dim);
    apply_rope_inplace(&mut kn, &cos, &sin, seq, n_kv, hd, rot_dim);

    // Expand K/V to query heads and append into cache.
    let mut k_chunk = vec![0.0; seq * n_heads * hd];
    let mut v_chunk = vec![0.0; seq * n_heads * hd];
    for t in 0..seq {
        for hq in 0..n_heads {
            let hk = hq / groups;
            let dst = (t * n_heads + hq) * hd;
            let src_k = (t * n_kv + hk) * hd;
            let src_v = (t * n_kv + hk) * hd;
            k_chunk[dst..dst + hd].copy_from_slice(&kn[src_k..src_k + hd]);
            v_chunk[dst..dst + hd].copy_from_slice(&v[src_v..src_v + hd]);
        }
    }
    kv.k.extend_from_slice(&k_chunk);
    kv.v.extend_from_slice(&v_chunk);
    let cache_len = kv.len();
    debug_assert_eq!(cache_len, pos_start + seq);

    let window = if cfg.is_sliding(layer_idx) {
        cfg.sliding_window.max(1)
    } else {
        cache_len
    };

    let mut attn_out = vec![0.0; seq * q_dim];
    for hq in 0..n_heads {
        for ti in 0..seq {
            let tq = pos_start + ti;
            let mut scores = vec![0.0f32; cache_len];
            let qrow = &qn[(ti * n_heads + hq) * hd..(ti * n_heads + hq + 1) * hd];
            let t_min = tq.saturating_sub(window - 1);
            for tk in 0..=tq {
                if tk < t_min {
                    scores[tk] = f32::NEG_INFINITY;
                    continue;
                }
                let krow = &kv.k[(tk * n_heads + hq) * hd..(tk * n_heads + hq + 1) * hd];
                let mut dot = 0.0;
                for j in 0..hd {
                    dot += qrow[j] * krow[j];
                }
                scores[tk] = dot * scale;
            }
            for tk in (tq + 1)..cache_len {
                scores[tk] = f32::NEG_INFINITY;
            }
            softmax_row(&mut scores);
            let out_base = (ti * n_heads + hq) * hd;
            for tk in t_min..=tq {
                let vrow = &kv.v[(tk * n_heads + hq) * hd..(tk * n_heads + hq + 1) * hd];
                let a = scores[tk];
                for j in 0..hd {
                    attn_out[out_base + j] += a * vrow[j];
                }
            }
        }
    }

    if cfg.gating != AttnGating::Off {
        let wg = layer
            .wg
            .as_ref()
            .ok_or_else(|| anyhow!("missing attn gate layer {layer_idx}"))?;
        let gate_out = match cfg.gating {
            AttnGating::PerHead => n_heads,
            AttnGating::PerElement => q_dim,
            AttnGating::Off => unreachable!(),
        };
        let g = linear_mat(loader, wg, x, seq, gate_out, h, accel.as_deref_mut())?;
        match cfg.gating {
            AttnGating::PerHead => {
                for t in 0..seq {
                    for hq in 0..n_heads {
                        let s = softplus(g[t * n_heads + hq]);
                        let base = (t * n_heads + hq) * hd;
                        for j in 0..hd {
                            attn_out[base + j] *= s;
                        }
                    }
                }
            }
            AttnGating::PerElement => {
                for i in 0..attn_out.len() {
                    attn_out[i] *= softplus(g[i]);
                }
            }
            AttnGating::Off => {}
        }
    }

    linear_mat(
        loader,
        &layer.wo,
        &attn_out,
        seq,
        h,
        q_dim,
        accel.as_deref_mut(),
    )
}

fn forward_hidden_with_cache(
    cfg: &LagunaConfig,
    weights: &LagunaPackedWeights,
    loader: &GgufLoader,
    prompt_ids: &[u32],
    pos_start: usize,
    cache: &mut PackedKvCache,
    mut accel: Option<&mut DeviceMatmul>,
) -> Result<Vec<f32>> {
    if prompt_ids.is_empty() {
        bail!("empty prompt");
    }
    let h = cfg.hidden_size;
    let seq = prompt_ids.len();
    let mut x = gather_embed(loader, &weights.token_embd, prompt_ids, h, cfg.vocab_size)?;

    for (layer_idx, layer) in weights.layers.iter().enumerate() {
        let residual = x.clone();
        let normed = rms_norm(&x, &layer.attn_norm, cfg.rms_norm_eps);
        let attn = attention(
            cfg,
            loader,
            layer,
            layer_idx,
            &normed,
            seq,
            pos_start,
            &mut cache.layers[layer_idx],
            accel.as_deref_mut(),
        )?;
        for i in 0..x.len() {
            x[i] = residual[i] + attn[i];
        }
        let residual = x.clone();
        let normed = rms_norm(&x, &layer.ffn_norm, cfg.rms_norm_eps);
        let ffn = match &layer.ffn {
            LagunaPackedFfn::Dense { gate, up, down } => dense_mlp_mat(
                loader,
                &normed,
                gate,
                up,
                down,
                seq,
                h,
                cfg.intermediate_size,
                accel.as_deref_mut(),
            )?,
            LagunaPackedFfn::Moe { .. } => {
                moe_mlp(cfg, loader, &normed, &layer.ffn, seq, accel.as_deref_mut())?
            }
        };
        for i in 0..x.len() {
            x[i] = residual[i] + ffn[i];
        }
    }

    cache.seq_len = pos_start + seq;
    let last = &x[(seq - 1) * h..seq * h];
    Ok(rms_norm(last, &weights.output_norm, cfg.rms_norm_eps))
}

/// Last-token hidden via full prefill (builds / resets [`PackedKvCache`]).
pub fn forward_hidden(
    cfg: &LagunaConfig,
    weights: &LagunaPackedWeights,
    loader: &GgufLoader,
    prompt_ids: &[u32],
    accel: Option<&mut DeviceMatmul>,
) -> Result<Vec<f32>> {
    let mut cache = PackedKvCache::new(cfg);
    forward_hidden_with_cache(cfg, weights, loader, prompt_ids, 0, &mut cache, accel)
}

fn lm_argmax_hidden(
    cfg: &LagunaConfig,
    weights: &LagunaPackedWeights,
    loader: &GgufLoader,
    hidden: &[f32],
) -> Result<u32> {
    let h = cfg.hidden_size;
    let (key, scheme, n_vocab, n_embd) = if let Some(out) = weights.output.as_ref() {
        match out {
            MatWeight::Packed { key, scheme, shape } => {
                (key.as_str(), *scheme, shape[0], shape[1])
            }
            MatWeight::F32(w) => {
                let n_vocab = w.len() / h;
                let (idx, _) = rlx_cpu::lm_head::f32_tied_lm_argmax(hidden, w, h, n_vocab);
                return Ok(idx);
            }
        }
    } else {
        match &weights.token_embd {
            MatWeight::Packed { key, scheme, shape } => {
                (key.as_str(), *scheme, shape[0], shape[1])
            }
            MatWeight::F32(w) => {
                let n_vocab = w.len() / h;
                let (idx, _) = rlx_cpu::lm_head::f32_tied_lm_argmax(hidden, w, h, n_vocab);
                return Ok(idx);
            }
        }
    };
    if n_embd != h {
        bail!("lm_head embd {n_embd} != hidden {h}");
    }
    let bytes = loader
        .tensor_bytes_borrowed(key)
        .ok_or_else(|| anyhow!("lm_head bytes missing for {key}"))?;
    let (idx, _) = gguf_tied_lm_argmax(hidden, bytes, h, n_vocab, scheme);
    Ok(idx)
}

pub fn greedy_next(
    cfg: &LagunaConfig,
    weights: &LagunaPackedWeights,
    loader: &GgufLoader,
    prompt_ids: &[u32],
    accel: Option<&mut DeviceMatmul>,
) -> Result<u32> {
    let hidden = forward_hidden(cfg, weights, loader, prompt_ids, accel)?;
    lm_argmax_hidden(cfg, weights, loader, &hidden)
}

/// Greedy generate with KV cache: one prefill, then per-token decode steps.
pub fn generate(
    cfg: &LagunaConfig,
    weights: &LagunaPackedWeights,
    loader: &GgufLoader,
    prompt_ids: &[u32],
    n_new: usize,
    mut on_token: impl FnMut(u32),
    mut accel: Option<&mut DeviceMatmul>,
) -> Result<Vec<u32>> {
    if prompt_ids.is_empty() {
        bail!("empty prompt");
    }
    if n_new == 0 {
        return Ok(prompt_ids.to_vec());
    }

    let mut cache = PackedKvCache::new(cfg);
    let hidden = forward_hidden_with_cache(
        cfg,
        weights,
        loader,
        prompt_ids,
        0,
        &mut cache,
        accel.as_deref_mut(),
    )?;
    let mut ids = prompt_ids.to_vec();
    let mut next = lm_argmax_hidden(cfg, weights, loader, &hidden)?;
    ids.push(next);
    on_token(next);
    if next == cfg.eos_token_id {
        return Ok(ids);
    }

    for _ in 1..n_new {
        let pos = cache.seq_len;
        let hidden = forward_hidden_with_cache(
            cfg,
            weights,
            loader,
            &[next],
            pos,
            &mut cache,
            accel.as_deref_mut(),
        )?;
        next = lm_argmax_hidden(cfg, weights, loader, &hidden)?;
        ids.push(next);
        on_token(next);
        if next == cfg.eos_token_id {
            break;
        }
    }
    Ok(ids)
}
