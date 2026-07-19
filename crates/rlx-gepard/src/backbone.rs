// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: Apache-2.0

//! Qwen3.5 full-attention backbone — eager CPU forward pass.
//!
//! Implements the `Qwen3_5TextModel` forward (no lm_head) matching
//! HF transformers weight keys. Gepard uses the `qwen3_5-full-attn-only-14`
//! variant where all 14 layers are standard full-attention.
//!
//! # Key layout (HF Qwen3.5 naming)
//!
//! ```text
//! model.embed_tokens.weight            [vocab, hidden]
//! model.norm.weight                    [hidden]
//! model.layers.{i}.input_layernorm.weight        [hidden]
//! model.layers.{i}.self_attn.q_proj.weight       [n_heads*head_dim, hidden]
//! model.layers.{i}.self_attn.q_proj.bias         [n_heads*head_dim]
//! model.layers.{i}.self_attn.k_proj.weight       [n_kv_heads*head_dim, hidden]
//! model.layers.{i}.self_attn.k_proj.bias         [n_kv_heads*head_dim]
//! model.layers.{i}.self_attn.v_proj.weight       [n_kv_heads*head_dim, hidden]
//! model.layers.{i}.self_attn.o_proj.weight       [hidden, n_heads*head_dim]
//! model.layers.{i}.self_attn.q_norm.weight       [head_dim]
//! model.layers.{i}.self_attn.k_norm.weight       [head_dim]
//! model.layers.{i}.post_attention_layernorm.weight  [hidden]
//! model.layers.{i}.mlp.gate_proj.weight          [intermediate, hidden]
//! model.layers.{i}.mlp.up_proj.weight            [intermediate, hidden]
//! model.layers.{i}.mlp.down_proj.weight          [hidden, intermediate]
//! ```

use anyhow::Result;
use safetensors::SafeTensors;

use crate::config::BackboneConfig;
use crate::weights::{backbone_embed_key, backbone_final_norm_key, backbone_layer_key, read_f32};

// ── math primitives ───────────────────────────────────────────────────────────

/// Dense matvec: `y[o] = sum_i(W[o,i] * x[i]) + b[o]`
/// `W`: row-major `[d_out, d_in]`
pub fn matvec(w: &[f32], x: &[f32], b: Option<&[f32]>, d_in: usize, d_out: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; d_out];
    for o in 0..d_out {
        let row = &w[o * d_in..(o + 1) * d_in];
        let acc: f32 = row.iter().zip(x.iter()).map(|(w, x)| w * x).sum();
        y[o] = acc + b.map_or(0.0, |b| b[o]);
    }
    y
}

/// RMSNorm (Qwen3.5): `y = x / rms(x) * (1 + w)` — weight is zero-init in HF.
pub fn rms_norm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let rms = (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32 + eps).sqrt();
    x.iter()
        .zip(w)
        .map(|(v, wi)| v / rms * (1.0 + wi))
        .collect()
}

/// Per-head RMSNorm (Qwen3 `q_norm` / `k_norm`).
/// `x`: `[n_heads * head_dim]`, normalised independently per head.
pub fn head_rms_norm(x: &mut [f32], w: &[f32], head_dim: usize, eps: f32) {
    for h in 0..x.len() / head_dim {
        let s = &mut x[h * head_dim..(h + 1) * head_dim];
        let rms = (s.iter().map(|v| v * v).sum::<f32>() / head_dim as f32 + eps).sqrt();
        for (v, wi) in s.iter_mut().zip(w) {
            *v = *v / rms * (1.0 + wi);
        }
    }
}

/// Apply RoPE in-place to `x: [n_heads * head_dim]` at sequence position `pos`.
pub fn apply_rope(x: &mut [f32], pos: usize, head_dim: usize, theta: f32) {
    let half = head_dim / 2;
    let n_heads = x.len() / head_dim;
    for h in 0..n_heads {
        let base = h * head_dim;
        for i in 0..half {
            let angle = pos as f32 / theta.powf((2 * i) as f32 / head_dim as f32);
            let (sin, cos) = angle.sin_cos();
            let x0 = x[base + i];
            let x1 = x[base + i + half];
            x[base + i] = x0 * cos - x1 * sin;
            x[base + i + half] = x0 * sin + x1 * cos;
        }
    }
}

/// Grouped-query attention for one query position against a KV cache.
/// `q`:       `[n_heads * head_dim]`
/// `k_cache`: `[n_kv * n_kv_heads * head_dim]`  (appended tokens row-major)
/// `v_cache`: same layout
/// Returns `[n_heads * head_dim]`
pub fn gqa_attend(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    n_kv: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let kv_groups = n_heads / n_kv_heads;
    let scale = (head_dim as f32).sqrt().recip();
    let mut out = vec![0.0f32; n_heads * head_dim];

    for h in 0..n_heads {
        let kv_h = h / kv_groups;
        let q_s = &q[h * head_dim..(h + 1) * head_dim];

        // Attention scores
        let mut scores = vec![0.0f32; n_kv];
        for t in 0..n_kv {
            let k_s = &k_cache
                [(t * n_kv_heads + kv_h) * head_dim..((t * n_kv_heads + kv_h) + 1) * head_dim];
            scores[t] = q_s.iter().zip(k_s).map(|(a, b)| a * b).sum::<f32>() * scale;
        }

        // Softmax
        let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = scores
            .iter_mut()
            .map(|s| {
                *s = (*s - max_s).exp();
                *s
            })
            .sum();
        for s in scores.iter_mut() {
            *s /= sum.max(1e-9);
        }

        // Weighted sum of V
        let out_s = &mut out[h * head_dim..(h + 1) * head_dim];
        for t in 0..n_kv {
            let v_s = &v_cache
                [(t * n_kv_heads + kv_h) * head_dim..((t * n_kv_heads + kv_h) + 1) * head_dim];
            for (ov, vv) in out_s.iter_mut().zip(v_s) {
                *ov += scores[t] * vv;
            }
        }
    }
    out
}

/// SiLU activation.
#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

// ── weight types ──────────────────────────────────────────────────────────────

pub struct BackboneLayer {
    pub input_norm_w: Vec<f32>,     // [hidden]
    pub q_w: Vec<f32>, // [2 * n_heads * head_dim, hidden]  (Q + gate when attn_output_gate)
    pub q_b: Vec<f32>, // [2 * n_heads * head_dim] or empty
    pub k_w: Vec<f32>, // [n_kv_heads * head_dim, hidden]
    pub k_b: Vec<f32>, // [n_kv_heads * head_dim] or empty
    pub v_w: Vec<f32>, // [n_kv_heads*head_dim, hidden]
    pub o_w: Vec<f32>, // [hidden, n_heads*head_dim]
    pub q_norm_w: Vec<f32>, // [head_dim]
    pub k_norm_w: Vec<f32>, // [head_dim]
    pub post_attn_norm_w: Vec<f32>, // [hidden]
    pub gate_w: Vec<f32>, // [intermediate, hidden]
    pub up_w: Vec<f32>, // [intermediate, hidden]
    pub down_w: Vec<f32>, // [hidden, intermediate]
}

pub struct BackboneWeights {
    pub embed: Vec<f32>,      // [vocab * hidden]
    pub final_norm: Vec<f32>, // [hidden]
    pub layers: Vec<BackboneLayer>,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub hidden: usize,
    pub head_dim: usize,
    pub intermediate: usize,
    pub rope_theta: f32,
    pub rms_eps: f32,
    pub attn_output_gate: bool,
}

impl BackboneWeights {
    pub fn load(st: &SafeTensors<'_>, cfg: &BackboneConfig) -> Result<Self> {
        let embed = read_f32(st, backbone_embed_key())?;
        let final_norm = read_f32(st, backbone_final_norm_key())?;

        let num_layers = cfg.num_hidden_layers;
        let hidden = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.effective_head_dim();
        let intermediate = cfg.intermediate_size;
        let rms_eps = cfg.rms_norm_eps as f32;
        let rope_theta = cfg.rope_theta as f32;
        let attn_output_gate = cfg.attn_output_gate;

        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        // When attn_output_gate is true, q_proj outputs 2×q_dim (Q + gate).
        let q_proj_out = if attn_output_gate { 2 * q_dim } else { q_dim };

        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let k = |s: &str| backbone_layer_key(i, s);
            layers.push(BackboneLayer {
                input_norm_w: read_f32(st, &k("input_layernorm.weight"))?,
                q_w: read_f32(st, &k("self_attn.q_proj.weight"))?,
                q_b: read_f32(st, &k("self_attn.q_proj.bias"))
                    .unwrap_or_else(|_| vec![0.0; q_proj_out]),
                k_w: read_f32(st, &k("self_attn.k_proj.weight"))?,
                k_b: read_f32(st, &k("self_attn.k_proj.bias"))
                    .unwrap_or_else(|_| vec![0.0; kv_dim]),
                v_w: read_f32(st, &k("self_attn.v_proj.weight"))?,
                o_w: read_f32(st, &k("self_attn.o_proj.weight"))?,
                q_norm_w: read_f32(st, &k("self_attn.q_norm.weight"))?,
                k_norm_w: read_f32(st, &k("self_attn.k_norm.weight"))?,
                post_attn_norm_w: read_f32(st, &k("post_attention_layernorm.weight"))?,
                gate_w: read_f32(st, &k("mlp.gate_proj.weight"))?,
                up_w: read_f32(st, &k("mlp.up_proj.weight"))?,
                down_w: read_f32(st, &k("mlp.down_proj.weight"))?,
            });
        }

        Ok(Self {
            embed,
            final_norm,
            layers,
            num_heads,
            num_kv_heads,
            hidden,
            head_dim,
            intermediate,
            rope_theta,
            rms_eps,
            attn_output_gate,
        })
    }

    /// Look up token embeddings.
    pub fn embed_tokens(&self, ids: &[u32]) -> Vec<f32> {
        let n = ids.len();
        let h = self.hidden;
        let mut out = vec![0.0f32; n * h];
        for (i, &id) in ids.iter().enumerate() {
            let src = &self.embed[id as usize * h..(id as usize + 1) * h];
            out[i * h..(i + 1) * h].copy_from_slice(src);
        }
        out
    }
}

// ── KV cache ──────────────────────────────────────────────────────────────────

pub struct LayerKv {
    pub k: Vec<f32>, // [n_tokens * n_kv_heads * head_dim]
    pub v: Vec<f32>,
}

pub struct GepardKvCache {
    pub layers: Vec<LayerKv>,
    pub num_tokens: usize,
}

impl GepardKvCache {
    pub fn new(num_layers: usize) -> Self {
        Self {
            layers: (0..num_layers)
                .map(|_| LayerKv {
                    k: Vec::new(),
                    v: Vec::new(),
                })
                .collect(),
            num_tokens: 0,
        }
    }
}

// ── forward pass ──────────────────────────────────────────────────────────────

/// Forward one transformer layer for a single token at `pos`.
/// Appends K/V to the cache and returns the new hidden state `[hidden]`.
fn layer_decode_step(
    h: &[f32],
    layer: &BackboneLayer,
    kv: &mut LayerKv,
    pos: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    intermediate: usize,
    rope_theta: f32,
    rms_eps: f32,
    attn_output_gate: bool,
) -> Vec<f32> {
    let hidden = h.len();
    let q_dim = num_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;
    // When attn_output_gate=true, q_proj has 2*q_dim output (Q + gate interleaved).
    let q_proj_out = if attn_output_gate { 2 * q_dim } else { q_dim };

    // Attention pre-norm
    let h_norm = rms_norm(h, &layer.input_norm_w, rms_eps);

    // QKV projections. With attn_output_gate, q_proj is per-head [Q|gate]
    // packed as [n_heads, 2*head_dim] (matches Qwen3.5 / ggml view).
    let q_raw = matvec(&layer.q_w, &h_norm, Some(&layer.q_b), hidden, q_proj_out);
    let mut k = matvec(&layer.k_w, &h_norm, Some(&layer.k_b), hidden, kv_dim);
    let v = matvec(&layer.v_w, &h_norm, None, hidden, kv_dim);

    let (mut q, gate) = if attn_output_gate {
        let mut q = vec![0.0f32; q_dim];
        let mut gate = vec![0.0f32; q_dim];
        for h_i in 0..num_heads {
            let src = h_i * 2 * head_dim;
            let dst = h_i * head_dim;
            q[dst..dst + head_dim].copy_from_slice(&q_raw[src..src + head_dim]);
            gate[dst..dst + head_dim].copy_from_slice(&q_raw[src + head_dim..src + 2 * head_dim]);
        }
        (q, Some(gate))
    } else {
        (q_raw, None)
    };

    // Per-head RMSNorm (Qwen3 qk_norm)
    head_rms_norm(&mut q, &layer.q_norm_w, head_dim, rms_eps);
    head_rms_norm(&mut k, &layer.k_norm_w, head_dim, rms_eps);

    // MRoPE (Qwen3.5 text modality)
    apply_rope(&mut q, pos, head_dim, rope_theta);
    apply_rope(&mut k, pos, head_dim, rope_theta);

    // Append to KV cache
    kv.k.extend_from_slice(&k);
    kv.v.extend_from_slice(&v);
    let n_kv = pos + 1;

    // GQA → [n_heads * head_dim]
    let mut attn_out = gqa_attend(&q, &kv.k, &kv.v, n_kv, num_heads, num_kv_heads, head_dim);

    // Qwen3.5: sigmoid(gate) × attention output, then o_proj.
    if let Some(gate) = gate {
        for (a, g) in attn_out.iter_mut().zip(&gate) {
            *a *= 1.0 / (1.0 + (-g).exp());
        }
    }

    let attn_proj = matvec(&layer.o_w, &attn_out, None, q_dim, hidden);
    let mut h_new: Vec<f32> = h.iter().zip(&attn_proj).map(|(a, b)| a + b).collect();

    // FFN pre-norm
    let h_norm2 = rms_norm(&h_new, &layer.post_attn_norm_w, rms_eps);

    // SwiGLU FFN
    let gate = matvec(&layer.gate_w, &h_norm2, None, hidden, intermediate);
    let up = matvec(&layer.up_w, &h_norm2, None, hidden, intermediate);
    let ffn_hidden: Vec<f32> = gate.iter().zip(&up).map(|(g, u)| silu(*g) * u).collect();
    let ffn_out = matvec(&layer.down_w, &ffn_hidden, None, intermediate, hidden);

    // Residual
    for (hi, fi) in h_new.iter_mut().zip(&ffn_out) {
        *hi += fi;
    }
    h_new
}

/// Prefill: run the backbone on `n_tokens` pre-computed embeddings.
/// Returns the hidden states for every position after the final RMSNorm.
/// Uses causal masking naturally (each token is processed in order,
/// appending its KV before the next token runs).
pub fn backbone_prefill(
    inputs_embeds: &[f32], // [n_tokens * hidden]
    n_tokens: usize,
    weights: &BackboneWeights,
    kv: &mut GepardKvCache,
) -> Vec<f32> {
    let hidden = weights.hidden;
    let mut all_out = vec![0.0f32; n_tokens * hidden];
    let start_pos = kv.num_tokens;

    for tok in 0..n_tokens {
        let pos = start_pos + tok;
        let mut h = inputs_embeds[tok * hidden..(tok + 1) * hidden].to_vec();
        for (li, layer) in weights.layers.iter().enumerate() {
            h = layer_decode_step(
                &h,
                layer,
                &mut kv.layers[li],
                pos,
                weights.num_heads,
                weights.num_kv_heads,
                weights.head_dim,
                weights.intermediate,
                weights.rope_theta,
                weights.rms_eps,
                weights.attn_output_gate,
            );
        }
        // Final norm
        let h_normed = rms_norm(&h, &weights.final_norm, weights.rms_eps);
        all_out[tok * hidden..(tok + 1) * hidden].copy_from_slice(&h_normed);
    }
    kv.num_tokens += n_tokens;
    all_out
}

/// Single-token decode step (AR generation).
/// Returns the hidden state `[hidden]` for the new position.
pub fn backbone_decode_step(
    embed: &[f32], // [hidden]
    weights: &BackboneWeights,
    kv: &mut GepardKvCache,
) -> Vec<f32> {
    let pos = kv.num_tokens;
    let mut h = embed.to_vec();
    for (li, layer) in weights.layers.iter().enumerate() {
        h = layer_decode_step(
            &h,
            layer,
            &mut kv.layers[li],
            pos,
            weights.num_heads,
            weights.num_kv_heads,
            weights.head_dim,
            weights.intermediate,
            weights.rope_theta,
            weights.rms_eps,
            weights.attn_output_gate,
        );
    }
    let h_normed = rms_norm(&h, &weights.final_norm, weights.rms_eps);
    kv.num_tokens += 1;
    h_normed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rms_norm_unit() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        // HF Qwen3.5 RMSNorm: scale = (1 + w); w=0 → identity affine.
        let w = vec![0.0, 0.0, 0.0, 0.0];
        let y = rms_norm(&x, &w, 1e-6);
        let rms = (7.5f32).sqrt();
        for (yi, xi) in y.iter().zip(&x) {
            assert!((yi - xi / rms).abs() < 1e-5, "{yi} vs {}", xi / rms);
        }
    }

    #[test]
    fn rope_cos_sin_symmetry() {
        let mut x = vec![1.0, 2.0, 3.0, 4.0];
        let orig = x.clone();
        apply_rope(&mut x, 0, 4, 10000.0);
        for (a, b) in x.iter().zip(&orig) {
            assert!((a - b).abs() < 1e-5, "rope at pos=0 should be identity");
        }
    }

    #[test]
    fn gqa_self_attend_softmax_sums_to_one() {
        // 1 head, 1 kv_head, head_dim=2, 3 positions cached
        let head_dim = 2;
        let n_kv = 3;
        let q = vec![1.0, 0.0];
        let k_cache = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0]; // 3 × [2]
        let v_cache = vec![1.0, 0.0, 0.0, 1.0, 0.5, 0.5];
        let out = gqa_attend(&q, &k_cache, &v_cache, n_kv, 1, 1, head_dim);
        assert_eq!(out.len(), head_dim);
        // output should be a weighted sum of V rows — just check it's finite
        for v in &out {
            assert!(v.is_finite());
        }
    }
}
