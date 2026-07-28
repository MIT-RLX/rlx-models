// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: Apache-2.0

//! Q-Former voice-cloning compressor — translates reference codec codes
//! into K speaker-prefix tokens.
//!
//! Mirrors `gepard/model/ref_compressor.py::RefCompressor`.
//!
//! # Key layout (state_dict prefix `ref_compressor.`)
//!
//! ```text
//! input_proj.weight                [d_model, c_total]
//! input_proj.bias                  [d_model]
//! queries                          [K, d_model]          (nn.Parameter)
//! blocks.{i}.norm_self.weight      [d_model]
//! blocks.{i}.self_attn.q_proj.weight   [d_model, d_model]
//! blocks.{i}.self_attn.k_proj.weight   [d_model, d_model]
//! blocks.{i}.self_attn.v_proj.weight   [d_model, d_model]
//! blocks.{i}.self_attn.out_proj.weight [d_model, d_model]
//! blocks.{i}.norm_cross.weight     [d_model]
//! blocks.{i}.cross_attn.q_proj.weight  [d_model, d_model]
//! blocks.{i}.cross_attn.k_proj.weight  [d_model, d_model]
//! blocks.{i}.cross_attn.v_proj.weight  [d_model, d_model]
//! blocks.{i}.cross_attn.out_proj.weight [d_model, d_model]
//! blocks.{i}.norm_ffn.weight       [d_model]
//! blocks.{i}.ffn.gate_proj.weight  [ffn_hidden, d_model]
//! blocks.{i}.ffn.up_proj.weight    [ffn_hidden, d_model]
//! blocks.{i}.ffn.down_proj.weight  [d_model, ffn_hidden]
//! final_norm.weight                [d_model]
//! output_scale                     [1]
//! ```

use anyhow::Result;
use safetensors::SafeTensors;

use crate::backbone::{rms_norm, silu};
use crate::codec_ops::{NUM_CHANNELS, dequantize_frame};
use crate::weights::read_f32;

// ── Q-Former weight types ─────────────────────────────────────────────────────

pub struct QFormerAttnWeights {
    pub q_proj: Vec<f32>,   // [d, d]
    pub k_proj: Vec<f32>,   // [d, d]
    pub v_proj: Vec<f32>,   // [d, d]
    pub out_proj: Vec<f32>, // [d, d]
}

pub struct QFormerBlockWeights {
    pub norm_self: Vec<f32>,
    pub self_attn: QFormerAttnWeights,
    pub norm_cross: Vec<f32>,
    pub cross_attn: QFormerAttnWeights,
    pub norm_ffn: Vec<f32>,
    pub gate_w: Vec<f32>,
    pub up_w: Vec<f32>,
    pub down_w: Vec<f32>,
}

pub struct QFormerWeights {
    pub input_proj_w: Vec<f32>, // [d_model, c_total]
    pub input_proj_b: Vec<f32>, // [d_model]
    pub queries: Vec<f32>,      // [K * d_model]
    pub blocks: Vec<QFormerBlockWeights>,
    pub final_norm_w: Vec<f32>, // [d_model]
    pub output_scale: f32,
    pub num_queries: usize,
    pub num_heads: usize,
    pub d_model: usize,
    pub c_total: usize,
    pub ffn_hidden: usize,
}

impl QFormerWeights {
    pub fn load(
        st: &SafeTensors<'_>,
        num_queries: usize,
        num_blocks: usize,
        num_heads: usize,
        d_model: usize,
        c_total: usize,
        ffn_hidden_multiplier: usize,
    ) -> Result<Self> {
        let p = "ref_compressor";
        let ffn_hidden = d_model * ffn_hidden_multiplier;

        let kf = |s: &str| format!("{p}.{s}");
        let ka = |b: usize, sub: &str, k: &str| format!("{p}.blocks.{b}.{sub}.{k}");

        let input_proj_w = read_f32(st, &kf("input_proj.weight"))?;
        let input_proj_b = read_f32(st, &kf("input_proj.bias"))?;
        let queries = read_f32(st, &kf("queries"))?;
        let final_norm_w = read_f32(st, &kf("final_norm.weight"))?;
        let output_scale = read_f32(st, &kf("output_scale"))
            .map(|v| v[0])
            .unwrap_or(1.0 / (d_model as f32).sqrt());

        let mut blocks = Vec::with_capacity(num_blocks);
        for b in 0..num_blocks {
            let load_attn = |sub: &str| -> Result<QFormerAttnWeights> {
                Ok(QFormerAttnWeights {
                    q_proj: read_f32(st, &ka(b, sub, "q_proj.weight"))?,
                    k_proj: read_f32(st, &ka(b, sub, "k_proj.weight"))?,
                    v_proj: read_f32(st, &ka(b, sub, "v_proj.weight"))?,
                    out_proj: read_f32(st, &ka(b, sub, "out_proj.weight"))?,
                })
            };

            blocks.push(QFormerBlockWeights {
                norm_self: read_f32(st, &format!("{p}.blocks.{b}.norm_self.weight"))?,
                self_attn: load_attn("self_attn")?,
                norm_cross: read_f32(st, &format!("{p}.blocks.{b}.norm_cross.weight"))?,
                cross_attn: load_attn("cross_attn")?,
                norm_ffn: read_f32(st, &format!("{p}.blocks.{b}.norm_ffn.weight"))?,
                gate_w: read_f32(st, &format!("{p}.blocks.{b}.ffn.gate_proj.weight"))?,
                up_w: read_f32(st, &format!("{p}.blocks.{b}.ffn.up_proj.weight"))?,
                down_w: read_f32(st, &format!("{p}.blocks.{b}.ffn.down_proj.weight"))?,
            });
        }

        Ok(Self {
            input_proj_w,
            input_proj_b,
            queries,
            blocks,
            final_norm_w,
            output_scale,
            num_queries,
            num_heads,
            d_model,
            c_total,
            ffn_hidden,
        })
    }

    /// Expected key names (for validation).
    pub fn expected_keys(num_blocks: usize) -> Vec<String> {
        let p = "ref_compressor";
        let mut keys = vec![
            format!("{p}.input_proj.weight"),
            format!("{p}.input_proj.bias"),
            format!("{p}.queries"),
        ];
        let attn_kinds = ["self_attn", "cross_attn"];
        let attn_parts = [
            "q_proj.weight",
            "k_proj.weight",
            "v_proj.weight",
            "out_proj.weight",
        ];
        let norm_kinds = ["norm_self", "norm_cross", "norm_ffn"];
        for b in 0..num_blocks {
            for nk in &norm_kinds {
                keys.push(format!("{p}.blocks.{b}.{nk}.weight"));
            }
            for ak in &attn_kinds {
                for ap in &attn_parts {
                    keys.push(format!("{p}.blocks.{b}.{ak}.{ap}"));
                }
            }
            for part in &["gate_proj.weight", "up_proj.weight", "down_proj.weight"] {
                keys.push(format!("{p}.blocks.{b}.ffn.{part}"));
            }
        }
        keys.push(format!("{p}.final_norm.weight"));
        keys.push(format!("{p}.output_scale"));
        keys
    }
}

// ── sinusoidal PE ─────────────────────────────────────────────────────────────

/// Compute sinusoidal positional encodings for `t` positions.
/// Returns `[t, d_model]` row-major.
fn sinusoidal_pe(t: usize, d_model: usize) -> Vec<f32> {
    let mut pe = vec![0.0f32; t * d_model];
    for pos in 0..t {
        for i in 0..d_model / 2 {
            let angle = pos as f32 / 10000.0f32.powf(2.0 * i as f32 / d_model as f32);
            pe[pos * d_model + 2 * i] = angle.sin();
            pe[pos * d_model + 2 * i + 1] = angle.cos();
        }
    }
    pe
}

// ── multi-head attention (bidirectional) ──────────────────────────────────────

/// Bidirectional multi-head attention (no causal mask).
/// `q_in`: `[tq * d]`, `kv_in`: `[tkv * d]`
/// Returns `[tq * d]`
fn mha_forward(
    q_in: &[f32],
    kv_in: &[f32],
    tq: usize,
    tkv: usize,
    attn: &QFormerAttnWeights,
    num_heads: usize,
    d: usize,
) -> Vec<f32> {
    let head_dim = d / num_heads;
    let scale = (head_dim as f32).sqrt().recip();

    // Project Q, K, V
    let q = batch_matvec(&attn.q_proj, q_in, tq, d, d);
    let k = batch_matvec(&attn.k_proj, kv_in, tkv, d, d);
    let v = batch_matvec(&attn.v_proj, kv_in, tkv, d, d);

    let mut out = vec![0.0f32; tq * d];

    for h in 0..num_heads {
        let qs = h * head_dim;
        let qe = qs + head_dim;

        // For each query position, compute softmax attention over all KV positions
        for tq_i in 0..tq {
            let q_h = &q[tq_i * d + qs..tq_i * d + qe];

            let mut scores = vec![0.0f32; tkv];
            for tkv_i in 0..tkv {
                let k_h = &k[tkv_i * d + qs..tkv_i * d + qe];
                scores[tkv_i] = q_h.iter().zip(k_h).map(|(a, b)| a * b).sum::<f32>() * scale;
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

            // Weighted V
            let out_base = tq_i * d + qs;
            for tkv_i in 0..tkv {
                let v_h = &v[tkv_i * d + qs..tkv_i * d + qe];
                for (o, &vv) in out[out_base..out_base + head_dim].iter_mut().zip(v_h) {
                    *o += scores[tkv_i] * vv;
                }
            }
        }
    }

    // Output projection
    batch_matvec(&attn.out_proj, &out, tq, d, d)
}

/// Row-wise matvec: `Y[i,:] = W * X[i,:] + b`
fn batch_matvec(w: &[f32], x: &[f32], n: usize, d_in: usize, d_out: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; n * d_out];
    for i in 0..n {
        let y_row = &mut y[i * d_out..(i + 1) * d_out];
        let x_row = &x[i * d_in..(i + 1) * d_in];
        for o in 0..d_out {
            y_row[o] = w[o * d_in..(o + 1) * d_in]
                .iter()
                .zip(x_row)
                .map(|(a, b)| a * b)
                .sum();
        }
    }
    y
}

/// Row-wise RMSNorm.
fn batch_rms_norm(x: &[f32], w: &[f32], n: usize, d: usize, eps: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; n * d];
    for i in 0..n {
        let row = &x[i * d..(i + 1) * d];
        let normed = rms_norm(row, w, eps);
        out[i * d..(i + 1) * d].copy_from_slice(&normed);
    }
    out
}

// ── Q-Former forward ──────────────────────────────────────────────────────────

/// Compute speaker prefix from reference codec codes.
///
/// - `ref_codes`: `[t_ref * NUM_CHANNELS]` integer codes (already unfolded, 32 ch)
/// - `t_ref`: number of reference frames
///
/// Returns `[K * d_model]` prefix tokens.
pub fn qformer_forward(
    ref_codes: &[u32],
    t_ref: usize,
    w: &QFormerWeights,
    rms_eps: f32,
) -> Vec<f32> {
    debug_assert_eq!(ref_codes.len(), t_ref * NUM_CHANNELS);

    let d = w.d_model;
    let nq = w.num_queries;

    // 1. Dequantize to [-1, 1] floats: [t_ref, NUM_CHANNELS] → Vec<f32>
    let mut ref_f = Vec::with_capacity(t_ref * NUM_CHANNELS);
    for frame in 0..t_ref {
        let codes = &ref_codes[frame * NUM_CHANNELS..(frame + 1) * NUM_CHANNELS];
        ref_f.extend(dequantize_frame(codes));
    }

    // 2. Input projection: [t_ref, c_total] → [t_ref, d]
    let ref_proj = batch_matvec(&w.input_proj_w, &ref_f, t_ref, w.c_total, d);
    // Add bias
    let mut ref_proj: Vec<f32> = ref_proj
        .iter()
        .enumerate()
        .map(|(i, &v)| v + w.input_proj_b[i % d])
        .collect();

    // 3. Add sinusoidal PE
    let pe = sinusoidal_pe(t_ref, d);
    for (v, p) in ref_proj.iter_mut().zip(&pe) {
        *v += p;
    }
    let ref_feats = ref_proj; // [t_ref * d]

    // 4. Expand learnable queries: [nq * d]
    let mut q = w.queries.clone(); // [nq * d]

    // 5. Q-Former blocks
    for block in &w.blocks {
        // Self-attention on queries
        let q_norm = batch_rms_norm(&q, &block.norm_self, nq, d, rms_eps);
        let q_self = mha_forward(&q_norm, &q_norm, nq, nq, &block.self_attn, w.num_heads, d);
        let q_res: Vec<f32> = q.iter().zip(&q_self).map(|(a, b)| a + b).collect();
        q = q_res;

        // Cross-attention: queries → ref_feats
        let q_norm2 = batch_rms_norm(&q, &block.norm_cross, nq, d, rms_eps);
        let q_cross = mha_forward(
            &q_norm2,
            &ref_feats,
            nq,
            t_ref,
            &block.cross_attn,
            w.num_heads,
            d,
        );
        let q_res: Vec<f32> = q.iter().zip(&q_cross).map(|(a, b)| a + b).collect();
        q = q_res;

        // SwiGLU FFN
        let q_norm3 = batch_rms_norm(&q, &block.norm_ffn, nq, d, rms_eps);
        let ffn_h = w.ffn_hidden;
        let gate = batch_matvec(&block.gate_w, &q_norm3, nq, d, ffn_h);
        let up = batch_matvec(&block.up_w, &q_norm3, nq, d, ffn_h);
        let act: Vec<f32> = gate.iter().zip(&up).map(|(g, u)| silu(*g) * u).collect();
        let ffn_out = batch_matvec(&block.down_w, &act, nq, ffn_h, d);
        q = q.iter().zip(&ffn_out).map(|(a, b)| a + b).collect();
    }

    // 6. Final RMSNorm + output_scale
    let q_normed = batch_rms_norm(&q, &w.final_norm_w, nq, d, rms_eps);
    q_normed.iter().map(|v| v * w.output_scale).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sinusoidal_pe_shape() {
        let pe = sinusoidal_pe(10, 64);
        assert_eq!(pe.len(), 10 * 64);
        // First position, even dim: sin(0) = 0
        assert!((pe[0]).abs() < 1e-6);
        // First position, odd dim: cos(0) = 1
        assert!((pe[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn mha_output_shape() {
        // Tiny config: 2 queries, 4 keys, d=4, 2 heads
        let d = 4;
        let num_heads = 2;
        let tq = 2;
        let tkv = 4;
        let attn = QFormerAttnWeights {
            q_proj: vec![0.0; d * d],
            k_proj: vec![0.0; d * d],
            v_proj: vec![0.0; d * d],
            out_proj: vec![0.0; d * d],
        };
        let q_in = vec![0.0f32; tq * d];
        let kv_in = vec![0.0f32; tkv * d];
        let out = mha_forward(&q_in, &kv_in, tq, tkv, &attn, num_heads, d);
        assert_eq!(out.len(), tq * d);
    }

    #[test]
    fn expected_keys_count() {
        // 2 blocks
        let keys = QFormerWeights::expected_keys(2);
        // 3 base + 2*(3 norms + 8 attn_proj + 3 ffn) + 2 final = 3 + 2*14 + 2 = 33
        assert_eq!(keys.len(), 33);
    }
}
