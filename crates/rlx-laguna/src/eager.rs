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

//! Eager CPU Laguna forward for synthetic / small configs.
//!
//! Mirrors transformers `LagunaForCausalLM` at float32 for unit checks.
//! Full-attention layers use YaRN inv-freq + `attention_factor` (same as
//! [`crate::packed_forward`]); SWA layers keep plain RoPE.

use crate::config::{AttnGating, LagunaConfig};
use crate::packed_forward::{rope_inv_freq, rotary_freqs};
use anyhow::{Result, anyhow, bail};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct TextWeights {
    pub tensors: HashMap<String, Vec<f32>>,
}

impl TextWeights {
    pub fn get(&self, key: &str) -> Result<&[f32]> {
        self.tensors
            .get(key)
            .map(|v| v.as_slice())
            .ok_or_else(|| anyhow!("missing weight {key}"))
    }
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { (1.0 + x.exp()).ln() }
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

/// Linear: y = x @ W^T where W is `[out, in]` row-major (HF / transformers).
fn linear(x: &[f32], w: &[f32], seq: usize, out_dim: usize, in_dim: usize) -> Vec<f32> {
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

fn dense_mlp(
    x: &[f32],
    w: &TextWeights,
    layer: usize,
    seq: usize,
    h: usize,
    inter: usize,
) -> Result<Vec<f32>> {
    let gate = linear(x, w.get(&format!("layers.{layer}.gate"))?, seq, inter, h);
    let up = linear(x, w.get(&format!("layers.{layer}.up"))?, seq, inter, h);
    let mut mid = vec![0.0; seq * inter];
    for i in 0..mid.len() {
        mid[i] = silu(gate[i]) * up[i];
    }
    Ok(linear(
        &mid,
        w.get(&format!("layers.{layer}.down"))?,
        seq,
        h,
        inter,
    ))
}

fn moe_mlp(
    cfg: &LagunaConfig,
    x: &[f32],
    w: &TextWeights,
    layer: usize,
    seq: usize,
) -> Result<Vec<f32>> {
    let h = cfg.hidden_size;
    let ne = cfg.num_experts;
    let top_k = cfg.num_experts_per_tok.min(ne).max(1);
    let inter = cfg.moe_intermediate_size;
    let shared_inter = cfg.shared_expert_intermediate_size;

    let shared = dense_mlp_named(
        x,
        w,
        &format!("layers.{layer}.shared_gate"),
        &format!("layers.{layer}.shared_up"),
        &format!("layers.{layer}.shared_down"),
        seq,
        h,
        shared_inter,
    )?;

    let router_w = w.get(&format!("layers.{layer}.gate_weight"))?; // [ne, h]
    let bias = w.get(&format!("layers.{layer}.gate_bias"))?; // [ne]
    let logits = linear(x, router_w, seq, ne, h);
    let mut scores = vec![0.0; seq * ne];
    for i in 0..logits.len() {
        let mut z = logits[i];
        if cfg.moe_router_logit_softcapping > 0.0 {
            let c = cfg.moe_router_logit_softcapping;
            z = (z / c).tanh() * c;
        }
        scores[i] = sigmoid(z);
    }

    let mut out = vec![0.0; seq * h];
    let eg = w.get(&format!("layers.{layer}.expert_gate"))?; // [ne, inter, h]
    let eu = w.get(&format!("layers.{layer}.expert_up"))?;
    let ed = w.get(&format!("layers.{layer}.expert_down"))?; // [ne, h, inter]

    for t in 0..seq {
        let row = &scores[t * ne..(t + 1) * ne];
        let mut order: Vec<(usize, f32)> = (0..ne)
            .map(|e| (e, row[e] + bias.get(e).copied().unwrap_or(0.0)))
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
            let gate_w = &eg[e * inter * h..(e + 1) * inter * h];
            let up_w = &eu[e * inter * h..(e + 1) * inter * h];
            let down_w = &ed[e * h * inter..(e + 1) * h * inter];
            let mut mid = vec![0.0; inter];
            for o in 0..inter {
                let mut g = 0.0;
                let mut u = 0.0;
                let gw = &gate_w[o * h..(o + 1) * h];
                let uw = &up_w[o * h..(o + 1) * h];
                for i in 0..h {
                    g += xt[i] * gw[i];
                    u += xt[i] * uw[i];
                }
                mid[o] = silu(g) * u;
            }
            for o in 0..h {
                let mut acc = 0.0;
                let dw = &down_w[o * inter..(o + 1) * inter];
                for i in 0..inter {
                    acc += mid[i] * dw[i];
                }
                out[t * h + o] += acc * rw * cfg.moe_routed_scaling_factor;
            }
        }
        for o in 0..h {
            out[t * h + o] += shared[t * h + o];
        }
    }
    Ok(out)
}

fn dense_mlp_named(
    x: &[f32],
    w: &TextWeights,
    gate_k: &str,
    up_k: &str,
    down_k: &str,
    seq: usize,
    h: usize,
    inter: usize,
) -> Result<Vec<f32>> {
    let gate = linear(x, w.get(gate_k)?, seq, inter, h);
    let up = linear(x, w.get(up_k)?, seq, inter, h);
    let mut mid = vec![0.0; seq * inter];
    for i in 0..mid.len() {
        mid[i] = silu(gate[i]) * up[i];
    }
    Ok(linear(&mid, w.get(down_k)?, seq, h, inter))
}

fn attention(
    cfg: &LagunaConfig,
    layer: usize,
    x: &[f32],
    w: &TextWeights,
    seq: usize,
) -> Result<Vec<f32>> {
    let h = cfg.hidden_size;
    let n_heads = cfg.n_heads(layer);
    let n_kv = cfg.num_key_value_heads;
    let hd = cfg.head_dim;
    let groups = n_heads / n_kv;
    let q_dim = n_heads * hd;
    let kv_dim = n_kv * hd;
    let rope = cfg.rope_for_layer(layer);
    let scale = (hd as f32).sqrt().recip() * rope.attention_factor.max(1e-6);
    let rot_dim = ((hd as f32) * rope.partial_rotary_factor).round() as usize;
    let rot_dim = rot_dim.max(2) & !1; // even

    let q = linear(x, w.get(&format!("layers.{layer}.wq"))?, seq, q_dim, h);
    let k = linear(x, w.get(&format!("layers.{layer}.wk"))?, seq, kv_dim, h);
    let v = linear(x, w.get(&format!("layers.{layer}.wv"))?, seq, kv_dim, h);

    // QK RMSNorm per head
    let qn_w = w.get(&format!("layers.{layer}.q_norm"))?;
    let kn_w = w.get(&format!("layers.{layer}.k_norm"))?;
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

    let (cos, sin) = rotary_freqs(0, seq, rot_dim, &rope_inv_freq(rope, rot_dim));
    apply_rope_inplace(&mut qn, &cos, &sin, seq, n_heads, hd, rot_dim);
    apply_rope_inplace(&mut kn, &cos, &sin, seq, n_kv, hd, rot_dim);

    // Expand KV for GQA
    let mut k_full = vec![0.0; seq * n_heads * hd];
    let mut v_full = vec![0.0; seq * n_heads * hd];
    for t in 0..seq {
        for hq in 0..n_heads {
            let hk = hq / groups;
            let dst = (t * n_heads + hq) * hd;
            let src_k = (t * n_kv + hk) * hd;
            let src_v = (t * n_kv + hk) * hd;
            k_full[dst..dst + hd].copy_from_slice(&kn[src_k..src_k + hd]);
            v_full[dst..dst + hd].copy_from_slice(&v[src_v..src_v + hd]);
        }
    }

    let window = if cfg.is_sliding(layer) {
        cfg.sliding_window.max(1)
    } else {
        seq
    };

    let mut attn_out = vec![0.0; seq * q_dim];
    for hq in 0..n_heads {
        for tq in 0..seq {
            let mut scores = vec![0.0f32; seq];
            let qrow = &qn[(tq * n_heads + hq) * hd..(tq * n_heads + hq + 1) * hd];
            let t_min = tq.saturating_sub(window - 1);
            for tk in 0..=tq {
                if tk < t_min {
                    scores[tk] = f32::NEG_INFINITY;
                    continue;
                }
                let krow = &k_full[(tk * n_heads + hq) * hd..(tk * n_heads + hq + 1) * hd];
                let mut dot = 0.0;
                for j in 0..hd {
                    dot += qrow[j] * krow[j];
                }
                scores[tk] = dot * scale;
            }
            for tk in (tq + 1)..seq {
                scores[tk] = f32::NEG_INFINITY;
            }
            softmax_row(&mut scores);
            let out_base = (tq * n_heads + hq) * hd;
            for tk in t_min..=tq {
                let vrow = &v_full[(tk * n_heads + hq) * hd..(tk * n_heads + hq + 1) * hd];
                let a = scores[tk];
                for j in 0..hd {
                    attn_out[out_base + j] += a * vrow[j];
                }
            }
        }
    }

    // Softplus output gate before o_proj
    if cfg.gating != AttnGating::Off {
        let gate_out = match cfg.gating {
            AttnGating::PerHead => n_heads,
            AttnGating::PerElement => q_dim,
            AttnGating::Off => unreachable!(),
        };
        let g = linear(x, w.get(&format!("layers.{layer}.wg"))?, seq, gate_out, h);
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

    Ok(linear(
        &attn_out,
        w.get(&format!("layers.{layer}.wo"))?,
        seq,
        h,
        q_dim,
    ))
}

/// Last-token logits `[vocab]`.
pub fn forward_logits(cfg: &LagunaConfig, w: &TextWeights, prompt_ids: &[u32]) -> Result<Vec<f32>> {
    if prompt_ids.is_empty() {
        bail!("empty prompt");
    }
    let h = cfg.hidden_size;
    let v = cfg.vocab_size;
    let seq = prompt_ids.len();
    let emb = w.get("embed")?;
    let mut x = vec![0.0; seq * h];
    for (t, &id) in prompt_ids.iter().enumerate() {
        let id = id as usize;
        if id >= v {
            bail!("token id {id} >= vocab {v}");
        }
        x[t * h..(t + 1) * h].copy_from_slice(&emb[id * h..(id + 1) * h]);
    }

    for layer in 0..cfg.num_hidden_layers {
        let residual = x.clone();
        let normed = rms_norm(
            &x,
            w.get(&format!("layers.{layer}.attn_norm"))?,
            cfg.rms_norm_eps,
        );
        let attn = attention(cfg, layer, &normed, w, seq)?;
        for i in 0..x.len() {
            x[i] = residual[i] + attn[i];
        }
        let residual = x.clone();
        let normed = rms_norm(
            &x,
            w.get(&format!("layers.{layer}.ffn_norm"))?,
            cfg.rms_norm_eps,
        );
        let ffn = if cfg.is_dense_mlp(layer) {
            dense_mlp(&normed, w, layer, seq, h, cfg.intermediate_size)?
        } else {
            moe_mlp(cfg, &normed, w, layer, seq)?
        };
        for i in 0..x.len() {
            x[i] = residual[i] + ffn[i];
        }
    }

    let last = &x[(seq - 1) * h..seq * h];
    let normed = rms_norm(last, w.get("norm")?, cfg.rms_norm_eps);
    let unemb = if cfg.tie_word_embeddings {
        w.get("embed")?
    } else {
        w.get("unembed")?
    };
    Ok(linear(&normed, unemb, 1, v, h))
}

pub fn greedy_next(cfg: &LagunaConfig, w: &TextWeights, prompt_ids: &[u32]) -> Result<u32> {
    let logits = forward_logits(cfg, w, prompt_ids)?;
    let (idx, _) = logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .ok_or_else(|| anyhow!("empty logits"))?;
    Ok(idx as u32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synth::{synthetic_text_weights, tiny_cfg};

    #[test]
    fn synth_forward_runs() {
        let cfg = tiny_cfg();
        let w = synthetic_text_weights(&cfg);
        let logits = forward_logits(&cfg, &w, &[1, 2, 3]).unwrap();
        assert_eq!(logits.len(), cfg.vocab_size);
        assert!(logits.iter().all(|v| v.is_finite()));
        let next = greedy_next(&cfg, &w, &[1, 2, 3]).unwrap();
        assert!((next as usize) < cfg.vocab_size);
    }
}
