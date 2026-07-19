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

//! Eager CPU text forward for synthetic / small Inkling configs.
//!
//! Mirrors transformers `InklingTextModel` + `InklingForCausalLM` at float32
//! for unit checks. Not a production path for the 975B checkpoint.

use crate::config::InklingTextConfig;
use anyhow::{Result, anyhow, bail};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct TextWeights {
    pub tensors: HashMap<String, Vec<f32>>,
}

impl TextWeights {
    fn get(&self, key: &str) -> Result<&[f32]> {
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

/// Linear: y = x @ W^T where W is `[out, in]` row-major.
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

/// Depthwise causal conv1d with residual: `y = conv(x) + x`.
/// Weight layout `[channels, kernel]` (channel-major kernels).
fn short_conv(x: &[f32], w: &[f32], seq: usize, channels: usize, kernel: usize) -> Vec<f32> {
    debug_assert_eq!(x.len(), seq * channels);
    debug_assert_eq!(w.len(), channels * kernel);
    let mut y = vec![0.0; seq * channels];
    for t in 0..seq {
        for c in 0..channels {
            let mut acc = 0.0;
            for k in 0..kernel {
                let src = t as isize - (kernel as isize - 1 - k as isize);
                if src >= 0 {
                    acc += x[src as usize * channels + c] * w[c * kernel + k];
                }
            }
            y[t * channels + c] = acc + x[t * channels + c];
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

fn relative_bias(
    rel_states: &[f32], // [T, H, d_rel]
    proj: &[f32],       // [d_rel, rel_extent]
    seq: usize,
    n_heads: usize,
    d_rel: usize,
    rel_extent: usize,
) -> Vec<f32> {
    // Mix profiles → [T, H, rel_extent]
    let mut mixed = vec![0.0; seq * n_heads * rel_extent];
    for t in 0..seq {
        for h in 0..n_heads {
            let rs = &rel_states[(t * n_heads + h) * d_rel..(t * n_heads + h + 1) * d_rel];
            for e in 0..rel_extent {
                let mut acc = 0.0;
                for d in 0..d_rel {
                    acc += rs[d] * proj[d * rel_extent + e];
                }
                mixed[(t * n_heads + h) * rel_extent + e] = acc;
            }
        }
    }
    // Gather into [H, Tq, Tk] materialised as [H * T * T]
    let mut bias = vec![0.0; n_heads * seq * seq];
    for h in 0..n_heads {
        for tq in 0..seq {
            for tk in 0..seq {
                let dist = tq as isize - tk as isize;
                let v = if dist >= 0 && (dist as usize) < rel_extent {
                    mixed[(tq * n_heads + h) * rel_extent + dist as usize]
                } else {
                    0.0
                };
                bias[h * seq * seq + tq * seq + tk] = v;
            }
        }
    }
    bias
}

fn attention(
    cfg: &InklingTextConfig,
    layer: usize,
    x: &[f32],
    w: &TextWeights,
    seq: usize,
) -> Result<Vec<f32>> {
    let h = cfg.hidden_size;
    let (n_heads, n_kv, hd) = cfg.attn_heads(layer);
    let groups = n_heads / n_kv;
    let q_dim = n_heads * hd;
    let kv_dim = n_kv * hd;
    let kconv = cfg.conv_kernel_size;
    let rel_extent = cfg.rel_extent_for_layer(layer);

    let q = linear(x, w.get(&format!("layers.{layer}.wq"))?, seq, q_dim, h);
    let mut k = linear(x, w.get(&format!("layers.{layer}.wk"))?, seq, kv_dim, h);
    let mut v = linear(x, w.get(&format!("layers.{layer}.wv"))?, seq, kv_dim, h);
    k = short_conv(
        &k,
        w.get(&format!("layers.{layer}.k_sconv"))?,
        seq,
        kv_dim,
        kconv,
    );
    v = short_conv(
        &v,
        w.get(&format!("layers.{layer}.v_sconv"))?,
        seq,
        kv_dim,
        kconv,
    );
    let r = linear(
        x,
        w.get(&format!("layers.{layer}.wr"))?,
        seq,
        n_heads * cfg.d_rel,
        h,
    );

    let q_norm_w = w.get(&format!("layers.{layer}.q_norm"))?;
    let k_norm_w = w.get(&format!("layers.{layer}.k_norm"))?;
    let mut qn = vec![0.0; q.len()];
    let mut kn = vec![0.0; k.len()];
    for t in 0..seq {
        for head in 0..n_heads {
            let off = t * q_dim + head * hd;
            let normed = rms_norm(&q[off..off + hd], q_norm_w, cfg.rms_norm_eps);
            qn[off..off + hd].copy_from_slice(&normed);
        }
        for head in 0..n_kv {
            let off = t * kv_dim + head * hd;
            let normed = rms_norm(&k[off..off + hd], k_norm_w, cfg.rms_norm_eps);
            kn[off..off + hd].copy_from_slice(&normed);
        }
    }

    // Relative states [T, H, d_rel]
    let mut rel_states = vec![0.0; seq * n_heads * cfg.d_rel];
    for t in 0..seq {
        for head in 0..n_heads {
            let src = t * n_heads * cfg.d_rel + head * cfg.d_rel;
            let dst = (t * n_heads + head) * cfg.d_rel;
            rel_states[dst..dst + cfg.d_rel].copy_from_slice(&r[src..src + cfg.d_rel]);
        }
    }
    let pos_bias = relative_bias(
        &rel_states,
        w.get(&format!("layers.{layer}.rel_proj"))?,
        seq,
        n_heads,
        cfg.d_rel,
        rel_extent,
    );

    let scale = 1.0 / hd as f32;
    let sliding = cfg.is_sliding(layer);
    let win = cfg.sliding_window_size;

    let mut ctx = vec![0.0; seq * q_dim];
    for head in 0..n_heads {
        let kv_head = head / groups;
        for tq in 0..seq {
            let mut logits = vec![f32::NEG_INFINITY; seq];
            for tk in 0..seq {
                if tk > tq {
                    continue;
                }
                if sliding && tq.saturating_sub(tk) >= win {
                    continue;
                }
                let mut dot = 0.0;
                let qo = tq * q_dim + head * hd;
                let ko = tk * kv_dim + kv_head * hd;
                for d in 0..hd {
                    dot += qn[qo + d] * kn[ko + d];
                }
                logits[tk] = dot * scale + pos_bias[head * seq * seq + tq * seq + tk];
            }
            softmax_row(&mut logits);
            let out_off = tq * q_dim + head * hd;
            for tk in 0..seq {
                let vo = tk * kv_dim + kv_head * hd;
                let a = logits[tk];
                if a == 0.0 {
                    continue;
                }
                for d in 0..hd {
                    ctx[out_off + d] += a * v[vo + d];
                }
            }
        }
    }

    Ok(linear(
        &ctx,
        w.get(&format!("layers.{layer}.wo"))?,
        seq,
        h,
        q_dim,
    ))
}

fn dense_mlp(
    cfg: &InklingTextConfig,
    layer: usize,
    x: &[f32],
    w: &TextWeights,
) -> Result<Vec<f32>> {
    let h = cfg.hidden_size;
    let inter = cfg.dense_intermediate_size;
    let seq = x.len() / h;
    let gate = linear(x, w.get(&format!("layers.{layer}.gate"))?, seq, inter, h);
    let up = linear(x, w.get(&format!("layers.{layer}.up"))?, seq, inter, h);
    let mut mid = vec![0.0; seq * inter];
    for i in 0..mid.len() {
        mid[i] = silu(gate[i]) * up[i];
    }
    let mut y = linear(&mid, w.get(&format!("layers.{layer}.down"))?, seq, h, inter);
    let scale = w.get(&format!("layers.{layer}.mlp_global_scale"))?[0];
    for v in &mut y {
        *v *= scale;
    }
    Ok(y)
}

fn moe_mlp(cfg: &InklingTextConfig, layer: usize, x: &[f32], w: &TextWeights) -> Result<Vec<f32>> {
    let h = cfg.hidden_size;
    let inter = cfg.moe_intermediate_size;
    let ne = cfg.n_routed_experts;
    let ns = cfg.n_shared_experts;
    let top_k = cfg.num_experts_per_tok;
    let seq = x.len() / h;
    let tokens = seq; // batch=1

    let gate_w = w.get(&format!("layers.{layer}.gate_weight"))?;
    let gate_bias = w.get(&format!("layers.{layer}.gate_bias"))?;
    let gscale = w.get(&format!("layers.{layer}.gate_global_scale"))?[0];
    let expert_w13 = w.get(&format!("layers.{layer}.expert_w13"))?;
    let expert_w2 = w.get(&format!("layers.{layer}.expert_w2"))?;
    let shared_gate = w.get(&format!("layers.{layer}.shared_gate"))?;
    let shared_up = w.get(&format!("layers.{layer}.shared_up"))?;
    let shared_down = w.get(&format!("layers.{layer}.shared_down"))?;

    let mut out = vec![0.0; x.len()];

    for t in 0..tokens {
        let xt = &x[t * h..(t + 1) * h];
        // router logits [ne+ns]
        let mut logits = vec![0.0; ne + ns];
        for e in 0..(ne + ns) {
            let mut acc = 0.0;
            let wr = &gate_w[e * h..(e + 1) * h];
            for i in 0..h {
                acc += xt[i] * wr[i];
            }
            logits[e] = acc;
        }
        // top-k over routed (sigmoid scores + correction bias)
        let mut scored: Vec<(usize, f32)> = (0..ne)
            .map(|e| (e, sigmoid(logits[e]) + gate_bias[e]))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let top: Vec<usize> = scored.into_iter().take(top_k).map(|(e, _)| e).collect();

        // Joint normalize selected routed + shared via logsigmoid
        let mut selected_logits = Vec::with_capacity(top_k + ns);
        for &e in &top {
            selected_logits.push(logits[e]);
        }
        for s in 0..ns {
            selected_logits.push(logits[ne + s]);
        }
        let log_sig: Vec<f32> = selected_logits
            .iter()
            .map(|&z| -((-z).exp()).ln_1p()) // logsigmoid
            .collect();
        let lse = {
            let m = log_sig.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            m + log_sig.iter().map(|v| (v - m).exp()).sum::<f32>().ln()
        };
        let mut weights: Vec<f32> = log_sig.iter().map(|v| (v - lse).exp()).collect();
        for wgt in &mut weights {
            *wgt *= cfg.route_scale * gscale;
        }
        let routed_w = &weights[..top_k];
        let shared_w = &weights[top_k..];

        // Routed experts
        for (ki, &e) in top.iter().enumerate() {
            let w13 = &expert_w13[e * 2 * inter * h..(e + 1) * 2 * inter * h];
            let w2 = &expert_w2[e * h * inter..(e + 1) * h * inter];
            let gate = linear(xt, &w13[..inter * h], 1, inter, h);
            let up = linear(xt, &w13[inter * h..], 1, inter, h);
            let mut mid = vec![0.0; inter];
            for i in 0..inter {
                mid[i] = silu(gate[i]) * up[i];
            }
            let y = linear(&mid, w2, 1, h, inter);
            let aw = routed_w[ki];
            for i in 0..h {
                out[t * h + i] += y[i] * aw;
            }
        }

        // Shared experts
        for s in 0..ns {
            let sg = &shared_gate[s * inter * h..(s + 1) * inter * h];
            let su = &shared_up[s * inter * h..(s + 1) * inter * h];
            let sd = &shared_down[s * h * inter..(s + 1) * h * inter];
            let gate = linear(xt, sg, 1, inter, h);
            let up = linear(xt, su, 1, inter, h);
            let mut mid = vec![0.0; inter];
            for i in 0..inter {
                mid[i] = silu(gate[i]) * up[i] * shared_w[s];
            }
            let y = linear(&mid, sd, 1, h, inter);
            for i in 0..h {
                out[t * h + i] += y[i];
            }
        }
    }
    Ok(out)
}

fn decoder_layer(
    cfg: &InklingTextConfig,
    layer: usize,
    x: &[f32],
    w: &TextWeights,
) -> Result<Vec<f32>> {
    let h = cfg.hidden_size;
    let seq = x.len() / h;
    let k = cfg.conv_kernel_size;

    let normed = rms_norm(
        x,
        w.get(&format!("layers.{layer}.attn_norm"))?,
        cfg.rms_norm_eps,
    );
    let mut attn = attention(cfg, layer, &normed, w, seq)?;
    attn = short_conv(
        &attn,
        w.get(&format!("layers.{layer}.attn_sconv"))?,
        seq,
        h,
        k,
    );
    let mut h_states: Vec<f32> = x.iter().zip(attn.iter()).map(|(a, b)| a + b).collect();

    let normed = rms_norm(
        &h_states,
        w.get(&format!("layers.{layer}.mlp_norm"))?,
        cfg.rms_norm_eps,
    );
    let mut mlp = if cfg.is_dense_mlp(layer) {
        dense_mlp(cfg, layer, &normed, w)?
    } else {
        moe_mlp(cfg, layer, &normed, w)?
    };
    mlp = short_conv(
        &mlp,
        w.get(&format!("layers.{layer}.mlp_sconv"))?,
        seq,
        h,
        k,
    );
    for i in 0..h_states.len() {
        h_states[i] += mlp[i];
    }
    Ok(h_states)
}

/// Prefill logits for the last token: `[vocab]`.
pub fn forward_logits(
    cfg: &InklingTextConfig,
    w: &TextWeights,
    input_ids: &[u32],
) -> Result<Vec<f32>> {
    if input_ids.is_empty() {
        bail!("rlx-inkling: empty input_ids");
    }
    let h = cfg.hidden_size;
    let v = cfg.vocab_size;
    let seq = input_ids.len();
    let embed = w.get("embed")?;
    let mut hidden = vec![0.0; seq * h];
    for (t, &id) in input_ids.iter().enumerate() {
        if id as usize >= v {
            bail!("token id {id} out of range (vocab={v})");
        }
        let src = (id as usize) * h;
        hidden[t * h..(t + 1) * h].copy_from_slice(&embed[src..src + h]);
    }
    if cfg.use_embed_norm {
        hidden = rms_norm(&hidden, w.get("embed_norm")?, cfg.rms_norm_eps);
    }
    for layer in 0..cfg.num_hidden_layers {
        hidden = decoder_layer(cfg, layer, &hidden, w)?;
    }
    hidden = rms_norm(&hidden, w.get("norm")?, cfg.rms_norm_eps);
    let last = &hidden[(seq - 1) * h..seq * h];
    let mut logits = linear(last, w.get("unembed")?, 1, v, h);
    let mup = cfg.logits_mup_width_multiplier;
    if mup != 1.0 {
        for z in &mut logits {
            *z *= mup;
        }
    }
    Ok(logits)
}

/// Argmax of last-token logits (greedy).
pub fn greedy_next(cfg: &InklingTextConfig, w: &TextWeights, input_ids: &[u32]) -> Result<u32> {
    let logits = forward_logits(cfg, w, input_ids)?;
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
        let logits = forward_logits(&cfg, &w, &[1, 2, 3, 4]).unwrap();
        assert_eq!(logits.len(), cfg.vocab_size);
        assert!(logits.iter().all(|v| v.is_finite()));
        let tok = greedy_next(&cfg, &w, &[1, 2, 3]).unwrap();
        assert!((tok as usize) < cfg.vocab_size);
    }

    #[test]
    fn longer_prompt_still_finite() {
        let cfg = tiny_cfg();
        let w = synthetic_text_weights(&cfg);
        let ids: Vec<u32> = (0..8).map(|i| i % cfg.vocab_size as u32).collect();
        let logits = forward_logits(&cfg, &w, &ids).unwrap();
        assert!(logits.iter().all(|v| v.is_finite()));
    }
}
