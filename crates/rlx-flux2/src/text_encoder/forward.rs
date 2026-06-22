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

//! Native CPU forward for the FLUX.2 Qwen3 text encoder.

use super::prompt::DEFAULT_TEXT_ENCODER_LAYERS;
use super::weights::{
    Flux2TextEncoderAttnWeights, Flux2TextEncoderLayerWeights, Flux2TextEncoderMlpWeights,
    Flux2TextEncoderWeights,
};
use anyhow::{Result, ensure};
use rlx_core::host_kernels::{layer_norm, linear};
use rlx_qwen3::Qwen3Config;

#[derive(Debug, Clone)]
pub struct Flux2PromptOutput {
    pub prompt_embeds: Vec<f32>,
    pub seq_len: usize,
    pub joint_dim: usize,
}

fn rms_norm(x: &[f32], scale: &[f32], dim: usize, eps: f32) -> Result<Vec<f32>> {
    let beta = vec![0.0f32; dim];
    layer_norm(x, scale, &beta, dim, eps)
}

fn rms_norm_heads(
    x: &[f32],
    scale: &[f32],
    batch: usize,
    seq: usize,
    heads: usize,
    head_dim: usize,
    eps: f32,
) -> Result<Vec<f32>> {
    let mut out = vec![0.0f32; x.len()];
    for b in 0..batch {
        for t in 0..seq {
            for h in 0..heads {
                let off = ((b * seq + t) * heads + h) * head_dim;
                let row = rms_norm(&x[off..off + head_dim], scale, head_dim, eps)?;
                out[off..off + head_dim].copy_from_slice(&row);
            }
        }
    }
    Ok(out)
}

fn mlp_forward(
    mlp: &Flux2TextEncoderMlpWeights,
    x: &[f32],
    rows: usize,
    _dim: usize,
) -> Result<Vec<f32>> {
    let gate = linear(
        x,
        rows,
        mlp.gate.in_dim,
        &mlp.gate.w_t,
        mlp.gate.out_dim,
        &mlp.gate.bias,
    )?;
    let up = linear(
        x,
        rows,
        mlp.up.in_dim,
        &mlp.up.w_t,
        mlp.up.out_dim,
        &mlp.up.bias,
    )?;
    let half = mlp.gate.out_dim;
    let mut h = vec![0.0f32; rows * half];
    for r in 0..rows {
        for c in 0..half {
            let a = gate[r * half + c];
            let b = up[r * half + c];
            let s = a / (1.0 + (-a).exp());
            h[r * half + c] = s * b;
        }
    }
    linear(
        &h,
        rows,
        mlp.down.in_dim,
        &mlp.down.w_t,
        mlp.down.out_dim,
        &mlp.down.bias,
    )
}

fn rope_cache(cfg: &Qwen3Config, seq: usize) -> (Vec<f32>, Vec<f32>) {
    let dh = cfg.head_dim;
    let half = dh / 2;
    let mut cos = vec![0.0f32; seq * dh];
    let mut sin = vec![0.0f32; seq * dh];
    for pos in 0..seq {
        for i in 0..half {
            let freq = 1.0 / cfg.rope_theta.powf((2 * i) as f64 / dh as f64);
            let angle = pos as f64 * freq;
            let c = angle.cos() as f32;
            let s = angle.sin() as f32;
            cos[pos * dh + 2 * i] = c;
            cos[pos * dh + 2 * i + 1] = c;
            sin[pos * dh + 2 * i] = s;
            sin[pos * dh + 2 * i + 1] = s;
        }
    }
    (cos, sin)
}

fn apply_rope_row(x: &mut [f32], cos: &[f32], sin: &[f32], head_dim: usize) {
    let mut rotated = vec![0.0f32; head_dim];
    let pairs = head_dim / 2;
    for i in 0..pairs {
        let xr = x[2 * i];
        let xi = x[2 * i + 1];
        rotated[2 * i] = -xi;
        rotated[2 * i + 1] = xr;
    }
    for d in 0..head_dim {
        x[d] = x[d] * cos[d] + rotated[d] * sin[d];
    }
}

fn repeat_kv(
    k: &[f32],
    v: &[f32],
    batch: usize,
    seq: usize,
    n_kv: usize,
    n_heads: usize,
    head_dim: usize,
) -> (Vec<f32>, Vec<f32>) {
    let group = n_heads / n_kv;
    let mut k_out = vec![0.0f32; batch * seq * n_heads * head_dim];
    let mut v_out = vec![0.0f32; batch * seq * n_heads * head_dim];
    for b in 0..batch {
        for t in 0..seq {
            for h in 0..n_heads {
                let kv_h = h / group;
                let src = ((b * seq + t) * n_kv + kv_h) * head_dim;
                let dst = ((b * seq + t) * n_heads + h) * head_dim;
                k_out[dst..dst + head_dim].copy_from_slice(&k[src..src + head_dim]);
                v_out[dst..dst + head_dim].copy_from_slice(&v[src..src + head_dim]);
            }
        }
    }
    (k_out, v_out)
}

fn causal_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    batch: usize,
    seq: usize,
    n_heads: usize,
    head_dim: usize,
    scale: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; batch * seq * n_heads * head_dim];
    for b in 0..batch {
        for h in 0..n_heads {
            for i in 0..seq {
                let q_off = ((b * seq + i) * n_heads + h) * head_dim;
                let q_h = &q[q_off..q_off + head_dim];
                let mut scores = vec![0.0f32; i + 1];
                let mut max_s = f32::NEG_INFINITY;
                for j in 0..=i {
                    let k_off = ((b * seq + j) * n_heads + h) * head_dim;
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q_h[d] * k[k_off + d];
                    }
                    let s = dot * scale;
                    scores[j] = s;
                    max_s = max_s.max(s);
                }
                let mut sum = 0.0f32;
                let mut probs = vec![0.0f32; i + 1];
                for j in 0..=i {
                    let e = (scores[j] - max_s).exp();
                    probs[j] = e;
                    sum += e;
                }
                for j in 0..=i {
                    probs[j] /= sum;
                }
                let o_off = ((b * seq + i) * n_heads + h) * head_dim;
                for d in 0..head_dim {
                    let mut acc = 0.0f32;
                    for j in 0..=i {
                        let v_off = ((b * seq + j) * n_heads + h) * head_dim;
                        acc += probs[j] * v[v_off + d];
                    }
                    out[o_off + d] = acc;
                }
            }
        }
    }
    out
}

fn attn_forward(
    attn: &Flux2TextEncoderAttnWeights,
    x: &[f32],
    cos: &[f32],
    sin: &[f32],
    batch: usize,
    seq: usize,
    cfg: &Qwen3Config,
) -> Result<Vec<f32>> {
    let nh = cfg.num_attention_heads;
    let nkv = cfg.num_key_value_heads;
    let hd = cfg.head_dim;
    let rows = batch * seq;

    let mut q = linear(
        x,
        rows,
        attn.q.in_dim,
        &attn.q.w_t,
        attn.q.out_dim,
        &attn.q.bias,
    )?;
    let mut k = linear(
        x,
        rows,
        attn.k.in_dim,
        &attn.k.w_t,
        attn.k.out_dim,
        &attn.k.bias,
    )?;
    let v = linear(
        x,
        rows,
        attn.v.in_dim,
        &attn.v.w_t,
        attn.v.out_dim,
        &attn.v.bias,
    )?;

    q = rms_norm_heads(
        &q,
        &attn.q_norm.scale,
        batch,
        seq,
        nh,
        hd,
        cfg.rms_norm_eps as f32,
    )?;
    k = rms_norm_heads(
        &k,
        &attn.k_norm.scale,
        batch,
        seq,
        nkv,
        hd,
        cfg.rms_norm_eps as f32,
    )?;

    for t in 0..seq {
        let c = &cos[t * hd..(t + 1) * hd];
        let s = &sin[t * hd..(t + 1) * hd];
        for b in 0..batch {
            for h in 0..nh {
                let off = ((b * seq + t) * nh + h) * hd;
                apply_rope_row(&mut q[off..off + hd], c, s, hd);
            }
            for h in 0..nkv {
                let off = ((b * seq + t) * nkv + h) * hd;
                apply_rope_row(&mut k[off..off + hd], c, s, hd);
            }
        }
    }

    let (k_rep, v_rep) = repeat_kv(&k, &v, batch, seq, nkv, nh, hd);
    let scale = 1.0 / (hd as f32).sqrt();
    let attn_out = causal_attention(&q, &k_rep, &v_rep, batch, seq, nh, hd, scale);
    linear(
        &attn_out,
        rows,
        attn.o.in_dim,
        &attn.o.w_t,
        attn.o.out_dim,
        &attn.o.bias,
    )
}

fn layer_forward(
    layer: &Flux2TextEncoderLayerWeights,
    x: &[f32],
    cos: &[f32],
    sin: &[f32],
    batch: usize,
    seq: usize,
    cfg: &Qwen3Config,
) -> Result<Vec<f32>> {
    let h = cfg.hidden_size;
    let rows = batch * seq;
    let eps = cfg.rms_norm_eps as f32;

    let normed = rms_norm(x, &layer.input_layernorm.scale, h, eps)?;
    let attn_out = attn_forward(&layer.attn, &normed, cos, sin, batch, seq, cfg)?;
    let mut hidden = vec![0.0f32; x.len()];
    for i in 0..hidden.len() {
        hidden[i] = x[i] + attn_out[i];
    }

    let normed2 = rms_norm(&hidden, &layer.post_attention_layernorm.scale, h, eps)?;
    let mlp_out = mlp_forward(&layer.mlp, &normed2, rows, h)?;
    for i in 0..hidden.len() {
        hidden[i] += mlp_out[i];
    }
    Ok(hidden)
}

fn embed_tokens(
    embed: &(Vec<f32>, usize, usize),
    input_ids: &[u32],
    batch: usize,
    seq: usize,
    hidden: usize,
) -> Vec<f32> {
    let (data, vocab, _) = embed;
    let mut out = vec![0.0f32; batch * seq * hidden];
    for b in 0..batch {
        for t in 0..seq {
            let id = input_ids[b * seq + t] as usize;
            let id = id.min(vocab.saturating_sub(1));
            let src = id * hidden;
            let dst = (b * seq + t) * hidden;
            out[dst..dst + hidden].copy_from_slice(&data[src..src + hidden]);
        }
    }
    out
}

/// Encode token ids → FLUX.2 `encoder_hidden_states` + metadata.
pub fn encode_prompt_embeds(
    weights: &Flux2TextEncoderWeights,
    cfg: &Qwen3Config,
    input_ids: &[u32],
    batch: usize,
    seq: usize,
    hidden_state_layers: &[usize],
) -> Result<Flux2PromptOutput> {
    ensure!(input_ids.len() == batch * seq, "input_ids length mismatch");
    let (cos, sin) = rope_cache(cfg, seq);
    let mut hidden = embed_tokens(
        &weights.embed_tokens,
        input_ids,
        batch,
        seq,
        cfg.hidden_size,
    );
    let mut hidden_states: Vec<Vec<f32>> = vec![hidden.clone()];
    for layer in &weights.layers {
        hidden = layer_forward(layer, &hidden, &cos, &sin, batch, seq, cfg)?;
        hidden_states.push(hidden.clone());
    }
    let eps = cfg.rms_norm_eps as f32;
    let _ = rms_norm(&hidden, &weights.norm.scale, cfg.hidden_size, eps)?;

    let h = cfg.hidden_size;
    let joint_dim = h * hidden_state_layers.len();
    let mut prompt_embeds = vec![0.0f32; batch * seq * joint_dim];
    for b in 0..batch {
        for t in 0..seq {
            let mut off = 0usize;
            for (li, &layer_idx) in hidden_state_layers.iter().enumerate() {
                ensure!(
                    layer_idx < hidden_states.len(),
                    "hidden_state_layers[{li}]={layer_idx} out of range (len={})",
                    hidden_states.len()
                );
                let src = (b * seq + t) * h;
                let dst = (b * seq + t) * joint_dim + off;
                prompt_embeds[dst..dst + h]
                    .copy_from_slice(&hidden_states[layer_idx][src..src + h]);
                off += h;
            }
        }
    }
    Ok(Flux2PromptOutput {
        prompt_embeds,
        seq_len: seq,
        joint_dim,
    })
}

/// Encode with default Klein layer indices (9, 18, 27).
pub fn encode_prompt_embeds_default_layers(
    weights: &Flux2TextEncoderWeights,
    cfg: &Qwen3Config,
    input_ids: &[u32],
    batch: usize,
    seq: usize,
) -> Result<Flux2PromptOutput> {
    encode_prompt_embeds(
        weights,
        cfg,
        input_ids,
        batch,
        seq,
        DEFAULT_TEXT_ENCODER_LAYERS,
    )
}
