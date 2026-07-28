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

//! gpt-oss architecture primitives + config — the two pieces that (per the
//! rlx audit) were **implemented nowhere**: the *clamped-SwiGLU* MoE expert and
//! *attention sinks*. These are host-reference implementations of the exact
//! gpt-oss math, with unit tests, so they double as the numeric oracle for the
//! eventual rlx-ir graph builder (which must reproduce them bit-for-bit).
//!
//! Sources: OpenAI `gpt-oss` reference (`GptOssForCausalLM`). 20B config:
//! 24 layers, 32 experts top-4, hidden 2880, GQA 64/8 head_dim 64, alternating
//! sliding(128)/full attention, YaRN rope (θ=150000, factor 32), `swiglu_limit
//! 7.0`, `attention_bias=true`, mixed quant (attn/embed affine-4, experts mxfp4).

use anyhow::{Context, Result};

/// gpt-oss hyperparameters parsed from a HF/mlx `config.json`.
#[derive(Debug, Clone)]
pub struct GptOssConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub num_local_experts: usize,
    pub experts_per_token: usize,
    pub sliding_window: usize,
    /// `layer_types[i] == "sliding_attention"` → sliding window on layer `i`,
    /// else full attention. gpt-oss alternates sliding/full.
    pub layer_is_sliding: Vec<bool>,
    pub swiglu_limit: f32,
    /// SwiGLU gate sigmoid coefficient (`glu = g·σ(α·g)`); gpt-oss uses 1.702.
    pub swiglu_alpha: f32,
    pub rms_norm_eps: f32,
    pub rope_theta: f64,
    pub attention_bias: bool,
    pub tie_word_embeddings: bool,
}

impl GptOssConfig {
    pub fn from_json_path(path: &std::path::Path) -> Result<Self> {
        let raw = std::fs::read(path).with_context(|| format!("gpt-oss: read {path:?}"))?;
        let v: serde_json::Value =
            serde_json::from_slice(&raw).with_context(|| format!("gpt-oss: parse {path:?}"))?;
        Self::from_json(&v)
    }

    pub fn from_json(v: &serde_json::Value) -> Result<Self> {
        let u = |k: &str| v.get(k).and_then(|x| x.as_u64()).map(|x| x as usize);
        let uu = |k: &str, d: usize| u(k).unwrap_or(d);
        let hidden_size = u("hidden_size").context("gpt-oss: hidden_size")?;
        let num_hidden_layers = u("num_hidden_layers").context("gpt-oss: num_hidden_layers")?;
        let num_attention_heads =
            u("num_attention_heads").context("gpt-oss: num_attention_heads")?;
        let head_dim = uu("head_dim", hidden_size / num_attention_heads.max(1));
        // gpt-oss stores the expert count as `num_local_experts`.
        let num_local_experts = u("num_local_experts")
            .or_else(|| u("num_experts"))
            .context("gpt-oss: num_local_experts")?;
        let experts_per_token = u("experts_per_token")
            .or_else(|| u("num_experts_per_tok"))
            .unwrap_or(4);
        let layer_is_sliding = v
            .get("layer_types")
            .and_then(|x| x.as_array())
            .map(|a| {
                a.iter()
                    .map(|t| t.as_str() == Some("sliding_attention"))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_else(|| (0..num_hidden_layers).map(|i| i % 2 == 0).collect());
        Ok(Self {
            vocab_size: u("vocab_size").context("gpt-oss: vocab_size")?,
            hidden_size,
            intermediate_size: uu("intermediate_size", hidden_size),
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads: uu("num_key_value_heads", num_attention_heads),
            head_dim,
            num_local_experts,
            experts_per_token,
            sliding_window: uu("sliding_window", 128),
            layer_is_sliding,
            swiglu_limit: v
                .get("swiglu_limit")
                .and_then(|x| x.as_f64())
                .unwrap_or(7.0) as f32,
            swiglu_alpha: 1.702,
            rms_norm_eps: v
                .get("rms_norm_eps")
                .and_then(|x| x.as_f64())
                .unwrap_or(1e-5) as f32,
            rope_theta: v
                .get("rope_theta")
                .and_then(|x| x.as_f64())
                .unwrap_or(150000.0),
            attention_bias: v
                .get("attention_bias")
                .and_then(|x| x.as_bool())
                .unwrap_or(true),
            tie_word_embeddings: v
                .get("tie_word_embeddings")
                .and_then(|x| x.as_bool())
                .unwrap_or(false),
        })
    }
}

/// gpt-oss **clamped-SwiGLU** expert activation, applied to the fused
/// `gate_up` projection output for one token.
///
/// gpt-oss interleaves gate/up columns (`gate = gate_up[..., 0::2]`,
/// `up = gate_up[..., 1::2]`), clamps them, then:
/// ```text
/// glu  = gate · σ(α · gate)          (α = 1.702, SiLU-ish)
/// out  = (up + 1) · glu
/// ```
/// with `gate` clamped to `≤ limit` (upper only) and `up` clamped to
/// `[-limit, limit]`. Returns the `[inter]` activation to feed `w_down`.
pub fn clamped_swiglu(gate_up: &[f32], limit: f32, alpha: f32) -> Vec<f32> {
    let inter = gate_up.len() / 2;
    let mut out = vec![0f32; inter];
    for i in 0..inter {
        let mut g = gate_up[2 * i]; // even = gate
        let mut u = gate_up[2 * i + 1]; // odd = up
        if g > limit {
            g = limit; // gate: clamp(max=limit), no lower bound
        }
        u = u.clamp(-limit, limit); // up: clamp(min=-limit, max=limit)
        let glu = g * (1.0 / (1.0 + (-(alpha * g)).exp()));
        out[i] = (u + 1.0) * glu;
    }
    out
}

/// Softmax over attention logits **with a per-head sink**: an extra learned
/// logit `sink` participates in the denominator but has no value column, so it
/// bleeds probability mass off the real keys. Returns the normalized weights
/// over the `n` real keys (the sink weight is dropped).
///
/// `logits[j]` are the (already scaled + masked) scores `q·kⱼ`; masked-out keys
/// should be `f32::NEG_INFINITY`. `probs = softmax([logits ; sink])[:n]`.
pub fn softmax_with_sink(logits: &[f32], sink: f32) -> Vec<f32> {
    let m = logits
        .iter()
        .copied()
        .fold(sink, |a, b| if b > a { b } else { a });
    let mut denom = (sink - m).exp();
    let mut out = vec![0f32; logits.len()];
    for (o, &l) in out.iter_mut().zip(logits) {
        let e = (l - m).exp();
        *o = e;
        denom += e;
    }
    let inv = 1.0 / denom;
    for o in out.iter_mut() {
        *o *= inv;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clamped_swiglu_matches_formula() {
        // gate/up interleaved: [g0,u0, g1,u1].
        let limit = 7.0f32;
        let alpha = 1.702f32;
        let gate_up = vec![2.0, 0.5, 100.0, -100.0]; // 2nd pair hits both clamps
        let out = clamped_swiglu(&gate_up, limit, alpha);
        // pair 0: g=2, u=0.5 → glu=2·σ(3.404); out=(1.5)·glu
        let glu0 = 2.0 * (1.0 / (1.0 + (-(alpha * 2.0f32)).exp()));
        assert!((out[0] - 1.5 * glu0).abs() < 1e-5);
        // pair 1: g=100→clamp 7; u=-100→clamp -7 → out=(-7+1)·(7·σ(1.702·7))
        let glu1 = 7.0 * (1.0 / (1.0 + (-(alpha * 7.0f32)).exp()));
        assert!((out[1] - (-6.0) * glu1).abs() < 1e-4);
    }

    #[test]
    fn sink_bleeds_probability_mass() {
        // With a sink logit equal to the max real logit, the real weights must
        // sum to < 1 (mass goes to the sink), and reduce to plain softmax when
        // the sink is −inf.
        let logits = vec![1.0f32, 2.0, 0.5];
        let with = softmax_with_sink(&logits, 2.0);
        let s: f32 = with.iter().sum();
        assert!(s < 1.0 && s > 0.0, "sink must remove mass, got {s}");

        let plain = softmax_with_sink(&logits, f32::NEG_INFINITY);
        let sp: f32 = plain.iter().sum();
        assert!((sp - 1.0).abs() < 1e-6, "no-sink must be a full softmax");
        // Reference plain softmax.
        let m = 2.0f32;
        let mut d = 0.0;
        let e: Vec<f32> = logits
            .iter()
            .map(|l| {
                let v = (l - m).exp();
                d += v;
                v
            })
            .collect();
        for (a, b) in plain.iter().zip(&e) {
            assert!((a - b / d).abs() < 1e-6);
        }
    }

    #[test]
    fn config_parses_gpt_oss_20b_shape() {
        let v = serde_json::json!({
            "model_type": "gpt_oss", "vocab_size": 201088, "hidden_size": 2880,
            "intermediate_size": 2880, "num_hidden_layers": 4,
            "num_attention_heads": 64, "num_key_value_heads": 8, "head_dim": 64,
            "num_local_experts": 32, "experts_per_token": 4, "sliding_window": 128,
            "swiglu_limit": 7.0, "attention_bias": true,
            "layer_types": ["sliding_attention","full_attention","sliding_attention","full_attention"],
        });
        let c = GptOssConfig::from_json(&v).unwrap();
        assert_eq!(c.num_local_experts, 32);
        assert_eq!(c.experts_per_token, 4);
        assert_eq!(c.head_dim, 64);
        assert!(c.attention_bias);
        assert_eq!(c.layer_is_sliding, vec![true, false, true, false]);
    }
}
