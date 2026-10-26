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

//! `config.json` of a MiniCPM-SALA checkpoint (`model_type = minicpm_sala`).

use anyhow::{Context, Result, anyhow, bail, ensure};
use serde_json::Value;
use std::path::Path;

/// Token mixer of one layer (`mixer_types[i]`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mixer {
    /// `minicpm4`: GQA softmax attention (InfLLM-v2 sparse past `dense_len`).
    Attention,
    /// `lightning-attn`: linear attention with a fixed per-head decay.
    Lightning,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SalaConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    /// KV heads of the attention layers.
    pub num_key_value_heads: usize,
    /// KV heads of the Lightning layers (`lightning_nkv`).
    pub lightning_kv_heads: usize,
    /// Per-head width of the Lightning layers (`head_dim`). The attention
    /// layers use `hidden_size / num_attention_heads`, as the reference does.
    pub head_dim: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f64,
    /// Embeddings are multiplied by this (muP).
    pub scale_emb: f32,
    /// Each residual branch is scaled by `scale_depth / sqrt(num_layers)`.
    pub scale_depth: f32,
    /// The LM head reads `hidden / (hidden_size / dim_model_base)`.
    pub dim_model_base: usize,
    pub mixers: Vec<Mixer>,
    /// `attn_use_output_gate`: attention output × σ(o_gate(x)).
    pub attn_output_gate: bool,
    /// `attn_use_rope` (off in the release: the sparse layers are NoPE).
    pub attn_rope: bool,
    /// Per-head RMSNorm on the attention q / k before RoPE (Qwen3's
    /// `q_norm` / `k_norm`).
    pub attn_qk_norm: bool,
    /// Biases on the attention q / k / v projections (Qwen2).
    pub attn_qkv_bias: bool,
    /// `use_output_gate`: Lightning output × σ(z_proj(x)).
    pub lightning_output_gate: bool,
    /// `use_output_norm`: RMSNorm over the whole Lightning output.
    pub lightning_output_norm: bool,
    /// `qk_norm` on the Lightning q/k (per head).
    pub lightning_qk_norm: bool,
    /// `lightning_use_rope`.
    pub lightning_rope: bool,
    /// `lightning_scale` as a number.
    pub lightning_scale: f32,
    /// Below this many keys InfLLM-v2 is ordinary dense attention.
    pub dense_len: Option<usize>,
    /// InfLLM-v2's block-sparse settings (`sparse_config`).
    pub sparse: Option<crate::sparse::SparseParams>,
    pub bos_token_id: Option<u32>,
    pub eos_token_ids: Vec<u32>,
    /// Prefix of every tensor name (`""` for SALA, `"llm."` for a MiniCPM-V
    /// trunk).
    pub weight_prefix: String,
    /// Head width of the attention layers when it is not
    /// `hidden_size / num_attention_heads` (Llama's `head_dim`).
    pub attn_head_dim_override: Option<usize>,
    /// RoPE frequencies when they are not `theta^(-2i/d)` (LongRoPE).
    pub rope_inv_freq: Option<Vec<f32>>,
}

impl SalaConfig {
    pub fn from_dir(dir: &Path) -> Result<Self> {
        let p = dir.join("config.json");
        let v: Value = serde_json::from_str(
            &std::fs::read_to_string(&p).with_context(|| format!("minicpm-sala: read {p:?}"))?,
        )
        .with_context(|| format!("minicpm-sala: parse {p:?}"))?;
        Self::from_value(&v)
    }

    pub fn from_value(v: &Value) -> Result<Self> {
        let mt = v
            .get("model_type")
            .and_then(Value::as_str)
            .unwrap_or_default();
        if mt != "minicpm_sala" {
            bail!("minicpm-sala: model_type {mt:?} (expected \"minicpm_sala\")");
        }
        let u = |k: &str| v.get(k).and_then(Value::as_u64).map(|n| n as usize);
        let need = |k: &str| u(k).ok_or_else(|| anyhow!("minicpm-sala: config.{k} missing"));
        let f = |k: &str, d: f64| v.get(k).and_then(Value::as_f64).unwrap_or(d);
        let b = |k: &str, d: bool| v.get(k).and_then(Value::as_bool).unwrap_or(d);

        let layers = need("num_hidden_layers")?;
        let mixers = v
            .get("mixer_types")
            .and_then(Value::as_array)
            .ok_or_else(|| anyhow!("minicpm-sala: config.mixer_types missing"))?
            .iter()
            .map(|m| match m.as_str() {
                Some("minicpm4") => Ok(Mixer::Attention),
                Some("lightning" | "lightning_attn" | "lightning-attn") => Ok(Mixer::Lightning),
                other => Err(anyhow!("minicpm-sala: unsupported mixer type {other:?}")),
            })
            .collect::<Result<Vec<_>>>()?;
        ensure!(
            mixers.len() >= layers,
            "minicpm-sala: {} mixer types for {layers} layers",
            mixers.len()
        );
        let heads = need("num_attention_heads")?;
        let hidden = need("hidden_size")?;
        let head_dim = u("head_dim").unwrap_or(hidden / heads);
        let lightning_scale = match v.get("lightning_scale").and_then(Value::as_str) {
            Some("1/sqrt(d)") | None => (head_dim as f32).powf(-0.5),
            Some("1/d") => 1.0 / head_dim as f32,
            Some(_) => 1.0,
        };
        if let Some(rs) = v.get("rope_scaling").filter(|r| !r.is_null()) {
            let kind = rs
                .get("rope_type")
                .or_else(|| rs.get("type"))
                .and_then(Value::as_str)
                .unwrap_or("default");
            if kind != "default" {
                bail!("minicpm-sala: rope_scaling {kind:?} is not implemented");
            }
        }
        let eos_token_ids = match v.get("eos_token_id") {
            Some(Value::Array(a)) => a
                .iter()
                .filter_map(Value::as_u64)
                .map(|x| x as u32)
                .collect(),
            Some(Value::Number(n)) => n.as_u64().map(|x| vec![x as u32]).unwrap_or_default(),
            _ => Vec::new(),
        };
        Ok(Self {
            vocab_size: need("vocab_size")?,
            hidden_size: hidden,
            intermediate_size: need("intermediate_size")?,
            num_attention_heads: heads,
            num_key_value_heads: u("num_key_value_heads").unwrap_or(heads),
            lightning_kv_heads: u("lightning_nkv").unwrap_or(heads),
            head_dim,
            rms_norm_eps: f("rms_norm_eps", 1e-6) as f32,
            rope_theta: f("rope_theta", 10_000.0),
            scale_emb: f("scale_emb", 1.0) as f32,
            scale_depth: f("scale_depth", 1.0) as f32,
            dim_model_base: u("dim_model_base").unwrap_or(hidden),
            mixers: mixers[..layers].to_vec(),
            attn_output_gate: b("attn_use_output_gate", false),
            attn_rope: b("attn_use_rope", true),
            attn_qk_norm: false,
            attn_qkv_bias: false,
            lightning_output_gate: b("use_output_gate", false),
            lightning_output_norm: b("use_output_norm", false),
            lightning_qk_norm: b("qk_norm", true),
            lightning_rope: b("lightning_use_rope", false),
            lightning_scale,
            dense_len: v
                .get("sparse_config")
                .and_then(|s| s.get("dense_len"))
                .and_then(Value::as_u64)
                .map(|n| n as usize),
            sparse: v
                .get("sparse_config")
                .and_then(crate::sparse::SparseParams::from_value),
            bos_token_id: u("bos_token_id").map(|x| x as u32),
            eos_token_ids,
            weight_prefix: String::new(),
            attn_head_dim_override: None,
            rope_inv_freq: None,
        })
    }

    /// A plain Llama trunk (`LlamaConfig` JSON, e.g. the language model of
    /// MiniCPM-V 4.0; set [`Self::attn_qk_norm`] for a Qwen3 one, as in 4.5)
    /// whose tensors sit under `prefix`: every layer softmax
    /// attention with RoPE, no muP scaling. `rope_scaling` may be absent or
    /// `longrope`; at the contexts this runs (below
    /// `original_max_position_embeddings`) LongRoPE divides each frequency by
    /// its `short_factor` and scales attention by 1 when `factor` is 1.
    pub fn from_llama_value(v: &Value, prefix: &str) -> Result<Self> {
        let u = |k: &str| v.get(k).and_then(Value::as_u64).map(|n| n as usize);
        let need = |k: &str| u(k).ok_or_else(|| anyhow!("minicpm-sala: config.{k} missing"));
        let f = |k: &str, d: f64| v.get(k).and_then(Value::as_f64).unwrap_or(d);
        let layers = need("num_hidden_layers")?;
        let heads = need("num_attention_heads")?;
        let hidden = need("hidden_size")?;
        let head_dim = u("head_dim").unwrap_or(hidden / heads);
        let theta = f("rope_theta", 10_000.0);
        let rope_inv_freq = match v.get("rope_scaling").filter(|r| !r.is_null()) {
            None => None,
            Some(rs) => {
                let kind = rs
                    .get("rope_type")
                    .or_else(|| rs.get("type"))
                    .and_then(Value::as_str)
                    .unwrap_or("default");
                match kind {
                    "default" => None,
                    "longrope" => {
                        let factor = rs.get("factor").and_then(Value::as_f64).unwrap_or(1.0);
                        ensure!(
                            factor <= 1.0 && rs.get("attention_factor").is_none(),
                            "minicpm-sala: LongRoPE with factor {factor} scales attention; \
                             not implemented"
                        );
                        let short: Vec<f64> = rs
                            .get("short_factor")
                            .and_then(Value::as_array)
                            .ok_or_else(|| anyhow!("minicpm-sala: longrope short_factor missing"))?
                            .iter()
                            .filter_map(Value::as_f64)
                            .collect();
                        ensure!(
                            short.len() == head_dim / 2,
                            "minicpm-sala: {} short factors for head_dim {head_dim}",
                            short.len()
                        );
                        Some(longrope_inv_freq(theta, head_dim, &short))
                    }
                    other => bail!("minicpm-sala: rope_scaling {other:?} is not implemented"),
                }
            }
        };
        let eos_token_ids = match v.get("eos_token_id") {
            Some(Value::Array(a)) => a
                .iter()
                .filter_map(Value::as_u64)
                .map(|x| x as u32)
                .collect(),
            Some(Value::Number(n)) => n.as_u64().map(|x| vec![x as u32]).unwrap_or_default(),
            _ => Vec::new(),
        };
        Ok(Self {
            vocab_size: need("vocab_size")?,
            hidden_size: hidden,
            intermediate_size: need("intermediate_size")?,
            num_attention_heads: heads,
            num_key_value_heads: u("num_key_value_heads").unwrap_or(heads),
            lightning_kv_heads: heads,
            head_dim,
            rms_norm_eps: f("rms_norm_eps", 1e-6) as f32,
            rope_theta: theta,
            scale_emb: 1.0,
            // residual scale = scale_depth / sqrt(layers) = 1.
            scale_depth: (layers as f32).sqrt(),
            dim_model_base: hidden,
            mixers: vec![Mixer::Attention; layers],
            attn_output_gate: false,
            attn_rope: true,
            attn_qk_norm: false,
            attn_qkv_bias: v
                .get("attention_bias")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            lightning_output_gate: false,
            lightning_output_norm: false,
            lightning_qk_norm: false,
            lightning_rope: false,
            lightning_scale: 1.0,
            dense_len: None,
            sparse: None,
            bos_token_id: u("bos_token_id").map(|x| x as u32),
            eos_token_ids,
            weight_prefix: prefix.to_string(),
            attn_head_dim_override: Some(head_dim),
            rope_inv_freq,
        })
    }

    pub fn num_layers(&self) -> usize {
        self.mixers.len()
    }

    /// Head width of the attention layers (`hidden_size / num_attention_heads`
    /// unless the config names one).
    pub fn attn_head_dim(&self) -> usize {
        self.attn_head_dim_override
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    /// `scale_depth / sqrt(num_hidden_layers)`.
    pub fn residual_scale(&self) -> f32 {
        self.scale_depth / (self.num_layers() as f32).sqrt()
    }

    /// Full tensor name of `name` (`model.norm.weight`, …).
    pub fn weight(&self, name: &str) -> String {
        format!("{}{name}", self.weight_prefix)
    }

    /// cos / sin tables `[positions, head_dim / 2]` for the RoPE layers.
    pub fn rope_tables(&self, positions: &[usize]) -> (Vec<f32>, Vec<f32>) {
        match &self.rope_inv_freq {
            None => rlx_fish::nn::rope_tables(positions, self.head_dim, self.rope_theta, false),
            Some(freqs) => {
                let mut cos = Vec::with_capacity(positions.len() * freqs.len());
                let mut sin = Vec::with_capacity(positions.len() * freqs.len());
                for &p in positions {
                    for &f in freqs {
                        let a = p as f32 * f;
                        cos.push(a.cos());
                        sin.push(a.sin());
                    }
                }
                (cos, sin)
            }
        }
    }

    /// Keep only the first `n` layers (for layer-subset tests). Like the
    /// reference with `num_hidden_layers = n`, the residual scale follows.
    pub fn truncate(&mut self, n: usize) {
        self.mixers.truncate(n);
    }

    /// Per-head log decay of the Lightning layers: `-slope_h`, with the
    /// ALiBi slopes of `_build_slope_tensor`, rounded to f32 as the
    /// reference's `torch.tensor(...)` does.
    pub fn lightning_log_decay(&self) -> Vec<f32> {
        alibi_slopes(self.num_attention_heads)
            .into_iter()
            .map(|s| -(s as f32))
            .collect()
    }
}

/// LongRoPE's short-context frequencies as HF computes them in f32:
/// `1 / (short_factor_i · theta^(2i/d))`.
pub fn longrope_inv_freq(theta: f64, head_dim: usize, short_factor: &[f64]) -> Vec<f32> {
    let base = theta as f32;
    short_factor
        .iter()
        .enumerate()
        .map(|(i, &s)| {
            let expo = (2 * i) as f32 / head_dim as f32;
            1.0 / (s as f32 * base.powf(expo))
        })
        .collect()
}

/// `_build_slope_tensor` (ALiBi): geometric slopes for a power-of-two head
/// count, interleaved extras otherwise.
pub fn alibi_slopes(n: usize) -> Vec<f64> {
    fn pow2(n: usize) -> Vec<f64> {
        let start = 2f64.powf(-(2f64.powf(-((n as f64).log2() - 3.0))));
        (0..n).map(|i| start * start.powi(i as i32)).collect()
    }
    let l = (n as f64).log2();
    if l.fract() == 0.0 {
        pow2(n)
    } else {
        let c = 2usize.pow(l.floor() as u32);
        let mut s = pow2(c);
        s.extend(alibi_slopes(2 * c).into_iter().step_by(2).take(n - c));
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slopes_match_reference_formula() {
        // n = 32: start = 2^(-2^-(5-3)) = 2^-0.25, ratio = start.
        let s = alibi_slopes(32);
        assert!((s[0] - 2f64.powf(-0.25)).abs() < 1e-15);
        assert!((s[31] - 2f64.powf(-8.0)).abs() < 1e-12);
        // Non power of two: 6 heads = 4 geometric + every other of 8.
        let s6 = alibi_slopes(6);
        let s8 = alibi_slopes(8);
        assert_eq!(s6[4], s8[0]);
        assert_eq!(s6[5], s8[2]);
    }

    #[test]
    fn parses_release_config() {
        let v: Value = serde_json::from_str(include_str!("../tests/fixtures/config.json")).unwrap();
        let c = SalaConfig::from_value(&v).unwrap();
        assert_eq!(c.num_layers(), 32);
        assert_eq!(
            c.mixers.iter().filter(|m| **m == Mixer::Attention).count(),
            8
        );
        assert_eq!(c.attn_head_dim(), 128);
        assert_eq!(c.dense_len, Some(8192));
        assert!(!c.attn_rope && c.lightning_rope && c.attn_output_gate);
        assert!((c.residual_scale() - 1.4 / 32f32.sqrt()).abs() < 1e-7);
        assert_eq!(c.eos_token_ids, vec![2, 73440]);
    }
}
