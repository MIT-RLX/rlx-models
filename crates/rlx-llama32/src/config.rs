// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// LLaMA-3.2 configuration — HF `config.json` and GGUF `llama.*` metadata.

use rlx_gguf::{GgufFile, MetaValue};
use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Llama32RopeType {
    #[default]
    Default,
    #[serde(rename = "llama3")]
    Llama3,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Llama32RopeScaling {
    pub factor: f32,
    #[serde(default = "default_low_freq_factor")]
    pub low_freq_factor: f32,
    #[serde(default = "default_high_freq_factor")]
    pub high_freq_factor: f32,
    pub original_max_position_embeddings: usize,
    #[serde(default)]
    pub rope_type: Llama32RopeType,
}

fn default_low_freq_factor() -> f32 {
    1.0
}
fn default_high_freq_factor() -> f32 {
    4.0
}

#[derive(Debug, Clone, Deserialize)]
pub struct Llama32Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,

    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default = "default_hidden_act")]
    pub hidden_act: String,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub attention_bias: bool,
    /// Explicit head dim (Llama 3.x); when absent, derived from hidden/heads.
    #[serde(default)]
    pub head_dim: Option<usize>,
    #[serde(default)]
    pub rope_scaling: Option<Llama32RopeScaling>,
}

fn default_rms_norm_eps() -> f64 {
    1e-5
}
fn default_rope_theta() -> f64 {
    500_000.0
}
fn default_hidden_act() -> String {
    "silu".into()
}

impl Llama32Config {
    pub fn from_file(path: &Path) -> anyhow::Result<Self> {
        let data = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&data)?)
    }

    pub fn from_gguf(raw: &GgufFile) -> anyhow::Result<Self> {
        llama32_cfg_from_gguf(raw)
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    pub fn kv_group_size(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }

    pub fn q_proj_dim(&self) -> usize {
        self.num_attention_heads * self.head_dim()
    }

    pub fn kv_proj_dim(&self) -> usize {
        self.num_key_value_heads * self.head_dim()
    }

    #[cfg(test)]
    pub(crate) fn tiny_test() -> Self {
        Self {
            vocab_size: 32,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            max_position_embeddings: 16,
            rms_norm_eps: 1e-5,
            rope_theta: 500_000.0,
            hidden_act: "silu".into(),
            tie_word_embeddings: false,
            attention_bias: false,
            head_dim: None,
            rope_scaling: None,
        }
    }
}

pub fn llama32_cfg_from_gguf(raw: &GgufFile) -> anyhow::Result<Llama32Config> {
    let arch_prefix = raw
        .metadata
        .get("general.architecture")
        .and_then(MetaValue::as_str)
        .unwrap_or("llama");
    let get_meta = |k: &str| -> Option<&MetaValue> {
        raw.metadata.get(k).or_else(|| {
            let suffix = k.strip_prefix("llama.")?;
            if arch_prefix == "llama" {
                None
            } else {
                let arch_key = format!("{arch_prefix}.{suffix}");
                raw.metadata.get(&arch_key)
            }
        })
    };
    let get_u32 = |k: &str| -> anyhow::Result<u32> {
        get_meta(k)
            .and_then(MetaValue::as_u32)
            .ok_or_else(|| anyhow::anyhow!("missing GGUF metadata key: {k}"))
    };
    let get_f32 = |k: &str| -> Option<f32> {
        get_meta(k).and_then(|v| match v {
            MetaValue::F32(x) => Some(*x),
            _ => None,
        })
    };
    let get_bool = |k: &str| -> Option<bool> {
        get_meta(k).and_then(|v| match v {
            MetaValue::Bool(b) => Some(*b),
            _ => None,
        })
    };

    let hidden_size = get_u32("llama.embedding_length")? as usize;
    let num_attention_heads = get_u32("llama.attention.head_count")? as usize;
    let head_dim = get_u32("llama.attention.key_length")
        .ok()
        .or_else(|| get_u32("llama.rope.dimension_count").ok())
        .map(|v| v as usize);

    let rope_scaling = match get_meta("llama.rope.scaling.type").and_then(MetaValue::as_str) {
        Some("none") | None => {
            // Llama 3.x often bakes scaling into rope_freqs.weight; HF fields may be absent.
            None
        }
        Some("linear") | Some("yarn") | Some("longrope") => {
            let factor = get_f32("llama.rope.scaling.factor")
                .or_else(|| get_f32("llama.rope.scale_linear"))
                .unwrap_or(1.0);
            let original = get_u32("llama.rope.scaling.original_context_length")
                .map(|v| v as usize)
                .unwrap_or(8192);
            Some(Llama32RopeScaling {
                factor,
                low_freq_factor: 1.0,
                high_freq_factor: 4.0,
                original_max_position_embeddings: original,
                rope_type: Llama32RopeType::Llama3,
            })
        }
        other => {
            return Err(anyhow::anyhow!(
                "unsupported llama.rope.scaling.type: {other:?}"
            ));
        }
    };

    Ok(Llama32Config {
        vocab_size: get_u32("llama.vocab_size").unwrap_or(128_256) as usize,
        hidden_size,
        intermediate_size: get_u32("llama.feed_forward_length")? as usize,
        num_hidden_layers: get_u32("llama.block_count")? as usize,
        num_attention_heads,
        num_key_value_heads: get_u32("llama.attention.head_count_kv")? as usize,
        max_position_embeddings: get_u32("llama.context_length").unwrap_or(8192) as usize,
        rms_norm_eps: get_f32("llama.attention.layer_norm_rms_epsilon").unwrap_or(1e-5) as f64,
        rope_theta: get_f32("llama.rope.freq_base").unwrap_or(500_000.0) as f64,
        hidden_act: "silu".into(),
        tie_word_embeddings: get_bool("llama.tie_word_embeddings").unwrap_or(true),
        attention_bias: false,
        head_dim,
        rope_scaling,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_llama32_1b_like() {
        let json = r#"{
            "vocab_size": 128256,
            "hidden_size": 2048,
            "intermediate_size": 8192,
            "num_hidden_layers": 16,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "max_position_embeddings": 131072,
            "rope_theta": 500000.0,
            "rms_norm_eps": 1e-05,
            "tie_word_embeddings": true,
            "rope_scaling": {
                "factor": 32.0,
                "high_freq_factor": 4.0,
                "low_freq_factor": 1.0,
                "original_max_position_embeddings": 8192,
                "rope_type": "llama3"
            }
        }"#;
        let cfg: Llama32Config = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.head_dim(), 64);
        assert_eq!(cfg.kv_group_size(), 4);
        assert!(cfg.rope_scaling.is_some());
    }
}
