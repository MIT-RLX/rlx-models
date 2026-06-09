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

//! Nemotron-H hybrid Mamba+attention configuration.
//!
//! The Nemotron-H architecture interleaves Mamba2 SSM layers with
//! standard GQA attention layers. The per-layer choice is encoded in
//! `nemotron_h.layer_kinds` (an array of 0/1 per layer where 0=Mamba2
//! and 1=attention) or derived from a periodic ratio
//! (`nemotron_h.attn_layer_period`).

use anyhow::{Result, anyhow};
use rlx_gguf::{GgufFile, MetaValue};
use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NemotronLayerKind {
    Mamba2,
    Attention,
}

#[derive(Debug, Clone)]
pub struct NemotronHybridConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    /// Mamba state size per head.
    pub mamba2_state_size: usize,
    /// Mamba head count (number of independent state-channels). Often
    /// equal to `num_attention_heads` in Nemotron-H but kept separate
    /// because some variants decouple the two.
    pub mamba2_num_heads: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub tie_word_embeddings: bool,
    /// Per-layer kind (length = num_hidden_layers).
    pub layer_kinds: Vec<NemotronLayerKind>,
}

impl NemotronHybridConfig {
    pub fn mamba2_state_bytes_per_layer(&self) -> usize {
        4 * self.mamba2_num_heads * self.mamba2_state_size
    }

    /// Default per-layer kinds when an explicit array is absent —
    /// every `attn_layer_period`-th layer is attention, rest Mamba2.
    pub fn periodic_layer_kinds(num_layers: usize, attn_period: usize) -> Vec<NemotronLayerKind> {
        (0..num_layers)
            .map(|i| {
                if attn_period > 0 && (i + 1) % attn_period == 0 {
                    NemotronLayerKind::Attention
                } else {
                    NemotronLayerKind::Mamba2
                }
            })
            .collect()
    }

    pub fn from_gguf(raw: &GgufFile) -> Result<Self> {
        let arch = raw
            .metadata
            .get("general.architecture")
            .and_then(MetaValue::as_str)
            .ok_or_else(|| anyhow!("missing general.architecture"))?
            .to_string();
        let prefix = match arch.as_str() {
            "nemotron_h" | "nemotron_h_moe" | "nemotron-h" => arch.replace('-', "_"),
            other => {
                return Err(anyhow!(
                    "NemotronHybridConfig::from_gguf: unsupported arch `{other}`"
                ));
            }
        };
        let get = |k: &str| raw.metadata.get(&format!("{prefix}.{k}"));
        let u = |k: &str| -> Result<u32> {
            get(k)
                .and_then(MetaValue::as_u32)
                .ok_or_else(|| anyhow!("missing {prefix}.{k}"))
        };
        let u_opt = |k: &str| get(k).and_then(MetaValue::as_u32);
        let f_opt = |k: &str| {
            get(k).and_then(|v| match v {
                MetaValue::F32(x) => Some(*x),
                _ => None,
            })
        };
        let b_opt = |k: &str| {
            get(k).and_then(|v| match v {
                MetaValue::Bool(x) => Some(*x),
                _ => None,
            })
        };

        let hidden_size = u("embedding_length")? as usize;
        let num_hidden_layers = u("block_count")? as usize;
        let num_attention_heads = u("attention.head_count")? as usize;
        let head_dim = u_opt("attention.key_length")
            .map(|v| v as usize)
            .unwrap_or(hidden_size / num_attention_heads);
        let attn_period = u_opt("attn_layer_period").unwrap_or(4) as usize;
        let layer_kinds = match get("layer_kinds") {
            Some(MetaValue::Array(a)) => a
                .iter()
                .filter_map(|x| x.as_u32())
                .map(|v| {
                    if v == 1 {
                        NemotronLayerKind::Attention
                    } else {
                        NemotronLayerKind::Mamba2
                    }
                })
                .collect::<Vec<_>>(),
            _ => Self::periodic_layer_kinds(num_hidden_layers, attn_period),
        };

        Ok(Self {
            vocab_size: u_opt("vocab_size").unwrap_or(128_000) as usize,
            hidden_size,
            intermediate_size: u("feed_forward_length")? as usize,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads: u_opt("attention.head_count_kv")
                .map(|v| v as usize)
                .unwrap_or(num_attention_heads),
            head_dim,
            mamba2_state_size: u_opt("ssm.state_size").unwrap_or(16) as usize,
            mamba2_num_heads: u_opt("ssm.head_count")
                .map(|v| v as usize)
                .unwrap_or(num_attention_heads),
            max_position_embeddings: u_opt("context_length").unwrap_or(8192) as usize,
            rms_norm_eps: f_opt("attention.layer_norm_rms_epsilon").unwrap_or(1e-5) as f64,
            rope_theta: f_opt("rope.freq_base").unwrap_or(500_000.0) as f64,
            tie_word_embeddings: b_opt("tie_word_embeddings").unwrap_or(false),
            layer_kinds,
        })
    }
}

#[derive(Debug, Deserialize)]
struct HfNemotron {
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    #[serde(default)]
    num_key_value_heads: Option<usize>,
    #[serde(default)]
    head_dim: Option<usize>,
    #[serde(default = "default_state_size")]
    mamba_state_size: usize,
    #[serde(default = "default_attn_period")]
    attn_layer_period: usize,
    #[serde(default = "default_max_pos")]
    max_position_embeddings: usize,
    #[serde(default = "default_eps")]
    rms_norm_eps: f64,
    #[serde(default = "default_rope")]
    rope_theta: f64,
    #[serde(default)]
    tie_word_embeddings: bool,
}

fn default_state_size() -> usize {
    16
}
fn default_attn_period() -> usize {
    4
}
fn default_max_pos() -> usize {
    8192
}
fn default_eps() -> f64 {
    1e-5
}
fn default_rope() -> f64 {
    500_000.0
}

impl NemotronHybridConfig {
    pub fn from_hf_config_json(path: &Path) -> Result<Self> {
        let raw =
            std::fs::read_to_string(path).map_err(|e| anyhow!("nemotron-h: read {path:?}: {e}"))?;
        let cfg: HfNemotron =
            serde_json::from_str(&raw).map_err(|e| anyhow!("nemotron-h: parse {path:?}: {e}"))?;
        let head_dim = cfg
            .head_dim
            .unwrap_or(cfg.hidden_size / cfg.num_attention_heads);
        let layer_kinds = Self::periodic_layer_kinds(cfg.num_hidden_layers, cfg.attn_layer_period);
        Ok(Self {
            vocab_size: cfg.vocab_size,
            hidden_size: cfg.hidden_size,
            intermediate_size: cfg.intermediate_size,
            num_hidden_layers: cfg.num_hidden_layers,
            num_attention_heads: cfg.num_attention_heads,
            num_key_value_heads: cfg.num_key_value_heads.unwrap_or(cfg.num_attention_heads),
            head_dim,
            mamba2_state_size: cfg.mamba_state_size,
            mamba2_num_heads: cfg.num_attention_heads,
            max_position_embeddings: cfg.max_position_embeddings,
            rms_norm_eps: cfg.rms_norm_eps,
            rope_theta: cfg.rope_theta,
            tie_word_embeddings: cfg.tie_word_embeddings,
            layer_kinds,
        })
    }
}
