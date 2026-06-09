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

//! MiniMax M2 configuration — parsed from `minimax-m2.*` GGUF metadata
//! keys or from a HuggingFace `config.json`.
//!
//! MiniMax M2/M2.5/M2.7 use **Lightning Attention** layers (linear
//! attention with a per-head log-decay state). Per-layer dims:
//!   * `q, k, v` projections: `[hidden] → [num_heads * head_dim]`
//!   * `gate, beta` projections: `[hidden] → [num_heads]` per token
//!   * `o_proj`: `[num_heads * head_dim] → [hidden]`
//!   * SwiGLU FFN with `intermediate_size` inner width

use anyhow::{Result, anyhow};
use rlx_gguf::{GgufFile, MetaValue};
use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct MiniMaxConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    /// Per-head value/key dim (Lightning state matrix is `[n, n]`).
    pub head_dim: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub tie_word_embeddings: bool,
}

impl MiniMaxConfig {
    pub fn lightning_state_bytes_per_layer(&self) -> usize {
        // f32 state buffer per layer: batch=1, [h, n, n].
        4 * self.num_attention_heads * self.head_dim * self.head_dim
    }

    pub fn from_gguf(raw: &GgufFile) -> Result<Self> {
        let arch = raw
            .metadata
            .get("general.architecture")
            .and_then(MetaValue::as_str)
            .ok_or_else(|| anyhow!("missing general.architecture"))?
            .to_string();
        let prefix = match arch.as_str() {
            "minimax-m2" | "minimax_m2" | "minimax" => arch.replace('-', "_"),
            other => {
                return Err(anyhow!(
                    "MiniMaxConfig::from_gguf: unsupported arch `{other}`"
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
        let num_attention_heads = u("attention.head_count")? as usize;
        let head_dim = u_opt("attention.key_length")
            .map(|v| v as usize)
            .unwrap_or(hidden_size / num_attention_heads);

        Ok(Self {
            vocab_size: u_opt("vocab_size").unwrap_or(64_000) as usize,
            hidden_size,
            intermediate_size: u("feed_forward_length")? as usize,
            num_hidden_layers: u("block_count")? as usize,
            num_attention_heads,
            head_dim,
            max_position_embeddings: u_opt("context_length").unwrap_or(8192) as usize,
            rms_norm_eps: f_opt("attention.layer_norm_rms_epsilon").unwrap_or(1e-5) as f64,
            tie_word_embeddings: b_opt("tie_word_embeddings").unwrap_or(false),
        })
    }
}

#[derive(Debug, Deserialize)]
struct HfMiniMax {
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    #[serde(default)]
    head_dim: Option<usize>,
    #[serde(default = "default_max_pos")]
    max_position_embeddings: usize,
    #[serde(default = "default_eps")]
    rms_norm_eps: f64,
    #[serde(default)]
    tie_word_embeddings: bool,
}

fn default_max_pos() -> usize {
    8192
}
fn default_eps() -> f64 {
    1e-5
}

impl MiniMaxConfig {
    pub fn from_hf_config_json(path: &Path) -> Result<Self> {
        let raw =
            std::fs::read_to_string(path).map_err(|e| anyhow!("minimax: read {path:?}: {e}"))?;
        let cfg: HfMiniMax =
            serde_json::from_str(&raw).map_err(|e| anyhow!("minimax: parse {path:?}: {e}"))?;
        let head_dim = cfg
            .head_dim
            .unwrap_or(cfg.hidden_size / cfg.num_attention_heads);
        Ok(Self {
            vocab_size: cfg.vocab_size,
            hidden_size: cfg.hidden_size,
            intermediate_size: cfg.intermediate_size,
            num_hidden_layers: cfg.num_hidden_layers,
            num_attention_heads: cfg.num_attention_heads,
            head_dim,
            max_position_embeddings: cfg.max_position_embeddings,
            rms_norm_eps: cfg.rms_norm_eps,
            tie_word_embeddings: cfg.tie_word_embeddings,
        })
    }
}
