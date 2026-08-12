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

//! DFlash drafter hyper-parameters, read from a `general.architecture =
//! "dflash"` GGUF.

use anyhow::{Context, Result, bail};
use rlx_gguf::{GgufFile, MetaValue};

/// DFlash draft head.
///
/// Eagle-style: it has **no token embedding and no LM head of its own** — it
/// consumes the TARGET model's intermediate residual streams and reuses the
/// target's `lm_head` to score its proposals. `fc` fuses the taps:
/// `[n_taps * hidden, hidden]`.
#[derive(Debug, Clone)]
pub struct DflashConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub max_position_embeddings: usize,
    /// Tokens proposed per draft step (`dflash.block_size`).
    pub block_size: usize,
    /// Which TARGET layers feed `fc` (`dflash.target_layers`), e.g.
    /// `[2, 14, 26, 38, 50]` for Muse-Glimmer-30B's 52 layers.
    pub target_layers: Vec<usize>,
    /// Sliding-window width; DFlash marks every layer local.
    pub sliding_window: Option<usize>,
}

fn u32_at(raw: &GgufFile, key: &str) -> Option<u32> {
    raw.metadata.get(key).and_then(MetaValue::as_u32)
}

fn f32_at(raw: &GgufFile, key: &str) -> Option<f32> {
    raw.metadata.get(key).and_then(|v| match v {
        MetaValue::F32(x) => Some(*x),
        _ => None,
    })
}

impl DflashConfig {
    pub fn from_gguf(raw: &GgufFile) -> Result<Self> {
        let arch = raw
            .metadata
            .get("general.architecture")
            .and_then(MetaValue::as_str)
            .unwrap_or_default();
        if arch != "dflash" {
            bail!("DflashConfig::from_gguf expected general.architecture=\"dflash\", got {arch:?}");
        }
        let hidden_size =
            u32_at(raw, "dflash.embedding_length").context("dflash.embedding_length")? as usize;
        let num_attention_heads = u32_at(raw, "dflash.attention.head_count")
            .context("dflash.attention.head_count")? as usize;
        let head_dim = u32_at(raw, "dflash.attention.key_length")
            .map(|v| v as usize)
            .unwrap_or_else(|| hidden_size / num_attention_heads.max(1));

        // `target_layers` is the whole point of the arch: without it there is
        // nothing to fuse, so treat a missing/empty array as a hard error rather
        // than silently drafting from noise.
        let target_layers: Vec<usize> = match raw.metadata.get("dflash.target_layers") {
            Some(MetaValue::Array(a)) => a
                .iter()
                .filter_map(MetaValue::as_u32)
                .map(|v| v as usize)
                .collect(),
            _ => Vec::new(),
        };
        if target_layers.is_empty() {
            bail!("dflash GGUF is missing `dflash.target_layers` — nothing to fuse");
        }

        Ok(Self {
            hidden_size,
            intermediate_size: u32_at(raw, "dflash.feed_forward_length")
                .context("dflash.feed_forward_length")? as usize,
            num_hidden_layers: u32_at(raw, "dflash.block_count").context("dflash.block_count")?
                as usize,
            num_attention_heads,
            num_key_value_heads: u32_at(raw, "dflash.attention.head_count_kv")
                .context("dflash.attention.head_count_kv")?
                as usize,
            head_dim,
            rms_norm_eps: f32_at(raw, "dflash.attention.layer_norm_rms_epsilon").unwrap_or(1e-5)
                as f64,
            rope_theta: f32_at(raw, "dflash.rope.freq_base").unwrap_or(500_000.0) as f64,
            max_position_embeddings: u32_at(raw, "dflash.context_length").unwrap_or(8192) as usize,
            block_size: u32_at(raw, "dflash.block_size").unwrap_or(16) as usize,
            target_layers,
            sliding_window: u32_at(raw, "dflash.attention.sliding_window").map(|v| v as usize),
        })
    }

    pub fn q_proj_dim(&self) -> usize {
        self.num_attention_heads * self.head_dim
    }

    pub fn kv_proj_dim(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }

    pub fn kv_group_size(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads.max(1)
    }

    /// Width of the concatenated tap vector `fc` consumes.
    pub fn fused_input_dim(&self) -> usize {
        self.target_layers.len() * self.hidden_size
    }
}
