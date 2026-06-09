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

//! LFM2.5 configuration — GGUF + HF config parsing.

use anyhow::{Result, anyhow};
use rlx_gguf::{GgufFile, MetaValue};
use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct LfmConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    /// Per-channel state size for the LFM SSM block.
    pub ssm_state_size: usize,
    /// Number of SSM input channels (`c`). Often == hidden_size; some
    /// variants project hidden → `c` via a linear before the SSM.
    pub ssm_channels: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub tie_word_embeddings: bool,
}

impl LfmConfig {
    pub fn lfm_state_bytes_per_layer(&self) -> usize {
        // f32 per-layer state buffer: batch=1, [c, n].
        4 * self.ssm_channels * self.ssm_state_size
    }

    pub fn from_gguf(raw: &GgufFile) -> Result<Self> {
        let arch = raw
            .metadata
            .get("general.architecture")
            .and_then(MetaValue::as_str)
            .ok_or_else(|| anyhow!("missing general.architecture"))?
            .to_string();
        let prefix = match arch.as_str() {
            "lfm2" | "lfm" | "lfm25" | "lfm2_5" | "lfm2moe" => arch.replace('-', "_"),
            other => return Err(anyhow!("LfmConfig::from_gguf: unsupported arch `{other}`")),
        };
        // LFM2.5-1.2B and friends use the "ShortConv" block (depthwise
        // causal conv1d + gated MLP) instead of the SSM block this
        // runner currently implements. Detect by tensor presence and
        // bail with a clear, actionable error so the next debugger
        // doesn't waste a cycle on shape mismatches inside the graph.
        let has_shortconv = raw.tensors.keys().any(|k| k.contains(".shortconv."));
        let has_ssm = raw.tensors.keys().any(|k| k.contains(".ssm_"));
        if has_shortconv && !has_ssm {
            return Err(anyhow!(
                "rlx-lfm: GGUF {arch:?} uses the ShortConv block variant \
                 (tensors like `blk.*.shortconv.{{conv,in_proj,out_proj}}.weight`). \
                 The current `LfmRunner` only implements the SSM variant. \
                 PLAN.md M5 follow-up: add `lfm_shortconv_layer_plugin`. \
                 See ShortConv config keys: `{prefix}.shortconv.l_cache` (kernel size)"
            ));
        }
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
        Ok(Self {
            vocab_size: u_opt("vocab_size").unwrap_or(65_536) as usize,
            hidden_size,
            intermediate_size: u("feed_forward_length")? as usize,
            num_hidden_layers: u("block_count")? as usize,
            ssm_state_size: u_opt("ssm.state_size").unwrap_or(16) as usize,
            ssm_channels: u_opt("ssm.inner_size").unwrap_or(hidden_size as u32) as usize,
            max_position_embeddings: u_opt("context_length").unwrap_or(8192) as usize,
            rms_norm_eps: f_opt("attention.layer_norm_rms_epsilon").unwrap_or(1e-5) as f64,
            tie_word_embeddings: b_opt("tie_word_embeddings").unwrap_or(true),
        })
    }
}

#[derive(Debug, Deserialize)]
struct HfLfm {
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    #[serde(default = "default_state_size")]
    ssm_state_size: usize,
    #[serde(default)]
    ssm_inner_size: Option<usize>,
    #[serde(default = "default_max_pos")]
    max_position_embeddings: usize,
    #[serde(default = "default_eps")]
    rms_norm_eps: f64,
    #[serde(default = "default_true")]
    tie_word_embeddings: bool,
}

fn default_state_size() -> usize {
    16
}
fn default_max_pos() -> usize {
    8192
}
fn default_eps() -> f64 {
    1e-5
}
fn default_true() -> bool {
    true
}

impl LfmConfig {
    pub fn from_hf_config_json(path: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(path).map_err(|e| anyhow!("lfm: read {path:?}: {e}"))?;
        let cfg: HfLfm =
            serde_json::from_str(&raw).map_err(|e| anyhow!("lfm: parse {path:?}: {e}"))?;
        let ssm_channels = cfg.ssm_inner_size.unwrap_or(cfg.hidden_size);
        Ok(Self {
            vocab_size: cfg.vocab_size,
            hidden_size: cfg.hidden_size,
            intermediate_size: cfg.intermediate_size,
            num_hidden_layers: cfg.num_hidden_layers,
            ssm_state_size: cfg.ssm_state_size,
            ssm_channels,
            max_position_embeddings: cfg.max_position_embeddings,
            rms_norm_eps: cfg.rms_norm_eps,
            tie_word_embeddings: cfg.tie_word_embeddings,
        })
    }
}
