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

//! TimesFM-3 configuration (`config.json` from Hugging Face).

use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::Path;

/// Default quantile levels (10th–90th percentile).
pub const DEFAULT_QUANTILES: [f32; 9] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];

/// Max supported context length (matches upstream forecaster).
pub const MAX_CONTEXT_LENGTH: usize = 15_360;

#[derive(Debug, Clone, Deserialize)]
pub struct ResidualBlockConfig {
    pub hidden_dims: usize,
    pub output_dims: usize,
    #[serde(default)]
    pub use_bias: bool,
    #[serde(default = "default_relu")]
    pub activation: String,
    #[serde(default)]
    pub dropout: f32,
    #[serde(default)]
    pub identity_skip: bool,
    #[serde(default = "default_none")]
    pub prenorm: String,
}

fn default_relu() -> String {
    "relu".into()
}
fn default_none() -> String {
    "none".into()
}

#[derive(Debug, Clone, Deserialize)]
pub struct TransformerInnerConfig {
    pub model_dims: usize,
    pub hidden_dims: usize,
    pub num_heads: usize,
    #[serde(default = "default_rms")]
    pub attention_norm: String,
    #[serde(default = "default_rms")]
    pub feedforward_norm: String,
    #[serde(default = "default_rms")]
    pub qk_norm: String,
    #[serde(default)]
    pub use_bias: bool,
    #[serde(default)]
    pub use_rope_seq: bool,
    #[serde(default)]
    pub use_rope_var: bool,
    #[serde(default = "default_relu")]
    pub ff_activation: String,
    #[serde(default)]
    pub deterministic: bool,
    #[serde(default = "default_none")]
    pub v_norm: String,
    #[serde(default)]
    pub causal_attention: bool,
    #[serde(default)]
    pub training: bool,
    #[serde(default = "default_true")]
    pub use_memory_efficient_attention: bool,
    #[serde(default = "default_true")]
    pub use_sdpa: bool,
    #[serde(default = "default_max_variates")]
    pub max_variates: usize,
}

fn default_rms() -> String {
    "rms".into()
}
fn default_true() -> bool {
    true
}
fn default_max_variates() -> usize {
    32
}

#[derive(Debug, Clone, Deserialize)]
pub struct StackedTransformersConfig {
    pub num_layers: usize,
    pub transformer: TransformerInnerConfig,
    #[serde(default = "default_true")]
    pub use_remat: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TimesFM3Config {
    pub input_patch_len: usize,
    pub output_patch_len: usize,
    #[serde(default = "default_quantiles")]
    pub quantiles: Vec<f32>,
    pub residual_block_config: ResidualBlockConfig,
    pub transformer_config: StackedTransformersConfig,
    #[serde(default = "default_true")]
    pub use_variate_attention: bool,
    #[serde(default = "default_value_clip")]
    pub value_clip: f32,
    #[serde(default = "default_true")]
    pub use_stitching: bool,
    #[serde(default = "default_true")]
    pub use_linear_detrending: bool,
    #[serde(default = "default_detrend_threshold")]
    pub linear_detrending_threshold: f32,
    #[serde(default = "default_true")]
    pub use_iterative_cpm_revin: bool,
    #[serde(default)]
    pub use_frozen_running_stats: bool,
    #[serde(default = "default_identity")]
    pub input_transform: String,
}

fn default_quantiles() -> Vec<f32> {
    DEFAULT_QUANTILES.to_vec()
}
fn default_value_clip() -> f32 {
    1e20
}
fn default_detrend_threshold() -> f32 {
    0.5
}
fn default_identity() -> String {
    "identity".into()
}

impl TimesFM3Config {
    /// Official TimesFM-3.0 checkpoint defaults (`google/timesfm-3.0-pytorch`).
    pub fn v3() -> Self {
        serde_json::from_value(serde_json::json!({
            "input_patch_len": 32,
            "output_patch_len": 64,
            "quantiles": DEFAULT_QUANTILES,
            "residual_block_config": {
                "activation": "relu",
                "dropout": 0.0,
                "hidden_dims": 1280,
                "identity_skip": false,
                "output_dims": 1280,
                "prenorm": "none",
                "use_bias": false
            },
            "transformer_config": {
                "num_layers": 20,
                "transformer": {
                    "attention_norm": "rms",
                    "causal_attention": true,
                    "deterministic": true,
                    "feedforward_norm": "rms",
                    "ff_activation": "relu",
                    "hidden_dims": 1280,
                    "max_variates": 32,
                    "model_dims": 1280,
                    "num_heads": 16,
                    "qk_norm": "rms",
                    "training": true,
                    "use_bias": false,
                    "use_memory_efficient_attention": true,
                    "use_rope_seq": true,
                    "use_rope_var": false,
                    "use_sdpa": true,
                    "v_norm": "none"
                },
                "use_remat": true
            },
            "use_frozen_running_stats": false,
            "use_iterative_cpm_revin": true,
            "use_linear_detrending": true,
            "linear_detrending_threshold": 0.5,
            "use_stitching": true,
            "use_variate_attention": true,
            "value_clip": 1e20,
            "input_transform": "identity"
        }))
        .expect("v3 config json")
    }

    /// Tiny config for unit tests (1 layer, 64-dim).
    pub fn synth_tiny() -> Self {
        let mut c = Self::v3();
        c.transformer_config.num_layers = 1;
        c.transformer_config.transformer.model_dims = 64;
        c.transformer_config.transformer.hidden_dims = 64;
        c.transformer_config.transformer.num_heads = 4;
        c.residual_block_config.hidden_dims = 64;
        c.residual_block_config.output_dims = 64;
        c
    }

    pub fn from_file(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("read config {}", path.display()))?;
        serde_json::from_str(&text).context("parse TimesFM3 config.json")
    }

    pub fn from_dir(dir: &Path) -> Result<Self> {
        Self::from_file(&dir.join("config.json"))
    }

    pub fn num_quantiles(&self) -> usize {
        self.quantiles.len()
    }

    pub fn rolls(&self) -> usize {
        self.output_patch_len / self.input_patch_len
    }

    pub fn model_dims(&self) -> usize {
        self.transformer_config.transformer.model_dims
    }

    pub fn num_layers(&self) -> usize {
        self.transformer_config.num_layers
    }

    pub fn num_heads(&self) -> usize {
        self.transformer_config.transformer.num_heads
    }

    pub fn head_dim(&self) -> usize {
        self.model_dims() / self.num_heads()
    }

    pub fn median_quantile_index(&self) -> usize {
        self.num_quantiles() / 2
    }

    pub fn resblock_input_dim(&self) -> usize {
        2 * (self.input_patch_len + self.output_patch_len)
    }
}
