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

//! LocateAnything configuration — HuggingFace `config.json` (`nvidia/LocateAnything-3B`).

use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use std::path::Path;

/// MoonViT-SO-400M vision tower (`vision_config` in HF JSON).
#[derive(Debug, Clone, Deserialize)]
pub struct MoonVitConfig {
    pub model_type: String,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub patch_size: usize,
    pub merge_kernel_size: [usize; 2],
    pub init_pos_emb_height: usize,
    pub init_pos_emb_width: usize,
}

impl MoonVitConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    pub fn merge_area(&self) -> usize {
        self.merge_kernel_size[0] * self.merge_kernel_size[1]
    }
}

/// Qwen2.5-3B text trunk (`text_config` in HF JSON).
#[derive(Debug, Clone, Deserialize)]
pub struct LocateAnythingTextConfig {
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
    /// MTP / parallel box block size (HF default 6).
    #[serde(default = "default_block_size")]
    pub block_size: usize,
    #[serde(default)]
    pub causal_attn: bool,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    #[serde(default)]
    pub null_token_id: Option<u32>,
    #[serde(default)]
    pub switch_token_id: Option<u32>,
    #[serde(default)]
    pub text_mask_token_id: Option<u32>,
}

fn default_rms_norm_eps() -> f64 {
    1e-6
}
fn default_rope_theta() -> f64 {
    1_000_000.0
}
fn default_hidden_act() -> String {
    "silu".into()
}
fn default_block_size() -> usize {
    6
}

impl LocateAnythingTextConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    pub fn kv_group_size(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }

    /// Map HF text trunk to [`rlx_qwen3::Qwen3Config`] (Qwen2.5 — biases on, no QK-norm).
    pub fn to_qwen3_config(&self) -> rlx_qwen3::Qwen3Config {
        rlx_qwen3::Qwen3Config {
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads,
            head_dim: self.head_dim(),
            max_position_embeddings: self.max_position_embeddings,
            rms_norm_eps: self.rms_norm_eps,
            rope_theta: self.rope_theta,
            hidden_act: self.hidden_act.clone(),
            tie_word_embeddings: self.tie_word_embeddings,
            attention_bias: true,
            qk_norm: false,
            sliding_window: None,
            max_window_layers: usize::MAX,
            use_sliding_window: false,
            num_experts: 0,
            num_experts_used: 0,
            expert_ffn_size: 0,
            shared_expert_ffn_size: 0,
            expert_weights_scale: 1.0,
        }
    }
}

/// `preprocessor_config.json` — native-resolution rescale limits (HF image processor).
#[derive(Debug, Clone, Deserialize)]
pub struct LocateAnythingPreprocessorConfig {
    #[serde(default = "default_in_token_limit")]
    pub in_token_limit: usize,
    #[serde(default = "default_image_mean")]
    pub image_mean: [f32; 3],
    #[serde(default = "default_image_std")]
    pub image_std: [f32; 3],
}

fn default_in_token_limit() -> usize {
    25_600
}

fn default_image_mean() -> [f32; 3] {
    [0.5, 0.5, 0.5]
}

fn default_image_std() -> [f32; 3] {
    [0.5, 0.5, 0.5]
}

impl LocateAnythingPreprocessorConfig {
    pub fn from_file(path: &Path) -> Result<Self> {
        let data = std::fs::read_to_string(path)
            .with_context(|| format!("read preprocessor config {path:?}"))?;
        serde_json::from_str(&data).with_context(|| format!("parse preprocessor config {path:?}"))
    }
}

/// Top-level LocateAnything checkpoint config.
#[derive(Debug, Clone, Deserialize)]
pub struct LocateAnythingConfig {
    pub model_type: String,
    pub image_token_index: u32,
    pub box_start_token_id: u32,
    pub box_end_token_id: u32,
    pub coord_start_token_id: u32,
    pub coord_end_token_id: u32,
    pub ref_start_token_id: u32,
    pub ref_end_token_id: u32,
    pub none_token_id: u32,
    #[serde(default = "default_mlp_connector_layers")]
    pub mlp_connector_layers: usize,
    #[serde(default)]
    pub mlp_checkpoint: bool,
    pub text_config: LocateAnythingTextConfig,
    pub vision_config: MoonVitConfig,
    /// Loaded from `preprocessor_config.json` via [`Self::from_model_dir`].
    #[serde(skip)]
    pub preprocessor: LocateAnythingPreprocessorConfig,
}

fn default_mlp_connector_layers() -> usize {
    2
}

impl Default for LocateAnythingPreprocessorConfig {
    fn default() -> Self {
        Self {
            in_token_limit: default_in_token_limit(),
            image_mean: default_image_mean(),
            image_std: default_image_std(),
        }
    }
}

impl LocateAnythingConfig {
    pub const HF_MODEL_ID: &'static str = "nvidia/LocateAnything-3B";

    pub fn from_file(path: &Path) -> Result<Self> {
        let data = std::fs::read_to_string(path)
            .with_context(|| format!("read LocateAnything config {path:?}"))?;
        let mut cfg: Self = serde_json::from_str(&data)
            .with_context(|| format!("parse LocateAnything config {path:?}"))?;
        let dir = path.parent().unwrap_or(Path::new("."));
        cfg.preprocessor =
            LocateAnythingPreprocessorConfig::from_file(&dir.join("preprocessor_config.json"))
                .unwrap_or_default();
        Ok(cfg)
    }

    pub fn from_model_dir(dir: &Path) -> Result<Self> {
        Self::from_file(&dir.join("config.json"))
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.model_type == "locateanything",
            "model_type must be locateanything, got {}",
            self.model_type
        );
        ensure!(
            self.vision_config.model_type == "moonvit",
            "vision_config.model_type must be moonvit, got {}",
            self.vision_config.model_type
        );
        ensure!(self.text_config.num_hidden_layers > 0, "text layers");
        ensure!(self.vision_config.num_hidden_layers > 0, "vision layers");
        ensure!(self.mlp_connector_layers == 2, "mlp_connector_layers");
        Ok(())
    }

    /// Projector input width: MoonViT hidden × merge kernel area (2×2).
    pub fn projector_input_dim(&self) -> usize {
        self.vision_config.hidden_size * self.vision_config.merge_area()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn moonvit_merge_area() {
        let cfg = MoonVitConfig {
            model_type: "moonvit".into(),
            hidden_size: 1152,
            intermediate_size: 4304,
            num_attention_heads: 16,
            num_hidden_layers: 27,
            patch_size: 14,
            merge_kernel_size: [2, 2],
            init_pos_emb_height: 64,
            init_pos_emb_width: 64,
        };
        assert_eq!(cfg.merge_area(), 4);
    }
}
