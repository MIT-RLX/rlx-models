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

//! Configuration for Llama-3.2-Vision (mllama).
//!
//! Mirrors `transformers` `MllamaConfig` = a `vision_config`
//! ([`MllamaVisionConfig`]) + a `text_config` ([`MllamaTextConfig`]) plus the
//! `image_token_index`. Parsed from the HF `config.json`; the defaults match
//! the 11B checkpoint.

use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::Path;

/// Vision tower configuration (`MllamaVisionConfig`).
#[derive(Debug, Clone, Deserialize)]
pub struct MllamaVisionConfig {
    #[serde(default = "d_vision_hidden")]
    pub hidden_size: usize,
    #[serde(default = "d_vision_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "d_vision_global_layers")]
    pub num_global_layers: usize,
    #[serde(default = "d_vision_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "d_vision_intermediate")]
    pub intermediate_size: usize,
    #[serde(default = "d_vision_output_dim")]
    pub vision_output_dim: usize,
    #[serde(default = "d_image_size")]
    pub image_size: usize,
    #[serde(default = "d_patch_size")]
    pub patch_size: usize,
    #[serde(default = "d_max_num_tiles")]
    pub max_num_tiles: usize,
    #[serde(default = "d_norm_eps")]
    pub norm_eps: f32,
    #[serde(default = "d_num_channels")]
    pub num_channels: usize,
    #[serde(default = "d_intermediate_layers_indices")]
    pub intermediate_layers_indices: Vec<usize>,
    #[serde(default = "d_supported_aspect_ratios")]
    pub supported_aspect_ratios: Vec<Vec<usize>>,
}

impl MllamaVisionConfig {
    /// Patches per tile (incl. the class token): `(image_size/patch_size)^2 + 1`.
    pub fn num_patches(&self) -> usize {
        let side = self.image_size / self.patch_size;
        side * side + 1
    }
    /// Per-head dimension of the vision attention.
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
    /// Number of supported aspect ratios (== max aspect-ratio id).
    pub fn max_aspect_ratio_id(&self) -> usize {
        self.supported_aspect_ratios.len()
    }
    /// The concatenated feature width fed to the projector.
    /// `vision_output_dim == hidden_size * (1 + intermediate_layers_indices.len())`.
    pub fn concat_width(&self) -> usize {
        self.hidden_size * (1 + self.intermediate_layers_indices.len())
    }
}

/// Text tower configuration (`MllamaTextConfig`).
#[derive(Debug, Clone, Deserialize)]
pub struct MllamaTextConfig {
    #[serde(default = "d_text_vocab")]
    pub vocab_size: usize,
    #[serde(default = "d_text_hidden")]
    pub hidden_size: usize,
    #[serde(default = "d_text_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "d_text_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "d_text_kv_heads")]
    pub num_key_value_heads: usize,
    #[serde(default = "d_text_intermediate")]
    pub intermediate_size: usize,
    #[serde(default = "d_norm_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "d_rope_theta")]
    pub rope_theta: f32,
    #[serde(default = "d_max_pos")]
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default = "d_cross_attention_layers")]
    pub cross_attention_layers: Vec<usize>,
    #[serde(default)]
    pub rope_scaling: Option<Llama3RopeScaling>,
    #[serde(default = "d_text_bos")]
    pub bos_token_id: u32,
    #[serde(default = "d_text_eos")]
    pub eos_token_id: u32,
}

impl MllamaTextConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
    pub fn kv_group_size(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }
    pub fn is_cross_attention_layer(&self, idx: usize) -> bool {
        self.cross_attention_layers.contains(&idx)
    }
}

/// Llama-3 RoPE scaling parameters (baked into the cos/sin tables host-side).
#[derive(Debug, Clone, Deserialize)]
pub struct Llama3RopeScaling {
    pub factor: f32,
    #[serde(default = "d_low_freq")]
    pub low_freq_factor: f32,
    #[serde(default = "d_high_freq")]
    pub high_freq_factor: f32,
    #[serde(default = "d_orig_max_pos")]
    pub original_max_position_embeddings: usize,
    #[serde(default)]
    pub rope_type: Option<String>,
}

/// Top-level mllama configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct MllamaConfig {
    pub text_config: MllamaTextConfig,
    pub vision_config: MllamaVisionConfig,
    #[serde(default = "d_image_token")]
    pub image_token_index: u32,
}

impl MllamaConfig {
    /// Parse `config.json` (the top-level `MllamaConfig` HF file).
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading mllama config {}", path.display()))?;
        Self::from_json(&text)
    }

    pub fn from_json(text: &str) -> Result<Self> {
        let cfg: MllamaConfig = serde_json::from_str(text).context("parsing mllama config json")?;
        Ok(cfg)
    }

    /// The 11B-Vision defaults (`meta-llama/Llama-3.2-11B-Vision`).
    pub fn llama_3_2_11b_vision() -> Self {
        Self::from_json("{\"text_config\":{},\"vision_config\":{}}")
            .expect("default mllama config parses")
    }
}

// ---- serde defaults (11B checkpoint values) --------------------------------
fn d_vision_hidden() -> usize {
    1280
}
fn d_vision_layers() -> usize {
    32
}
fn d_vision_global_layers() -> usize {
    8
}
fn d_vision_heads() -> usize {
    16
}
fn d_vision_intermediate() -> usize {
    5120
}
fn d_vision_output_dim() -> usize {
    7680
}
fn d_image_size() -> usize {
    448
}
fn d_patch_size() -> usize {
    14
}
fn d_max_num_tiles() -> usize {
    4
}
fn d_num_channels() -> usize {
    3
}
fn d_norm_eps() -> f32 {
    1e-5
}
fn d_intermediate_layers_indices() -> Vec<usize> {
    vec![3, 7, 15, 23, 30]
}
fn d_supported_aspect_ratios() -> Vec<Vec<usize>> {
    vec![
        vec![1, 1],
        vec![1, 2],
        vec![1, 3],
        vec![1, 4],
        vec![2, 1],
        vec![2, 2],
        vec![3, 1],
        vec![4, 1],
    ]
}
fn d_text_vocab() -> usize {
    128256
}
fn d_text_hidden() -> usize {
    4096
}
fn d_text_layers() -> usize {
    40
}
fn d_text_heads() -> usize {
    32
}
fn d_text_kv_heads() -> usize {
    8
}
fn d_text_intermediate() -> usize {
    14336
}
fn d_rope_theta() -> f32 {
    500000.0
}
fn d_max_pos() -> usize {
    131072
}
fn d_cross_attention_layers() -> Vec<usize> {
    vec![3, 8, 13, 18, 23, 28, 33, 38]
}
fn d_text_bos() -> u32 {
    128000
}
fn d_text_eos() -> u32 {
    128001
}
fn d_image_token() -> u32 {
    128256
}
fn d_low_freq() -> f32 {
    1.0
}
fn d_high_freq() -> f32 {
    4.0
}
fn d_orig_max_pos() -> usize {
    8192
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_11b() {
        let cfg = MllamaConfig::llama_3_2_11b_vision();
        assert_eq!(cfg.text_config.num_hidden_layers, 40);
        assert_eq!(
            cfg.text_config.cross_attention_layers,
            vec![3, 8, 13, 18, 23, 28, 33, 38]
        );
        assert_eq!(cfg.text_config.head_dim(), 128);
        assert_eq!(cfg.vision_config.num_patches(), 1025);
        assert_eq!(cfg.vision_config.head_dim(), 80);
        assert_eq!(cfg.vision_config.concat_width(), 7680);
        assert_eq!(cfg.image_token_index, 128256);
    }
}
