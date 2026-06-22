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

//! Grounding DINO configuration. Mirrors `IDEA-Research/grounding-dino-base`
//! (HF `transformers` `GroundingDinoConfig`) `config.json`.

use serde::Deserialize;
use std::path::Path;

/// ImageNet-1k mean/std applied to RGB pixels in `[0, 1]` (HF `GroundingDinoImageProcessor`).
pub const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
pub const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// Default aspect-preserving resize bounds used by the HF processor.
pub const DEFAULT_SHORTEST_EDGE: usize = 800;
pub const DEFAULT_LONGEST_EDGE: usize = 1333;

/// Swin Transformer backbone configuration (`backbone_config` in `config.json`).
#[derive(Debug, Clone, Deserialize)]
pub struct SwinConfig {
    #[serde(default = "swin_embed_dim")]
    pub embed_dim: usize,
    #[serde(default = "swin_depths")]
    pub depths: Vec<usize>,
    #[serde(default = "swin_num_heads")]
    pub num_heads: Vec<usize>,
    #[serde(default = "swin_window_size")]
    pub window_size: usize,
    #[serde(default = "swin_image_size")]
    pub image_size: usize,
    #[serde(default = "swin_patch_size")]
    pub patch_size: usize,
    #[serde(default = "swin_mlp_ratio")]
    pub mlp_ratio: f64,
    #[serde(default = "swin_num_channels")]
    pub num_channels: usize,
    /// `out_indices` into the 1-based stage list (e.g. `[2, 3, 4]`).
    #[serde(default = "swin_out_indices")]
    pub out_indices: Vec<usize>,
    #[serde(default = "swin_eps")]
    pub layer_norm_eps: f64,
    #[serde(default = "default_true")]
    pub qkv_bias: bool,
}

fn swin_embed_dim() -> usize {
    128
}
fn swin_depths() -> Vec<usize> {
    vec![2, 2, 18, 2]
}
fn swin_num_heads() -> Vec<usize> {
    vec![4, 8, 16, 32]
}
fn swin_window_size() -> usize {
    12
}
fn swin_image_size() -> usize {
    384
}
fn swin_patch_size() -> usize {
    4
}
fn swin_mlp_ratio() -> f64 {
    4.0
}
fn swin_num_channels() -> usize {
    3
}
fn swin_out_indices() -> Vec<usize> {
    vec![2, 3, 4]
}
fn swin_eps() -> f64 {
    1e-5
}
fn default_true() -> bool {
    true
}

impl Default for SwinConfig {
    fn default() -> Self {
        Self {
            embed_dim: swin_embed_dim(),
            depths: swin_depths(),
            num_heads: swin_num_heads(),
            window_size: swin_window_size(),
            image_size: swin_image_size(),
            patch_size: swin_patch_size(),
            mlp_ratio: swin_mlp_ratio(),
            num_channels: swin_num_channels(),
            out_indices: swin_out_indices(),
            layer_norm_eps: swin_eps(),
            qkv_bias: true,
        }
    }
}

impl SwinConfig {
    pub fn num_stages(&self) -> usize {
        self.depths.len()
    }
    /// Feature dimension produced by stage `i` (0-based): `embed_dim * 2^i`.
    pub fn stage_dim(&self, stage: usize) -> usize {
        self.embed_dim * (1 << stage)
    }
    /// Output channel dims for the stages selected by `out_indices` (1-based).
    pub fn out_channels(&self) -> Vec<usize> {
        self.out_indices
            .iter()
            .map(|&idx| self.stage_dim(idx - 1))
            .collect()
    }
}

/// BERT text-backbone configuration (`text_config` in `config.json`).
/// Defaults match `bert-base-uncased`.
#[derive(Debug, Clone, Deserialize)]
pub struct TextConfig {
    #[serde(default = "bert_vocab")]
    pub vocab_size: usize,
    #[serde(default = "bert_hidden")]
    pub hidden_size: usize,
    #[serde(default = "bert_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "bert_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "bert_intermediate")]
    pub intermediate_size: usize,
    #[serde(default = "bert_max_pos")]
    pub max_position_embeddings: usize,
    #[serde(default = "bert_type_vocab")]
    pub type_vocab_size: usize,
    #[serde(default = "bert_eps")]
    pub layer_norm_eps: f64,
    #[serde(default = "bert_act")]
    pub hidden_act: String,
}

fn bert_vocab() -> usize {
    30522
}
fn bert_hidden() -> usize {
    768
}
fn bert_layers() -> usize {
    12
}
fn bert_heads() -> usize {
    12
}
fn bert_intermediate() -> usize {
    3072
}
fn bert_max_pos() -> usize {
    512
}
fn bert_type_vocab() -> usize {
    2
}
fn bert_eps() -> f64 {
    1e-12
}
fn bert_act() -> String {
    "gelu".to_string()
}

impl Default for TextConfig {
    fn default() -> Self {
        Self {
            vocab_size: bert_vocab(),
            hidden_size: bert_hidden(),
            num_hidden_layers: bert_layers(),
            num_attention_heads: bert_heads(),
            intermediate_size: bert_intermediate(),
            max_position_embeddings: bert_max_pos(),
            type_vocab_size: bert_type_vocab(),
            layer_norm_eps: bert_eps(),
            hidden_act: bert_act(),
        }
    }
}

impl TextConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

/// Top-level Grounding DINO configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct GroundingDinoConfig {
    #[serde(default)]
    pub backbone_config: SwinConfig,
    #[serde(default)]
    pub text_config: TextConfig,
    #[serde(default = "default_d_model")]
    pub d_model: usize,
    #[serde(default = "default_layers")]
    pub encoder_layers: usize,
    #[serde(default = "default_layers")]
    pub decoder_layers: usize,
    #[serde(default = "default_heads")]
    pub encoder_attention_heads: usize,
    #[serde(default = "default_heads")]
    pub decoder_attention_heads: usize,
    #[serde(default = "default_ffn")]
    pub encoder_ffn_dim: usize,
    #[serde(default = "default_ffn")]
    pub decoder_ffn_dim: usize,
    #[serde(default = "default_feature_levels")]
    pub num_feature_levels: usize,
    #[serde(default = "default_num_queries")]
    pub num_queries: usize,
    #[serde(default = "default_max_text_len")]
    pub max_text_len: usize,
    #[serde(default = "default_n_points")]
    pub encoder_n_points: usize,
    #[serde(default = "default_n_points")]
    pub decoder_n_points: usize,
    #[serde(default = "default_activation")]
    pub activation_function: String,
    #[serde(default = "default_pos_embed")]
    pub position_embedding_type: String,
    #[serde(default = "default_pos_temp")]
    pub positional_embedding_temperature: f64,
}

fn default_d_model() -> usize {
    256
}
fn default_layers() -> usize {
    6
}
fn default_heads() -> usize {
    8
}
fn default_ffn() -> usize {
    2048
}
fn default_feature_levels() -> usize {
    4
}
fn default_num_queries() -> usize {
    900
}
fn default_max_text_len() -> usize {
    256
}
fn default_n_points() -> usize {
    4
}
fn default_activation() -> String {
    "relu".to_string()
}
fn default_pos_embed() -> String {
    "sine".to_string()
}
fn default_pos_temp() -> f64 {
    20.0
}

impl Default for GroundingDinoConfig {
    fn default() -> Self {
        Self {
            backbone_config: SwinConfig::default(),
            text_config: TextConfig::default(),
            d_model: default_d_model(),
            encoder_layers: default_layers(),
            decoder_layers: default_layers(),
            encoder_attention_heads: default_heads(),
            decoder_attention_heads: default_heads(),
            encoder_ffn_dim: default_ffn(),
            decoder_ffn_dim: default_ffn(),
            num_feature_levels: default_feature_levels(),
            num_queries: default_num_queries(),
            max_text_len: default_max_text_len(),
            encoder_n_points: default_n_points(),
            decoder_n_points: default_n_points(),
            activation_function: default_activation(),
            position_embedding_type: default_pos_embed(),
            positional_embedding_temperature: default_pos_temp(),
        }
    }
}

impl GroundingDinoConfig {
    /// The canonical `grounding-dino-base` configuration.
    pub fn base() -> Self {
        Self::default()
    }

    /// Load the config from a HF `config.json` on disk.
    pub fn from_file(path: &Path) -> anyhow::Result<Self> {
        let data = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&data)?)
    }

    /// Parse the config from a HF `config.json` string.
    pub fn from_json_str(data: &str) -> anyhow::Result<Self> {
        Ok(serde_json::from_str(data)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base_config_matches_hf() {
        let c = GroundingDinoConfig::base();
        assert_eq!(c.d_model, 256);
        assert_eq!(c.encoder_layers, 6);
        assert_eq!(c.decoder_layers, 6);
        assert_eq!(c.num_queries, 900);
        assert_eq!(c.num_feature_levels, 4);
        assert_eq!(c.max_text_len, 256);
        assert_eq!(c.backbone_config.embed_dim, 128);
        assert_eq!(c.backbone_config.depths, vec![2, 2, 18, 2]);
        assert_eq!(c.backbone_config.num_heads, vec![4, 8, 16, 32]);
        assert_eq!(c.backbone_config.window_size, 12);
        assert_eq!(c.backbone_config.out_channels(), vec![256, 512, 1024]);
        assert_eq!(c.text_config.hidden_size, 768);
        assert_eq!(c.text_config.num_hidden_layers, 12);
    }

    #[test]
    fn parses_real_config_json() {
        // Subset of the published config.json — verifies serde defaults + nesting.
        let json = r#"{
            "activation_function": "relu",
            "architectures": ["GroundingDinoForObjectDetection"],
            "backbone_config": {
                "depths": [2, 2, 18, 2],
                "embed_dim": 128,
                "hidden_size": 1024,
                "image_size": 384,
                "model_type": "swin",
                "num_heads": [4, 8, 16, 32],
                "out_features": ["stage2", "stage3", "stage4"],
                "out_indices": [2, 3, 4],
                "window_size": 12
            },
            "d_model": 256,
            "decoder_attention_heads": 8,
            "decoder_ffn_dim": 2048,
            "decoder_layers": 6,
            "encoder_attention_heads": 8,
            "encoder_ffn_dim": 2048,
            "encoder_layers": 6,
            "max_text_len": 256,
            "model_type": "grounding-dino",
            "num_feature_levels": 4,
            "num_queries": 900,
            "position_embedding_type": "sine",
            "text_config": {"model_type": "bert"},
            "torch_dtype": "float32"
        }"#;
        let c = GroundingDinoConfig::from_json_str(json).unwrap();
        assert_eq!(c.num_queries, 900);
        assert_eq!(c.backbone_config.out_indices, vec![2, 3, 4]);
        assert_eq!(c.text_config.hidden_size, 768); // default (bert-base)
        assert_eq!(c.encoder_ffn_dim, 2048);
    }
}
