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

use serde::Deserialize;
use std::path::Path;

/// Wav2Vec2-BERT model configuration (e.g. facebook/w2v-bert-2.0).
#[derive(Debug, Clone, Deserialize)]
pub struct Wav2Vec2BertConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub feature_projection_input_dim: usize,
    #[serde(default = "default_layer_norm_eps")]
    pub layer_norm_eps: f64,
    #[serde(default = "default_hidden_act")]
    pub hidden_act: String,
    #[serde(default = "default_position_embeddings_type")]
    pub position_embeddings_type: String,
    #[serde(default = "default_left_max_position_embeddings")]
    pub left_max_position_embeddings: usize,
    #[serde(default = "default_right_max_position_embeddings")]
    pub right_max_position_embeddings: usize,
    #[serde(default = "default_conv_depthwise_kernel_size")]
    pub conv_depthwise_kernel_size: usize,
    #[serde(default)]
    pub add_adapter: bool,
    #[serde(default)]
    pub apply_spec_augment: bool,
    #[serde(default)]
    pub use_intermediate_ffn_before_adapter: bool,
    /// Present in HF configs; ignored at inference when `apply_spec_augment=false`.
    #[serde(default)]
    pub model_type: Option<String>,
}

fn default_layer_norm_eps() -> f64 {
    1e-5
}
fn default_hidden_act() -> String {
    "swish".into()
}
fn default_position_embeddings_type() -> String {
    "relative_key".into()
}
fn default_left_max_position_embeddings() -> usize {
    64
}
fn default_right_max_position_embeddings() -> usize {
    8
}
fn default_conv_depthwise_kernel_size() -> usize {
    31
}

impl Wav2Vec2BertConfig {
    pub fn from_file(path: &Path) -> anyhow::Result<Self> {
        let data = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&data)?)
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    pub fn num_relative_positions(&self) -> usize {
        self.left_max_position_embeddings + self.right_max_position_embeddings + 1
    }

    /// Factory for the public W2v-BERT 2.0 checkpoint dimensions.
    pub fn w2v_bert_2_0() -> Self {
        Self {
            hidden_size: 1024,
            num_hidden_layers: 24,
            num_attention_heads: 16,
            intermediate_size: 4096,
            feature_projection_input_dim: 160,
            layer_norm_eps: 1e-5,
            hidden_act: "swish".into(),
            position_embeddings_type: "relative_key".into(),
            left_max_position_embeddings: 64,
            right_max_position_embeddings: 8,
            conv_depthwise_kernel_size: 31,
            add_adapter: false,
            apply_spec_augment: false,
            use_intermediate_ffn_before_adapter: false,
            model_type: Some("wav2vec2-bert".into()),
        }
    }
}
