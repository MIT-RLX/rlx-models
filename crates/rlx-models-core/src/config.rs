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

//! Model configuration structs — parsed from HuggingFace config.json.

use serde::{Deserialize, Deserializer};
use std::path::Path;

fn deserialize_usize_or_float<'de, D: Deserializer<'de>>(d: D) -> Result<usize, D::Error> {
    let v: serde_json::Value = Deserialize::deserialize(d)?;
    match v {
        serde_json::Value::Number(n) => {
            if let Some(u) = n.as_u64() {
                Ok(u as usize)
            } else if let Some(f) = n.as_f64() {
                Ok(f as usize)
            } else {
                Err(serde::de::Error::custom("expected number"))
            }
        }
        _ => Err(serde::de::Error::custom("expected number")),
    }
}

/// BERT model configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct BertConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    #[serde(default = "default_type_vocab_size")]
    pub type_vocab_size: usize,
    #[serde(default = "default_layer_norm_eps")]
    pub layer_norm_eps: f64,
    #[serde(default = "default_hidden_act")]
    pub hidden_act: String,
}

fn default_type_vocab_size() -> usize {
    2
}
fn default_layer_norm_eps() -> f64 {
    1e-12
}
fn default_hidden_act() -> String {
    "gelu".into()
}

impl BertConfig {
    pub fn from_file(path: &Path) -> anyhow::Result<Self> {
        let data = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&data)?)
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

/// NomicBERT model configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct NomicBertConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    #[serde(default = "default_type_vocab_size")]
    pub type_vocab_size: usize,
    #[serde(default = "default_layer_norm_eps")]
    pub layer_norm_eps: f64,
    #[serde(default = "default_head_dim")]
    pub head_dim: usize,
    #[serde(default = "default_rotary_emb_base")]
    pub rotary_emb_base: f64,
}

fn default_head_dim() -> usize {
    64
}
fn default_rotary_emb_base() -> f64 {
    1000.0
}

impl NomicBertConfig {
    pub fn from_file(path: &Path) -> anyhow::Result<Self> {
        let data = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&data)?)
    }
}

/// NomicVision model configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct NomicVisionConfig {
    #[serde(alias = "n_embd")]
    pub hidden_size: usize,
    #[serde(alias = "n_layer")]
    pub num_hidden_layers: usize,
    #[serde(alias = "n_head")]
    pub num_attention_heads: usize,
    #[serde(
        default = "default_vision_intermediate",
        deserialize_with = "deserialize_usize_or_float"
    )]
    pub n_inner: usize,
    pub img_size: usize,
    pub patch_size: usize,
    #[serde(default = "default_vision_ln_eps")]
    pub layer_norm_epsilon: f64,
}

fn default_vision_intermediate() -> usize {
    2048
}
fn default_vision_ln_eps() -> f64 {
    1e-6
}

impl NomicVisionConfig {
    pub fn from_file(path: &Path) -> anyhow::Result<Self> {
        let data = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&data)?)
    }
    pub fn intermediate_size(&self) -> usize {
        self.n_inner
    }
    pub fn layer_norm_eps(&self) -> f64 {
        self.layer_norm_epsilon
    }
}

// DinoV2Config moved to crate::dinov2::config (subfolder module).

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_bert_config() {
        let json = r#"{
            "vocab_size": 30522,
            "hidden_size": 384,
            "num_hidden_layers": 6,
            "num_attention_heads": 12,
            "intermediate_size": 1536,
            "max_position_embeddings": 512
        }"#;
        let cfg: BertConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.hidden_size, 384);
        assert_eq!(cfg.head_dim(), 32);
        assert_eq!(cfg.layer_norm_eps, 1e-12);
    }
}
