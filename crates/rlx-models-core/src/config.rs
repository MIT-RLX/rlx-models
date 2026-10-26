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

/// Restore the RoPE keys transformers 5 folds into `rope_parameters` in the
/// `config.json` it saves: `rope_theta` (and `partial_rotary_factor`) back
/// at the top level, and a non-default `rope_type`'s parameters back as
/// `rope_scaling` (with the old `type` alias), each only where the config
/// lacks it (`rope_scaling` also for a default type carrying more, such as
/// Qwen-VL's `mrope_section`). A reader of the older layout otherwise falls back to its
/// default theta and no scaling, and every position rotates wrong with no
/// error. Applied to the top level and to each `*_config` object in it
/// (`text_config`, …); a per-layer-type map (`rope_parameters:
/// {full_attention: {…}, …}`, Gemma) is left for its own reader.
pub fn legacy_rope_keys(config: &mut serde_json::Value) {
    use serde_json::Value;
    let Some(obj) = config.as_object_mut() else {
        return;
    };
    for (key, child) in obj.iter_mut() {
        if key.ends_with("_config") && child.is_object() {
            legacy_rope_keys(child);
        }
    }
    let Some(rp) = obj
        .get("rope_parameters")
        .and_then(Value::as_object)
        .cloned()
    else {
        return;
    };
    if !rp.contains_key("rope_theta") && !rp.contains_key("rope_type") {
        return;
    }
    for key in ["rope_theta", "partial_rotary_factor"] {
        if !obj.contains_key(key)
            && let Some(v) = rp.get(key)
        {
            obj.insert(key.into(), v.clone());
        }
    }
    let kind = rp
        .get("rope_type")
        .and_then(Value::as_str)
        .unwrap_or("default");
    // Scaling, or anything else the old `rope_scaling` carried
    // (`mrope_section` with a default type, Qwen-VL).
    let more = rp.keys().any(|k| {
        !matches!(
            k.as_str(),
            "rope_type" | "rope_theta" | "partial_rotary_factor"
        )
    });
    if (kind != "default" || more) && obj.get("rope_scaling").is_none_or(Value::is_null) {
        let mut scaling = rp.clone();
        scaling.remove("rope_theta");
        scaling
            .entry("type")
            .or_insert_with(|| Value::String(kind.into()));
        obj.insert("rope_scaling".into(), Value::Object(scaling));
    }
}

/// A Hugging Face `config.json` as `T`, read through [`legacy_rope_keys`].
pub fn from_hf_json<T: serde::de::DeserializeOwned>(text: &str) -> anyhow::Result<T> {
    let mut value: serde_json::Value = serde_json::from_str(text)?;
    legacy_rope_keys(&mut value);
    Ok(serde_json::from_value(value)?)
}

#[cfg(test)]
mod legacy_rope_tests {
    use super::legacy_rope_keys;
    use serde_json::json;

    #[test]
    fn a_transformers_5_config_gets_its_theta_and_scaling_back() {
        // What transformers 5 saves for Llama 3.2.
        let mut c = json!({"rope_parameters": {"rope_type": "llama3", "factor": 32.0,
            "high_freq_factor": 4.0, "low_freq_factor": 1.0,
            "original_max_position_embeddings": 8192, "rope_theta": 500000.0},
            "text_config": {"rope_parameters": {"rope_theta": 5e6, "rope_type": "default"}}});
        legacy_rope_keys(&mut c);
        assert_eq!(c["rope_theta"], json!(500000.0));
        assert_eq!(c["rope_scaling"]["rope_type"], json!("llama3"));
        assert_eq!(c["rope_scaling"]["type"], json!("llama3"));
        assert_eq!(c["rope_scaling"]["factor"], json!(32.0));
        assert!(c["rope_scaling"].get("rope_theta").is_none());
        assert_eq!(c["text_config"]["rope_theta"], json!(5e6));
        assert!(c["text_config"].get("rope_scaling").is_none());
        let mut vl = json!({"rope_parameters": {"rope_type": "default", "rope_theta": 1e6,
            "mrope_section": [16, 24, 24]}});
        legacy_rope_keys(&mut vl);
        assert_eq!(vl["rope_scaling"]["mrope_section"], json!([16, 24, 24]));
        assert_eq!(vl["rope_scaling"]["type"], json!("default"));
    }

    #[test]
    fn keys_a_config_has_and_per_layer_maps_are_left_alone() {
        let mut c = json!({"rope_theta": 1e4, "rope_scaling": {"type": "linear", "factor": 2.0},
            "rope_parameters": {"rope_theta": 5e6, "rope_type": "yarn", "factor": 4.0}});
        let before = c.clone();
        legacy_rope_keys(&mut c);
        assert_eq!(c, before);
        let mut g = json!({"rope_parameters": {"full_attention": {"rope_theta": 1e6}}});
        let before = g.clone();
        legacy_rope_keys(&mut g);
        assert_eq!(g, before);
    }
}
