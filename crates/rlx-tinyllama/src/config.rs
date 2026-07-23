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

//! TinyLlama-1.1B config validation and reference hyperparameters.

use anyhow::{Context, Result, bail};
use rlx_cli::WeightFormat;
use rlx_llama_base::LlamaBaseConfig;
use rlx_llama32::Llama32Config;
use serde::Deserialize;
use std::path::{Path, PathBuf};

/// Expected `hidden_size` for TinyLlama-1.1B (`2048`).
pub const TINYLLAMA_1_1B_HIDDEN_SIZE: usize = 2048;
/// Expected transformer block count for TinyLlama-1.1B (`22`).
pub const TINYLLAMA_1_1B_NUM_LAYERS: usize = 22;

/// Hugging Face `config.json` fields used to recognize TinyLlama checkpoints.
#[derive(Debug, Clone, Deserialize)]
struct HfConfigProbe {
    #[serde(default)]
    model_type: Option<String>,
    #[serde(default)]
    architectures: Option<Vec<String>>,
    #[serde(default)]
    hidden_size: Option<usize>,
    #[serde(default)]
    num_hidden_layers: Option<usize>,
}

/// Resolve `config.json` next to a safetensors file or inside a model directory.
pub fn config_json_path(weights_or_dir: &Path) -> PathBuf {
    if weights_or_dir.is_dir() {
        return weights_or_dir.join("config.json");
    }
    weights_or_dir
        .parent()
        .map(|p| p.join("config.json"))
        .unwrap_or_else(|| PathBuf::from("config.json"))
}

/// Load [`Llama32Config`] from the HF layout beside `weights_or_dir`.
pub fn llama_config_from_hf(weights_or_dir: &Path) -> Result<Llama32Config> {
    let cfg_path = config_json_path(weights_or_dir);
    Llama32Config::from_file(&cfg_path)
        .with_context(|| format!("reading TinyLlama HF config {cfg_path:?}"))
}

fn validate_tinyllama_1_1b_dims(hidden_size: usize, num_hidden_layers: usize) -> Result<()> {
    if hidden_size != TINYLLAMA_1_1B_HIDDEN_SIZE || num_hidden_layers != TINYLLAMA_1_1B_NUM_LAYERS {
        bail!(
            "rlx-tinyllama: expected TinyLlama-1.1B dims \
             (hidden_size={TINYLLAMA_1_1B_HIDDEN_SIZE}, num_hidden_layers={TINYLLAMA_1_1B_NUM_LAYERS}); \
             got hidden_size={hidden_size}, num_hidden_layers={num_hidden_layers}"
        );
    }
    Ok(())
}

/// Ensure `config.json` describes a TinyLlama-1.1B checkpoint.
pub fn validate_hf_config(weights_or_dir: &Path) -> Result<()> {
    let cfg_path = config_json_path(weights_or_dir);
    let raw =
        std::fs::read_to_string(&cfg_path).with_context(|| format!("reading {cfg_path:?}"))?;
    let probe: HfConfigProbe =
        serde_json::from_str(&raw).with_context(|| format!("parsing {cfg_path:?}"))?;

    match probe.model_type.as_deref() {
        Some("llama") => {}
        Some(other) => bail!(
            "rlx-tinyllama: {cfg_path:?} has model_type={other:?}; expected `llama` \
             (TinyLlama-1.1B is LlamaForCausalLM-shaped)"
        ),
        None => bail!("rlx-tinyllama: {cfg_path:?} missing model_type"),
    }

    if let Some(archs) = &probe.architectures {
        let ok = archs.iter().any(|a| a == "LlamaForCausalLM");
        if !ok {
            bail!(
                "rlx-tinyllama: {cfg_path:?} architectures={archs:?}; \
                 expected LlamaForCausalLM (TinyLlama)"
            );
        }
    }

    let hidden_size = probe
        .hidden_size
        .ok_or_else(|| anyhow::anyhow!("rlx-tinyllama: {cfg_path:?} missing hidden_size"))?;
    let num_hidden_layers = probe
        .num_hidden_layers
        .ok_or_else(|| anyhow::anyhow!("rlx-tinyllama: {cfg_path:?} missing num_hidden_layers"))?;
    validate_tinyllama_1_1b_dims(hidden_size, num_hidden_layers)
}

/// GGUF arch tag or HF `config.json` checks, depending on weight format.
pub fn validate_weights_kind(weights: &Path) -> Result<()> {
    match WeightFormat::from_path(weights)? {
        WeightFormat::Gguf => {
            let cfg = LlamaBaseConfig::from_gguf_path(weights)
                .with_context(|| format!("rlx-tinyllama: parse GGUF {weights:?}"))?;
            if cfg.arch != "llama" {
                bail!(
                    "rlx-tinyllama: expected `general.architecture = llama`; \
                     got `{}` at {weights:?}",
                    cfg.arch
                );
            }
            validate_tinyllama_1_1b_dims(cfg.hidden_size, cfg.num_hidden_layers)?;
        }
        WeightFormat::Safetensors => validate_hf_config(weights)?,
    }
    Ok(())
}

/// Reference dims for [TinyLlama/TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0).
pub fn tinyllama_1_1b_preset() -> Llama32Config {
    Llama32Config {
        vocab_size: 32_000,
        hidden_size: TINYLLAMA_1_1B_HIDDEN_SIZE,
        intermediate_size: 5632,
        num_hidden_layers: TINYLLAMA_1_1B_NUM_LAYERS,
        num_attention_heads: 32,
        num_key_value_heads: 4,
        max_position_embeddings: 2048,
        rms_norm_eps: 1e-5,
        rope_theta: 10_000.0,
        hidden_act: "silu".into(),
        tie_word_embeddings: false,
        attention_bias: false,
        head_dim: None,
        rope_scaling: None,
        num_loops: 1,
        skip_loop_final_norm: false,
        rope_style: rlx_llama32::RopeStyle::NeoX,
        gguf_arch: None,
        rope_dim: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preset_matches_hf_card() {
        let p = tinyllama_1_1b_preset();
        assert_eq!(p.hidden_size, 2048);
        assert_eq!(p.num_hidden_layers, 22);
        assert_eq!(p.head_dim(), 64);
        assert_eq!(p.num_key_value_heads, 4);
        assert!((p.rope_theta - 10_000.0).abs() < 1.0);
    }

    #[test]
    fn validates_llama_model_type_and_dims() {
        let dir = std::env::temp_dir().join(format!(
            "rlx_tinyllama_cfg_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("config.json"),
            r#"{
                "model_type": "llama",
                "architectures": ["LlamaForCausalLM"],
                "vocab_size": 32000,
                "hidden_size": 2048,
                "intermediate_size": 5632,
                "num_hidden_layers": 22,
                "num_attention_heads": 32,
                "num_key_value_heads": 4,
                "max_position_embeddings": 2048,
                "rms_norm_eps": 1e-5,
                "rope_theta": 10000
            }"#,
        )
        .unwrap();
        validate_hf_config(&dir).expect("valid tinyllama config");
        let cfg = llama_config_from_hf(&dir).unwrap();
        assert_eq!(cfg.hidden_size, 2048);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn rejects_wrong_layer_count() {
        let dir = std::env::temp_dir().join(format!(
            "rlx_tinyllama_bad_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("config.json"),
            r#"{
                "model_type": "llama",
                "architectures": ["LlamaForCausalLM"],
                "hidden_size": 2048,
                "num_hidden_layers": 32
            }"#,
        )
        .unwrap();
        let err = validate_hf_config(&dir).unwrap_err();
        assert!(
            err.to_string().contains("num_hidden_layers=32"),
            "unexpected error: {err:#}"
        );
        std::fs::remove_dir_all(&dir).ok();
    }
}
