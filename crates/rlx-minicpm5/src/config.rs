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

use anyhow::{Context, Result, bail};
use rlx_cli::WeightFormat;
use rlx_llama_base::LlamaBaseConfig;
use rlx_llama32::Llama32Config;
use serde::Deserialize;
use std::path::{Path, PathBuf};

/// Hugging Face `config.json` fields used to recognize MiniCPM5 checkpoints.
#[derive(Debug, Clone, Deserialize)]
struct HfConfigProbe {
    #[serde(default)]
    model_type: Option<String>,
    #[serde(default)]
    architectures: Option<Vec<String>>,
    #[serde(default)]
    _name_or_path: Option<String>,
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
        .with_context(|| format!("reading MiniCPM5 HF config {cfg_path:?}"))
}

/// Ensure `config.json` describes a Llama-shaped MiniCPM5 checkpoint.
pub fn validate_hf_config(weights_or_dir: &Path) -> Result<()> {
    let cfg_path = config_json_path(weights_or_dir);
    let raw =
        std::fs::read_to_string(&cfg_path).with_context(|| format!("reading {cfg_path:?}"))?;
    let probe: HfConfigProbe =
        serde_json::from_str(&raw).with_context(|| format!("parsing {cfg_path:?}"))?;

    match probe.model_type.as_deref() {
        Some("llama") => {}
        Some(other) => bail!(
            "rlx-minicpm5: {cfg_path:?} has model_type={other:?}; expected `llama` \
             (MiniCPM5-1B is LlamaForCausalLM-shaped)"
        ),
        None => bail!("rlx-minicpm5: {cfg_path:?} missing model_type"),
    }

    if let Some(archs) = &probe.architectures {
        let ok = archs.iter().any(|a| a == "LlamaForCausalLM");
        if !ok {
            bail!(
                "rlx-minicpm5: {cfg_path:?} architectures={archs:?}; \
                 expected LlamaForCausalLM (MiniCPM5)"
            );
        }
    }

    Ok(())
}

/// GGUF arch tag or HF `config.json` checks, depending on weight format.
pub fn validate_weights_kind(weights: &Path) -> Result<()> {
    match WeightFormat::from_path(weights)? {
        WeightFormat::Gguf => {
            let cfg = LlamaBaseConfig::from_gguf_path(weights)
                .with_context(|| format!("rlx-minicpm5: parse GGUF {weights:?}"))?;
            if cfg.arch != "llama" {
                bail!(
                    "rlx-minicpm5: expected `general.architecture = llama`; \
                     got `{}` at {weights:?}",
                    cfg.arch
                );
            }
        }
        WeightFormat::Safetensors => validate_hf_config(weights)?,
    }
    Ok(())
}

/// Reference dims for [openbmb/MiniCPM5-1B](https://huggingface.co/openbmb/MiniCPM5-1B).
pub fn minicpm5_1b_preset() -> Llama32Config {
    Llama32Config {
        vocab_size: 130_560,
        hidden_size: 1536,
        intermediate_size: 4608,
        num_hidden_layers: 24,
        num_attention_heads: 16,
        num_key_value_heads: 2,
        max_position_embeddings: 131_072,
        rms_norm_eps: 1e-6,
        rope_theta: 5_000_000.0,
        hidden_act: "silu".into(),
        tie_word_embeddings: false,
        attention_bias: false,
        head_dim: Some(128),
        rope_scaling: None,
        num_loops: 1,
        skip_loop_final_norm: false,
        rope_style: rlx_llama32::RopeStyle::NeoX,
        gguf_arch: None,
        rope_dim: None,
        // Arch-quirk scales (Cohere/Granite/GLM/…) — off for MiniCPM (matches the
        // pre-existing behavior before these fields existed).
        embedding_scale: None,
        residual_scale: None,
        attention_scale: None,
        logit_scale: None,
        // Sliding-window / final-logit-softcap knobs added to Llama32Config
        // upstream — off for MiniCPM (matches behavior before these fields existed).
        sliding_window: None,
        sliding_window_pattern: None,
        final_logit_softcap: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preset_matches_hf_card() {
        let p = minicpm5_1b_preset();
        assert_eq!(p.hidden_size, 1536);
        assert_eq!(p.num_hidden_layers, 24);
        assert_eq!(p.head_dim(), 128);
        assert!((p.rope_theta - 5_000_000.0).abs() < 1.0);
    }

    #[test]
    fn validates_llama_model_type() {
        let dir = std::env::temp_dir().join(format!(
            "rlx_minicpm5_cfg_{}_{}",
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
                "vocab_size": 130560,
                "hidden_size": 1536,
                "intermediate_size": 4608,
                "num_hidden_layers": 24,
                "num_attention_heads": 16,
                "num_key_value_heads": 2,
                "head_dim": 128,
                "max_position_embeddings": 131072,
                "rms_norm_eps": 1e-6,
                "rope_theta": 5000000
            }"#,
        )
        .unwrap();
        validate_hf_config(&dir).expect("valid llama config");
        let cfg = llama_config_from_hf(&dir).unwrap();
        assert_eq!(cfg.hidden_size, 1536);
        std::fs::remove_dir_all(&dir).ok();
    }
}
