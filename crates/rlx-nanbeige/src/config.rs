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

/// Hugging Face `config.json` fields used to recognize Nanbeige checkpoints.
#[derive(Debug, Clone, Deserialize)]
struct HfConfigProbe {
    #[serde(default)]
    model_type: Option<String>,
    #[serde(default)]
    architectures: Option<Vec<String>>,
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
///
/// Nanbeige is Llama-shaped with an extra `num_loops` field; serde maps that
/// into [`Llama32Config::num_loops`] so the shared decoder unrolls correctly.
pub fn llama_config_from_hf(weights_or_dir: &Path) -> Result<Llama32Config> {
    let cfg_path = config_json_path(weights_or_dir);
    Llama32Config::from_file(&cfg_path)
        .with_context(|| format!("reading Nanbeige HF config {cfg_path:?}"))
}

/// Ensure `config.json` describes a Nanbeige Looped Transformer checkpoint.
pub fn validate_hf_config(weights_or_dir: &Path) -> Result<()> {
    let cfg_path = config_json_path(weights_or_dir);
    let raw =
        std::fs::read_to_string(&cfg_path).with_context(|| format!("reading {cfg_path:?}"))?;
    let probe: HfConfigProbe =
        serde_json::from_str(&raw).with_context(|| format!("parsing {cfg_path:?}"))?;

    match probe.model_type.as_deref() {
        Some("nanbeige") => {}
        Some(other) => {
            bail!("rlx-nanbeige: {cfg_path:?} has model_type={other:?}; expected `nanbeige`")
        }
        None => bail!("rlx-nanbeige: {cfg_path:?} missing model_type"),
    }

    if let Some(archs) = &probe.architectures {
        let ok = archs
            .iter()
            .any(|a| a == "NanbeigeForCausalLM" || a == "NanbeigeModel");
        if !ok {
            bail!(
                "rlx-nanbeige: {cfg_path:?} architectures={archs:?}; \
                 expected NanbeigeForCausalLM"
            );
        }
    }

    Ok(())
}

/// GGUF arch tag or HF `config.json` checks, depending on weight format.
pub fn validate_weights_kind(weights: &Path) -> Result<()> {
    let fmt = if weights.is_dir() {
        WeightFormat::detect(weights)?
    } else {
        WeightFormat::from_path(weights)?
    };
    match fmt {
        WeightFormat::Gguf => {
            let cfg = LlamaBaseConfig::from_gguf_path(weights)
                .with_context(|| format!("rlx-nanbeige: parse GGUF {weights:?}"))?;
            // Nanbeige's public GGUF fork may still tag as `llama` / `nanbeige`.
            if !matches!(cfg.arch.as_str(), "llama" | "nanbeige") {
                bail!(
                    "rlx-nanbeige: expected `general.architecture` ∈ [llama, nanbeige]; \
                     got `{}` at {weights:?}",
                    cfg.arch
                );
            }
        }
        WeightFormat::Safetensors => validate_hf_config(weights)?,
    }
    Ok(())
}

/// Reference dims for [Nanbeige/Nanbeige4.2-3B](https://huggingface.co/Nanbeige/Nanbeige4.2-3B).
pub fn nanbeige42_3b_preset() -> Llama32Config {
    Llama32Config {
        embedding_scale: None,
        residual_scale: None,
        attention_scale: None,
        logit_scale: None,
        vocab_size: 166_144,
        hidden_size: 3072,
        intermediate_size: 10_752,
        num_hidden_layers: 22,
        num_attention_heads: 48,
        num_key_value_heads: 8,
        max_position_embeddings: 262_144,
        rms_norm_eps: 1e-5,
        rope_theta: 70_000_000.0,
        hidden_act: "silu".into(),
        tie_word_embeddings: false,
        attention_bias: false,
        head_dim: Some(128),
        rope_scaling: None,
        num_loops: 2,
        skip_loop_final_norm: false,
        rope_style: rlx_llama32::RopeStyle::NeoX,
        gguf_arch: None,
        rope_dim: None,
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
        let p = nanbeige42_3b_preset();
        assert_eq!(p.hidden_size, 3072);
        assert_eq!(p.num_hidden_layers, 22);
        assert_eq!(p.num_loops, 2);
        assert_eq!(p.kv_layers(), 44);
        assert_eq!(p.head_dim(), 128);
        assert!((p.rope_theta - 70_000_000.0).abs() < 1.0);
        assert!(!p.skip_loop_final_norm);
    }

    #[test]
    fn validates_nanbeige_model_type() {
        let dir = std::env::temp_dir().join(format!(
            "rlx_nanbeige_cfg_{}_{}",
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
                "model_type": "nanbeige",
                "architectures": ["NanbeigeForCausalLM"],
                "vocab_size": 166144,
                "hidden_size": 3072,
                "intermediate_size": 10752,
                "num_hidden_layers": 22,
                "num_attention_heads": 48,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "max_position_embeddings": 262144,
                "rms_norm_eps": 1e-5,
                "rope_theta": 70000000,
                "num_loops": 2,
                "skip_loop_final_norm": false
            }"#,
        )
        .unwrap();
        validate_hf_config(&dir).expect("valid nanbeige config");
        let cfg = llama_config_from_hf(&dir).unwrap();
        assert_eq!(cfg.num_loops, 2);
        assert_eq!(cfg.kv_layers(), 44);
        assert_eq!(cfg.weight_layer_index(22), 0);
        assert_eq!(cfg.weight_layer_index(43), 21);
        std::fs::remove_dir_all(&dir).ok();
    }
}
