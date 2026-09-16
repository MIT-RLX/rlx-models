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

//! HY-MT1.5 (`hunyuan_v1_dense`) config validation and Qwen3 mapping.

use anyhow::{Context, Result, bail};
use rlx_cli::WeightFormat;
use rlx_core::gguf_architecture_from_path;
use rlx_qwen3::Qwen3Config;
use serde::Deserialize;
use std::path::{Path, PathBuf};

/// Expected HF / GGUF architecture tags for HY-MT dense.
pub const HF_MODEL_TYPES: &[&str] = &["hunyuan_v1_dense", "hunyuan-dense", "hunyuan_dense"];
/// GGUF `general.architecture` values accepted by this crate.
pub const GGUF_ARCHES: &[&str] = &["hunyuan-dense", "hunyuan_dense", "hunyuan-v1-dense"];

/// HY-MT1.5-1.8B reference dims.
pub const HY_MT_1_8B_HIDDEN: usize = 2048;
pub const HY_MT_1_8B_LAYERS: usize = 32;
/// HY-MT1.5-7B reference dims.
pub const HY_MT_7B_HIDDEN: usize = 4096;
pub const HY_MT_7B_LAYERS: usize = 32;

/// Effective RoPE base baked into official HY-MT1.5-1.8B GGUF
/// (`hunyuan-dense.rope.freq_base` ≈ 1.115884e7 from NTK dynamic α=1000).
pub const HY_MT_1_8B_ROPE_THETA_GGUF: f64 = 11_158_840.0;

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
    #[serde(default)]
    vocab_size: Option<usize>,
    #[serde(default)]
    intermediate_size: Option<usize>,
    #[serde(default)]
    num_attention_heads: Option<usize>,
    #[serde(default)]
    num_key_value_heads: Option<usize>,
    #[serde(default)]
    head_dim: Option<usize>,
    #[serde(default)]
    attention_head_dim: Option<usize>,
    #[serde(default)]
    max_position_embeddings: Option<usize>,
    #[serde(default)]
    rms_norm_eps: Option<f64>,
    #[serde(default)]
    rope_theta: Option<f64>,
    #[serde(default)]
    hidden_act: Option<String>,
    #[serde(default)]
    tie_word_embeddings: Option<bool>,
    #[serde(default)]
    attention_bias: Option<bool>,
    #[serde(default)]
    use_qk_norm: Option<bool>,
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

fn is_hy_mt_model_type(mt: &str) -> bool {
    let a = mt.to_ascii_lowercase();
    HF_MODEL_TYPES.iter().any(|t| t.eq_ignore_ascii_case(&a))
        || (a.contains("hunyuan") && a.contains("dense"))
}

fn is_hy_mt_arch_name(name: &str) -> bool {
    let a = name.to_ascii_lowercase().replace(['_', '-'], "");
    a.contains("hunyuandense")
}

/// Ensure HF `config.json` describes a dense Hunyuan / HY-MT checkpoint.
pub fn validate_hf_config(weights_or_dir: &Path) -> Result<()> {
    let cfg_path = config_json_path(weights_or_dir);
    let raw =
        std::fs::read_to_string(&cfg_path).with_context(|| format!("reading {cfg_path:?}"))?;
    let probe: HfConfigProbe =
        serde_json::from_str(&raw).with_context(|| format!("parsing {cfg_path:?}"))?;

    match probe.model_type.as_deref() {
        Some(mt) if is_hy_mt_model_type(mt) => {}
        Some(other) => {
            bail!("rlx-hy-mt: {cfg_path:?} has model_type={other:?}; expected hunyuan_v1_dense")
        }
        None => bail!("rlx-hy-mt: {cfg_path:?} missing model_type"),
    }

    if let Some(archs) = &probe.architectures {
        let ok = archs.iter().any(|a| is_hy_mt_arch_name(a));
        if !ok {
            bail!(
                "rlx-hy-mt: {cfg_path:?} architectures={archs:?}; \
                 expected HunYuanDenseV1ForCausalLM"
            );
        }
    }

    let hidden = probe
        .hidden_size
        .ok_or_else(|| anyhow::anyhow!("rlx-hy-mt: {cfg_path:?} missing hidden_size"))?;
    let layers = probe
        .num_hidden_layers
        .ok_or_else(|| anyhow::anyhow!("rlx-hy-mt: {cfg_path:?} missing num_hidden_layers"))?;
    validate_known_size(hidden, layers)
}

fn validate_known_size(hidden_size: usize, num_hidden_layers: usize) -> Result<()> {
    let ok_1_8b = hidden_size == HY_MT_1_8B_HIDDEN && num_hidden_layers == HY_MT_1_8B_LAYERS;
    let ok_7b = hidden_size == HY_MT_7B_HIDDEN && num_hidden_layers == HY_MT_7B_LAYERS;
    if ok_1_8b || ok_7b {
        return Ok(());
    }
    bail!(
        "rlx-hy-mt: unexpected dims hidden_size={hidden_size}, num_hidden_layers={num_hidden_layers}; \
         known: 1.8B ({HY_MT_1_8B_HIDDEN}/{HY_MT_1_8B_LAYERS}) or 7B ({HY_MT_7B_HIDDEN}/{HY_MT_7B_LAYERS})"
    )
}

/// Map HF Hunyuan dense `config.json` → [`Qwen3Config`] for the shared decoder.
pub fn qwen3_config_from_hf(weights_or_dir: &Path) -> Result<Qwen3Config> {
    let cfg_path = config_json_path(weights_or_dir);
    let raw =
        std::fs::read_to_string(&cfg_path).with_context(|| format!("reading {cfg_path:?}"))?;
    let probe: HfConfigProbe =
        serde_json::from_str(&raw).with_context(|| format!("parsing {cfg_path:?}"))?;

    let hidden_size = probe
        .hidden_size
        .ok_or_else(|| anyhow::anyhow!("missing hidden_size"))?;
    let num_attention_heads = probe
        .num_attention_heads
        .ok_or_else(|| anyhow::anyhow!("missing num_attention_heads"))?;
    let head_dim = probe
        .head_dim
        .or(probe.attention_head_dim)
        .unwrap_or_else(|| hidden_size / num_attention_heads.max(1));

    // Prefer the GGUF-baked NTK base for 1.8B so safetensors matches the
    // published quant; otherwise keep HF `rope_theta`.
    let rope_theta = if hidden_size == HY_MT_1_8B_HIDDEN {
        HY_MT_1_8B_ROPE_THETA_GGUF
    } else {
        probe.rope_theta.unwrap_or(10_000.0)
    };

    let mut cfg = Qwen3Config {
        vocab_size: probe.vocab_size.unwrap_or(120_818),
        hidden_size,
        intermediate_size: probe
            .intermediate_size
            .ok_or_else(|| anyhow::anyhow!("missing intermediate_size"))?,
        num_hidden_layers: probe
            .num_hidden_layers
            .ok_or_else(|| anyhow::anyhow!("missing num_hidden_layers"))?,
        num_attention_heads,
        num_key_value_heads: probe.num_key_value_heads.unwrap_or(num_attention_heads),
        head_dim,
        max_position_embeddings: probe.max_position_embeddings.unwrap_or(262_144),
        rms_norm_eps: probe.rms_norm_eps.unwrap_or(1e-5),
        rope_theta,
        hidden_act: probe.hidden_act.unwrap_or_else(|| "silu".into()),
        tie_word_embeddings: probe.tie_word_embeddings.unwrap_or(true),
        attention_bias: probe.attention_bias.unwrap_or(false),
        qk_norm: probe.use_qk_norm.unwrap_or(true),
        sliding_window: None,
        max_window_layers: 0,
        use_sliding_window: false,
        num_experts: 0,
        num_experts_used: 0,
        expert_ffn_size: 0,
        shared_expert_ffn_size: 0,
        expert_weights_scale: 1.0,
    };
    cfg.fill_derived_defaults()?;
    Ok(cfg)
}

/// GGUF arch tag or HF `config.json` checks.
pub fn validate_weights_kind(weights: &Path) -> Result<()> {
    match WeightFormat::from_path(weights)? {
        WeightFormat::Gguf => {
            let arch = gguf_architecture_from_path(weights)
                .with_context(|| format!("rlx-hy-mt: parse GGUF {weights:?}"))?;
            if !GGUF_ARCHES.iter().any(|a| a.eq_ignore_ascii_case(&arch)) {
                bail!(
                    "rlx-hy-mt: expected GGUF architecture in {GGUF_ARCHES:?}; \
                     got `{arch}` at {weights:?}"
                );
            }
        }
        WeightFormat::Safetensors => validate_hf_config(weights)?,
    }
    Ok(())
}

/// Reference 1.8B preset (matches official GGUF metadata).
pub fn hy_mt_1_8b_preset() -> Qwen3Config {
    Qwen3Config {
        vocab_size: 120_818,
        hidden_size: HY_MT_1_8B_HIDDEN,
        intermediate_size: 6144,
        num_hidden_layers: HY_MT_1_8B_LAYERS,
        num_attention_heads: 16,
        num_key_value_heads: 4,
        head_dim: 128,
        max_position_embeddings: 262_144,
        rms_norm_eps: 1e-5,
        rope_theta: HY_MT_1_8B_ROPE_THETA_GGUF,
        hidden_act: "silu".into(),
        tie_word_embeddings: true,
        attention_bias: false,
        qk_norm: true,
        sliding_window: None,
        max_window_layers: 0,
        use_sliding_window: false,
        num_experts: 0,
        num_experts_used: 0,
        expert_ffn_size: 0,
        shared_expert_ffn_size: 0,
        expert_weights_scale: 1.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preset_1_8b_qk_norm() {
        let p = hy_mt_1_8b_preset();
        assert!(p.qk_norm);
        assert!(!p.attention_bias);
        assert_eq!(p.hidden_size, 2048);
        assert_eq!(p.num_key_value_heads, 4);
    }

    #[test]
    fn validates_hf_probe() {
        let dir = std::env::temp_dir().join(format!(
            "rlx_hy_mt_cfg_{}_{}",
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
                "model_type": "hunyuan_v1_dense",
                "architectures": ["HunYuanDenseV1ForCausalLM"],
                "vocab_size": 120818,
                "hidden_size": 2048,
                "intermediate_size": 6144,
                "num_hidden_layers": 32,
                "num_attention_heads": 16,
                "num_key_value_heads": 4,
                "attention_head_dim": 128,
                "max_position_embeddings": 262144,
                "rms_norm_eps": 1e-5,
                "rope_theta": 10000,
                "use_qk_norm": true,
                "tie_word_embeddings": true
            }"#,
        )
        .unwrap();
        validate_hf_config(&dir).unwrap();
        let cfg = qwen3_config_from_hf(&dir).unwrap();
        assert!(cfg.qk_norm);
        assert_eq!(cfg.head_dim, 128);
        assert!((cfg.rope_theta - HY_MT_1_8B_ROPE_THETA_GGUF).abs() < 1.0);
        std::fs::remove_dir_all(&dir).ok();
    }
}
