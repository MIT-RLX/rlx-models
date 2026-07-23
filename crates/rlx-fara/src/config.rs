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

//! Fara1.5 size presets and HuggingFace ids.

use anyhow::{Result, bail};
use rlx_qwen35::{MmProjConfig, Qwen35Config};
use std::path::{Path, PathBuf};

pub const FAMILY: &str = "Fara1.5";
pub const HF_MODEL_ID_4B: &str = "microsoft/Fara1.5-4B";
pub const HF_MODEL_ID_9B: &str = "microsoft/Fara1.5-9B";

/// Vision / special token ids shared by Fara1.5-4B and 9B.
pub const IMAGE_TOKEN_ID: u32 = 248_056;
pub const VIDEO_TOKEN_ID: u32 = 248_057;
pub const VISION_START_TOKEN_ID: u32 = 248_053;
pub const VISION_END_TOKEN_ID: u32 = 248_054;
pub const EOS_TOKEN_ID: u32 = 248_044;

/// Recommended sandbox resolution from the Fara model card.
pub const TRAIN_SCREEN_W: usize = 1440;
pub const TRAIN_SCREEN_H: usize = 900;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FaraSize {
    B4,
    B9,
}

impl FaraSize {
    pub fn parse(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "4b" | "4" | "fara1.5-4b" | "fara-4b" => Ok(Self::B4),
            "9b" | "9" | "fara1.5-9b" | "fara-9b" => Ok(Self::B9),
            other => bail!("unknown Fara size `{other}` (expected 4b or 9b)"),
        }
    }

    pub fn hf_model_id(self) -> &'static str {
        match self {
            Self::B4 => HF_MODEL_ID_4B,
            Self::B9 => HF_MODEL_ID_9B,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::B4 => "4B",
            Self::B9 => "9B",
        }
    }

    pub fn cache_subdir(self) -> &'static str {
        match self {
            Self::B4 => "4b",
            Self::B9 => "9b",
        }
    }
}

/// Built-in Qwen3.5 text config matching the published Fara HF cards.
pub fn fara_qwen35_config(size: FaraSize) -> Qwen35Config {
    match size {
        FaraSize::B4 => Qwen35Config {
            vocab_size: 248_320,
            hidden_size: 2560,
            intermediate_size: 9216,
            num_hidden_layers: 32,
            nextn_predict_layers: 0,
            num_attention_heads: 16,
            num_key_value_heads: 4,
            key_length: 256,
            value_length: 256,
            max_position_embeddings: 262_144,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000_000.0,
            rope_dim_count: 64,
            rope_dim_sections: vec![11, 11, 10],
            mrope_interleaved: true,
            rms_norm_offset: true,
            full_attention_interval: 4,
            ssm_conv_kernel: 4,
            ssm_group_count: 16,
            ssm_inner_size: 4096,
            ssm_state_size: 128,
            // Must match HF `linear_num_value_heads` (not key heads).
            ssm_time_step_rank: 32,
            tie_word_embeddings: true,
            num_experts: 0,
            num_experts_used: 0,
            expert_ffn_size: 0,
            shared_expert_ffn_size: 0,
            expert_weights_scale: 1.0,
        },
        FaraSize::B9 => Qwen35Config {
            vocab_size: 248_320,
            hidden_size: 4096,
            intermediate_size: 12288,
            num_hidden_layers: 32,
            nextn_predict_layers: 0,
            num_attention_heads: 16,
            num_key_value_heads: 4,
            key_length: 256,
            value_length: 256,
            max_position_embeddings: 262_144,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000_000.0,
            rope_dim_count: 64,
            rope_dim_sections: vec![11, 11, 10],
            mrope_interleaved: true,
            rms_norm_offset: true,
            full_attention_interval: 4,
            ssm_conv_kernel: 4,
            ssm_group_count: 16,
            ssm_inner_size: 4096,
            ssm_state_size: 128,
            // Must match HF `linear_num_value_heads` (not key heads).
            ssm_time_step_rank: 32,
            tie_word_embeddings: false,
            num_experts: 0,
            num_experts_used: 0,
            expert_ffn_size: 0,
            shared_expert_ffn_size: 0,
            expert_weights_scale: 1.0,
        },
    }
}

/// Built-in vision tower config matching the published Fara HF cards.
pub fn fara_vision_config(size: FaraSize) -> MmProjConfig {
    match size {
        FaraSize::B4 => MmProjConfig {
            patch_size: 16,
            n_embd: 1024,
            n_head: 16,
            n_layer: 24,
            image_size: 768,
            image_min_pixels: 1024 * 2 * 2 * 16 * 16,
            image_max_pixels: 4096 * 2 * 2 * 16 * 16,
            n_merge: 2,
            eps: 1e-6,
            projector_type: "qwen3vl".into(),
            image_mean: [0.5, 0.5, 0.5],
            image_std: [0.5, 0.5, 0.5],
            spatial_merge_size: 2,
            llm_hidden_size: 2560,
            n_ff: 4096,
            deepstack_layers: vec![],
        },
        FaraSize::B9 => MmProjConfig {
            patch_size: 16,
            n_embd: 1152,
            n_head: 16,
            n_layer: 27,
            image_size: 768,
            image_min_pixels: 1024 * 2 * 2 * 16 * 16,
            image_max_pixels: 4096 * 2 * 2 * 16 * 16,
            n_merge: 2,
            eps: 1e-6,
            projector_type: "qwen3vl".into(),
            image_mean: [0.5, 0.5, 0.5],
            image_std: [0.5, 0.5, 0.5],
            spatial_merge_size: 2,
            llm_hidden_size: 4096,
            n_ff: 4304,
            deepstack_layers: vec![],
        },
    }
}

/// Default local cache root: `.cache/fara/`.
pub fn default_cache_root() -> PathBuf {
    PathBuf::from(".cache/fara")
}

pub fn default_model_dir(size: FaraSize) -> PathBuf {
    default_cache_root().join(size.cache_subdir())
}

/// True when `dir` looks like a usable Fara / Qwen3.5 multimodal snapshot.
pub fn is_model_dir(dir: &Path) -> bool {
    dir.join("config.json").is_file()
        && (dir.join("model.safetensors.index.json").is_file()
            || dir.join("model.safetensors").is_file()
            || dir
                .read_dir()
                .ok()
                .into_iter()
                .flatten()
                .filter_map(|e| e.ok())
                .any(|e| {
                    e.path()
                        .extension()
                        .and_then(|s| s.to_str())
                        == Some("safetensors")
                }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn size_parse_and_presets() {
        assert_eq!(FaraSize::parse("4b").unwrap(), FaraSize::B4);
        assert_eq!(FaraSize::parse("9B").unwrap(), FaraSize::B9);
        let c4 = fara_qwen35_config(FaraSize::B4);
        assert_eq!(c4.hidden_size, 2560);
        assert!(c4.tie_word_embeddings);
        let c9 = fara_qwen35_config(FaraSize::B9);
        assert_eq!(c9.hidden_size, 4096);
        assert!(!c9.tie_word_embeddings);
        assert_eq!(fara_vision_config(FaraSize::B4).n_layer, 24);
        assert_eq!(fara_vision_config(FaraSize::B9).n_layer, 27);
    }
}
