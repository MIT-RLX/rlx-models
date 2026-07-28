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

//! Voxtral configuration — HuggingFace `config.json` (`mistralai/Voxtral-Mini-3B-2507`).

use anyhow::{Context, Result, ensure};
use rlx_llama32::Llama32Config;
use serde::Deserialize;
use std::path::Path;

/// Whisper-style audio encoder section (`audio_config` in HF JSON).
#[derive(Debug, Clone, Deserialize)]
pub struct VoxtralAudioConfig {
    pub num_mel_bins: usize,
    pub max_source_positions: usize,
    #[serde(rename = "hidden_size", alias = "d_model")]
    pub d_model: usize,
    #[serde(rename = "num_attention_heads", alias = "encoder_attention_heads")]
    pub encoder_attention_heads: usize,
    #[serde(rename = "num_hidden_layers", alias = "encoder_layers")]
    pub encoder_layers: usize,
    pub intermediate_size: usize,
    #[serde(default)]
    pub scale_embedding: bool,
}

impl VoxtralAudioConfig {
    pub fn head_dim(&self) -> usize {
        self.d_model / self.encoder_attention_heads
    }

    /// Sequence length after two stride convolutions (same as Whisper).
    pub fn encoder_seq_len(&self, mel_frames: usize) -> usize {
        let after_conv1 = mel_frames;
        let pad = 1usize;
        let k = 3usize;
        let stride2 = 2usize;
        (after_conv1 + 2 * pad - k) / stride2 + 1
    }

    /// Audio frames after the 4× projector grouping.
    pub fn audio_token_count(&self, mel_frames: usize) -> usize {
        self.encoder_seq_len(mel_frames) / 4
    }

    pub fn tiny_synthetic() -> Self {
        Self {
            num_mel_bins: 4,
            max_source_positions: 16,
            d_model: 8,
            encoder_attention_heads: 2,
            encoder_layers: 1,
            intermediate_size: 32,
            scale_embedding: false,
        }
    }

    pub fn mini_3b() -> Self {
        Self {
            num_mel_bins: 128,
            max_source_positions: 1500,
            d_model: 1280,
            encoder_attention_heads: 20,
            encoder_layers: 32,
            intermediate_size: 5120,
            scale_embedding: false,
        }
    }
}

/// Top-level Voxtral checkpoint config.
#[derive(Debug, Clone, Deserialize)]
pub struct VoxtralConfig {
    pub audio_config: VoxtralAudioConfig,
    pub text_config: Llama32Config,
    #[serde(default = "default_audio_token_id")]
    pub audio_token_id: u32,
    /// End-of-stream token that halts generation (tekken `</s>` = 2).
    #[serde(default = "default_eos_token_id")]
    pub eos_token_id: u32,
    #[serde(default = "default_projector_act")]
    pub projector_hidden_act: String,
    pub vocab_size: usize,
}

fn default_audio_token_id() -> u32 {
    24
}

fn default_eos_token_id() -> u32 {
    2
}

fn default_projector_act() -> String {
    "gelu".into()
}

impl VoxtralConfig {
    pub fn from_file(path: &Path) -> Result<Self> {
        let data = std::fs::read_to_string(path)?;
        serde_json::from_str(&data).with_context(|| format!("parse Voxtral config {path:?}"))
    }

    pub fn llama_config(&self) -> &Llama32Config {
        &self.text_config
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.text_config.hidden_size > 0,
            "text_config.hidden_size must be > 0"
        );
        ensure!(
            self.audio_config.intermediate_size == self.audio_config.d_model * 4,
            "audio_config.intermediate_size should be 4× d_model for the projector reshape"
        );
        Ok(())
    }

    pub fn tiny_synthetic() -> Self {
        Self {
            audio_config: VoxtralAudioConfig::tiny_synthetic(),
            text_config: Llama32Config {
                embedding_scale: None,
                residual_scale: None,
                attention_scale: None,
                logit_scale: None,
                vocab_size: 32,
                hidden_size: 16,
                intermediate_size: 32,
                num_hidden_layers: 1,
                num_attention_heads: 4,
                num_key_value_heads: 2,
                max_position_embeddings: 16,
                rms_norm_eps: 1e-5,
                rope_theta: 100_000_000.0,
                hidden_act: "silu".into(),
                tie_word_embeddings: true,
                attention_bias: false,
                head_dim: Some(4),
                rope_scaling: None,
                num_loops: 1,
                skip_loop_final_norm: false,
                rope_style: rlx_ir::RopeStyle::NeoX,
                gguf_arch: None,
                rope_dim: None,
            },
            audio_token_id: 24,
            eos_token_id: 2,
            projector_hidden_act: "gelu".into(),
            vocab_size: 32,
        }
    }

    pub fn mini_3b() -> Self {
        Self {
            audio_config: VoxtralAudioConfig::mini_3b(),
            text_config: Llama32Config {
                embedding_scale: None,
                residual_scale: None,
                attention_scale: None,
                logit_scale: None,
                vocab_size: 131_072,
                hidden_size: 3072,
                intermediate_size: 8192,
                num_hidden_layers: 30,
                num_attention_heads: 32,
                num_key_value_heads: 8,
                max_position_embeddings: 131_072,
                rms_norm_eps: 1e-5,
                rope_theta: 100_000_000.0,
                hidden_act: "silu".into(),
                tie_word_embeddings: true,
                attention_bias: false,
                head_dim: Some(128),
                rope_scaling: None,
                num_loops: 1,
                skip_loop_final_norm: false,
                rope_style: rlx_ir::RopeStyle::NeoX,
                gguf_arch: None,
                rope_dim: None,
            },
            audio_token_id: 24,
            eos_token_id: 2,
            projector_hidden_act: "gelu".into(),
            vocab_size: 131_072,
        }
    }
}
