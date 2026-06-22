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

//! Qwen3-ASR configuration — mirrors HF `Qwen3ASRForConditionalGeneration`.
//!
//! The HF `config.json` nests everything under `thinker_config`:
//!   - `audio_config` → [`AudioEncoderConfig`] (Qwen3-Omni audio tower).
//!   - `text_config`  → [`rlx_qwen3::Qwen3Config`] (Qwen3 dense decoder).
//!   - `audio_token_id` / `audio_start_token_id` / `audio_end_token_id`.

use anyhow::{Context, Result};
use rlx_qwen3::Qwen3Config;
use serde::Deserialize;
use std::path::Path;

/// Qwen3-Omni audio encoder ("qwen3_asr_audio_encoder") parameters.
#[derive(Debug, Clone, Deserialize)]
pub struct AudioEncoderConfig {
    /// Transformer width (`d_model`).
    pub d_model: usize,
    /// Input log-mel channels.
    pub num_mel_bins: usize,
    /// Number of transformer encoder layers (`num_hidden_layers`; the HF
    /// config also carries a redundant `encoder_layers` that serde ignores).
    pub num_hidden_layers: usize,
    /// Attention heads per encoder layer.
    pub encoder_attention_heads: usize,
    /// Feed-forward inner width.
    pub encoder_ffn_dim: usize,
    /// Conv2d channel count (`downsample_hidden_size`).
    pub downsample_hidden_size: usize,
    /// Adapter output width (== text `hidden_size`).
    pub output_dim: usize,
    /// Length of the sinusoidal positional table.
    pub max_source_positions: usize,
    /// Half-window size in raw mel frames (chunk = `2 * n_window`).
    pub n_window: usize,
    /// Inference attention-window size in raw mel frames.
    pub n_window_infer: usize,
    /// Convolution batch chunk size (irrelevant for correctness, kept for parity).
    #[serde(default = "default_conv_chunksize")]
    pub conv_chunksize: usize,
    #[serde(default = "default_activation")]
    pub activation_function: String,
    #[serde(default)]
    pub scale_embedding: bool,
}

fn default_conv_chunksize() -> usize {
    500
}
fn default_activation() -> String {
    "gelu".into()
}

impl AudioEncoderConfig {
    /// Per-head attention dimension (`d_model / heads`).
    pub fn head_dim(&self) -> usize {
        self.d_model / self.encoder_attention_heads
    }

    /// Raw mel frames per pre-CNN chunk (`2 * n_window`).
    pub fn chunk_frames(&self) -> usize {
        self.n_window * 2
    }
}

#[derive(Debug, Clone, Deserialize)]
struct ThinkerConfig {
    audio_config: AudioEncoderConfig,
    text_config: Qwen3Config,
    audio_token_id: u32,
    audio_start_token_id: u32,
    audio_end_token_id: u32,
}

#[derive(Debug, Clone, Deserialize)]
struct RawConfig {
    thinker_config: ThinkerConfig,
}

/// Full Qwen3-ASR model configuration.
#[derive(Debug, Clone)]
pub struct Qwen3AsrConfig {
    pub audio: AudioEncoderConfig,
    pub text: Qwen3Config,
    /// `<|audio_pad|>` placeholder token id (151676).
    pub audio_token_id: u32,
    /// `<|audio_start|>` token id (151669).
    pub audio_start_token_id: u32,
    /// `<|audio_end|>` token id (151670).
    pub audio_end_token_id: u32,
}

impl Qwen3AsrConfig {
    pub fn from_file(path: &Path) -> Result<Self> {
        let data =
            std::fs::read_to_string(path).with_context(|| format!("reading config {path:?}"))?;
        Self::from_json(&data)
    }

    pub fn from_json(data: &str) -> Result<Self> {
        let raw: RawConfig = serde_json::from_str(data).context("parsing qwen3-asr config.json")?;
        let t = raw.thinker_config;
        Ok(Self {
            audio: t.audio_config,
            text: t.text_config,
            audio_token_id: t.audio_token_id,
            audio_start_token_id: t.audio_start_token_id,
            audio_end_token_id: t.audio_end_token_id,
        })
    }

    pub fn validate(&self) -> Result<()> {
        anyhow::ensure!(self.audio.d_model > 0, "audio d_model must be > 0");
        anyhow::ensure!(
            self.audio
                .d_model
                .is_multiple_of(self.audio.encoder_attention_heads),
            "audio d_model {} not divisible by heads {}",
            self.audio.d_model,
            self.audio.encoder_attention_heads
        );
        anyhow::ensure!(
            self.audio.output_dim == self.text.hidden_size,
            "audio output_dim {} != text hidden_size {}",
            self.audio.output_dim,
            self.text.hidden_size
        );
        anyhow::ensure!(
            self.text
                .num_attention_heads
                .is_multiple_of(self.text.num_key_value_heads),
            "text heads not divisible by kv heads"
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"{
        "model_type": "qwen3_asr",
        "thinker_config": {
            "audio_config": {
                "d_model": 896, "num_mel_bins": 128, "encoder_layers": 18,
                "encoder_attention_heads": 14, "encoder_ffn_dim": 3584,
                "downsample_hidden_size": 480, "output_dim": 1024,
                "max_source_positions": 1500, "n_window": 50,
                "n_window_infer": 800, "conv_chunksize": 500,
                "num_hidden_layers": 18, "activation_function": "gelu",
                "scale_embedding": false
            },
            "audio_token_id": 151676,
            "audio_start_token_id": 151669,
            "audio_end_token_id": 151670,
            "text_config": {
                "vocab_size": 151936, "hidden_size": 1024, "intermediate_size": 3072,
                "num_hidden_layers": 28, "num_attention_heads": 16,
                "num_key_value_heads": 8, "head_dim": 128,
                "max_position_embeddings": 65536, "rope_theta": 1000000,
                "rms_norm_eps": 1e-6, "tie_word_embeddings": true, "model_type": "qwen3"
            }
        },
        "transformers_version": "4.57.6"
    }"#;

    #[test]
    fn parses_sample_config() {
        let cfg = Qwen3AsrConfig::from_json(SAMPLE).unwrap();
        cfg.validate().unwrap();
        assert_eq!(cfg.audio.d_model, 896);
        assert_eq!(cfg.audio.head_dim(), 64);
        assert_eq!(cfg.audio.chunk_frames(), 100);
        assert_eq!(cfg.audio.num_hidden_layers, 18);
        assert_eq!(cfg.text.num_hidden_layers, 28);
        assert_eq!(cfg.text.head_dim, 128);
        assert_eq!(cfg.audio_token_id, 151676);
        assert!(cfg.text.tie_word_embeddings);
    }
}
