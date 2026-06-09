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

//! Speaker encoder config — defaults match `Qwen3TTSSpeakerEncoderConfig`.
//!
//! The Base `config.json` only stores `sample_rate` and `enc_dim`; remaining
//! ECAPA params come from the HF defaults.

use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Clone, Deserialize)]
pub struct SpeakerEncoderConfig {
    #[serde(default = "default_mel_dim")]
    pub mel_dim: usize,
    #[serde(default = "default_sample_rate")]
    pub sample_rate: u32,
    #[serde(default = "default_enc_channels")]
    pub enc_channels: Vec<usize>,
    #[serde(default = "default_enc_kernel_sizes")]
    pub enc_kernel_sizes: Vec<usize>,
    #[serde(default = "default_enc_dilations")]
    pub enc_dilations: Vec<usize>,
    #[serde(default = "default_enc_attention_channels")]
    pub enc_attention_channels: usize,
    #[serde(default = "default_enc_res2net_scale")]
    pub enc_res2net_scale: usize,
    #[serde(default = "default_enc_se_channels")]
    pub enc_se_channels: usize,
    #[serde(default = "default_enc_dim")]
    pub enc_dim: usize,
}

fn default_mel_dim() -> usize {
    128
}
fn default_sample_rate() -> u32 {
    24_000
}
fn default_enc_channels() -> Vec<usize> {
    vec![512, 512, 512, 512, 1536]
}
fn default_enc_kernel_sizes() -> Vec<usize> {
    vec![5, 3, 3, 3, 1]
}
fn default_enc_dilations() -> Vec<usize> {
    vec![1, 2, 3, 4, 1]
}
fn default_enc_attention_channels() -> usize {
    128
}
fn default_enc_res2net_scale() -> usize {
    8
}
fn default_enc_se_channels() -> usize {
    128
}
fn default_enc_dim() -> usize {
    1024
}

impl Default for SpeakerEncoderConfig {
    fn default() -> Self {
        Self {
            mel_dim: default_mel_dim(),
            sample_rate: default_sample_rate(),
            enc_channels: default_enc_channels(),
            enc_kernel_sizes: default_enc_kernel_sizes(),
            enc_dilations: default_enc_dilations(),
            enc_attention_channels: default_enc_attention_channels(),
            enc_res2net_scale: default_enc_res2net_scale(),
            enc_se_channels: default_enc_se_channels(),
            enc_dim: default_enc_dim(),
        }
    }
}

impl SpeakerEncoderConfig {
    /// Reads `speaker_encoder_config` from a Qwen3-TTS Base `config.json`.
    pub fn from_model_dir(dir: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(dir.join("config.json"))
            .with_context(|| format!("read config.json under {}", dir.display()))?;
        let v: serde_json::Value = serde_json::from_str(&text).context("parse config.json")?;
        if let Some(sub) = v.get("speaker_encoder_config") {
            return serde_json::from_value(sub.clone()).context("parse speaker_encoder_config");
        }
        Ok(Self::default())
    }

    /// Mel-spectrogram params: (n_fft, hop, win, num_mels, fmin, fmax).
    /// Fixed at HF defaults — `extract_speaker_embedding` hardcodes them.
    pub fn mel_params(&self) -> MelParams {
        MelParams {
            n_fft: 1024,
            hop: 256,
            win: 1024,
            num_mels: self.mel_dim,
            fmin: 0.0,
            fmax: 12_000.0,
            sample_rate: self.sample_rate as f64,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MelParams {
    pub n_fft: usize,
    pub hop: usize,
    pub win: usize,
    pub num_mels: usize,
    pub fmin: f64,
    pub fmax: f64,
    pub sample_rate: f64,
}
