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

//! TADA configuration.
//!
//! `TadaConfig` mirrors `tada.modules.tada.TadaConfig` (a `LlamaConfig`
//! subclass) and is read straight from the checkpoint's `config.json`. The
//! codec sub-modules (`Encoder` / `Decoder` / `Aligner`) ship a `config.json`
//! that carries only `architectures` + `dtype`, so their shapes live as
//! defaults here — exactly as they do in the upstream Python classes.

use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::Path;

/// Frames per second of the acoustic token grid (24 kHz / 480).
pub const FRAME_RATE: usize = 50;
/// Codec sample rate.
pub const SAMPLE_RATE: usize = 24_000;
/// Sample rate the aligner's wav2vec2 stack expects.
pub const ALIGNER_SAMPLE_RATE: usize = 16_000;

fn d_acoustic_dim() -> usize {
    512
}
fn d_num_time_classes() -> usize {
    1024
}
fn d_shift_acoustic() -> usize {
    5
}
fn d_head_layers() -> usize {
    4
}
fn d_head_ffn_ratio() -> f32 {
    3.0
}
fn d_acoustic_std() -> f32 {
    1.5
}
fn d_rms_eps() -> f32 {
    1e-5
}
fn d_rope_theta() -> f32 {
    500_000.0
}
fn d_true() -> bool {
    true
}

/// Rope scaling block (`rope_type: "llama3"` for every shipped TADA variant).
#[derive(Debug, Clone, Deserialize)]
pub struct RopeScaling {
    pub factor: f32,
    pub high_freq_factor: f32,
    pub low_freq_factor: f32,
    pub original_max_position_embeddings: usize,
    #[serde(default)]
    pub rope_type: String,
}

/// The TADA causal LM config — a Llama config plus the acoustic/time fields.
#[derive(Debug, Clone, Deserialize)]
pub struct TadaConfig {
    // --- Llama backbone ---
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub vocab_size: usize,
    #[serde(default)]
    pub head_dim: Option<usize>,
    #[serde(default = "d_rms_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "d_rope_theta")]
    pub rope_theta: f32,
    #[serde(default)]
    pub rope_scaling: Option<RopeScaling>,
    #[serde(default)]
    pub max_position_embeddings: usize,
    #[serde(default = "d_true")]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub bos_token_id: u32,
    #[serde(default)]
    pub eos_token_id: u32,

    // --- TADA extensions ---
    #[serde(default = "d_acoustic_dim")]
    pub acoustic_dim: usize,
    #[serde(default = "d_num_time_classes")]
    pub num_time_classes: usize,
    #[serde(default = "d_shift_acoustic")]
    pub shift_acoustic: usize,
    #[serde(default = "d_head_layers")]
    pub head_layers: usize,
    #[serde(default = "d_head_ffn_ratio")]
    pub head_ffn_ratio: f32,
    #[serde(default)]
    pub bottleneck_dim: Option<usize>,
    #[serde(default)]
    pub acoustic_mean: f32,
    #[serde(default = "d_acoustic_std")]
    pub acoustic_std: f32,
}

impl TadaConfig {
    pub fn from_file(path: &Path) -> Result<Self> {
        let raw =
            std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
        let cfg: Self =
            serde_json::from_str(&raw).with_context(|| format!("parse {}", path.display()))?;
        cfg.validate()?;
        Ok(cfg)
    }

    fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            self.num_time_classes.is_power_of_two(),
            "num_time_classes {} is not a power of two — the Gray-code time \
             field assumes ceil(log2(classes)) bits round-trips exactly",
            self.num_time_classes
        );
        anyhow::ensure!(
            self.num_attention_heads
                .is_multiple_of(self.num_key_value_heads),
            "num_attention_heads {} not divisible by num_key_value_heads {}",
            self.num_attention_heads,
            self.num_key_value_heads
        );
        Ok(())
    }

    /// Bits used by one Gray-coded frame gap (`ceil(log2(num_time_classes))`).
    pub fn num_time_bits(&self) -> usize {
        self.num_time_classes.next_power_of_two().trailing_zeros() as usize
    }

    /// Width of the time field the diffusion head predicts: one Gray code for
    /// the gap *before* the token and one for the gap *after*.
    pub fn time_dim(&self) -> usize {
        2 * self.num_time_bits()
    }

    /// Full latent the diffusion head denoises: acoustic ‖ time.
    pub fn latent_dim(&self) -> usize {
        self.acoustic_dim + self.time_dim()
    }

    /// Conditioning width fed to the head (`bottleneck_dim` when set).
    pub fn cond_dim(&self) -> usize {
        self.bottleneck_dim.unwrap_or(self.hidden_size)
    }

    /// Number of `<|eot_id|>` tokens appended to the text — upstream ties this
    /// to `shift_acoustic` so the last real token still gets `shift` steps of
    /// acoustic lookahead.
    pub fn num_eos_tokens(&self) -> usize {
        self.shift_acoustic
    }
}

/// Codec encoder shape constants (`tada.modules.encoder.EncoderConfig`).
#[derive(Debug, Clone)]
pub struct EncoderConfig {
    pub hidden_dim: usize,
    pub embed_dim: usize,
    /// Downsampling strides, outermost first. Product must be
    /// `SAMPLE_RATE / FRAME_RATE`.
    pub strides: Vec<usize>,
    pub num_attn_layers: usize,
    pub num_attn_heads: usize,
    pub attn_dim_feedforward: usize,
}

impl Default for EncoderConfig {
    fn default() -> Self {
        Self {
            hidden_dim: 1024,
            embed_dim: 512,
            strides: vec![6, 5, 4, 4],
            num_attn_layers: 6,
            num_attn_heads: 8,
            attn_dim_feedforward: 4096,
        }
    }
}

impl EncoderConfig {
    /// Samples of audio consumed per acoustic frame.
    pub fn hop(&self) -> usize {
        self.strides.iter().product()
    }
}

/// Codec decoder shape constants (`tada.modules.decoder.DecoderConfig`).
#[derive(Debug, Clone)]
pub struct DecoderConfig {
    pub embed_dim: usize,
    pub hidden_dim: usize,
    pub num_attn_layers: usize,
    pub num_attn_heads: usize,
    pub attn_dim_feedforward: usize,
    pub wav_decoder_channels: usize,
    /// Upsampling strides, innermost first.
    pub strides: Vec<usize>,
}

impl Default for DecoderConfig {
    fn default() -> Self {
        Self {
            embed_dim: 512,
            hidden_dim: 1024,
            num_attn_layers: 6,
            num_attn_heads: 8,
            attn_dim_feedforward: 4096,
            wav_decoder_channels: 1536,
            strides: vec![4, 4, 5, 6],
        }
    }
}

impl DecoderConfig {
    pub fn hop(&self) -> usize {
        self.strides.iter().product()
    }
}

/// Aligner backbone shape (`facebook/wav2vec2-large` with the Llama vocabulary
/// bolted onto the CTC head).
#[derive(Debug, Clone)]
pub struct AlignerConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub layer_norm_eps: f32,
    /// Conv feature extractor: `(out_channels, kernel, stride)` per layer.
    pub conv_layers: Vec<(usize, usize, usize)>,
    pub conv_dim: usize,
    pub num_conv_pos_embeddings: usize,
    pub num_conv_pos_embedding_groups: usize,
    pub feat_extract_norm_groups: usize,
}

impl Default for AlignerConfig {
    fn default() -> Self {
        Self {
            hidden_size: 1024,
            num_hidden_layers: 24,
            num_attention_heads: 16,
            intermediate_size: 4096,
            vocab_size: 128_256,
            layer_norm_eps: 1e-5,
            conv_layers: vec![
                (512, 10, 5),
                (512, 3, 2),
                (512, 3, 2),
                (512, 3, 2),
                (512, 3, 2),
                (512, 2, 2),
                (512, 2, 2),
            ],
            conv_dim: 512,
            num_conv_pos_embeddings: 128,
            num_conv_pos_embedding_groups: 16,
            feat_extract_norm_groups: 512,
        }
    }
}

impl AlignerConfig {
    /// Total temporal downsampling of the conv feature extractor (320 for
    /// wav2vec2-large → 50 Hz at 16 kHz, matching [`FRAME_RATE`]).
    pub fn downsample(&self) -> usize {
        self.conv_layers.iter().map(|&(_, _, s)| s).product()
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg_1b() -> TadaConfig {
        serde_json::from_str(
            r#"{"hidden_size":2048,"intermediate_size":8192,"num_hidden_layers":16,
                "num_attention_heads":32,"num_key_value_heads":8,"vocab_size":128256,
                "head_dim":64,"acoustic_dim":512,"num_time_classes":256,
                "shift_acoustic":5,"head_layers":6,"head_ffn_ratio":4.0,
                "acoustic_std":1.5,"max_position_embeddings":131072}"#,
        )
        .unwrap()
    }

    #[test]
    fn derived_widths_match_checkpoint_shapes() {
        let c = cfg_1b();
        // prediction_head.final_layer.linear.weight is [528, 2048] in tada-1b.
        assert_eq!(c.num_time_bits(), 8);
        assert_eq!(c.time_dim(), 16);
        assert_eq!(c.latent_dim(), 528);
        assert_eq!(c.cond_dim(), 2048);
        assert_eq!(c.num_eos_tokens(), 5);
    }

    #[test]
    fn codec_strides_land_on_the_frame_rate() {
        assert_eq!(EncoderConfig::default().hop(), SAMPLE_RATE / FRAME_RATE);
        assert_eq!(DecoderConfig::default().hop(), SAMPLE_RATE / FRAME_RATE);
        assert_eq!(
            AlignerConfig::default().downsample(),
            ALIGNER_SAMPLE_RATE / FRAME_RATE
        );
    }

    #[test]
    fn non_power_of_two_time_classes_are_rejected() {
        let mut c = cfg_1b();
        c.num_time_classes = 300;
        assert!(c.validate().is_err());
    }
}
