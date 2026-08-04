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

//! # rlx-inflect-v2
//!
//! **Inflect v2** is a VITS-style end-to-end flow TTS (espeak phonemes → text
//! encoder + stochastic duration predictor + normalizing flow → HiFiGAN-style
//! decoder, 24 kHz). Note this is a *different architecture* from Inflect-Nano v1
//! ([`rlx_inflect_nano`], a mel-acoustic model + separate vocoder) — v2 is a VITS2
//! family model, so the synthesis graph is shared with
//! [`rlx-tiny-tts`](https://docs.rs/rlx-tiny-tts) (MeloTTS / VITS2), which already
//! runs on all RLX backends.
//!
//! This crate currently provides the model-defining pieces that are checkpoint-
//! free and unit-testable:
//!
//! - [`InflectV2Config`] — the architecture config, ported from audio.cpp's
//!   `InflectV2Config` (`community_models/inflect_v2`).
//! - [`GenerationOptions`] — VITS sampling controls (`speaking_rate`, `variation`
//!   noise scale, `seed`).
//! - [`sample_flow_prior`] — the VITS flow prior `z ~ variation · N(0, I)`, drawn
//!   from the shared seedable RNG in [`rlx_audio_blocks::sampling`] so the same
//!   seed yields the same audio across backends.
//!
//! **Next step** (needs a checkpoint): map [`InflectV2Config`] onto
//! `rlx_tiny_tts::BundleConfig`, wire the espeak phoneme frontend, and drive
//! `rlx_tiny_tts::TinyModel` for end-to-end synthesis + per-backend parity.

use anyhow::{Result, ensure};
use rlx_audio_blocks::sampling::Rng;

/// Inflect v2 architecture config. Fields defaulted to `0` here are read from the
/// checkpoint (the audio.cpp header defaults them to `0` too); the rest carry the
/// model's fixed topology.
#[derive(Debug, Clone, PartialEq)]
pub struct InflectV2Config {
    /// Phoneme vocabulary size.
    pub vocab_size: usize,
    /// Output sample rate (Hz).
    pub sample_rate: usize,
    /// Decoder hop length (samples per frame).
    pub hop_length: usize,
    /// Flow / prior latent channels (from checkpoint).
    pub inter_channels: usize,
    /// Text-encoder hidden channels (from checkpoint).
    pub hidden_channels: usize,
    /// Text-encoder FFN channels (from checkpoint).
    pub filter_channels: usize,
    /// Duration-predictor channels.
    pub duration_channels: usize,
    /// Text-encoder attention heads.
    pub attention_heads: usize,
    /// Text-encoder transformer layers.
    pub encoder_layers: usize,
    /// Coupling layers per flow.
    pub flow_layers: usize,
    /// Number of normalizing flows.
    pub flow_count: usize,
    /// HiFiGAN decoder initial channels (from checkpoint).
    pub upsample_initial_channels: usize,
    pub upsample_rates: Vec<usize>,
    pub upsample_kernel_sizes: Vec<usize>,
    pub resblock_kernel_sizes: Vec<usize>,
    pub resblock_dilations: Vec<Vec<usize>>,
    /// Free-form variant tag from the bundle.
    pub variant: String,
}

impl Default for InflectV2Config {
    /// Matches the audio.cpp `InflectV2Config` header defaults.
    fn default() -> Self {
        Self {
            vocab_size: 178,
            sample_rate: 24_000,
            hop_length: 256,
            inter_channels: 0,
            hidden_channels: 0,
            filter_channels: 0,
            duration_channels: 256,
            attention_heads: 2,
            encoder_layers: 3,
            flow_layers: 4,
            flow_count: 4,
            upsample_initial_channels: 0,
            upsample_rates: Vec::new(),
            upsample_kernel_sizes: Vec::new(),
            resblock_kernel_sizes: Vec::new(),
            resblock_dilations: Vec::new(),
            variant: String::new(),
        }
    }
}

impl InflectV2Config {
    /// Acoustic frame rate: `sample_rate / hop_length` (frames per second).
    pub fn frames_per_second(&self) -> f32 {
        self.sample_rate as f32 / self.hop_length as f32
    }

    /// Validate the topology fields that must be non-zero regardless of the
    /// checkpoint (the `0`-defaulted channel widths are filled in at load time).
    pub fn validate(&self) -> Result<()> {
        ensure!(self.vocab_size > 0, "inflect-v2 vocab_size must be > 0");
        ensure!(self.sample_rate > 0, "inflect-v2 sample_rate must be > 0");
        ensure!(self.hop_length > 0, "inflect-v2 hop_length must be > 0");
        ensure!(self.flow_count > 0, "inflect-v2 flow_count must be > 0");
        ensure!(self.flow_layers > 0, "inflect-v2 flow_layers must be > 0");
        ensure!(
            self.encoder_layers > 0,
            "inflect-v2 encoder_layers must be > 0"
        );
        Ok(())
    }
}

/// VITS generation controls, ported from audio.cpp `InflectV2GenerationOptions`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GenerationOptions {
    /// Duration scale (>1 slower, <1 faster).
    pub speaking_rate: f32,
    /// Prior noise scale (VITS `noise_scale`); controls prosodic variation.
    pub variation: f32,
    /// RNG seed for reproducible sampling.
    pub seed: u64,
}

impl Default for GenerationOptions {
    fn default() -> Self {
        Self {
            speaking_rate: 1.0,
            variation: 0.667,
            seed: 0,
        }
    }
}

/// Draw the VITS flow prior `z` of shape `[channels, frames]` (row-major), equal
/// to `variation · N(0, I)`, from the shared seedable RNG. Deterministic for a
/// given `seed`, so two RLX backends fed the same options produce identical noise.
pub fn sample_flow_prior(channels: usize, frames: usize, opts: &GenerationOptions) -> Vec<f32> {
    let mut rng = Rng::seeded(opts.seed);
    let n = channels * frames;
    let mut z = rng.standard_normal_vec(n);
    if opts.variation != 1.0 {
        for v in z.iter_mut() {
            *v *= opts.variation;
        }
    }
    z
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_matches_audio_cpp() {
        let c = InflectV2Config::default();
        assert_eq!(c.vocab_size, 178);
        assert_eq!(c.sample_rate, 24_000);
        assert_eq!(c.hop_length, 256);
        assert_eq!(c.duration_channels, 256);
        assert_eq!(c.attention_heads, 2);
        assert_eq!(c.encoder_layers, 3);
        assert_eq!(c.flow_layers, 4);
        assert_eq!(c.flow_count, 4);
        c.validate().unwrap();
    }

    #[test]
    fn frame_rate_is_sr_over_hop() {
        let c = InflectV2Config::default();
        assert!((c.frames_per_second() - 24_000.0 / 256.0).abs() < 1e-3);
    }

    #[test]
    fn generation_defaults() {
        let o = GenerationOptions::default();
        assert_eq!(o.speaking_rate, 1.0);
        assert!((o.variation - 0.667).abs() < 1e-6);
        assert_eq!(o.seed, 0);
    }

    #[test]
    fn flow_prior_is_deterministic_and_shaped() {
        let opts = GenerationOptions::default();
        let a = sample_flow_prior(4, 10, &opts);
        let b = sample_flow_prior(4, 10, &opts);
        assert_eq!(a.len(), 40);
        assert_eq!(a, b, "same seed must reproduce the prior");
    }

    #[test]
    fn variation_scales_prior_magnitude() {
        let base = GenerationOptions {
            variation: 1.0,
            ..Default::default()
        };
        let scaled = GenerationOptions {
            variation: 3.0,
            ..Default::default()
        };
        let n = 20_000;
        let a = sample_flow_prior(1, n, &base);
        let b = sample_flow_prior(1, n, &scaled);
        let std = |v: &[f32]| -> f32 {
            let m = v.iter().sum::<f32>() / v.len() as f32;
            (v.iter().map(|x| (x - m).powi(2)).sum::<f32>() / v.len() as f32).sqrt()
        };
        let (sa, sb) = (std(&a), std(&b));
        // b is drawn with 3x the noise scale → ~3x the standard deviation.
        assert!((sb / sa - 3.0).abs() < 0.1, "ratio={}", sb / sa);
    }

    #[test]
    fn zero_variation_is_deterministic_silence_prior() {
        let opts = GenerationOptions {
            variation: 0.0,
            ..Default::default()
        };
        let z = sample_flow_prior(2, 8, &opts);
        assert!(z.iter().all(|&v| v == 0.0));
    }
}
