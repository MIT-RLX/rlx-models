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

//! `config.json` of a LiquidCodec checkpoint (`LiquidCodecConfig`).

use std::path::Path;

use anyhow::{Context, Result, bail};
use serde::Deserialize;

/// The codec's shape. Fields default to the released values, so the
/// `audio_tokenizer/config.json` inside LFM3-Audio checkpoints (same fields)
/// and a missing file both work.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(default)]
pub struct LiquidCodecConfig {
    /// Width of the encoder and decoder LFM3 stacks.
    pub encoder_dim: usize,
    /// Channels of the STFT front end.
    pub encoder_conv_dim: usize,
    pub stft_n_fft: usize,
    pub stft_hop_length: usize,
    pub stft_win_length: usize,
    /// STFT frames per codec frame.
    pub stft_frame_stack: usize,
    /// Width of the latent handed to the quantizer.
    pub quantizer_dim: usize,
    /// Levels of each FSQ dimension; their product is the codebook size.
    pub fsq_levels: Vec<usize>,
    /// Codebooks (tokens per frame).
    pub n_q: usize,
    /// Sliding attention window of the decoder, in STFT frames.
    pub decoder_window: usize,
    pub sampling_rate: u32,
}

impl Default for LiquidCodecConfig {
    fn default() -> Self {
        Self {
            encoder_dim: 512,
            encoder_conv_dim: 256,
            stft_n_fft: 1280,
            stft_hop_length: 320,
            stft_win_length: 1280,
            stft_frame_stack: 6,
            quantizer_dim: 256,
            fsq_levels: vec![8, 7, 6, 6],
            n_q: 12,
            decoder_window: 30,
            sampling_rate: 24_000,
        }
    }
}

impl LiquidCodecConfig {
    /// Read `config.json`, or the defaults when `path` does not exist.
    pub fn load(path: &Path) -> Result<Self> {
        if !path.exists() {
            return Ok(Self::default());
        }
        let text =
            std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
        let cfg: Self =
            serde_json::from_str(&text).with_context(|| format!("parse {}", path.display()))?;
        cfg.validate()?;
        Ok(cfg)
    }

    fn validate(&self) -> Result<()> {
        if self.stft_win_length != self.stft_n_fft {
            bail!(
                "STFT window {} differs from n_fft {}: not a released LiquidCodec layout",
                self.stft_win_length,
                self.stft_n_fft
            );
        }
        if self.stft_hop_length == 0 || self.stft_hop_length > self.stft_n_fft {
            bail!(
                "STFT hop {} for n_fft {}",
                self.stft_hop_length,
                self.stft_n_fft
            );
        }
        if self.fsq_levels.is_empty() || self.fsq_levels.contains(&0) || self.n_q == 0 {
            bail!("FSQ levels {:?} × {} codebooks", self.fsq_levels, self.n_q);
        }
        Ok(())
    }

    /// Entries per codebook (the product of the FSQ levels).
    pub fn codebook_size(&self) -> usize {
        self.fsq_levels.iter().product()
    }

    /// Dimensions per codebook.
    pub fn codebook_dim(&self) -> usize {
        self.fsq_levels.len()
    }

    /// Samples per codec frame (1920: 80 ms at 24 kHz).
    pub fn frame_size(&self) -> usize {
        self.stft_hop_length * self.stft_frame_stack
    }

    /// Codec frames per second (12.5).
    pub fn frame_rate(&self) -> f32 {
        self.sampling_rate as f32 / self.frame_size() as f32
    }

    /// One-sided STFT bins (641).
    pub fn bins(&self) -> usize {
        self.stft_n_fft / 2 + 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn released_shape() {
        let c = LiquidCodecConfig::default();
        assert_eq!(c.codebook_size(), 2016);
        assert_eq!(c.frame_size(), 1920);
        assert_eq!(c.frame_rate(), 12.5);
        let parsed: LiquidCodecConfig =
            serde_json::from_str(r#"{"n_q": 12, "model_type": "liquid_codec"}"#).unwrap();
        assert_eq!(parsed, c);
    }
}
