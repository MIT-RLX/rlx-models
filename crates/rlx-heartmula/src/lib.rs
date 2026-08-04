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

//! # rlx-heartmula
//!
//! **HeartMula** music generation on RLX — a MusicGen-style codec-token language
//! model: a transformer autoregressively emits the RVQ codebooks of a neural music
//! codec (**HeartCodec**), interleaved with the **delay pattern**, then the codec
//! decodes them to a waveform.
//!
//! Native Rust, composing rlx pieces:
//!
//! - **LM backbone** → Llama-style (`rlx-llama32`).
//! - **RVQ delay pattern** → [`rlx_audio_blocks::codec`].
//! - **HeartCodec decode** → RVQ + GAN codec (`rlx-dac` / `rlx-encodec` patterns).
//!
//! Checkpoint-free, unit-tested core: the config, duration→token control, and the
//! codebook delay interleave. (HeartMula's exact dims come from its checkpoint; the
//! values here are the MusicGen-family conventions.)

use anyhow::{Result, ensure};
use rlx_audio_blocks::codec::{build_delay_pattern, revert_delay_pattern};

/// HeartMula / HeartCodec config.
#[derive(Debug, Clone, PartialEq)]
pub struct HeartMulaConfig {
    pub sample_rate: usize,
    // LM backbone.
    pub backbone_hidden: usize,
    pub backbone_layers: usize,
    pub backbone_heads: usize,
    /// Text/condition prompt embedding width.
    pub cond_dim: usize,
    // HeartCodec RVQ.
    pub num_codebooks: usize,
    pub codebook_size: usize,
    /// Codec token frame rate (tokens/sec).
    pub frame_rate: f32,
    /// Delay-pattern padding sentinel.
    pub pad_id: i32,
    /// Maximum generatable duration (seconds).
    pub max_seconds: f32,
}

impl Default for HeartMulaConfig {
    fn default() -> Self {
        Self {
            sample_rate: 44_100,
            backbone_hidden: 1536,
            backbone_layers: 24,
            backbone_heads: 16,
            cond_dim: 768,
            num_codebooks: 4,
            codebook_size: 2048,
            frame_rate: 50.0,
            pad_id: -1,
            max_seconds: 300.0,
        }
    }
}

impl HeartMulaConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(self.num_codebooks > 0, "num_codebooks must be > 0");
        ensure!(self.codebook_size > 0, "codebook_size must be > 0");
        ensure!(self.frame_rate > 0.0, "frame_rate must be > 0");
        Ok(())
    }

    /// Number of codec frames for a target clip of `seconds` (clamped to
    /// `max_seconds`).
    pub fn frames_for_duration(&self, seconds: f32) -> usize {
        let s = seconds.clamp(0.0, self.max_seconds);
        (s * self.frame_rate).round() as usize
    }

    /// Interleave RVQ `codes` (`[num_codebooks][frames]`) into the delay pattern the
    /// AR head predicts.
    pub fn delay_encode(&self, codes: &[Vec<i32>]) -> Result<Vec<Vec<i32>>> {
        build_delay_pattern(codes, self.pad_id)
    }

    /// Recover `[num_codebooks][frames]` codes from the generated delay pattern.
    pub fn delay_decode(&self, delayed: &[Vec<i32>]) -> Result<Vec<Vec<i32>>> {
        revert_delay_pattern(delayed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_and_validate() {
        let c = HeartMulaConfig::default();
        assert_eq!(c.num_codebooks, 4);
        assert_eq!(c.codebook_size, 2048);
        c.validate().unwrap();
    }

    #[test]
    fn duration_to_frames_clamps() {
        let c = HeartMulaConfig::default(); // 50 fps, max 300 s
        assert_eq!(c.frames_for_duration(2.0), 100);
        assert_eq!(c.frames_for_duration(1000.0), (300.0 * 50.0) as usize);
        assert_eq!(c.frames_for_duration(0.0), 0);
    }

    #[test]
    fn delay_roundtrip() {
        let c = HeartMulaConfig::default();
        let codes = vec![
            vec![1, 2, 3],
            vec![4, 5, 6],
            vec![7, 8, 9],
            vec![10, 11, 12],
        ];
        let delayed = c.delay_encode(&codes).unwrap();
        assert_eq!(delayed[0].len(), 3 + 4 - 1);
        assert_eq!(c.delay_decode(&delayed).unwrap(), codes);
    }
}
