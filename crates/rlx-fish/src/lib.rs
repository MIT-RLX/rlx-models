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

//! # rlx-fish
//!
//! **Fish-Speech** (Fish Audio) TTS / voice-cloning on RLX — a **dual-AR**
//! architecture: a "slow" Llama-style backbone emits one semantic step per audio
//! frame, and a small "fast" (depth) transformer autoregressively emits that
//! frame's `num_codebooks` acoustic codes; a **Firefly-GAN** codec decodes the
//! `[frames, num_codebooks]` code matrix to a waveform.
//!
//! Native Rust, composing rlx pieces:
//!
//! - **Backbone + fast transformer** → both are Llama-style (`rlx-llama32`).
//! - **Firefly codec** → a VQ + GAN vocoder (conv/GAN stack, cf. `rlx-neutts`
//!   BigVGAN / `rlx-dac`).
//!
//! Checkpoint-free, unit-tested core here: the config ([`FishConfig`],
//! [`FireflyConfig`]) and the dual-AR **codebook-matrix** packing
//! ([`codebook_matrix`] / [`flatten_codebook_matrix`] / [`validate_codes`]) — the
//! bridge between the fast transformer's flat token stream and the codec's
//! per-frame codebook rows. Wiring the two transformers + Firefly decode
//! end-to-end is the next step.

use anyhow::{Result, ensure};

/// Firefly-GAN codec parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct FireflyConfig {
    pub sample_rate: usize,
    pub hop_length: usize,
    pub num_codebooks: usize,
    pub codebook_size: usize,
    /// Codec latent / VQ dimension.
    pub codebook_dim: usize,
}

impl Default for FireflyConfig {
    fn default() -> Self {
        Self {
            sample_rate: 44_100,
            hop_length: 512,
            num_codebooks: 8,
            codebook_size: 1024,
            codebook_dim: 512,
        }
    }
}

/// Fish-Speech dual-AR model config. Dimensional fields carry typical values;
/// exact widths come from the checkpoint.
#[derive(Debug, Clone, PartialEq)]
pub struct FishConfig {
    pub vocab_size: usize,
    // Slow backbone (Llama-style).
    pub backbone_dim: usize,
    pub backbone_layers: usize,
    pub backbone_heads: usize,
    // Fast / depth transformer (per-frame codebook AR).
    pub fast_dim: usize,
    pub fast_layers: usize,
    pub fast_heads: usize,
    /// Firefly codec config; `num_codebooks` here must match the fast transformer.
    pub codec: FireflyConfig,
}

impl Default for FishConfig {
    fn default() -> Self {
        Self {
            vocab_size: 32_000,
            backbone_dim: 1024,
            backbone_layers: 24,
            backbone_heads: 16,
            fast_dim: 1024,
            fast_layers: 4,
            fast_heads: 16,
            codec: FireflyConfig::default(),
        }
    }
}

impl FishConfig {
    /// Number of acoustic codebooks the fast transformer emits per frame.
    pub fn num_codebooks(&self) -> usize {
        self.codec.num_codebooks
    }

    /// Acoustic frame rate (`sample_rate / hop_length`).
    pub fn frames_per_second(&self) -> f32 {
        self.codec.sample_rate as f32 / self.codec.hop_length as f32
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(self.num_codebooks() > 0, "num_codebooks must be > 0");
        ensure!(self.codec.codebook_size > 0, "codebook_size must be > 0");
        ensure!(self.codec.hop_length > 0, "hop_length must be > 0");
        ensure!(
            self.backbone_dim > 0 && self.fast_dim > 0,
            "invalid transformer dims"
        );
        Ok(())
    }
}

/// Reshape the fast transformer's flat, frame-major code stream into per-frame
/// codebook rows (`[frames][num_codebooks]`). The stream length must be a whole
/// number of frames.
pub fn codebook_matrix(flat: &[i32], num_codebooks: usize) -> Result<Vec<Vec<i32>>> {
    ensure!(num_codebooks > 0, "num_codebooks must be > 0");
    ensure!(
        flat.len().is_multiple_of(num_codebooks),
        "code stream length {} is not a multiple of num_codebooks {num_codebooks}",
        flat.len()
    );
    Ok(flat
        .chunks_exact(num_codebooks)
        .map(|row| row.to_vec())
        .collect())
}

/// Flatten per-frame codebook rows back into the frame-major stream.
pub fn flatten_codebook_matrix(frames: &[Vec<i32>]) -> Vec<i32> {
    frames.iter().flatten().copied().collect()
}

/// Check that every code lies in `[0, codebook_size)` and every row has the
/// expected width.
pub fn validate_codes(
    frames: &[Vec<i32>],
    num_codebooks: usize,
    codebook_size: usize,
) -> Result<()> {
    for (f, row) in frames.iter().enumerate() {
        ensure!(
            row.len() == num_codebooks,
            "frame {f} has {} codes, expected {num_codebooks}",
            row.len()
        );
        for (k, &code) in row.iter().enumerate() {
            ensure!(
                code >= 0 && (code as usize) < codebook_size,
                "frame {f} codebook {k} code {code} out of range [0, {codebook_size})"
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_defaults_and_validate() {
        let c = FishConfig::default();
        assert_eq!(c.num_codebooks(), 8);
        assert_eq!(c.codec.sample_rate, 44_100);
        assert_eq!(c.codec.codebook_size, 1024);
        c.validate().unwrap();
    }

    #[test]
    fn frame_rate() {
        let c = FishConfig::default();
        assert!((c.frames_per_second() - 44_100.0 / 512.0).abs() < 1e-3);
    }

    #[test]
    fn codebook_matrix_roundtrips() {
        let flat = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
        let frames = codebook_matrix(&flat, 4).unwrap();
        assert_eq!(
            frames,
            vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7], vec![8, 9, 10, 11]]
        );
        assert_eq!(flatten_codebook_matrix(&frames), flat);
    }

    #[test]
    fn codebook_matrix_rejects_ragged_stream() {
        let flat = vec![0, 1, 2, 3, 4]; // 5 not divisible by 4
        assert!(codebook_matrix(&flat, 4).is_err());
    }

    #[test]
    fn validate_codes_checks_width_and_range() {
        let ok = vec![vec![0, 1, 2], vec![3, 1023, 0]];
        validate_codes(&ok, 3, 1024).unwrap();
        // out of range
        let bad = vec![vec![0, 1, 2000]];
        assert!(validate_codes(&bad, 3, 1024).is_err());
        // wrong width
        let ragged = vec![vec![0, 1]];
        assert!(validate_codes(&ragged, 3, 1024).is_err());
    }
}
