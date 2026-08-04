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

//! # rlx-demucs
//!
//! **Hybrid-Transformer Demucs** (`htdemucs`) music source separation on RLX. A
//! dual-branch model: a **time branch** (1D conv U-Net on the waveform) and a
//! **spectral branch** (2D conv U-Net on the STFT), joined by a **cross-domain
//! Transformer** at the bottleneck; it outputs 4 stems (drums / bass / other /
//! vocals). Long audio is processed in **overlapping segments** with a triangular
//! overlap-add.
//!
//! Native Rust. STFT/ISTFT reuse [`rlx-fft`](https://docs.rs/rlx-fft); the conv
//! U-Nets + transformer reuse `rlx-flow`. This crate contributes the checkpoint-
//! free, unit-tested inference glue: the config, the U-Net channel progression
//! ([`encoder_channels`]), and the overlap-add segmentation
//! ([`DemucsConfig::segment_starts`] / [`transition_weight`]).

use anyhow::{Result, ensure};

/// HTDemucs config.
#[derive(Debug, Clone, PartialEq)]
pub struct DemucsConfig {
    pub sample_rate: usize,
    pub audio_channels: usize,
    /// Output stems (drums, bass, other, vocals = 4).
    pub num_sources: usize,
    // Spectral branch STFT.
    pub n_fft: usize,
    pub hop_length: usize,
    // Conv U-Net.
    /// Initial conv channels (doubles each encoder layer by `growth`).
    pub base_channels: usize,
    pub growth: usize,
    pub depth: usize,
    // Cross-domain transformer.
    pub transformer_layers: usize,
    pub transformer_heads: usize,
    // Inference segmentation.
    /// Segment length in samples.
    pub segment_length: usize,
    /// Overlap fraction between consecutive segments (0..1).
    pub overlap: f32,
}

impl Default for DemucsConfig {
    fn default() -> Self {
        Self {
            sample_rate: 44_100,
            audio_channels: 2,
            num_sources: 4,
            n_fft: 4096,
            hop_length: 1024,
            base_channels: 48,
            growth: 2,
            depth: 4,
            transformer_layers: 5,
            transformer_heads: 8,
            segment_length: 44_100 * 10, // ~10 s
            overlap: 0.25,
        }
    }
}

impl DemucsConfig {
    pub fn num_freqs(&self) -> usize {
        self.n_fft / 2 + 1
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(self.num_sources > 0, "num_sources must be > 0");
        ensure!(self.depth > 0, "depth must be > 0");
        ensure!(self.growth >= 1, "growth must be ≥ 1");
        ensure!(self.segment_length > 0, "segment_length must be > 0");
        ensure!(
            (0.0..1.0).contains(&self.overlap),
            "overlap must be in [0, 1)"
        );
        Ok(())
    }

    /// Segment start offsets (in samples) for overlap-add inference over
    /// `total_len` samples. The final segment may run past the end (the model pads).
    pub fn segment_starts(&self, total_len: usize) -> Vec<usize> {
        if total_len == 0 {
            return Vec::new();
        }
        let stride = ((self.segment_length as f32) * (1.0 - self.overlap)).round() as usize;
        let stride = stride.max(1);
        let mut starts = Vec::new();
        let mut s = 0;
        while s < total_len {
            starts.push(s);
            s += stride;
        }
        starts
    }
}

/// Encoder channel counts per layer: `base · growth^i` for `i in 0..depth`.
pub fn encoder_channels(base: usize, growth: usize, depth: usize) -> Vec<usize> {
    (0..depth).map(|i| base * growth.pow(i as u32)).collect()
}

/// A triangular overlap-add weight of length `len` (ramp up to the centre, back
/// down), used to cross-fade overlapping segments. Peak in the middle, ≥ 1 at the
/// edges, symmetric.
pub fn transition_weight(len: usize) -> Vec<f32> {
    (0..len).map(|i| (i + 1).min(len - i) as f32).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_and_validate() {
        let c = DemucsConfig::default();
        assert_eq!(c.num_sources, 4);
        assert_eq!(c.num_freqs(), 4096 / 2 + 1);
        c.validate().unwrap();
    }

    #[test]
    fn channels_double_per_layer() {
        assert_eq!(encoder_channels(48, 2, 4), vec![48, 96, 192, 384]);
        assert_eq!(encoder_channels(64, 2, 1), vec![64]);
    }

    #[test]
    fn segment_starts_cover_audio() {
        let c = DemucsConfig {
            segment_length: 400,
            overlap: 0.25,
            ..Default::default()
        };
        // stride = 400 * 0.75 = 300
        assert_eq!(c.segment_starts(1000), vec![0, 300, 600, 900]);
        // last segment (900..1300) runs past 1000 → the model pads.
        assert_eq!(c.segment_starts(0), Vec::<usize>::new());
        // short audio → a single segment
        assert_eq!(c.segment_starts(200), vec![0]);
    }

    #[test]
    fn transition_weight_is_symmetric_triangle() {
        assert_eq!(transition_weight(5), vec![1.0, 2.0, 3.0, 2.0, 1.0]);
        let w = transition_weight(6);
        assert_eq!(w, vec![1.0, 2.0, 3.0, 3.0, 2.0, 1.0]);
        // symmetric
        let n = w.len();
        for i in 0..n {
            assert_eq!(w[i], w[n - 1 - i]);
        }
    }
}
