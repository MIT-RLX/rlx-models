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

//! **MarbleNet** voice activity detection (NVIDIA NeMo).
//!
//! MarbleNet is a compact 1D **time-channel separable (TCS) convolution** network
//! (a QuartzNet-style residual conv stack) over MFCC features that emits per-frame
//! speech / non-speech logits. Those per-frame speech probabilities feed the shared
//! [`speech_segments_from_probs`](crate::segments::speech_segments_from_probs) to
//! produce speech segments — the same segmentation path as the other rlx-vad
//! backends.
//!
//! This module provides the checkpoint-free config; the TCS-conv graph + NeMo
//! weight loading (reusing `rlx-ir`/`rlx-flow`, which already back the crate's other
//! backends on every RLX device) is the next step.

/// MarbleNet architecture config. The canonical released model is `MarbleNet-3x2x64`.
#[derive(Debug, Clone, PartialEq)]
pub struct MarbleNetConfig {
    pub sample_rate: usize,
    /// MFCC feature dimension (input channels).
    pub feature_dim: usize,
    /// Feature window length, milliseconds.
    pub window_size_ms: usize,
    /// Feature hop, milliseconds (the per-frame rate the model scores at).
    pub window_stride_ms: usize,
    /// Number of residual TCS-conv blocks (the `R` in `RxSxC`).
    pub num_blocks: usize,
    /// Sub-blocks per block (the `S`).
    pub sub_blocks: usize,
    /// Base channel width (the `C`).
    pub channels: usize,
    /// Output classes (speech vs non-speech = 2).
    pub num_classes: usize,
}

impl Default for MarbleNetConfig {
    /// MarbleNet-3x2x64 @ 16 kHz (25 ms / 10 ms features).
    fn default() -> Self {
        Self {
            sample_rate: 16_000,
            feature_dim: 64,
            window_size_ms: 25,
            window_stride_ms: 10,
            num_blocks: 3,
            sub_blocks: 2,
            channels: 64,
            num_classes: 2,
        }
    }
}

impl MarbleNetConfig {
    /// Samples between consecutive feature frames (`sample_rate · stride_ms / 1000`).
    pub fn frame_hop_samples(&self) -> usize {
        self.sample_rate * self.window_stride_ms / 1000
    }

    /// Frames per second the model scores at.
    pub fn frames_per_second(&self) -> f32 {
        1000.0 / self.window_stride_ms as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::segments::{SegmentParams, speech_segments_from_probs};

    #[test]
    fn config_defaults_3x2x64() {
        let c = MarbleNetConfig::default();
        assert_eq!(c.num_blocks, 3);
        assert_eq!(c.sub_blocks, 2);
        assert_eq!(c.channels, 64);
        assert_eq!(c.num_classes, 2);
        // 10 ms hop @ 16 kHz = 160 samples = 100 fps.
        assert_eq!(c.frame_hop_samples(), 160);
        assert!((c.frames_per_second() - 100.0).abs() < 1e-3);
    }

    #[test]
    fn probs_segment_via_shared_path() {
        // A run of speech frames surrounded by silence → one segment.
        let cfg = MarbleNetConfig::default();
        let hop = cfg.frame_hop_samples();
        // 50 silence, 60 speech, 50 silence frames.
        let mut probs = vec![0.0f32; 50];
        probs.extend(std::iter::repeat_n(0.95, 60));
        probs.extend(std::iter::repeat_n(0.0, 50));
        let n_samples = probs.len() * hop;
        let params = SegmentParams::marblenet();
        let segs = speech_segments_from_probs(n_samples, hop, &params, &probs);
        assert_eq!(segs.len(), 1, "expected exactly one speech segment");
        // The segment sits within the middle speech run.
        assert!(segs[0].start < segs[0].end);
    }
}
