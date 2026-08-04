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

//! # rlx-citrinet
//!
//! **Citrinet** ASR (NVIDIA NeMo) on RLX — a 1D **time-channel separable
//! convolution** network (QuartzNet-style) with **squeeze-and-excitation** blocks
//! over mel features, trained with **CTC**. Decoding is CTC greedy: argmax per
//! frame, collapse consecutive repeats, drop blanks.
//!
//! Native Rust. The conv stack reuses `rlx-conformer-ctc` conv modules and the
//! `.nemo` loader reuses `rlx-nemo`; this crate contributes the config plus the
//! CTC decode ([`ctc_greedy_decode`] / [`ctc_greedy_decode_logits`]), which any CTC
//! model (Conformer-CTC, Wav2Vec2-CTC, …) can reuse.

use anyhow::{Result, ensure};

/// Citrinet config. The canonical models are `Citrinet-512` / `Citrinet-1024`.
#[derive(Debug, Clone, PartialEq)]
pub struct CitrinetConfig {
    pub sample_rate: usize,
    /// Mel feature dimension.
    pub feature_dim: usize,
    /// Base channel width (512 / 1024).
    pub channels: usize,
    /// Number of TCS-conv mega-blocks.
    pub num_blocks: usize,
    /// Squeeze-and-excitation channel reduction ratio.
    pub se_reduction: usize,
    /// Total time subsampling factor.
    pub subsampling_factor: usize,
    /// Acoustic vocabulary size (excluding the blank).
    pub vocab_size: usize,
    /// CTC blank id (NeMo puts it last, at index `vocab_size`).
    pub blank_id: i32,
}

impl Default for CitrinetConfig {
    fn default() -> Self {
        let vocab_size = 1024;
        Self {
            sample_rate: 16_000,
            feature_dim: 80,
            channels: 512,
            num_blocks: 21,
            se_reduction: 8,
            subsampling_factor: 8,
            vocab_size,
            blank_id: vocab_size as i32,
        }
    }
}

impl CitrinetConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(self.channels > 0, "channels must be > 0");
        ensure!(self.num_blocks > 0, "num_blocks must be > 0");
        ensure!(self.vocab_size > 0, "vocab_size must be > 0");
        Ok(())
    }

    /// Total number of classifier outputs (`vocab + 1` for the blank).
    pub fn num_classes(&self) -> usize {
        self.vocab_size + 1
    }
}

/// CTC greedy decode over per-frame argmax token ids: collapse consecutive
/// duplicates, then drop the blank. Returns the emitted token sequence.
pub fn ctc_greedy_decode(frame_ids: &[i32], blank_id: i32) -> Vec<i32> {
    let mut out = Vec::new();
    let mut prev: Option<i32> = None;
    for &id in frame_ids {
        if prev != Some(id) {
            if id != blank_id {
                out.push(id);
            }
            prev = Some(id);
        }
    }
    out
}

/// CTC greedy decode straight from per-frame logits (`[frames][num_classes]`):
/// argmax each frame, then [`ctc_greedy_decode`].
pub fn ctc_greedy_decode_logits(logits: &[Vec<f32>], blank_id: i32) -> Vec<i32> {
    let ids: Vec<i32> = logits
        .iter()
        .map(|frame| {
            let mut best = 0usize;
            let mut bv = f32::NEG_INFINITY;
            for (i, &v) in frame.iter().enumerate() {
                if v > bv {
                    bv = v;
                    best = i;
                }
            }
            best as i32
        })
        .collect();
    ctc_greedy_decode(&ids, blank_id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_and_classes() {
        let c = CitrinetConfig::default();
        assert_eq!(c.num_classes(), 1025);
        assert_eq!(c.blank_id, 1024);
        c.validate().unwrap();
    }

    #[test]
    fn ctc_collapses_repeats_and_drops_blank() {
        let blank = 0;
        // [a a _ a] → collapse → [a _ a] → drop blank → [a a]  (blank separates)
        assert_eq!(ctc_greedy_decode(&[1, 1, 0, 1], blank), vec![1, 1]);
        // [_ a a a _ b] → [a b]
        assert_eq!(ctc_greedy_decode(&[0, 1, 1, 1, 0, 2], blank), vec![1, 2]);
        // all blank → empty
        assert_eq!(ctc_greedy_decode(&[0, 0, 0], blank), Vec::<i32>::new());
        // no repeats, no blanks → identity
        assert_eq!(ctc_greedy_decode(&[3, 1, 2], blank), vec![3, 1, 2]);
    }

    #[test]
    fn ctc_from_logits_argmaxes_then_decodes() {
        // frames: token1, token1, blank(0), token2
        let logits = vec![
            vec![-1.0, 5.0, 0.0], // argmax 1
            vec![-1.0, 5.0, 0.0], // argmax 1
            vec![5.0, -1.0, 0.0], // argmax 0 (blank)
            vec![-1.0, 0.0, 5.0], // argmax 2
        ];
        assert_eq!(ctc_greedy_decode_logits(&logits, 0), vec![1, 2]);
    }
}
