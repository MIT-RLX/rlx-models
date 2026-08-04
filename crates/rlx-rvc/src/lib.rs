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

//! # rlx-rvc
//!
//! **RVC** (Retrieval-based Voice Conversion) on RLX. RVC extracts self-supervised
//! **content features** (HuBERT / ContentVec) from the source, **retrieves** the
//! nearest target-speaker features from a trained index and blends them in
//! (the "retrieval" that fixes timbre), conditions on **F0** (pitch, optionally
//! transposed), and synthesizes with an **NSF-HiFiGAN** generator.
//!
//! Native Rust, composing rlx pieces:
//!
//! - **Content encoder** → HuBERT / Wav2Vec2-BERT (`rlx-neutts` / `rlx-wav2vec2-bert`).
//! - **Generator** → NSF-HiFiGAN (conv/GAN vocoder, cf. `rlx-nanocodec` / `rlx-neutts`).
//!
//! The checkpoint-free, unit-tested core here is the two RVC-defining operations:
//! [`retrieval_blend`] (k-NN feature-index blend) and [`transpose_f0`] (pitch
//! shift). The HuBERT + generator graphs are the next step.

use anyhow::{Result, ensure};

/// RVC config.
#[derive(Debug, Clone, PartialEq)]
pub struct RvcConfig {
    /// Content-encoder input sample rate (HuBERT = 16 kHz).
    pub content_sample_rate: usize,
    /// Output/generator sample rate (RVC v2 commonly 40 kHz).
    pub output_sample_rate: usize,
    /// Content-feature width (ContentVec/HuBERT: 768 for v2, 256 for v1).
    pub content_dim: usize,
    /// Default retrieval mix (0 = ignore index, 1 = fully retrieved).
    pub index_rate: f32,
    /// Default pitch transpose in semitones.
    pub transpose_semitones: f32,
}

impl Default for RvcConfig {
    fn default() -> Self {
        Self {
            content_sample_rate: 16_000,
            output_sample_rate: 40_000,
            content_dim: 768,
            index_rate: 0.75,
            transpose_semitones: 0.0,
        }
    }
}

impl RvcConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(self.content_dim > 0, "content_dim must be > 0");
        ensure!(
            (0.0..=1.0).contains(&self.index_rate),
            "index_rate must be in [0, 1]"
        );
        Ok(())
    }
}

/// Blend a query content feature with its `k` nearest target-speaker features
/// (from the RVC index) by `index_rate`. Neighbour contributions are weighted by
/// inverse squared distance (closer = more weight):
/// `out = (1 − rate)·query + rate·Σ wᵢ·featureᵢ`. Returns the query unchanged when
/// `rate ≤ 0` or there are no neighbours.
pub fn retrieval_blend(
    query: &[f32],
    neighbor_features: &[Vec<f32>],
    distances: &[f32],
    index_rate: f32,
) -> Result<Vec<f32>> {
    ensure!(
        neighbor_features.len() == distances.len(),
        "neighbour/distance count mismatch"
    );
    let rate = index_rate.clamp(0.0, 1.0);
    if neighbor_features.is_empty() || rate == 0.0 {
        return Ok(query.to_vec());
    }
    let dim = query.len();
    ensure!(
        neighbor_features.iter().all(|f| f.len() == dim),
        "neighbour features must match query dimension {dim}"
    );

    // Inverse-square-distance weights, normalised.
    let mut weights: Vec<f32> = distances.iter().map(|&d| 1.0 / (d * d + 1e-8)).collect();
    let sum: f32 = weights.iter().sum();
    for w in &mut weights {
        *w /= sum;
    }

    let mut knn = vec![0.0f32; dim];
    for (feat, &w) in neighbor_features.iter().zip(&weights) {
        for (k, &v) in knn.iter_mut().zip(feat) {
            *k += w * v;
        }
    }
    Ok((0..dim)
        .map(|j| (1.0 - rate) * query[j] + rate * knn[j])
        .collect())
}

/// Transpose an F0 (pitch, Hz) contour by `semitones`: `f0 · 2^(semitones/12)`.
/// Unvoiced frames (`f0 == 0`) are left at zero.
pub fn transpose_f0(f0: &[f32], semitones: f32) -> Vec<f32> {
    let mult = 2f32.powf(semitones / 12.0);
    f0.iter()
        .map(|&x| if x > 0.0 { x * mult } else { 0.0 })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_validate() {
        RvcConfig::default().validate().unwrap();
        let bad = RvcConfig {
            index_rate: 1.5,
            ..Default::default()
        };
        assert!(bad.validate().is_err());
    }

    #[test]
    fn retrieval_rate_zero_returns_query() {
        let q = vec![1.0, 2.0];
        let out = retrieval_blend(&q, &[vec![9.0, 9.0]], &[1.0], 0.0).unwrap();
        assert_eq!(out, q);
        // no neighbours → query
        assert_eq!(retrieval_blend(&q, &[], &[], 0.8).unwrap(), q);
    }

    #[test]
    fn retrieval_full_rate_single_neighbor_is_neighbor() {
        let q = vec![0.0, 0.0];
        let out = retrieval_blend(&q, &[vec![10.0, 20.0]], &[1.0], 1.0).unwrap();
        assert_eq!(out, vec![10.0, 20.0]);
        // half rate → midpoint
        let mid = retrieval_blend(&q, &[vec![10.0, 20.0]], &[1.0], 0.5).unwrap();
        assert_eq!(mid, vec![5.0, 10.0]);
    }

    #[test]
    fn retrieval_weights_favor_closer_neighbor() {
        // two neighbours, one much closer → blend leans toward it.
        let q = vec![0.0];
        let out = retrieval_blend(
            &q,
            &[vec![10.0], vec![0.0]],
            &[0.1, 10.0], // first is far closer
            1.0,
        )
        .unwrap();
        assert!(out[0] > 9.0, "expected near 10, got {}", out[0]);
    }

    #[test]
    fn f0_transpose_shifts_by_octave() {
        assert_eq!(
            transpose_f0(&[100.0, 200.0, 0.0], 12.0),
            vec![200.0, 400.0, 0.0]
        );
        assert_eq!(transpose_f0(&[440.0], 0.0), vec![440.0]);
        assert_eq!(transpose_f0(&[400.0], -12.0), vec![200.0]);
    }
}
