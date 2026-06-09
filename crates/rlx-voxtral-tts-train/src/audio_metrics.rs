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

//! Lightweight audio similarity metrics for rig tests.

use crate::asr_loss::whisper_mel_mse;

/// Map mel-band MSE to a 0..1 similarity score (1 = identical bands).
pub fn mel_similarity(a: &[f32], b: &[f32]) -> f32 {
    let mse = whisper_mel_mse(a, b);
    1.0 / (1.0 + mse)
}

/// Cosine similarity between flat voice-embedding vectors.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let mut dot = 0f32;
    let mut na = 0f32;
    let mut nb = 0f32;
    for i in 0..n {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom <= 1e-12 {
        0.0
    } else {
        (dot / denom).clamp(-1.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_pcm_has_high_mel_similarity() {
        let pcm = vec![0.1f32; 2400];
        assert!(mel_similarity(&pcm, &pcm) > 0.99);
    }

    #[test]
    fn cosine_one_for_same_vector() {
        let v = vec![1.0, 2.0, 3.0];
        assert!((cosine_similarity(&v, &v) - 1.0).abs() < 1e-5);
    }
}
