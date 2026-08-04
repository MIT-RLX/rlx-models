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

//! RVQ **delay pattern** — the MusicGen / Parler / Higgs interleaving that lets an
//! autoregressive LM predict all `K` residual codebooks in parallel while
//! respecting their coarse-to-fine dependency: codebook `k` is shifted by `k`
//! steps, so at any output position the model has already committed to the coarser
//! codebooks of the current frame.
//!
//! Codes are `[num_codebooks][frames]`. The delayed form is
//! `[num_codebooks][frames + num_codebooks - 1]`, padded with a sentinel where a
//! codebook has no valid value yet.

use anyhow::{Result, ensure};

/// Apply the delay pattern to `codes` (`[K][T]`), padding empty slots with `pad`.
/// Returns `[K][T + K - 1]`.
pub fn build_delay_pattern(codes: &[Vec<i32>], pad: i32) -> Result<Vec<Vec<i32>>> {
    let k = codes.len();
    ensure!(k > 0, "delay pattern needs ≥1 codebook");
    let t = codes[0].len();
    ensure!(
        codes.iter().all(|row| row.len() == t),
        "all codebooks must have the same frame count"
    );
    let l = t + k - 1;
    let mut out = vec![vec![pad; l]; k];
    for (cb, row) in out.iter_mut().enumerate() {
        for (pos, slot) in row.iter_mut().enumerate() {
            // Codebook `cb` holds valid data at positions [cb, cb + t).
            if pos >= cb && pos < cb + t {
                *slot = codes[cb][pos - cb];
            }
        }
    }
    Ok(out)
}

/// Invert [`build_delay_pattern`]: recover `[K][T]` from a delayed `[K][L]`
/// (`T = L - K + 1`).
pub fn revert_delay_pattern(delayed: &[Vec<i32>]) -> Result<Vec<Vec<i32>>> {
    let k = delayed.len();
    ensure!(k > 0, "delay pattern needs ≥1 codebook");
    let l = delayed[0].len();
    ensure!(
        delayed.iter().all(|row| row.len() == l),
        "all codebooks must have the same delayed length"
    );
    ensure!(l >= k, "delayed length {l} shorter than codebook count {k}");
    let t = l - k + 1;
    let mut out = vec![vec![0i32; t]; k];
    for (cb, row) in out.iter_mut().enumerate() {
        for (j, slot) in row.iter_mut().enumerate() {
            *slot = delayed[cb][j + cb];
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_matches_hand_computed() {
        let codes = vec![vec![1, 2], vec![3, 4], vec![5, 6]];
        let d = build_delay_pattern(&codes, 0).unwrap();
        assert_eq!(d[0], vec![1, 2, 0, 0]);
        assert_eq!(d[1], vec![0, 3, 4, 0]);
        assert_eq!(d[2], vec![0, 0, 5, 6]);
    }

    #[test]
    fn roundtrips() {
        let codes = vec![
            vec![10, 11, 12, 13],
            vec![20, 21, 22, 23],
            vec![30, 31, 32, 33],
            vec![40, 41, 42, 43],
        ];
        let d = build_delay_pattern(&codes, -1).unwrap();
        assert_eq!(d[0].len(), 4 + 4 - 1);
        let back = revert_delay_pattern(&d).unwrap();
        assert_eq!(back, codes);
    }

    #[test]
    fn single_codebook_is_identity() {
        let codes = vec![vec![7, 8, 9]];
        let d = build_delay_pattern(&codes, 0).unwrap();
        assert_eq!(d, codes);
        assert_eq!(revert_delay_pattern(&d).unwrap(), codes);
    }

    #[test]
    fn rejects_ragged_input() {
        let codes = vec![vec![1, 2], vec![3]];
        assert!(build_delay_pattern(&codes, 0).is_err());
    }
}
