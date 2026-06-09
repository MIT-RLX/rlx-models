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

//! Reusable batch-prep utilities (plan #83).
//!
//! Borrowed from MAX's `pipelines/dataprocessing/` module:
//! `causal_attention_mask.py`, `collate_batch.py`. Pulling these into
//! a dedicated module keeps each model file (bert.rs, nomic.rs, etc.)
//! focused on the graph builder and stops the same padding logic from
//! being reimplemented per architecture.

/// Build a causal (lower-triangular) attention mask of size
/// `[seq, seq]`. `out[qi, ki] = 1.0` iff `ki <= qi`. Values match
/// the existing 0/1 mask convention used in burnembed (see
/// `RuntimeConfig::mask_binary_threshold`).
///
/// Allocates and returns a fresh `Vec<f32>` of length `seq * seq`.
/// Callers usually want to broadcast this across the batch
/// dimension; do that at the call site so a single causal mask
/// can be shared across batches.
pub fn causal_mask(seq: usize) -> Vec<f32> {
    let mut m = vec![0.0f32; seq * seq];
    for qi in 0..seq {
        for ki in 0..=qi {
            m[qi * seq + ki] = 1.0;
        }
    }
    m
}

/// Build a sliding-window mask of size `[seq, seq]`:
/// `out[qi, ki] = 1.0` iff `qi - window <= ki <= qi`. Used by
/// long-context models (Mistral, Gemma, Phi) that don't attend
/// past a fixed lookback.
pub fn sliding_window_mask(seq: usize, window: usize) -> Vec<f32> {
    let mut m = vec![0.0f32; seq * seq];
    for qi in 0..seq {
        let lo = qi.saturating_sub(window);
        for ki in lo..=qi {
            m[qi * seq + ki] = 1.0;
        }
    }
    m
}

/// Build a padding mask from per-row sequence lengths: rows with
/// `lengths[i] == k` get `1.0` in the first k columns and `0.0`
/// after. Output is `[batch, max_seq]` flattened row-major. The
/// existing burnembed BERT path uses this shape for its mask.
pub fn padding_mask(lengths: &[usize], max_seq: usize) -> Vec<f32> {
    let mut m = vec![0.0f32; lengths.len() * max_seq];
    for (i, &len) in lengths.iter().enumerate() {
        let n = len.min(max_seq);
        for j in 0..n {
            m[i * max_seq + j] = 1.0;
        }
    }
    m
}

/// Collate a batch of variable-length f32 sequences into a single
/// `[batch, max_seq]` tensor padded with `pad_value`. Returns the
/// flat tensor + the per-row lengths (so callers can pair with
/// [`padding_mask`]). Order of rows matches input order.
pub fn collate_padded_f32(rows: &[Vec<f32>], pad_value: f32) -> (Vec<f32>, Vec<usize>) {
    let lengths: Vec<usize> = rows.iter().map(|r| r.len()).collect();
    let max_seq = lengths.iter().copied().max().unwrap_or(0);
    let mut out = vec![pad_value; rows.len() * max_seq];
    for (i, r) in rows.iter().enumerate() {
        out[i * max_seq..i * max_seq + r.len()].copy_from_slice(r);
    }
    (out, lengths)
}

/// `i64` variant of [`collate_padded_f32`] — for token IDs.
pub fn collate_padded_i64(rows: &[Vec<i64>], pad_value: i64) -> (Vec<i64>, Vec<usize>) {
    let lengths: Vec<usize> = rows.iter().map(|r| r.len()).collect();
    let max_seq = lengths.iter().copied().max().unwrap_or(0);
    let mut out = vec![pad_value; rows.len() * max_seq];
    for (i, r) in rows.iter().enumerate() {
        out[i * max_seq..i * max_seq + r.len()].copy_from_slice(r);
    }
    (out, lengths)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn causal_is_lower_triangular() {
        let m = causal_mask(4);
        assert_eq!(
            m,
            vec![
                1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0,
            ]
        );
    }

    #[test]
    fn sliding_window_band() {
        let m = sliding_window_mask(5, 1);
        assert_eq!(
            m,
            vec![
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0,
            ]
        );
    }

    #[test]
    fn padding_zeros_after_length() {
        let m = padding_mask(&[3, 1, 2], 4);
        assert_eq!(
            m,
            vec![1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0,]
        );
    }

    #[test]
    fn collate_pads_to_longest() {
        let rows = vec![vec![1, 2, 3], vec![4], vec![5, 6]];
        let (flat, lens) = collate_padded_i64(&rows, 0);
        assert_eq!(lens, vec![3, 1, 2]);
        assert_eq!(flat, vec![1, 2, 3, 4, 0, 0, 5, 6, 0]);
    }
}
