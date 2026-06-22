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

//! Dynamic time warping for cross-attention word alignment.

use crate::timestamp_parse::TIMESTAMP_QUANTUM_SEC;

/// Slice flat encoder hidden `[enc_seq, d_model]` by absolute time (20 ms frames).
pub fn slice_encoder_hidden(
    enc: &[f32],
    enc_seq: usize,
    d_model: usize,
    start_sec: f32,
    end_sec: f32,
) -> (Vec<f32>, usize) {
    if enc_seq == 0 || enc.is_empty() {
        return (Vec::new(), 0);
    }
    let mut f0 = (start_sec / TIMESTAMP_QUANTUM_SEC).floor() as isize;
    let mut f1 = (end_sec / TIMESTAMP_QUANTUM_SEC).ceil() as isize;
    if f1 <= f0 {
        f0 = 0;
        f1 = enc_seq as isize;
    }
    let f0 = f0.clamp(0, enc_seq as isize) as usize;
    let f1 = f1.clamp(f0 as isize + 1, enc_seq as isize) as usize;
    let out_seq = f1 - f0;
    let mut out = vec![0f32; out_seq * d_model];
    for f in f0..f1 {
        let dst = (f - f0) * d_model;
        let src = f * d_model;
        out[dst..dst + d_model].copy_from_slice(&enc[src..src + d_model]);
    }
    (out, out_seq)
}

/// Median filter along the last axis (1D).
pub fn median_filter_1d(x: &[f32], width: usize) -> Vec<f32> {
    if width <= 1 || x.is_empty() {
        return x.to_vec();
    }
    let pad = width / 2;
    let mut out = vec![0f32; x.len()];
    for i in 0..x.len() {
        let lo = i.saturating_sub(pad);
        let hi = (i + pad + 1).min(x.len());
        let mut window: Vec<f32> = x[lo..hi].to_vec();
        window.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        out[i] = window[window.len() / 2];
    }
    out
}

/// DTW on cost matrix `x` with shape `(n_tokens, n_frames)`.
/// Returns `(text_indices, time_indices)` path arrays.
pub fn dtw(x: &[f32], n_tokens: usize, n_frames: usize) -> (Vec<usize>, Vec<usize>) {
    if n_tokens == 0 || n_frames == 0 {
        return (Vec::new(), Vec::new());
    }
    let mut cost = vec![f32::INFINITY; (n_tokens + 1) * (n_frames + 1)];
    let mut trace = vec![-1i8; (n_tokens + 1) * (n_frames + 1)];
    cost[0] = 0.0;

    let idx = |i: usize, j: usize| i * (n_frames + 1) + j;

    for j in 1..=n_frames {
        for i in 1..=n_tokens {
            let c0 = cost[idx(i - 1, j - 1)];
            let c1 = cost[idx(i - 1, j)];
            let c2 = cost[idx(i, j - 1)];
            let (c, t) = if c0 <= c1 && c0 <= c2 {
                (c0, 0i8)
            } else if c1 <= c0 && c1 <= c2 {
                (c1, 1i8)
            } else {
                (c2, 2i8)
            };
            cost[idx(i, j)] = x[(i - 1) * n_frames + (j - 1)] + c;
            trace[idx(i, j)] = t;
        }
    }

    let mut i = n_tokens;
    let mut j = n_frames;
    let mut text_indices = Vec::new();
    let mut time_indices = Vec::new();
    while i > 0 || j > 0 {
        text_indices.push(i.saturating_sub(1));
        time_indices.push(j.saturating_sub(1));
        if i == 0 {
            j -= 1;
            continue;
        }
        if j == 0 {
            i -= 1;
            continue;
        }
        match trace[idx(i, j)] {
            0 => {
                i -= 1;
                j -= 1;
            }
            1 => i -= 1,
            _ => j -= 1,
        }
    }
    text_indices.reverse();
    time_indices.reverse();
    (text_indices, time_indices)
}

/// Split token ids into word groups using decoded text whitespace boundaries.
pub fn split_to_word_tokens(
    tokenizer: &tokenizers::Tokenizer,
    text_tokens: &[u32],
) -> (Vec<String>, Vec<Vec<u32>>) {
    if text_tokens.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let mut words = Vec::new();
    let mut word_tokens: Vec<Vec<u32>> = Vec::new();
    let mut current = Vec::new();
    for &tok in text_tokens {
        current.push(tok);
        let piece = tokenizer.decode(&[tok], false).unwrap_or_default();
        if (piece.starts_with(' ') || piece.starts_with('\u{3000}')) && current.len() > 1 {
            let prev: Vec<u32> = current[..current.len() - 1].to_vec();
            let w = tokenizer.decode(&prev, false).unwrap_or_default();
            if !w.trim().is_empty() {
                words.push(w);
                word_tokens.push(prev);
            }
            current = vec![tok];
        }
    }
    if !current.is_empty() {
        let w = tokenizer.decode(&current, false).unwrap_or_default();
        if !w.trim().is_empty() {
            words.push(w);
            word_tokens.push(current);
        }
    }
    if words.is_empty() {
        let w = tokenizer.decode(text_tokens, false).unwrap_or_default();
        words.push(w);
        word_tokens.push(text_tokens.to_vec());
    }
    (words, word_tokens)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slice_encoder_hidden_covers_range() {
        let enc_seq = 10;
        let d = 4;
        let enc: Vec<f32> = (0..enc_seq * d).map(|x| x as f32).collect();
        let (sl, n) = slice_encoder_hidden(&enc, enc_seq, d, 0.04, 0.10);
        assert_eq!(n, 3); // frames 2..4
        assert_eq!(sl.len(), n * d);
    }

    #[test]
    fn dtw_toy_path() {
        let costs = vec![
            1.0, 2.0, 3.0, //
            2.0, 1.0, 2.0,
        ];
        let (ti, fi) = dtw(&costs, 2, 3);
        assert_eq!(ti.len(), fi.len());
        assert!(!ti.is_empty());
    }
}
