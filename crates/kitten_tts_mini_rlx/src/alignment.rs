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

//! Fused SplitToSequence + Loop + ConcatFromSequence for Kitten duration alignment.

/// Prefix constant in ONNX Loop body `Concat_2` (`/Constant_31_output_0` = 1).
#[cfg(test)]
const LOOP_PREFIX: i64 = 1;

fn split_1d(data: &[i64], lens: &[i64]) -> Vec<Vec<i64>> {
    let mut out = Vec::new();
    let mut pos = 0usize;
    for &len in lens {
        let n = len.max(0) as usize;
        let end = (pos + n).min(data.len());
        out.push(data[pos..end].to_vec());
        pos = end;
        if pos >= data.len() {
            break;
        }
    }
    if out.is_empty() && !data.is_empty() {
        out.push(data.to_vec());
    }
    out
}

/// Legacy loop-body model (prefix + frame); kept for tests comparing old behavior.
#[cfg(test)]
fn loop_body_frame_legacy(frame0: &[i64], range_id: i64) -> Vec<i64> {
    let mut concat2 = vec![LOOP_PREFIX];
    concat2.extend_from_slice(frame0);
    let len = concat2.len().max(1);
    vec![range_id; len]
}

/// ONNX Loop body for Kitten: `Expand(range_id, duration)` → `duration` copies of `range_id`.
fn loop_body_frame(duration: i64, range_id: i64) -> Vec<i64> {
    let d = duration.max(0) as usize;
    vec![range_id; d]
}

/// Total alignment frame count (sum of per-token durations).
pub fn alignment_frame_count(duration_mask: &[i64]) -> usize {
    duration_mask.iter().map(|&d| d.max(0) as usize).sum()
}

/// Upper bound on alignment frames for static compile shapes (`seq * max_frames_per_token`).
pub fn alignment_frame_upper_bound(sequence_length: usize) -> usize {
    sequence_length.saturating_mul(crate::bundle_compile::MAX_FRAMES_PER_TOKEN)
}

fn per_trip_scalars(data: &[i64], split_lens: &[i64], trip_count: usize) -> Vec<i64> {
    if split_lens.len() <= 1 {
        return (0..trip_count)
            .map(|i| data.get(i).copied().unwrap_or(0))
            .collect();
    }
    let splits = split_1d(data, split_lens);
    (0..trip_count)
        .map(|i| splits.get(i).and_then(|v| v.first().copied()).unwrap_or(0))
        .collect()
}

/// Concatenate per-trip alignment rows (i64), matching Kitten ONNX Loop + ConcatFromSequence.
pub fn concat_alignment_durations(
    duration_mask: &[i64],
    range_ids: &[i64],
    split_lens: &[i64],
    trip_count: usize,
    out: &mut [i64],
) {
    let trips = per_trip_scalars(duration_mask, split_lens, trip_count);
    let ranges = per_trip_scalars(range_ids, split_lens, trip_count);
    let mut pos = 0usize;
    for i in 0..trip_count {
        let duration = trips.get(i).copied().unwrap_or(0);
        let rid = ranges.get(i).copied().unwrap_or(i as i64);
        let row = loop_body_frame(duration, rid);
        for v in row {
            if pos < out.len() {
                out[pos] = v;
                pos += 1;
            }
        }
    }
    for slot in out.iter_mut().skip(pos) {
        *slot = 0;
    }
}

/// Build ONNX `/Concat_4` scatter index pairs `[token_row, mel_col]` for `frames` steps.
pub fn alignment_scatter_index_pairs(token_ids: &[i64], frames: usize, out: &mut [i64]) {
    let pairs = out.len() / 2;
    let n = frames.min(pairs).min(token_ids.len());
    for j in 0..n {
        out[j * 2] = token_ids[j];
        out[j * 2 + 1] = j as i64;
    }
    for slot in out.iter_mut().skip(n.saturating_mul(2)) {
        *slot = 0;
    }
}

#[cfg(test)]
mod scatter_index_tests {
    use super::*;

    #[test]
    fn scatter_pairs_match_ort_hello() {
        let mask = vec![19i64, 2, 1, 2, 3, 2, 3, 2];
        let range = (0i64..8).collect::<Vec<_>>();
        let mut tokens = vec![0i64; 64];
        concat_alignment_durations(&mask, &range, &[1i64; 8], 8, &mut tokens);
        let mut idx = vec![0i64; 64 * 2];
        alignment_scatter_index_pairs(&tokens, 34, &mut idx);
        assert_eq!(idx[0], 0);
        assert_eq!(idx[1], 0);
        assert_eq!(idx[2], 0);
        assert_eq!(idx[3], 1);
        assert_eq!(idx[36], 0);
        assert_eq!(idx[37], 18);
        assert_eq!(idx[38], 1);
        assert_eq!(idx[39], 19);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loop_body_repeats_range_id_by_duration() {
        assert_eq!(loop_body_frame(19, 0), vec![0; 19]);
        assert_eq!(loop_body_frame(2, 1), vec![1, 1]);
    }

    #[test]
    fn concat_broadcast_scalar_split_lens() {
        let mask = vec![19i64, 2, 1, 2, 3, 2, 3, 2];
        let range = (0i64..8).collect::<Vec<_>>();
        let mut out_full = vec![0i64; 64];
        concat_alignment_durations(&mask, &range, &[1i64; 8], 8, &mut out_full);
        let mut out_scalar = vec![0i64; 64];
        concat_alignment_durations(&mask, &range, &[1i64], 8, &mut out_scalar);
        assert_eq!(out_scalar, out_full);
    }

    #[test]
    fn concat_matches_ort_jfk_pattern() {
        let mask = vec![19i64, 2, 1, 2, 3, 2, 3, 2];
        let range = (0i64..8).collect::<Vec<_>>();
        let lens = vec![1i64; 8];
        let mut out = vec![0i64; 64];
        concat_alignment_durations(&mask, &range, &lens, 8, &mut out);
        let expected = [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 3, 3, 4, 4, 4, 5, 5,
            6,
        ];
        assert_eq!(&out[..expected.len()], expected);
        assert_eq!(alignment_frame_count(&mask), 34);
    }

    #[test]
    fn legacy_loop_body_differs() {
        let row = loop_body_frame_legacy(&[2, 3], 5);
        assert_eq!(row, vec![5, 5, 5]);
    }
}
