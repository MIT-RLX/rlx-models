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

//! Segment ("block") attention masks for the codec's local attention stacks.
//!
//! Both the codec encoder and the codec decoder run a 6-layer transformer over
//! the 50 Hz frame grid, but each restricts attention with a *different* rule
//! keyed off `token_mask` — the binary vector marking which frames a text
//! token landed on. Upstream defines two same-named `_create_segment_attention_mask`
//! functions, one in `encoder.py` and one in `decoder.py`; they are **not**
//! interchangeable, so both are ported here under distinct names.
//!
//! Returned masks are additive attention biases (`0.0` = attend,
//! `f32::NEG_INFINITY` = blocked), laid out `[q, k]` row-major, which is what
//! `HirGraphExt::attention_bias` consumes.

const BLOCKED: f32 = f32::NEG_INFINITY;

/// Prefix sums of `token_mask`, i.e. `cumsum(mask)`.
fn cumsum(token_mask: &[u8]) -> Vec<i32> {
    let mut acc = 0i32;
    token_mask
        .iter()
        .map(|&m| {
            acc += m as i32;
            acc
        })
        .collect()
}

/// `cumsum(mask) - mask` — marked positions share the block id of the frames
/// that precede them, so a block *ends* at its marked frame.
fn cumsum_exclusive(token_mask: &[u8]) -> Vec<i32> {
    let mut acc = 0i32;
    token_mask
        .iter()
        .map(|&m| {
            acc += m as i32;
            acc - m as i32
        })
        .collect()
}

/// Codec **encoder** mask (`encoder.py`, `version="v2"`).
///
/// Blocks start at a marked frame. A frame may attend inside its own block but
/// never *to* a marked frame — unless it is itself marked, in which case it may
/// also reach back into the previous block's unmarked frames. That keeps the
/// marked frames (the ones whose activations become the token's acoustic
/// latent) from being read by their own neighbours, so each latent stays a
/// summary of its span rather than a copy of the frame under it.
pub fn encoder_segment_bias(token_mask: &[u8]) -> Vec<f32> {
    let n = token_mask.len();
    let block = cumsum(token_mask);
    let mut out = vec![BLOCKED; n * n];
    for i in 0..n {
        let marked_i = token_mask[i] != 0;
        for j in 0..n {
            let marked_j = token_mask[j] != 0;
            let same = block[i] == block[j];
            // `same & (~marked_j | (marked_i & same))` — a marked key is only
            // readable by a marked query in the same block.
            let same_ok = same && (!marked_j || marked_i);
            let prev_ok = marked_i && block[j] == block[i] - 1 && !marked_j;
            if same_ok || prev_ok {
                out[i * n + j] = 0.0;
            }
        }
    }
    out
}

/// Codec **decoder** mask (`decoder.py`, `version="v2"`).
///
/// Blocks end at a marked frame; every frame sees its own block and the one
/// before it. Simpler than the encoder rule — the decoder is reconstructing a
/// waveform, so it wants local context, not information isolation.
pub fn decoder_segment_bias(token_mask: &[u8]) -> Vec<f32> {
    let n = token_mask.len();
    let block = cumsum_exclusive(token_mask);
    let mut out = vec![BLOCKED; n * n];
    for i in 0..n {
        for j in 0..n {
            if block[j] == block[i] || block[j] == block[i] - 1 {
                out[i * n + j] = 0.0;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn attends(bias: &[f32], n: usize, i: usize, j: usize) -> bool {
        bias[i * n + j] == 0.0
    }

    #[test]
    fn decoder_blocks_end_at_marks_and_see_one_block_back() {
        // frames:      0  1  2  3  4
        // mask:        0  1  0  1  0   → block ids 0 0 1 1 2
        let m = [0u8, 1, 0, 1, 0];
        let b = decoder_segment_bias(&m);
        assert!(attends(&b, 5, 0, 1), "same block");
        assert!(attends(&b, 5, 2, 1), "previous block");
        assert!(!attends(&b, 5, 4, 1), "two blocks back is blocked");
        assert!(attends(&b, 5, 4, 3), "previous block");
    }

    #[test]
    fn encoder_hides_marked_frames_from_unmarked_queries() {
        // mask: 1 0 0 1 0 → block ids (inclusive cumsum) 1 1 1 2 2
        let m = [1u8, 0, 0, 1, 0];
        let b = encoder_segment_bias(&m);
        assert!(
            !attends(&b, 5, 1, 0),
            "unmarked query cannot read a marked key"
        );
        assert!(
            attends(&b, 5, 1, 2),
            "unmarked keys in the same block are fine"
        );
        assert!(attends(&b, 5, 0, 0), "marked query reads itself");
        assert!(
            attends(&b, 5, 3, 1),
            "marked query reaches the previous block"
        );
        assert!(!attends(&b, 5, 3, 0), "…but not its marked frames");
    }

    #[test]
    fn every_query_can_reach_at_least_one_key() {
        // A fully blocked row would make softmax produce NaN on every backend.
        for m in [
            vec![0u8; 8],
            vec![1u8; 8],
            vec![1, 0, 0, 1, 0, 0, 1, 0],
            vec![0, 0, 1, 1, 0, 1, 0, 0],
        ] {
            for (name, bias) in [
                ("encoder", encoder_segment_bias(&m)),
                ("decoder", decoder_segment_bias(&m)),
            ] {
                let n = m.len();
                for i in 0..n {
                    assert!(
                        (0..n).any(|j| bias[i * n + j] == 0.0),
                        "{name}: row {i} of {m:?} is fully masked"
                    );
                }
            }
        }
    }

    #[test]
    fn unmarked_input_makes_one_all_to_all_block() {
        let m = vec![0u8; 4];
        let b = decoder_segment_bias(&m);
        assert!(b.iter().all(|&v| v == 0.0));
    }
}
