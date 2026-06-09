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

//! MTP 2D attention mask layout (no weights).

use rlx_locateanything::mask::{
    attn_bias_for_incremental, attn_bias_from_2d, causal_mask_f32, mtp_prefill_mask_2d,
    mtp_window_mask_f32,
};

#[test]
fn mtp_prefill_tail_is_bidirectional() {
    let seq = 10usize;
    let block = 6usize;
    let ids: Vec<u32> = (0..seq as u32).collect();
    let m = mtp_prefill_mask_2d(&ids, 99, block, false, false);
    assert_eq!(m[(seq - 1) * seq + (seq - 2)], 0.0);
    assert_eq!(m[(seq - 1) * seq + (seq - 1)], 0.0);
    assert!(m[seq - 1].is_infinite());
}

#[test]
fn attn_bias_layout_matches_heads() {
    let seq = 4usize;
    let m2d = mtp_window_mask_f32(seq, 2, false);
    let bias = attn_bias_from_2d(1, 2, seq, &m2d);
    assert_eq!(bias.len(), 2 * seq * seq);
    assert_eq!(bias[0], m2d[0]);
    assert_eq!(bias[seq * seq], m2d[0]);
}

#[test]
fn incremental_bias_slices_query_rows() {
    let seq = 8usize;
    let past = 5usize;
    let q = 3usize;
    let full = mtp_window_mask_f32(seq, q, true);
    let inc = attn_bias_for_incremental(1, 2, past, q, &full, seq);
    assert_eq!(inc.len(), 2 * q * (past + q));
    assert_eq!(inc[0], full[past * seq]);
}

#[test]
fn causal_upper_triangle_blocked() {
    let m = causal_mask_f32(5);
    assert_eq!(m[0], 0.0);
    assert!(m[1].is_infinite());
}

#[test]
fn mtp_prefill_mask_hf_fixture_cells() {
    let seq = 18usize;
    let block = 6usize;
    let text_mask = 151_676u32;
    let mut ids: Vec<u32> = (0..seq - block).map(|i| 10 + i as u32 * 10).collect();
    ids.extend(std::iter::repeat_n(text_mask, block - 2));
    ids.extend([1000, 1001]);
    let m = mtp_prefill_mask_2d(&ids, text_mask, block, false, false);
    assert_eq!(m[(seq - 1) * seq + (seq - 1)], 0.0);
    assert_eq!(m[(seq - 1) * seq + (seq - 2)], 0.0);
    assert!(m[seq - 1].is_infinite());
    assert!(m[11].is_infinite(), "causal block at (0,11): got {}", m[11]);
}

#[test]
fn mtp_pad_non_visible_matches_hf_visible_rule() {
    let seq = 18usize;
    let text_mask = 151_676u32;
    let mut ids: Vec<u32> = (0..seq - 6).map(|i| 10 + i as u32 * 10).collect();
    ids.extend(std::iter::repeat_n(text_mask, 4));
    ids.extend([1000, 1001]);
    let m = mtp_prefill_mask_2d(&ids, text_mask, 6, false, false);
    assert!(m[11].is_infinite());
}
