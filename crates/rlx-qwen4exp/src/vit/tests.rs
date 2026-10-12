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

//! The tower's host-side invariants: the config's derived widths, the merge
//! permutation, the axial-RoPE layout against its closed form, the position
//! table's resize, and the name/shape tables against each other.
//!
//! The *graph* is covered by `tests/vit_parity.rs`, against `rlx-qwen35`'s
//! independent implementation of the same architecture.

use super::preprocess::position_embd_for_grid;
use super::*;

fn tiny() -> VisionConfig {
    VisionConfig {
        patch_size: 2,
        n_embd: 16,
        n_head: 2,
        n_layer: 2,
        image_size: 8,
        n_ff: 24,
        n_merge: 2,
        eps: 1e-6,
        projector_type: "qwen3vl_merger".into(),
        image_mean: [0.5; 3],
        image_std: [0.5; 3],
        llm_hidden_size: 16,
        image_min_pixels: 16,
        image_max_pixels: 1024,
        rope_theta: 10_000.0,
        deepstack_layers: vec![],
    }
}

#[test]
fn the_tiny_config_validates() {
    tiny().validate().unwrap();
    assert_eq!(tiny().head_dim(), 8);
    assert_eq!(tiny().align_size(), 4);
    assert_eq!(tiny().merger_in(), 64);
}

/// A head that is not a multiple of 4 cannot carry the axial RoPE's
/// row/column split-then-duplicate layout.
#[test]
fn a_head_that_cannot_hold_the_axial_rope_is_rejected() {
    let mut c = tiny();
    c.n_embd = 18;
    c.n_head = 3; // head_dim 6
    let err = c.validate().unwrap_err().to_string();
    assert!(err.contains("multiple of 4"), "{err}");
}

#[test]
fn smart_resize_aligns_and_respects_the_bounds() {
    let c = tiny();
    let (w, h) = smart_resize(&c, 100, 50);
    assert_eq!(w % c.align_size(), 0);
    assert_eq!(h % c.align_size(), 0);
    assert!(w * h >= c.image_min_pixels && w * h <= c.image_max_pixels);
}

#[test]
fn the_merge_permutation_groups_raster_blocks() {
    assert_eq!(
        spatial_merge_gather_idx(4, 4, 2),
        vec![
            0.0, 1.0, 4.0, 5.0, 2.0, 3.0, 6.0, 7.0, 8.0, 9.0, 12.0, 13.0, 10.0, 11.0, 14.0, 15.0
        ]
    );
}

/// The RoPE layout is what makes `rotate_half` pair row with row: the
/// second half of each head must duplicate the first.
#[test]
fn the_axial_rope_duplicates_its_first_half() {
    let c = tiny();
    let hd = c.head_dim();
    let (cos, sin) = vision_rope_tables(&c, &[(3, 5)]);
    assert_eq!(cos.len(), hd);
    for i in 0..hd / 2 {
        assert_eq!(cos[i], cos[i + hd / 2], "cos half {i}");
        assert_eq!(sin[i], sin[i + hd / 2], "sin half {i}");
    }
    // The first quarter is row frequencies, the second column.
    let axis = hd / 4;
    let f0 = 1.0f64; // j = 0 → inv_freq 1.0
    assert!((sin[0] as f64 - (3.0 * f0).sin()).abs() < 1e-6, "row");
    assert!((sin[axis] as f64 - (5.0 * f0).sin()).abs() < 1e-6, "column");
}

/// At position 0 the rotation is the identity, whatever the grid.
#[test]
fn position_zero_is_the_identity_rotation() {
    let c = tiny();
    let (cos, sin) = vision_rope_tables(&c, &[(0, 0)]);
    assert!(cos.iter().all(|v| (*v - 1.0).abs() < 1e-7));
    assert!(sin.iter().all(|v| v.abs() < 1e-7));
}

/// Resizing a table that already matches the grid must be the identity, and
/// the permutation must be applied in both cases.
#[test]
fn the_position_table_resizes_and_permutes() {
    let n = 2usize;
    // A 2x2 source grid, rows = [0, 1, 2, 3] scaled.
    let table: Vec<f32> = (0..4 * n).map(|i| i as f32).collect();
    let idx = spatial_merge_gather_idx(2, 2, 2);
    let out = position_embd_for_grid(&table, n, 2, 2, &idx).unwrap();
    // 2x2 with merge 2 is one block in raster order, so the permutation is
    // the identity and the resize is a copy.
    assert_eq!(out, table);

    // A 4x4 target from the 2x2 source: corners are preserved exactly.
    let idx4 = spatial_merge_gather_idx(4, 4, 2);
    let out4 = position_embd_for_grid(&table, n, 4, 4, &idx4).unwrap();
    assert_eq!(out4.len(), 16 * n);
    // Target (0,0) ← source (0,0) = row 0.
    let first = idx4.iter().position(|&i| i == 0.0).unwrap();
    assert_eq!(&out4[first * n..first * n + n], &table[0..n]);
}

#[test]
fn a_non_square_position_table_is_rejected() {
    let table = vec![0f32; 3 * 2];
    let idx = spatial_merge_gather_idx(2, 2, 2);
    let err = position_embd_for_grid(&table, 2, 2, 2, &idx)
        .unwrap_err()
        .to_string();
    assert!(err.contains("square grid"), "{err}");
}

#[test]
fn the_shape_table_covers_exactly_the_name_list() {
    let c = tiny();
    let mut from_shapes: Vec<String> = vision_tensor_shapes(&c, 16)
        .into_iter()
        .map(|(n, _)| n)
        .collect();
    from_shapes.sort();
    assert_eq!(from_shapes, vision_tensor_names(&c));
}
