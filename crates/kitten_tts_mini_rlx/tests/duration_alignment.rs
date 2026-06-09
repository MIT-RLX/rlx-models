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

//! Duration `ConcatFromSequence` fusion reference (ORT alignment semantics).

use rlx_onnx_import::{
    alignment_frame_count, concat_alignment_durations, resolve_duration_align_inputs,
};

mod common;

#[test]
fn resolve_duration_align_inputs_from_bundle() {
    let Some(bundle) = common::load_bundle() else {
        return;
    };
    let inputs = resolve_duration_align_inputs(&bundle.nodes).expect("resolve");
    assert_eq!(inputs.duration_mask, "/Where_1_output_0");
    assert_eq!(inputs.range_ids, "/Reshape_1_output_0");
    assert_eq!(inputs.trip_count, "/Unsqueeze_4_output_0");
}

#[test]
fn concat_alignment_matches_reference_pattern() {
    let mask = vec![19i64, 2, 1, 2, 3, 2, 3, 2];
    let range = (0i64..8).collect::<Vec<_>>();
    let lens = vec![1i64; 8];
    let mut out = vec![0i64; 64];
    concat_alignment_durations(&mask, &range, &lens, 8, &mut out);
    let expected = [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 3, 3, 4, 4, 4, 5, 5, 6,
        6, 6, 7, 7,
    ];
    assert_eq!(&out[..expected.len()], expected);
    assert_eq!(alignment_frame_count(&mask), 34);
}
