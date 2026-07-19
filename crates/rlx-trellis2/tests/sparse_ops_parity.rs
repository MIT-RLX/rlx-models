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

//! Parity for the submanifold sparse conv against a dense `conv3d` oracle.
//! Gated on `RLX_TRELLIS2_SUBM_REF` (`scripts/subm_conv_ref.py`).

use rlx_trellis2::sparse::{SparseTensor, submanifold_conv3d};
use safetensors::SafeTensors;

fn f32s(st: &SafeTensors, name: &str) -> Vec<f32> {
    st.tensor(name)
        .unwrap()
        .data()
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}
fn i32s(st: &SafeTensors, name: &str) -> Vec<i32> {
    st.tensor(name)
        .unwrap()
        .data()
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[test]
fn submanifold_conv_matches_dense() {
    let Ok(refp) = std::env::var("RLX_TRELLIS2_SUBM_REF") else {
        eprintln!("skipping subm conv parity: set RLX_TRELLIS2_SUBM_REF");
        return;
    };
    let bytes = std::fs::read(&refp).unwrap();
    let st = SafeTensors::deserialize(&bytes).unwrap();

    let coords_flat = i32s(&st, "coords");
    let coords: Vec<[i32; 3]> = coords_flat
        .chunks_exact(3)
        .map(|c| [c[0], c[1], c[2]])
        .collect();
    let feats = f32s(&st, "feats");
    let weight = f32s(&st, "weight"); // [out,3,3,3,in]
    let bias = f32s(&st, "bias");
    let out_ref = f32s(&st, "out");
    let out_c = bias.len();
    let in_c = feats.len() / coords.len();

    let x = SparseTensor::new(feats, coords, in_c);
    let out = submanifold_conv3d(&x, &weight, &bias, out_c);

    let maxabs = out
        .feats
        .iter()
        .zip(&out_ref)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!(
        "subm conv: n={} in={in_c} out={out_c} maxabs {maxabs:.3e}",
        x.n()
    );
    assert!(maxabs < 1e-4, "submanifold conv mismatch: {maxabs}");
}
