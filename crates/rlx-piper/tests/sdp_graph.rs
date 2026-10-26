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

//! The duration predictor as an rlx graph predicts the frames the Rust
//! one does, on the fixture's encoder output, with no noise and with some.
#![cfg(feature = "export")]

use std::path::PathBuf;

use rlx_piper::sdp::Sdp;
use rlx_piper::sdp_graph;

#[test]
fn the_graph_predicts_the_durations_the_rust_predictor_does() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../weights/tts/piper/rlx-split");
    let fix = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/sdp_dp_in.f32");
    if !dir.join("dp_manifest.json").is_file() || !fix.is_file() {
        eprintln!("skip: no piper split bundle");
        return;
    }
    let dp_in: Vec<f32> = std::fs::read(&fix)
        .unwrap()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    let t = dp_in.len() / 192;
    let (dims, values) = sdp_graph::weights(&dir).unwrap();
    let sdp = Sdp::load(&dir).unwrap();
    let mut c =
        rlx_runtime::Session::new(rlx_runtime::Device::Cpu).compile(sdp_graph::build(&dims, t));
    for (k, v) in &values {
        c.set_param(k, v);
    }
    c.finalize_params();
    for (noise_w, scale) in [(0.0f32, 1.0f32), (0.8, 1.0), (0.8, 1.3)] {
        let noise: Vec<f32> = (0..2 * t)
            .map(|i| noise_w * ((i as f32 * 0.7).sin() + 0.5 * (i as f32 * 1.9).cos()))
            .collect();
        let want = sdp.durations(&dp_in, t, scale, &noise);
        let got = c
            .run(&[
                ("dp_in", &dp_in),
                ("noise", &noise),
                ("length_scale", &[scale]),
            ])
            .remove(0);
        let got: Vec<usize> = got.iter().map(|v| *v as usize).collect();
        eprintln!("noise {noise_w}, scale {scale}: {got:?}");
        assert_eq!(got, want, "noise {noise_w}, scale {scale}");
    }
}
