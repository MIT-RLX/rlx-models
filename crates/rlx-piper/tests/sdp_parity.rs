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

//! Parity of the native Rust StochasticDurationPredictor vs onnxruntime.
//!
//! Fixtures (`tests/fixtures/sdp_*`) are the dp input (`/enc_p/encoder/Mul_2`)
//! and the ort-predicted durations (`/Ceil`) for `scales=[0,1,0]` (deterministic:
//! zero flow noise), dumped by `scripts/split_piper.py`'s reference path. The dp
//! weight bundle lives beside the piper voice under `weights/tts/piper/rlx-split/`.
#![cfg(feature = "native")]

use std::path::PathBuf;

use rlx_piper::sdp::Sdp;

fn split_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../weights/tts/piper/rlx-split")
}

fn read_f32(p: &PathBuf) -> Vec<f32> {
    std::fs::read(p)
        .unwrap_or_else(|_| panic!("read {}", p.display()))
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn read_i64(p: &PathBuf) -> Vec<i64> {
    std::fs::read(p)
        .unwrap_or_else(|_| panic!("read {}", p.display()))
        .chunks_exact(8)
        .map(|b| i64::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

#[test]
fn sdp_durations_match_ort() {
    let dir = split_dir();
    let fix = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    if !dir.join("dp_manifest.json").is_file() || !fix.join("sdp_dp_in.f32").is_file() {
        eprintln!("skip: piper split bundle / fixtures not present (run scripts/split_piper.py)");
        return;
    }
    let sdp = Sdp::load(&dir).expect("load dp weights");
    let dp_in = read_f32(&fix.join("sdp_dp_in.f32"));
    let ort_dur: Vec<usize> = read_i64(&fix.join("sdp_durations.i64"))
        .into_iter()
        .map(|d| d.max(0) as usize)
        .collect();
    let seq = ort_dur.len();
    assert_eq!(dp_in.len(), 192 * seq, "dp_in must be [192, seq]");

    // scales=[0,1,0]: zero flow noise, length_scale 1 → deterministic.
    let noise = vec![0.0f32; 2 * seq];
    let dur = sdp.durations(&dp_in, seq, 1.0, &noise);

    // Spline coupling flows are bin-boundary sensitive: allow ±1 at rare phonemes
    // (perceptually identical, whisper-validated), require the bulk to match and
    // the total frame count to stay within 2%.
    let exact = dur.iter().zip(&ort_dur).filter(|(a, b)| a == b).count();
    let off_by_one = dur
        .iter()
        .zip(&ort_dur)
        .all(|(a, b)| (*a as i64 - *b as i64).abs() <= 1);
    let sum_np: usize = dur.iter().sum();
    let sum_ort: usize = ort_dur.iter().sum();
    eprintln!("native durations: {dur:?}");
    eprintln!("ort    durations: {ort_dur:?}");
    eprintln!("exact {exact}/{seq}, native sum {sum_np}, ort sum {sum_ort}");
    assert!(off_by_one, "some duration differs by >1 from ort");
    assert!(
        exact * 100 >= seq * 85,
        "only {exact}/{seq} durations exact (<85%)"
    );
    let tol = (sum_ort as f64 * 0.02).ceil() as i64 + 1;
    assert!(
        (sum_np as i64 - sum_ort as i64).abs() <= tol,
        "total frames {sum_np} vs ort {sum_ort} exceeds 2% tolerance"
    );
}
