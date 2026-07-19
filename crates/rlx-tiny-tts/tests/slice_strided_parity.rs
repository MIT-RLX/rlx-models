// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//! Regression: ONNX `Slice` with `step > 1` (strided). The importer bailed on any
//! `step != 1`, so strided slices silently kept the FULL axis dim — e.g. the RoPE
//! interleaved even/odd head-dim split `x[..., 0::2]` / `x[..., 1::2]` (halving
//! 64→32) stayed 64, doubling every downstream shape and blowing up the rope
//! broadcast in moss-nano / maya1 / chatterbox LMs. Fix lowers a single-axis
//! positive strided slice to `Gather{axis}` with strided indices. Fixture: two
//! strided slices (even+odd) concatenated. Committed; no Python at test time.
use rlx_runtime::{DType, Device};
use std::path::PathBuf;

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/slice_strided")
}
fn read_f32(p: &std::path::Path) -> Vec<f32> {
    std::fs::read(p)
        .unwrap_or_default()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

#[test]
fn onnx_strided_slice_matches_onnxruntime() {
    let d = dir();
    if !d.join("slice_strided.onnx").is_file() {
        eprintln!("skip: missing fixture");
        return;
    }
    let x = std::fs::read(d.join("x.f32")).expect("x.f32");
    let refy = read_f32(&d.join("y_ref.f32"));
    assert!(!refy.is_empty());
    let mut g =
        rlx_tiny_tts::model::compile_graph(&d, "slice_strided", Device::Cpu, 8).expect("compile");
    let out = g.run_typed(&[("x", &x, DType::F32)]);
    let y = out
        .into_iter()
        .next()
        .expect("out")
        .0
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect::<Vec<f32>>();
    assert_eq!(
        y.len(),
        refy.len(),
        "strided-slice length mismatch (step ignored?)"
    );
    let maxd = y
        .iter()
        .zip(&refy)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("strided slice max_abs={maxd:.3e}");
    assert!(maxd < 1e-5, "strided-slice max_abs {maxd} above threshold");
}
