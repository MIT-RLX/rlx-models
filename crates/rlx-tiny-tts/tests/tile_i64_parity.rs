// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//! Regression: `Tile` (and any `Concat`) of an **i64** tensor. The CPU `Concat`
//! thunk only special-cased F64 for the 8-byte copy path; every other dtype
//! (including I64) went through the f32 4-byte path, copying only the first half
//! of each input and zero-filling the tail — `Tile([9..0], 3)` on i64 gave
//! `[9,8,7,6,5]×3 + zeros`. This corrupted the Zipformer relative-position
//! rel-shift indices in LuxTTS/ZipVoice's fm_decoder (and any i64 concat). Fix
//! routes concat by element SIZE (8-byte → f64 copy path). Fixture committed.
use rlx_runtime::{DType, Device};
use std::path::PathBuf;

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tile_i64")
}

#[test]
fn onnx_tile_i64_matches_onnxruntime() {
    let d = dir();
    if !d.join("tile_i64.onnx").is_file() {
        eprintln!("skip: missing fixture");
        return;
    }
    let x = std::fs::read(d.join("x.i64")).expect("x.i64");
    let refy: Vec<i64> = std::fs::read(d.join("y_ref.i64"))
        .expect("y_ref")
        .chunks_exact(8)
        .map(|b| i64::from_le_bytes(b.try_into().unwrap()))
        .collect();
    assert!(!refy.is_empty());
    let mut g =
        rlx_tiny_tts::model::compile_graph(&d, "tile_i64", Device::Cpu, 10).expect("compile");
    let out = g.run_typed(&[("x", &x, DType::I64)]);
    let (bytes, _dt) = out.into_iter().next().expect("out");
    let y: Vec<i64> = bytes
        .chunks_exact(8)
        .map(|b| i64::from_le_bytes(b.try_into().unwrap()))
        .collect();
    assert_eq!(
        y.len(),
        refy.len(),
        "tile-i64 length mismatch (4-byte concat path?)"
    );
    assert_eq!(y, refy, "tile-i64 values mismatch (i64 concat corruption)");
}
