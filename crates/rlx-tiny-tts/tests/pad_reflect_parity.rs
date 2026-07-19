// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//! Regression: ONNX `Pad` with `mode='reflect'`. The importer only handled
//! `constant` (zero) and `edge` (replicate); reflect fell through to zero-pad,
//! which corrupts every reflect-padded frontend — e.g. Kokoro's ISTFTNet source
//! STFT centering pad (10 samples each side), whose wrong edge fed the STFT phase
//! and blew the vocoder up. Reflect mirrors WITHOUT repeating the edge sample:
//! `[1,2,3,4]` pad (2,2) → `[3,2,1,2,3,4,3,2]`. Fixture committed; no Python.
use rlx_runtime::{DType, Device};
use std::path::PathBuf;

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/pad_reflect")
}
fn read_f32(p: &std::path::Path) -> Vec<f32> {
    std::fs::read(p)
        .unwrap_or_default()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

#[test]
fn onnx_pad_reflect_matches_onnxruntime() {
    let d = dir();
    if !d.join("pad_reflect.onnx").is_file() {
        eprintln!("skip: missing fixture");
        return;
    }
    let x = std::fs::read(d.join("x.f32")).expect("x.f32");
    let refy = read_f32(&d.join("y_ref.f32"));
    assert!(!refy.is_empty());
    let mut g =
        rlx_tiny_tts::model::compile_graph(&d, "pad_reflect", Device::Cpu, 10).expect("compile");
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
        "reflect-pad length mismatch (zero-pad fallthrough?)"
    );
    let maxd = y
        .iter()
        .zip(&refy)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("pad reflect max_abs={maxd:.3e}");
    assert!(maxd < 1e-5, "reflect-pad max_abs {maxd} above threshold");
}
