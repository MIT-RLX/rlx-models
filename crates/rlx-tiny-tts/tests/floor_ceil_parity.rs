// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//! Regression: ONNX `Floor`/`Ceil` lower as `round(x) ∓ (round(x) ≷ x)`. The
//! `Compare` term is a *bool* tensor — before the cast fix it was read as f32
//! (a `0x01` byte → denormal ~1e-45), so the ±1 correction vanished and both ops
//! collapsed to bare `Round` whenever rounding crossed an integer (`Floor(0.6)→1`,
//! `Ceil(0.4)→0`). Kokoro's ISTFTNet NSF phase (`Div/Floor/Sub` = `mod 1`) needs
//! this exact. Fixture committed; no Python at test time.
use rlx_runtime::{DType, Device};
use std::path::{Path, PathBuf};

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/floor_ceil")
}
fn read_f32(p: &Path) -> Vec<f32> {
    std::fs::read(p)
        .unwrap_or_default()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn run_component(d: &Path, comp: &'static str, x: &[u8]) -> Vec<f32> {
    let mut g = rlx_tiny_tts::model::compile_graph(d, comp, Device::Cpu, 12).expect("compile");
    let out = g.run_typed(&[("x", x, DType::F32)]);
    out.into_iter()
        .next()
        .expect("out")
        .0
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

#[test]
fn onnx_floor_matches_onnxruntime() {
    let d = dir();
    if !d.join("floor_min.onnx").is_file() {
        eprintln!("skip: missing fixture");
        return;
    }
    let x = std::fs::read(d.join("x.f32")).expect("x.f32");
    let refy = read_f32(&d.join("y_ref.f32"));
    assert!(!refy.is_empty());
    let y = run_component(&d, "floor_min", &x);
    assert_eq!(y.len(), refy.len());
    let maxd = y
        .iter()
        .zip(&refy)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("floor max_abs={maxd:.3e}");
    assert!(
        maxd < 1e-5,
        "floor max_abs {maxd} above threshold (round-crossing bug)"
    );
}

#[test]
fn onnx_ceil_matches_onnxruntime() {
    let d = dir();
    if !d.join("ceil_min.onnx").is_file() {
        eprintln!("skip: missing fixture");
        return;
    }
    let x = std::fs::read(d.join("x.f32")).expect("x.f32");
    let refy = read_f32(&d.join("y_ceil_ref.f32"));
    assert!(!refy.is_empty());
    let y = run_component(&d, "ceil_min", &x);
    assert_eq!(y.len(), refy.len());
    let maxd = y
        .iter()
        .zip(&refy)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("ceil max_abs={maxd:.3e}");
    assert!(
        maxd < 1e-5,
        "ceil max_abs {maxd} above threshold (round-crossing bug)"
    );
}
