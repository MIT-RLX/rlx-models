// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//! Regression: an ONNX linear `Resize` (half_pixel) over a 1-D length axis
//! lowers to a native 2-gather interpolation matching onnxruntime — the op the
//! Kokoro (ISTFTNet) NSF m_source needs. Fixture committed; no Python at test time.
use rlx_runtime::{DType, Device};
use std::path::PathBuf;

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/resize_linear")
}
fn read_f32(p: &std::path::Path) -> Vec<f32> {
    std::fs::read(p)
        .unwrap_or_default()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

#[test]
fn onnx_linear_resize_matches_onnxruntime() {
    let d = dir();
    if !d.join("resize_linear.onnx").is_file() {
        eprintln!("skip: missing fixture");
        return;
    }
    let x = std::fs::read(d.join("x.f32")).expect("x.f32");
    let refy = read_f32(&d.join("y_ref.f32"));
    assert!(!refy.is_empty());
    let mut g =
        rlx_tiny_tts::model::compile_graph(&d, "resize_linear", Device::Cpu, 7).expect("compile");
    let out = g.run_typed(&[("x", &x, DType::F32)]);
    let y = out
        .into_iter()
        .next()
        .expect("out")
        .0
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect::<Vec<f32>>();
    assert_eq!(y.len(), refy.len(), "linear resize length mismatch");
    let maxd = y
        .iter()
        .zip(&refy)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("linear resize max_abs={maxd:.3e}");
    assert!(maxd < 1e-4, "max_abs {maxd} above threshold");
}
