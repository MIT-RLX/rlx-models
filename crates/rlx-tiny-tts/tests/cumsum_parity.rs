// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//! Regression: ONNX `CumSum` over a *non-last* axis. The kernel scans the last
//! axis, so the importer swaps `axis`↔`last`, scans, then must swap back. Before
//! the fix the swap-back was missing, so the result stayed in the transposed
//! layout while wearing the original output shape (`[1,9,148]` mislabelled as
//! `[1,148,9]`) — Kokoro's ISTFTNet NSF phase accumulation `CumSum(axis=1)` came
//! out permuted. Fixture committed; no Python at test time.
use rlx_runtime::{DType, Device};
use std::path::{Path, PathBuf};

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/cumsum_axis1")
}
fn read_f32(p: &Path) -> Vec<f32> {
    std::fs::read(p)
        .unwrap_or_default()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

#[test]
fn onnx_cumsum_nonlast_axis_matches_onnxruntime() {
    let d = dir();
    if !d.join("cumsum_axis1.onnx").is_file() {
        eprintln!("skip: missing fixture");
        return;
    }
    let x = std::fs::read(d.join("x.f32")).expect("x.f32");
    let refy = read_f32(&d.join("y_ref.f32"));
    assert!(!refy.is_empty());
    let mut g =
        rlx_tiny_tts::model::compile_graph(&d, "cumsum_axis1", Device::Cpu, 4).expect("compile");
    let out = g.run_typed(&[("x", &x, DType::F32)]);
    let y = out
        .into_iter()
        .next()
        .expect("out")
        .0
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect::<Vec<f32>>();
    // Element order matters here: a missing swap-back permutes the layout while
    // preserving the multiset of values, so compare position-by-position.
    assert_eq!(y.len(), refy.len(), "cumsum length mismatch");
    let maxd = y
        .iter()
        .zip(&refy)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("cumsum axis1 max_abs={maxd:.3e}  y={y:?}");
    assert!(
        maxd < 1e-4,
        "cumsum max_abs {maxd} above threshold (transpose-back layout bug)"
    );
}
