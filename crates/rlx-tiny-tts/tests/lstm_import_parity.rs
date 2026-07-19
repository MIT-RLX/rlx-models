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

//! Regression: importing a float ONNX `LSTM` to the native `Op::Lstm` matches
//! onnxruntime bit-for-bit (bidirectional, so it exercises the gate-reorder,
//! `Wb+Rb` bias fold, and X/Y layout transposes in `rlx-onnx-import`).
//!
//! Fixtures (`tests/fixtures/lstm_import/`) are a tiny bidirectional LSTM and its
//! onnxruntime `Y` reference, produced by the co-located `gen_lstm.py`. No Python
//! is needed at test time — the reference is committed.

use std::path::PathBuf;

use rlx_runtime::{DType, Device};

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/lstm_import")
}

fn read_f32(path: &std::path::Path) -> Vec<f32> {
    std::fs::read(path)
        .unwrap_or_default()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

#[test]
fn onnx_lstm_imports_with_onnxruntime_parity() {
    let dir = fixture_dir();
    let onnx = dir.join("lstm_import_fixture.onnx");
    if !onnx.is_file() {
        eprintln!("skip: missing {}", onnx.display());
        return;
    }

    let x_bytes = std::fs::read(dir.join("x.f32")).expect("x.f32");
    let y_ref = read_f32(&dir.join("y_ref.f32"));
    assert!(!y_ref.is_empty(), "empty reference");

    // seq = 5 in the fixture; component name matches the .onnx filename.
    let mut g = rlx_tiny_tts::model::compile_graph(&dir, "lstm_import_fixture", Device::Cpu, 5)
        .expect("import + compile LSTM");
    let out = g.run_typed(&[("X", &x_bytes, DType::F32)]);
    let (y_bytes, dt) = out.into_iter().next().expect("LSTM output");
    assert_eq!(dt, DType::F32);
    let y: Vec<f32> = y_bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    assert_eq!(
        y.len(),
        y_ref.len(),
        "output element count mismatch (bidirectional direction dropped?)"
    );
    let dot: f32 = y.iter().zip(&y_ref).map(|(a, b)| a * b).sum();
    let na: f32 = y.iter().map(|v| v * v).sum::<f32>().sqrt();
    let nb: f32 = y_ref.iter().map(|v| v * v).sum::<f32>().sqrt();
    let cosine = dot / (na * nb + 1e-12);
    let max_abs = y
        .iter()
        .zip(&y_ref)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    eprintln!("LSTM import parity: cosine={cosine:.6}, max_abs_diff={max_abs:.3e}");
    assert!(cosine > 0.9999, "cosine {cosine} below parity threshold");
    assert!(
        max_abs < 1e-4,
        "max_abs_diff {max_abs} above parity threshold"
    );
}
