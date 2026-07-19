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

//! Regression: importing an ONNX `STFT` (opset 17) to a native constant-DFT
//! subgraph (framing gather + window-folded cos/-sin matmuls + real/imag
//! interleave) matches onnxruntime. Exercises `lower_stft` in `rlx-onnx-import`
//! — the op the Kokoro (ISTFTNet) decoder needs.
//!
//! Fixtures (`tests/fixtures/stft_import/`) are a tiny STFT (n_fft=20, hop=5,
//! Hann window, onesided) and its onnxruntime reference, produced by the
//! co-located `gen_stft.py`. No Python is needed at test time.

use std::path::PathBuf;

use rlx_runtime::{DType, Device};

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/stft_import")
}

fn read_f32(path: &std::path::Path) -> Vec<f32> {
    std::fs::read(path)
        .unwrap_or_default()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

#[test]
fn onnx_stft_imports_with_onnxruntime_parity() {
    let dir = fixture_dir();
    let onnx = dir.join("stft_import_fixture.onnx");
    if !onnx.is_file() {
        eprintln!("skip: missing {}", onnx.display());
        return;
    }

    let sig_bytes = std::fs::read(dir.join("signal.f32")).expect("signal.f32");
    let ref_out = read_f32(&dir.join("stft_ref.f32"));
    assert!(!ref_out.is_empty(), "empty reference");

    // signal length = 200 in the fixture; component name matches the .onnx filename.
    let mut g = rlx_tiny_tts::model::compile_graph(&dir, "stft_import_fixture", Device::Cpu, 200)
        .expect("import + compile STFT");
    let out = g.run_typed(&[("signal", &sig_bytes, DType::F32)]);
    let (y_bytes, dt) = out.into_iter().next().expect("STFT output");
    assert_eq!(dt, DType::F32);
    let y = y_bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect::<Vec<f32>>();

    assert_eq!(y.len(), ref_out.len(), "STFT output element count mismatch");
    let dot: f32 = y.iter().zip(&ref_out).map(|(a, b)| a * b).sum();
    let na: f32 = y.iter().map(|v| v * v).sum::<f32>().sqrt();
    let nb: f32 = ref_out.iter().map(|v| v * v).sum::<f32>().sqrt();
    let cosine = dot / (na * nb + 1e-12);
    let max_abs = y
        .iter()
        .zip(&ref_out)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    eprintln!("STFT import parity: cosine={cosine:.7}, max_abs_diff={max_abs:.3e}");
    assert!(cosine > 0.9999, "cosine {cosine} below parity threshold");
    assert!(
        max_abs < 1e-3,
        "max_abs_diff {max_abs} above parity threshold"
    );
}
