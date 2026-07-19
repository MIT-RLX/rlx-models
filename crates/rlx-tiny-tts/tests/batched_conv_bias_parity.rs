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

//! Regression: importing a BATCH-2 ONNX `Conv`-with-bias matches onnxruntime on
//! BOTH batch elements. Guards the `rlx-onnx-import` conv-bias broadcast bug
//! (conv_pool.rs): the per-channel bias `[C]` was reshaped with the activation's
//! actual batch as its leading dim (`[N,C,1]`), so for N>1 batch elements ≥1 read
//! `C·N > C` elements out of a `C`-element buffer → garbage bias on every batch
//! but the first. Silent on the batch-1 inference path (`N=1` is a no-op) but it
//! corrupted every batched conv — e.g. the Supertonic CFG (batch-2) vector
//! estimator, whose guidance cosine was 0.26 until the bias reshaped from 1.
//!
//! Fixtures (`tests/fixtures/batched_conv_bias/`) are a tiny N=2 depthwise conv
//! (L_in≠L_out, large bias) and its onnxruntime reference, from the co-located
//! `gen_batched_conv_bias.py`. No Python at test time — the reference is committed.

use std::path::PathBuf;

use rlx_runtime::{DType, Device};

fn read_f32(path: &std::path::Path) -> Vec<f32> {
    std::fs::read(path)
        .unwrap_or_default()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|v| v * v).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|v| v * v).sum::<f32>().sqrt();
    dot / (na * nb + 1e-12)
}

#[test]
fn onnx_batched_conv_bias_imports_with_onnxruntime_parity() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/batched_conv_bias");
    let onnx = dir.join("batched_conv_bias_fixture.onnx");
    if !onnx.is_file() {
        eprintln!("skip: missing {}", onnx.display());
        return;
    }
    let x_bytes = std::fs::read(dir.join("x.f32")).expect("x.f32");
    let y_ref = read_f32(&dir.join("y_ref.f32"));
    assert!(!y_ref.is_empty(), "empty reference");

    // Fixture: x [2,8,20] -> Conv(depthwise,k5) -> y [2,8,16]. Fully static; the
    // length arg is just a fallback for any dynamic dim (there are none here).
    let mut g =
        rlx_tiny_tts::model::compile_graph(&dir, "batched_conv_bias_fixture", Device::Cpu, 20)
            .expect("import + compile batched conv+bias");
    let out = g.run_typed(&[("x", &x_bytes, DType::F32)]);
    let (y_bytes, dt) = out.into_iter().next().expect("conv output");
    assert_eq!(dt, DType::F32);
    let y = read_f32_bytes(&y_bytes);
    assert_eq!(y.len(), y_ref.len(), "output element count mismatch");

    // Whole-tensor parity (fails if either batch is wrong)...
    let cos = cosine(&y, &y_ref);
    let max_abs = y
        .iter()
        .zip(&y_ref)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    // ...plus an explicit PER-BATCH check so a regression names the batch: the
    // old bug left batch 0 bit-exact and only corrupted batch 1.
    let per = y_ref.len() / 2;
    let cos_b0 = cosine(&y[..per], &y_ref[..per]);
    let cos_b1 = cosine(&y[per..], &y_ref[per..]);
    eprintln!(
        "batched conv+bias parity: cosine={cos:.6} max_abs={max_abs:.3e} (b0={cos_b0:.6} b1={cos_b1:.6})"
    );
    assert!(cos_b0 > 0.9999, "batch 0 cosine {cos_b0} below parity");
    assert!(
        cos_b1 > 0.9999,
        "batch 1 cosine {cos_b1} below parity — conv bias broadcast regressed (OOB per-batch bias)"
    );
    assert!(
        max_abs < 1e-4,
        "max_abs_diff {max_abs} above parity threshold"
    );
}

fn read_f32_bytes(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}
