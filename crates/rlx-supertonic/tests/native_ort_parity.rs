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

//! Native-RLX vs ONNX-Runtime parity for the full Supertonic-3 pipeline. Both
//! paths sample the same seeded noise and run the same four subgraphs, so — with
//! the importer's per-graph bit-exactness — the end-to-end waveforms must match.
//! Requires the `onnx` feature (builds the reference ORT sessions) + the weights.

#![cfg(feature = "onnx")]

use std::path::PathBuf;

use rlx_runtime::Device;
use rlx_supertonic::{InferOpts, Supertonic, Voice};

const TEXT: &str = "Hello from Supertonic, running natively on RLX.";

fn supertonic_dir() -> Option<PathBuf> {
    if let Ok(d) = std::env::var("RLX_SUPERTONIC_DIR") {
        let p = PathBuf::from(d);
        if p.join("onnx/tts.json").is_file() {
            return Some(p);
        }
    }
    // repo-root weights, relative to this crate.
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../weights/tts/supertonic-3");
    p.join("onnx/tts.json").is_file().then_some(p)
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let dot: f32 = a[..n].iter().zip(&b[..n]).map(|(x, y)| x * y).sum();
    let na: f32 = a[..n].iter().map(|v| v * v).sum::<f32>().sqrt();
    let nb: f32 = b[..n].iter().map(|v| v * v).sum::<f32>().sqrt();
    dot / (na * nb + 1e-12)
}

#[test]
fn native_matches_onnxruntime_end_to_end() {
    let Some(dir) = supertonic_dir() else {
        eprintln!("skip: set RLX_SUPERTONIC_DIR (or place weights/tts/supertonic-3)");
        return;
    };
    let tts = Supertonic::load_on(&dir, Device::Cpu).expect("load supertonic");
    let voice = Voice::load(&dir.join("voice_styles/F1.json")).expect("voice F1");
    // Fewer ODE steps keeps the test fast; parity holds at any step count.
    let opts = InferOpts {
        total_step: 4,
        speed: 1.05,
        seed: 42,
    };

    let native = tts
        .synthesize(TEXT, "en", &voice, &opts)
        .expect("native synthesize");
    let reference = tts
        .synthesize_ort(TEXT, "en", &voice, &opts)
        .expect("ort synthesize");

    assert!(!native.is_empty(), "native produced no audio");
    assert!(
        rlx_supertonic::peak_amplitude(&native) > 0.01,
        "native audio not audible (peak={:.2e})",
        rlx_supertonic::peak_amplitude(&native)
    );
    let len_ratio = native.len() as f32 / reference.len().max(1) as f32;
    let cos = cosine(&native, &reference);
    let max_abs = native
        .iter()
        .zip(&reference)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!(
        "native vs ort: samples {} vs {} (ratio {len_ratio:.4}), cosine={cos:.6}, max_abs={max_abs:.3e}",
        native.len(),
        reference.len()
    );
    assert!(
        (len_ratio - 1.0).abs() < 0.02,
        "sample-count mismatch (ratio {len_ratio})"
    );
    assert!(
        cos > 0.999,
        "native/ort waveform cosine {cos} below parity threshold"
    );
}

#[test]
fn which_subgraph_diverges() {
    let Some(dir) = supertonic_dir() else {
        eprintln!("skip: no supertonic weights");
        return;
    };
    let tts = Supertonic::load_on(&dir, Device::Cpu).expect("load supertonic");
    let voice = Voice::load(&dir.join("voice_styles/F1.json")).expect("voice F1");
    let opts = InferOpts {
        total_step: 4,
        speed: 1.05,
        seed: 42,
    };
    tts.debug_subgraph_parity(TEXT, "en", &voice, &opts)
        .expect("parity probe");
}
