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

//! `speech_encoder` against the ONNX Runtime reference.
//!
//! The speaker embedding is the conditioning half of ChatterBox voice cloning,
//! and it was wrong in five independent ways at once — a 2-D conv collapsing its
//! height axis, a stale meta length trusted over the computed one, a
//! BatchNormalization taking its shape from that meta, `ceil_mode` ignored so
//! pooling produced the wrong window count, and the CPU pool kernel averaging an
//! overhanging window over its nominal size. None of it crashed. The embedding
//! came out plausible and identified nothing: two different speakers scored
//! cosine 0.908 where the reference says 0.438.
//!
//! So this compares against something external. The fixture holds onnxruntime's
//! output for a probe signal that the test rebuilds, which keeps the comparison
//! honest without needing onnxruntime at test time.

use rlx_chatterbox::native::NativeChatterBox;
use std::path::Path;

const SR: u32 = 24_000;

/// Deterministic, non-degenerate probe: two tones plus an integer-LCG dither.
/// A pure tone is a bad probe here — its activations are near-degenerate and the
/// comparison goes slack.
fn probe_signal(n: usize) -> Vec<f32> {
    let mut seed: u64 = 12_345;
    (0..n)
        .map(|i| {
            seed = (seed.wrapping_mul(1_103_515_245).wrapping_add(12_345)) % (1 << 31);
            let dither = (seed as f64 / (1u64 << 31) as f64) * 0.05 - 0.025;
            let t = i as f64 / SR as f64;
            let two_pi = std::f64::consts::TAU;
            (0.25 * (two_pi * 180.0 * t).sin() + 0.15 * (two_pi * 450.0 * t).sin() + dither) as f32
        })
        .collect()
}

#[derive(serde::Deserialize)]
struct Fixture {
    samples: usize,
    embedding: Vec<f32>,
}

#[test]
fn speaker_embedding_matches_onnxruntime() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace root");
    let dir = root.join("weights/tts/chatterbox");
    if !dir.join("onnx/speech_encoder.onnx").exists() {
        eprintln!("no ChatterBox weights at {} — skipping", dir.display());
        return;
    }
    let f: Fixture = serde_json::from_str(include_str!("fixtures/speech_encoder_reference.json"))
        .expect("parse speech_encoder_reference.json");

    let cb = NativeChatterBox::load(&dir).expect("load ChatterBox");
    let got = cb
        .speaker_embedding(&probe_signal(f.samples), SR)
        .expect("speaker_embedding");

    assert_eq!(got.len(), f.embedding.len(), "embedding width");

    let dot: f64 = got
        .iter()
        .zip(&f.embedding)
        .map(|(a, b)| *a as f64 * *b as f64)
        .sum();
    let na: f64 = got.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = f
        .embedding
        .iter()
        .map(|b| (*b as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    let cos = dot / (na * nb).max(1e-12);
    let max_abs = got
        .iter()
        .zip(&f.embedding)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);

    // Cosine is the property that matters — it is what decides "same speaker" —
    // and it is what collapsed when the encoder was broken (0.908 vs 0.438 on a
    // pair that should not match). The absolute bound catches a drift that
    // happens to preserve direction.
    assert!(
        cos > 0.9999,
        "speaker embedding diverged from the reference: cosine {cos:.8} (max|Δ| {max_abs:.2e})"
    );
    assert!(
        max_abs < 0.05,
        "speaker embedding drifted: max|Δ| {max_abs:.4} (cosine {cos:.8})"
    );
}
