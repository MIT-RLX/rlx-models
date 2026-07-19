// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//! Native-RLX vs ONNX-Runtime end-to-end parity for LuxTTS. The native path
//! (`synthesize`) replaces the single `text_encoder` graph with Rust token
//! concat+pad → native `encoder_body` → Rust length regulator; both paths then
//! share the same CFM loop + vocoder + ISTFT and the SAME seeded noise, so — with
//! per-graph bit-exactness — the waveforms must match. Requires the `onnx`
//! feature (builds the reference ort sessions) + weights incl. `encoder_body.onnx`.
#![cfg(all(feature = "onnx", feature = "espeak"))]

use std::path::PathBuf;

use rlx_luxtts::{InferOpts, LuxTts};
use rlx_runtime::Device;

const PROMPT_TEXT: &str = "Hello from Kokoro. This is a test of speech synthesis in Rust.";
const TEXT: &str = "The quick brown fox jumps over the lazy dog near the river bank.";

fn model_dir() -> Option<PathBuf> {
    if let Ok(d) = std::env::var("RLX_LUXTTS_DIR") {
        let p = PathBuf::from(d);
        if p.join("tokens.txt").is_file() {
            return Some(p);
        }
    }
    let p = PathBuf::from("weights/tts/luxtts");
    (p.join("tokens.txt").is_file()
        && (p.join("encoder_body.onnx").is_file() || p.join("onnx/encoder_body.onnx").is_file()))
    .then_some(p)
}

fn read_prompt() -> Vec<f32> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/prompt.wav");
    let mut r = hound::WavReader::open(p).expect("prompt.wav");
    let max = (1i64 << (r.spec().bits_per_sample - 1)) as f32;
    r.samples::<i32>()
        .map(|s| s.unwrap() as f32 / max)
        .collect()
}

#[test]
fn native_matches_onnxruntime_end_to_end() {
    let Some(dir) = model_dir() else {
        eprintln!(
            "skip: set RLX_LUXTTS_DIR (needs encoder_body.onnx — see scripts/export_encoder_body.py)"
        );
        return;
    };
    let tts = LuxTts::load_on(&dir, Device::Cpu).expect("load luxtts");
    let prompt = read_prompt();
    let opts = InferOpts::default();
    let native = tts
        .synthesize(TEXT, &prompt, PROMPT_TEXT, &opts)
        .expect("native synthesize");
    let reference = tts
        .synthesize_ort(TEXT, &prompt, PROMPT_TEXT, &opts)
        .expect("ort synthesize");

    assert_eq!(
        native.len(),
        reference.len(),
        "native/ort sample count mismatch"
    );
    let n = native.len().min(reference.len());
    let (a, b) = (&native[..n], &reference[..n]);
    let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
    let na: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let cos = dot / (na * nb + 1e-12);
    let maxabs = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    eprintln!("luxtts native-vs-ort: cos={cos:.7} max_abs={maxabs:.6} n={n}");
    assert!(cos > 0.999, "native-vs-ort cosine {cos} too low");
}
