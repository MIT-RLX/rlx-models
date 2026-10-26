// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only

//! An exported VibeVoice-ASR-Streaming
//! (`weights/asr/vibevoice-asr-streaming-7b`), opened by rlx alone,
//! transcribes as the official `streaming_generate` does: each chunk's text
//! the reference's (`tests/fixtures/streaming_reference/jfk_rust_speech.json`,
//! from `scripts/streaming_reference.py`, the acoustic latent at its mean),
//! the transcript those texts trimmed and joined by spaces. The crate's
//! runner (`split_then_encode`, as the reference) is held to them first, then
//! dropped (its decoder in f32 is 28 GB), then the package runs.

#![cfg(feature = "export")]

use std::path::PathBuf;

use rlx_runtime::Device;
use rlx_vibevoice_asr::export::{ExportOptions, export_rlxp};
use rlx_vibevoice_asr::{EncodeMode, StreamingAsr};
use serde_json::{Value, json};

/// A 16-bit mono WAV's samples in [-1, 1).
fn wav(path: &std::path::Path) -> (Vec<f32>, usize) {
    let b = std::fs::read(path).unwrap();
    let rate = u32::from_le_bytes(b[24..28].try_into().unwrap()) as usize;
    let mut i = 12;
    while &b[i..i + 4] != b"data" {
        i += 8 + u32::from_le_bytes(b[i + 4..i + 8].try_into().unwrap()) as usize;
    }
    let n = u32::from_le_bytes(b[i + 4..i + 8].try_into().unwrap()) as usize;
    let pcm = b[i + 8..i + 8 + n]
        .chunks_exact(2)
        .map(|s| f32::from(i16::from_le_bytes([s[0], s[1]])) / 32768.0)
        .collect();
    (pcm, rate)
}

#[test]
fn an_exported_vibevoice_asr_transcribes_as_the_reference_does() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let dir = root.join("../../weights/asr/vibevoice-asr-streaming-7b");
    if !dir.join("config.json").is_file() {
        eprintln!("skip: no VibeVoice-ASR-Streaming checkpoint");
        return;
    }
    rlx_text::install_model_tokenizer();
    let case: Value = serde_json::from_slice(
        &std::fs::read(root.join("tests/fixtures/streaming_reference/jfk_rust_speech.json"))
            .unwrap(),
    )
    .unwrap();
    let (pcm, rate) = wav(&root.join("../../assets/jfk/jfk_rust_speech.wav"));
    assert_eq!(rate, 24_000);
    let texts: Vec<String> = serde_json::from_value(case["chunk_texts"].clone()).unwrap();
    let want = texts
        .iter()
        .map(|t| t.trim())
        .filter(|t| !t.is_empty())
        .collect::<Vec<_>>()
        .join(" ");
    eprintln!("reference: {want:?}");

    {
        let mut runner = StreamingAsr::load(&dir, Device::Cpu).unwrap();
        runner.set_encode_mode(EncodeMode::SplitThenEncode);
        let got = runner.transcribe(&pcm, rate, None, 256).unwrap();
        eprintln!("runner: {got:?}");
        assert_eq!(got, want, "the runner");
    }

    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("vibevoice-asr.rlxp");
    eprintln!(
        "{}",
        export_rlxp(&dir, &out, &ExportOptions::default()).unwrap()
    );
    let mut m = rlx_runtime::model::open_source(out.to_str().unwrap(), json!({})).unwrap();
    let got = m.transcribe(&pcm, rate as u32, &json!({})).unwrap();
    eprintln!("package: {got:?}");
    assert_eq!(got, want, "the package");
}
