// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//! End-to-end whisper round-trip for the **native** (ort-free) Kokoro path:
//! text → RLX graph-split synthesis → Whisper transcription → word coverage.
//!
//! Requires (else the test SKIPS): a Kokoro model dir with the native split
//! bundle under `onnx/rlx-split/` (produce with `scripts/split_kokoro.py`) and a
//! Whisper checkpoint. Point at them with `RLX_KOKORO_DIR` / `RLX_WHISPER_DIR`
//! (or the `.cache/whisper-*` fallbacks). No ort anywhere on this path.
#![cfg(all(feature = "native", feature = "espeak"))]

use std::path::PathBuf;

use rlx_kokoro::{Device, NativeKokoro, SAMPLE_RATE as TTS_RATE};
use rlx_whisper::{SAMPLE_RATE as WHISPER_RATE, WhisperRunner};

const TEXT: &str = "The quick brown fox jumps over the lazy dog.";

fn kokoro_dir() -> Option<PathBuf> {
    if let Ok(d) = std::env::var("RLX_KOKORO_DIR") {
        let p = PathBuf::from(d);
        return has_split(&p).then_some(p);
    }
    [
        PathBuf::from("weights/tts/kokoro-82m"),
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../weights/tts/kokoro-82m"),
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.cache/kokoro-82m"),
    ]
    .into_iter()
    .find(|cand| has_split(cand))
}

fn has_split(model_dir: &std::path::Path) -> bool {
    model_dir.join("onnx/rlx-split/encoder.onnx").is_file()
}

fn whisper_dir() -> Option<PathBuf> {
    if let Ok(d) = std::env::var("RLX_WHISPER_DIR") {
        return whisper_if_ready(PathBuf::from(d));
    }
    let cache = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.cache");
    for name in [
        "whisper-base.en",
        "whisper-small.en",
        "whisper-tiny.en",
        "whisper-tiny",
    ] {
        if let Some(d) = whisper_if_ready(cache.join(name)) {
            return Some(d);
        }
    }
    None
}

fn whisper_if_ready(dir: PathBuf) -> Option<PathBuf> {
    (dir.join("model.safetensors").is_file() && dir.join("tokenizer.json").is_file()).then_some(dir)
}

fn resample_linear(samples: &[f32], from_hz: u32, to_hz: u32) -> Vec<f32> {
    if from_hz == to_hz || samples.is_empty() {
        return samples.to_vec();
    }
    let out_len = (samples.len() as u64 * to_hz as u64 / from_hz as u64).max(1) as usize;
    (0..out_len)
        .map(|i| {
            let src = i as f64 * from_hz as f64 / to_hz as f64;
            let idx = src.floor() as usize;
            let frac = (src - idx as f64) as f32;
            let a = samples[idx.min(samples.len() - 1)];
            let b = samples[(idx + 1).min(samples.len() - 1)];
            a + (b - a) * frac
        })
        .collect()
}

fn normalize_words(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() > 2)
        .map(str::to_string)
        .collect()
}

fn coverage(reference: &str, transcript: &str) -> f32 {
    let refs = normalize_words(reference);
    if refs.is_empty() {
        return 0.0;
    }
    let heard = normalize_words(transcript);
    let hits = refs
        .iter()
        .filter(|w| heard.iter().any(|h| h == *w || h.contains(w.as_str())))
        .count();
    hits as f32 / refs.len() as f32
}

#[test]
fn native_kokoro_text_roundtrip_via_whisper() {
    let Some(kokoro) = kokoro_dir() else {
        eprintln!(
            "skip: no Kokoro native split bundle (set RLX_KOKORO_DIR; run scripts/split_kokoro.py)"
        );
        return;
    };
    let Some(whisper) = whisper_dir() else {
        eprintln!("skip: set RLX_WHISPER_DIR (or fetch .cache/whisper-*)");
        return;
    };

    let tts = NativeKokoro::load(&kokoro, Device::Cpu).expect("load native kokoro");
    let audio = tts
        .generate_from_text(TEXT, "af_heart", 1.0)
        .expect("synthesize");
    assert!(audio.len() > TTS_RATE as usize, "audio too short");
    assert!(
        rlx_kokoro::peak_amplitude(&audio) > 0.05,
        "audio not audible"
    );

    let pcm_16k = resample_linear(&audio, TTS_RATE, WHISPER_RATE as u32);
    let mut runner = WhisperRunner::builder()
        .weights(whisper.join("model.safetensors"))
        .config_path(whisper.join("config.json"))
        .tokenizer_path(whisper.join("tokenizer.json"))
        .device(Device::Cpu)
        .language("en")
        .build()
        .expect("whisper runner");
    let transcript = runner.transcribe_greedy(&pcm_16k).expect("transcribe");

    eprintln!("reference:  {TEXT}");
    eprintln!("whisper:    {transcript}");
    let cov = coverage(TEXT, &transcript);
    eprintln!("coverage:   {cov:.2}");
    assert!(
        cov >= 0.6,
        "native Kokoro whisper coverage {cov:.2} too low.\nref: {TEXT}\ngot: {transcript}"
    );
}
