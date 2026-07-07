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

//! Kokoro synthesis → resample → Whisper ASR round-trip (intelligibility check).
//!
//! Skips gracefully when the model or Whisper weights are not present. Point at
//! them with `RLX_KOKORO_DIR` and `RLX_WHISPER_DIR` (or `.cache/whisper-*.en`).

#![cfg(all(feature = "onnx", feature = "espeak"))]

use std::path::PathBuf;

use rlx_kokoro::{Kokoro, SAMPLE_RATE as TTS_RATE};
use rlx_runtime::Device;
use rlx_whisper::{SAMPLE_RATE as WHISPER_RATE, WhisperRunner};

const TEXT: &str = "The quick brown fox jumps over the lazy dog near the river bank.";

fn kokoro_dir() -> Option<PathBuf> {
    if let Ok(d) = std::env::var("RLX_KOKORO_DIR") {
        let p = PathBuf::from(d);
        if p.join("tokenizer.json").is_file() {
            return Some(p);
        }
    }
    for cand in [
        PathBuf::from("weights/tts/kokoro-82m"),
        PathBuf::from(".cache/kokoro-82m"),
        dirs_home().join(".cache/rlx/kokoro-82m"),
    ] {
        if cand.join("tokenizer.json").is_file() {
            return Some(cand);
        }
    }
    None
}

fn dirs_home() -> PathBuf {
    std::env::var("HOME").map(PathBuf::from).unwrap_or_default()
}

fn whisper_dir() -> Option<PathBuf> {
    if let Ok(d) = std::env::var("RLX_WHISPER_DIR") {
        return whisper_if_ready(PathBuf::from(d));
    }
    let cache = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.cache");
    for name in ["whisper-base.en", "whisper-small.en", "whisper-tiny.en"] {
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
fn kokoro_text_roundtrip_via_whisper() {
    let Some(kokoro) = kokoro_dir() else {
        eprintln!("skip: set RLX_KOKORO_DIR (or place .cache/kokoro-82m)");
        return;
    };
    let Some(whisper) = whisper_dir() else {
        eprintln!("skip: set RLX_WHISPER_DIR (or fetch .cache/whisper-base.en)");
        return;
    };

    let tts = Kokoro::load_on(&kokoro, "model.onnx", Device::Cpu).expect("load kokoro");
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
        "Whisper transcript missed too much of the reference (coverage {cov:.2}).\nref: {TEXT}\ngot: {transcript}"
    );
}
