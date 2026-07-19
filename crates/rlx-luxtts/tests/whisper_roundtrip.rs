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

//! LuxTTS voice-clone synthesis → Whisper ASR round-trip (intelligibility).
//! Needs the model weights (RLX_LUXTTS_DIR or weights/tts/luxtts) + Whisper
//! (RLX_WHISPER_DIR or .cache/whisper-*.en); skips otherwise. The prompt is a
//! committed 2.5 s clip fixture.

// Runs the NATIVE RLX path (no ort); needs `encoder_body.onnx` (see
// scripts/export_encoder_body.py) alongside the other subgraphs.
#![cfg(feature = "espeak")]

use std::path::PathBuf;

use rlx_luxtts::{InferOpts, LuxTts};
use rlx_runtime::Device;
use rlx_whisper::{SAMPLE_RATE as WR, WhisperRunner};

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
    (p.join("tokens.txt").is_file() && p.join("onnx/vocoder_spec.onnx").is_file()).then_some(p)
}

fn whisper_dir() -> Option<PathBuf> {
    if let Ok(d) = std::env::var("RLX_WHISPER_DIR") {
        return ready(PathBuf::from(d));
    }
    let c = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.cache");
    ["whisper-base.en", "whisper-small.en"]
        .into_iter()
        .find_map(|n| ready(c.join(n)))
}
fn ready(d: PathBuf) -> Option<PathBuf> {
    (d.join("model.safetensors").is_file() && d.join("tokenizer.json").is_file()).then_some(d)
}

fn read_prompt() -> Vec<f32> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/prompt.wav");
    let mut r = hound::WavReader::open(p).expect("prompt.wav");
    let max = (1i64 << (r.spec().bits_per_sample - 1)) as f32;
    r.samples::<i32>()
        .map(|s| s.unwrap() as f32 / max)
        .collect()
}

fn words(s: &str) -> Vec<String> {
    s.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() > 2)
        .map(str::to_string)
        .collect()
}

#[test]
fn luxtts_clone_roundtrip_via_whisper() {
    let (Some(dir), Some(wd)) = (model_dir(), whisper_dir()) else {
        eprintln!("skip: set RLX_LUXTTS_DIR + RLX_WHISPER_DIR");
        return;
    };
    let tts = LuxTts::load_on(&dir, Device::Cpu).expect("load luxtts");
    let prompt = read_prompt();
    let audio = tts
        .synthesize(TEXT, &prompt, PROMPT_TEXT, &InferOpts::default())
        .expect("synthesize");
    assert!(
        rlx_luxtts::peak_amplitude(&audio) > 0.05,
        "audio not audible"
    );

    let n = (audio.len() as u64 * WR as u64 / tts.sample_rate() as u64).max(1) as usize;
    let pcm: Vec<f32> = (0..n)
        .map(|i| {
            let s = i as f64 * tts.sample_rate() as f64 / WR as f64;
            let idx = s.floor() as usize;
            let f = (s - idx as f64) as f32;
            let a = audio[idx.min(audio.len() - 1)];
            let b = audio[(idx + 1).min(audio.len() - 1)];
            a + (b - a) * f
        })
        .collect();
    let mut w = WhisperRunner::builder()
        .weights(wd.join("model.safetensors"))
        .config_path(wd.join("config.json"))
        .tokenizer_path(wd.join("tokenizer.json"))
        .device(Device::Cpu)
        .language("en")
        .build()
        .expect("whisper");
    let transcript = w.transcribe_greedy(&pcm).expect("transcribe");

    let (refs, heard) = (words(TEXT), words(&transcript));
    let hits = refs
        .iter()
        .filter(|x| heard.iter().any(|h| h == *x || h.contains(x.as_str())))
        .count();
    let cov = hits as f32 / refs.len() as f32;
    eprintln!("target:  {TEXT}\nwhisper: {transcript}\ncoverage: {cov:.2}");
    // espeak (vs the reference piper_phonemize) shifts a few vowels, so we accept
    // ≥0.6 word coverage rather than exact parity.
    assert!(
        cov >= 0.6,
        "coverage {cov:.2} too low.\ntarget: {TEXT}\ngot: {transcript}"
    );
}
