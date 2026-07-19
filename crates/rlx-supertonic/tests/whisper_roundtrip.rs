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

//! Supertonic-3 synthesis → resample → Whisper ASR round-trip (intelligibility).
//! Skips gracefully when weights are absent. Point at them with
//! `RLX_SUPERTONIC_DIR` and `RLX_WHISPER_DIR` (or `.cache/whisper-*.en`).
//!
//! Runs on the default (native RLX) synthesis path — no ONNX Runtime needed.

use std::path::PathBuf;

use rlx_runtime::Device;
use rlx_supertonic::{InferOpts, Supertonic, Voice};
use rlx_whisper::{SAMPLE_RATE as WHISPER_RATE, WhisperRunner};

const TEXT: &str = "The quick brown fox jumps over the lazy dog near the river bank.";

fn supertonic_dir() -> Option<PathBuf> {
    if let Ok(d) = std::env::var("RLX_SUPERTONIC_DIR") {
        let p = PathBuf::from(d);
        if p.join("onnx/tts.json").is_file() {
            return Some(p);
        }
    }
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../weights/tts/supertonic-3");
    p.join("onnx/tts.json").is_file().then_some(p)
}

fn whisper_dir() -> Option<PathBuf> {
    if let Ok(d) = std::env::var("RLX_WHISPER_DIR") {
        return ready(PathBuf::from(d));
    }
    let cache = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.cache");
    [
        "whisper-base.en",
        "whisper-small.en",
        "whisper-tiny.en",
        "whisper-tiny",
    ]
    .into_iter()
    .find_map(|n| ready(cache.join(n)))
}

fn ready(d: PathBuf) -> Option<PathBuf> {
    (d.join("model.safetensors").is_file() && d.join("tokenizer.json").is_file()).then_some(d)
}

fn resample(x: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == to || x.is_empty() {
        return x.to_vec();
    }
    let n = (x.len() as u64 * to as u64 / from as u64).max(1) as usize;
    (0..n)
        .map(|i| {
            let s = i as f64 * from as f64 / to as f64;
            let idx = s.floor() as usize;
            let f = (s - idx as f64) as f32;
            let a = x[idx.min(x.len() - 1)];
            let b = x[(idx + 1).min(x.len() - 1)];
            a + (b - a) * f
        })
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
fn supertonic_text_roundtrip_via_whisper() {
    let Some(dir) = supertonic_dir() else {
        eprintln!("skip: set RLX_SUPERTONIC_DIR");
        return;
    };
    let Some(whisper) = whisper_dir() else {
        eprintln!("skip: set RLX_WHISPER_DIR");
        return;
    };

    let tts = Supertonic::load_on(&dir, Device::Cpu).expect("load supertonic");
    let voice = Voice::load(&dir.join("voice_styles/F1.json")).expect("voice");
    let audio = tts
        .synthesize(TEXT, "en", &voice, &InferOpts::default())
        .expect("synthesize");
    assert!(
        rlx_supertonic::peak_amplitude(&audio) > 0.05,
        "audio not audible"
    );

    let pcm16k = resample(&audio, tts.sample_rate(), WHISPER_RATE as u32);
    let mut w = WhisperRunner::builder()
        .weights(whisper.join("model.safetensors"))
        .config_path(whisper.join("config.json"))
        .tokenizer_path(whisper.join("tokenizer.json"))
        .device(Device::Cpu)
        .language("en")
        .build()
        .expect("whisper");
    let transcript = w.transcribe_greedy(&pcm16k).expect("transcribe");

    let refs = words(TEXT);
    let heard = words(&transcript);
    let hits = refs
        .iter()
        .filter(|x| heard.iter().any(|h| h == *x || h.contains(x.as_str())))
        .count();
    let cov = hits as f32 / refs.len() as f32;
    eprintln!("reference: {TEXT}\nwhisper:   {transcript}\ncoverage:  {cov:.2}");
    assert!(
        cov >= 0.6,
        "coverage {cov:.2} too low.\nref: {TEXT}\ngot: {transcript}"
    );
}
