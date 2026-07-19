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

//! F5-TTS **native RLX** (no ONNX Runtime) voice-clone synthesis → Whisper ASR.
//! The 3 F5 graphs are imported + compiled + run through rlx-tiny-tts. Needs
//! model weights (RLX_F5TTS_DIR or weights/tts/f5tts) + Whisper; skips otherwise.
//! Slow (NFE denoising over a 664 MB DiT on CPU).

use std::path::PathBuf;

use rlx_f5tts::{F5Native, InferOpts};
use rlx_runtime::Device;
use rlx_whisper::{SAMPLE_RATE as WR, WhisperRunner};

const REF_TEXT: &str = "Hello from Kokoro. This is a test of speech synthesis in Rust.";
const TEXT: &str = "The quick brown fox jumps over the lazy dog.";

fn model_dir() -> Option<PathBuf> {
    if let Ok(d) = std::env::var("RLX_F5TTS_DIR") {
        let p = PathBuf::from(d);
        if p.join("F5_Transformer.onnx").is_file() {
            return Some(p);
        }
    }
    let p = PathBuf::from("weights/tts/f5tts");
    p.join("F5_Transformer.onnx").is_file().then_some(p)
}
fn whisper_dir() -> Option<PathBuf> {
    if let Ok(d) = std::env::var("RLX_WHISPER_DIR") {
        return ready(PathBuf::from(d));
    }
    let c = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.cache");
    ["whisper-base.en", "whisper-tiny.en", "whisper-tiny"]
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
fn f5tts_native_clone_roundtrip_via_whisper() {
    let (Some(dir), Some(wd)) = (model_dir(), whisper_dir()) else {
        eprintln!("skip: set RLX_F5TTS_DIR + RLX_WHISPER_DIR");
        return;
    };
    let tts = F5Native::load_on(&dir, Device::Cpu).expect("load f5tts native");
    let audio = tts
        .synthesize(TEXT, &read_prompt(), REF_TEXT, &InferOpts::default())
        .expect("native synthesize");
    let peak = audio.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
    eprintln!(
        "native f5: {} samples ({:.2}s), peak {peak:.3}",
        audio.len(),
        audio.len() as f32 / tts.sample_rate() as f32
    );
    assert!(peak > 0.02, "native audio not audible (peak {peak:.4})");

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
    let cov = hits as f32 / refs.len().max(1) as f32;
    eprintln!("target:  {TEXT}\nwhisper: {transcript}\ncoverage: {cov:.2}");
    assert!(
        cov >= 0.5,
        "native coverage {cov:.2} too low.\ntarget: {TEXT}\ngot: {transcript}"
    );
}
