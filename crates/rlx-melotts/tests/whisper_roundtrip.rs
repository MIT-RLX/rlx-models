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

//! MeloTTS synthesis → Whisper round-trip. Skips without weights.
//! Set RLX_MELOTTS_DIR (or weights/tiny-tts-rlx) + RLX_WHISPER_DIR.

use std::path::PathBuf;

use rlx_melotts::{InferOpts, MeloTts};
use rlx_runtime::Device;
use rlx_whisper::{SAMPLE_RATE as WR, WhisperRunner};

const TEXT: &str = "The quick brown fox jumps over the lazy dog.";

fn model_dir() -> Option<PathBuf> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    for cand in [
        std::env::var("RLX_MELOTTS_DIR").ok().map(PathBuf::from),
        std::env::var("RLX_TINY_TTS_DIR").ok().map(PathBuf::from),
        Some(root.join("weights/tts/melotts")),
        Some(root.join("weights/tts/tiny-tts-rlx")),
        Some(root.join("weights/tiny-tts-rlx")),
    ]
    .into_iter()
    .flatten()
    {
        if cand.join("config.json").is_file() && cand.join("onnx/decoder.onnx").is_file() {
            return Some(cand);
        }
    }
    None
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
fn words(s: &str) -> Vec<String> {
    s.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() > 2)
        .map(str::to_string)
        .collect()
}

#[test]
fn melotts_roundtrip_via_whisper() {
    let (Some(dir), Some(wd)) = (model_dir(), whisper_dir()) else {
        eprintln!("skip: set RLX_MELOTTS_DIR + RLX_WHISPER_DIR");
        return;
    };
    let tts = MeloTts::load(&dir).expect("load melotts");
    let opts = InferOpts::from_config(tts.config());
    let wav = tts.synthesize(TEXT, &opts).expect("synthesize");
    assert!(
        rlx_melotts::peak_amplitude(&wav.samples) > 0.05,
        "audio not audible"
    );

    let n = (wav.samples.len() as u64 * WR as u64 / wav.sample_rate as u64).max(1) as usize;
    let pcm: Vec<f32> = (0..n)
        .map(|i| {
            let s = i as f64 * wav.sample_rate as f64 / WR as f64;
            let idx = s.floor() as usize;
            let f = (s - idx as f64) as f32;
            let a = wav.samples[idx.min(wav.samples.len() - 1)];
            let b = wav.samples[(idx + 1).min(wav.samples.len() - 1)];
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
    assert!(
        cov >= 0.6,
        "coverage {cov:.2} too low.\ntarget: {TEXT}\ngot: {transcript}"
    );
}
