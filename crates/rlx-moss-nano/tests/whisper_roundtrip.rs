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

//! MOSS-TTS-Nano synthesis (builtin voice) → Whisper round-trip. Skips without
//! weights. Set RLX_MOSS_NANO_DIR (or weights/tts/moss-nano) + RLX_WHISPER_DIR.

#![cfg(feature = "onnx")]

use std::path::PathBuf;

use rlx_moss_nano::{MossNano, SynthOpts};
use rlx_runtime::Device;
use rlx_whisper::{SAMPLE_RATE as WR, WhisperRunner};

const TEXT: &str = "The quick brown fox jumps over the lazy dog.";

fn model_dir() -> Option<PathBuf> {
    if let Ok(d) = std::env::var("RLX_MOSS_NANO_DIR") {
        let p = PathBuf::from(d);
        if p.join("tokenizer.json").is_file() {
            return Some(p);
        }
    }
    let p = PathBuf::from("weights/tts/moss-nano");
    (p.join("tokenizer.json").is_file() && p.join("moss_tts_prefill.onnx").is_file()).then_some(p)
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
fn moss_nano_roundtrip_via_whisper() {
    let (Some(dir), Some(wd)) = (model_dir(), whisper_dir()) else {
        eprintln!("skip: set RLX_MOSS_NANO_DIR + RLX_WHISPER_DIR");
        return;
    };
    let tts = MossNano::load_on(&dir, Device::Cpu).expect("load moss-nano");
    let voice = tts
        .voice_names()
        .into_iter()
        .find(|v| v == "Trump")
        .unwrap_or_else(|| tts.voice_names()[0].clone());
    let audio = tts
        .synthesize(TEXT, &voice, &SynthOpts::default())
        .expect("synthesize");
    assert!(
        rlx_moss_nano::peak_amplitude(&audio) > 0.05,
        "audio not audible"
    );

    // interleaved stereo @ 48 kHz → mono 16 kHz
    let ch = tts.channels() as usize;
    let frames = audio.len() / ch;
    let mono: Vec<f32> = (0..frames)
        .map(|i| (0..ch).map(|c| audio[i * ch + c]).sum::<f32>() / ch as f32)
        .collect();
    let n = (frames as u64 * WR as u64 / tts.sample_rate() as u64).max(1) as usize;
    let pcm: Vec<f32> = (0..n)
        .map(|i| {
            let s = i as f64 * tts.sample_rate() as f64 / WR as f64;
            let idx = s.floor() as usize;
            let f = (s - idx as f64) as f32;
            let a = mono[idx.min(mono.len() - 1)];
            let b = mono[(idx + 1).min(mono.len() - 1)];
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
