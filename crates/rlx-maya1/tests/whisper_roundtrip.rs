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

//! Maya1 voice-design synthesis → Whisper round-trip. Skips without weights.
//! Needs a Maya1 GGUF (RLX_MAYA1_GGUF or weights/tts/maya1/*.gguf),
//! ORPHEUS_SNAC_PATH, and RLX_WHISPER_DIR.

use std::path::PathBuf;

use rlx_maya1::{Maya1, SAMPLE_RATE};
use rlx_runtime::Device;
use rlx_whisper::{SAMPLE_RATE as WR, WhisperRunner};

const DESC: &str = "Realistic female voice in her 20s with a British accent. Normal pitch, warm timbre, conversational pacing.";
const TEXT: &str = "The quick brown fox jumps over the lazy dog.";

fn gguf_path() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("RLX_MAYA1_GGUF") {
        let p = PathBuf::from(p);
        if p.is_file() {
            return Some(p);
        }
    }
    let dir = PathBuf::from("weights/tts/maya1");
    let pref = dir.join("maya1.Q4_K_M.gguf");
    if pref.is_file() {
        return Some(pref);
    }
    std::fs::read_dir(&dir).ok()?.filter_map(|e| e.ok()).map(|e| e.path()).find(|p| p.extension().is_some_and(|x| x == "gguf"))
}
fn whisper_dir() -> Option<PathBuf> {
    if let Ok(d) = std::env::var("RLX_WHISPER_DIR") {
        return ready(PathBuf::from(d));
    }
    let c = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.cache");
    ["whisper-base.en", "whisper-small.en"].into_iter().find_map(|n| ready(c.join(n)))
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
fn maya1_roundtrip_via_whisper() {
    let (Some(gguf), Some(wd)) = (gguf_path(), whisper_dir()) else {
        eprintln!("skip: set RLX_MAYA1_GGUF + ORPHEUS_SNAC_PATH + RLX_WHISPER_DIR");
        return;
    };
    if std::env::var("ORPHEUS_SNAC_PATH").is_err() {
        eprintln!("skip: set ORPHEUS_SNAC_PATH to the SNAC decoder safetensors");
        return;
    }
    let tts = Maya1::load_on(&gguf, Device::Cpu).expect("load maya1");
    let audio = tts.synthesize(DESC, TEXT).expect("synthesize");
    assert!(rlx_maya1::peak_amplitude(&audio) > 0.03, "audio not audible");

    let n = (audio.len() as u64 * WR as u64 / SAMPLE_RATE as u64).max(1) as usize;
    let pcm: Vec<f32> = (0..n)
        .map(|i| {
            let s = i as f64 * SAMPLE_RATE as f64 / WR as f64;
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
    let hits = refs.iter().filter(|x| heard.iter().any(|h| h == *x || h.contains(x.as_str()))).count();
    let cov = hits as f32 / refs.len() as f32;
    eprintln!("target:  {TEXT}\nwhisper: {transcript}\ncoverage: {cov:.2}");
    assert!(cov >= 0.6, "coverage {cov:.2} too low.\ntarget: {TEXT}\ngot: {transcript}");
}
