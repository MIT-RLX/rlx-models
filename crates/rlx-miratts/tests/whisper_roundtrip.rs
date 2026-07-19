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

//! MiraTTS FastBiCodec decoder → Whisper ASR intelligibility round-trip.
//!
//! The codec decode is bit-exact vs onnxruntime (`codec_parity.rs`); this test
//! additionally proves the **native** rlx decode of a real utterance's acoustic
//! codes yields *intelligible speech* (Whisper hears real words), on the default
//! RLX backend — no ONNX Runtime at runtime. Skips gracefully when the codec
//! weights, fixtures, or a Whisper model are absent. Point Whisper at it with
//! `RLX_WHISPER_DIR` (or drop a model under `.cache/whisper-*.en`).

use std::path::PathBuf;

use rlx_miratts::codec::MiraCodec;
use rlx_runtime::Device;
use rlx_whisper::{SAMPLE_RATE as WHISPER_RATE, WhisperRunner};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn read_i64(p: &PathBuf) -> Vec<i64> {
    std::fs::read(p)
        .unwrap()
        .chunks_exact(8)
        .map(|b| i64::from_le_bytes(b.try_into().unwrap()))
        .collect()
}
fn read_i32(p: &PathBuf) -> Vec<i32> {
    std::fs::read(p)
        .unwrap()
        .chunks_exact(4)
        .map(|b| i32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

fn whisper_dir() -> Option<PathBuf> {
    if let Ok(d) = std::env::var("RLX_WHISPER_DIR") {
        return ready(PathBuf::from(d));
    }
    let cache = root().join(".cache");
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

/// Decode a real utterance's acoustic + global codes with the native rlx codec
/// and confirm Whisper hears intelligible speech.
#[test]
fn codec_decode_is_intelligible_speech() {
    let dir = root().join("weights/tts/miratts/decoders");
    let fix = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    if !dir.join("detokenizer.onnx").is_file() || !fix.join("codec_speech.i64").is_file() {
        eprintln!("skip: detokenizer.onnx / codec fixtures not present");
        return;
    }
    let Some(whisper) = whisper_dir() else {
        eprintln!("skip: set RLX_WHISPER_DIR (or drop a whisper model in .cache)");
        return;
    };

    let speech: Vec<u32> = read_i64(&fix.join("codec_speech.i64"))
        .into_iter()
        .map(|x| x as u32)
        .collect();
    let context: Vec<u32> = read_i32(&fix.join("codec_context.i32"))
        .into_iter()
        .map(|x| x as u32)
        .collect();

    let codec = MiraCodec::load(&dir, Device::Cpu).expect("load codec");
    let wav = codec.decode(&speech, &context).expect("decode");
    let peak = wav.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
    eprintln!(
        "decoded {} samples ({:.2}s), peak {peak:.3}",
        wav.len(),
        wav.len() as f32 / codec.sample_rate() as f32
    );
    assert!(peak > 0.02, "decoded audio not audible (peak {peak:.4})");

    let pcm16k = resample(&wav, codec.sample_rate(), WHISPER_RATE as u32);
    let mut w = WhisperRunner::builder()
        .weights(whisper.join("model.safetensors"))
        .config_path(whisper.join("config.json"))
        .tokenizer_path(whisper.join("tokenizer.json"))
        .device(Device::Cpu)
        .language("en")
        .build()
        .expect("whisper");
    let transcript = w.transcribe_greedy(&pcm16k).expect("transcribe");
    let heard = words(&transcript);
    eprintln!("whisper heard: {transcript:?}  ({} words)", heard.len());
    // Ground-truth text of the fixture utterance is unknown, so assert
    // intelligibility: real speech transcribes to several real words, noise does not.
    assert!(
        heard.len() >= 3,
        "codec output not intelligible speech: {transcript:?}"
    );
}
