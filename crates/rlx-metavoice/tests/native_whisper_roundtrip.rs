// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0

//! MetaVoice sentence → Whisper ASR round-trip (≥5/6 fox content words).
//! Needs `weights/tts/metavoice` (+ `bria_16k.wav`), `weights/tts/encodec24`,
//! and a Whisper cache.

use std::path::PathBuf;

use rlx_metavoice::{
    DEFAULT_ENCODEC_PATH, DEFAULT_LOCAL_DIR, DEFAULT_REFERENCE, FOX_WORDS, InferOpts, MetaVoice,
    peak_amplitude,
};
use rlx_runtime::Device;
use rlx_whisper::{SAMPLE_RATE as WHISPER_RATE, WhisperRunner};

const TEXT: &str = "The quick brown fox jumps over the lazy dog.";

fn env_dir(var: &str, default: &str) -> Option<PathBuf> {
    let d = std::env::var(var)
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(default));
    d.exists().then_some(d)
}

fn whisper_dir() -> Option<PathBuf> {
    if let Ok(d) = std::env::var("RLX_WHISPER_DIR") {
        let p = PathBuf::from(d);
        if p.join("config.json").exists() {
            return Some(p);
        }
    }
    for name in [
        "whisper-base.en",
        "whisper-small.en",
        "whisper-tiny.en",
        "whisper-tiny",
    ] {
        let p = PathBuf::from(".cache").join(name);
        if p.join("config.json").exists() {
            return Some(p);
        }
    }
    None
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
        .filter(|w| !w.is_empty())
        .map(str::to_string)
        .collect()
}

#[test]
fn infer_opts_default_greedy_long() {
    let o = InferOpts::default();
    assert!(o.greedy, "defaults should be greedy for intelligibility");
    assert!(o.max_new_tokens >= 864);
    assert_eq!(o.seed, 1337);
}

#[test]
fn metavoice_sentence_whisper_roundtrip() {
    let Some(dir) = env_dir("RLX_METAVOICE_DIR", DEFAULT_LOCAL_DIR) else {
        eprintln!("skip: set RLX_METAVOICE_DIR / place weights at {DEFAULT_LOCAL_DIR}");
        return;
    };
    if !dir.join("first_stage.safetensors").is_file() {
        eprintln!(
            "skip: missing first_stage.safetensors under {}",
            dir.display()
        );
        return;
    }
    let enc = PathBuf::from(
        std::env::var("RLX_ENCODEC_PATH").unwrap_or_else(|_| DEFAULT_ENCODEC_PATH.to_string()),
    );
    if !enc.is_file() {
        eprintln!("skip: missing EnCodec at {}", enc.display());
        return;
    }
    let Some(whisper) = whisper_dir() else {
        eprintln!("skip: set RLX_WHISPER_DIR");
        return;
    };
    let reference = dir.join("bria_16k.wav");
    let reference = if reference.is_file() {
        reference
    } else {
        PathBuf::from(DEFAULT_REFERENCE)
    };
    if !reference.is_file() {
        eprintln!("skip: missing speaker reference {}", reference.display());
        return;
    }

    let mv = MetaVoice::open_with_encodec(&dir, &enc, Device::Cpu).expect("open metavoice");
    let opts = InferOpts::default(); // greedy + 864 + seed 1337
    let pcm = mv
        .synthesize(TEXT, Some(reference.as_path()), &opts)
        .expect("synthesize");
    let peak = peak_amplitude(&pcm);
    assert!(peak > 0.05, "audio not audible (peak {peak:.4})");

    let pcm16k = resample(&pcm, mv.sample_rate(), WHISPER_RATE as u32);
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
    let hits = FOX_WORDS
        .iter()
        .filter(|x| heard.iter().any(|h| h == *x || h.contains(*x)))
        .count();
    eprintln!(
        "reference: {TEXT}\nwhisper:   {transcript}\nfox hits:  {hits}/{}  peak={peak:.3}",
        FOX_WORDS.len()
    );
    assert!(
        hits >= 5,
        "fox Whisper {hits}/{} < 5.\nref: {TEXT}\ngot: {transcript}",
        FOX_WORDS.len()
    );
}
