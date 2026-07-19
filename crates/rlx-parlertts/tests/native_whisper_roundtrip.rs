// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0

//! Native (ort-free) Parler-TTS synthesis → resample → Whisper ASR round-trip
//! (intelligibility bar). Needs weights under `weights/tts/parlertts` +
//! `weights/tts/parler-dac`, and a Whisper cache (or `RLX_WHISPER_DIR`).

use std::path::PathBuf;

use rlx_parlertts::{
    DEFAULT_DAC_DIR, DEFAULT_DESCRIPTION, DEFAULT_LOCAL_DIR, InferOpts, NativeParler,
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
        .filter(|w| w.len() > 2)
        .map(str::to_string)
        .collect()
}

#[test]
fn parler_text_roundtrip_via_whisper() {
    let Some(dir) = env_dir("RLX_PARLER_DIR", DEFAULT_LOCAL_DIR) else {
        eprintln!("skip: set RLX_PARLER_DIR / place weights at {DEFAULT_LOCAL_DIR}");
        return;
    };
    if !dir.join("onnx/decoder.onnx").is_file() {
        eprintln!("skip: missing onnx/decoder.onnx under {}", dir.display());
        return;
    }
    let Some(dac) = env_dir("RLX_DAC_DIR", DEFAULT_DAC_DIR) else {
        eprintln!("skip: set RLX_DAC_DIR / place weights at {DEFAULT_DAC_DIR}");
        return;
    };
    let Some(whisper) = whisper_dir() else {
        eprintln!("skip: set RLX_WHISPER_DIR");
        return;
    };

    let p = NativeParler::open(&dir, &dac, Device::Cpu).expect("open parler");
    let opts = InferOpts {
        max_steps: 172,
        greedy: true,
        ..Default::default()
    };
    let pcm = p
        .synthesize(TEXT, DEFAULT_DESCRIPTION, &opts)
        .expect("synthesize");
    let peak = pcm.iter().fold(0f32, |m, &v| m.max(v.abs()));
    assert!(peak > 0.02, "audio not audible (peak {peak:.4})");

    let pcm16k = resample(&pcm, p.sample_rate(), WHISPER_RATE as u32);
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
    let cov = hits as f32 / refs.len().max(1) as f32;
    eprintln!("reference: {TEXT}\nwhisper:   {transcript}\ncoverage:  {cov:.2}");
    assert!(
        cov >= 0.5,
        "coverage {cov:.2} too low.\nref: {TEXT}\ngot: {transcript}"
    );
}
