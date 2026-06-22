// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// Licensed under GNU GPL v3. See top-level LICENSE.

//! Whisper-validate Pocket TTS output.
//!
//! Generates audio for a known prompt, runs it through Whisper, and prints
//! both transcripts side-by-side plus a coverage score. Use this to tell
//! whether the model is rushing words (low coverage) vs. truncating sentences
//! vs. producing clean output.
//!
//! ```bash
//! RLX_WHISPER_DIR=.cache/whisper-base.en \
//! cargo run -p rlx-pocket-tts --example whisper_check \
//!   --features hf-download --release
//! ```

use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use rlx_pocket_tts::config::PocketTtsConfig;
use rlx_pocket_tts::{GenerationOptions, TtsModel};
use rlx_runtime::Device;
use rlx_whisper::{SAMPLE_RATE as WHISPER_RATE, WhisperRunner};

fn main() -> Result<()> {
    let text = std::env::var("POCKET_TTS_TEXT").unwrap_or_else(|_| {
        "Hello world. I am Kyutai's Pocket TTS, running natively in Rust. \
         I hope you like the way I sound."
            .to_string()
    });
    let voice_name = std::env::var("POCKET_TTS_VOICE").unwrap_or_else(|_| "alba".to_string());
    let seed: u64 = std::env::var("POCKET_TTS_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1729);

    let whisper_dir = resolve_whisper_dir()?;
    eprintln!("whisper:  {}", whisper_dir.display());

    // ── generate ────────────────────────────────────────────────────────────
    let assets = rlx_pocket_tts::download::fetch_default_assets()?;
    let voice_path = rlx_pocket_tts::download::fetch_voice(&voice_name)?;
    let mut cfg = PocketTtsConfig::english();
    if let Ok(s) = std::env::var("POCKET_TTS_TEMP") {
        if let Ok(v) = s.parse() {
            cfg.temperature = v;
        }
    }
    if let Ok(s) = std::env::var("POCKET_TTS_STEPS") {
        if let Ok(v) = s.parse() {
            cfg.lsd_decode_steps = v;
        }
    }
    eprintln!("temp={} steps={}", cfg.temperature, cfg.lsd_decode_steps);
    let model = TtsModel::open_with_config(&assets.weights, &assets.tokenizer, cfg)?;
    let voice = model.load_voice(&voice_path)?;
    let mut opts = GenerationOptions::default();
    opts.seed = seed;

    eprintln!("prompt:   {text:?}");
    eprintln!("voice:    {voice_name}  seed={seed}");
    let t = std::time::Instant::now();
    let audio = model.generate(&text, &voice, opts)?;
    let dt = t.elapsed().as_secs_f32();
    eprintln!(
        "generated {} samples ({:.2}s audio) in {:.2}s — {:.2}× realtime",
        audio.samples.len(),
        audio.duration_secs(),
        dt,
        audio.duration_secs() / dt.max(1e-6),
    );

    // ── transcribe ──────────────────────────────────────────────────────────
    let pcm_16k = resample_linear(
        &audio.samples,
        rlx_pocket_tts::SAMPLE_RATE,
        WHISPER_RATE as u32,
    );
    let mut whisper = WhisperRunner::builder()
        .weights(whisper_dir.join("model.safetensors"))
        .config_path(whisper_dir.join("config.json"))
        .tokenizer_path(whisper_dir.join("tokenizer.json"))
        .device(Device::Cpu)
        .language("en")
        .build()
        .context("build WhisperRunner")?;
    let t = std::time::Instant::now();
    let transcript = whisper
        .transcribe_greedy(&pcm_16k)
        .context("whisper transcribe")?;
    eprintln!("whisper:  {:.2}s", t.elapsed().as_secs_f32());

    let expected = text.trim();
    let got = transcript.trim();
    println!();
    println!("expected : {expected}");
    println!("whisper  : {got}");
    println!();
    print_metrics(expected, got, audio.duration_secs(), &voice_name);

    Ok(())
}

fn resolve_whisper_dir() -> Result<PathBuf> {
    if let Ok(dir) = std::env::var("RLX_WHISPER_DIR") {
        let p = PathBuf::from(dir);
        if p.join("model.safetensors").is_file() {
            return Ok(p);
        }
    }
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    for name in [
        "whisper-base.en",
        "whisper-small.en",
        "whisper-tiny.en",
        "whisper-tiny",
    ] {
        let p = root.join(".cache").join(name);
        if p.join("model.safetensors").is_file() {
            return Ok(p);
        }
    }
    bail!("Whisper weights not found — set RLX_WHISPER_DIR=<dir-with-model.safetensors>")
}

fn resample_linear(samples: &[f32], from_hz: u32, to_hz: u32) -> Vec<f32> {
    if from_hz == to_hz || samples.is_empty() {
        return samples.to_vec();
    }
    let out_len = (samples.len() as u64 * to_hz as u64 / from_hz as u64).max(1) as usize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let src = i as f64 * from_hz as f64 / to_hz as f64;
        let idx = src.floor() as usize;
        let frac = (src - idx as f64) as f32;
        let a = samples[idx.min(samples.len() - 1)];
        let b = samples[(idx + 1).min(samples.len() - 1)];
        out.push(a + (b - a) * frac);
    }
    out
}

fn normalize(s: &str) -> Vec<String> {
    s.chars()
        .map(|c| {
            if c.is_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                ' '
            }
        })
        .collect::<String>()
        .split_whitespace()
        .map(|s| s.to_string())
        .collect()
}

fn print_metrics(expected: &str, got: &str, audio_secs: f32, voice: &str) {
    let exp_words = normalize(expected);
    let got_words = normalize(got);
    use std::collections::HashSet;
    let exp_set: HashSet<_> = exp_words.iter().collect();
    let got_set: HashSet<_> = got_words.iter().collect();
    let overlap = exp_set.intersection(&got_set).count() as f32;
    let coverage = if exp_set.is_empty() {
        0.0
    } else {
        overlap / exp_set.len() as f32
    };

    // Expected speech rate: ≈ 2.5 words per second at a natural English pace.
    let words_per_sec = if audio_secs > 0.0 {
        exp_words.len() as f32 / audio_secs
    } else {
        0.0
    };
    let typical_wps = 2.5_f32;
    let speed_ratio = words_per_sec / typical_wps;

    println!("--- metrics ---");
    println!("voice            : {voice}");
    println!("expected words   : {}", exp_words.len());
    println!("transcribed words: {}", got_words.len());
    println!(
        "word coverage    : {:.0}% ({} of {} expected words appear in Whisper output)",
        coverage * 100.0,
        overlap as usize,
        exp_words.len()
    );
    println!("audio duration   : {audio_secs:.2}s");
    println!(
        "speech rate      : {words_per_sec:.2} expected_words/sec  ({:.2}× normal ~2.5 wps)",
        speed_ratio
    );
    if speed_ratio > 1.3 {
        println!("→ TOO FAST: model is packing more words into less audio than natural");
    } else if speed_ratio < 0.7 {
        println!("→ too slow / dragging");
    }
    if coverage < 0.5 {
        println!("→ LOW COVERAGE: Whisper is not finding most expected words");
    }
}
