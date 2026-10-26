// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0

//! End to end: synthesize the fox pangram with a built-in voice, transcribe
//! it with Whisper, and require the content words back.
//!
//! Skips without the model (`just fetch-kittentts2`) or a Whisper checkpoint
//! (`.cache/whisper-base.en`, or `RLX_WHISPER_DIR`). Device:
//! `RLX_KITTENTTS2_TEST_DEVICE` (default cpu).

use std::path::PathBuf;

use rlx_kittentts2::pipeline::{GenerateOpts, KittenTts2, resolve_s3gen};
use rlx_kittentts2::{Device, SAMPLE_RATE, parse_device};
use rlx_whisper::{SAMPLE_RATE as WHISPER_RATE, WhisperRunner};

const TEXT: &str = "The quick brown fox jumps over the lazy dog.";
const WORDS: [&str; 6] = ["quick", "brown", "fox", "jumps", "lazy", "dog"];

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn whisper_dir() -> Option<PathBuf> {
    let dirs = match rlx_ir::env::var_result("RLX_WHISPER_DIR") {
        Ok(d) => vec![PathBuf::from(d)],
        Err(_) => ["whisper-base.en", "whisper-small.en", "whisper-tiny.en"]
            .iter()
            .map(|n| root().join(".cache").join(n))
            .collect(),
    };
    dirs.into_iter()
        .find(|d| d.join("model.safetensors").is_file() && d.join("tokenizer.json").is_file())
}

#[test]
fn fox_pangram_survives_a_whisper_round_trip() {
    let repo = rlx_ir::env::var_result("RLX_KITTENTTS2_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| root().join("weights/tts/kitten-tts-2"));
    if !repo.join("config.json").is_file() {
        eprintln!("skip: no KittenTTS 2 at {}", repo.display());
        return;
    }
    let Some(whisper) = whisper_dir() else {
        eprintln!("skip: no Whisper checkpoint");
        return;
    };
    let Ok(s3gen) = resolve_s3gen(&repo) else {
        eprintln!("skip: no s3gen_meanflow.safetensors");
        return;
    };
    let device = rlx_ir::env::var_result("RLX_KITTENTTS2_TEST_DEVICE")
        .map(|d| parse_device(&d).expect("device"))
        .unwrap_or(Device::Cpu);

    let mut tts = KittenTts2::open(&repo, &s3gen, device).expect("open");
    let opts = GenerateOpts {
        voice: Some("Bruno".into()),
        seed: 1,
        ..Default::default()
    };
    let (wav, chunks) = tts.generate(TEXT, &opts, |_| {}).expect("generate");
    assert_eq!(chunks.len(), 1);
    let secs = wav.len() as f32 / SAMPLE_RATE as f32;
    assert!(
        (1.0..12.0).contains(&secs),
        "{secs:.2}s of audio for one sentence"
    );

    let pcm = rlx_core::audio::resample_linear(&wav, SAMPLE_RATE, WHISPER_RATE as u32);
    let mut asr = WhisperRunner::builder()
        .weights(whisper.join("model.safetensors"))
        .config_path(whisper.join("config.json"))
        .tokenizer_path(whisper.join("tokenizer.json"))
        .device(Device::Cpu)
        .language("en")
        .build()
        .expect("whisper");
    let heard = asr
        .transcribe_greedy(&pcm)
        .expect("transcribe")
        .to_lowercase();
    let hits = WORDS.iter().filter(|w| heard.contains(*w)).count();
    eprintln!("whisper: {heard} ({hits}/6)");
    assert!(hits >= 5, "only {hits}/6 words recognized: {heard:?}");
}
