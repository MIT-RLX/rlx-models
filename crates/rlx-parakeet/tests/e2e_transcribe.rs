// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0-only

//! End-to-end Parakeet-TDT: `.nemo` → transcribe. Skips without a checkpoint.
//! Set `RLX_PARAKEET_NEMO` (a Parakeet-TDT `.nemo`) and optionally
//! `RLX_PARAKEET_WAV` (defaults to the JFK harness clip).

use std::path::Path;

use rlx_nemotron_asr::wav;
use rlx_parakeet::Parakeet;
use rlx_runtime::Device;

#[test]
fn parakeet_tdt_transcribes_end_to_end() {
    let Ok(nemo) = std::env::var("RLX_PARAKEET_NEMO") else {
        eprintln!("skip: set RLX_PARAKEET_NEMO to a Parakeet-TDT .nemo checkpoint");
        return;
    };
    let wav_path = std::env::var("RLX_PARAKEET_WAV")
        .unwrap_or_else(|_| "assets/harness/jfk_16k.wav".to_string());

    let pk = Parakeet::open(Path::new(&nemo), Device::Cpu).expect("open parakeet");
    let target_sr = pk.config().sample_rate as u32;
    // A non-empty duration table is required for the TDT decode.
    assert!(!pk.durations().is_empty(), "empty TDT duration table");

    let bytes = std::fs::read(&wav_path).expect("read wav");
    let w = wav::parse(&bytes).expect("parse wav");
    let pcm = if w.sample_rate != target_sr {
        wav::resample(&w.samples, w.sample_rate, target_sr)
    } else {
        w.samples
    };

    let text = pk.transcribe(&pcm).expect("transcribe");
    eprintln!("parakeet transcript: {text:?}");
    // Real audio must produce some tokens (not an empty string).
    assert!(
        !text.trim().is_empty(),
        "empty transcript for non-empty audio"
    );
}
