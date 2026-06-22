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

//! Real-audio encoder + greedy transcript vs paired reference (`just fetch-whisper-bench`).
//!
//! ```sh
//! just test-whisper-jfk
//! ```

use anyhow::Result;
use rlx_models::whisper::{
    WhisperRunner, assert_transcript_matches_reference, ensure_jfk_fixture, load_wav_mono_f32,
    pcm_to_mel,
};
use rlx_runtime::Device;
use std::path::PathBuf;

fn tiny_dir() -> PathBuf {
    std::env::var("RLX_WHISPER_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.cache/whisper-tiny")
        })
}

#[test]
fn whisper_jfk_encoder_and_transcript() -> Result<()> {
    let dir = tiny_dir();
    let weights = dir.join("model.safetensors");
    if !weights.is_file() {
        eprintln!("skip: need weights (just fetch-whisper)");
        return Ok(());
    }
    let (wav, reference) = match ensure_jfk_fixture() {
        Ok(f) => f,
        Err(e) => {
            eprintln!("skip: {e}");
            return Ok(());
        }
    };

    let pcm = load_wav_mono_f32(&wav)?;
    let mut runner = WhisperRunner::builder()
        .weights(&weights)
        .config_path(dir.join("config.json"))
        .tokenizer_path(dir.join("tokenizer.json"))
        .device(Device::Cpu)
        .language("en")
        .build()?;

    let mel = pcm_to_mel(runner.config(), &pcm);
    let n = mel.data.len();
    let mean = mel.data.iter().sum::<f32>() / n as f32;
    let var = mel.data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n as f32;
    eprintln!("rlx mel mean={mean:.6} std={:.6}", var.sqrt());

    let enc = runner.encode_mel(&mel)?;
    let emean = enc.iter().sum::<f32>() / enc.len() as f32;
    let evar = enc.iter().map(|x| (x - emean).powi(2)).sum::<f32>() / enc.len() as f32;
    eprintln!(
        "rlx enc len={} mean={emean:.6} std={:.6} (hf ref ~0.035)",
        enc.len(),
        evar.sqrt()
    );

    let text = runner.transcribe_greedy(&pcm)?;
    eprintln!("transcript={text:?}");
    eprintln!("reference={reference:?}");
    assert_transcript_matches_reference(&text, &reference);
    let words: Vec<_> = text.split_whitespace().collect();
    let repeats = words.windows(2).filter(|w| w[0] == w[1]).count();
    assert!(repeats < 2, "decode stuck repeating tokens: {text:?}");
    Ok(())
}
