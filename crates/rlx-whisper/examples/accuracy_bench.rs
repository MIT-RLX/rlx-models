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

//! Whisper accuracy + latency bench: transcribe a clip and report WER / CER /
//! RTFx using the shared `rlx_core::asr_metrics`. End-to-end proof that the
//! metrics work against a real, loaded ASR model.
//!
//! Model dir (with `model.safetensors`, `config.json`, `tokenizer.json`) comes
//! from `WHISPER_MODEL_DIR`; the audio + reference come from the existing
//! whisper bench fixture (`bench_fixture`), overridable via `RLX_WHISPER_WAV`
//! and `RLX_WHISPER_REFERENCE`. The example skips cleanly when assets are
//! missing.
//!
//! Run, e.g.:
//!   WHISPER_MODEL_DIR=/models/whisper-tiny \
//!   cargo run -p rlx-whisper --example accuracy_bench --release
//!
//!   # custom clip + reference:
//!   WHISPER_MODEL_DIR=/models/whisper-tiny \
//!   RLX_WHISPER_WAV=clip.wav RLX_WHISPER_REFERENCE=clip.txt \
//!   cargo run -p rlx-whisper --example accuracy_bench --release

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use rlx_cli::parse_standard_device;
use rlx_core::asr_metrics::{character_error_rate, rtfx, word_error_rate};
use rlx_whisper::WhisperRunner;
use rlx_whisper::audio::{SAMPLE_RATE, load_wav_mono_f32};
use rlx_whisper::bench_fixture::{jfk_wav_path, load_jfk_reference};

fn main() -> Result<()> {
    let Some(dir) = std::env::var("WHISPER_MODEL_DIR").ok().map(PathBuf::from) else {
        return skip(
            "WHISPER_MODEL_DIR is unset (dir with model.safetensors + config.json + tokenizer.json)",
        );
    };
    let weights = dir.join("model.safetensors");
    if !weights.is_file() {
        return skip(&format!("model.safetensors not found in {}", dir.display()));
    }
    let wav = jfk_wav_path();
    if !wav.is_file() {
        return skip(&format!(
            "audio clip not found at {} (run `just fetch-whisper-bench` or set RLX_WHISPER_WAV)",
            wav.display()
        ));
    }
    let reference = load_jfk_reference().context("load reference transcript")?;

    let device_s = std::env::var("RLX_WHISPER_DEVICE").unwrap_or_else(|_| "cpu".into());
    let device = parse_standard_device("whisper", &device_s)?;

    let pcm = load_wav_mono_f32(&wav).with_context(|| format!("read {}", wav.display()))?;
    let audio_s = pcm.len() as f64 / SAMPLE_RATE as f64;

    let mut runner = WhisperRunner::builder()
        .weights(&weights)
        .device(device)
        .build()
        .context("build WhisperRunner")?;

    eprintln!(
        "[accuracy_bench] device={device:?} audio={audio_s:.2}s clip={}",
        wav.display()
    );

    let t0 = Instant::now();
    let hyp = runner.transcribe_greedy(&pcm).context("transcribe")?;
    let wall = t0.elapsed().as_secs_f64();

    let wer = word_error_rate(&reference, &hyp);
    let cer = character_error_rate(&reference, &hyp);
    let rtf = rtfx(audio_s, wall);

    println!("\nreference: {reference}");
    println!("hypothesis: {hyp}");
    println!(
        "\n{:>8}  {:>8}  {:>8}  {:>9}",
        "WER%", "CER%", "RTFx", "wall_s"
    );
    println!("{}", "-".repeat(40));
    println!(
        "{:>8.2}  {:>8.2}  {:>8.2}  {:>9.3}",
        wer * 100.0,
        cer * 100.0,
        rtf,
        wall
    );
    Ok(())
}

fn skip(reason: &str) -> Result<()> {
    eprintln!("[accuracy_bench] skipped: {reason}");
    Ok(())
}
