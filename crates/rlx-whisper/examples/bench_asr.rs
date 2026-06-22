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

//! Whisper ASR benchmark driven by the SHARED `rlx_core::asr_bench` harness, so
//! results line up directly with every other rlx ASR crate (batch baseline +
//! chunked-streaming sweep, uniform `ASRBENCH` lines + table).
//!
//! Uniform env convention (shared across all rlx ASR `bench_asr` examples):
//!   RLX_ASR_MODEL     — whisper model dir (model.safetensors + config.json +
//!                       tokenizer.json). Falls back to WHISPER_MODEL_DIR.
//!   RLX_ASR_WAV       — WAV clip to benchmark (required; skips if unset/missing)
//!   RLX_ASR_REFERENCE — optional reference transcript .txt (enables WER/CER/BSF)
//!   RLX_ASR_DEVICE    — cpu|metal|mlx|gpu (default cpu)
//!   RLX_ASR_CHUNKS    — comma list of streaming chunk seconds (default 4,8)
//!
//! Run, e.g.:
//!   RLX_ASR_MODEL=/models/whisper-tiny RLX_ASR_WAV=clip.wav \
//!   cargo run -p rlx-whisper --example bench_asr --release
//!
//! Whisper weights may need downloading first (a model dir with
//! model.safetensors + config.json + tokenizer.json). With no weights locally
//! the example still compiles and skips cleanly at runtime.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use rlx_core::asr_bench::{
    AsrBenchConfig, load_clip_16k, parse_chunks, parse_device, run_asr_bench,
};
use rlx_whisper::WhisperRunner;

fn main() -> Result<()> {
    // ── model dir: RLX_ASR_MODEL, fallback WHISPER_MODEL_DIR ──────────────
    let Some(dir) = std::env::var("RLX_ASR_MODEL")
        .ok()
        .or_else(|| std::env::var("WHISPER_MODEL_DIR").ok())
        .map(PathBuf::from)
    else {
        return skip(
            "RLX_ASR_MODEL (or WHISPER_MODEL_DIR) unset (dir with model.safetensors + config.json + tokenizer.json)",
        );
    };
    let weights = dir.join("model.safetensors");
    if !weights.is_file() {
        return skip(&format!(
            "model.safetensors not found in {} (download whisper weights first)",
            dir.display()
        ));
    }

    // ── audio clip: RLX_ASR_WAV (required) ────────────────────────────────
    let Some(wav) = std::env::var("RLX_ASR_WAV").ok().map(PathBuf::from) else {
        return skip("RLX_ASR_WAV is unset (path to a WAV clip to benchmark)");
    };
    if !wav.is_file() {
        return skip(&format!("RLX_ASR_WAV not found: {}", wav.display()));
    }

    // ── device ────────────────────────────────────────────────────────────
    let device_s = std::env::var("RLX_ASR_DEVICE").unwrap_or_else(|_| "cpu".into());
    let device = parse_device(&device_s)?;

    // ── optional reference transcript ─────────────────────────────────────
    let reference = match std::env::var("RLX_ASR_REFERENCE").ok().map(PathBuf::from) {
        Some(p) => Some(
            std::fs::read_to_string(&p)
                .with_context(|| format!("read reference {}", p.display()))?,
        ),
        None => None,
    };

    // ── chunk sweep ───────────────────────────────────────────────────────
    let chunks = parse_chunks(&std::env::var("RLX_ASR_CHUNKS").unwrap_or_default());

    // ── load clip (16 kHz mono PCM, exactly what Whisper expects) ─────────
    let (pcm, audio_s) =
        load_clip_16k(Path::new(&wav)).with_context(|| format!("load clip {}", wav.display()))?;

    // Multilingual Whisper checkpoints need a forced decoder language token;
    // without it the auto-detect path mis-fires and transcribes to noise. Force
    // the language (RLX_ASR_LANG, default "en"); ".en" models ignore it.
    let lang = std::env::var("RLX_ASR_LANG").unwrap_or_else(|_| "en".into());
    let mut runner = WhisperRunner::builder()
        .weights(&weights)
        .device(device)
        .language(lang)
        .build()
        .context("build WhisperRunner")?;

    eprintln!(
        "[bench_asr] crate=rlx-whisper device={device_s} audio={audio_s:.2}s clip={}",
        wav.display()
    );

    let cfg = AsrBenchConfig {
        chunks,
        reference,
        ..Default::default()
    };

    let transcribe = |window: &[f32]| -> Result<String> {
        runner.transcribe_greedy(window).context("transcribe")
    };

    run_asr_bench("rlx-whisper", &device_s, &pcm, audio_s, &cfg, transcribe)?;
    Ok(())
}

fn skip(reason: &str) -> Result<()> {
    eprintln!("[bench_asr] skipped: {reason}");
    Ok(())
}
