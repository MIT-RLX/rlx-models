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

//! Uniform ASR benchmark for Qwen3-ASR, driven by the shared
//! [`rlx_core::asr_bench`] harness so results are directly comparable with
//! every other rlx ASR crate. Runs a batch baseline plus a chunked-streaming
//! sweep and prints machine-readable `ASRBENCH ...` lines plus a table.
//!
//! Uniform env convention (shared by all rlx ASR `bench_asr` examples):
//!   RLX_ASR_MODEL      model dir (model.safetensors/config/tokenizer)  (required)
//!                      falls back to RLX_QWEN3_MODEL_DIR if set
//!   RLX_ASR_WAV        path to a WAV clip                              (required)
//!   RLX_ASR_REFERENCE  path to a reference transcript (.txt)          (optional)
//!   RLX_ASR_DEVICE     cpu | metal | mlx | gpu                        (default cpu)
//!   RLX_ASR_CHUNKS     comma list of chunk seconds                    (default 4,8)
//!
//! Run, e.g.:
//!   RLX_ASR_MODEL=/models/qwen3-asr-0.6b RLX_ASR_WAV=clip.wav \
//!   RLX_ASR_REFERENCE=clip.txt RLX_ASR_DEVICE=cpu \
//!   cargo run -p rlx-qwen3-asr --example bench_asr --release

use std::path::PathBuf;

use anyhow::Result;
use rlx_core::asr_bench::{
    AsrBenchConfig, load_clip_16k, parse_chunks, parse_device, run_asr_bench,
};
use rlx_qwen3_asr::AsrRunner;

fn main() -> Result<()> {
    let Some(dir) = env_path("RLX_ASR_MODEL").or_else(|| env_path("RLX_QWEN3_MODEL_DIR")) else {
        return skip("RLX_ASR_MODEL is unset (model dir)");
    };
    let Some(wav_path) = env_path("RLX_ASR_WAV") else {
        return skip("RLX_ASR_WAV is unset (WAV clip)");
    };
    if !dir.is_dir() {
        return skip(&format!("RLX_ASR_MODEL not a dir: {}", dir.display()));
    }
    if !wav_path.is_file() {
        return skip(&format!("RLX_ASR_WAV not found: {}", wav_path.display()));
    }

    let device_s = std::env::var("RLX_ASR_DEVICE").unwrap_or_else(|_| "cpu".into());
    let device = parse_device(&device_s)?;
    let device_label = device_s.to_lowercase();

    let chunks = parse_chunks(&std::env::var("RLX_ASR_CHUNKS").unwrap_or_default());
    let reference = std::env::var("RLX_ASR_REFERENCE")
        .ok()
        .map(std::fs::read_to_string)
        .transpose()?
        .map(|s| s.trim().to_string());

    // The builder derives config.json + tokenizer from the weights' parent dir.
    let runner = AsrRunner::builder()
        .weights(dir.join("model.safetensors"))
        .device(device)
        .build()?;

    let (pcm, audio_s) = load_clip_16k(&wav_path)?;

    eprintln!(
        "[bench_asr] crate=rlx-qwen3-asr device={device_label} audio={audio_s:.2}s \
         reference={} chunks={chunks:?}",
        if reference.is_some() { "yes" } else { "none" }
    );

    let cfg = AsrBenchConfig {
        chunks,
        reference,
        ..Default::default()
    };

    run_asr_bench(
        "rlx-qwen3-asr",
        &device_label,
        &pcm,
        audio_s,
        &cfg,
        |window| runner.transcribe_pcm(window, ""),
    )?;

    Ok(())
}

fn env_path(key: &str) -> Option<PathBuf> {
    std::env::var(key).ok().map(PathBuf::from)
}

fn skip(reason: &str) -> Result<()> {
    eprintln!("[bench_asr] skipped: {reason}");
    eprintln!(
        "  set RLX_ASR_MODEL (or RLX_QWEN3_MODEL_DIR) and RLX_ASR_WAV \
         (and optionally RLX_ASR_REFERENCE) to run."
    );
    Ok(())
}
