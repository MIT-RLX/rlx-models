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

//! Uniform ASR benchmark for Nemotron 3.5 ASR, driven by the shared
//! [`rlx_core::asr_bench`] harness so its numbers are directly comparable with
//! every other rlx ASR crate.
//!
//! Assets are supplied via the **uniform** ASR env convention (with a fallback
//! to this crate's legacy var):
//!
//!   RLX_ASR_MODEL      path to the `.nemo` checkpoint
//!                      (fallback: RLX_NEMOTRON_NEMO)              (required)
//!   RLX_ASR_WAV        path to a mono WAV clip                    (required)
//!   RLX_ASR_REFERENCE  path to a reference transcript (.txt)      (optional)
//!   RLX_ASR_DEVICE     cpu | metal | mlx | gpu  (default cpu)
//!   RLX_ASR_CHUNKS     comma list of chunk seconds (default "4,8")
//!
//! Run, e.g.:
//!   RLX_ASR_MODEL=model.nemo RLX_ASR_WAV=clip.wav \
//!   RLX_ASR_REFERENCE=clip.txt RLX_ASR_DEVICE=cpu \
//!   cargo run -p rlx-nemotron-asr --example bench_asr --release
//!
//! Missing model/WAV → prints `[bench_asr] skipped: ...` and exits Ok.

use std::path::PathBuf;

use anyhow::Result;
use rlx_core::asr_bench::{self, AsrBenchConfig};
use rlx_nemotron_asr::NemotronAsr;

fn main() -> Result<()> {
    // RLX_ASR_MODEL with fallback to the crate's legacy RLX_NEMOTRON_NEMO.
    let Some(model) = env_path("RLX_ASR_MODEL").or_else(|| env_path("RLX_NEMOTRON_NEMO")) else {
        return skip("RLX_ASR_MODEL (or RLX_NEMOTRON_NEMO) unset (path to a .nemo checkpoint)");
    };
    let Some(wav_path) = env_path("RLX_ASR_WAV") else {
        return skip("RLX_ASR_WAV unset (path to a mono WAV clip)");
    };
    if !model.is_file() {
        return skip(&format!("model not found: {}", model.display()));
    }
    if !wav_path.is_file() {
        return skip(&format!("WAV not found: {}", wav_path.display()));
    }

    let device_s = std::env::var("RLX_ASR_DEVICE").unwrap_or_else(|_| "cpu".into());
    let device = asr_bench::parse_device(&device_s)?;

    let chunks = asr_bench::parse_chunks(&std::env::var("RLX_ASR_CHUNKS").unwrap_or_default());
    let reference = std::env::var("RLX_ASR_REFERENCE")
        .ok()
        .map(std::fs::read_to_string)
        .transpose()?
        .map(|s| s.trim().to_string());

    // ── load model + audio (16 kHz mono via the shared loader) ───────────────
    let asr = NemotronAsr::open(&model, device)?;
    let (pcm, audio_s) = asr_bench::load_clip_16k(&wav_path)?;

    eprintln!(
        "[bench_asr] crate=rlx-nemotron-asr device={device_s} audio={audio_s:.2}s \
         reference={} chunks={chunks:?}",
        if reference.is_some() { "yes" } else { "none" }
    );

    let cfg = AsrBenchConfig {
        chunks,
        reference,
        ..Default::default()
    };

    // Batch path mirrors streaming_wer.rs: feed the PCM window to `transcribe`.
    let transcribe = |window: &[f32]| -> Result<String> { asr.transcribe(window) };

    asr_bench::run_asr_bench(
        "rlx-nemotron-asr",
        &device_s,
        &pcm,
        audio_s,
        &cfg,
        transcribe,
    )?;
    Ok(())
}

fn env_path(key: &str) -> Option<PathBuf> {
    std::env::var(key).ok().map(PathBuf::from)
}

fn skip(reason: &str) -> Result<()> {
    eprintln!("[bench_asr] skipped: {reason}");
    eprintln!(
        "  set RLX_ASR_MODEL (or RLX_NEMOTRON_NEMO) and RLX_ASR_WAV \
         (optionally RLX_ASR_REFERENCE / RLX_ASR_DEVICE / RLX_ASR_CHUNKS) to run."
    );
    Ok(())
}
