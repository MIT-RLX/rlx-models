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

//! Uniform ASR benchmark for Voxtral, driven by the shared
//! [`rlx_core::asr_bench`] harness so results are directly comparable with
//! every other rlx ASR crate. Runs a batch baseline plus a chunked-streaming
//! sweep and prints machine-readable `ASRBENCH ...` lines plus a table.
//!
//! Voxtral is an audio-LLM: each window is transcribed by fusing the audio
//! encoder/projector output into a Mistral transcription prompt and decoding
//! the Llama trunk (see [`rlx_voxtral::transcription_prompt_ids`]).
//!
//! Uniform env convention (shared by all rlx ASR `bench_asr` examples):
//!   RLX_ASR_MODEL      Voxtral model dir (safetensors/config/tekken)  (required)
//!   RLX_ASR_WAV        path to a WAV clip                             (required)
//!   RLX_ASR_REFERENCE  path to a reference transcript (.txt)          (optional)
//!   RLX_ASR_DEVICE     cpu | metal | mlx | gpu                        (default cpu)
//!   RLX_ASR_CHUNKS     comma list of chunk seconds                    (default 4,8)
//!
//! Run, e.g.:
//!   RLX_ASR_MODEL=/models/Voxtral-Mini-3B-2507 RLX_ASR_WAV=clip.wav \
//!   RLX_ASR_REFERENCE=clip.txt RLX_ASR_DEVICE=cpu \
//!   cargo run -p rlx-voxtral --example bench_asr --release

use std::path::PathBuf;

use anyhow::Result;
use rlx_core::asr_bench::{
    AsrBenchConfig, load_clip_16k, parse_chunks, parse_device, run_asr_bench,
};
use rlx_voxtral::{VoxtralRunner, decode_token_ids, pcm_to_mel, transcription_prompt_ids};

fn main() -> Result<()> {
    let Some(dir) = env_path("RLX_ASR_MODEL") else {
        return skip("RLX_ASR_MODEL is unset (Voxtral model dir)");
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

    // The builder resolves config.json + tekken tokenizer from the model dir.
    let runner = VoxtralRunner::builder()
        .weights(&dir)
        .device(device)
        .build()?;

    let (pcm, audio_s) = load_clip_16k(&wav_path)?;

    eprintln!(
        "[bench_asr] crate=rlx-voxtral device={device_label} audio={audio_s:.2}s \
         reference={} chunks={chunks:?}",
        if reference.is_some() { "yes" } else { "none" }
    );

    let cfg = AsrBenchConfig {
        chunks,
        reference,
        ..Default::default()
    };

    // Voxtral transcribes by: native log-mel -> Mistral transcription prompt
    // (audio placeholders + transcribe template) -> fused encoder/decoder ->
    // decode token ids with the tekken tokenizer next to the weights.
    let transcribe = |window: &[f32]| -> Result<String> {
        let vcfg = runner.config();
        let mel = pcm_to_mel(&vcfg.audio_config, window)?;
        let n_audio = vcfg.audio_config.audio_token_count(mel.n_frames);
        let prompt = transcription_prompt_ids(vcfg, n_audio, None, Some(runner.model_dir()))?;
        let tokens = runner.generate(&prompt, &mel)?;
        // Strip the prompt prefix so only the generated transcript is scored.
        let generated = &tokens[prompt.len().min(tokens.len())..];
        decode_token_ids(Some(runner.model_dir()), generated)
    };

    run_asr_bench(
        "rlx-voxtral",
        &device_label,
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
        "  set RLX_ASR_MODEL (Voxtral model dir) and RLX_ASR_WAV \
         (and optionally RLX_ASR_REFERENCE) to run."
    );
    Ok(())
}
