// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// Licensed under GNU GPL v3. See top-level LICENSE.
//
//! Run a WAV file through the same Whisper config that whisper_check uses.
//! Lets us tell whether the transcription pipeline itself is broken.

use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use rlx_runtime::Device;
use rlx_whisper::{SAMPLE_RATE as WHISPER_RATE, WhisperRunner};

fn main() -> Result<()> {
    let path = std::env::args()
        .nth(1)
        .context("usage: whisper_file <path.wav>")?;
    let whisper_dir = resolve_whisper_dir()?;

    let mut reader = hound::WavReader::open(&path)?;
    let spec = reader.spec();
    let pcm: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().map(|s| s.unwrap()).collect(),
        hound::SampleFormat::Int => {
            let max = (1u32 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.unwrap() as f32 / max)
                .collect()
        }
    };
    eprintln!(
        "{}: {} samples @ {} Hz, {}ch",
        path,
        pcm.len(),
        spec.sample_rate,
        spec.channels
    );

    let pcm_16k = resample_linear(&pcm, spec.sample_rate, WHISPER_RATE as u32);
    eprintln!(
        "resampled to {} samples @ {} Hz",
        pcm_16k.len(),
        WHISPER_RATE
    );

    let mut whisper = WhisperRunner::builder()
        .weights(whisper_dir.join("model.safetensors"))
        .config_path(whisper_dir.join("config.json"))
        .tokenizer_path(whisper_dir.join("tokenizer.json"))
        .device(Device::Cpu)
        .language("en")
        .build()?;
    let transcript = whisper.transcribe_greedy(&pcm_16k)?;
    println!("transcript: {transcript:?}");
    Ok(())
}

fn resolve_whisper_dir() -> Result<PathBuf> {
    if let Ok(dir) = std::env::var("RLX_WHISPER_DIR") {
        let p = PathBuf::from(dir);
        if p.join("model.safetensors").is_file() {
            return Ok(p);
        }
    }
    bail!("set RLX_WHISPER_DIR")
}

fn resample_linear(samples: &[f32], from_hz: u32, to_hz: u32) -> Vec<f32> {
    if from_hz == to_hz || samples.is_empty() {
        return samples.to_vec();
    }
    let out_len = (samples.len() as u64 * to_hz as u64 / from_hz as u64).max(1) as usize;
    (0..out_len)
        .map(|i| {
            let src = i as f64 * from_hz as f64 / to_hz as f64;
            let idx = src.floor() as usize;
            let frac = (src - idx as f64) as f32;
            let a = samples[idx.min(samples.len() - 1)];
            let b = samples[(idx + 1).min(samples.len() - 1)];
            a + (b - a) * frac
        })
        .collect()
}
