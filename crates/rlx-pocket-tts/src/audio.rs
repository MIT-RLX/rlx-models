// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// Licensed under GNU GPL v3. See top-level LICENSE.

//! WAV writer / reader (24 kHz mono).

use std::path::Path;

use anyhow::{Context, Result};

use crate::SAMPLE_RATE;

pub fn write_wav_mono(path: impl AsRef<Path>, samples: &[f32]) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: SAMPLE_RATE,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create(path.as_ref(), spec)
        .with_context(|| format!("create wav {}", path.as_ref().display()))?;
    for s in samples {
        writer
            .write_sample(s.clamp(-1.0, 1.0))
            .context("write sample")?;
    }
    writer.finalize().context("finalize wav")?;
    Ok(())
}

/// Read a WAV file as mono f32 (channels averaged) plus its sample rate.
pub fn read_wav_mono(path: impl AsRef<Path>) -> Result<(Vec<f32>, u32)> {
    let path = path.as_ref();
    let mut reader =
        hound::WavReader::open(path).with_context(|| format!("open wav {}", path.display()))?;
    let spec = reader.spec();
    let ch = spec.channels.max(1) as usize;
    let interleaved: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<std::result::Result<_, _>>()?,
        hound::SampleFormat::Int => {
            let scale = 1.0 / (1u64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 * scale))
                .collect::<std::result::Result<_, _>>()?
        }
    };
    let mono = interleaved
        .chunks(ch)
        .map(|f| f.iter().sum::<f32>() / ch as f32)
        .collect();
    Ok((mono, spec.sample_rate))
}

/// `pocket_tts.data.audio_utils.end_on_pause`: cut trailing silence (20 ms
/// frames more than 35 dB below the loudest), fade the last 20 ms out, and
/// append 80 ms of zeros, so a voice prompt ends inside a pause.
pub fn end_on_pause(wav: &[f32], sample_rate: usize) -> Vec<f32> {
    let frame = ((0.02 * sample_rate as f64) as usize).max(1);
    let n = wav.len() / frame;
    if n == 0 {
        return wav.to_vec();
    }
    let db: Vec<f32> = wav[..n * frame]
        .chunks(frame)
        .map(|f| {
            let rms = (f.iter().map(|x| x * x).sum::<f32>() / frame as f32).sqrt();
            20.0 * (rms + 1e-12).log10()
        })
        .collect();
    let max = db.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let last = db.iter().rposition(|&d| d > max - 35.0).unwrap_or(n - 1);
    let end = (last + 1) * frame;
    let mut out = wav[..end].to_vec();
    let fade = ((0.02 * sample_rate as f64) as usize).min(end);
    for i in 0..fade {
        // torch.linspace(1, 0, fade)
        let g = if fade > 1 {
            1.0 - i as f32 / (fade - 1) as f32
        } else {
            1.0
        };
        out[end - fade + i] *= g;
    }
    out.resize(end + (0.08 * sample_rate as f64) as usize, 0.0);
    out
}
