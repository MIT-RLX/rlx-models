// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! Audio file loading. WAV goes through the in-crate RIFF reader; compressed
//! formats (mp3 / m4a-aac / flac) are decoded with `symphonia`. Everything is
//! returned as mono `f32` resampled to the requested rate.

use std::path::Path;

use anyhow::{Context, Result, bail};

use crate::wav;

/// Load any supported audio file as mono `f32` at `target_sr` Hz.
pub fn load_mono(path: &Path, target_sr: u32) -> Result<Vec<f32>> {
    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    let (samples, sr) = if ext == "wav" {
        let bytes = std::fs::read(path)?;
        let w = wav::parse(&bytes)?;
        (w.samples, w.sample_rate)
    } else {
        decode_compressed(path)?
    };
    Ok(wav::resample(&samples, sr, target_sr))
}

/// Decode mp3 / m4a / flac via symphonia → (mono f32, sample_rate).
fn decode_compressed(path: &Path) -> Result<(Vec<f32>, u32)> {
    use symphonia::core::audio::SampleBuffer;
    use symphonia::core::codecs::DecoderOptions;
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;
    use symphonia::core::probe::Hint;

    let file = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
        hint.with_extension(ext);
    }
    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .with_context(|| format!("probe {}", path.display()))?;
    let mut format = probed.format;
    let track = format
        .default_track()
        .context("no default audio track")?
        .clone();
    let track_id = track.id;
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .context("make decoder")?;
    let sr = track
        .codec_params
        .sample_rate
        .context("unknown sample rate")?;

    let mut out: Vec<f32> = Vec::new();
    while let Ok(packet) = format.next_packet() {
        if packet.track_id() != track_id {
            continue;
        }
        let decoded = match decoder.decode(&packet) {
            Ok(d) => d,
            Err(symphonia::core::errors::Error::DecodeError(_)) => continue,
            Err(_) => break,
        };
        let spec = *decoded.spec();
        let ch = spec.channels.count().max(1);
        let mut buf = SampleBuffer::<f32>::new(decoded.capacity() as u64, spec);
        buf.copy_interleaved_ref(decoded);
        for frame in buf.samples().chunks(ch) {
            out.push(frame.iter().sum::<f32>() / ch as f32);
        }
    }
    if out.is_empty() {
        bail!("decoded no audio from {}", path.display());
    }
    Ok((out, sr))
}
