// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! WAV helpers — no gain normalization, no resampling.

use std::path::Path;

use anyhow::{Context, Result, bail};
use hound::{SampleFormat, WavReader, WavSpec, WavWriter};

use crate::AudioOutput;

/// Write PCM as 16-bit integer WAV without peak normalization.
///
/// Float samples are clamped to [-1, 1] only for integer conversion; the
/// waveform is not rescaled.
pub fn write_wav(output: &AudioOutput, path: &Path) -> Result<()> {
    if output.channels == 0 {
        bail!("cannot write WAV with zero channels");
    }
    if output.sample_rate == 0 {
        bail!("cannot write WAV with zero sample rate");
    }
    let spec = WavSpec {
        channels: output.channels,
        sample_rate: output.sample_rate,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };
    let mut writer =
        WavWriter::create(path, spec).with_context(|| format!("create WAV {}", path.display()))?;
    for &s in &output.pcm {
        let s16 = (s * i16::MAX as f32).clamp(i16::MIN as f32, i16::MAX as f32) as i16;
        writer.write_sample(s16).context("WAV write sample")?;
    }
    writer.finalize().context("WAV finalize")?;
    Ok(())
}

/// Read a mono or multi-channel 16-bit/float WAV into interleaved f32 PCM.
pub fn read_wav_f32(path: &Path) -> Result<AudioOutput> {
    let mut reader =
        WavReader::open(path).with_context(|| format!("open WAV {}", path.display()))?;
    let spec = reader.spec();
    let pcm: Vec<f32> = match spec.sample_format {
        SampleFormat::Int => {
            let bits = spec.bits_per_sample;
            let max = match bits {
                8 => i8::MAX as f32,
                16 => i16::MAX as f32,
                24 | 32 => i32::MAX as f32,
                other => bail!("unsupported integer WAV bit depth {other}"),
            };
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 / max))
                .collect::<Result<Vec<_>, _>>()
                .context("read int WAV samples")?
        }
        SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<Result<Vec<_>, _>>()
            .context("read float WAV samples")?,
    };
    Ok(AudioOutput {
        pcm,
        sample_rate: spec.sample_rate,
        channels: spec.channels,
        voice_identifier: String::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DEFAULT_VOICE_ID;

    #[test]
    fn wav_roundtrip_preserves_samples() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("rlx_tts_wav_roundtrip_{}.wav", std::process::id()));
        let original = AudioOutput {
            // Exact representable values after i16 quantization.
            pcm: vec![0.0, 0.5, -0.25, 1.0, -1.0],
            sample_rate: 16_000,
            channels: 1,
            voice_identifier: DEFAULT_VOICE_ID.into(),
        };
        write_wav(&original, &path).unwrap();
        let loaded = read_wav_f32(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(loaded.sample_rate, 16_000);
        assert_eq!(loaded.channels, 1);
        assert_eq!(loaded.pcm.len(), original.pcm.len());
        for (a, b) in original.pcm.iter().zip(loaded.pcm.iter()) {
            assert!((a - b).abs() < 1.0 / 32767.0 + 1e-6, "{a} vs {b}");
        }
    }
}
