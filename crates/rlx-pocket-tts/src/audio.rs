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
