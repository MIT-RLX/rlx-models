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

//! Shared F5-TTS helpers: per-call options, waveform peak, and WAV writing.
//! The synthesis pipeline lives in the native RLX path ([`crate::native`]).

use std::path::Path;

use anyhow::{Context, Result};

pub fn peak_amplitude(a: &[f32]) -> f32 {
    a.iter()
        .filter(|s| s.is_finite())
        .map(|s| s.abs())
        .fold(0.0, f32::max)
}

/// Write mono f32 PCM to a 16-bit WAV at `sample_rate`.
pub fn write_wav(audio: &[f32], sample_rate: u32, path: &Path) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut w = hound::WavWriter::create(path, spec)
        .with_context(|| format!("create {}", path.display()))?;
    for &s in audio {
        w.write_sample((s * i16::MAX as f32).clamp(i16::MIN as f32, i16::MAX as f32) as i16)
            .context("wav write")?;
    }
    w.finalize().context("wav finalize")?;
    Ok(())
}

/// Per-call options.
#[derive(Debug, Clone, Copy)]
pub struct InferOpts {
    pub nfe: usize,
    pub speed: f32,
}
impl Default for InferOpts {
    fn default() -> Self {
        Self {
            nfe: crate::config::DEFAULT_NFE,
            speed: 1.0,
        }
    }
}
