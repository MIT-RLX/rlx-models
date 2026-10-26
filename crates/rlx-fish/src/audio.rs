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

//! WAV in/out at the codec rate.

use std::path::Path;

use anyhow::{Context, Result};

use crate::codec::SAMPLE_RATE;

/// Read a WAV (any rate; channels are averaged) as mono f32 at 44.1 kHz,
/// resampled the way the reference does (`torchaudio.functional.resample`).
pub fn read_wav_44k(path: &Path) -> Result<Vec<f32>> {
    let (pcm, rate) = rlx_core::asr_bench::read_wav_mono(path)
        .with_context(|| format!("read {}", path.display()))?;
    Ok(rlx_core::audio::resample_sinc(
        &pcm,
        rate as usize,
        SAMPLE_RATE as usize,
    ))
}

/// Write mono f32 PCM at 44.1 kHz as 16-bit WAV.
pub fn write_wav(path: &Path, pcm: &[f32]) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut w = hound::WavWriter::create(path, spec)
        .with_context(|| format!("create {}", path.display()))?;
    for &s in pcm {
        w.write_sample((s.clamp(-1.0, 1.0) * i16::MAX as f32).round() as i16)?;
    }
    w.finalize()?;
    Ok(())
}
