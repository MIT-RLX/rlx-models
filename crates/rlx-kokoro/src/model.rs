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

//! Shared Kokoro audio helpers (waveform peak + WAV writer), used by the native
//! ([`crate::native`]) inference path.

use std::path::Path;

use anyhow::{Context, Result};

use crate::config::SAMPLE_RATE;

/// Peak amplitude below this is treated as silent (failed) output.
pub const MIN_AUDIBLE_PEAK: f32 = 1e-3;

/// Peak absolute amplitude of a waveform.
pub fn peak_amplitude(audio: &[f32]) -> f32 {
    audio
        .iter()
        .filter(|s| s.is_finite())
        .map(|s| s.abs())
        .fold(0.0f32, f32::max)
}

/// Write mono 16-bit PCM WAV at 24 kHz.
pub fn write_wav(audio: &[f32], output_path: &Path) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(output_path, spec)
        .with_context(|| format!("create WAV: {}", output_path.display()))?;
    for &s in audio {
        let s16 = (s * i16::MAX as f32).clamp(i16::MIN as f32, i16::MAX as f32) as i16;
        writer.write_sample(s16).context("WAV write")?;
    }
    writer.finalize().context("WAV finalize")?;
    Ok(())
}
