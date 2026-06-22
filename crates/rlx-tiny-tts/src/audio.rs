//! WAV output helpers.

use std::path::Path;

use anyhow::Result;

/// Peak-normalize to avoid clipping while preserving relative dynamics. TinyTTS
/// already emits `tanh`-bounded audio in `[-1, 1]`, so this only trims rare
/// overshoot; it is a no-op when the peak is already within range.
pub fn normalize_audio(samples: &[f32]) -> Vec<f32> {
    let peak = samples.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
    if peak <= 1.0 || peak == 0.0 {
        return samples.to_vec();
    }
    let inv = 1.0 / peak;
    samples.iter().map(|&x| x * inv).collect()
}

/// Write a mono 16-bit PCM WAV file.
pub fn write_wav(path: &Path, samples: &[f32], sample_rate: u32) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec)?;
    for &s in samples {
        let v = (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        writer.write_sample(v)?;
    }
    writer.finalize()?;
    Ok(())
}
