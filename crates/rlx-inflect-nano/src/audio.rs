//! Output audio post-processing + WAV writing. Mirrors `inference.py::normalize_audio`.

use std::path::Path;

use anyhow::Result;

/// Normalize to -20 dB RMS with a -1 dB peak limiter, then clip to [-1, 1].
/// Uses f64 accumulation for the RMS (numpy uses float64), matching the reference.
pub fn normalize_audio(audio: &[f32]) -> Vec<f32> {
    if audio.is_empty() {
        return vec![0.0];
    }
    let n = audio.len() as f64;
    let mean = audio.iter().map(|&v| v as f64).sum::<f64>() / n;
    let mut out: Vec<f32> = audio.iter().map(|&v| (v as f64 - mean) as f32).collect();

    let ms = out.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / n;
    let rms_db = 20.0 * (ms.sqrt() + 1e-9).log10();
    let gain = 10f64.powf((-20.0 - rms_db) / 20.0) as f32;
    for v in out.iter_mut() {
        *v *= gain;
    }

    let peak = out.iter().map(|v| v.abs()).fold(0.0f32, f32::max) + 1e-9;
    let peak_limit = 10f32.powf(-1.0 / 20.0);
    if peak > peak_limit {
        let g = peak_limit / peak;
        for v in out.iter_mut() {
            *v *= g;
        }
    }
    for v in out.iter_mut() {
        *v = v.clamp(-1.0, 1.0);
    }
    out
}

/// Write mono f32 samples as 16-bit PCM WAV.
pub fn write_wav(path: &Path, samples: &[f32], sample_rate: u32) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec)?;
    for &s in samples {
        let v = (s.clamp(-1.0, 1.0) * 32767.0).round() as i16;
        writer.write_sample(v)?;
    }
    writer.finalize()?;
    Ok(())
}
