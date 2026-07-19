//! WAV output helpers.

use std::path::Path;

use anyhow::Result;

/// Minimum peak absolute amplitude treated as audible speech (not silence).
pub const MIN_AUDIBLE_PEAK: f32 = 1e-3;

/// Decoder upsamples 512× per latent frame; `y_len=1` (broken duration / CT) yields
/// exactly 512 samples — reject anything shorter than ~36 ms @ 44.1 kHz.
pub const MIN_AUDIBLE_SAMPLES: usize = 1600;

/// Peak absolute amplitude.
pub fn peak_amplitude(samples: &[f32]) -> f32 {
    samples.iter().fold(0.0f32, |m, &x| m.max(x.abs()))
}

/// Fail loudly on silent / collapsed waveforms (MSI wgpu/vulkan used to emit
/// 512 samples when duration collapsed to `y_len=1`).
pub fn ensure_audible(samples: &[f32]) -> Result<()> {
    let peak = peak_amplitude(samples);
    anyhow::ensure!(
        samples.len() >= MIN_AUDIBLE_SAMPLES && peak >= MIN_AUDIBLE_PEAK,
        "synthesized audio is silent/empty (samples={}, peak={peak:.2e}); \
         check ConvTranspose / AOT cache (CACHE_TAG tiny_tts_v3_ct)",
        samples.len()
    );
    Ok(())
}

/// Peak-normalize to avoid clipping while preserving relative dynamics. TinyTTS
/// already emits `tanh`-bounded audio in `[-1, 1]`, so this only trims rare
/// overshoot; it is a no-op when the peak is already within range.
pub fn normalize_audio(samples: &[f32]) -> Vec<f32> {
    let peak = peak_amplitude(samples);
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
