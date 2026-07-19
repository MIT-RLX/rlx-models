//! WAV I/O + light DSP helpers.

use std::path::Path;

use anyhow::{Context, Result};

pub fn read_wav_mono(path: &Path) -> Result<(Vec<f32>, u32)> {
    let mut r = hound::WavReader::open(path).with_context(|| format!("open {}", path.display()))?;
    let sr = r.spec().sample_rate;
    let ch = r.spec().channels as usize;
    let raw: Vec<f32> = match r.spec().sample_format {
        hound::SampleFormat::Int => {
            let m = (1i64 << (r.spec().bits_per_sample - 1)) as f32;
            r.samples::<i32>()
                .filter_map(|s| s.ok())
                .map(|s| s as f32 / m)
                .collect()
        }
        hound::SampleFormat::Float => r.samples::<f32>().filter_map(|s| s.ok()).collect(),
    };
    let mono = if ch > 1 {
        raw.chunks(ch)
            .map(|c| c.iter().sum::<f32>() / ch as f32)
            .collect()
    } else {
        raw
    };
    Ok((mono, sr))
}

pub fn write_wav_mono(path: &Path, pcm: &[f32], sample_rate: u32) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut w = hound::WavWriter::create(path, spec)
        .with_context(|| format!("create {}", path.display()))?;
    for &s in pcm {
        w.write_sample((s.clamp(-1.0, 1.0) * 32767.0) as i16)?;
    }
    w.finalize()?;
    Ok(())
}

pub fn resample_linear(samples: &[f32], from_hz: u32, to_hz: u32) -> Vec<f32> {
    if from_hz == to_hz || samples.is_empty() {
        return samples.to_vec();
    }
    let out_len = (samples.len() as u64 * to_hz as u64 / from_hz as u64).max(1) as usize;
    (0..out_len)
        .map(|i| {
            let src = i as f64 * from_hz as f64 / to_hz as f64;
            let idx = src.floor() as usize;
            let frac = (src - idx as f64) as f32;
            let a = samples[idx.min(samples.len() - 1)];
            let b = samples[(idx + 1).min(samples.len() - 1)];
            a + (b - a) * frac
        })
        .collect()
}

pub fn peak_normalize(pcm: &[f32], target: f32) -> Vec<f32> {
    let peak = pcm
        .iter()
        .filter(|s| s.is_finite())
        .map(|s| s.abs())
        .fold(0.0f32, f32::max);
    if peak < 1e-8 {
        return pcm.to_vec();
    }
    let g = target / peak;
    pcm.iter().map(|s| s * g).collect()
}

pub fn add_gaussian_noise(pcm: &[f32], snr_db: f32, seed: u64) -> Vec<f32> {
    let mut rng = seed;
    let signal_power = pcm.iter().map(|x| (x * x) as f64).sum::<f64>() / pcm.len().max(1) as f64;
    let noise_power = signal_power / 10f64.powf(snr_db as f64 / 10.0);
    let sigma = noise_power.sqrt() as f32;
    pcm.iter()
        .map(|&x| {
            // Box-Muller
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u1 = ((rng >> 33) as f32 / u32::MAX as f32).clamp(1e-7, 1.0);
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u2 = ((rng >> 33) as f32 / u32::MAX as f32).clamp(0.0, 1.0);
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
            x + sigma * z
        })
        .collect()
}

pub fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..n {
        let x = a[i] as f64;
        let y = b[i] as f64;
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

pub fn median(mut v: Vec<f64>) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    v[v.len() / 2]
}
