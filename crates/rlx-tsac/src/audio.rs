use anyhow::{Context, Result, bail, ensure};
use hound::{WavReader, WavSpec, WavWriter};
use std::path::Path;

use crate::SAMPLE_RATE;

pub fn load_wav_f32(path: &Path, target_hz: u32) -> Result<(Vec<f32>, u32)> {
    let mut reader =
        WavReader::open(path).with_context(|| format!("open wav {}", path.display()))?;
    let spec = reader.spec();
    ensure!(
        spec.sample_format == hound::SampleFormat::Float
            || spec.sample_format == hound::SampleFormat::Int,
        "unsupported wav sample format"
    );
    let mut samples = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<Result<Vec<_>, _>>()
            .context("read f32 wav")?,
        hound::SampleFormat::Int => {
            let max = (1i32 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 / max))
                .collect::<Result<Vec<_>, _>>()
                .context("read int wav")?
        }
    };
    let channels = spec.channels as usize;
    if channels > 1 {
        let frames = samples.len() / channels;
        let mut interleaved = Vec::with_capacity(samples.len());
        for fi in 0..frames {
            for ch in 0..channels {
                interleaved.push(samples[fi * channels + ch]);
            }
        }
        samples = interleaved;
    }
    let rate = spec.sample_rate;
    if rate != target_hz {
        samples = resample_linear_interleaved(&samples, channels.max(1), rate, target_hz);
    }
    Ok((samples, target_hz))
}

pub fn load_wav_mono_f32(path: &Path) -> Result<Vec<f32>> {
    let (pcm, _) = load_wav_f32(path, SAMPLE_RATE)?;
    Ok(pcm)
}

pub fn write_wav_f32(path: &Path, pcm: &[f32], sample_rate: u32, channels: u16) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("create dir {}", parent.display()))?;
        }
    }
    ensure!(channels >= 1, "channels must be >= 1");
    let frames = pcm.len() / channels as usize;
    ensure!(
        frames * channels as usize == pcm.len(),
        "pcm length {} not divisible by {channels} channels",
        pcm.len()
    );
    let spec = WavSpec {
        channels,
        sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer =
        WavWriter::create(path, spec).with_context(|| format!("create {}", path.display()))?;
    for &s in pcm {
        writer.write_sample(s)?;
    }
    writer.finalize()?;
    Ok(())
}

pub fn prepare_tsac_wav(in_wav: &Path, out_wav: &Path) -> Result<u16> {
    let reader = WavReader::open(in_wav).with_context(|| format!("open {}", in_wav.display()))?;
    let channels = reader.spec().channels;
    let (pcm, _) = load_wav_f32(in_wav, SAMPLE_RATE)?;
    // Reference codecs accept PCM int16 (fmt=1) or plain float32 (fmt=3), not WAVE_FORMAT_EXTENSIBLE.
    write_wav_i16(out_wav, &pcm, SAMPLE_RATE, channels)?;
    Ok(channels)
}

fn write_wav_i16(path: &Path, pcm: &[f32], sample_rate: u32, channels: u16) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("create dir {}", parent.display()))?;
        }
    }
    ensure!(channels >= 1, "channels must be >= 1");
    let frames = pcm.len() / channels as usize;
    ensure!(
        frames * channels as usize == pcm.len(),
        "pcm length {} not divisible by {channels} channels",
        pcm.len()
    );
    let spec = WavSpec {
        channels,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer =
        WavWriter::create(path, spec).with_context(|| format!("create {}", path.display()))?;
    for &s in pcm {
        let clipped = s.clamp(-1.0, 1.0);
        let sample = (clipped * 32767.0).round() as i32;
        writer.write_sample(sample as i16)?;
    }
    writer.finalize()?;
    Ok(())
}

fn resample_linear_interleaved(
    samples: &[f32],
    channels: usize,
    from_hz: u32,
    to_hz: u32,
) -> Vec<f32> {
    if from_hz == to_hz || samples.is_empty() || channels == 0 {
        return samples.to_vec();
    }
    let in_frames = samples.len() / channels;
    if in_frames == 0 {
        return Vec::new();
    }
    let ratio = from_hz as f64 / to_hz as f64;
    let out_frames = (in_frames as f64 / ratio).ceil() as usize;
    let mut out = Vec::with_capacity(out_frames * channels);
    for fi in 0..out_frames {
        let src = fi as f64 * ratio;
        let lo = src.floor() as usize;
        let hi = (lo + 1).min(in_frames - 1);
        let frac = (src - lo as f64) as f32;
        for ch in 0..channels {
            let a = samples[lo * channels + ch];
            let b = samples[hi * channels + ch];
            out.push(a * (1.0 - frac) + b * frac);
        }
    }
    out
}

#[derive(Debug, Clone)]
pub struct PcmCompare {
    pub mse: f32,
    pub correlation: f32,
    pub max_abs: f32,
    pub samples: usize,
}

impl PcmCompare {
    pub fn compare(a: &[f32], b: &[f32]) -> Self {
        let n = a.len().min(b.len());
        Self {
            mse: mse(a, b),
            correlation: correlation(a, b),
            max_abs: max_abs_error(a, b),
            samples: n,
        }
    }

    pub fn passes(&self, min_corr: f32, max_mse: f32) -> bool {
        self.correlation >= min_corr && self.mse <= max_mse
    }
}

pub fn mse(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum::<f32>()
        / n as f32
}

pub fn correlation(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let mut sa = 0.0f32;
    let mut sb = 0.0f32;
    let mut sab = 0.0f32;
    let mut sa2 = 0.0f32;
    let mut sb2 = 0.0f32;
    for i in 0..n {
        let x = a[i];
        let y = b[i];
        sa += x;
        sb += y;
        sab += x * y;
        sa2 += x * x;
        sb2 += y * y;
    }
    let nf = n as f32;
    let denom = (nf * sa2 - sa * sa) * (nf * sb2 - sb * sb);
    if denom <= 1e-12 {
        return 0.0;
    }
    (nf * sab - sa * sb) / denom.sqrt()
}

pub fn max_abs_error(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

pub fn load_pcm_from_wav(path: &Path) -> Result<Vec<f32>> {
    let (pcm, rate) = load_wav_f32(path, SAMPLE_RATE)?;
    if rate != SAMPLE_RATE {
        bail!("expected {} Hz after load, got {rate}", SAMPLE_RATE);
    }
    Ok(pcm)
}
