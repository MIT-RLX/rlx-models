use anyhow::{Context, Result, ensure};
use hound::{WavReader, WavSpec, WavWriter};
use std::path::Path;

pub fn load_wav_mono(path: &Path, target_hz: u32) -> Result<Vec<f32>> {
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
    if spec.channels == 2 {
        samples = samples
            .chunks_exact(2)
            .map(|c| 0.5 * (c[0] + c[1]))
            .collect();
    } else {
        ensure!(spec.channels == 1, "expected mono or stereo wav");
    }
    if spec.sample_rate != target_hz {
        samples = resample_linear(&samples, spec.sample_rate, target_hz);
    }
    Ok(samples)
}

pub fn write_wav_mono(path: &Path, pcm: &[f32], sample_rate: u32) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("create dir {}", parent.display()))?;
        }
    }
    let spec = WavSpec {
        channels: 1,
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

fn resample_linear(samples: &[f32], from_hz: u32, to_hz: u32) -> Vec<f32> {
    rlx_core::resample_linear(samples, from_hz, to_hz)
}
