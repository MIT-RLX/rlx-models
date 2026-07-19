//! High-level Sesame CSM session: LM (eager) + Mimi encode/decode.

use anyhow::{Context, Result};
use rlx_mimi::{MimiCodec, MimiCodes, SAMPLE_RATE};
use rlx_runtime::Device;
use std::path::{Path, PathBuf};

use crate::generate::{GenerateOpts, generate_codes};
use crate::tokenize::{SesameTokenizer, default_mimi_dir, default_model_dir, ensure_model_dir};
use crate::weights::CsmWeights;

#[derive(Debug, Clone)]
pub struct SynthesisResult {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub audio_frames: Vec<Vec<u32>>,
}

pub struct SesameSession {
    weights: CsmWeights,
    tokenizer: SesameTokenizer,
    mimi: MimiCodec,
    device: Device,
    model_dir: PathBuf,
    mimi_dir: PathBuf,
}

impl SesameSession {
    pub fn open(model_dir: impl AsRef<Path>, mimi_dir: impl AsRef<Path>) -> Result<Self> {
        Self::open_on(model_dir, mimi_dir, Device::Cpu)
    }

    pub fn open_on(
        model_dir: impl AsRef<Path>,
        mimi_dir: impl AsRef<Path>,
        device: Device,
    ) -> Result<Self> {
        let model_dir = model_dir.as_ref().to_path_buf();
        let mimi_dir = mimi_dir.as_ref().to_path_buf();
        ensure_model_dir(&model_dir)?;
        rlx_mimi::ensure_weights(&mimi_dir).context("mimi weights")?;
        let weights = CsmWeights::load(&model_dir).context("load CSM weights")?;
        let tokenizer = SesameTokenizer::load(&model_dir)?;
        let mimi = MimiCodec::open_on(&mimi_dir, device).context("open Mimi")?;
        Ok(Self {
            weights,
            tokenizer,
            mimi,
            device,
            model_dir,
            mimi_dir,
        })
    }

    pub fn open_defaults(device: Device) -> Result<Self> {
        Self::open_on(default_model_dir(), default_mimi_dir(), device)
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn model_dir(&self) -> &Path {
        &self.model_dir
    }

    pub fn mimi_dir(&self) -> &Path {
        &self.mimi_dir
    }

    pub fn config(&self) -> &crate::config::SesameConfig {
        &self.weights.cfg
    }

    /// Text-only synthesis.
    pub fn synthesize(&mut self, text: &str, opts: &GenerateOpts) -> Result<SynthesisResult> {
        self.synthesize_with_context(text, None, opts)
    }

    /// Optional context PCM @ 24 kHz mono for conversational continuity.
    pub fn synthesize_with_context(
        &mut self,
        text: &str,
        context_pcm: Option<&[f32]>,
        opts: &GenerateOpts,
    ) -> Result<SynthesisResult> {
        let frames = self.generate_frames(text, context_pcm, opts)?;
        let samples = self.decode_frames(&frames)?;
        Ok(SynthesisResult {
            samples,
            sample_rate: SAMPLE_RATE,
            audio_frames: frames,
        })
    }

    /// Eager LM only — emit Mimi codebook frames (no codec decode).
    pub fn generate_frames(
        &mut self,
        text: &str,
        context_pcm: Option<&[f32]>,
        opts: &GenerateOpts,
    ) -> Result<Vec<Vec<u32>>> {
        let context_codes = if let Some(pcm) = context_pcm {
            let codes = self
                .mimi
                .encode_pcm(pcm, Some(self.weights.cfg.num_codebooks))?;
            Some(codes.frames)
        } else {
            None
        };
        let frames = generate_codes(
            &self.weights,
            &self.tokenizer,
            text,
            context_codes.as_deref(),
            opts,
        )?;
        if frames.is_empty() {
            anyhow::bail!("CSM produced zero audio frames");
        }
        Ok(frames)
    }

    /// Decode codebook frames with the session's Mimi device.
    pub fn decode_frames(&mut self, frames: &[Vec<u32>]) -> Result<Vec<f32>> {
        let mimi_codes = MimiCodes {
            frames: frames.to_vec(),
            num_quantizers: self
                .weights
                .cfg
                .num_codebooks
                .min(frames.first().map(|r| r.len()).unwrap_or(0)),
        };
        self.mimi.decode_codes(&mimi_codes)
    }

    /// Decode the same frames on an arbitrary Mimi device (backend matrix).
    pub fn decode_frames_on(
        mimi_dir: impl AsRef<Path>,
        device: Device,
        frames: &[Vec<u32>],
        num_codebooks: usize,
    ) -> Result<Vec<f32>> {
        let mimi = MimiCodec::open_on(mimi_dir.as_ref(), device).context("open Mimi")?;
        let mimi_codes = MimiCodes {
            frames: frames.to_vec(),
            num_quantizers: num_codebooks.min(frames.first().map(|r| r.len()).unwrap_or(0)),
        };
        mimi.decode_codes(&mimi_codes)
    }
}

/// Write mono f32 PCM to a WAV file.
pub fn write_wav(path: impl AsRef<Path>, samples: &[f32], sample_rate: u32) -> Result<()> {
    let path = path.as_ref();
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec)
        .with_context(|| format!("create {}", path.display()))?;
    for &s in samples {
        let v = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
        writer.write_sample(v)?;
    }
    writer.finalize()?;
    Ok(())
}

/// Load mono WAV, resampled to 24 kHz if needed (nearest-neighbour for non-24k).
pub fn load_wav_mono_24k(path: impl AsRef<Path>) -> Result<Vec<f32>> {
    let path = path.as_ref();
    let mut reader =
        hound::WavReader::open(path).with_context(|| format!("open {}", path.display()))?;
    let spec = reader.spec();
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().map(|s| s.unwrap_or(0.0)).collect(),
        hound::SampleFormat::Int => {
            let max = (1i32 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.unwrap_or(0) as f32 / max)
                .collect()
        }
    };
    let ch = spec.channels as usize;
    let mono: Vec<f32> = if ch <= 1 {
        samples
    } else {
        samples
            .chunks(ch)
            .map(|c| c.iter().sum::<f32>() / ch as f32)
            .collect()
    };
    if spec.sample_rate == SAMPLE_RATE {
        return Ok(mono);
    }
    // Linear resample to 24 kHz.
    let ratio = SAMPLE_RATE as f64 / spec.sample_rate as f64;
    let out_len = ((mono.len() as f64) * ratio).round() as usize;
    let mut out = vec![0.0f32; out_len];
    for (i, o) in out.iter_mut().enumerate() {
        let src = i as f64 / ratio;
        let i0 = src.floor() as usize;
        let i1 = (i0 + 1).min(mono.len().saturating_sub(1));
        let t = (src - i0 as f64) as f32;
        *o = mono[i0] * (1.0 - t) + mono[i1] * t;
    }
    Ok(out)
}
