//! High-level inference session for Kyutai TTS.
//!
//! The runtime generation path is not yet implemented — the model uses
//! per-step DepFormer weights, cross-attention conditioners, and a demuxed
//! second stream, none of which exist in [`rlx_moshi`]'s eager backbone.
//!
//! The upstream Kyutai `moshi` 0.6.4 `tts` module is wired in as a
//! **dev-dependency only** for parity tests (see `tests/whisper_validate.rs`).

use crate::checkpoint::{KyutaiTtsCheckpoint, KyutaiTtsVoice};
use crate::config::KyutaiTtsConfig;
use crate::download::{default_mimi_dir, ensure_weights_checkpoint};
use anyhow::{Result, bail};
use rlx_mimi::SAMPLE_RATE as MIMI_RATE;
use rlx_runtime::Device;
use std::path::{Path, PathBuf};

/// Sampling overrides for Kyutai TTS generation.
#[derive(Debug, Clone)]
pub struct GenerationConfig {
    pub max_steps: usize,
    pub text_temperature: f64,
    pub audio_temperature: f64,
    pub cfg_alpha: f32,
    pub seed: u64,
    /// Mimi codebooks to decode (defaults to all 32; lower → faster, lower fidelity).
    pub mimi_codebooks: usize,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_steps: 100,
            text_temperature: 0.6,
            audio_temperature: 0.6,
            cfg_alpha: 2.0,
            seed: 42,
            mimi_codebooks: 32,
        }
    }
}

/// Synthesis output: mono PCM @ 24 kHz + token trace.
#[derive(Debug, Clone)]
pub struct GenerationResult {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub text_tokens: Vec<u32>,
    pub audio_frames: Vec<Vec<u32>>,
}

/// Kyutai TTS session — config + on-disk weight paths + device handle.
pub struct KyutaiTtsSession {
    cfg: KyutaiTtsConfig,
    checkpoint: KyutaiTtsCheckpoint,
    model_dir: PathBuf,
    mimi_dir: PathBuf,
    device: Device,
    voice: KyutaiTtsVoice,
}

impl KyutaiTtsSession {
    /// Open with the env-default checkpoint (`kyutai/tts-1.6b-en_fr`) on CPU.
    pub fn open(model_dir: impl AsRef<Path>, mimi_dir: impl AsRef<Path>) -> Result<Self> {
        Self::open_on(model_dir, mimi_dir, Device::Cpu)
    }

    /// Open on a specific device with the env-default checkpoint.
    pub fn open_on(
        model_dir: impl AsRef<Path>,
        mimi_dir: impl AsRef<Path>,
        device: Device,
    ) -> Result<Self> {
        Self::open_with_checkpoint(
            model_dir,
            mimi_dir,
            device,
            KyutaiTtsCheckpoint::from_env_or_default(),
        )
    }

    /// Open with explicit checkpoint preset.
    pub fn open_with_checkpoint(
        model_dir: impl AsRef<Path>,
        mimi_dir: impl AsRef<Path>,
        device: Device,
        checkpoint: KyutaiTtsCheckpoint,
    ) -> Result<Self> {
        let model_dir = model_dir.as_ref().to_path_buf();
        let mimi_dir = mimi_dir.as_ref().to_path_buf();
        ensure_weights_checkpoint(&model_dir, checkpoint)?;
        rlx_mimi::ensure_weights(&mimi_dir)?;
        let cfg = KyutaiTtsConfig::v1_6b_en_fr();
        Ok(Self {
            cfg,
            checkpoint,
            model_dir,
            mimi_dir,
            device,
            voice: KyutaiTtsVoice::unconditional(),
        })
    }

    /// Open with default cache dirs.
    pub fn open_default() -> Result<Self> {
        Self::open(
            crate::download::default_kyutai_tts_dir(),
            default_mimi_dir(),
        )
    }

    pub fn config(&self) -> &KyutaiTtsConfig {
        &self.cfg
    }

    pub fn checkpoint(&self) -> KyutaiTtsCheckpoint {
        self.checkpoint
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

    pub fn voice(&self) -> &KyutaiTtsVoice {
        &self.voice
    }

    /// Select a pre-computed voice embedding (see `kyutai/tts-voices`).
    pub fn set_voice(&mut self, voice: KyutaiTtsVoice) {
        self.voice = voice;
    }

    /// Sample rate of the synthesised audio (Mimi: 24 kHz).
    pub fn sample_rate(&self) -> u32 {
        MIMI_RATE
    }

    /// One-shot TTS from a text prompt.
    ///
    /// Currently returns an unimplemented error — the depth-multiplexed Kyutai
    /// TTS architecture (per-step DepFormer weights, cross-attention speaker
    /// conditioning, demuxed second stream) is not yet wired into the eager
    /// CPU backbone. Use the upstream Kyutai `moshi` pipeline against the
    /// same `model_dir` in the meantime.
    pub fn generate(&mut self, prompt: &str, cfg: &GenerationConfig) -> Result<GenerationResult> {
        let _ = (prompt, cfg);
        bail!(
            "Kyutai TTS generation is not yet implemented in rlx-kyutai-tts.\n\
             The weights at {} are valid and can be loaded by the upstream Kyutai\n\
             `moshi` (Python) pipeline. Rust generation will follow once the\n\
             cross-attention + per-step DepFormer path lands.",
            self.model_dir.display()
        )
    }
}
