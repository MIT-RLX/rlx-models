//! High-level inference session for Kyutai TTS.
//!
//! End-to-end synthesis: DSM LM (RLX backbone on GPU) → Mimi decode (same device).

use crate::backend::{KyutaiTtsBackend, resolve_lm_device};
use crate::checkpoint::{KyutaiTtsCheckpoint, KyutaiTtsVoice};
use crate::config::KyutaiTtsConfig;
use crate::download::{
    default_mimi_dir, default_voices_dir, ensure_voice_embedding, ensure_weights_checkpoint,
    tokenizer_path,
};
use crate::generate::GenerateConfig;
use crate::model::load_voice_speaker_wavs;
use crate::tokenizer::KyutaiTokenizer;
use anyhow::{Result, bail};
use ndarray::Array2;
use rlx_mimi::{MimiCodec, MimiCodes, SAMPLE_RATE as MIMI_RATE};
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
        fn env_f64(key: &str, default: f64) -> f64 {
            std::env::var(key)
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(default)
        }
        fn env_u64(key: &str, default: u64) -> u64 {
            std::env::var(key)
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(default)
        }
        Self {
            max_steps: 100,
            text_temperature: env_f64("RLX_KYUTAI_TTS_TEXT_TEMP", 0.6),
            audio_temperature: env_f64("RLX_KYUTAI_TTS_AUDIO_TEMP", 0.6),
            cfg_alpha: 2.0,
            seed: env_u64("RLX_KYUTAI_TTS_SEED", 42),
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
    voices_dir: PathBuf,
    device: Device,
    voice: KyutaiTtsVoice,
    backend: Option<KyutaiTtsBackend>,
    mimi: Option<MimiCodec>,
    tokenizer: Option<KyutaiTokenizer>,
    speaker: Option<Array2<f32>>,
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
        let device = resolve_lm_device(device);
        Ok(Self {
            cfg,
            checkpoint,
            model_dir,
            mimi_dir,
            voices_dir: default_voices_dir(),
            device,
            voice: KyutaiTtsVoice::unconditional(),
            backend: None,
            mimi: None,
            tokenizer: None,
            speaker: None,
        })
    }

    /// Override the voice-embedding cache directory (`kyutai/tts-voices`).
    pub fn set_voices_dir(&mut self, dir: impl AsRef<Path>) {
        self.voices_dir = dir.as_ref().to_path_buf();
        self.speaker = None;
    }

    /// Open with default cache dirs.
    pub fn open_default() -> Result<Self> {
        Self::open(
            crate::download::default_kyutai_tts_dir(),
            default_mimi_dir(),
        )
    }

    fn ensure_loaded(&mut self) -> Result<()> {
        if self.backend.is_none() {
            self.backend = Some(KyutaiTtsBackend::open(
                &self.model_dir,
                self.cfg.clone(),
                self.device,
            )?);
        }
        if self.tokenizer.is_none() {
            let sp = tokenizer_path(&self.model_dir);
            self.tokenizer = Some(KyutaiTokenizer::load(sp)?);
        }
        if self.mimi.is_none() {
            self.mimi = Some(MimiCodec::open_on(&self.mimi_dir, self.device)?);
        }
        Ok(())
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
        self.speaker = None;
    }

    /// Load a speaker embedding from `kyutai/tts-voices` into the session.
    pub fn load_voice_embedding(&mut self, path: impl AsRef<Path>) -> Result<()> {
        self.speaker = Some(load_voice_speaker_wavs(path.as_ref())?);
        Ok(())
    }

    fn ensure_speaker(&mut self) -> Result<()> {
        if self.speaker.is_some() || self.voice.is_unconditional() {
            return Ok(());
        }
        let path = ensure_voice_embedding(&self.voices_dir, self.checkpoint, &self.voice.name)?;
        self.speaker = Some(load_voice_speaker_wavs(&path)?);
        Ok(())
    }

    /// Sample rate of the synthesised audio (Mimi: 24 kHz).
    pub fn sample_rate(&self) -> u32 {
        MIMI_RATE
    }

    /// One-shot TTS from a text prompt.
    pub fn generate(&mut self, prompt: &str, cfg: &GenerationConfig) -> Result<GenerationResult> {
        self.ensure_loaded()?;
        self.ensure_speaker()?;
        let gen_cfg = GenerateConfig::from_session(cfg, &self.cfg);
        let tokenizer = self.tokenizer.as_ref().unwrap();
        let backend = self.backend.as_mut().unwrap();
        let frames = backend.synthesize_codes(tokenizer, prompt, gen_cfg, self.speaker.as_ref())?;
        if frames.is_empty() {
            bail!("generation produced no audio frames (try a longer --max-steps)");
        }
        let codes = MimiCodes {
            frames: frames.clone(),
            num_quantizers: cfg.mimi_codebooks.min(self.cfg.dep_q),
        };
        let mimi = self.mimi.take().unwrap();
        let mut samples = mimi.decode_codes(&codes)?;
        self.mimi = Some(mimi);
        normalize_peak(&mut samples, 0.95);
        Ok(GenerationResult {
            samples,
            sample_rate: MIMI_RATE,
            text_tokens: vec![],
            audio_frames: frames,
        })
    }
}

/// Scale quiet PCM to a target peak (TTS Mimi output is often low without mastering).
fn normalize_peak(samples: &mut [f32], target_peak: f32) {
    let peak = samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    if peak > 1e-7 && peak < target_peak {
        let scale = target_peak / peak;
        for s in samples {
            *s *= scale;
        }
    }
}
