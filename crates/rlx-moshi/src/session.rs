use crate::backend::{MoshiLm, resolve_lm_device};
use crate::checkpoint::MoshiCheckpoint;
use crate::config::{GenerateConfig, MoshiVariant};
use crate::download::{default_mimi_dir, ensure_weights_checkpoint, tokenizer_path};
use crate::sampling::LogitsProcessor;
use crate::tokenizer::MoshiTokenizer;
use anyhow::{Result, ensure};
use rlx_mimi::{MimiCodec, MimiCodes, SAMPLE_RATE as MIMI_RATE};
use rlx_runtime::Device;
use std::path::{Path, PathBuf};

/// Sampling overrides for Moshi generation.
#[derive(Debug, Clone)]
pub struct GenerationConfig {
    pub max_steps: usize,
    pub text_temperature: f64,
    pub text_top_k: usize,
    pub audio_temperature: f64,
    pub audio_top_k: usize,
    pub text_seed: u64,
    pub audio_seed: u64,
    pub mimi_codebooks: usize,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_steps: 25,
            text_temperature: 0.8,
            text_top_k: 250,
            audio_temperature: 0.8,
            audio_top_k: 250,
            text_seed: 42,
            audio_seed: 43,
            mimi_codebooks: 8,
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
    pub transcript: String,
}

/// Decomposed session for streaming worker ownership.
pub struct MoshiSessionParts {
    pub lm: MoshiLm,
    pub mimi: MimiCodec,
    pub tokenizer: MoshiTokenizer,
    pub gen_cfg: GenerateConfig,
    pub device: Device,
    pub variant: MoshiVariant,
    pub checkpoint: MoshiCheckpoint,
    pub moshi_dir: PathBuf,
}

/// Moshi session — LM + Mimi codec + tokenizer.
pub struct MoshiSession {
    variant: MoshiVariant,
    lm: MoshiLm,
    mimi: MimiCodec,
    tokenizer: MoshiTokenizer,
    gen_cfg: GenerateConfig,
    moshi_dir: PathBuf,
    device: Device,
    checkpoint: MoshiCheckpoint,
}

impl MoshiSession {
    pub fn open(
        moshi_dir: impl AsRef<Path>,
        mimi_dir: impl AsRef<Path>,
        variant: MoshiVariant,
    ) -> Result<Self> {
        Self::open_on(moshi_dir, mimi_dir, variant, Device::Cpu)
    }

    pub fn open_on(
        moshi_dir: impl AsRef<Path>,
        mimi_dir: impl AsRef<Path>,
        variant: MoshiVariant,
        device: Device,
    ) -> Result<Self> {
        Self::open_with_checkpoint(
            moshi_dir,
            mimi_dir,
            variant,
            device,
            MoshiCheckpoint::from_env_or_default(),
        )
    }

    pub fn open_with_checkpoint(
        moshi_dir: impl AsRef<Path>,
        mimi_dir: impl AsRef<Path>,
        variant: MoshiVariant,
        device: Device,
        checkpoint: MoshiCheckpoint,
    ) -> Result<Self> {
        let moshi_dir = moshi_dir.as_ref().to_path_buf();
        ensure_weights_checkpoint(&moshi_dir, variant, checkpoint)?;
        rlx_mimi::ensure_weights(mimi_dir.as_ref())?;
        let device = resolve_lm_device(device, checkpoint);
        let lm = MoshiLm::open(&moshi_dir, variant, checkpoint, device)?;
        let mimi =
            MimiCodec::open_on_with_moshi(mimi_dir.as_ref(), Some(&moshi_dir), device, Some(8))?;
        let tokenizer = MoshiTokenizer::open(tokenizer_path(&moshi_dir))?;
        Ok(Self {
            variant,
            lm,
            mimi,
            tokenizer,
            gen_cfg: variant.generate_config(),
            moshi_dir,
            device,
            checkpoint,
        })
    }

    pub fn open_default(variant: MoshiVariant) -> Result<Self> {
        Self::open_default_on(variant, Device::Cpu)
    }

    pub fn open_default_on(variant: MoshiVariant, device: Device) -> Result<Self> {
        Self::open_on(
            crate::download::default_moshi_dir(),
            default_mimi_dir(),
            variant,
            device,
        )
    }

    pub fn variant(&self) -> MoshiVariant {
        self.variant
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn checkpoint(&self) -> MoshiCheckpoint {
        self.checkpoint
    }

    pub fn moshi_dir(&self) -> &Path {
        &self.moshi_dir
    }

    pub fn gen_cfg_internal(&self) -> &GenerateConfig {
        &self.gen_cfg
    }

    pub fn into_parts(self) -> Result<MoshiSessionParts> {
        Ok(MoshiSessionParts {
            lm: self.lm,
            mimi: self.mimi,
            tokenizer: self.tokenizer,
            gen_cfg: self.gen_cfg,
            device: self.device,
            variant: self.variant,
            checkpoint: self.checkpoint,
            moshi_dir: self.moshi_dir,
        })
    }

    /// One-way TTS from a text prompt (blank user audio).
    pub fn generate_one_way(
        &mut self,
        prompt: &str,
        cfg: &GenerationConfig,
    ) -> Result<GenerationResult> {
        ensure!(
            self.gen_cfg.input_audio_codebooks == 0,
            "generate_one_way requires a one-way variant (MoshikoOneWay or MoshikaOneWay)"
        );
        self.run_generation(prompt, cfg)
    }

    /// Full-duplex: encode user WAV with Mimi, condition generation, decode Moshi reply.
    pub fn generate_duplex(
        &mut self,
        user_wav: impl AsRef<Path>,
        cfg: &GenerationConfig,
    ) -> Result<GenerationResult> {
        ensure!(
            self.gen_cfg.input_audio_codebooks > 0,
            "generate_duplex requires a full-duplex variant (Moshiko or Moshika)"
        );
        let user_codes = self
            .mimi
            .encode_wav(user_wav.as_ref(), Some(cfg.mimi_codebooks))?;
        let num_frames = user_codes.num_frames().min(cfg.max_steps);
        self.run_generation_with_user("", &user_codes, num_frames, cfg)
    }

    fn run_generation(&mut self, prompt: &str, cfg: &GenerationConfig) -> Result<GenerationResult> {
        let text_frames = self.tokenizer.prompt_frame_tokens(prompt, cfg.max_steps)?;
        let text_lp = LogitsProcessor::new(cfg.text_temperature, cfg.text_top_k, cfg.text_seed);
        let audio_lp = LogitsProcessor::new(cfg.audio_temperature, cfg.audio_top_k, cfg.audio_seed);
        let mut state = self.lm.new_gen_state(
            cfg.max_steps,
            text_lp.clone(),
            audio_lp.clone(),
            self.gen_cfg.clone(),
        )?;
        state.reset(&mut self.lm, cfg.max_steps, text_lp, audio_lp)?;
        let empty_user: Vec<u32> = vec![];
        let mut decoded_pcm = Vec::new();
        let mut audio_trace = Vec::new();
        for step in 0..cfg.max_steps {
            let tt = text_frames[step];
            state.step(&mut self.lm, tt, &empty_user)?;
            if let Some(frame) = state.last_audio_frame() {
                audio_trace.push(frame.clone());
                let pcm = self.decode_frame(&frame, cfg.mimi_codebooks)?;
                decoded_pcm.extend(pcm);
            }
        }
        let text_tokens = state.text_tokens().to_vec();
        let transcript = self.tokens_to_text(&text_tokens)?;
        Ok(GenerationResult {
            samples: decoded_pcm,
            sample_rate: MIMI_RATE,
            text_tokens,
            audio_frames: audio_trace,
            transcript,
        })
    }

    fn run_generation_with_user(
        &mut self,
        prompt: &str,
        user_codes: &MimiCodes,
        num_frames: usize,
        cfg: &GenerationConfig,
    ) -> Result<GenerationResult> {
        let text_frames = self.tokenizer.prompt_frame_tokens(prompt, num_frames)?;
        let text_lp = LogitsProcessor::new(cfg.text_temperature, cfg.text_top_k, cfg.text_seed);
        let audio_lp = LogitsProcessor::new(cfg.audio_temperature, cfg.audio_top_k, cfg.audio_seed);
        let mut state = self.lm.new_gen_state(
            num_frames,
            text_lp.clone(),
            audio_lp.clone(),
            self.gen_cfg.clone(),
        )?;
        state.reset(&mut self.lm, num_frames, text_lp, audio_lp)?;
        let pad = state.config().audio_pad_token();
        let mut decoded_pcm = Vec::new();
        let mut audio_trace = Vec::new();
        for step in 0..num_frames {
            let user_frame: Vec<u32> = if step < user_codes.frames.len() {
                user_codes.frames[step].clone()
            } else {
                vec![pad; cfg.mimi_codebooks]
            };
            state.step(&mut self.lm, text_frames[step], &user_frame)?;
            if let Some(frame) = state.last_audio_frame() {
                audio_trace.push(frame.clone());
                let pcm = self.decode_frame(&frame, cfg.mimi_codebooks)?;
                decoded_pcm.extend(pcm);
            }
        }
        let text_tokens = state.text_tokens().to_vec();
        let transcript = self.tokens_to_text(&text_tokens)?;
        Ok(GenerationResult {
            samples: decoded_pcm,
            sample_rate: MIMI_RATE,
            text_tokens,
            audio_frames: audio_trace,
            transcript,
        })
    }

    fn decode_frame(&mut self, frame: &[u32], nq: usize) -> Result<Vec<f32>> {
        let codes = MimiCodes {
            frames: vec![frame.to_vec()],
            num_quantizers: nq,
        };
        self.mimi.decode_codes(&codes)
    }

    fn tokens_to_text(&self, tokens: &[u32]) -> Result<String> {
        let mut out = String::new();
        let cfg = &self.gen_cfg;
        let mut prev = cfg.text_pad_token;
        for &t in tokens {
            if t != cfg.text_start_token
                && t != cfg.text_pad_token
                && t != cfg.text_eop_token
                && prev == cfg.text_start_token
            {
                out.push_str(&self.tokenizer.decode_piece(t)?);
            }
            prev = t;
        }
        Ok(out)
    }
}
