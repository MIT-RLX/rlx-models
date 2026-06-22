use crate::backend::{MoshiGenState, MoshiLm};
use crate::config::GenerateConfig;
use crate::sampling::LogitsProcessor;
use crate::session::{GenerationConfig, MoshiSession};
use crate::tokenizer::MoshiTokenizer;
use anyhow::{Result, ensure};
use rlx_mimi::{MimiCodec, MimiCodes};
use rlx_runtime::Device;

/// One streaming step's output.
#[derive(Debug, Clone)]
pub struct StreamStepOutput {
    pub step: usize,
    pub text_token: u32,
    pub moshi_pcm: Vec<f32>,
    pub transcript_delta: Option<String>,
}

/// Incremental full-duplex engine — feed 24 kHz PCM, receive Moshi PCM chunks.
pub struct DuplexStreamEngine {
    lm: MoshiLm,
    mimi: MimiCodec,
    tokenizer: MoshiTokenizer,
    state: MoshiGenState,
    gen_cfg: GenerateConfig,
    run_cfg: GenerationConfig,
    text_frames: Vec<u32>,
    pcm_buf: Vec<f32>,
    frame_samples: usize,
    frame_idx: usize,
    finished: bool,
    device: Device,
}

impl DuplexStreamEngine {
    pub fn from_session(
        session: MoshiSession,
        prompt: &str,
        run_cfg: &GenerationConfig,
    ) -> Result<Self> {
        let parts = session.into_parts()?;
        ensure!(
            parts.gen_cfg.input_audio_codebooks > 0,
            "DuplexStreamEngine requires a full-duplex variant (Moshiko or Moshika)"
        );
        let max = run_cfg.max_steps;
        let text_frames = parts.tokenizer.prompt_frame_tokens(prompt, max)?;
        let text_lp = LogitsProcessor::new(
            run_cfg.text_temperature,
            run_cfg.text_top_k,
            run_cfg.text_seed,
        );
        let audio_lp = LogitsProcessor::new(
            run_cfg.audio_temperature,
            run_cfg.audio_top_k,
            run_cfg.audio_seed,
        );
        let mut state = parts
            .lm
            .new_gen_state(max, text_lp, audio_lp, parts.gen_cfg.clone())?;
        let mut lm = parts.lm;
        state.reset(
            &mut lm,
            max,
            LogitsProcessor::new(
                run_cfg.text_temperature,
                run_cfg.text_top_k,
                run_cfg.text_seed,
            ),
            LogitsProcessor::new(
                run_cfg.audio_temperature,
                run_cfg.audio_top_k,
                run_cfg.audio_seed,
            ),
        )?;
        let frame_samples = parts.mimi.config().samples_per_codec_frame();
        Ok(Self {
            lm,
            mimi: parts.mimi,
            tokenizer: parts.tokenizer,
            state,
            gen_cfg: parts.gen_cfg,
            run_cfg: run_cfg.clone(),
            text_frames,
            pcm_buf: Vec::new(),
            frame_samples,
            frame_idx: 0,
            finished: false,
            device: parts.device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn frame_samples(&self) -> usize {
        self.frame_samples
    }

    pub fn steps_done(&self) -> usize {
        self.frame_idx
    }

    /// Append PCM (mono f32 @ 24 kHz). Returns zero or more completed step outputs.
    pub fn feed_pcm(&mut self, pcm: &[f32]) -> Result<Vec<StreamStepOutput>> {
        ensure!(!self.finished, "stream already finished");
        self.pcm_buf.extend_from_slice(pcm);
        let mut outs = Vec::new();
        while self.pcm_buf.len() >= self.frame_samples && self.frame_idx < self.run_cfg.max_steps {
            let frame_pcm: Vec<f32> = self.pcm_buf.drain(..self.frame_samples).collect();
            outs.push(self.step_frame(&frame_pcm)?);
        }
        Ok(outs)
    }

    /// Pad tail, run remaining buffered audio, mark finished.
    pub fn finish(&mut self) -> Result<Vec<StreamStepOutput>> {
        if self.finished {
            return Ok(Vec::new());
        }
        let mut outs = Vec::new();
        if !self.pcm_buf.is_empty() && self.frame_idx < self.run_cfg.max_steps {
            let mut frame_pcm = std::mem::take(&mut self.pcm_buf);
            frame_pcm.resize(self.frame_samples, 0.0);
            outs.push(self.step_frame(&frame_pcm)?);
        }
        self.finished = true;
        Ok(outs)
    }

    pub fn collected_text_tokens(&self) -> Vec<u32> {
        self.state.text_tokens().to_vec()
    }

    fn step_frame(&mut self, frame_pcm: &[f32]) -> Result<StreamStepOutput> {
        let user_codes = self
            .mimi
            .encode_pcm(frame_pcm, Some(self.run_cfg.mimi_codebooks))?;
        ensure!(
            user_codes.num_frames() >= 1,
            "mimi encode produced no frames"
        );
        let user_frame = user_codes.frames[0].clone();
        let text_tok = self.text_frames[self.frame_idx];
        let sampled = self.state.step(&mut self.lm, text_tok, &user_frame)?;
        let mut moshi_pcm = Vec::new();
        if let Some(frame) = self.state.last_audio_frame() {
            moshi_pcm = self.decode_frame(&frame)?;
        }
        let transcript_delta = self.token_delta(sampled);
        let step = self.frame_idx;
        self.frame_idx += 1;
        Ok(StreamStepOutput {
            step,
            text_token: sampled,
            moshi_pcm,
            transcript_delta,
        })
    }

    fn decode_frame(&mut self, frame: &[u32]) -> Result<Vec<f32>> {
        let codes = MimiCodes {
            frames: vec![frame.to_vec()],
            num_quantizers: self.run_cfg.mimi_codebooks,
        };
        self.mimi.decode_codes(&codes)
    }

    fn token_delta(&self, token: u32) -> Option<String> {
        let g = &self.gen_cfg;
        if token == g.text_start_token || token == g.text_pad_token || token == g.text_eop_token {
            return None;
        }
        self.tokenizer.decode_piece(token).ok()
    }
}
