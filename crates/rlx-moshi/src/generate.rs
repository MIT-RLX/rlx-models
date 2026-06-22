use crate::config::GenerateConfig;
use crate::lm::LmModel;
use crate::sampling::LogitsProcessor;
use anyhow::{Result, ensure};

pub const UNGENERATED: u32 = u32::MAX;

/// Multistream autoregressive state (ported from kyutai `lm_generate_multistream`).
pub struct GenerateState {
    audio_tokens: Vec<Vec<u32>>,
    text_tokens: Vec<u32>,
    text_lp: LogitsProcessor,
    audio_lp: LogitsProcessor,
    step_idx: usize,
    forced_audio_tokens: ForcedAudioTokens,
    cfg: GenerateConfig,
}

#[derive(Debug, Clone)]
struct ForcedAudioTokens {
    delay: usize,
    pad: u32,
    pattern: Vec<usize>,
}

impl ForcedAudioTokens {
    fn new(delay: usize, pad: u32, pattern: &[usize]) -> Self {
        Self {
            delay,
            pad,
            pattern: pattern.to_vec(),
        }
    }

    fn forced_tokens(&self, step: usize) -> Vec<Option<u32>> {
        if step >= self.delay {
            return vec![None; self.pattern.len()];
        }
        self.pattern
            .iter()
            .map(|&v| if v == 0 { None } else { Some(self.pad) })
            .collect()
    }
}

impl GenerateState {
    pub fn new(
        max_steps: usize,
        text_lp: LogitsProcessor,
        audio_lp: LogitsProcessor,
        cfg: GenerateConfig,
    ) -> Self {
        let buf = max_steps + cfg.acoustic_delay;
        let audio_tokens = vec![vec![UNGENERATED; cfg.total_audio_codebooks()]; buf];
        let text_tokens = vec![UNGENERATED; buf];
        let forced = ForcedAudioTokens::new(cfg.acoustic_delay, cfg.audio_pad_token(), &[8, 8]);
        Self {
            audio_tokens,
            text_tokens,
            text_lp,
            audio_lp,
            step_idx: 0,
            forced_audio_tokens: forced,
            cfg,
        }
    }

    pub fn config(&self) -> &GenerateConfig {
        &self.cfg
    }

    pub fn step_idx(&self) -> usize {
        self.step_idx
    }

    pub fn text_tokens(&self) -> &[u32] {
        let n = self.step_idx.min(self.text_tokens.len());
        &self.text_tokens[..n]
    }

    /// Advance one 12.5 Hz frame. `input_audio` is user codebooks (empty for one-way).
    pub fn step(&mut self, lm: &mut LmModel, text_token: u32, input_audio: &[u32]) -> Result<u32> {
        ensure!(
            input_audio.len() == self.cfg.input_audio_codebooks,
            "expected {} user codebooks, got {}",
            self.cfg.input_audio_codebooks,
            input_audio.len()
        );
        for (ci, &t) in input_audio.iter().enumerate() {
            let idx = ci + self.cfg.generated_audio_codebooks;
            self.audio_tokens[self.step_idx][idx] = t;
        }
        let pad = self.cfg.audio_pad_token();
        let mut delayed = Vec::with_capacity(self.cfg.total_audio_codebooks());
        for codebook in 0..self.cfg.total_audio_codebooks() {
            let t = if codebook == 0 || codebook == self.cfg.generated_audio_codebooks {
                if self.step_idx == 0 {
                    pad
                } else {
                    self.audio_tokens[self.step_idx - 1][codebook]
                }
            } else if self.step_idx <= self.cfg.acoustic_delay {
                pad
            } else {
                self.audio_tokens[self.step_idx - self.cfg.acoustic_delay - 1][codebook]
            };
            ensure!(
                t != UNGENERATED,
                "internal: ungenerated audio at step {}",
                self.step_idx
            );
            delayed.push(Some(t));
        }
        let (text_logits, hidden) = lm.forward_step(Some(text_token), &delayed)?;
        let sampled_text = self.text_lp.sample(text_logits.view())?;
        self.text_tokens[self.step_idx] = sampled_text;
        let forced = self.forced_audio_tokens.forced_tokens(self.step_idx);
        if let Some(tokens) =
            lm.depformer_sample(&hidden, Some(sampled_text), &forced, &mut self.audio_lp)?
        {
            for (ci, &tok) in tokens.iter().enumerate() {
                let delay = if ci == 0 { 0 } else { self.cfg.acoustic_delay };
                let pos = self.step_idx.saturating_sub(delay);
                self.audio_tokens[pos][ci] = tok;
            }
        }
        self.step_idx += 1;
        Ok(sampled_text)
    }

    /// Moshi output codebooks ready for Mimi decode (past acoustic delay).
    pub fn last_audio_frame(&self) -> Option<Vec<u32>> {
        if self.step_idx <= self.cfg.acoustic_delay {
            return None;
        }
        let pos = self.step_idx - self.cfg.acoustic_delay - 1;
        let frame = &self.audio_tokens[pos];
        let pad = self.cfg.audio_pad_token();
        if frame[..self.cfg.generated_audio_codebooks]
            .iter()
            .any(|&t| t >= pad)
        {
            return None;
        }
        Some(frame[..self.cfg.generated_audio_codebooks].to_vec())
    }

    pub fn reset(&mut self, lm: &mut LmModel) {
        lm.reset_state();
        self.step_idx = 0;
        let buf = self.audio_tokens.len();
        let tc = self.cfg.total_audio_codebooks();
        self.audio_tokens = vec![vec![UNGENERATED; tc]; buf];
        self.text_tokens = vec![UNGENERATED; self.text_tokens.len()];
    }
}
