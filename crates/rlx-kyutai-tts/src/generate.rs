//! Kyutai TTS autoregressive generation loop (DSM + state machine).

use crate::config::KyutaiTtsConfig;
use crate::delays::StreamLayout;
use crate::model::KyutaiLm;
use crate::sampling::StreamSampler;
use crate::state_machine::{StateMachine, TtsState, script_to_entries};
use crate::tokenizer::KyutaiTokenizer;
use anyhow::{Result, ensure};
use ndarray::Array2;

pub const UNGENERATED: u32 = u32::MAX;

/// Extra steps after `end_step` before stopping (matches Moshi TTS `final_padding`).
const FINAL_PADDING: usize = 4;

#[derive(Clone)]
pub struct GenerateConfig {
    pub max_steps: usize,
    pub n_q: usize,
    pub cfg_alpha: f32,
    pub text_temperature: f32,
    pub audio_temperature: f32,
    pub seed: u64,
}

impl GenerateConfig {
    pub fn from_session(
        cfg: &crate::session::GenerationConfig,
        model_cfg: &KyutaiTtsConfig,
    ) -> Self {
        Self {
            max_steps: cfg.max_steps,
            n_q: cfg.mimi_codebooks.min(model_cfg.dep_q),
            cfg_alpha: cfg.cfg_alpha,
            text_temperature: cfg.text_temperature as f32,
            audio_temperature: cfg.audio_temperature as f32,
            seed: cfg.seed,
        }
    }
}

pub struct GenerateState {
    machine: StateMachine,
    tts_state: TtsState,
    layout: StreamLayout,
    stream_delay: usize,
    audio_tokens: Vec<Vec<u32>>,
    /// Per-step LM output frames (after per-codebook delay alignment), matching Moshi `LMGen`.
    lm_frames: Vec<Vec<u32>>,
    text_tokens: Vec<u32>,
    sampler: StreamSampler,
    cfg: GenerateConfig,
    step_idx: usize,
    text_input: u32,
}

impl GenerateState {
    pub fn new(
        model_cfg: &KyutaiTtsConfig,
        tokenizer: &KyutaiTokenizer,
        prompt: &str,
        cfg: GenerateConfig,
    ) -> Result<Self> {
        let entries = script_to_entries(tokenizer, prompt)?;
        let layout = StreamLayout::from_config(model_cfg);
        let stream_delay = model_cfg.audio_delay_frames();
        let machine = StateMachine::for_config(
            model_cfg.text_card as u32,
            model_cfg.tts_config.second_stream_ahead,
        );
        let tts_state = machine.new_state(entries);
        let buf = layout.total_steps_for(cfg.max_steps) + stream_delay + FINAL_PADDING + 4;
        Ok(Self {
            machine,
            tts_state,
            layout,
            stream_delay,
            audio_tokens: vec![vec![UNGENERATED; cfg.n_q]; buf],
            lm_frames: Vec::new(),
            text_tokens: vec![UNGENERATED; buf],
            sampler: StreamSampler::new(cfg.seed, cfg.text_temperature, cfg.audio_temperature),
            cfg,
            step_idx: 0,
            text_input: model_cfg.text_card as u32,
        })
    }

    pub fn step_idx(&self) -> usize {
        self.step_idx
    }

    pub fn multiplex_text_input(&self) -> u32 {
        self.text_input
    }

    pub fn raw_lm_frames(&self) -> &[Vec<u32>] {
        &self.lm_frames
    }

    pub fn end_step(&self) -> Option<usize> {
        self.tts_state.end_step
    }

    pub fn transcript(&self) -> &[(String, usize)] {
        &self.tts_state.transcript
    }

    pub fn text_tokens(&self) -> &[u32] {
        let n = self.step_idx.min(self.text_tokens.len());
        &self.text_tokens[..n]
    }

    fn delayed_audio(&self, zero: u32, pad: u32) -> Vec<u32> {
        let mut out = Vec::with_capacity(self.cfg.n_q);
        for cb in 0..self.cfg.n_q {
            let cb_delay = self.layout.audio_delay(cb) as usize;
            // Moshi `is_init`: `offset <= delays[q]` → initial audio token is `card`.
            if self.step_idx <= cb_delay {
                out.push(pad);
                continue;
            }
            if self.step_idx < self.stream_delay {
                // Text→audio stream delay: depformer not running yet.
                out.push(zero);
                continue;
            }
            let tok = self.audio_tokens[self.step_idx][cb];
            out.push(if tok == UNGENERATED { zero } else { tok });
        }
        out
    }

    pub fn step(&mut self, model: &mut impl KyutaiLm) -> Result<()> {
        ensure!(self.step_idx < self.cfg.max_steps, "max_steps reached");

        let zero = model.lm_zero_token();
        let pad = model.audio_pad_token();
        let delayed = self.delayed_audio(zero, pad);
        let text_in = self.text_input;
        let (sampled, hidden) = model.forward_step(text_in, &delayed, &mut self.sampler)?;

        // Moshi `on_text_hook`: state machine runs before DepFormer.
        let depformer_text = self
            .machine
            .process(self.step_idx, &mut self.tts_state, sampled);
        self.text_input = depformer_text;

        if std::env::var_os("RLX_KYUTAI_TTS_TRACE").is_some() && self.step_idx < 35 {
            eprintln!(
                "trace step={} end={:?} text_in={} sampled={} depformer_text={} lm_frames={}",
                self.step_idx,
                self.tts_state.end_step,
                text_in,
                sampled,
                depformer_text,
                self.lm_frames.len(),
            );
        }

        let audio_frame = if self.step_idx < self.stream_delay {
            vec![UNGENERATED; self.cfg.n_q]
        } else {
            model.depformer_step(&hidden, depformer_text, &mut self.sampler)?
        };

        if std::env::var_os("RLX_KYUTAI_TTS_TRACE").is_some()
            && self.step_idx >= self.stream_delay
            && self.step_idx < self.stream_delay + 8
        {
            eprintln!(
                "  audio step={} cb0..7 {:?}",
                self.step_idx,
                &audio_frame[..8.min(audio_frame.len())]
            );
        }

        // Moshi `tts.generate` `_on_audio_hook`: zero audio codebooks until
        // `offset >= delays[q] + delay_steps` before writing to the LM cache.
        let offset = self.step_idx;
        self.step_idx += 1;
        self.text_tokens[self.step_idx] = sampled;
        for (cb, &tok) in audio_frame.iter().enumerate().take(self.cfg.n_q) {
            let cb_delay = self.layout.audio_delay(cb) as usize;
            let stored = if offset < cb_delay + self.stream_delay {
                zero
            } else if tok == UNGENERATED {
                zero
            } else {
                tok
            };
            self.audio_tokens[self.step_idx][cb] = stored;
        }

        let max_delay = self.layout.max_delay.max(0) as usize;
        if self.step_idx > max_delay {
            self.lm_frames.push(self.gather_lm_frame(self.step_idx));
        }
        Ok(())
    }

    /// Gather one delay-aligned audio frame at generation offset `offset` (post-increment).
    fn gather_lm_frame(&self, offset: usize) -> Vec<u32> {
        let max_delay = self.layout.max_delay.max(0) as usize;
        let pad = self.layout.audio_pad;
        let mut frame = Vec::with_capacity(self.cfg.n_q);
        for cb in 0..self.cfg.n_q {
            let src = offset
                .saturating_sub(max_delay)
                .saturating_add(self.layout.audio_delay(cb) as usize);
            let tok = self
                .audio_tokens
                .get(src)
                .and_then(|row| row.get(cb))
                .copied()
                .unwrap_or(UNGENERATED);
            frame.push(if tok == UNGENERATED || tok >= pad || tok == u32::MAX {
                0
            } else {
                tok
            });
        }
        frame
    }

    /// Audio frame ready for Mimi decode (past stream + per-codebook delay).
    pub fn last_audio_frame(&self) -> Option<Vec<u32>> {
        self.collect_audio_frames().last().cloned()
    }

    pub fn collect_audio_frames(&self) -> Vec<Vec<u32>> {
        self.lm_frames
            .iter()
            .skip(self.stream_delay)
            .map(|frame| self.sanitize_frame(frame))
            .collect()
    }

    /// Moshi `simple_generate` frame window: skip stream delay + 2 lead-in frames, cap at `end_step`.
    pub fn trim_for_mimi(
        lm_frames: &[Vec<u32>],
        end_step: Option<usize>,
        stream_delay: usize,
        n_q: usize,
        audio_pad: u32,
    ) -> Vec<Vec<u32>> {
        const LEAD_TRIM: usize = 2;
        let sanitize = |frame: &[u32]| -> Vec<u32> {
            frame
                .iter()
                .take(n_q)
                .map(|&t| {
                    if t == UNGENERATED || t >= audio_pad || t == u32::MAX {
                        0
                    } else {
                        t
                    }
                })
                .collect()
        };
        let mut out: Vec<_> = lm_frames
            .iter()
            .skip(stream_delay + LEAD_TRIM)
            .map(|f| sanitize(f))
            .collect();
        if let Some(end) = end_step {
            if end < out.len() {
                out.truncate(end);
            }
        }
        out
    }

    fn sanitize_frame(&self, frame: &[u32]) -> Vec<u32> {
        let pad = self.layout.audio_pad;
        frame
            .iter()
            .take(self.cfg.n_q)
            .map(|&t| {
                if t == UNGENERATED || t >= pad || t == u32::MAX {
                    0
                } else {
                    t
                }
            })
            .collect()
    }
}

/// Run generation to completion and return Mimi-ready code frames + `end_step`.
pub fn generate_codes(
    model: &mut impl KyutaiLm,
    tokenizer: &KyutaiTokenizer,
    prompt: &str,
    cfg: GenerateConfig,
    speaker: Option<&Array2<f32>>,
) -> Result<(Vec<Vec<u32>>, Option<usize>)> {
    model.reset_state();
    model.set_generation_conditions(cfg.cfg_alpha, speaker)?;
    let mut state = GenerateState::new(model.config(), tokenizer, prompt, cfg)?;
    let max_steps = state.cfg.max_steps;
    while state.step_idx() < max_steps {
        if let Some(e) = state.tts_state.end_step {
            if state.step_idx() >= e + state.stream_delay + FINAL_PADDING {
                break;
            }
        }
        state.step(model)?;
    }
    Ok((
        GenerateState::trim_for_mimi(
            &state.lm_frames,
            state.tts_state.end_step,
            state.stream_delay,
            state.cfg.n_q,
            state.layout.audio_pad,
        ),
        state.tts_state.end_step,
    ))
}
