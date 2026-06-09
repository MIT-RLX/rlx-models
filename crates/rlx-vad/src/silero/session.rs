// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

use anyhow::{Result, bail};

use super::model::{LstmState, SileroScratch, forward_frame};
use super::weights::{SileroWeights, context_samples, frame_samples};
use crate::SampleRate;

#[derive(Debug, Clone)]
pub struct SileroConfig {
    pub sample_rate: SampleRate,
}

impl Default for SileroConfig {
    fn default() -> Self {
        Self {
            sample_rate: SampleRate::Hz16000,
        }
    }
}

pub struct SileroSession {
    cfg: SileroConfig,
    weights: SileroWeights,
    context: Vec<f32>,
    frame_input: Vec<f32>,
    frame_buf: Vec<f32>,
    state: LstmState,
    scratch: SileroScratch,
}

impl SileroSession {
    pub fn new(weights: SileroWeights, cfg: SileroConfig) -> Self {
        let ctx = context_samples(cfg.sample_rate);
        let frame = frame_samples(cfg.sample_rate);
        Self {
            cfg,
            weights,
            context: vec![0.0; ctx],
            frame_input: vec![0.0; ctx + frame],
            frame_buf: vec![0.0; frame],
            state: LstmState::default(),
            scratch: SileroScratch::for_max_frame(ctx + frame),
        }
    }

    pub fn reset(&mut self) {
        self.context.fill(0.0);
        self.state = LstmState::default();
    }

    pub fn frame_samples(&self) -> usize {
        frame_samples(self.cfg.sample_rate)
    }

    pub fn context_samples(&self) -> usize {
        context_samples(self.cfg.sample_rate)
    }

    /// Score one frame of normalized f32 PCM (exactly `frame_samples()` long).
    pub fn predict_frame(&mut self, frame: &[f32]) -> Result<f32> {
        let expect = self.frame_samples();
        if frame.len() != expect {
            bail!("expected {expect} samples, got {}", frame.len());
        }
        let ctx = self.context.len();
        self.frame_input[..ctx].copy_from_slice(&self.context);
        self.frame_input[ctx..ctx + expect].copy_from_slice(frame);
        let prob = forward_frame(
            &self.weights,
            &self.frame_input[..ctx + expect],
            &mut self.state,
            &mut self.scratch,
        );
        self.context.copy_from_slice(&frame[frame.len() - ctx..]);
        Ok(prob)
    }

    /// Pad a short tail chunk to `frame_samples()` and score it.
    pub fn predict_frame_padded(&mut self, chunk: &[f32]) -> Result<f32> {
        let expect = self.frame_samples();
        let frame_owned;
        let frame: &[f32] = if chunk.len() == expect {
            chunk
        } else {
            self.frame_buf.fill(0.0);
            self.frame_buf[..chunk.len()].copy_from_slice(chunk);
            frame_owned = self.frame_buf.clone();
            &frame_owned
        };
        self.predict_frame(frame)
    }

    /// Score one frame of i16 PCM.
    pub fn predict_i16(&mut self, frame: &[i16]) -> Result<f32> {
        let expect = self.frame_samples();
        let mut f32s = vec![0.0f32; expect];
        for (dst, &s) in f32s.iter_mut().zip(frame.iter().take(expect)) {
            *dst = s as f32 / i16::MAX as f32;
        }
        self.predict_frame(&f32s)
    }
}
