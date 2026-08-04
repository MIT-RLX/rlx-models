// RLX models — LLM benchmark harness.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! A weightless [`LmRunner`] for exercising the harness with no checkpoint on
//! disk — powers `--dry-run` and the unit tests.
//!
//! Its logits are a fixed, context-independent ramp `logit[i] = -scale * i`, so
//! smaller token ids always score higher. That makes every dimension
//! deterministic: greedy generation emits token `0` repeatedly, and
//! multiple-choice scoring prefers the choice with the smallest ids — the same
//! intuition as `rlx_eval`'s `IdModel`, but through the real `prefill_logits` /
//! `decode_logits` surface the harness drives.

use anyhow::Result;
use rlx_runtime::lm::LmRunner;

/// Deterministic, weightless runner. See module docs.
pub struct MockRunner {
    vocab: usize,
    scale: f32,
}

impl MockRunner {
    /// Runner with `vocab` tokens and the default ramp scale.
    pub fn new(vocab: usize) -> Self {
        Self {
            vocab: vocab.max(1),
            scale: 0.1,
        }
    }

    /// Override the ramp scale (steeper ⇒ sharper preference for small ids).
    pub fn with_scale(mut self, scale: f32) -> Self {
        self.scale = scale;
        self
    }

    fn logits(&self) -> Vec<f32> {
        (0..self.vocab).map(|i| -(i as f32) * self.scale).collect()
    }
}

impl LmRunner for MockRunner {
    fn family(&self) -> &'static str {
        "mock"
    }
    fn vocab_size(&self) -> usize {
        self.vocab
    }
    fn predict_logits(&mut self, _prompt_ids: &[u32]) -> Result<Vec<f32>> {
        Ok(self.logits())
    }
    fn prefill_logits(&mut self, _prompt_ids: &[u32]) -> Result<Vec<f32>> {
        Ok(self.logits())
    }
    fn decode_logits(&mut self, _token: u32) -> Result<Vec<f32>> {
        Ok(self.logits())
    }
    // `generate` uses the default trait impl (argmax over `predict_logits`),
    // which emits token 0 each step — fine for timing/plumbing.
}
