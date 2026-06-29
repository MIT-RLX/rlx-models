// RLX models — language-model evaluation.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! Evaluation harness: perplexity over a corpus and lm-eval-style
//! multiple-choice scoring, built on teacher-forced next-token log-probs.
//!
//! Generic over [`LmLogprobs`] so any model exposing one full-sequence
//! forward can be evaluated. The whole harness is **host-side** — it consumes
//! the `Vec<f32>` log-probs the model produces on any backend.

pub mod multiple_choice;
pub mod perplexity;

pub use multiple_choice::{McItem, McResult, score_mc};
pub use perplexity::{PerplexityConfig, perplexity};

/// A model that can produce teacher-forced next-token log-probabilities.
///
/// `sequence_logprobs(tokens)` returns `tokens.len() - 1` values where index
/// `i` is `log P(tokens[i+1] | tokens[..=i])` (empty for a 1-token input).
pub trait LmLogprobs {
    fn sequence_logprobs(&mut self, tokens: &[u32]) -> anyhow::Result<Vec<f32>>;
}

impl LmLogprobs for rlx_qwen3::Qwen3Generator {
    fn sequence_logprobs(&mut self, tokens: &[u32]) -> anyhow::Result<Vec<f32>> {
        rlx_qwen3::Qwen3Generator::sequence_logprobs(self, tokens)
    }
}
