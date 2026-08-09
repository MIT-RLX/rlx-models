// RLX models — LLM benchmark harness.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! [`BenchModel`] — the model-agnostic thing every bench dimension drives.
//!
//! It wraps a boxed [`LmRunner`] (the seam every rlx LM crate already
//! implements) plus an optional tokenizer and EOS ids. The wrapper adds exactly
//! what the trait omits: text ⇄ token conversion and two scoring drivers built
//! on the trait's host-driven `prefill_logits` / `decode_logits`:
//!
//! - [`BenchModel::score_mc`] — lm-eval-style multiple-choice, `1` prefill +
//!   `k-1` decodes per choice. Same output as [`rlx_eval::score_mc`] but on the
//!   KV-cached path instead of a full teacher-forced forward.
//! - [`LmLogprobs`] impl — teacher-forced next-token log-probs for the whole
//!   sequence, so [`fn@rlx_eval::perplexity`] runs unchanged on any runner.

use anyhow::{Result, anyhow, bail};
use rlx_eval::{LmLogprobs, McItem, McResult};
use rlx_runtime::lm::LmRunner;
use tokenizers::Tokenizer;

use crate::metrics::log_softmax_at;

/// A benchmarkable language model: a runner + the text/tokenizer context the
/// bench needs around it.
pub struct BenchModel {
    /// The inference runner. Any crate implementing [`LmRunner`] plugs in.
    pub runner: Box<dyn LmRunner>,
    /// Tokenizer for text tasks (GSM8K prompts, MMLU rendering). `None` ⇒
    /// token-id-only mode (parity / raw-id speed still work).
    pub tokenizer: Option<Tokenizer>,
    /// End-of-sequence ids that stop generation.
    pub eos_ids: Vec<u32>,
    /// Display name for the leaderboard (e.g. `"qwen3-0.6b"`).
    pub name: String,
    /// Device label for the leaderboard (e.g. `"metal"`).
    pub device: String,
}

impl BenchModel {
    /// Assemble from already-built parts. Adapters (`adapters::qwen3`, …) call
    /// this after constructing their concrete runner.
    pub fn new(
        name: impl Into<String>,
        device: impl Into<String>,
        runner: Box<dyn LmRunner>,
        tokenizer: Option<Tokenizer>,
        eos_ids: Vec<u32>,
    ) -> Self {
        Self {
            runner,
            tokenizer,
            eos_ids,
            name: name.into(),
            device: device.into(),
        }
    }

    /// LM head vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.runner.vocab_size()
    }

    /// Encode text to ids. Errors if no tokenizer was supplied.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let tk = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| anyhow!("this benchmark needs a tokenizer; pass --tokenizer <file>"))?;
        let enc = tk
            .encode(text, false)
            .map_err(|e| anyhow!("tokenizer encode failed: {e}"))?;
        Ok(enc.get_ids().to_vec())
    }

    /// Decode ids to text (special tokens stripped). Errors if no tokenizer.
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        let tk = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| anyhow!("this benchmark needs a tokenizer; pass --tokenizer <file>"))?;
        tk.decode(ids, true)
            .map_err(|e| anyhow!("tokenizer decode failed: {e}"))
    }

    /// Greedy-generate up to `max_new` tokens after `prompt_ids`, stopping at
    /// any EOS id. Returns the produced ids (EOS trimmed) and their decoded
    /// text (empty string when no tokenizer is present).
    pub fn generate_text(
        &mut self,
        prompt_ids: &[u32],
        max_new: usize,
    ) -> Result<(Vec<u32>, String)> {
        let eos = self.eos_ids.clone();
        let mut produced: Vec<u32> = Vec::with_capacity(max_new);
        self.runner.generate(prompt_ids, max_new, &mut |tok| {
            produced.push(tok);
            // Return false to stop *after* recording the EOS token.
            !eos.contains(&tok)
        })?;
        // Trim a trailing EOS so callers see only real content.
        if produced.last().is_some_and(|t| eos.contains(t)) {
            produced.pop();
        }
        let text = if self.tokenizer.is_some() {
            self.decode(&produced)?
        } else {
            String::new()
        };
        Ok((produced, text))
    }

    /// Last-position logits for `context`, on whichever path the runner exposes.
    ///
    /// Tries the host-driven F32 `prefill_logits` first (real logits, seeds the
    /// decode KV cache so [`score_mc`](Self::score_mc) can continue multi-token
    /// choices). Falls back to `predict_logits` — the **packed / quantized**
    /// forward, which pads every context to one fixed `max_seq` shape and reads
    /// the true last position via `last_token_idx`. That fixed shape is the
    /// whole point: a compiled backend (Metal MPSGraph, CUDA, ROCm, wgpu,
    /// Vulkan) compiles the prefill graph **once** and reuses it for every
    /// context length, instead of recompiling per distinct length — the churn
    /// that makes many-length workloads (MMLU) crawl on those backends.
    pub fn context_last_logits(&mut self, context: &[u32]) -> Result<Vec<f32>> {
        let vocab = self.runner.vocab_size();
        if let Ok(l) = self.runner.prefill_logits(context) {
            if l.len() >= vocab {
                return Ok(l);
            }
        }
        let l = self.runner.predict_logits(context)?;
        if l.len() < vocab {
            bail!(
                "runner returned {} logits, expected >= vocab {vocab} (the F32 `predict_logits` \
                 is a placeholder — need `prefill_logits` or the packed path)",
                l.len()
            );
        }
        Ok(l)
    }

    /// lm-eval-style multiple-choice scoring.
    ///
    /// For each choice: read the shared `context`'s last-position logits (via
    /// [`context_last_logits`](Self::context_last_logits), so single-token
    /// choices work on the fast bucketed packed path too), then feed any further
    /// choice tokens one at a time, accumulating
    /// `Σ log P(choice_tok | context ++ prefix)`. Multi-token choices use
    /// `decode_logits` and therefore require the F32 path. Produces the same
    /// [`McResult`] as [`rlx_eval::score_mc`].
    pub fn score_mc(&mut self, item: &McItem) -> Result<McResult> {
        if item.context.is_empty() {
            bail!("multiple-choice scoring requires a non-empty context");
        }
        if item.choices.is_empty() {
            bail!("multiple-choice item has no choices");
        }

        let mut scores = Vec::with_capacity(item.choices.len());
        let mut scores_norm = Vec::with_capacity(item.choices.len());

        for choice in &item.choices {
            if choice.is_empty() {
                bail!("multiple-choice choice is empty");
            }
            // Fresh context read per choice — both paths reset state, so
            // choices never leak into one another.
            let mut logits = self.context_last_logits(&item.context)?;
            let mut sum = 0.0f32;
            for (k, &tok) in choice.iter().enumerate() {
                sum += log_softmax_at(&logits, tok as usize);
                if k + 1 < choice.len() {
                    logits = self.runner.decode_logits(tok)?;
                }
            }
            scores.push(sum);
            scores_norm.push(sum / choice.len() as f32);
        }

        let argmax = |v: &[f32]| {
            v.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0)
        };
        let best = argmax(&scores);
        let best_norm = argmax(&scores_norm);

        Ok(McResult {
            scores,
            scores_norm,
            best,
            best_norm,
        })
    }
}

/// Teacher-forced next-token log-probs for **any** runner, computed on the
/// host-driven KV path so [`fn@rlx_eval::perplexity`] (and any other
/// [`LmLogprobs`] consumer) works uniformly. `sequence_logprobs(tokens)`
/// returns `tokens.len() - 1` values where index `i` is
/// `log P(tokens[i+1] | tokens[..=i])`.
///
/// This walks the sequence one decode step at a time (`O(len)` forwards); a
/// per-family single-pass path (e.g. `Qwen3Generator::sequence_logprobs`) is
/// faster when the concrete type is in hand, but this keeps the harness generic.
impl LmLogprobs for BenchModel {
    fn sequence_logprobs(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        if tokens.len() < 2 {
            return Ok(Vec::new());
        }
        let mut out = Vec::with_capacity(tokens.len() - 1);
        // Prefill the first token → logits that predict tokens[1].
        let logits = self.runner.prefill_logits(&tokens[..1])?;
        out.push(log_softmax_at(&logits, tokens[1] as usize));
        // Each decode step feeds tokens[i] and predicts tokens[i+1].
        for i in 1..tokens.len() - 1 {
            let logits = self.runner.decode_logits(tokens[i])?;
            out.push(log_softmax_at(&logits, tokens[i + 1] as usize));
        }
        Ok(out)
    }
}
