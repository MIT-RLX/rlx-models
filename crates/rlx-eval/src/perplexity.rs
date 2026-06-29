// RLX models — language-model evaluation.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! Sliding-window perplexity over a tokenized corpus.

use crate::LmLogprobs;
use anyhow::{Result, bail};

/// Striding configuration for [`perplexity`].
#[derive(Debug, Clone, Copy)]
pub struct PerplexityConfig {
    /// Tokens per forward window (context length used for scoring).
    pub seq_len: usize,
    /// How far the window advances each step. Clamped to `seq_len - 1` so
    /// windows overlap by ≥1 token, giving contiguous target coverage.
    pub stride: usize,
}

impl Default for PerplexityConfig {
    fn default() -> Self {
        Self {
            seq_len: 512,
            stride: 512,
        }
    }
}

/// Perplexity = `exp(mean NLL)` over every scored next-token position.
///
/// Slides a window of `seq_len` over `token_ids`; each next-token position is
/// scored exactly once (overlap is de-duplicated), so longer contexts improve
/// the estimate without double-counting.
pub fn perplexity<M: LmLogprobs>(
    model: &mut M,
    token_ids: &[u32],
    cfg: PerplexityConfig,
) -> Result<f64> {
    let n = token_ids.len();
    if n < 2 {
        bail!("perplexity needs at least 2 tokens, got {n}");
    }
    let seq_len = cfg.seq_len.clamp(2, n);
    let stride = cfg.stride.clamp(1, seq_len.saturating_sub(1).max(1));

    let mut total_nll = 0.0f64;
    let mut count = 0usize;
    let mut next_target = 1usize; // first global target index not yet scored
    let mut begin = 0usize;

    loop {
        let end = (begin + seq_len).min(n);
        let window = &token_ids[begin..end];
        let lps = model.sequence_logprobs(window)?; // lps[i] → target begin+i+1
        // Window covers targets begin+1 ..= end-1; score the as-yet-unscored.
        let first = next_target.max(begin + 1);
        for tgt in first..end {
            let i = tgt - (begin + 1);
            total_nll += -(lps[i] as f64);
            count += 1;
        }
        next_target = end;
        if end >= n {
            break;
        }
        begin += stride;
    }

    if count == 0 {
        bail!("perplexity scored no positions");
    }
    Ok((total_nll / count as f64).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Returns a constant log-prob for every position.
    struct ConstModel(f32);
    impl LmLogprobs for ConstModel {
        fn sequence_logprobs(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
            Ok(vec![self.0; tokens.len().saturating_sub(1)])
        }
    }

    #[test]
    fn constant_logprob_gives_exp_nll() {
        // Every position has logprob -1 → NLL 1 → PPL = e.
        let toks: Vec<u32> = (0..20).collect();
        let ppl = perplexity(
            &mut ConstModel(-1.0),
            &toks,
            PerplexityConfig {
                seq_len: 8,
                stride: 4,
            },
        )
        .unwrap();
        assert!((ppl - std::f64::consts::E).abs() < 1e-6, "ppl={ppl}");
    }

    #[test]
    fn scores_each_position_once() {
        // n-1 positions scored; with logprob 0 → PPL = 1 regardless of stride.
        let toks: Vec<u32> = (0..30).collect();
        for stride in [1usize, 3, 7, 16] {
            let ppl = perplexity(
                &mut ConstModel(0.0),
                &toks,
                PerplexityConfig { seq_len: 8, stride },
            )
            .unwrap();
            assert!((ppl - 1.0).abs() < 1e-9, "stride={stride} ppl={ppl}");
        }
    }

    #[test]
    fn too_short_errors() {
        assert!(perplexity(&mut ConstModel(0.0), &[1], PerplexityConfig::default()).is_err());
    }
}
