// RLX models — language-model evaluation.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! lm-eval-style multiple-choice scoring: pick the continuation with the
//! highest (length-normalized) log-probability under the model.

use crate::LmLogprobs;
use anyhow::{Result, bail};

/// One multiple-choice item: a shared `context` and several tokenized
/// `choices` (continuations).
#[derive(Debug, Clone)]
pub struct McItem {
    pub context: Vec<u32>,
    pub choices: Vec<Vec<u32>>,
}

/// Per-choice scores and the winners.
#[derive(Debug, Clone)]
pub struct McResult {
    /// Summed continuation log-prob per choice (`acc`).
    pub scores: Vec<f32>,
    /// Length-normalized log-prob per choice (`acc_norm`).
    pub scores_norm: Vec<f32>,
    /// Argmax of `scores`.
    pub best: usize,
    /// Argmax of `scores_norm`.
    pub best_norm: usize,
}

/// Score each choice by `Σ log P(choice_tok | context ++ choice_prefix)` using
/// one forward per choice. Returns raw + length-normalized scores and their
/// argmaxes. Requires a non-empty context and non-empty choices.
pub fn score_mc<M: LmLogprobs>(model: &mut M, item: &McItem) -> Result<McResult> {
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
        let seq: Vec<u32> = item.context.iter().chain(choice).copied().collect();
        let lps = model.sequence_logprobs(&seq)?; // len = seq.len() - 1
        // The first choice token sits at global position context.len(); its
        // log-prob is lps[context.len() - 1]. Sum over the choice span.
        let start = item.context.len() - 1;
        let span = &lps[start..start + choice.len()];
        let sum: f32 = span.iter().sum();
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Assigns each token id a fixed per-token log-prob (id → -id*0.1), so a
    /// choice made of smaller ids scores higher.
    struct IdModel;
    impl LmLogprobs for IdModel {
        fn sequence_logprobs(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
            // logprob of predicting tokens[i+1] depends only on that token.
            Ok(tokens[1..].iter().map(|&t| -(t as f32) * 0.1).collect())
        }
    }

    #[test]
    fn picks_highest_logprob_choice() {
        let item = McItem {
            context: vec![1, 2],
            choices: vec![vec![3, 3], vec![9, 9]],
        };
        let r = score_mc(&mut IdModel, &item).unwrap();
        // Choice 0 (smaller ids) has higher (less negative) summed logprob.
        assert_eq!(r.best, 0);
        assert!(r.scores[0] > r.scores[1]);
        assert_eq!(r.scores_norm.len(), 2);
    }

    #[test]
    fn length_normalization_can_flip_winner() {
        // A long low-per-token choice vs a short choice.
        let item = McItem {
            context: vec![1],
            choices: vec![vec![2, 2, 2, 2], vec![5]],
        };
        let r = score_mc(&mut IdModel, &item).unwrap();
        // Raw sum favors... choice 0 sum=-0.8, choice 1 sum=-0.5 → best=1.
        assert_eq!(r.best, 1);
        // Norm: choice0 -0.2/tok, choice1 -0.5/tok → best_norm=0.
        assert_eq!(r.best_norm, 0);
    }

    #[test]
    fn empty_context_errors() {
        let item = McItem {
            context: vec![],
            choices: vec![vec![1]],
        };
        assert!(score_mc(&mut IdModel, &item).is_err());
    }
}
