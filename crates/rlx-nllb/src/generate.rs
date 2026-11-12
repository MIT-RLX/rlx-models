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

//! Autoregressive generation: greedy and beam search with NLLB's forced
//! target-language token (`forced_bos_token_id` = tgt FLORES lang id).

use crate::config::NllbConfig;
use crate::runner::{Encoded, NllbModel};
use anyhow::Result;

/// Generation options. Defaults follow HF `GenerationConfig`, and beam
/// search follows HF `_beam_search` step for step, so the same options give
/// the same tokens as `transformers`.
#[derive(Debug, Clone)]
pub struct GenerateConfig {
    pub max_new_tokens: usize,
    pub num_beams: usize,
    /// Finished beams score `sum_logprobs / generated_len^length_penalty`.
    pub length_penalty: f64,
    /// Stop as soon as `num_beams` hypotheses have finished (HF `True`).
    /// When false, stop once no running beam can beat the worst finished one
    /// at its current length (HF `False`, the transformers default).
    pub early_stopping: bool,
    /// Target language token id forced at decoder position 1 (NLLB convention).
    pub forced_bos_token_id: Option<u32>,
    /// Token forced on the last step allowed by `max_new_tokens`. NLLB's
    /// config sets none, so HF ends a length-capped output without EOS.
    pub forced_eos_token_id: Option<u32>,
    pub no_repeat_ngram_size: usize,
}

impl GenerateConfig {
    pub fn new(max_new_tokens: usize) -> Self {
        Self {
            max_new_tokens,
            num_beams: 1,
            length_penalty: 1.0,
            early_stopping: true,
            forced_bos_token_id: None,
            forced_eos_token_id: None,
            no_repeat_ngram_size: 0,
        }
    }

    pub fn from_config(cfg: &NllbConfig, max_new_tokens: usize) -> Self {
        Self {
            num_beams: cfg.num_beams.max(1),
            no_repeat_ngram_size: cfg.no_repeat_ngram_size,
            ..Self::new(max_new_tokens)
        }
    }

    pub fn with_forced_bos(mut self, lang_id: u32) -> Self {
        self.forced_bos_token_id = Some(lang_id);
        self
    }

    pub fn with_beams(mut self, n: usize) -> Self {
        self.num_beams = n.max(1);
        self
    }
}

fn log_softmax_in_place(logits: &mut [f32]) {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0f64;
    for &l in logits.iter() {
        sum += ((l - max) as f64).exp();
    }
    let lse = max as f64 + sum.ln();
    for l in logits.iter_mut() {
        *l = (*l as f64 - lse) as f32;
    }
}

/// HF logits processors, in HF order: no-repeat-ngram, forced BOS (tgt lang
/// at position 1), forced EOS (last allowed position). `seq` is the decoder
/// prefix; the token about to be produced lands at index `seq.len()`.
fn apply_processors(scores: &mut [f32], seq: &[u32], max_len: usize, opts: &GenerateConfig) {
    let n = opts.no_repeat_ngram_size;
    if n > 0 && seq.len() + 1 >= n {
        let prefix = &seq[seq.len() + 1 - n..];
        for i in 0..=seq.len() + 1 - n {
            if i + n > seq.len() {
                break;
            }
            if seq[i..i + n - 1] == *prefix {
                let banned = seq[i + n - 1] as usize;
                if banned < scores.len() {
                    scores[banned] = f32::NEG_INFINITY;
                }
            }
        }
    }
    let cur_len = seq.len();
    if cur_len == 1
        && let Some(tok) = opts.forced_bos_token_id
    {
        force_token(scores, tok);
    }
    if cur_len == max_len - 1
        && let Some(tok) = opts.forced_eos_token_id
    {
        force_token(scores, tok);
    }
}

fn force_token(scores: &mut [f32], token: u32) {
    for (i, l) in scores.iter_mut().enumerate() {
        *l = if i as u32 == token {
            0.0
        } else {
            f32::NEG_INFINITY
        };
    }
}

/// The token the processors force at this step without needing its logits:
/// the target language at position 1. That step's LM head (a 1 GB pass over
/// the vocabulary table) is skipped — every other score would be −∞. Not
/// when a forced EOS lands on the same step and overrides it.
fn forced_without_logits(seq: &[u32], max_len: usize, opts: &GenerateConfig) -> Option<u32> {
    let eos_here = opts.forced_eos_token_id.is_some() && seq.len() == max_len - 1;
    (seq.len() == 1 && !eos_here)
        .then_some(opts.forced_bos_token_id)
        .flatten()
}

fn argmax(a: &[f32]) -> u32 {
    rlx_runtime::search::argmax_u32(a)
}

/// Indices of the `k` largest finite entries of `v`, best first.
fn top_k(v: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..v.len()).filter(|&i| v[i].is_finite()).collect();
    let k = k.min(idx.len());
    if k == 0 {
        return idx;
    }
    if k < idx.len() {
        idx.select_nth_unstable_by(k - 1, |&a, &b| v[b].total_cmp(&v[a]));
        idx.truncate(k);
    }
    idx.sort_by(|&a, &b| v[b].total_cmp(&v[a]).then(a.cmp(&b)));
    idx
}

impl NllbModel {
    /// Greedy decode. Starts with `decoder_start_token_id` (eos=2); forces
    /// `forced_bos_token_id` (tgt lang) as the first generated token.
    pub fn generate_greedy(&mut self, enc: &Encoded, opts: &GenerateConfig) -> Result<Vec<u32>> {
        let eos = self.config().eos_token_id;
        let max_len = opts.max_new_tokens + 1;
        let mut state = self.start_decode(enc, 1, max_len)?;
        let mut seq = vec![self.config().decoder_start_token_id];
        while seq.len() < max_len {
            let next = if let Some(tok) = forced_without_logits(&seq, max_len, opts) {
                self.step_hidden(&mut state, &seq[seq.len() - 1..])?;
                tok
            } else {
                let mut scores = self.step(&mut state, &seq[seq.len() - 1..])?;
                apply_processors(&mut scores, &seq, max_len, opts);
                argmax(&scores)
            };
            seq.push(next);
            if next == eos {
                break;
            }
        }
        Ok(seq)
    }

    /// Beam search with HF `_beam_search` semantics:
    ///
    /// * candidates are the top `2·num_beams` of all beams × vocab by summed
    ///   log-prob (processors act on log-probs, after `log_softmax`);
    /// * a candidate finishes on EOS or at `max_new_tokens`, but only one
    ///   ranked inside the top `num_beams` may join the finished set, which
    ///   keeps the best `num_beams`, scored `logp / generated_len^penalty`
    ///   (`generated_len` excludes the decoder start token);
    /// * the best `num_beams` unfinished candidates keep running.
    ///
    /// All beams decode as one batch; the KV cache follows each beam's parent.
    pub fn generate_beam(&mut self, enc: &Encoded, opts: &GenerateConfig) -> Result<Vec<u32>> {
        let nb = opts.num_beams.max(1);
        if nb == 1 {
            return self.generate_greedy(enc, opts);
        }
        let eos = self.config().eos_token_id;
        let vocab = self.config().vocab_size;
        let max_len = opts.max_new_tokens + 1;
        let keep = 2 * nb;
        let lp = opts.length_penalty;

        let mut state = self.start_decode(enc, nb, max_len)?;
        // Exactly `nb` rows; a dead row (score −∞) only pads the batch — the
        // first step has a single live beam, as in HF's −1e9 initialisation.
        let start = vec![self.config().decoder_start_token_id];
        let mut running: Vec<(Vec<u32>, f64)> = (0..nb)
            .map(|i| (start.clone(), if i == 0 { 0.0 } else { f64::NEG_INFINITY }))
            .collect();
        // Best `nb` finished hypotheses, best first.
        let mut finished: Vec<(Vec<u32>, f64)> = Vec::new();

        loop {
            let cur_len = running[0].0.len();
            let last: Vec<u32> = running.iter().map(|(s, _)| *s.last().unwrap()).collect();
            // (beam, token, summed log-prob), best first.
            let mut cands: Vec<(usize, u32, f64)> = Vec::new();
            let forced = forced_without_logits(&running[0].0, max_len, opts);
            let mut logits = if let Some(tok) = forced {
                // Only beam 0 is live here; the forced token scores log 1 = 0.
                self.step_hidden(&mut state, &last)?;
                cands.push((0, tok, running[0].1));
                Vec::new()
            } else {
                self.step(&mut state, &last)?
            };
            for (b, (seq, score)) in running.iter().enumerate() {
                if forced.is_some() {
                    break;
                }
                if *score == f64::NEG_INFINITY {
                    continue;
                }
                let scores = &mut logits[b * vocab..(b + 1) * vocab];
                log_softmax_in_place(scores);
                apply_processors(scores, seq, max_len, opts);
                for t in top_k(scores, keep) {
                    cands.push((b, t as u32, score + scores[t] as f64));
                }
            }
            cands.sort_by(|a, b| b.2.total_cmp(&a.2));
            cands.truncate(keep);

            let at_limit = cur_len + 1 >= max_len;
            // As in HF, fullness is judged before this step's finishers merge.
            let full = opts.early_stopping && finished.len() >= nb;
            let mut next: Vec<(Vec<u32>, f64)> = Vec::with_capacity(nb);
            let mut parents = Vec::with_capacity(nb);
            for (rank, &(b, tok, score)) in cands.iter().enumerate() {
                let mut seq = running[b].0.clone();
                seq.push(tok);
                if tok == eos || at_limit {
                    if rank < nb && !full {
                        let norm = score / ((seq.len() - 1) as f64).powf(lp);
                        finished.push((seq, norm));
                    }
                } else if next.len() < nb {
                    next.push((seq, score));
                    parents.push(b);
                }
            }
            finished.sort_by(|a, b| b.1.total_cmp(&a.1));
            finished.truncate(nb);

            if next.is_empty() || at_limit || (opts.early_stopping && finished.len() >= nb) {
                running = next;
                break;
            }
            // HF's `early_stopping=False` heuristic: the best running beam,
            // scored at its current length, cannot beat the worst finished.
            if !opts.early_stopping && finished.len() >= nb {
                let (seq, score) = &next[0];
                if score / ((seq.len() - 1) as f64).powf(lp) <= finished[nb - 1].1 {
                    running = next;
                    break;
                }
            }
            while next.len() < nb {
                next.push((next[0].0.clone(), f64::NEG_INFINITY));
                parents.push(parents[0]);
            }
            state.reorder(&parents);
            running = next;
        }

        Ok(finished
            .into_iter()
            .next()
            .or_else(|| running.into_iter().next())
            .map(|(s, _)| s)
            .unwrap_or_default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_repeat_ngram_bans_completion_of_seen_bigram() {
        let opts = GenerateConfig {
            no_repeat_ngram_size: 2,
            ..GenerateConfig::new(16)
        };
        // Seen bigrams: (2,5) (5,7) (7,5) — prefix ends in 5, so 7 is banned.
        let mut s = vec![0f32; 10];
        apply_processors(&mut s, &[2, 5, 7, 5], 16, &opts);
        assert_eq!(s[7], f32::NEG_INFINITY);
        assert!(s.iter().enumerate().all(|(i, v)| i == 7 || *v == 0.0));
    }

    #[test]
    fn forced_tokens_follow_options() {
        let opts = GenerateConfig::new(4).with_forced_bos(9);
        let mut s = vec![0f32; 10];
        apply_processors(&mut s, &[2], 5, &opts);
        assert_eq!(argmax(&s), 9);
        // No forced EOS unless configured.
        let mut s = vec![0f32; 10];
        s[3] = 1.0;
        apply_processors(&mut s, &[2, 9, 4, 4], 5, &opts);
        assert_eq!(argmax(&s), 3);
        let opts = GenerateConfig {
            forced_eos_token_id: Some(2),
            ..opts
        };
        apply_processors(&mut s, &[2, 9, 4, 4], 5, &opts);
        assert_eq!(argmax(&s), 2);
    }

    #[test]
    fn top_k_skips_masked_and_orders() {
        let v = [0.1, f32::NEG_INFINITY, 0.5, 0.3, 0.5];
        assert_eq!(top_k(&v, 3), vec![2, 4, 3]);
        assert_eq!(top_k(&v, 10), vec![2, 4, 3, 0]);
    }
}
