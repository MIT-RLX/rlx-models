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

//! Autoregressive generation: greedy and beam search with the same logits
//! processors HuggingFace applies for Florence-2 (`forced_bos_token_id`,
//! `forced_eos_token_id`, `no_repeat_ngram_size`).

use crate::config::Florence2TextConfig;
use crate::runner::Florence2Model;
use anyhow::Result;

/// Generation knobs (defaults mirror Florence-2's `generation_config`).
#[derive(Debug, Clone)]
pub struct GenerateConfig {
    pub max_new_tokens: usize,
    pub num_beams: usize,
    pub length_penalty: f64,
    pub early_stopping: bool,
}

impl GenerateConfig {
    pub fn from_text(t: &Florence2TextConfig, max_new_tokens: usize) -> Self {
        Self {
            max_new_tokens,
            num_beams: t.num_beams,
            length_penalty: 1.0,
            early_stopping: true,
        }
    }
}

fn log_softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0f64;
    for &l in logits {
        sum += ((l - max) as f64).exp();
    }
    let lse = max as f64 + sum.ln();
    logits.iter().map(|&l| (l as f64 - lse) as f32).collect()
}

/// Apply Florence-2's logits processors in place (HF order: no-repeat-ngram,
/// forced-bos, forced-eos). `cur_len` is the current decoder length, i.e. the
/// position index of the token about to be produced.
fn apply_processors(
    logits: &mut [f32],
    seq: &[u32],
    cur_len: usize,
    max_len: usize,
    t: &Florence2TextConfig,
    no_repeat_ngram: usize,
) {
    // no_repeat_ngram_size: ban tokens completing a previously seen n-gram.
    if no_repeat_ngram > 0 && seq.len() >= no_repeat_ngram {
        let n = no_repeat_ngram;
        let prefix = &seq[seq.len() + 1 - n..]; // last (n-1) tokens
        for i in 0..=seq.len() - n {
            if seq[i..i + n - 1] == *prefix {
                let banned = seq[i + n - 1] as usize;
                if banned < logits.len() {
                    logits[banned] = f32::NEG_INFINITY;
                }
            }
        }
    }
    // forced_bos_token_id: first generated token (position 1).
    if cur_len == 1 {
        force_token(logits, t.forced_bos_token_id);
    }
    // forced_eos_token_id: last position.
    if cur_len == max_len - 1 {
        force_token(logits, t.forced_eos_token_id);
    }
}

fn force_token(logits: &mut [f32], token: u32) {
    for (i, l) in logits.iter_mut().enumerate() {
        *l = if i as u32 == token {
            0.0
        } else {
            f32::NEG_INFINITY
        };
    }
}

fn argmax(a: &[f32]) -> u32 {
    let mut bi = 0u32;
    let mut bv = f32::NEG_INFINITY;
    for (i, &v) in a.iter().enumerate() {
        if v > bv {
            bv = v;
            bi = i as u32;
        }
    }
    bi
}

impl Florence2Model {
    /// Greedy decode. `encoder_hidden` is `[enc_seq · d]`. Returns the full
    /// token sequence including the start token (and trailing EOS).
    pub fn generate_greedy(
        &mut self,
        encoder_hidden: &[f32],
        enc_seq: usize,
        opts: &GenerateConfig,
    ) -> Result<Vec<u32>> {
        let t = self.config().text.clone();
        let max_len = opts.max_new_tokens + 1;
        let no_ngram = t.no_repeat_ngram_size;
        let mut seq = vec![t.decoder_start_token_id];
        while seq.len() < max_len {
            let mut last = self.decode_logits(&seq, encoder_hidden, enc_seq, max_len)?;
            apply_processors(&mut last, &seq, seq.len(), max_len, &t, no_ngram);
            let next = argmax(&last);
            seq.push(next);
            if next == t.eos_token_id {
                break;
            }
        }
        Ok(seq)
    }

    /// Beam search (HF semantics: log-prob accumulation, `2·num_beams`
    /// candidate pool, length-penalty normalization, early stopping).
    pub fn generate_beam(
        &mut self,
        encoder_hidden: &[f32],
        enc_seq: usize,
        opts: &GenerateConfig,
    ) -> Result<Vec<u32>> {
        let t = self.config().text.clone();
        let nb = opts.num_beams.max(1);
        if nb == 1 {
            return self.generate_greedy(encoder_hidden, enc_seq, opts);
        }
        let max_len = opts.max_new_tokens + 1;
        let no_ngram = t.no_repeat_ngram_size;
        let eos = t.eos_token_id;

        // Active beams: (tokens, cumulative_logprob). Seed only beam 0 so the
        // first expansion does not produce `num_beams` identical hypotheses.
        let mut beams: Vec<(Vec<u32>, f64)> = vec![(vec![t.decoder_start_token_id], 0.0)];
        let mut finished: Vec<(Vec<u32>, f64)> = Vec::new();

        for _ in 0..max_len - 1 {
            let mut cands: Vec<(Vec<u32>, f64)> = Vec::new();
            for (seq, score) in &beams {
                let mut last = self.decode_logits(seq, encoder_hidden, enc_seq, max_len)?;
                apply_processors(&mut last, seq, seq.len(), max_len, &t, no_ngram);
                let logp = log_softmax(&last);
                // Top 2*nb tokens for this beam.
                let mut idx: Vec<usize> = (0..logp.len()).collect();
                idx.select_nth_unstable_by(2 * nb, |&a, &b| logp[b].partial_cmp(&logp[a]).unwrap());
                for &tok in idx.iter().take(2 * nb) {
                    if logp[tok] == f32::NEG_INFINITY {
                        continue;
                    }
                    let mut ns = seq.clone();
                    ns.push(tok as u32);
                    cands.push((ns, score + logp[tok] as f64));
                }
            }
            // Keep the best `2*nb` candidates, route EOS to finished.
            cands.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let mut next_beams: Vec<(Vec<u32>, f64)> = Vec::new();
            for (seq, score) in cands {
                if *seq.last().unwrap() == eos {
                    let norm = score / (seq.len() as f64).powf(opts.length_penalty);
                    finished.push((seq, norm));
                } else {
                    next_beams.push((seq, score));
                }
                if next_beams.len() >= nb {
                    break;
                }
            }
            beams = next_beams;
            if beams.is_empty() {
                break;
            }
            // Early stopping: enough finished hypotheses and the best active
            // beam cannot beat them.
            if opts.early_stopping && finished.len() >= nb {
                break;
            }
        }

        // Add any unfinished beams as candidates (normalized).
        for (seq, score) in beams {
            let norm = score / (seq.len() as f64).powf(opts.length_penalty);
            finished.push((seq, norm));
        }
        finished.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        Ok(finished
            .into_iter()
            .next()
            .map(|(s, _)| s)
            .unwrap_or_default())
    }
}
