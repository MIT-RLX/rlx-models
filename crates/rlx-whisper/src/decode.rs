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

//! Greedy / beam decoding helpers (host-side token loop).

use crate::config::WhisperConfig;

pub const SOT_TOKEN: &str = "<|startoftranscript|>";
pub const TRANSCRIBE_TOKEN: &str = "<|transcribe|>";
pub const TRANSLATE_TOKEN: &str = "<|translate|>";
pub const NO_TIMESTAMPS_TOKEN: &str = "<|notimestamps|>";
pub const EOT_TOKEN: &str = "<|endoftext|>";

/// Build the initial decoder prompt: SOT + optional language/task tokens.
pub fn initial_prompt(
    tokenizer: &tokenizers::Tokenizer,
    language: Option<&str>,
    translate: bool,
) -> anyhow::Result<Vec<u32>> {
    initial_prompt_opts(tokenizer, language, translate, false)
}

/// Like [`initial_prompt`] but omits `<|notimestamps|>` when `timestamps` is true.
pub fn initial_prompt_opts(
    tokenizer: &tokenizers::Tokenizer,
    language: Option<&str>,
    translate: bool,
    timestamps: bool,
) -> anyhow::Result<Vec<u32>> {
    let mut ids = vec![
        tokenizer
            .token_to_id(SOT_TOKEN)
            .ok_or_else(|| anyhow::anyhow!("tokenizer missing {SOT_TOKEN}"))?,
    ];
    if let Some(lang) = language {
        let tok = format!("<|{lang}|>");
        if let Some(id) = tokenizer.token_to_id(&tok) {
            ids.push(id);
        }
    }
    let task = if translate {
        TRANSLATE_TOKEN
    } else {
        TRANSCRIBE_TOKEN
    };
    if let Some(id) = tokenizer.token_to_id(task) {
        ids.push(id);
    }
    if !timestamps {
        if let Some(id) = tokenizer.token_to_id(NO_TIMESTAMPS_TOKEN) {
            ids.push(id);
        }
    }
    Ok(ids)
}

/// View of batch row `b` in `[batch, dec_seq, vocab]` logits (last time step).
pub fn batched_logits_row(
    logits: &[f32],
    batch_ix: usize,
    batch: usize,
    dec_seq: usize,
    vocab: usize,
) -> &[f32] {
    let plane = dec_seq * vocab;
    let off = batch_ix * plane + dec_seq.saturating_sub(1) * vocab;
    if logits.len() >= off + vocab {
        &logits[off..off + vocab]
    } else if batch == 1 && logits.len() >= vocab {
        &logits[..vocab]
    } else {
        &[]
    }
}

/// Owned copy of [`batched_logits_row`] (tests / legacy callers).
pub fn batched_logits_row_owned(
    logits: &[f32],
    batch_ix: usize,
    batch: usize,
    dec_seq: usize,
    vocab: usize,
) -> Vec<f32> {
    batched_logits_row(logits, batch_ix, batch, dec_seq, vocab).to_vec()
}

/// Last row of a `[dec_seq, vocab]` logits tensor in flat row-major layout.
pub fn last_logits_row(logits: &[f32], dec_seq: usize, vocab: usize) -> Vec<f32> {
    let off = dec_seq.saturating_sub(1) * vocab;
    if logits.len() >= off + vocab {
        logits[off..off + vocab].to_vec()
    } else if logits.len() == vocab {
        logits.to_vec()
    } else {
        vec![0.0; vocab]
    }
}

pub fn argmax_logits(logits: &[f32]) -> u32 {
    argmax_logits_masked(logits, None)
}

/// Argmax over one row in `[batch, dec_seq, vocab]` logits without allocating the row.
pub fn argmax_batched_row(
    logits: &[f32],
    batch_ix: usize,
    batch: usize,
    dec_seq: usize,
    vocab: usize,
    suppression: Option<&SuppressionMask>,
) -> u32 {
    let plane = dec_seq * vocab;
    let off = batch_ix * plane + dec_seq.saturating_sub(1) * vocab;
    let row = if logits.len() >= off + vocab {
        &logits[off..off + vocab]
    } else if batch == 1 && logits.len() >= vocab {
        &logits[..vocab]
    } else {
        return 0;
    };
    argmax_logits_masked(row, suppression.map(|s| s.data.as_slice()))
}

/// Argmax with optional precomputed suppression mask (`-inf` entries skipped).
pub fn argmax_logits_masked(logits: &[f32], mask: Option<&[f32]>) -> u32 {
    let mut best_i = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if mask.is_some_and(|m| i < m.len() && !m[i].is_finite()) {
            continue;
        }
        if v > best_v {
            best_v = v;
            best_i = i;
        }
    }
    best_i as u32
}

/// Precomputed token suppression for the full vocabulary.
#[derive(Debug, Clone)]
pub struct SuppressionMask {
    pub data: Vec<f32>,
    begin: Vec<u32>,
}

impl SuppressionMask {
    pub fn from_config(cfg: &WhisperConfig) -> Self {
        let mut data = vec![0f32; cfg.vocab_size];
        for &id in &cfg.suppress_tokens {
            let i = id as usize;
            if i < data.len() {
                data[i] = f32::NEG_INFINITY;
            }
        }
        Self {
            data,
            begin: cfg.begin_suppress_tokens.clone(),
        }
    }

    pub fn apply<'a>(&self, logits: &'a mut [f32]) -> &'a mut [f32] {
        let n = logits.len().min(self.data.len());
        for i in 0..n {
            if !self.data[i].is_finite() {
                logits[i] = f32::NEG_INFINITY;
            }
        }
        logits
    }

    /// HF `begin_suppress_tokens` — first generated token only.
    pub fn apply_begin(&self, logits: &mut [f32]) {
        for &id in &self.begin {
            let i = id as usize;
            if i < logits.len() {
                logits[i] = f32::NEG_INFINITY;
            }
        }
    }

    /// Argmax after log-softmax with global + optional begin suppression.
    pub fn argmax_next(&self, logits: &mut [f32], at_generation_start: bool) -> u32 {
        if at_generation_start {
            self.apply_begin(logits);
        }
        self.apply(logits);
        log_softmax_last(logits);
        argmax_logits_masked(logits, None)
    }
}

pub fn argmax_last_logits(logits: &[f32], dec_seq: usize, vocab: usize) -> u32 {
    let last = dec_seq - 1;
    let off = last * vocab;
    argmax_logits(&logits[off..off + vocab])
}

pub fn log_softmax_last(logits: &mut [f32]) {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0f32;
    for v in logits.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    let log_z = sum.max(1e-20).ln();
    for v in logits.iter_mut() {
        *v = v.ln() - log_z;
    }
}

pub fn is_eot(tokenizer: &tokenizers::Tokenizer, id: u32) -> bool {
    tokenizer
        .id_to_token(id)
        .map(|t| t == EOT_TOKEN)
        .unwrap_or(false)
}

pub fn filter_suppressed(logits: &mut [f32], cfg: &WhisperConfig) {
    SuppressionMask::from_config(cfg).apply(logits);
}

/// One beam hypothesis during search.
#[derive(Debug, Clone)]
pub struct BeamHypothesis {
    pub tokens: Vec<u32>,
    pub score: f32,
    pub completed: bool,
}

/// Beam state for KV-incremental search (suffix tokens only; prompt handled separately).
#[derive(Debug, Clone)]
pub struct BeamKvState {
    pub suffix: Vec<u32>,
    pub score: f32,
    pub cache: crate::cache::WhisperKvCache,
    pub next_logits: Vec<f32>,
    pub completed: bool,
}

/// Beam search with per-hypothesis self-attention KV (no full prefill replay).
pub fn beam_search_decode_kv(
    prefill_logits: &[f32],
    prompt_len: usize,
    initial_cache: crate::cache::WhisperKvCache,
    eot_id: u32,
    beam_size: usize,
    max_steps: usize,
    vocab: usize,
    mut decode_step: impl FnMut(
        u32,
        &crate::cache::WhisperKvCache,
    ) -> anyhow::Result<(Vec<f32>, crate::cache::WhisperKvCache)>,
) -> anyhow::Result<Vec<u32>> {
    let beam_size = beam_size.max(1);
    let mut beams = vec![BeamKvState {
        suffix: Vec::new(),
        score: 0.0,
        cache: initial_cache,
        next_logits: last_logits_row(prefill_logits, prompt_len, vocab),
        completed: false,
    }];

    for _ in 0..max_steps {
        let mut candidates = Vec::new();
        for beam in &beams {
            if beam.completed {
                candidates.push(beam.clone());
                continue;
            }
            let mut lp = beam.next_logits.clone();
            log_softmax_last(&mut lp);
            let mut indexed: Vec<(usize, f32)> = lp.into_iter().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            for (tok, logp) in indexed.into_iter().take(beam_size) {
                let tok = tok as u32;
                if beam.suffix.is_empty() && tok == eot_id {
                    continue;
                }
                let (next_logits, new_cache) = decode_step(tok, &beam.cache)?;
                let row = if next_logits.len() == vocab {
                    next_logits
                } else {
                    last_logits_row(&next_logits, 1, vocab)
                };
                let mut suffix = beam.suffix.clone();
                suffix.push(tok);
                candidates.push(BeamKvState {
                    suffix,
                    score: beam.score + logp,
                    cache: new_cache,
                    next_logits: row,
                    completed: tok == eot_id,
                });
            }
        }
        candidates.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        candidates.truncate(beam_size);
        beams = candidates;
        if beams.iter().all(|b| b.completed) {
            break;
        }
    }
    Ok(beams
        .into_iter()
        .max_by(|a, b| {
            a.score
                .partial_cmp(&b.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|b| b.suffix)
        .unwrap_or_default())
}

/// Top-k token ids from log-probs (descending). Full sort — fine for small `vocab`.
pub fn topk_tokens(logits: &mut [f32], k: usize) -> Vec<(u32, f32)> {
    log_softmax_last(logits);
    topk_from_log_probs(logits, k)
}

/// Top-k on already log-softmaxed logits — `O(vocab * k)` (small `k` / beam width).
pub fn topk_from_log_probs(log_probs: &[f32], k: usize) -> Vec<(u32, f32)> {
    use std::cmp::Ordering;
    if k == 0 {
        return Vec::new();
    }
    let mut best: Vec<(u32, f32)> = Vec::with_capacity(k);
    for (i, &p) in log_probs.iter().enumerate() {
        if !p.is_finite() {
            continue;
        }
        if best.len() < k {
            best.push((i as u32, p));
        } else {
            let mut min_j = 0usize;
            for (j, &(_, v)) in best.iter().enumerate() {
                if v < best[min_j].1 {
                    min_j = j;
                }
            }
            if p > best[min_j].1 {
                best[min_j] = (i as u32, p);
            }
        }
    }
    best.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    best
}

/// Copy `row` into `scratch`, log-softmax, return top-k (reuses `scratch` each call).
pub fn topk_tokens_row(scratch: &mut [f32], row: &[f32], k: usize) -> Vec<(u32, f32)> {
    let n = row.len().min(scratch.len());
    scratch[..n].copy_from_slice(&row[..n]);
    log_softmax_last(&mut scratch[..n]);
    topk_from_log_probs(&scratch[..n], k)
}

/// One beam slot in a batched (regions × beams) decode.
#[derive(Debug, Clone)]
struct BatchedBeamSlot {
    region: usize,
    suffix: Vec<u32>,
    score: f32,
    logits: Vec<f32>,
    completed: bool,
}

/// Batched beam search: `batch = n_regions * beam_size`, one GPU forward per time step.
pub fn beam_search_decode_kv_batched(
    prefill_logits: &[f32],
    prompt_len: usize,
    mut cache: crate::cache::WhisperKvCache,
    n_regions: usize,
    beam_size: usize,
    max_steps: usize,
    vocab: usize,
    eot_id: u32,
    mut decode_step_batch: impl FnMut(
        &[u32],
        &mut crate::cache::WhisperKvCache,
    ) -> anyhow::Result<Vec<f32>>,
) -> anyhow::Result<Vec<Vec<u32>>> {
    use crate::batch::reorder_kv_beams;

    let mut logits_scratch = vec![0f32; vocab];
    let beam_size = beam_size.max(1);
    let batch = n_regions * beam_size;

    let mut slots: Vec<BatchedBeamSlot> = (0..batch)
        .map(|b| {
            let ri = b / beam_size;
            BatchedBeamSlot {
                region: ri,
                suffix: Vec::new(),
                score: 0.0,
                logits: batched_logits_row_owned(prefill_logits, b, batch, prompt_len, vocab),
                completed: false,
            }
        })
        .collect();

    for _ in 0..max_steps {
        if slots.iter().all(|s| s.completed) {
            break;
        }

        let mut step_tokens = vec![eot_id; batch];
        for (b, slot) in slots.iter().enumerate() {
            if slot.completed {
                continue;
            }
            let row = batched_logits_row(&slot.logits, 0, 1, 1, vocab);
            let (tok, _) = topk_tokens_row(&mut logits_scratch, row, 1)
                .into_iter()
                .next()
                .unwrap_or((eot_id, 0.0));
            step_tokens[b] = tok;
        }

        let step_logits = decode_step_batch(&step_tokens, &mut cache)?;

        let mut per_region_cands: Vec<Vec<(BatchedBeamSlot, usize)>> = vec![Vec::new(); n_regions];

        for (b, slot) in slots.iter().enumerate() {
            if slot.completed {
                per_region_cands[slot.region].push((slot.clone(), b));
                continue;
            }
            let row = batched_logits_row(&step_logits, b, batch, 1, vocab);
            for (tok, logp) in topk_tokens_row(&mut logits_scratch, row, beam_size) {
                if slot.suffix.is_empty() && tok == eot_id {
                    continue;
                }
                let mut suffix = slot.suffix.clone();
                suffix.push(tok);
                per_region_cands[slot.region].push((
                    BatchedBeamSlot {
                        region: slot.region,
                        suffix,
                        score: slot.score + logp,
                        logits: row.to_vec(),
                        completed: tok == eot_id,
                    },
                    b,
                ));
            }
        }

        let mut reorder_idx = vec![0usize; batch];
        let mut next_slots = Vec::with_capacity(batch);
        for (ri, cands) in per_region_cands.iter_mut().enumerate() {
            cands.sort_by(|a, b| {
                b.0.score
                    .partial_cmp(&a.0.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            cands.truncate(beam_size);
            while cands.len() < beam_size {
                let pad = slots[ri * beam_size].clone();
                cands.push((pad, ri * beam_size));
            }
            for (bi, (cand, parent)) in cands.iter().enumerate() {
                let dst = ri * beam_size + bi;
                reorder_idx[dst] = *parent;
                next_slots.push(cand.clone());
            }
        }
        slots = next_slots;
        cache = reorder_kv_beams(&cache, &reorder_idx, batch).map_err(|e| anyhow::anyhow!(e))?;
    }

    let mut out = vec![Vec::new(); n_regions];
    for ri in 0..n_regions {
        out[ri] = slots
            .iter()
            .filter(|s| s.region == ri)
            .max_by(|a, b| {
                a.score
                    .partial_cmp(&b.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|s| s.suffix.clone())
            .unwrap_or_default();
    }
    Ok(out)
}

/// Standard log-sum beam search over a closure that returns next-token log-probs.
pub fn beam_search_decode(
    mut next_logprobs: impl FnMut(&[u32]) -> Vec<f32>,
    eot_id: u32,
    beam_size: usize,
    max_steps: usize,
) -> Vec<u32> {
    let beam_size = beam_size.max(1);
    let mut beams = vec![BeamHypothesis {
        tokens: Vec::new(),
        score: 0.0,
        completed: false,
    }];

    for _ in 0..max_steps {
        let mut candidates = Vec::new();
        for beam in &beams {
            if beam.completed {
                candidates.push(beam.clone());
                continue;
            }
            let mut lp = next_logprobs(&beam.tokens);
            log_softmax_last(&mut lp);
            let mut indexed: Vec<(usize, f32)> = lp.into_iter().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            for (tok, logp) in indexed.into_iter().take(beam_size) {
                let mut t = beam.tokens.clone();
                t.push(tok as u32);
                candidates.push(BeamHypothesis {
                    tokens: t,
                    score: beam.score + logp,
                    completed: tok as u32 == eot_id,
                });
            }
        }
        candidates.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        candidates.truncate(beam_size);
        beams = candidates;
        if beams.iter().all(|b| b.completed) {
            break;
        }
    }
    beams
        .into_iter()
        .max_by(|a, b| {
            a.score
                .partial_cmp(&b.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|b| b.tokens)
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn topk_heap_matches_sort_on_toy_vocab() {
        let logits = [0.1f32, 3.0, 1.0, 2.0, 0.5];
        let mut scratch = logits.to_vec();
        let heap = topk_tokens_row(&mut scratch, &logits, 3);
        let mut full = logits.to_vec();
        let sort = topk_tokens(&mut full, 3);
        assert_eq!(heap.len(), sort.len());
        for ((hi, _), (si, _)) in heap.iter().zip(sort.iter()) {
            assert_eq!(hi, si);
        }
    }
}
