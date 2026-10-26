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

use crate::cache::WhisperKvCache;
use crate::config::WhisperConfig;
use rlx_search::{BeamConfig, BeamSearch, Constraint, Hypothesis};

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
///
/// When `english_only` is true, language and task tokens are skipped (OpenAI
/// `.en` checkpoints); only SOT (+ optional notimestamps) is used.
pub fn initial_prompt_opts(
    tokenizer: &tokenizers::Tokenizer,
    language: Option<&str>,
    translate: bool,
    timestamps: bool,
) -> anyhow::Result<Vec<u32>> {
    initial_prompt_opts_ex(tokenizer, language, translate, timestamps, false)
}

/// Extended prompt builder with an English-only flag.
pub fn initial_prompt_opts_ex(
    tokenizer: &tokenizers::Tokenizer,
    language: Option<&str>,
    translate: bool,
    timestamps: bool,
    english_only: bool,
) -> anyhow::Result<Vec<u32>> {
    let mut ids = vec![
        tokenizer
            .token_to_id(SOT_TOKEN)
            .ok_or_else(|| anyhow::anyhow!("tokenizer missing {SOT_TOKEN}"))?,
    ];
    if !english_only {
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
    }
    if !timestamps && let Some(id) = tokenizer.token_to_id(NO_TIMESTAMPS_TOKEN) {
        ids.push(id);
    }
    Ok(ids)
}

/// View of batch row `b` in `[batch, dec_seq, vocab]` logits (last time step).
///
/// After the first bucketed decode step, `dec_seq` is `1` even though the token list
/// grows — prefill uses `dec_seq = prompt.len()`.
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
    let Some(mask) = mask else {
        return rlx_search::argmax_u32(logits);
    };
    let mask = Some(mask);
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

    /// Argmax with global + optional begin suppression. (Log-softmax is a
    /// shift, so the argmax of the logits is the argmax of the log-probs.)
    pub fn argmax_next(&self, logits: &mut [f32], at_generation_start: bool) -> u32 {
        if at_generation_start {
            self.apply_begin(logits);
        }
        self.apply(logits);
        rlx_search::argmax_u32(logits)
    }
}

/// Beam-search masks: `suppress_tokens` every step, `begin_suppress_tokens`
/// at the first generated token.
impl Constraint for SuppressionMask {
    fn mask(&mut self, _sequence: &[u32], generated: usize, scores: &mut [f32]) {
        if generated == 0 {
            self.apply_begin(scores);
        }
        self.apply(scores);
    }
}

pub fn argmax_last_logits(logits: &[f32], dec_seq: usize, vocab: usize) -> u32 {
    let last = dec_seq - 1;
    let off = last * vocab;
    argmax_logits(&logits[off..off + vocab])
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

/// A beam result's tokens, ending in `eot` when it finished (as the greedy
/// loop's do).
fn with_eot(h: Option<Hypothesis>, eot: u32) -> Vec<u32> {
    h.map(|h| {
        let mut t = h.tokens;
        if h.finished {
            t.push(eot);
        }
        t
    })
    .unwrap_or_default()
}

/// Beam search with per-hypothesis self-attention KV (no full prefill replay):
/// `openai/whisper`'s rule ([`BeamConfig::whisper`]). `decode_step(token,
/// parent_cache)` runs one step on a copy of the parent's cache, once per kept
/// hypothesis. Returns the generated tokens.
#[allow(clippy::too_many_arguments)]
pub fn beam_search_decode_kv(
    prefill_logits: &[f32],
    prompt_len: usize,
    initial_cache: WhisperKvCache,
    eot_id: u32,
    beam_size: usize,
    max_steps: usize,
    vocab: usize,
    suppression: &mut SuppressionMask,
    mut decode_step: impl FnMut(u32, &WhisperKvCache) -> anyhow::Result<(Vec<f32>, WhisperKvCache)>,
) -> anyhow::Result<Vec<u32>> {
    let cfg = BeamConfig::whisper(beam_size, max_steps, eot_id);
    let first = last_logits_row(prefill_logits, prompt_len, vocab);
    let mut caches = vec![initial_cache];
    let best = rlx_search::beam_search::<anyhow::Error>(cfg, &[], &first, suppression, |step| {
        let mut rows = Vec::with_capacity(step.len());
        let mut next = Vec::with_capacity(step.len());
        for (&parent, &token) in step.parents.iter().zip(step.tokens) {
            let (logits, cache) = decode_step(token, &caches[parent])?;
            rows.push(if logits.len() == vocab {
                logits
            } else {
                last_logits_row(&logits, 1, vocab)
            });
            next.push(cache);
        }
        caches = next;
        Ok(rows)
    })?;
    Ok(with_eot(best.into_iter().next(), eot_id))
}

/// Batched beam search: `batch = n_regions * beam_size`, one GPU forward per time
/// step. Region `r`'s live hypothesis `j` is batch slot `r * beam_size + j`; each
/// step reorders the KV cache to the chosen parents, then feeds the chosen tokens.
#[allow(clippy::too_many_arguments)]
pub fn beam_search_decode_kv_batched(
    prefill_logits: &[f32],
    prompt_len: usize,
    mut cache: WhisperKvCache,
    n_regions: usize,
    beam_size: usize,
    max_steps: usize,
    vocab: usize,
    eot_id: u32,
    suppression: &mut SuppressionMask,
    mut decode_step_batch: impl FnMut(&[u32], &mut WhisperKvCache) -> anyhow::Result<Vec<f32>>,
) -> anyhow::Result<Vec<Vec<u32>>> {
    use crate::batch::reorder_kv_beams;

    let beam_size = beam_size.max(1);
    let batch = n_regions * beam_size;
    let cfg = BeamConfig::whisper(beam_size, max_steps, eot_id);
    let mut searches: Vec<BeamSearch> = (0..n_regions)
        .map(|_| BeamSearch::new(cfg.clone(), &[]))
        .collect();
    // Every slot of a region holds the same prefill; read the first.
    let mut rows: Vec<Vec<Vec<f32>>> = (0..n_regions)
        .map(|r| {
            vec![batched_logits_row_owned(
                prefill_logits,
                r * beam_size,
                batch,
                prompt_len,
                vocab,
            )]
        })
        .collect();

    loop {
        // Idle slots (finished regions, short beams) keep their cache and
        // decode `eot`; their output is never read.
        let mut reorder: Vec<usize> = (0..batch).collect();
        let mut tokens = vec![eot_id; batch];
        let mut stepped = false;
        for (r, bs) in searches.iter_mut().enumerate() {
            if bs.is_done() {
                continue;
            }
            let refs: Vec<&[f32]> = rows[r].iter().map(Vec::as_slice).collect();
            let Some(step) = bs.advance(&refs, suppression) else {
                continue;
            };
            stepped = true;
            for (j, (&parent, &token)) in step.parents.iter().zip(step.tokens).enumerate() {
                reorder[r * beam_size + j] = r * beam_size + parent;
                tokens[r * beam_size + j] = token;
            }
        }
        if !stepped {
            break;
        }
        cache = reorder_kv_beams(&cache, &reorder, batch).map_err(|e| anyhow::anyhow!(e))?;
        let logits = decode_step_batch(&tokens, &mut cache)?;
        for (r, bs) in searches.iter().enumerate() {
            if !bs.is_done() {
                rows[r] = (0..bs.live_len())
                    .map(|j| batched_logits_row_owned(&logits, r * beam_size + j, batch, 1, vocab))
                    .collect();
            }
        }
    }
    Ok(searches
        .into_iter()
        .map(|bs| with_eot(bs.finish().into_iter().next(), eot_id))
        .collect())
}

/// Beam search over a closure returning next-token logits for the generated
/// tokens so far (no KV cache): [`BeamConfig::whisper`].
pub fn beam_search_decode(
    mut next_logits: impl FnMut(&[u32]) -> Vec<f32>,
    eot_id: u32,
    beam_size: usize,
    max_steps: usize,
) -> Vec<u32> {
    let cfg = BeamConfig::whisper(beam_size, max_steps, eot_id);
    let first = next_logits(&[]);
    let best =
        rlx_search::beam_search::<std::convert::Infallible>(cfg, &[], &first, &mut (), |s| {
            Ok((0..s.len()).map(|i| next_logits(s.generated(i))).collect())
        });
    match best {
        Ok(b) => with_eot(b.into_iter().next(), eot_id),
        Err(e) => match e {},
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::cache::WhisperKvCache;

    /// A tiny "decoder" whose logits depend on the generated tokens only, so
    /// the batched and per-hypothesis searches must agree.
    fn logits_for(seq: &[u32], vocab: usize) -> Vec<f32> {
        let mut h = 0x9e37_79b9u64;
        for &t in seq {
            h = (h ^ t as u64).wrapping_mul(0x0100_0000_01b3);
        }
        (0..vocab)
            .map(|i| ((h ^ (i as u64 * 0x2545_f491)) % 1000) as f32 / 97.0)
            .collect()
    }

    /// The KV "cache" stores the generated tokens in `layers_k[0]` (one f32 per
    /// token per batch slot), so a wrong parent reorder shows up as wrong logits.
    fn kv(slots: &[Vec<u32>]) -> WhisperKvCache {
        let width = slots.iter().map(Vec::len).max().unwrap_or(0) + 1;
        let mut k = Vec::new();
        for s in slots {
            k.push(s.len() as f32);
            k.extend(s.iter().map(|&t| t as f32));
            k.resize(k.len() + width - 1 - s.len(), 0.0);
        }
        WhisperKvCache {
            past_len: 0,
            layers_kv_base: vec![0],
            layers_k: vec![k.clone()],
            layers_v: vec![k],
        }
    }

    fn slots_of(cache: &WhisperKvCache, batch: usize) -> Vec<Vec<u32>> {
        let per = cache.layers_k[0].len() / batch;
        (0..batch)
            .map(|b| {
                let row = &cache.layers_k[0][b * per..(b + 1) * per];
                let n = row[0] as usize;
                row[1..=n].iter().map(|&t| t as u32).collect()
            })
            .collect()
    }

    fn mask(vocab: usize) -> SuppressionMask {
        SuppressionMask {
            data: vec![0.0; vocab],
            begin: Vec::new(),
        }
    }

    #[test]
    fn batched_regions_match_the_single_stream_search() {
        let (vocab, beam, eot) = (9usize, 3usize, 4u32);
        let single = beam_search_decode(|g| logits_for(g, vocab), eot, beam, 7);
        let n_regions = 2;
        let batch = n_regions * beam;
        let prefill: Vec<f32> = (0..batch).flat_map(|_| logits_for(&[], vocab)).collect();
        let out = beam_search_decode_kv_batched(
            &prefill,
            1,
            kv(&vec![Vec::new(); batch]),
            n_regions,
            beam,
            7,
            vocab,
            eot,
            &mut mask(vocab),
            |tokens, cache| {
                // Append each slot's token to its (reordered) history.
                let mut slots = slots_of(cache, batch);
                for (s, &t) in slots.iter_mut().zip(tokens) {
                    s.push(t);
                }
                *cache = kv(&slots);
                Ok(slots.iter().flat_map(|s| logits_for(s, vocab)).collect())
            },
        )
        .unwrap();
        assert_eq!(out, vec![single.clone(), single]);
    }

    #[test]
    fn kv_search_matches_the_closure_search() {
        let (vocab, beam, eot) = (8usize, 4usize, 2u32);
        let want = beam_search_decode(|g| logits_for(g, vocab), eot, beam, 6);
        let got = beam_search_decode_kv(
            &logits_for(&[], vocab),
            1,
            kv(&[Vec::new()]),
            eot,
            beam,
            6,
            vocab,
            &mut mask(vocab),
            |token, cache| {
                let mut s = slots_of(cache, 1).remove(0);
                s.push(token);
                Ok((logits_for(&s, vocab), kv(&[s])))
            },
        )
        .unwrap();
        assert_eq!(got, want);
    }

    #[test]
    fn suppressed_tokens_never_appear_in_beam_output() {
        let (vocab, beam, eot) = (8usize, 3usize, 2u32);
        let mut m = mask(vocab);
        m.data[5] = f32::NEG_INFINITY;
        m.data[6] = f32::NEG_INFINITY;
        let got = beam_search_decode_kv(
            &logits_for(&[], vocab),
            1,
            kv(&[Vec::new()]),
            eot,
            beam,
            8,
            vocab,
            &mut m,
            |token, cache| {
                let mut s = slots_of(cache, 1).remove(0);
                s.push(token);
                Ok((logits_for(&s, vocab), kv(&[s])))
            },
        )
        .unwrap();
        assert!(!got.contains(&5) && !got.contains(&6), "{got:?}");
        assert_ne!(got.first(), Some(&eot), "end-of-text can't come first");
    }
}
