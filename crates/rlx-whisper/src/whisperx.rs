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

//! WhisperX 3.8.6's batched transcription of one VAD chunk
//! (`WhisperModel.generate_segment_batched`): the chunk zero-padded to 30 s
//! and turned into a log-mel, one CTranslate2 `generate` call with
//! `[sot, lang, transcribe, notimestamps]`, `beam_size 5, patience 1,
//! length_penalty 1, suppress_tokens [-1], suppress_blank`, and the text
//! decoded from the ids below `eot`.
//!
//! [`ct2_beam_search`] is CTranslate2 4.8's `BeamSearch::search` for that
//! configuration: `2 * beam` candidates per step from the flattened
//! cumulative log-probabilities; an EOS candidate among the first `beam`
//! registers a finished hypothesis and is replaced by the next non-EOS
//! candidate; the batch stops once `round(beam * patience)` hypotheses are
//! finished (or at `max_length`); hypotheses are scored `cum / len^lp`
//! without the EOS and the best is returned. Suppressed ids get
//! `f32::MIN` before the log-softmax, `suppress_ids_begin` only at step 0.

use crate::batch::reorder_kv_beams;
use crate::runner::WhisperRunner;
use anyhow::{Context, Result};
use std::collections::HashMap;

/// `config.json` of the CTranslate2 model (`suppress_ids`,
/// `suppress_ids_begin`) — the defaults `suppress_tokens=[-1]` and
/// `suppress_blank=True` expand to.
pub struct Ct2Suppression {
    pub ids: Vec<u32>,
    pub begin: Vec<u32>,
}

impl Ct2Suppression {
    pub fn from_config(json: &str) -> Result<Self> {
        let v: serde_json::Value = serde_json::from_str(json)?;
        let list = |k: &str| -> Result<Vec<u32>> {
            Ok(v[k]
                .as_array()
                .with_context(|| format!("config has no {k}"))?
                .iter()
                .map(|x| x.as_u64().unwrap() as u32)
                .collect())
        };
        Ok(Self {
            ids: list("suppress_ids")?,
            begin: list("suppress_ids_begin")?,
        })
    }

    /// The same lists from the HuggingFace `generation_config.json`
    /// (`suppress_tokens`, `begin_suppress_tokens`), which CTranslate2's
    /// converter copies into `suppress_ids` / `suppress_ids_begin`.
    pub fn from_generation_config(json: &str) -> Result<Self> {
        let v: serde_json::Value = serde_json::from_str(json)?;
        let list = |k: &str| -> Result<Vec<u32>> {
            Ok(v[k]
                .as_array()
                .with_context(|| format!("generation_config has no {k}"))?
                .iter()
                .map(|x| x.as_u64().unwrap() as u32)
                .collect())
        };
        Ok(Self {
            ids: list("suppress_tokens")?,
            begin: list("begin_suppress_tokens")?,
        })
    }
}

#[derive(Clone, Debug)]
pub struct BeamResult {
    /// Generated ids without the prompt and without `eot`.
    pub ids: Vec<u32>,
    /// `cum_logprob / len^length_penalty`.
    pub score: f32,
}

fn log_softmax(row: &mut [f32]) {
    let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let s: f32 = row.iter().map(|&z| (z - m).exp()).sum();
    let ls = m + s.ln();
    for z in row.iter_mut() {
        *z -= ls;
    }
}

/// The `k` best `(index, value)` of `v`, best first (ties: lower index).
fn topk(v: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    let k = k.min(v.len());
    idx.select_nth_unstable_by(k - 1, |&a, &b| {
        v[b].partial_cmp(&v[a]).unwrap().then(a.cmp(&b))
    });
    idx.truncate(k);
    idx.sort_by(|&a, &b| v[b].partial_cmp(&v[a]).unwrap().then(a.cmp(&b)));
    idx.into_iter().map(|i| (i, v[i])).collect()
}

/// CTranslate2's beam search over `runner`'s decoder for one encoded chunk.
#[allow(clippy::too_many_arguments)]
pub fn ct2_beam_search(
    runner: &mut WhisperRunner,
    enc: &[f32],
    prompt: &[u32],
    beam: usize,
    patience: f32,
    length_penalty: f32,
    max_length: usize,
    sup: &Ct2Suppression,
    eot: u32,
) -> Result<BeamResult> {
    let vocab = runner.config().vocab_size;
    // Cross-attention K/V once, then replicated per beam (CT2 projects the
    // encoder output once and replicates the decoder state).
    let dbg = std::env::var_os("WHISPERX_TIMING").is_some();
    let t0 = std::time::Instant::now();
    let one = runner.cross_cache_batch(enc, 1)?;
    let cross = crate::cache::WhisperCrossCache {
        past_len: one.past_len,
        layers_kv_base: one.layers_kv_base.clone(),
        layers_k: one.layers_k.iter().map(|k| k.repeat(beam)).collect(),
        layers_v: one.layers_v.iter().map(|v| v.repeat(beam)).collect(),
    };
    let t1 = std::time::Instant::now();
    let (prefill, mut cache) = runner.prefill_prompt(&cross, prompt, beam)?;
    if dbg {
        eprintln!(
            "   cross {:.2}s prefill {:.2}s",
            (t1 - t0).as_secs_f64(),
            t1.elapsed().as_secs_f64()
        );
    }
    let num_candidates = 2 * beam;
    let max_candidates = (beam as f32 * patience).round() as usize;
    // CT2: max_length = min(448 / 2, 448 - start_step) generated steps.
    let start_step = prompt.len() - 1;
    let max_steps = (max_length / 2).min(max_length - start_step);
    let mut finished: Vec<(Vec<u32>, f32)> = Vec::new();
    // Live beams: (tokens, cumulative score).
    let mut alive: Vec<(Vec<u32>, f32)> = Vec::new();
    let mut logits = prefill[(prompt.len() - 1) * vocab..prompt.len() * vocab].to_vec(); // slot 0, last position
    for step in 0..max_steps {
        let expanded = step > 0;
        let nrows = if expanded { beam } else { 1 };
        let mut lp = Vec::with_capacity(nrows * vocab);
        for r in 0..nrows {
            let mut row = logits[r * vocab..(r + 1) * vocab].to_vec();
            for &t in &sup.ids {
                row[t as usize] = f32::MIN;
            }
            if step == 0 {
                for &t in &sup.begin {
                    row[t as usize] = f32::MIN;
                }
            }
            log_softmax(&mut row);
            if expanded {
                let s = alive[r].1;
                for z in row.iter_mut() {
                    *z += s;
                }
            }
            lp.extend(row);
        }
        let cands: Vec<(usize, u32, f32)> = topk(&lp, num_candidates)
            .into_iter()
            .map(|(flat, sc)| (flat / vocab, (flat % vocab) as u32, sc))
            .collect();
        let is_last = step + 1 == max_steps;
        let mut secondary = beam;
        let mut next: Vec<(usize, u32, f32)> = Vec::with_capacity(beam);
        for k in 0..beam {
            let (origin, id, sc) = cands[k];
            let mut pick = k;
            if id == eot || is_last {
                let mut toks = if expanded {
                    alive[origin].0.clone()
                } else {
                    Vec::new()
                };
                if id != eot {
                    toks.push(id);
                }
                finished.push((toks, sc));
                for (j, c) in cands.iter().enumerate().skip(secondary) {
                    if c.1 != eot {
                        pick = j;
                        secondary = j + 1;
                        break;
                    }
                }
            }
            next.push(cands[pick]);
        }
        if is_last || finished.len() >= max_candidates {
            break;
        }
        let parents: Vec<usize> = next
            .iter()
            .map(|c| if expanded { c.0 } else { 0 })
            .collect();
        alive = next
            .iter()
            .map(|&(o, id, sc)| {
                let mut t = if expanded {
                    alive[o].0.clone()
                } else {
                    Vec::new()
                };
                t.push(id);
                (t, sc)
            })
            .collect();
        cache = reorder_kv_beams(&cache, &parents, beam).map_err(|e| anyhow::anyhow!(e))?;
        let toks: Vec<u32> = next.iter().map(|c| c.1).collect();
        let ts = std::time::Instant::now();
        logits = runner.decode_step_batch(&cross, &toks, &mut cache, beam, false)?;
        if dbg {
            eprintln!("   step {step} {:.2}s", ts.elapsed().as_secs_f64());
        }
    }
    let mut scored: Vec<(Vec<u32>, f32)> = finished
        .into_iter()
        .map(|(t, s)| {
            let l = t.len() as f32;
            (t, s / l.powf(length_penalty))
        })
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let (ids, score) = scored
        .into_iter()
        .next()
        .context("beam search produced no hypothesis")?;
    Ok(BeamResult { ids, score })
}

/// Byte-level BPE decoding of token ids (`tokenizers`' `ByteLevel` decoder),
/// from `tokenizer.json`'s vocabulary — no regex, no `tokenizers` crate.
pub struct ByteLevelDecoder {
    tokens: HashMap<u32, String>,
    byte_of: HashMap<char, u8>,
}

impl ByteLevelDecoder {
    pub fn from_tokenizer_json(json: &str) -> Result<Self> {
        let v: serde_json::Value = serde_json::from_str(json)?;
        let mut tokens = HashMap::new();
        for (tok, id) in v["model"]["vocab"]
            .as_object()
            .context("tokenizer.json: model.vocab")?
        {
            tokens.insert(id.as_u64().unwrap() as u32, tok.clone());
        }
        for t in v["added_tokens"].as_array().into_iter().flatten() {
            tokens.insert(
                t["id"].as_u64().unwrap() as u32,
                t["content"].as_str().unwrap().to_string(),
            );
        }
        // GPT-2 `bytes_to_unicode`.
        let mut bs: Vec<u32> = (b'!' as u32..=b'~' as u32)
            .chain(0xA1..=0xAC)
            .chain(0xAE..=0xFF)
            .collect();
        let mut cs = bs.clone();
        let mut n = 0;
        for b in 0..256u32 {
            if !bs.contains(&b) {
                bs.push(b);
                cs.push(256 + n);
                n += 1;
            }
        }
        let byte_of = bs
            .iter()
            .zip(&cs)
            .map(|(&b, &c)| (char::from_u32(c).unwrap(), b as u8))
            .collect();
        Ok(Self { tokens, byte_of })
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for id in ids {
            if let Some(t) = self.tokens.get(id) {
                for ch in t.chars() {
                    match self.byte_of.get(&ch) {
                        Some(&b) => bytes.push(b),
                        None => bytes.extend(ch.to_string().as_bytes()),
                    }
                }
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

/// [`ct2_beam_search`] over the host [`FastDecoder`](crate::fast_decoder::FastDecoder),
/// given the chunk's cross-attention K/V (`[enc_seq * d]` per layer, as
/// `WhisperRunner::cross_cache_batch(enc, 1)` returns them).
#[allow(clippy::too_many_arguments)]
pub fn ct2_beam_search_fast(
    dec: &crate::fast_decoder::FastDecoder,
    cross_k: &[Vec<f32>],
    cross_v: &[Vec<f32>],
    prompt: &[u32],
    beam: usize,
    patience: f32,
    length_penalty: f32,
    max_length: usize,
    sup: &Ct2Suppression,
    eot: u32,
) -> Result<BeamResult> {
    use crate::fast_decoder::SelfCache;
    let vocab = dec.vocab;
    let num_candidates = 2 * beam;
    let max_candidates = (beam as f32 * patience).round() as usize;
    let start_step = prompt.len() - 1;
    let max_steps = (max_length / 2).min(max_length - start_step);
    // Prompt: one beam until the first search step (CT2 expands after it).
    let cross = dec.prepare_cross(cross_k, cross_v);
    let mut cache = SelfCache::new(dec.num_layers(), 1);
    let mut logits = Vec::new();
    for &t in prompt {
        logits = dec.step(&[t], &mut cache, &cross);
    }
    let mut finished: Vec<(Vec<u32>, f32)> = Vec::new();
    let mut alive: Vec<(Vec<u32>, f32)> = Vec::new();
    for step in 0..max_steps {
        let expanded = step > 0;
        let nrows = if expanded { beam } else { 1 };
        let mut lp = Vec::with_capacity(nrows * vocab);
        for r in 0..nrows {
            let mut row = logits[r * vocab..(r + 1) * vocab].to_vec();
            for &t in &sup.ids {
                row[t as usize] = f32::MIN;
            }
            if step == 0 {
                for &t in &sup.begin {
                    row[t as usize] = f32::MIN;
                }
            }
            log_softmax(&mut row);
            if expanded {
                let s = alive[r].1;
                for z in row.iter_mut() {
                    *z += s;
                }
            }
            lp.extend(row);
        }
        let cands: Vec<(usize, u32, f32)> = topk(&lp, num_candidates)
            .into_iter()
            .map(|(flat, sc)| (flat / vocab, (flat % vocab) as u32, sc))
            .collect();
        let is_last = step + 1 == max_steps;
        let mut secondary = beam;
        let mut next: Vec<(usize, u32, f32)> = Vec::with_capacity(beam);
        for k in 0..beam {
            let (origin, id, sc) = cands[k];
            let mut pick = k;
            if id == eot || is_last {
                let mut toks = if expanded {
                    alive[origin].0.clone()
                } else {
                    Vec::new()
                };
                if id != eot {
                    toks.push(id);
                }
                finished.push((toks, sc));
                for (j, c) in cands.iter().enumerate().skip(secondary) {
                    if c.1 != eot {
                        pick = j;
                        secondary = j + 1;
                        break;
                    }
                }
            }
            next.push(cands[pick]);
        }
        if is_last || finished.len() >= max_candidates {
            break;
        }
        let parents: Vec<usize> = next
            .iter()
            .map(|c| if expanded { c.0 } else { 0 })
            .collect();
        alive = next
            .iter()
            .map(|&(o, id, sc)| {
                let mut t = if expanded {
                    alive[o].0.clone()
                } else {
                    Vec::new()
                };
                t.push(id);
                (t, sc)
            })
            .collect();
        cache.reorder(&parents);
        let toks: Vec<u32> = next.iter().map(|c| c.1).collect();
        logits = dec.step(&toks, &mut cache, &cross);
    }
    let mut scored: Vec<(Vec<u32>, f32)> = finished
        .into_iter()
        .map(|(t, s)| {
            let l = t.len() as f32;
            (t, s / l.powf(length_penalty))
        })
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let (ids, score) = scored
        .into_iter()
        .next()
        .context("beam search produced no hypothesis")?;
    Ok(BeamResult { ids, score })
}
