// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0

//! Delay-pattern AR generate (Zyphra/Zonos `model.py::generate`).

use anyhow::{Context, Result, bail};

use crate::backbone::{self, BackboneState, KvCache};
use crate::conditioner::{CondOpts, PrefixConditioner};
use crate::config::{CODEBOOK_SIZE, EOS_TOKEN_ID, MASKED_TOKEN_ID, N_CODEBOOKS, ZonosFileConfig};
use crate::delay::{apply_delay_pattern, revert_delay_pattern};
use crate::engine::BackboneEngine;
use crate::ops::{argmax, linear, softmax_inplace};
use crate::weights::WeightMap;

#[derive(Debug, Clone)]
pub struct GenerateOpts {
    pub max_new_tokens: usize,
    pub cfg_scale: f32,
    pub greedy: bool,
    pub seed: u64,
    pub min_p: f32,
    pub temperature: f32,
    pub repetition_penalty: f32,
    pub cond: CondOpts,
}

impl Default for GenerateOpts {
    fn default() -> Self {
        Self {
            max_new_tokens: 86 * 8,
            cfg_scale: 2.0,
            greedy: false,
            seed: 1337,
            min_p: 0.1,
            temperature: 1.0,
            repetition_penalty: 3.0,
            cond: CondOpts::default(),
        }
    }
}

/// Generate aligned DAC codes using a compiled (device) backbone engine.
pub fn generate_codes_compiled(
    cfg: &ZonosFileConfig,
    w: &WeightMap,
    engine: &mut BackboneEngine,
    phoneme_ids: &[i64],
    opts: &GenerateOpts,
) -> Result<Vec<Vec<i64>>> {
    if (opts.cfg_scale - 1.0).abs() < 1e-6 {
        bail!("cfg_scale=1 not supported (Zonos needs CFG pairs)");
    }
    let d = cfg.backbone.d_model;
    let eps = cfg.backbone.norm_epsilon;
    engine.reset();

    let cond_mod = PrefixConditioner::new(d, eps);
    let prefix_c = cond_mod.forward(w, phoneme_ids, &opts.cond, false)?;
    let prefix_u = cond_mod.forward(w, phoneme_ids, &opts.cond, true)?;
    let prefix_len = prefix_c.len() / d;
    anyhow::ensure!(prefix_u.len() / d == prefix_len);

    let prefix_audio_len = 0usize;
    let audio_seq_len = prefix_audio_len + opts.max_new_tokens;
    let mut rng = Rng::new(opts.seed);

    let unknown = -1i64;
    let codes = vec![vec![unknown; audio_seq_len]; N_CODEBOOKS];
    let mut delayed = apply_delay_pattern(&codes, MASKED_TOKEN_ID);

    let prefix_codes: Vec<Vec<i64>> = delayed
        .iter()
        .map(|row| row[..prefix_audio_len + 1].to_vec())
        .collect();
    let emb = embed_codes(w, &prefix_codes, d)?;
    let audio_t = prefix_audio_len + 1;

    // Prefill: prefix (cond≠uncond) + first delayed audio frame (shared).
    let pre_seq = prefix_len + audio_t;
    let mut cond_seq = Vec::with_capacity(pre_seq * d);
    let mut uncond_seq = Vec::with_capacity(pre_seq * d);
    cond_seq.extend_from_slice(&prefix_c);
    uncond_seq.extend_from_slice(&prefix_u);
    for t in 0..audio_t {
        let row = &emb[t * d..(t + 1) * d];
        cond_seq.extend_from_slice(row);
        uncond_seq.extend_from_slice(row);
    }
    let mut last = engine.prefill_pair(&cond_seq, &uncond_seq, pre_seq)?;

    let mut logits = apply_heads_cfg(w, &last, d, opts.cfg_scale)?;
    let mut next = sample_tokens(&logits, opts, None, &mut rng)?;
    // Prefill sample lands at `offset`. Decode loop feeds `delayed[offset - 1]`
    // then writes `delayed[offset]` (RLX layout). Zyphra bumps offset first; that
    // schedule goes near-silent on long phoneme prefixes with our backbone, so we
    // keep this order which matches audible Metal/MLX/eager output.
    let mut offset = prefix_audio_len + 1;
    write_unknown_frame(&mut delayed, offset, &next);

    let min_frames = min_frames_before_eos(phoneme_ids.len(), opts.cond.speaking_rate);
    let mut remaining_steps = delayed[0].len() - offset;
    let mut stopping = false;

    while remaining_steps > 0 {
        let frame_in: Vec<Vec<i64>> = delayed.iter().map(|row| vec![row[offset - 1]]).collect();
        let emb1 = embed_codes(w, &frame_in, d)?;
        last = engine.step_cfg(&emb1, &emb1)?;
        logits = apply_heads_cfg(w, &last, d, opts.cfg_scale)?;
        for q in 1..N_CODEBOOKS {
            logits[q * (CODEBOOK_SIZE + 1) + EOS_TOKEN_ID as usize] = f32::NEG_INFINITY;
        }

        let gen_so_far: Option<Vec<Vec<i64>>> = if opts.greedy {
            None
        } else {
            Some(delayed.iter().map(|row| row[..offset].to_vec()).collect())
        };
        next = sample_tokens(&logits, opts, gen_so_far.as_deref(), &mut rng)?;

        let audio_frames = offset.saturating_sub(N_CODEBOOKS);
        if next[0] == EOS_TOKEN_ID {
            if !stopping && audio_frames < min_frames {
                // Greedy often emits EOS early on long text; hold off until the
                // phoneme-derived duration floor so the ending is not truncated.
                next[0] =
                    argmax_excluding(&logits[..CODEBOOK_SIZE + 1], EOS_TOKEN_ID as usize) as i64;
            } else {
                remaining_steps = remaining_steps.min(N_CODEBOOKS);
                stopping = true;
            }
        }
        if stopping {
            let idx = (N_CODEBOOKS - remaining_steps).min(N_CODEBOOKS - 1);
            for q in 0..N_CODEBOOKS {
                if q < idx {
                    next[q] = MASKED_TOKEN_ID;
                } else if q == idx {
                    next[q] = EOS_TOKEN_ID;
                }
            }
        }

        write_unknown_frame(&mut delayed, offset, &next);
        offset += 1;
        remaining_steps -= 1;
    }

    if stopping {
        eprintln!(
            "zonos: AR stop=eos offset={offset} budget={} min_frames={min_frames}",
            opts.max_new_tokens
        );
    } else {
        eprintln!(
            "zonos: AR stop=budget offset={offset} max_tokens={} — raise --max-tokens if audio cuts off",
            opts.max_new_tokens
        );
    }

    finalize_codes(&delayed, offset)
}

/// Eager host path (reference / `RLX_ZONOS_EAGER=1`).
pub fn generate_codes_eager(
    cfg: &ZonosFileConfig,
    w: &WeightMap,
    phoneme_ids: &[i64],
    opts: &GenerateOpts,
) -> Result<Vec<Vec<i64>>> {
    if (opts.cfg_scale - 1.0).abs() < 1e-6 {
        bail!("cfg_scale=1 not supported (Zonos needs CFG pairs)");
    }
    let d = cfg.backbone.d_model;
    let eps = cfg.backbone.norm_epsilon;
    let head_dim = cfg.head_dim();
    let n_kv = cfg.backbone.attn_cfg.num_heads_kv;
    let n_layer = cfg.backbone.n_layer;

    let cond_mod = PrefixConditioner::new(d, eps);
    let prefix_c = cond_mod.forward(w, phoneme_ids, &opts.cond, false)?;
    let prefix_u = cond_mod.forward(w, phoneme_ids, &opts.cond, true)?;
    let prefix_len = prefix_c.len() / d;
    anyhow::ensure!(prefix_u.len() / d == prefix_len);

    let batch = 2usize;
    let prefix_audio_len = 0usize;
    let audio_seq_len = prefix_audio_len + opts.max_new_tokens;
    let max_seq = prefix_len + audio_seq_len + N_CODEBOOKS + 8;

    let mut cache = KvCache::new(n_layer, batch, max_seq, n_kv, head_dim);
    let mut state = BackboneState::new(head_dim);
    let mut rng = Rng::new(opts.seed);

    let unknown = -1i64;
    let codes = vec![vec![unknown; audio_seq_len]; N_CODEBOOKS];
    let mut delayed = apply_delay_pattern(&codes, MASKED_TOKEN_ID);

    let prefix_codes: Vec<Vec<i64>> = delayed
        .iter()
        .map(|row| row[..prefix_audio_len + 1].to_vec())
        .collect();
    let emb = embed_codes(w, &prefix_codes, d)?;
    let audio_t = prefix_audio_len + 1;
    let seq = prefix_len + audio_t;
    let mut hidden = vec![0.0f32; batch * seq * d];
    for (bi, prefix) in [prefix_c.as_slice(), prefix_u.as_slice()]
        .into_iter()
        .enumerate()
    {
        let base = bi * seq * d;
        hidden[base..base + prefix_len * d].copy_from_slice(prefix);
        for t in 0..audio_t {
            let dst = base + (prefix_len + t) * d;
            let src = t * d;
            hidden[dst..dst + d].copy_from_slice(&emb[src..src + d]);
        }
    }

    let last = backbone::forward_last(cfg, w, &hidden, batch, seq, &mut cache, &state)?;
    state.seqlen_offset = seq;

    let mut logits = apply_heads_cfg(w, &last, d, opts.cfg_scale)?;
    let mut next = sample_tokens(&logits, opts, None, &mut rng)?;
    // Same schedule as the compiled path (see comment there).
    let mut offset = prefix_audio_len + 1;
    write_unknown_frame(&mut delayed, offset, &next);

    let min_frames = min_frames_before_eos(phoneme_ids.len(), opts.cond.speaking_rate);
    let mut remaining_steps = delayed[0].len() - offset;
    let mut stopping = false;

    while remaining_steps > 0 {
        let frame_in: Vec<Vec<i64>> = delayed.iter().map(|row| vec![row[offset - 1]]).collect();
        let emb1 = embed_codes(w, &frame_in, d)?;
        let mut h1 = vec![0.0f32; batch * d];
        h1[..d].copy_from_slice(&emb1);
        h1[d..].copy_from_slice(&emb1);
        let last = backbone::forward_last(cfg, w, &h1, batch, 1, &mut cache, &state)?;
        state.seqlen_offset += 1;

        logits = apply_heads_cfg(w, &last, d, opts.cfg_scale)?;
        for q in 1..N_CODEBOOKS {
            logits[q * (CODEBOOK_SIZE + 1) + EOS_TOKEN_ID as usize] = f32::NEG_INFINITY;
        }

        let gen_so_far: Option<Vec<Vec<i64>>> = if opts.greedy {
            None
        } else {
            Some(delayed.iter().map(|row| row[..offset].to_vec()).collect())
        };
        next = sample_tokens(&logits, opts, gen_so_far.as_deref(), &mut rng)?;

        let audio_frames = offset.saturating_sub(N_CODEBOOKS);
        if next[0] == EOS_TOKEN_ID {
            if !stopping && audio_frames < min_frames {
                next[0] =
                    argmax_excluding(&logits[..CODEBOOK_SIZE + 1], EOS_TOKEN_ID as usize) as i64;
            } else {
                remaining_steps = remaining_steps.min(N_CODEBOOKS);
                stopping = true;
            }
        }
        if stopping {
            let idx = (N_CODEBOOKS - remaining_steps).min(N_CODEBOOKS - 1);
            for q in 0..N_CODEBOOKS {
                if q < idx {
                    next[q] = MASKED_TOKEN_ID;
                } else if q == idx {
                    next[q] = EOS_TOKEN_ID;
                }
            }
        }

        write_unknown_frame(&mut delayed, offset, &next);
        offset += 1;
        remaining_steps -= 1;
    }

    if stopping {
        eprintln!(
            "zonos: AR stop=eos offset={offset} budget={} min_frames={min_frames}",
            opts.max_new_tokens
        );
    } else {
        eprintln!(
            "zonos: AR stop=budget offset={offset} max_tokens={} — raise --max-tokens if audio cuts off",
            opts.max_new_tokens
        );
    }

    finalize_codes(&delayed, offset)
}

/// Lower bound on aligned DAC frames before codebook-0 EOS is accepted.
///
/// Uses the speaking-rate dial (phonemes × 86 / rate). A 0.9× floor still let
/// greedy/sample EOS in the mush zone right after the hold; 1.05× keeps the
/// tail through the last content words.
pub(crate) fn min_frames_before_eos(phoneme_len: usize, speaking_rate: f32) -> usize {
    let rate = speaking_rate.clamp(5.0, 40.0);
    let frames = ((phoneme_len as f32) * (86.0 / rate) * 1.05).ceil() as usize;
    frames.max(32)
}

fn argmax_excluding(row: &[f32], ban: usize) -> usize {
    let mut best_i = if ban == 0 { 1 } else { 0 };
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in row.iter().enumerate() {
        if i == ban {
            continue;
        }
        if v > best_v {
            best_v = v;
            best_i = i;
        }
    }
    best_i
}

fn finalize_codes(delayed: &[Vec<i64>], offset: usize) -> Result<Vec<Vec<i64>>> {
    let mut aligned = revert_delay_pattern(delayed);
    let t_out = offset.saturating_sub(N_CODEBOOKS);
    for q in 0..N_CODEBOOKS {
        if aligned[q].len() > t_out {
            aligned[q].truncate(t_out);
        }
        for v in &mut aligned[q] {
            if *v >= 1024 {
                *v = 0;
            }
        }
    }
    Ok(aligned)
}

fn write_unknown_frame(delayed: &mut [Vec<i64>], t: usize, next: &[i64]) {
    for q in 0..N_CODEBOOKS {
        if delayed[q][t] == -1 {
            delayed[q][t] = next[q];
        }
    }
}

fn embed_codes(w: &WeightMap, codes: &[Vec<i64>], d: usize) -> Result<Vec<f32>> {
    anyhow::ensure!(codes.len() == N_CODEBOOKS, "n_q");
    let t = codes[0].len();
    let mut out = vec![0.0f32; t * d];
    for q in 0..N_CODEBOOKS {
        let emb = w
            .get(&format!("embeddings.{q}.weight"))
            .with_context(|| format!("embeddings.{q}"))?;
        let vocab = w.shape(&format!("embeddings.{q}.weight"))?[0];
        for ti in 0..t {
            let id = codes[q][ti];
            if id < 0 {
                bail!("embed negative id at q={q} t={ti}");
            }
            let id = id as usize;
            if id >= vocab {
                bail!("code {id} >= emb vocab {vocab}");
            }
            let src = &emb[id * d..(id + 1) * d];
            let dst = &mut out[ti * d..(ti + 1) * d];
            for i in 0..d {
                dst[i] += src[i];
            }
        }
    }
    Ok(out)
}

fn apply_heads_cfg(w: &WeightMap, last: &[f32], d: usize, cfg_scale: f32) -> Result<Vec<f32>> {
    let cond = &last[..d];
    let uncond = &last[d..];
    let vocab = CODEBOOK_SIZE + 1;
    let mut logits = vec![0.0f32; N_CODEBOOKS * vocab];
    for q in 0..N_CODEBOOKS {
        let hw = w.get(&format!("heads.{q}.weight"))?;
        let lc = linear(cond, hw, None, 1, vocab, d);
        let lu = linear(uncond, hw, None, 1, vocab, d);
        for i in 0..vocab {
            logits[q * vocab + i] = lu[i] + (lc[i] - lu[i]) * cfg_scale;
        }
    }
    Ok(logits)
}

fn sample_tokens(
    logits: &[f32],
    opts: &GenerateOpts,
    generated: Option<&[Vec<i64>]>,
    rng: &mut Rng,
) -> Result<Vec<i64>> {
    let vocab = CODEBOOK_SIZE + 1;
    let mut out = vec![0i64; N_CODEBOOKS];
    if opts.greedy {
        for q in 0..N_CODEBOOKS {
            let row = &logits[q * vocab..(q + 1) * vocab];
            out[q] = argmax(row) as i64;
        }
        return Ok(out);
    }
    for q in 0..N_CODEBOOKS {
        let mut row = logits[q * vocab..(q + 1) * vocab].to_vec();
        if opts.repetition_penalty != 1.0 {
            if let Some(g) = generated {
                let window = 2usize.min(g[q].len());
                let start = g[q].len() - window;
                for &tok in &g[q][start..] {
                    let t = tok.clamp(0, (vocab - 1) as i64) as usize;
                    if row[t] <= 0.0 {
                        row[t] *= opts.repetition_penalty;
                    } else {
                        row[t] /= opts.repetition_penalty;
                    }
                }
            }
        }
        let temp = opts.temperature.max(1e-5);
        for v in &mut row {
            *v /= temp;
        }
        softmax_inplace(&mut row);
        if opts.min_p > 0.0 {
            let top = row.iter().copied().fold(0.0f32, f32::max);
            let thr = opts.min_p * top;
            let mut sum = 0.0f32;
            for v in &mut row {
                if *v < thr {
                    *v = 0.0;
                }
                sum += *v;
            }
            if sum > 0.0 {
                for v in &mut row {
                    *v /= sum;
                }
            }
        }
        out[q] = sample_multinomial(&row, rng) as i64;
    }
    Ok(out)
}

fn sample_multinomial(probs: &[f32], rng: &mut Rng) -> usize {
    let r = rng.f32();
    let mut acc = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        acc += p;
        if r <= acc {
            return i;
        }
    }
    probs.len().saturating_sub(1)
}

struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }
    fn f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}
