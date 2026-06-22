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

//! Eager cross-attention capture + DTW word alignment (OpenAI `timing.py`).
//!
//! Two paths:
//!
//! - **CPU** — full decoder forward through the alignment layer, then batched QK matmul
//!   ([`find_word_alignment`]).
//! - **GPU** — compiled align-hidden graph on Metal/CUDA ([`find_word_alignment_from_hidden`]),
//!   invoked from [`crate::runner::WhisperRunner::apply_word_alignment`] when
//!   `align_device != Cpu`.

use crate::alignment_heads::AlignmentHeads;
use crate::config::WhisperConfig;
use crate::dtw::{dtw, median_filter_1d, split_to_word_tokens};
use crate::timestamp_parse::TOKENS_PER_SECOND;
use crate::transcript::WordTiming;
use crate::weights::WhisperWeightPrefix;
use anyhow::Result;
use rlx_core::weight_map::WeightMap;
use std::collections::HashMap;

fn layer_norm(x: &[f32], w: &[f32], b: &[f32], d: usize) -> Vec<f32> {
    let mut out = vec![0f32; x.len()];
    for t in 0..(x.len() / d) {
        let slice = &x[t * d..(t + 1) * d];
        let mean = slice.iter().sum::<f32>() / d as f32;
        let var = slice.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / d as f32;
        let inv = 1.0 / (var + 1e-5).sqrt();
        for i in 0..d {
            out[t * d + i] = (slice[i] - mean) * inv * w[i] + b[i];
        }
    }
    out
}

fn linear(x: &[f32], w: &[f32], b: Option<&[f32]>, in_f: usize, out_f: usize) -> Vec<f32> {
    let seq = x.len() / in_f;
    let mut y = vec![0f32; seq * out_f];
    for t in 0..seq {
        for o in 0..out_f {
            let mut sum = b.map(|b| b[o]).unwrap_or(0.0);
            for i in 0..in_f {
                sum += x[t * in_f + i] * w[o * in_f + i];
            }
            y[t * out_f + o] = sum;
        }
    }
    y
}

fn gelu(x: &[f32]) -> Vec<f32> {
    x.iter()
        .map(|&v| {
            let c = 0.797_884_6 * (v + 0.044715 * v.powi(3));
            0.5 * v * (1.0 + c.tanh())
        })
        .collect()
}

fn get_w<'a>(wm: &'a WeightMap, key: &str) -> Result<&'a [f32]> {
    wm.get(key)
        .map(|(d, _)| d)
        .ok_or_else(|| anyhow::anyhow!("missing weight {key}"))
}

fn embed_tokens(
    wm: &WeightMap,
    pfx: &WhisperWeightPrefix,
    ids: &[u32],
    pos_emb: &[f32],
    d: usize,
) -> Result<Vec<f32>> {
    let w = get_w(wm, &pfx.dec_embed_tokens())?;
    let mut x = vec![0f32; ids.len() * d];
    for (i, &id) in ids.iter().enumerate() {
        let row = id as usize * d;
        x[i * d..(i + 1) * d].copy_from_slice(&w[row..row + d]);
        for j in 0..d {
            x[i * d + j] += pos_emb[i * d + j];
        }
    }
    Ok(x)
}

fn layer_cross_ln_q(
    x: &[f32],
    wm: &WeightMap,
    pfx: &WhisperWeightPrefix,
    layer: usize,
    d: usize,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let lp = |s: &str| pfx.dec_layer(layer, s);
    let ln_w = get_w(wm, &lp("encoder_attn_layer_norm.weight"))?;
    let ln_b = get_w(wm, &lp("encoder_attn_layer_norm.bias"))?;
    let ln = layer_norm(x, ln_w, ln_b, d);
    let qw = get_w(wm, &lp("encoder_attn.q_proj.weight"))?;
    let qb = wm.get(&lp("encoder_attn.q_proj.bias")).map(|(d, _)| d);
    let q = linear(&ln, qw, qb, d, d);
    Ok((ln, q))
}

fn softmax_row(row: &mut [f32]) {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut denom = 0f32;
    for v in row.iter_mut() {
        *v = (*v - max).exp();
        denom += *v;
    }
    let inv = 1.0 / denom.max(1e-9);
    for v in row.iter_mut() {
        *v *= inv;
    }
}

fn alignment_matrix_from_hidden(
    heads: &AlignmentHeads,
    x: &[f32],
    encoder_hidden: &[f32],
    wm: &WeightMap,
    pfx: &WhisperWeightPrefix,
    sot_len: usize,
    n_text: usize,
    enc_seq: usize,
    d: usize,
    head_dim: usize,
) -> Result<(Vec<f32>, usize)> {
    let n_frames = (enc_seq / 2).max(1);
    let mut matrix = vec![0f32; n_text * n_frames];
    let mut ln_q_cache: HashMap<usize, (Vec<f32>, Vec<f32>)> = HashMap::new();
    let n_align = heads.pairs.len().max(1);
    let scale = (head_dim as f32).powf(-0.25);
    let enc = &encoder_hidden[..enc_seq * d];
    for &(layer, head) in &heads.pairs {
        let q = if let Some((_, q)) = ln_q_cache.get(&layer) {
            q.clone()
        } else {
            let (_, q) = layer_cross_ln_q(x, wm, pfx, layer, d)?;
            ln_q_cache.insert(layer, (Vec::new(), q.clone()));
            q
        };
        for ti in 0..n_text {
            let tok_ix = sot_len + ti;
            let mut row = vec![0f32; enc_seq];
            for f in 0..enc_seq {
                let mut sum = 0f32;
                for hd in 0..head_dim {
                    sum += q[tok_ix * d + head * head_dim + hd]
                        * scale
                        * enc[f * d + head * head_dim + hd]
                        * scale;
                }
                row[f] = sum;
            }
            softmax_row(&mut row);
            for f in 0..n_frames.min(enc_seq) {
                matrix[ti * n_frames + f] += row[f] / n_align as f32;
            }
        }
    }
    Ok((matrix, n_frames))
}

fn word_timings_from_matrix(
    tokenizer: &tokenizers::Tokenizer,
    text_tokens: &[u32],
    matrix: &[f32],
    n_text: usize,
    n_frames: usize,
    time_offset_sec: f32,
    medfilt_width: usize,
) -> Result<Vec<WordTiming>> {
    let mut filtered = matrix.to_vec();
    for ti in 0..n_text {
        let row = &matrix[ti * n_frames..(ti + 1) * n_frames];
        let med = median_filter_1d(row, medfilt_width);
        filtered[ti * n_frames..(ti + 1) * n_frames].copy_from_slice(&med);
    }

    let neg: Vec<f32> = filtered.iter().map(|&v| -v).collect();
    let (text_idx, time_idx) = dtw(&neg, n_text, n_frames);

    let (words, word_token_groups) = split_to_word_tokens(tokenizer, text_tokens);
    if word_token_groups.is_empty() {
        return Ok(Vec::new());
    }

    let jump_times: Vec<f32> = time_idx
        .iter()
        .map(|&fi| fi as f32 / TOKENS_PER_SECOND)
        .collect();

    let mut boundaries = vec![0usize];
    boundaries.extend(
        word_token_groups
            .iter()
            .take(word_token_groups.len().saturating_sub(1))
            .scan(0usize, |acc, g| {
                *acc += g.len();
                Some(*acc)
            }),
    );

    let mut out = Vec::new();
    for (wi, w) in words.iter().enumerate() {
        let b0 = boundaries
            .get(wi)
            .copied()
            .unwrap_or(0)
            .min(text_idx.len().saturating_sub(1));
        let b1 = boundaries
            .get(wi + 1)
            .copied()
            .unwrap_or(n_text)
            .min(text_idx.len().saturating_sub(1));
        let start = jump_times.get(b0).copied().unwrap_or(0.0);
        let end = jump_times.get(b1).copied().unwrap_or(start);
        out.push(WordTiming {
            word: w.clone(),
            start: time_offset_sec + start,
            end: time_offset_sec + end.max(start),
            prob: None,
        });
    }
    Ok(out)
}

/// Word alignment from precomputed decoder hidden states (GPU align-hidden path).
pub fn find_word_alignment_from_hidden(
    tokenizer: &tokenizers::Tokenizer,
    heads: &AlignmentHeads,
    wm: &WeightMap,
    pfx: &WhisperWeightPrefix,
    sot_tokens: &[u32],
    text_tokens: &[u32],
    hidden: &[f32],
    encoder_hidden: &[f32],
    enc_seq: usize,
    d: usize,
    head_dim: usize,
    time_offset_sec: f32,
    medfilt_width: usize,
) -> Result<Vec<WordTiming>> {
    if text_tokens.is_empty() {
        return Ok(Vec::new());
    }
    let sot_len = sot_tokens.len();
    let n_text = text_tokens.len();
    let (matrix, n_frames) = alignment_matrix_from_hidden(
        heads,
        hidden,
        encoder_hidden,
        wm,
        pfx,
        sot_len,
        n_text,
        enc_seq,
        d,
        head_dim,
    )?;
    word_timings_from_matrix(
        tokenizer,
        text_tokens,
        &matrix,
        n_text,
        n_frames,
        time_offset_sec,
        medfilt_width,
    )
}

fn decoder_forward_through(
    cfg: &WhisperConfig,
    wm: &WeightMap,
    pfx: &WhisperWeightPrefix,
    token_ids: &[u32],
    encoder_hidden: &[f32],
    enc_seq: usize,
    max_layer: usize,
) -> Result<Vec<f32>> {
    let d = cfg.d_model;
    let n_head = cfg.decoder_attention_heads;
    let head_dim = cfg.decoder_head_dim();
    let mlp = d * 4;
    let pos_w = get_w(wm, &pfx.dec_embed_positions())?;
    let pos_emb: Vec<f32> = token_ids
        .iter()
        .enumerate()
        .flat_map(|(i, _)| pos_w[i * d..(i + 1) * d].iter().copied())
        .collect();
    let mut x = embed_tokens(wm, pfx, token_ids, &pos_emb, d)?;

    let enc = &encoder_hidden[..enc_seq * d];
    let n_layers = cfg.decoder_layers;
    let align_layer = max_layer.min(n_layers.saturating_sub(1));

    for layer in 0..align_layer {
        let lp = |s: &str| pfx.dec_layer(layer, s);
        let ln_w = get_w(wm, &lp("self_attn_layer_norm.weight"))?;
        let ln_b = get_w(wm, &lp("self_attn_layer_norm.bias"))?;
        let ln = layer_norm(&x, ln_w, ln_b, d);
        let qw = get_w(wm, &lp("self_attn.q_proj.weight"))?;
        let kw = get_w(wm, &lp("self_attn.k_proj.weight"))?;
        let vw = get_w(wm, &lp("self_attn.v_proj.weight"))?;
        let vb = get_w(wm, &lp("self_attn.v_proj.bias"))?;
        let ow = get_w(wm, &lp("self_attn.out_proj.weight"))?;
        let ob = get_w(wm, &lp("self_attn.out_proj.bias"))?;
        let q = linear(&ln, qw, None, d, d);
        let k = linear(&ln, kw, None, d, d);
        let v = linear(&ln, vw, Some(vb), d, d);
        let seq = x.len() / d;
        let scale = (head_dim as f32).powf(-0.25);
        let mut sa = vec![0f32; x.len()];
        for t in 0..seq {
            for t2 in 0..=t {
                let mut scores = vec![0f32; n_head];
                for h in 0..n_head {
                    let mut sum = 0f32;
                    for hd in 0..head_dim {
                        sum += q[t * d + h * head_dim + hd]
                            * scale
                            * k[t2 * d + h * head_dim + hd]
                            * scale;
                    }
                    scores[h] = sum;
                }
                let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut denom = 0f32;
                for s in &mut scores {
                    *s = (*s - max).exp();
                    denom += *s;
                }
                for h in 0..n_head {
                    let w = scores[h] / denom.max(1e-9);
                    for hd in 0..head_dim {
                        sa[t * d + h * head_dim + hd] += w * v[t2 * d + h * head_dim + hd];
                    }
                }
            }
        }
        let sa_out = linear(&sa, ow, Some(ob), d, d);
        for i in 0..x.len() {
            x[i] += sa_out[i];
        }

        let kw = get_w(wm, &lp("encoder_attn.k_proj.weight"))?;
        let ck = linear(enc, kw, None, d, d);
        let vw = get_w(wm, &lp("encoder_attn.v_proj.weight"))?;
        let vb = wm.get(&lp("encoder_attn.v_proj.bias")).map(|(d, _)| d);
        let cv = linear(enc, vw, vb, d, d);

        let ln_w = get_w(wm, &lp("encoder_attn_layer_norm.weight"))?;
        let ln_b = get_w(wm, &lp("encoder_attn_layer_norm.bias"))?;
        let ln = layer_norm(&x, ln_w, ln_b, d);
        let qw = get_w(wm, &lp("encoder_attn.q_proj.weight"))?;
        let qb = wm.get(&lp("encoder_attn.q_proj.bias")).map(|(d, _)| d);
        let q = linear(&ln, qw, qb, d, d);
        let ow = get_w(wm, &lp("encoder_attn.out_proj.weight"))?;
        let ob = get_w(wm, &lp("encoder_attn.out_proj.bias"))?;
        let seq = x.len() / d;
        let scale = (head_dim as f32).powf(-0.25);
        let mut ca = vec![0f32; x.len()];
        for t in 0..seq {
            for h in 0..n_head {
                for f in 0..enc_seq {
                    let mut sum = 0f32;
                    for hd in 0..head_dim {
                        sum += q[t * d + h * head_dim + hd]
                            * scale
                            * ck[f * d + h * head_dim + hd]
                            * scale;
                    }
                    let w = sum;
                    for hd in 0..head_dim {
                        ca[t * d + h * head_dim + hd] += w * cv[f * d + h * head_dim + hd];
                    }
                }
            }
        }
        let ca_out = linear(&ca, ow, Some(ob), d, d);
        for i in 0..x.len() {
            x[i] += ca_out[i];
        }

        let ln_w = get_w(wm, &lp("final_layer_norm.weight"))?;
        let ln_b = get_w(wm, &lp("final_layer_norm.bias"))?;
        let ln = layer_norm(&x, ln_w, ln_b, d);
        let fc1w = get_w(wm, &lp("fc1.weight"))?;
        let fc1b = get_w(wm, &lp("fc1.bias"))?;
        let h1 = gelu(&linear(&ln, fc1w, Some(fc1b), d, mlp));
        let fc2w = get_w(wm, &lp("fc2.weight"))?;
        let fc2b = get_w(wm, &lp("fc2.bias"))?;
        let mlp_out = linear(&h1, fc2w, Some(fc2b), mlp, d);
        for i in 0..x.len() {
            x[i] = ln[i] + mlp_out[i];
        }
    }

    if align_layer < n_layers {
        let layer = align_layer;
        let lp = |s: &str| pfx.dec_layer(layer, s);
        let ln_w = get_w(wm, &lp("self_attn_layer_norm.weight"))?;
        let ln_b = get_w(wm, &lp("self_attn_layer_norm.bias"))?;
        let ln = layer_norm(&x, ln_w, ln_b, d);
        let qw = get_w(wm, &lp("self_attn.q_proj.weight"))?;
        let kw = get_w(wm, &lp("self_attn.k_proj.weight"))?;
        let vw = get_w(wm, &lp("self_attn.v_proj.weight"))?;
        let vb = get_w(wm, &lp("self_attn.v_proj.bias"))?;
        let ow = get_w(wm, &lp("self_attn.out_proj.weight"))?;
        let ob = get_w(wm, &lp("self_attn.out_proj.bias"))?;
        let q = linear(&ln, qw, None, d, d);
        let k = linear(&ln, kw, None, d, d);
        let v = linear(&ln, vw, Some(vb), d, d);
        let seq = x.len() / d;
        let scale = (head_dim as f32).powf(-0.25);
        let mut sa = vec![0f32; x.len()];
        for t in 0..seq {
            for t2 in 0..=t {
                let mut scores = vec![0f32; n_head];
                for h in 0..n_head {
                    let mut sum = 0f32;
                    for hd in 0..head_dim {
                        sum += q[t * d + h * head_dim + hd]
                            * scale
                            * k[t2 * d + h * head_dim + hd]
                            * scale;
                    }
                    scores[h] = sum;
                }
                let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut denom = 0f32;
                for s in &mut scores {
                    *s = (*s - max).exp();
                    denom += *s;
                }
                for h in 0..n_head {
                    let w = scores[h] / denom.max(1e-9);
                    for hd in 0..head_dim {
                        sa[t * d + h * head_dim + hd] += w * v[t2 * d + h * head_dim + hd];
                    }
                }
            }
        }
        let sa_out = linear(&sa, ow, Some(ob), d, d);
        for i in 0..x.len() {
            x[i] += sa_out[i];
        }
    }
    Ok(x)
}

/// CPU path: decoder forward + cross-attention QK + DTW ([`find_word_alignment_from_hidden`]).
pub fn find_word_alignment(
    tokenizer: &tokenizers::Tokenizer,
    heads: &AlignmentHeads,
    cfg: &WhisperConfig,
    wm: &WeightMap,
    pfx: &WhisperWeightPrefix,
    sot_tokens: &[u32],
    text_tokens: &[u32],
    eot: u32,
    encoder_hidden: &[f32],
    enc_seq: usize,
    time_offset_sec: f32,
    medfilt_width: usize,
) -> Result<Vec<WordTiming>> {
    if text_tokens.is_empty() {
        return Ok(Vec::new());
    }
    let d = cfg.d_model;
    let head_dim = cfg.decoder_head_dim();
    let max_layer = heads
        .pairs
        .iter()
        .map(|(l, _)| *l)
        .max()
        .unwrap_or(cfg.decoder_layers.saturating_sub(1));
    let mut all_ids: Vec<u32> = sot_tokens.to_vec();
    all_ids.extend_from_slice(text_tokens);
    all_ids.push(eot);
    let x = decoder_forward_through(cfg, wm, pfx, &all_ids, encoder_hidden, enc_seq, max_layer)?;
    find_word_alignment_from_hidden(
        tokenizer,
        heads,
        wm,
        pfx,
        sot_tokens,
        text_tokens,
        &x,
        encoder_hidden,
        enc_seq,
        d,
        head_dim,
        time_offset_sec,
        medfilt_width,
    )
}

pub fn interpolate_words_in_segment(text: &str, seg_start: f32, seg_end: f32) -> Vec<WordTiming> {
    let parts: Vec<&str> = text.split_whitespace().collect();
    if parts.is_empty() {
        return Vec::new();
    }
    let dur = (seg_end - seg_start).max(0.01);
    let step = dur / parts.len() as f32;
    parts
        .iter()
        .enumerate()
        .map(|(i, &w)| WordTiming {
            word: w.to_string(),
            start: seg_start + i as f32 * step,
            end: seg_start + (i + 1) as f32 * step,
            prob: None,
        })
        .collect()
}
