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

//! Host-side LM head for decode (skips full `[vocab]` logits when possible).

use crate::config::Qwen35Config;
use crate::weights::{MatWeight, Qwen35Weights};
use anyhow::{Result, anyhow};
use rlx_core::weight_loader::GgufLoader;
use rlx_qwen3::sampling::{SampleOpts, sample_token_at};

/// How many vocab rows to score for sampling (partial LM head).
/// Rotate a single hidden row into the LM head's basis.
///
/// Ternary Bonsai 2 stores `output.weight` Hadamard-folded, so these
/// host paths must apply the same activation transform the graph
/// builder does. Uses the butterfly rather than the `[block, block]`
/// matmul — it is one row, and this runs per decoded token.
///
/// Returns `hidden` untouched for every other checkpoint.
fn lm_head_input<'a>(weights: &Qwen35Weights, hidden: &'a [f32]) -> std::borrow::Cow<'a, [f32]> {
    let Some(fold) = weights.output_fold.as_ref() else {
        return std::borrow::Cow::Borrowed(hidden);
    };
    let mut v = hidden.to_vec();
    let width = v.len();
    crate::prism_hadamard::apply_activation_transform(&mut v, width, fold);
    std::borrow::Cow::Owned(v)
}

pub fn sample_lm_cap(opts: &SampleOpts, n_vocab: usize) -> usize {
    if opts.greedy {
        return 1;
    }
    if opts.top_k > 0 {
        return opts.top_k.max(32).min(n_vocab);
    }
    if opts.top_p < 1.0 {
        return 512.min(n_vocab);
    }
    n_vocab
}

fn lm_topk(
    weights: &Qwen35Weights,
    cfg: &Qwen35Config,
    hidden: &[f32],
    loader: Option<&GgufLoader>,
    cap: usize,
) -> Result<Vec<(u32, f32)>> {
    let n_embd = cfg.hidden_size;
    let n_vocab = weights.lm_vocab_size(cfg);
    if hidden.len() != n_embd {
        return Err(anyhow!(
            "lm_head: hidden len {} != n_embd {n_embd}",
            hidden.len()
        ));
    }

    let hidden = lm_head_input(weights, hidden);
    let hidden = hidden.as_ref();

    match &weights.output {
        Some(MatWeight::F32(data)) => Ok(rlx_cpu::lm_head::f32_tied_lm_topk(
            hidden, data, n_embd, n_vocab, cap,
        )),
        Some(MatWeight::Packed { key, scheme, shape }) => {
            let expected_out = shape.first().copied().unwrap_or(n_vocab);
            let expected_in = shape.get(1).copied().unwrap_or(n_embd);
            let bytes = loader
                .and_then(|l| l.tensor_bytes_borrowed(key))
                .ok_or_else(|| anyhow!("packed lm_head: bytes missing for {key}"))?;
            Ok(rlx_cpu::lm_head::gguf_tied_lm_topk(
                hidden,
                bytes,
                expected_in,
                expected_out,
                *scheme,
                cap,
            ))
        }
        None => match &weights.token_embd_lm {
            Some(MatWeight::Packed { key, scheme, shape }) => {
                let expected_out = shape.first().copied().unwrap_or(n_vocab);
                let expected_in = shape.get(1).copied().unwrap_or(n_embd);
                let bytes = loader
                    .and_then(|l| l.tensor_bytes_borrowed(key))
                    .ok_or_else(|| anyhow!("packed tied lm_head: bytes missing for {key}"))?;
                Ok(rlx_cpu::lm_head::gguf_tied_lm_topk(
                    hidden,
                    bytes,
                    expected_in,
                    expected_out,
                    *scheme,
                    cap,
                ))
            }
            Some(MatWeight::F32(data)) => Ok(rlx_cpu::lm_head::f32_tied_lm_topk(
                hidden, data, n_embd, n_vocab, cap,
            )),
            None => Ok(rlx_cpu::lm_head::f32_tied_lm_topk(
                hidden,
                weights.token_embd(),
                n_embd,
                n_vocab,
                cap,
            )),
        },
    }
}

/// Argmax token + logit for one `[n_embd]` hidden vector.
pub fn greedy_lm_head_argmax(
    weights: &Qwen35Weights,
    cfg: &Qwen35Config,
    hidden: &[f32],
    loader: Option<&GgufLoader>,
) -> Result<(u32, f32)> {
    let n_embd = cfg.hidden_size;
    let n_vocab = weights.lm_vocab_size(cfg);
    if hidden.len() != n_embd {
        return Err(anyhow!(
            "lm_head: hidden len {} != n_embd {n_embd}",
            hidden.len()
        ));
    }

    let hidden = lm_head_input(weights, hidden);
    let hidden = hidden.as_ref();

    match &weights.output {
        Some(MatWeight::F32(data)) => {
            let (idx, val) = rlx_cpu::lm_head::f32_tied_lm_argmax(hidden, data, n_embd, n_vocab);
            Ok((idx, val))
        }
        Some(MatWeight::Packed { key, scheme, shape }) => {
            let expected_out = shape.first().copied().unwrap_or(n_vocab);
            let expected_in = shape.get(1).copied().unwrap_or(n_embd);
            let bytes = loader
                .and_then(|l| l.tensor_bytes_borrowed(key))
                .ok_or_else(|| anyhow!("packed lm_head: bytes missing for {key}"))?;
            let (idx, val) = rlx_cpu::lm_head::gguf_tied_lm_argmax_parallel(
                hidden,
                bytes,
                expected_in,
                expected_out,
                *scheme,
            );
            Ok((idx, val))
        }
        None => match &weights.token_embd_lm {
            Some(MatWeight::Packed { key, scheme, shape }) => {
                let expected_out = shape.first().copied().unwrap_or(n_vocab);
                let expected_in = shape.get(1).copied().unwrap_or(n_embd);
                let bytes = loader
                    .and_then(|l| l.tensor_bytes_borrowed(key))
                    .ok_or_else(|| anyhow!("packed tied lm_head: bytes missing for {key}"))?;
                let (idx, val) = rlx_cpu::lm_head::gguf_tied_lm_argmax_parallel(
                    hidden,
                    bytes,
                    expected_in,
                    expected_out,
                    *scheme,
                );
                Ok((idx, val))
            }
            Some(MatWeight::F32(data)) => {
                let (idx, val) =
                    rlx_cpu::lm_head::f32_tied_lm_argmax(hidden, data, n_embd, n_vocab);
                Ok((idx, val))
            }
            None => {
                let (idx, val) = rlx_cpu::lm_head::f32_tied_lm_argmax(
                    hidden,
                    weights.token_embd(),
                    n_embd,
                    n_vocab,
                );
                Ok((idx, val))
            }
        },
    }
}

/// Sample one token from hidden via partial LM head (top-k logits only).
/// `step` is the index of the token being sampled ([`sample_token_at`]).
pub fn sample_lm_head_from_hidden(
    weights: &Qwen35Weights,
    cfg: &Qwen35Config,
    hidden: &[f32],
    loader: Option<&GgufLoader>,
    opts: SampleOpts,
    step: u64,
) -> Result<u32> {
    let n_vocab = weights.lm_vocab_size(cfg);
    let cap = sample_lm_cap(&opts, n_vocab);
    if cap >= n_vocab {
        let (idx, val) = greedy_lm_head_argmax(weights, cfg, hidden, loader)?;
        if opts.greedy {
            return Ok(idx);
        }
        let mut logits = vec![f32::NEG_INFINITY; n_vocab];
        logits[idx as usize] = val;
        return Ok(sample_token_at(&logits, opts, step) as u32);
    }
    let top = lm_topk(weights, cfg, hidden, loader, cap)?;
    let mut logits = vec![f32::NEG_INFINITY; n_vocab];
    for (idx, val) in top {
        if (idx as usize) < n_vocab {
            logits[idx as usize] = val;
        }
    }
    Ok(sample_token_at(&logits, opts, step) as u32)
}

/// Full `[vocab]` logits row for one hidden vector (host matmul).
pub fn lm_head_logits_row(
    weights: &Qwen35Weights,
    cfg: &Qwen35Config,
    hidden: &[f32],
    loader: Option<&GgufLoader>,
) -> Result<Vec<f32>> {
    let n_embd = cfg.hidden_size;
    let n_vocab = weights.lm_vocab_size(cfg);
    if hidden.len() != n_embd {
        return Err(anyhow!(
            "lm_head: hidden len {} != n_embd {n_embd}",
            hidden.len()
        ));
    }
    let mut logits = vec![0f32; n_vocab];

    let hidden = lm_head_input(weights, hidden);
    let hidden = hidden.as_ref();

    match &weights.output {
        Some(MatWeight::F32(data)) => {
            matmul_row(hidden, data, n_embd, n_vocab, &mut logits);
        }
        Some(MatWeight::Packed { key, scheme, shape }) => {
            let expected_out = shape.first().copied().unwrap_or(n_vocab);
            let expected_in = shape.get(1).copied().unwrap_or(n_embd);
            let bytes = loader
                .and_then(|l| l.tensor_bytes_borrowed(key))
                .ok_or_else(|| anyhow!("packed lm_head: bytes missing for {key}"))?;
            rlx_cpu::gguf_matmul::gguf_matmul_bt(
                hidden,
                bytes,
                &mut logits,
                1,
                expected_in,
                expected_out,
                *scheme,
            );
        }
        None => match &weights.token_embd_lm {
            Some(MatWeight::Packed { key, scheme, shape }) => {
                let expected_out = shape.first().copied().unwrap_or(n_vocab);
                let expected_in = shape.get(1).copied().unwrap_or(n_embd);
                let bytes = loader
                    .and_then(|l| l.tensor_bytes_borrowed(key))
                    .ok_or_else(|| anyhow!("packed tied lm_head: bytes missing for {key}"))?;
                rlx_cpu::gguf_matmul::gguf_matmul_bt(
                    hidden,
                    bytes,
                    &mut logits,
                    1,
                    expected_in,
                    expected_out,
                    *scheme,
                );
            }
            Some(MatWeight::F32(data)) => {
                matmul_row(hidden, data, n_embd, n_vocab, &mut logits);
            }
            None => matmul_row(hidden, weights.token_embd(), n_embd, n_vocab, &mut logits),
        },
    }
    Ok(logits)
}

fn matmul_row(x: &[f32], w: &[f32], k: usize, n: usize, out: &mut [f32]) {
    out.fill(0.0);
    for j in 0..n {
        let row = &w[j * k..(j + 1) * k];
        let mut dot = 0f32;
        for p in 0..k {
            dot += x[p] * row[p];
        }
        out[j] = dot;
    }
}

/// Expand argmax to a sparse logits row (only `best_idx` set) for API compat.
#[allow(dead_code)]
pub fn logits_from_argmax(n_vocab: usize, best_idx: u32, best_val: f32) -> Vec<f32> {
    let mut logits = vec![f32::NEG_INFINITY; n_vocab];
    if (best_idx as usize) < n_vocab {
        logits[best_idx as usize] = best_val;
    }
    logits
}

/// Build sparse logits from partial top-k scores.
#[allow(dead_code)]
pub fn logits_from_topk(n_vocab: usize, top: &[(u32, f32)]) -> Vec<f32> {
    let mut logits = vec![f32::NEG_INFINITY; n_vocab];
    for &(idx, val) in top {
        if (idx as usize) < n_vocab {
            logits[idx as usize] = val;
        }
    }
    logits
}
