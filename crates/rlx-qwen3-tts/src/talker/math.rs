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

//! Host-side helpers (codec head, sampling).

use anyhow::{Result, ensure};
use ndarray::linalg::general_mat_vec_mul;
use ndarray::{ArrayView1, ArrayView2, ArrayViewMut1};

/// Bucketed decode graphs may return padded `hidden_states`; take the last token row.
pub fn last_decode_hidden(buf: &[f32], hidden: usize) -> Result<Vec<f32>> {
    let mut out = vec![0f32; hidden];
    last_decode_hidden_into(buf, hidden, &mut out)?;
    Ok(out)
}

pub fn last_decode_hidden_into(buf: &[f32], hidden: usize, out: &mut [f32]) -> Result<()> {
    ensure!(
        out.len() == hidden,
        "last_decode_hidden_into out len {} != {hidden}",
        out.len()
    );
    ensure!(
        !buf.is_empty() && buf.len().is_multiple_of(hidden),
        "decode hidden len {} not multiple of {hidden}",
        buf.len()
    );
    let off = buf.len() - hidden;
    out.copy_from_slice(&buf[off..]);
    Ok(())
}

/// Bucketed decode `hidden_states` for embeds graphs are extent-independent `[batch, 1, hidden]`;
/// when the runtime returns a padded buffer, the active row is always index 0 (unlike K/V,
/// where the new token row sits at bucket `upper`).
pub fn bucket_decode_hidden_into(buf: &[f32], hidden: usize, out: &mut [f32]) -> Result<()> {
    ensure!(
        out.len() == hidden,
        "bucket_decode_hidden_into out len {} != {hidden}",
        out.len()
    );
    ensure!(
        buf.len() >= hidden && buf.len().is_multiple_of(hidden),
        "decode hidden len {} invalid for hidden={hidden}",
        buf.len()
    );
    out.copy_from_slice(&buf[..hidden]);
    Ok(())
}

pub fn linear_logits(hidden: ArrayView1<f32>, head: ArrayView2<f32>) -> Result<Vec<f32>> {
    let (vocab, _) = head.dim();
    let mut logits = vec![0f32; vocab];
    linear_logits_into(hidden, head, &mut logits)?;
    Ok(logits)
}

/// `[out0, out1] = weight @ [x0, x1]` with `weight` `[out_dim, in_dim]` (one BLAS gemm).
#[inline]
pub fn matmul2_cols_into(
    weight: ArrayView2<f32>,
    x0: &[f32],
    x1: &[f32],
    out0: &mut [f32],
    out1: &mut [f32],
    b_stack: &mut [f32],
    tmp: &mut [f32],
) -> Result<()> {
    let (out_dim, in_dim) = weight.dim();
    ensure!(x0.len() == in_dim && x1.len() == in_dim);
    ensure!(out0.len() == out_dim && out1.len() == out_dim);
    ensure!(b_stack.len() >= in_dim * 2);
    ensure!(tmp.len() >= out_dim * 2);
    for i in 0..in_dim {
        b_stack[i * 2] = x0[i];
        b_stack[i * 2 + 1] = x1[i];
    }
    if let Some(w) = weight.as_slice() {
        rlx_cpu::blas::sgemm(
            w,
            &b_stack[..in_dim * 2],
            &mut tmp[..out_dim * 2],
            out_dim,
            in_dim,
            2,
        );
        for o in 0..out_dim {
            out0[o] = tmp[o * 2];
            out1[o] = tmp[o * 2 + 1];
        }
        return Ok(());
    }
    matvec_into(weight, x0, out0)?;
    matvec_into(weight, x1, out1)?;
    Ok(())
}

/// `out = weight @ x` with `weight` shaped `[out_dim, in_dim]` (no alloc).
#[inline]
pub fn matvec_into(weight: ArrayView2<f32>, x: &[f32], out: &mut [f32]) -> Result<()> {
    let (out_dim, in_dim) = weight.dim();
    ensure!(
        x.len() == in_dim,
        "matvec x len {} != in_dim {in_dim}",
        x.len()
    );
    ensure!(
        out.len() == out_dim,
        "matvec out len {} != out_dim {out_dim}",
        out.len()
    );
    if let Some(w) = weight.as_slice() {
        rlx_cpu::blas::sgemm(w, x, out, out_dim, in_dim, 1);
        return Ok(());
    }
    let x_view = ArrayView1::from(x);
    let mut y_view = ArrayViewMut1::from(out);
    general_mat_vec_mul(1.0, &weight, &x_view, 0.0, &mut y_view);
    Ok(())
}

pub fn linear_logits_into(
    hidden: ArrayView1<f32>,
    head: ArrayView2<f32>,
    logits: &mut [f32],
) -> Result<()> {
    let h = hidden.len();
    let (vocab, h2) = head.dim();
    ensure!(h == h2, "hidden {h} != head cols {h2}");
    ensure!(
        logits.len() == vocab,
        "logits len {} != vocab {vocab}",
        logits.len()
    );
    matvec_into(head, hidden.as_slice().unwrap(), logits)
}

/// Flat row-major `[vocab, hidden]` lm_head; no `ArrayView2::from_shape` per step.
#[inline]
pub fn linear_logits_flat_into(
    hidden: &[f32],
    head: &[f32],
    vocab: usize,
    hidden_dim: usize,
    logits: &mut [f32],
) -> Result<()> {
    ensure!(hidden.len() == hidden_dim);
    ensure!(head.len() == vocab * hidden_dim);
    ensure!(logits.len() == vocab);
    linear_logits_flat_unchecked(hidden, head, vocab, hidden_dim, logits);
    Ok(())
}

/// Hot-path lm_head matvec (caller guarantees shapes).
#[inline(always)]
pub fn linear_logits_flat_unchecked(
    hidden: &[f32],
    head: &[f32],
    vocab: usize,
    hidden_dim: usize,
    logits: &mut [f32],
) {
    debug_assert_eq!(hidden.len(), hidden_dim);
    debug_assert_eq!(head.len(), vocab * hidden_dim);
    debug_assert!(logits.len() >= vocab);
    rlx_cpu::blas::sgemm(head, hidden, logits, vocab, hidden_dim, 1);
}

/// Scalar GQA for short sequences (CP decode); avoids BLAS setup on tiny `n_keys`.
#[inline(always)]
fn gqa_attention1_small(
    q: &[f32],
    k_flat: &[f32],
    v_flat: &[f32],
    n_keys: usize,
    kv_dim: usize,
    out: &mut [f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    scale: f32,
    repeats: usize,
) {
    debug_assert!(n_keys <= 16);
    debug_assert_eq!(n_heads, n_kv_heads * repeats);
    let mut scores = [0f32; 16];
    for kv_h in 0..n_kv_heads {
        let h_base = kv_h * repeats;
        for b in 0..repeats {
            let hi = h_base + b;
            let q_off = hi * head_dim;
            let o_off = hi * head_dim;
            for ki in 0..n_keys {
                let kv_off = ki * kv_dim + kv_h * head_dim;
                let mut dot = 0f32;
                for d in 0..head_dim {
                    dot += q[q_off + d] * k_flat[kv_off + d];
                }
                scores[ki] = dot * scale;
            }
            let mut max_w = f32::NEG_INFINITY;
            for ki in 0..n_keys {
                max_w = max_w.max(scores[ki]);
            }
            let mut sum = 0f32;
            for ki in 0..n_keys {
                let e = (scores[ki] - max_w).exp();
                scores[ki] = e;
                sum += e;
            }
            let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
            for d in 0..head_dim {
                let mut acc = 0f32;
                for ki in 0..n_keys {
                    let kv_off = ki * kv_dim + kv_h * head_dim;
                    acc += scores[ki] * inv * v_flat[kv_off + d];
                }
                out[o_off + d] = acc;
            }
        }
    }
}

/// CP decode attention (`t_k ≤ 16`); always uses the register softmax micro-kernel.
#[inline(always)]
pub fn gqa_attention1_cp(
    q: &[f32],
    k_flat: &[f32],
    v_flat: &[f32],
    t_k: usize,
    kv_dim: usize,
    out: &mut [f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
) {
    if t_k == 1 {
        let repeats = n_heads / n_kv_heads;
        for hi in 0..n_heads {
            let kv_h = hi / repeats;
            let o_off = hi * head_dim;
            let v_off = kv_h * head_dim;
            out[o_off..o_off + head_dim].copy_from_slice(&v_flat[v_off..v_off + head_dim]);
        }
        return;
    }
    let repeats = n_heads / n_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let n_keys = t_k;
    debug_assert!(n_keys <= 16);
    gqa_attention1_small(
        q, k_flat, v_flat, n_keys, kv_dim, out, n_heads, n_kv_heads, head_dim, scale, repeats,
    );
}

/// GQA single-query attention (`t_k` cached keys); BLAS for `q@Kᵀ` and weighted `V`.
#[inline]
pub fn gqa_attention1_into(
    q: &[f32],
    k_flat: &[f32],
    v_flat: &[f32],
    t_k: usize,
    kv_dim: usize,
    out: &mut [f32],
    weights: &mut [f32],
    kv_head_scratch: &mut [f32],
    max_attn: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
) {
    if t_k == 1 {
        let repeats = n_heads / n_kv_heads;
        for hi in 0..n_heads {
            let kv_h = hi / repeats;
            let o_off = hi * head_dim;
            let v_off = kv_h * head_dim;
            out[o_off..o_off + head_dim].copy_from_slice(&v_flat[v_off..v_off + head_dim]);
        }
        return;
    }
    let repeats = n_heads / n_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let kq = t_k.saturating_sub(1);
    let n_keys = kq + 1;
    if n_keys <= 16 && repeats >= 2 {
        gqa_attention1_small(
            q, k_flat, v_flat, n_keys, kv_dim, out, n_heads, n_kv_heads, head_dim, scale, repeats,
        );
        return;
    }
    let (k_gather, rest) = kv_head_scratch.split_at_mut(n_keys * head_dim);
    let (v_gather, batch_scores) = rest.split_at_mut(n_keys * head_dim);
    debug_assert!(batch_scores.len() >= repeats * n_keys);

    if repeats > 1 {
        for kv_h in 0..n_kv_heads {
            for ki in 0..n_keys {
                let kv_off = ki * kv_dim + kv_h * head_dim;
                let row = ki * head_dim;
                k_gather[row..row + head_dim].copy_from_slice(&k_flat[kv_off..kv_off + head_dim]);
                v_gather[row..row + head_dim].copy_from_slice(&v_flat[kv_off..kv_off + head_dim]);
            }
            let h_base = kv_h * repeats;
            let q_batch = &q[h_base * head_dim..(h_base + repeats) * head_dim];
            let scores = &mut batch_scores[..repeats * n_keys];
            rlx_cpu::blas::sgemm_bt(q_batch, k_gather, scores, repeats, head_dim, n_keys, scale);
            for b in 0..repeats {
                let row = &mut scores[b * n_keys..(b + 1) * n_keys];
                let mut max_w = f32::NEG_INFINITY;
                for w in row.iter_mut() {
                    max_w = max_w.max(*w);
                }
                let mut sum = 0f32;
                for w in row.iter_mut() {
                    *w = (*w - max_w).exp();
                    sum += *w;
                }
                let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
                for w in row.iter_mut() {
                    *w *= inv;
                }
            }
            let o_batch = &mut out[h_base * head_dim..(h_base + repeats) * head_dim];
            rlx_cpu::blas::sgemm(scores, v_gather, o_batch, repeats, n_keys, head_dim);
        }
        return;
    }

    for hi in 0..n_heads {
        let kv_h = hi;
        let q_off = hi * head_dim;
        let o_off = hi * head_dim;
        let head_w = &mut weights[hi * max_attn..hi * max_attn + t_k];
        for ki in 0..n_keys {
            let kv_off = ki * kv_dim + kv_h * head_dim;
            let row = ki * head_dim;
            k_gather[row..row + head_dim].copy_from_slice(&k_flat[kv_off..kv_off + head_dim]);
            v_gather[row..row + head_dim].copy_from_slice(&v_flat[kv_off..kv_off + head_dim]);
        }
        let q_head = &q[q_off..q_off + head_dim];
        let scores = &mut head_w[..n_keys];
        rlx_cpu::blas::sgemm_bt(q_head, k_gather, scores, 1, head_dim, n_keys, scale);
        let mut max_w = f32::NEG_INFINITY;
        for w in scores.iter_mut() {
            max_w = max_w.max(*w);
        }
        let mut sum = 0f32;
        for w in scores.iter_mut() {
            *w = (*w - max_w).exp();
            sum += *w;
        }
        let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
        for w in scores.iter_mut() {
            *w *= inv;
        }
        rlx_cpu::blas::sgemm(
            scores,
            v_gather,
            &mut out[o_off..o_off + head_dim],
            1,
            n_keys,
            head_dim,
        );
    }
}

pub fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

pub fn sample_greedy(logits: &[f32]) -> u32 {
    argmax(logits)
}

/// Greedy argmax over the first `vocab` logits (CP lm_heads use smaller tables).
#[inline(always)]
pub fn sample_greedy_vocab(logits: &[f32], vocab: usize) -> u32 {
    let n = vocab.min(logits.len());
    let mut best_i = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    let mut i = 0usize;
    while i + 4 <= n {
        let v0 = logits[i];
        let v1 = logits[i + 1];
        let v2 = logits[i + 2];
        let v3 = logits[i + 3];
        if v0 > best_v {
            best_v = v0;
            best_i = i;
        }
        if v1 > best_v {
            best_v = v1;
            best_i = i + 1;
        }
        if v2 > best_v {
            best_v = v2;
            best_i = i + 2;
        }
        if v3 > best_v {
            best_v = v3;
            best_i = i + 3;
        }
        i += 4;
    }
    while i < n {
        let v = logits[i];
        if v > best_v {
            best_v = v;
            best_i = i;
        }
        i += 1;
    }
    best_i as u32
}

/// `[out0, out1] = weight @ [x0, x1]`; unchecked (hot CP 2-token prefill).
#[inline(always)]
pub fn matmul2_cols_blas(
    w: &[f32],
    x0: &[f32],
    x1: &[f32],
    out0: &mut [f32],
    out1: &mut [f32],
    b_stack: &mut [f32],
    tmp: &mut [f32],
    out_dim: usize,
    in_dim: usize,
) {
    debug_assert_eq!(x0.len(), in_dim);
    debug_assert_eq!(x1.len(), in_dim);
    debug_assert_eq!(out0.len(), out_dim);
    debug_assert_eq!(out1.len(), out_dim);
    debug_assert_eq!(w.len(), out_dim * in_dim);
    for i in 0..in_dim {
        b_stack[i * 2] = x0[i];
        b_stack[i * 2 + 1] = x1[i];
    }
    rlx_cpu::blas::sgemm(
        w,
        &b_stack[..in_dim * 2],
        &mut tmp[..out_dim * 2],
        out_dim,
        in_dim,
        2,
    );
    for o in 0..out_dim {
        out0[o] = tmp[o * 2];
        out1[o] = tmp[o * 2 + 1];
    }
}

/// `out = weight @ x` with row-major `weight` `[out_dim, in_dim]`; no checks (hot CP path).
#[inline(always)]
pub fn matvec_blas(w: &[f32], x: &[f32], out: &mut [f32], out_dim: usize, in_dim: usize) {
    debug_assert_eq!(x.len(), in_dim);
    debug_assert_eq!(out.len(), out_dim);
    debug_assert_eq!(w.len(), out_dim * in_dim);
    rlx_cpu::blas::sgemm(w, x, out, out_dim, in_dim, 1);
}

/// `out = weight @ x + out` (fused residual GEMV for transformer blocks).
#[inline(always)]
pub fn matvec_accumulate_blas(
    w: &[f32],
    x: &[f32],
    out: &mut [f32],
    out_dim: usize,
    in_dim: usize,
) {
    debug_assert_eq!(x.len(), in_dim);
    debug_assert_eq!(out.len(), out_dim);
    debug_assert_eq!(w.len(), out_dim * in_dim);
    rlx_cpu::blas::sgemm_accumulate(w, x, out, out_dim, in_dim, 1);
}

/// Greedy argmax with HF talker `suppress_tokens` (reserved codec band).
/// Greedy argmax with HF code-predictor reserved codec band (no eos carve-out).
pub fn sample_greedy_codec(logits: &[f32], vocab_size: usize) -> u32 {
    let reserved_start = vocab_size.saturating_sub(1024);
    let mut best_i = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if i >= reserved_start {
            continue;
        }
        if v > best_v {
            best_v = v;
            best_i = i;
        }
    }
    best_i as u32
}

/// HF `RepetitionPenaltyLogitsProcessor` (penalty > 1 down-weights seen tokens).
pub fn apply_repetition_penalty(logits: &mut [f32], past_tokens: &[u32], penalty: f32) {
    if (penalty - 1.0).abs() < 1e-6 {
        return;
    }
    for &tok in past_tokens {
        let i = tok as usize;
        if i >= logits.len() {
            continue;
        }
        if logits[i] > 0.0 {
            logits[i] /= penalty;
        } else {
            logits[i] *= penalty;
        }
    }
}

pub fn sample_greedy_talker_codec(logits: &[f32], vocab_size: usize, codec_eos: u32) -> u32 {
    // RLX_QWEN3_TTS_SAMPLE=1: temperature + top_k sampling (HF defaults
    // temperature=0.9, top_k=50 for voice clone).
    // RLX_QWEN3_TTS_SEED: seed for reproducible sampling.
    if std::env::var("RLX_QWEN3_TTS_SAMPLE").ok().as_deref() == Some("1") {
        let temp: f32 = std::env::var("RLX_QWEN3_TTS_TEMP")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.9);
        let top_k: usize = std::env::var("RLX_QWEN3_TTS_TOP_K")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(50);
        return sample_topk_talker_codec(logits, vocab_size, codec_eos, temp, top_k);
    }
    let reserved_start = vocab_size.saturating_sub(1024);
    let mut best_i = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if i >= reserved_start && i != codec_eos as usize {
            continue;
        }
        if v > best_v {
            best_v = v;
            best_i = i;
        }
    }
    best_i as u32
}

fn sample_topk_talker_codec(
    logits: &[f32],
    vocab_size: usize,
    codec_eos: u32,
    temperature: f32,
    top_k: usize,
) -> u32 {
    use rand::Rng;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    let reserved_start = vocab_size.saturating_sub(1024);
    // Collect (idx, scaled_logit) for valid tokens.
    let mut pool: Vec<(usize, f32)> = Vec::with_capacity(reserved_start + 1);
    for (i, &v) in logits.iter().enumerate() {
        if i >= reserved_start && i != codec_eos as usize {
            continue;
        }
        pool.push((i, v / temperature.max(1e-6)));
    }
    let k = top_k.max(1).min(pool.len());
    // Partial sort: keep top_k by logit.
    pool.select_nth_unstable_by(k - 1, |a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
    });
    pool.truncate(k);
    // Softmax over top_k.
    let max_l = pool
        .iter()
        .map(|(_, v)| *v)
        .fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0f32;
    for (_, v) in pool.iter_mut() {
        *v = (*v - max_l).exp();
        sum += *v;
    }
    if sum <= 0.0 {
        return pool[0].0 as u32;
    }
    thread_local! {
        static RNG: std::cell::RefCell<Option<StdRng>> = const { std::cell::RefCell::new(None) };
    }
    let r: f32 = RNG.with(|cell| {
        let mut slot = cell.borrow_mut();
        if slot.is_none() {
            let rng = match std::env::var("RLX_QWEN3_TTS_SEED")
                .ok()
                .and_then(|s| s.parse::<u64>().ok())
            {
                Some(s) => StdRng::seed_from_u64(s),
                None => StdRng::from_entropy(),
            };
            *slot = Some(rng);
        }
        slot.as_mut().unwrap().r#gen::<f32>() * sum
    });
    let mut acc = 0f32;
    for (idx, w) in &pool {
        acc += *w;
        if r <= acc {
            return *idx as u32;
        }
    }
    pool.last().unwrap().0 as u32
}
