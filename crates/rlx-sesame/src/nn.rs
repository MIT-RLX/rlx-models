//! Eager CPU math primitives for CSM (Llama-style, NeoX RoPE).

use crate::config::RopeScaling;

/// Dense matvec: `y[o] = sum_i(W[o,i] * x[i])`. `W` is row-major `[d_out, d_in]`
/// (PyTorch `nn.Linear` weight layout).
pub fn matvec(w: &[f32], x: &[f32], d_in: usize, d_out: usize) -> Vec<f32> {
    debug_assert_eq!(w.len(), d_out * d_in);
    debug_assert_eq!(x.len(), d_in);
    let mut y = vec![0.0f32; d_out];
    for o in 0..d_out {
        let row = &w[o * d_in..(o + 1) * d_in];
        y[o] = row.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
    }
    y
}

/// `y = x @ W` with `W` row-major `[d_in, d_out]` (CSM `codebooks_head` slices).
pub fn matmul_in_out(w: &[f32], x: &[f32], d_in: usize, d_out: usize) -> Vec<f32> {
    debug_assert_eq!(w.len(), d_in * d_out);
    debug_assert_eq!(x.len(), d_in);
    let mut y = vec![0.0f32; d_out];
    for i in 0..d_in {
        let row = &w[i * d_out..(i + 1) * d_out];
        let xi = x[i];
        for o in 0..d_out {
            y[o] += xi * row[o];
        }
    }
    y
}

/// Llama RMSNorm: `y = x / rms(x) * w`.
pub fn rms_norm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let rms = (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32 + eps).sqrt();
    x.iter().zip(w).map(|(v, wi)| v / rms * wi).collect()
}

#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Llama-3 RoPE inverse frequencies (HF / candle formula).
pub fn llama3_inv_freq(
    rope_theta: f64,
    head_dim: usize,
    scaling: Option<&RopeScaling>,
) -> Vec<f64> {
    let base: Vec<f64> = (0..head_dim)
        .step_by(2)
        .map(|i| 1.0 / rope_theta.powf(i as f64 / head_dim as f64))
        .collect();
    let Some(s) = scaling else {
        return base;
    };
    if s.rope_type != "llama3" && !s.rope_type.is_empty() {
        return base;
    }
    let low_freq_wavelen = s.original_max_position_embeddings as f64 / s.low_freq_factor as f64;
    let high_freq_wavelen = s.original_max_position_embeddings as f64 / s.high_freq_factor as f64;
    base.into_iter()
        .map(|freq| {
            let wavelen = 2.0 * std::f64::consts::PI / freq;
            if wavelen < high_freq_wavelen {
                freq
            } else if wavelen > low_freq_wavelen {
                freq / s.factor as f64
            } else {
                let smooth = (s.original_max_position_embeddings as f64 / wavelen
                    - s.low_freq_factor as f64)
                    / (s.high_freq_factor as f64 - s.low_freq_factor as f64);
                (1.0 - smooth) * freq / s.factor as f64 + smooth * freq
            }
        })
        .collect()
}

/// Apply NeoX-style RoPE in-place to `x: [n_heads * head_dim]` at `pos`.
pub fn apply_rope(x: &mut [f32], pos: usize, head_dim: usize, inv_freq: &[f64]) {
    let half = head_dim / 2;
    debug_assert_eq!(inv_freq.len(), half);
    let n_heads = x.len() / head_dim;
    for h in 0..n_heads {
        let base = h * head_dim;
        for i in 0..half {
            let angle = (pos as f64 * inv_freq[i]) as f32;
            let (sin, cos) = angle.sin_cos();
            let x0 = x[base + i];
            let x1 = x[base + i + half];
            x[base + i] = x0 * cos - x1 * sin;
            x[base + i + half] = x0 * sin + x1 * cos;
        }
    }
}

/// Grouped-query attention for one query against a KV cache.
pub fn gqa_attend(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    n_kv: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let kv_groups = n_heads / n_kv_heads;
    let scale = (head_dim as f32).sqrt().recip();
    let mut out = vec![0.0f32; n_heads * head_dim];

    for h in 0..n_heads {
        let kv_h = h / kv_groups;
        let q_s = &q[h * head_dim..(h + 1) * head_dim];
        let mut scores = vec![0.0f32; n_kv];
        for t in 0..n_kv {
            let k_s = &k_cache
                [(t * n_kv_heads + kv_h) * head_dim..((t * n_kv_heads + kv_h) + 1) * head_dim];
            scores[t] = q_s.iter().zip(k_s).map(|(a, b)| a * b).sum::<f32>() * scale;
        }
        let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = scores
            .iter_mut()
            .map(|s| {
                *s = (*s - max_s).exp();
                *s
            })
            .sum();
        for s in &mut scores {
            *s /= sum.max(1e-9);
        }
        let out_s = &mut out[h * head_dim..(h + 1) * head_dim];
        for t in 0..n_kv {
            let v_s = &v_cache
                [(t * n_kv_heads + kv_h) * head_dim..((t * n_kv_heads + kv_h) + 1) * head_dim];
            for (ov, vv) in out_s.iter_mut().zip(v_s) {
                *ov += scores[t] * vv;
            }
        }
    }
    out
}

/// Top-k sampling (Sesame `sample_topk`).
pub fn sample_topk(logits: &[f32], topk: usize, temperature: f32, rng: &mut fastrand::Rng) -> u32 {
    let temp = temperature.max(1e-5);
    let mut indexed: Vec<(usize, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, v / temp))
        .collect();
    let k = topk.min(indexed.len()).max(1);
    indexed.select_nth_unstable_by(k - 1, |a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
    });
    indexed.truncate(k);
    let max_v = indexed
        .iter()
        .map(|(_, v)| *v)
        .fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = indexed.iter().map(|(_, v)| (v - max_v).exp()).collect();
    let sum: f32 = probs.iter().sum();
    for p in &mut probs {
        *p /= sum.max(1e-9);
    }
    // Multinomial via CDF.
    let r = rng.f32();
    let mut acc = 0.0f32;
    for (i, p) in probs.iter().enumerate() {
        acc += *p;
        if r <= acc {
            return indexed[i].0 as u32;
        }
    }
    indexed[k - 1].0 as u32
}

/// Argmax (greedy).
pub fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}
