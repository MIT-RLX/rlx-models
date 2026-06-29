// RLX models — calibration quantization.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! AWQ — Activation-aware Weight Quantization (Lin et al. 2023).
//!
//! Salient weight channels (those multiplied by large activations) dominate the
//! output error. AWQ scales each input channel `c` by `s[c]` before quantizing
//! (`W·diag(s)`) and folds `1/s` into the preceding op, leaving the math
//! unchanged but shrinking the quant error on the channels that matter. The
//! per-channel scale `s = act_scale^α` (geometric-mean-normalized) is chosen by
//! a small grid search over `α ∈ [0, 1]` (α=0 reduces to round-to-nearest).

use crate::quant::{GroupQuant, dequantize, quantize_rtn};

/// Per-input-channel scale `s = act_scale^α`, normalized to geometric mean 1 so
/// the overall weight magnitude is preserved.
fn channel_scale(act_scale: &[f32], alpha: f32) -> Vec<f32> {
    let eps = 1e-8;
    let raw: Vec<f32> = act_scale.iter().map(|&a| (a + eps).powf(alpha)).collect();
    let logmean = raw.iter().map(|v| v.ln()).sum::<f32>() / raw.len().max(1) as f32;
    let gm = logmean.exp();
    raw.iter().map(|v| (v / gm).max(eps)).collect()
}

/// AWQ-quantize `w [out, in]` given per-input-channel activation magnitudes
/// `act_scale [in]`. Returns the quantized **scaled** weight and the chosen
/// per-channel scale `s [in]`; the effective dequantized weight is
/// `dequant(q) / s` ([`awq_effective_weight`]). The grid search minimizes the
/// activation-weighted reconstruction error.
pub fn awq_quantize(
    w: &[f32],
    out: usize,
    inn: usize,
    act_scale: &[f32],
    bits: u32,
    group_size: usize,
) -> (GroupQuant, Vec<f32>) {
    let steps = 21usize;
    let mut best: Option<(f32, GroupQuant, Vec<f32>)> = None;
    for i in 0..steps {
        let alpha = i as f32 / (steps - 1) as f32;
        let s = channel_scale(act_scale, alpha);
        let mut ws = vec![0f32; out * inn];
        for o in 0..out {
            for c in 0..inn {
                ws[o * inn + c] = w[o * inn + c] * s[c];
            }
        }
        let q = quantize_rtn(&ws, out, inn, bits, group_size);
        let dq = dequantize(&q);
        // Activation-weighted error of the effective weight (∝ output MSE).
        let mut err = 0.0f32;
        for o in 0..out {
            for c in 0..inn {
                let w_eff = dq[o * inn + c] / s[c];
                let d = w[o * inn + c] - w_eff;
                err += act_scale[c] * act_scale[c] * d * d;
            }
        }
        if best.as_ref().map_or(true, |(e, _, _)| err < *e) {
            best = Some((err, q, s));
        }
    }
    let (_, q, s) = best.expect("grid has >= 1 point");
    (q, s)
}

/// The effective dequantized weight after AWQ: `dequant(W·diag(s)) / s`.
pub fn awq_effective_weight(q: &GroupQuant, s: &[f32]) -> Vec<f32> {
    let dq = dequantize(q);
    let mut out = vec![0f32; q.out * q.inn];
    for o in 0..q.out {
        for c in 0..q.inn {
            out[o * q.inn + c] = dq[o * q.inn + c] / s[c];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::{dequantize, matmul_wt, mse, quantize_rtn};

    fn pseudo(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed.wrapping_mul(2654435761).wrapping_add(1);
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                ((s >> 8) as f32 / u32::MAX as f32 - 0.5) * 2.0
            })
            .collect()
    }

    #[test]
    fn awq_beats_rtn_on_salient_channels() {
        // Each quant group holds one *salient* channel (high activation but
        // small weight) plus larger low-activation channels that set the
        // group's scale. RTN gives the salient channel poor resolution; AWQ
        // scales it up within the group and cuts output error.
        let (out, inn, gs) = (8usize, 16usize, 4usize);
        let samples = 96usize;
        let mut w = pseudo(out * inn, 1);
        for o in 0..out {
            for c in 0..inn {
                if c % gs == 0 {
                    w[o * inn + c] *= 0.12; // salient channels have small weights
                }
            }
        }

        let base = pseudo(samples * inn, 9);
        let mut x = vec![0f32; samples * inn];
        for sm in 0..samples {
            for c in 0..inn {
                let chan = if c % gs == 0 { 6.0 } else { 0.3 };
                x[sm * inn + c] = base[sm * inn + c] * chan;
            }
        }
        let act_scale: Vec<f32> = (0..inn)
            .map(|c| (0..samples).map(|sm| x[sm * inn + c].abs()).sum::<f32>() / samples as f32)
            .collect();

        let target = matmul_wt(&x, &w, samples, inn, out);
        let bits = 4u32;

        let rtn = quantize_rtn(&w, out, inn, bits, gs);
        let rtn_err = mse(
            &target,
            &matmul_wt(&x, &dequantize(&rtn), samples, inn, out),
        );

        let (aq, s) = awq_quantize(&w, out, inn, &act_scale, bits, gs);
        let awq_err = mse(
            &target,
            &matmul_wt(&x, &awq_effective_weight(&aq, &s), samples, inn, out),
        );

        assert!(awq_err < rtn_err, "AWQ {awq_err} should beat RTN {rtn_err}");
    }
}
