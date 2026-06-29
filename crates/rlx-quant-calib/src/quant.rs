// RLX models — calibration quantization.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! Group-wise symmetric integer quantization — the shared primitive AWQ /
//! GPTQ / dynamic build on.
//!
//! A weight is laid out `[out, in]` (HF convention); quantization groups along
//! the `in` (contraction) axis with one scale per `(out_row, group)`.

/// A group-wise symmetric-quantized weight.
#[derive(Debug, Clone)]
pub struct GroupQuant {
    /// Quantized integers, `[out * in]` row-major.
    pub q: Vec<i32>,
    /// Per-`(row, group)` scales, `[out * num_groups]`.
    pub scales: Vec<f32>,
    pub out: usize,
    pub inn: usize,
    pub bits: u32,
    pub group_size: usize,
}

impl GroupQuant {
    pub fn num_groups(&self) -> usize {
        self.inn.div_ceil(self.group_size)
    }
}

/// Largest representable magnitude for symmetric `bits`-bit quant.
pub fn qmax(bits: u32) -> f32 {
    ((1i64 << (bits - 1)) - 1) as f32
}

/// Round-to-nearest group-wise quantization — the baseline every learned
/// method must beat. `group_size >= in` ⇒ one scale per row.
pub fn quantize_rtn(w: &[f32], out: usize, inn: usize, bits: u32, group_size: usize) -> GroupQuant {
    let gs = group_size.clamp(1, inn.max(1));
    let ng = inn.div_ceil(gs);
    let qm = qmax(bits);
    let mut q = vec![0i32; out * inn];
    let mut scales = vec![0f32; out * ng];
    for r in 0..out {
        for g in 0..ng {
            let c0 = g * gs;
            let c1 = (c0 + gs).min(inn);
            let amax = (c0..c1)
                .map(|c| w[r * inn + c].abs())
                .fold(0.0f32, f32::max);
            let s = if amax > 0.0 { amax / qm } else { 1.0 };
            scales[r * ng + g] = s;
            for c in c0..c1 {
                q[r * inn + c] = (w[r * inn + c] / s).round().clamp(-qm, qm) as i32;
            }
        }
    }
    GroupQuant {
        q,
        scales,
        out,
        inn,
        bits,
        group_size: gs,
    }
}

/// Dequantize back to `[out, in]` f32.
pub fn dequantize(qd: &GroupQuant) -> Vec<f32> {
    let ng = qd.num_groups();
    let mut w = vec![0f32; qd.out * qd.inn];
    for r in 0..qd.out {
        for c in 0..qd.inn {
            let g = c / qd.group_size;
            w[r * qd.inn + c] = qd.q[r * qd.inn + c] as f32 * qd.scales[r * ng + g];
        }
    }
    w
}

/// Mean squared error between two equal-length tensors.
pub fn mse(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f32>() / a.len().max(1) as f32
}

/// `X · Wᵀ` where `x` is `[samples, in]` and `w` is `[out, in]` → `[samples, out]`.
pub fn matmul_wt(x: &[f32], w: &[f32], samples: usize, inn: usize, out: usize) -> Vec<f32> {
    let mut y = vec![0f32; samples * out];
    for s in 0..samples {
        for o in 0..out {
            let mut acc = 0.0;
            for i in 0..inn {
                acc += x[s * inn + i] * w[o * inn + i];
            }
            y[s * out + o] = acc;
        }
    }
    y
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rtn_round_trips_within_resolution() {
        let w = vec![0.0, 0.5, -0.5, 1.0, -1.0, 0.25, -0.25, 0.75];
        let q = quantize_rtn(&w, 1, 8, 4, 8); // 1×8, one group, 4-bit
        let dq = dequantize(&q);
        // Quant error bounded by half a quantization step (amax/qmax / 2).
        let step = 1.0 / qmax(4);
        for (a, b) in w.iter().zip(&dq) {
            assert!((a - b).abs() <= step / 2.0 + 1e-6, "{a} vs {b}");
        }
    }

    #[test]
    fn more_bits_means_less_error() {
        let w: Vec<f32> = (0..64).map(|i| (i * 7 % 23) as f32 / 23.0 - 0.5).collect();
        let e4 = mse(&w, &dequantize(&quantize_rtn(&w, 8, 8, 4, 8)));
        let e8 = mse(&w, &dequantize(&quantize_rtn(&w, 8, 8, 8, 8)));
        assert!(e8 < e4, "8-bit ({e8}) should beat 4-bit ({e4})");
    }
}
