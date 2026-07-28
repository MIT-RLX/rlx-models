// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Exact ternary `{−1,0,+1}` weights for bake TQ2 / fused add-sub kernels.

use alloc::vec;
use alloc::vec::Vec;

/// Which WakeCnn weight tensors to ternarize (biases stay f32).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TernaryOpts {
    pub conv: bool,
    pub fc: bool,
    /// Keep the top `keep_frac` of |w| as ±1; rest → 0. Default ≈ ⅓ (BitNet-ish).
    pub keep_frac: f32,
}

impl Default for TernaryOpts {
    fn default() -> Self {
        Self {
            // FC MatMuls are what rlx-bake packs as TQ2_0 today.
            conv: false,
            fc: true,
            keep_frac: 1.0 / 3.0,
        }
    }
}

impl TernaryOpts {
    pub fn fc_only() -> Self {
        Self::default()
    }

    pub fn all_weights() -> Self {
        Self {
            conv: true,
            fc: true,
            keep_frac: 1.0 / 3.0,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct TernaryStats {
    pub tensors: usize,
    pub elems: usize,
    pub nonzero: usize,
    pub bytes_f32: usize,
    pub bytes_packed: usize,
}

impl TernaryStats {
    pub fn compression_ratio(&self) -> f32 {
        if self.bytes_packed == 0 {
            1.0
        } else {
            self.bytes_f32 as f32 / self.bytes_packed as f32
        }
    }
}

/// Exact ternary values `{−1, 0, +1}` (eligible for rlx-bake TQ2_0).
pub fn is_ternary_f32(v: &[f32]) -> bool {
    v.iter().all(|&x| x == -1.0 || x == 0.0 || x == 1.0)
}

/// Magnitude threshold: top `keep_frac` of |w| → ±1 by sign, rest → 0.
pub fn ternarize(w: &[f32], keep_frac: f32) -> Vec<f32> {
    if w.is_empty() {
        return Vec::new();
    }
    let keep = keep_frac.clamp(0.01, 1.0);
    let mut abs: Vec<f32> = w.iter().map(|v| v.abs()).collect();
    abs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
    let n_keep = (((abs.len() as f32) * keep).round() as usize).clamp(1, abs.len());
    let thr = if n_keep >= abs.len() {
        0.0
    } else {
        abs[abs.len() - n_keep]
    };
    w.iter()
        .map(|&v| {
            if v.abs() < thr {
                0.0
            } else if v > 0.0 {
                1.0
            } else if v < 0.0 {
                -1.0
            } else {
                0.0
            }
        })
        .collect()
}

pub fn ternarize_inplace(w: &mut [f32], keep_frac: f32) {
    let t = ternarize(w, keep_frac);
    w.copy_from_slice(&t);
}

/// Pack trits as 2 bits each: `00=0`, `01=+1`, `10=−1` (little-endian within bytes).
pub fn pack_trits(w: &[f32]) -> Vec<u8> {
    let mut out = vec![0u8; w.len().div_ceil(4)];
    for (i, &v) in w.iter().enumerate() {
        let code: u8 = if v > 0.5 {
            0b01
        } else if v < -0.5 {
            0b10
        } else {
            0b00
        };
        let byte = i / 4;
        let shift = (i % 4) * 2;
        out[byte] |= code << shift;
    }
    out
}

pub fn unpack_trits(packed: &[u8], n: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let byte = packed.get(i / 4).copied().unwrap_or(0);
        let code = (byte >> ((i % 4) * 2)) & 0b11;
        out.push(match code {
            0b01 => 1.0,
            0b10 => -1.0,
            _ => 0.0,
        });
    }
    out
}

/// `y = A @ x` when A is exact ternary (skip zeros; ±1 → add/sub).
pub fn gemv_ternary(m: usize, n: usize, a: &[f32], x: &[f32], y: &mut [f32]) {
    debug_assert!(a.len() >= m * n);
    debug_assert!(x.len() >= n);
    debug_assert!(y.len() >= m);
    for i in 0..m {
        let mut s = 0.0f32;
        let row = i * n;
        for j in 0..n {
            let w = a[row + j];
            if w == 0.0 {
                continue;
            }
            if w > 0.0 {
                s += x[j];
            } else {
                s -= x[j];
            }
        }
        y[i] = s;
    }
}

pub fn gemv_bias_ternary(m: usize, n: usize, a: &[f32], x: &[f32], bias: &[f32], y: &mut [f32]) {
    gemv_ternary(m, n, a, x, y);
    for i in 0..m {
        y[i] += bias[i];
    }
}

/// Conv1d when weights are exact ternary (add/sub/skip, no dense mul).
pub fn conv1d_ternary(
    x: &[f32],
    in_ch: usize,
    t_in: usize,
    w: &[f32],
    out_ch: usize,
    k: usize,
    stride: usize,
    pad: usize,
    bias: Option<&[f32]>,
    out: &mut [f32],
) -> usize {
    let t_out = if t_in + 2 * pad >= k {
        (t_in + 2 * pad - k) / stride + 1
    } else {
        0
    };
    out.fill(0.0);
    for oc in 0..out_ch {
        for ot in 0..t_out {
            let mut sum = bias.map(|b| b[oc]).unwrap_or(0.0);
            for ic in 0..in_ch {
                for ki in 0..k {
                    let ti = ot * stride + ki;
                    let ti = ti as isize - pad as isize;
                    if ti < 0 || ti >= t_in as isize {
                        continue;
                    }
                    let x_idx = ic * t_in + ti as usize;
                    let w_idx = oc * (in_ch * k) + ic * k + ki;
                    let ww = w[w_idx];
                    if ww == 0.0 {
                        continue;
                    }
                    if ww > 0.0 {
                        sum += x[x_idx];
                    } else {
                        sum -= x[x_idx];
                    }
                }
            }
            out[oc * t_out + ot] = sum;
        }
    }
    t_out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_roundtrip() {
        let w = vec![-1.0, 0.0, 1.0, 0.0, 1.0, -1.0];
        let p = pack_trits(&w);
        assert_eq!(unpack_trits(&p, w.len()), w);
        assert!(is_ternary_f32(&w));
    }

    #[test]
    fn ternarize_exact() {
        let w = vec![0.01, -0.9, 0.5, 0.0, 0.8, -0.02];
        let t = ternarize(&w, 1.0 / 3.0);
        assert!(is_ternary_f32(&t));
        assert_eq!(t.iter().filter(|&&v| v != 0.0).count(), 2);
    }
}
