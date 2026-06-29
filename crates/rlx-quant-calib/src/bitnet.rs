// RLX models — calibration quantization.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! BitNet b1.58 — **ternary** weight quantization.
//!
//! BitNet quantizes each linear weight to one of three values `{-1, 0, +1}`
//! ("1.58 bits" = log2(3)) with a single per-tensor scale, and quantizes
//! activations to int8. The weight quantizer is the *absmean* rule from the
//! BitNet b1.58 paper:
//!
//! ```text
//! scale = mean(|W|)
//! W_q   = clamp(round(W / scale), -1, +1)
//! W ≈ scale · W_q
//! ```
//!
//! Two storage wins: ternary values pack 4-per-byte (2 bits each) — 16× vs
//! f32 — and the matmul becomes add/subtract/skip (no multiplies). This module
//! is host-side: it produces the ternary weights, the 2-bit packing, and the
//! int8 activation quantization. Running them is a `DequantMatMul` over the
//! dequantized `{-scale, 0, +scale}` weight (works today on every backend), or
//! a future packed-ternary kernel for the multiply-free speedup.

/// A ternary-quantized linear weight. `t[i] ∈ {-1, 0, 1}`, row-major
/// `[out * inn]`; the dequantized weight is `scale · t`.
#[derive(Debug, Clone)]
pub struct TernaryQuant {
    pub t: Vec<i8>,
    /// Per-tensor absmean scale.
    pub scale: f32,
    pub out: usize,
    pub inn: usize,
}

/// BitNet b1.58 absmean weight quantization (per-tensor scale).
pub fn quantize_bitnet(w: &[f32], out: usize, inn: usize) -> TernaryQuant {
    let n = (out * inn).max(1);
    let mean_abs = w.iter().map(|x| x.abs()).sum::<f32>() / n as f32;
    let scale = if mean_abs > 0.0 { mean_abs } else { 1.0 };
    let inv = 1.0 / scale;
    let t = w
        .iter()
        .map(|&x| (x * inv).round().clamp(-1.0, 1.0) as i8)
        .collect();
    TernaryQuant { t, scale, out, inn }
}

/// Dequantize: `scale · t` (exact reconstruction of the ternary approximation).
pub fn dequantize_bitnet(q: &TernaryQuant) -> Vec<f32> {
    q.t.iter().map(|&v| v as f32 * q.scale).collect()
}

/// Pack ternary `{-1, 0, 1}` into 2 bits each, 4 values per byte
/// (encoding `v + 1 ∈ {0, 1, 2}`, low bits first). The tail is zero-padded.
pub fn pack_ternary(t: &[i8]) -> Vec<u8> {
    let mut out = vec![0u8; t.len().div_ceil(4)];
    for (i, &v) in t.iter().enumerate() {
        let code = (v + 1) as u8 & 0b11; // -1→0, 0→1, 1→2
        out[i / 4] |= code << ((i % 4) * 2);
    }
    out
}

/// Inverse of [`pack_ternary`]: unpack `n` ternary values from packed bytes.
pub fn unpack_ternary(packed: &[u8], n: usize) -> Vec<i8> {
    (0..n)
        .map(|i| {
            let code = (packed[i / 4] >> ((i % 4) * 2)) & 0b11;
            code as i8 - 1
        })
        .collect()
}

/// Per-row (per-token) int8 absmax activation quantization — the activation
/// half of BitNet. Returns the int8 values `[rows * cols]` and the per-row
/// scale `[rows]`; dequant is `q[r,c] · scale[r]`.
pub fn quantize_activations_int8(x: &[f32], rows: usize, cols: usize) -> (Vec<i8>, Vec<f32>) {
    let mut q = vec![0i8; rows * cols];
    let mut scales = vec![0f32; rows];
    for r in 0..rows {
        let row = &x[r * cols..(r + 1) * cols];
        let amax = row.iter().fold(0f32, |m, v| m.max(v.abs()));
        let scale = if amax > 0.0 { amax / 127.0 } else { 1.0 };
        let inv = 1.0 / scale;
        for c in 0..cols {
            q[r * cols + c] = (row[c] * inv).round().clamp(-127.0, 127.0) as i8;
        }
        scales[r] = scale;
    }
    (q, scales)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let (mut d, mut na, mut nb) = (0.0f32, 0.0f32, 0.0f32);
        for (x, y) in a.iter().zip(b) {
            d += x * y;
            na += x * x;
            nb += y * y;
        }
        d / (na.sqrt() * nb.sqrt() + 1e-12)
    }

    #[test]
    fn ternary_values_in_range_and_dequant_exact() {
        let w: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).sin() * 0.7).collect();
        let q = quantize_bitnet(&w, 8, 8);
        assert!(q.t.iter().all(|&v| (-1..=1).contains(&v)), "ternary range");
        assert!(q.scale > 0.0);
        let dq = dequantize_bitnet(&q);
        for (i, &v) in q.t.iter().enumerate() {
            assert_eq!(dq[i], v as f32 * q.scale, "dequant is scale·t exactly");
        }
    }

    #[test]
    fn reconstructs_near_ternary_weights_well() {
        // A weight already drawn from {-s, 0, +s} must reconstruct almost
        // exactly (the quantizer recovers the ternary structure).
        let s = 0.42f32;
        let pattern = [-1i8, 0, 1, 1, -1, 0, 0, 1];
        let w: Vec<f32> = (0..128).map(|i| pattern[i % 8] as f32 * s).collect();
        let q = quantize_bitnet(&w, 16, 8);
        let dq = dequantize_bitnet(&q);
        assert!(cosine(&dq, &w) > 0.999, "near-ternary reconstruction");
    }

    #[test]
    fn pack_unpack_roundtrips_exact() {
        let t: Vec<i8> = (0..103).map(|i| (i % 3) as i8 - 1).collect(); // -1,0,1,...
        let packed = pack_ternary(&t);
        assert_eq!(packed.len(), t.len().div_ceil(4));
        let back = unpack_ternary(&packed, t.len());
        assert_eq!(back, t, "2-bit pack round-trip");
    }

    #[test]
    fn packing_is_16x_smaller_than_f32() {
        let n = 4096;
        let t = vec![1i8; n];
        let packed = pack_ternary(&t);
        // 2 bits/weight vs 32 bits/weight = 16×.
        assert_eq!(packed.len(), n / 4);
        assert!(packed.len() * 16 <= n * 4);
    }

    #[test]
    fn int8_activation_quant_roundtrips_close() {
        let (rows, cols) = (3usize, 32usize);
        let x: Vec<f32> = (0..rows * cols)
            .map(|i| (i as f32 * 0.05).cos() * 2.0)
            .collect();
        let (q, scales) = quantize_activations_int8(&x, rows, cols);
        let mut dq = vec![0f32; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                dq[r * cols + c] = q[r * cols + c] as f32 * scales[r];
            }
        }
        assert!(cosine(&dq, &x) > 0.999, "int8 activation round-trip");
    }
}
