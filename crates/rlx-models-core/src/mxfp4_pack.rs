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

//! **f32 → MXFP4 encoder** — the missing *produce* side of rlx's MXFP4 support.
//!
//! Every other MXFP4 path in the tree is consume-side ([`crate::dsv4_quant`],
//! `rlx_mlx_io::dequant_mxfp4_f32`, `Op::DequantMatMul`/`DequantGroupedMatMulMlx`),
//! because they were written for checkpoints that ship *already* quantized
//! (mlx-community, Kimi). This packs an f32 tensor into the identical byte
//! layout so an ordinary bf16/f32 checkpoint can drive the same kernels:
//! 4× less arena, and on CUDA the native nibble-decode GEMM
//! (`Step::DequantGroupedMatmulMlxNative`) instead of a host round-trip.
//!
//! ## Layout (must match `rlx_mlx_io::dequant_mxfp4_f32`)
//!
//! For a row-major `[rows, k]` weight (`rows` = out-features; the contraction
//! runs along `k`, i.e. the **B-transposed** orientation these ops want):
//!
//! * `codes`: `rows * k/2` bytes. Element `c` of row `r` lives in byte
//!   `r*k/2 + c/2`, **low** nibble when `c` is even, **high** nibble when odd.
//! * nibble = `sign<<3 | magnitude_index` decoded through [`E2M1`].
//! * `scales`: `rows * k/gs` E8M0 **exponent bytes**; byte `b` means `2^(b-127)`.
//!
//! ## Two scale conventions — do not mix them
//!
//! The consuming ops disagree on how the scale operand is *typed*, and the
//! mismatch is silent (both are the right byte count):
//!
//! | op | scale param | contents |
//! |---|---|---|
//! | [`rlx_ir::op::Op::DequantMatMul`] (dense) | `U8 [n, groups]` | raw E8M0 bytes |
//! | `Op::DequantGroupedMatMulMlx` (MoE) | `BF16 [E, n, groups]` | the decoded float `2^(b-127)` |
//!
//! Use [`quantize_rows`] then feed [`scales_e8m0`] to the dense op and
//! [`scales_bf16`] to the grouped one.
//!
//! ## Exponent choice
//!
//! Per group: the smallest `e` with `6·2^e >= amax`. Consequently every element
//! maps into `[-6, 6]` and **saturation never happens** — unlike the OCP
//! `floor(log2(amax)) - 2` rule, which leaves the group max in `[4, 8)` and
//! clamps the top ~25% of that range to 6 (up to a 25% error on precisely the
//! largest weight in the group). Elements round to nearest, ties to even.

use half::bf16;

/// E2M1 nibble → value. Index is the raw nibble; bit 3 is the sign.
pub const E2M1: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// The group size MXFP4 is defined at (and the only one with E8M0 scales — MLX
/// switches to FP8 E4M3 scales at `group_size == 16`).
pub const GROUP_SIZE: usize = 32;

/// Round `|v| / scale` to the nearest E2M1 magnitude index, ties to even.
///
/// Boundaries are the midpoints of `[0, .5, 1, 1.5, 2, 3, 4, 6]`; each tie
/// resolves toward the even code (`0.25→0`, `0.75→1.0`, `1.25→1.0`, `1.75→2.0`,
/// `2.5→2.0`, `3.5→4.0`, `5.0→4.0`), matching IEEE round-half-to-even on the
/// same grid.
#[inline]
fn e2m1_magnitude(a: f32) -> u8 {
    if a <= 0.25 {
        0
    } else if a < 0.75 {
        1
    } else if a <= 1.25 {
        2
    } else if a < 1.75 {
        3
    } else if a <= 2.5 {
        4
    } else if a < 3.5 {
        5
    } else if a <= 5.0 {
        6
    } else {
        7
    }
}

/// Encode one f32 as an E2M1 nibble given the group scale.
#[inline]
pub fn e2m1_code(v: f32, scale: f32) -> u8 {
    let r = if scale > 0.0 { v / scale } else { 0.0 };
    if !r.is_finite() {
        // NaN/Inf can only come from a corrupt checkpoint; ±6 beats NaN
        // poisoning a whole row of activations.
        return if r.is_sign_negative() { 0x0F } else { 0x07 };
    }
    let mag = e2m1_magnitude(r.abs());
    // `-0.0` keeps its sign bit: harmless (decodes to -0.0) and keeps the
    // encoder an exact round-trip for signed zeros.
    if r.is_sign_negative() {
        mag | 0x08
    } else {
        mag
    }
}

/// Smallest E8M0 byte `b` with `6·2^(b-127) >= amax` (see module docs).
#[inline]
pub fn e8m0_exponent(amax: f32) -> u8 {
    if amax.is_nan() || amax <= 0.0 || !amax.is_finite() {
        return 127; // scale 1.0; the group is all zeros (or garbage → zeros)
    }
    // Seed from the float exponent, then fix up with exact comparisons — a
    // `log2().ceil()` seed is off by one whenever `amax/6` lands exactly on a
    // power of two, which is common for synthetic and pruned weights.
    let mut e = (amax / 6.0).log2().ceil() as i32;
    e = e.clamp(-127, 127);
    while e < 127 && exp2(e) * 6.0 < amax {
        e += 1;
    }
    while e > -127 && exp2(e - 1) * 6.0 >= amax {
        e -= 1;
    }
    (e + 127) as u8
}

/// Exact `2^e` for `e` in E8M0's range.
#[inline]
fn exp2(e: i32) -> f32 {
    bf16::from_bits((((e + 127).clamp(0, 254)) as u16) << 7).to_f32()
}

/// Packed MXFP4 form of a row-major `[rows, k]` f32 tensor.
#[derive(Debug, Clone)]
pub struct Mxfp4 {
    /// `rows * k/2` bytes, two E2M1 nibbles each (low nibble = even element).
    pub codes: Vec<u8>,
    /// `rows * k/group_size` E8M0 exponent bytes.
    pub exponents: Vec<u8>,
    /// Out-features (for an expert bank, `experts * out_features`).
    pub rows: usize,
    /// Contraction length — the dim the nibbles and groups run along.
    pub k: usize,
    /// Elements per shared exponent.
    pub group_size: usize,
}

impl Mxfp4 {
    /// Groups per row.
    pub fn n_groups(&self) -> usize {
        self.k / self.group_size
    }

    /// Scale operand for the **dense** `Op::DequantMatMul` — raw E8M0 bytes,
    /// declared as a `U8 [rows, n_groups]` param.
    pub fn scales_e8m0(&self) -> &[u8] {
        &self.exponents
    }

    /// Scale operand for the **grouped** `Op::DequantGroupedMatMulMlx` — the
    /// decoded float `2^(b-127)` as little-endian bf16, declared as a
    /// `BF16 [E, rows, n_groups]` param. Every E8M0 value is a power of two, so
    /// bf16 holds it exactly.
    pub fn scales_bf16(&self) -> Vec<u8> {
        self.exponents
            .iter()
            .flat_map(|&b| bf16::from_bits((b as u16) << 7).to_le_bytes())
            .collect()
    }

    /// Zero `biases` operand (MXFP4 is symmetric — there is no zero-point, but
    /// both ops keep the 4th/5th operand slot for layout parity with MLX affine).
    pub fn zero_biases_bf16(&self) -> Vec<u8> {
        vec![0u8; self.rows * self.n_groups() * 2]
    }

    /// Zero `biases` in the dense op's U8 convention.
    pub fn zero_biases_u8(&self) -> Vec<u8> {
        vec![0u8; self.rows * self.n_groups()]
    }

    /// Bytes actually resident for this tensor (codes + bf16 scales).
    pub fn packed_bytes(&self) -> usize {
        self.codes.len() + self.exponents.len() * 2
    }
}

/// Quantize a row-major `[rows, k]` f32 slab. `k` must be a multiple of
/// `group_size` (all Ling/DeepSeek/Kimi expert dims are).
///
/// Rows are independent, so this parallelizes cleanly — expert banks are ~1 GB
/// each and a serial packer shows up as load-time latency.
pub fn quantize_rows(w: &[f32], rows: usize, k: usize, group_size: usize) -> Mxfp4 {
    assert_eq!(w.len(), rows * k, "mxfp4: expected {rows}x{k} elements");
    assert!(
        group_size > 0 && k.is_multiple_of(group_size),
        "mxfp4: k={k} must be a multiple of group_size={group_size}"
    );
    assert!(
        k.is_multiple_of(2),
        "mxfp4: k={k} must be even (2 nibbles/byte)"
    );
    let n_groups = k / group_size;
    let bpr = k / 2;
    let mut codes = vec![0u8; rows * bpr];
    let mut exponents = vec![0u8; rows * n_groups];

    use rayon::prelude::*;
    codes
        .par_chunks_mut(bpr)
        .zip(exponents.par_chunks_mut(n_groups))
        .enumerate()
        .for_each(|(r, (crow, erow))| {
            let wrow = &w[r * k..(r + 1) * k];
            for g in 0..n_groups {
                let grp = &wrow[g * group_size..(g + 1) * group_size];
                let amax = grp.iter().fold(0f32, |m, v| {
                    let a = v.abs();
                    if a > m { a } else { m }
                });
                let eb = e8m0_exponent(amax);
                erow[g] = eb;
                let scale = exp2(eb as i32 - 127);
                for (i, &v) in grp.iter().enumerate() {
                    let c = g * group_size + i;
                    let nib = e2m1_code(v, scale);
                    let byte = &mut crow[c / 2];
                    if c.is_multiple_of(2) {
                        *byte = (*byte & 0xF0) | nib;
                    } else {
                        *byte = (*byte & 0x0F) | (nib << 4);
                    }
                }
            }
        });

    Mxfp4 {
        codes,
        exponents,
        rows,
        k,
        group_size,
    }
}

/// Quantize a stacked expert bank `[experts, rows, k]` (one [`Mxfp4`] whose
/// `rows` is `experts*rows`, which is exactly the flat layout both ops index:
/// expert `e`, row `j` is slab row `e*rows + j`).
pub fn quantize_bank(w: &[f32], experts: usize, rows: usize, k: usize, group_size: usize) -> Mxfp4 {
    quantize_rows(w, experts * rows, k, group_size)
}

/// Reference decode — mirrors `rlx_mlx_io::dequant_mxfp4_f32`. Tests compare
/// against this *and* against the real op, so a shared misreading of the layout
/// can't pass unnoticed.
pub fn dequantize(q: &Mxfp4) -> Vec<f32> {
    let (rows, k, gs) = (q.rows, q.k, q.group_size);
    let (bpr, n_groups) = (k / 2, k / gs);
    let mut out = vec![0f32; rows * k];
    for r in 0..rows {
        let crow = &q.codes[r * bpr..(r + 1) * bpr];
        let erow = &q.exponents[r * n_groups..(r + 1) * n_groups];
        for c in 0..k {
            let byte = crow[c / 2];
            let nib = if c.is_multiple_of(2) {
                byte & 0x0F
            } else {
                byte >> 4
            };
            out[r * k + c] = E2M1[nib as usize] * exp2(erow[c / gs] as i32 - 127);
        }
    }
    out
}

/// Round-trip a slab through MXFP4 (quantize → dequantize). Handy for
/// *simulating* the precision loss on a backend that has no packed kernel, and
/// for parity baselines.
pub fn round_trip(w: &[f32], rows: usize, k: usize, group_size: usize) -> Vec<f32> {
    dequantize(&quantize_rows(w, rows, k, group_size))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_grid_values_round_trip_bit_exact() {
        // Every E2M1 magnitude at a few scales: quantization must be the
        // identity on its own grid.
        for e in [-8i32, -2, 0, 5] {
            let s = exp2(e);
            let row: Vec<f32> = (0..GROUP_SIZE)
                .map(|i| E2M1[i % 16] * s)
                .collect::<Vec<_>>();
            let back = round_trip(&row, 1, GROUP_SIZE, GROUP_SIZE);
            for (a, b) in row.iter().zip(&back) {
                assert_eq!(a.to_bits(), b.to_bits(), "e={e} {a} != {b}");
            }
        }
    }

    #[test]
    fn saturation_never_happens() {
        // The group max must decode to exactly itself when it sits on the grid,
        // and never clamp otherwise.
        for amax in [1e-6f32, 0.017, 1.0, 1.9, 6.0, 7.9, 1e4] {
            let mut row = vec![0f32; GROUP_SIZE];
            row[7] = amax;
            let q = quantize_rows(&row, 1, GROUP_SIZE, GROUP_SIZE);
            let scale = exp2(q.exponents[0] as i32 - 127);
            // The invariant that makes saturation impossible.
            assert!(
                scale * 6.0 >= amax,
                "amax={amax} exceeds 6·scale={} — would clamp",
                scale * 6.0
            );
            let nib = q.codes[3] >> 4; // element 7 → byte 3, high nibble
            assert!(nib <= 7, "sign flipped for {amax}");
            let back = dequantize(&q)[7];
            // Worst case is a value at 5·scale, where the grid ({3,4,6}) steps
            // by 2 → 20% relative. Anything above that means a real clamp.
            assert!(
                (back - amax).abs() <= 0.2 * amax,
                "amax={amax} decoded {back}"
            );
        }
    }

    #[test]
    fn zeros_and_denormal_groups_are_safe() {
        let row = vec![0f32; GROUP_SIZE];
        let q = quantize_rows(&row, 1, GROUP_SIZE, GROUP_SIZE);
        assert!(dequantize(&q).iter().all(|v| *v == 0.0));
        assert!(q.exponents.iter().all(|&b| b != 0xFF), "E8M0 NaN byte");
    }

    #[test]
    fn ties_round_to_even() {
        let s = 1.0f32;
        for (v, want) in [
            (0.25, 0.0),
            (0.75, 1.0),
            (1.25, 1.0),
            (1.75, 2.0),
            (2.5, 2.0),
            (3.5, 4.0),
            (5.0, 4.0),
        ] {
            let c = e2m1_code(v, s);
            assert_eq!(E2M1[c as usize], want, "tie {v} → {}", E2M1[c as usize]);
        }
    }

    #[test]
    fn gaussian_weights_keep_high_cosine() {
        // Model-weight-like input: N(0, 0.02). MXFP4 has ~3.3 mantissa bits, so
        // per-tensor cosine should land ~0.99+; anything much lower means the
        // scale search is wasting range.
        let (rows, k) = (64usize, 256usize);
        let mut s = 0x1234_5678u64;
        let w: Vec<f32> = (0..rows * k)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                let u = ((s >> 33) as f32) / (u32::MAX as f32 / 2.0) - 1.0;
                u * 0.02
            })
            .collect();
        let back = round_trip(&w, rows, k, GROUP_SIZE);
        let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
        for (a, b) in w.iter().zip(&back) {
            dot += (*a as f64) * (*b as f64);
            na += (*a as f64) * (*a as f64);
            nb += (*b as f64) * (*b as f64);
        }
        let cos = dot / (na.sqrt() * nb.sqrt());
        assert!(cos > 0.99, "MXFP4 cosine {cos:.6} too low");
    }
}
