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

//! Benchmark BF16-weight × F32-input matvec vs Apple Accelerate sgemv.
//!
//! Validates correctness and measures speedup at CP-relevant matrix sizes
//! before integrating into the CP forward path.
//!
//! Approach: weights stored as BF16 (`&[u16]`, raw bits). On the fly we
//! widen 8 BF16 values to 8 F32 values via NEON `vshll_n_u16` (zero-extend
//! then shift left 16 bits — exactly BF16→F32 since BF16 is the upper 16
//! bits of an F32). Then standard F32 FMA. Halves weight memory bandwidth
//! without needing the BF16 hardware path.

use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rayon::prelude::*;
use std::time::Instant;

/// Convert one f32 to its BF16 bit representation (truncating; HF-style).
#[inline(always)]
fn f32_to_bf16_bits(x: f32) -> u16 {
    (x.to_bits() >> 16) as u16
}

/// Parallel version: distribute rows across rayon worker threads.
#[cfg(target_arch = "aarch64")]
fn matvec_bf16_neon_par(w: &[u16], x: &[f32], out: &mut [f32], out_dim: usize, in_dim: usize) {
    debug_assert_eq!(w.len(), out_dim * in_dim);
    debug_assert_eq!(x.len(), in_dim);
    debug_assert_eq!(out.len(), out_dim);
    out.par_chunks_mut(64)
        .enumerate()
        .for_each(|(chunk_idx, out_chunk)| {
            let row_off = chunk_idx * 64;
            let rows = out_chunk.len();
            let w_chunk = &w[row_off * in_dim..(row_off + rows) * in_dim];
            unsafe {
                matvec_bf16_neon(w_chunk, x, out_chunk, rows, in_dim);
            }
        });
}

/// out = W · x, where W is row-major [out_dim, in_dim] stored as BF16 bits.
///
/// SAFETY: requires aarch64 NEON. Caller must ensure `w.len() == out_dim * in_dim`
/// and that `in_dim` is a multiple of 8 (slow path for tail otherwise).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn matvec_bf16_neon(w: &[u16], x: &[f32], out: &mut [f32], out_dim: usize, in_dim: usize) {
    use std::arch::aarch64::*;
    debug_assert_eq!(w.len(), out_dim * in_dim);
    debug_assert_eq!(x.len(), in_dim);
    debug_assert_eq!(out.len(), out_dim);

    let lanes = 8usize;
    let bulk = in_dim - (in_dim % lanes);
    let x_ptr = x.as_ptr();
    // SAFETY: caller upholds the function's safety contract (aarch64 NEON,
    // correct slice lengths). All pointer arithmetic stays in bounds.
    unsafe {
        for co in 0..out_dim {
            let w_row = w.as_ptr().add(co * in_dim);
            let mut acc0 = vdupq_n_f32(0.0);
            let mut acc1 = vdupq_n_f32(0.0);
            let mut ci = 0usize;
            while ci < bulk {
                let w_u16 = vld1q_u16(w_row.add(ci));
                let w_lo_u32 = vshll_n_u16(vget_low_u16(w_u16), 16);
                let w_hi_u32 = vshll_high_n_u16::<16>(w_u16);
                let w_lo_f32 = vreinterpretq_f32_u32(w_lo_u32);
                let w_hi_f32 = vreinterpretq_f32_u32(w_hi_u32);
                let x_lo = vld1q_f32(x_ptr.add(ci));
                let x_hi = vld1q_f32(x_ptr.add(ci + 4));
                acc0 = vfmaq_f32(acc0, w_lo_f32, x_lo);
                acc1 = vfmaq_f32(acc1, w_hi_f32, x_hi);
                ci += lanes;
            }
            let mut sum = vaddvq_f32(vaddq_f32(acc0, acc1));
            while ci < in_dim {
                let bf = *w_row.add(ci);
                let f = f32::from_bits((bf as u32) << 16);
                sum += f * *x_ptr.add(ci);
                ci += 1;
            }
            *out.get_unchecked_mut(co) = sum;
        }
    }
}

/// Reference F32 matvec via Accelerate sgemm (N=1).
fn matvec_f32_sgemm(w: &[f32], x: &[f32], out: &mut [f32], out_dim: usize, in_dim: usize) {
    rlx_cpu::blas::sgemm(w, x, out, out_dim, in_dim, 1);
}

/// Pure Rust f32 scalar reference (correctness witness).
fn matvec_f32_scalar(w: &[f32], x: &[f32], out: &mut [f32], out_dim: usize, in_dim: usize) {
    for co in 0..out_dim {
        let row = &w[co * in_dim..(co + 1) * in_dim];
        let mut s = 0f32;
        for i in 0..in_dim {
            s += row[i] * x[i];
        }
        out[co] = s;
    }
}

fn max_abs_err(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
}

fn mean_abs(a: &[f32]) -> f32 {
    a.iter().map(|v| v.abs()).sum::<f32>() / a.len().max(1) as f32
}

#[derive(Debug, Clone, Copy)]
struct Shape {
    out_dim: usize,
    in_dim: usize,
    label: &'static str,
}

fn bench_one(seed: u64, shape: Shape, iters: usize) {
    let Shape {
        out_dim,
        in_dim,
        label,
    } = shape;
    let mut rng = StdRng::seed_from_u64(seed);

    let w_f32: Vec<f32> = (0..out_dim * in_dim)
        .map(|_| rng.r#gen::<f32>() * 2.0 - 1.0)
        .collect();
    let x_f32: Vec<f32> = (0..in_dim)
        .map(|_| rng.r#gen::<f32>() * 2.0 - 1.0)
        .collect();
    // BF16-quantize the weight matrix (truncating round) into u16 storage.
    let w_bf16: Vec<u16> = w_f32.iter().map(|&v| f32_to_bf16_bits(v)).collect();
    // Reconstruct the f32-as-bf16 reference for an apples-to-apples accuracy check.
    let w_f32_as_bf16: Vec<f32> = w_bf16
        .iter()
        .map(|&bits| f32::from_bits((bits as u32) << 16))
        .collect();

    let mut out_ref = vec![0f32; out_dim];
    let mut out_sgemm = vec![0f32; out_dim];
    let mut out_bf16 = vec![0f32; out_dim];
    let mut out_bf16_par = vec![0f32; out_dim];

    matvec_f32_scalar(&w_f32_as_bf16, &x_f32, &mut out_ref, out_dim, in_dim);
    matvec_f32_sgemm(&w_f32_as_bf16, &x_f32, &mut out_sgemm, out_dim, in_dim);
    #[cfg(target_arch = "aarch64")]
    unsafe {
        matvec_bf16_neon(&w_bf16, &x_f32, &mut out_bf16, out_dim, in_dim);
    }
    #[cfg(target_arch = "aarch64")]
    matvec_bf16_neon_par(&w_bf16, &x_f32, &mut out_bf16_par, out_dim, in_dim);
    #[cfg(not(target_arch = "aarch64"))]
    {
        out_bf16.copy_from_slice(&out_ref);
    }

    let err_bf16 = max_abs_err(&out_bf16, &out_ref);
    let err_sgemm = max_abs_err(&out_sgemm, &out_ref);
    let mean = mean_abs(&out_ref);

    // Warmup.
    for _ in 0..3 {
        matvec_f32_sgemm(&w_f32_as_bf16, &x_f32, &mut out_sgemm, out_dim, in_dim);
        #[cfg(target_arch = "aarch64")]
        unsafe {
            matvec_bf16_neon(&w_bf16, &x_f32, &mut out_bf16, out_dim, in_dim);
        }
    }

    let t = Instant::now();
    for _ in 0..iters {
        matvec_f32_sgemm(&w_f32_as_bf16, &x_f32, &mut out_sgemm, out_dim, in_dim);
    }
    let t_sgemm = t.elapsed().as_secs_f64() / iters as f64;

    #[cfg(target_arch = "aarch64")]
    let t_bf16 = {
        let t = Instant::now();
        for _ in 0..iters {
            unsafe {
                matvec_bf16_neon(&w_bf16, &x_f32, &mut out_bf16, out_dim, in_dim);
            }
        }
        t.elapsed().as_secs_f64() / iters as f64
    };
    #[cfg(not(target_arch = "aarch64"))]
    let t_bf16 = t_sgemm;

    #[cfg(target_arch = "aarch64")]
    let t_bf16_par = {
        let t = Instant::now();
        for _ in 0..iters {
            matvec_bf16_neon_par(&w_bf16, &x_f32, &mut out_bf16_par, out_dim, in_dim);
        }
        t.elapsed().as_secs_f64() / iters as f64
    };
    #[cfg(not(target_arch = "aarch64"))]
    let t_bf16_par = t_sgemm;

    let speedup = t_sgemm / t_bf16.max(1e-12);
    let speedup_par = t_sgemm / t_bf16_par.max(1e-12);
    println!(
        "{label:>22}  M={out_dim:>5} K={in_dim:>5}  sgemm={:>7.2}µs  bf16_1t={:>7.2}µs ({:>4.2}×)  bf16_par={:>7.2}µs ({:>4.2}×)  err={:.2e}",
        t_sgemm * 1e6,
        t_bf16 * 1e6,
        speedup,
        t_bf16_par * 1e6,
        speedup_par,
        err_bf16,
    );
    let _ = (err_sgemm, mean);
}

fn main() {
    println!("=== BF16-weight × F32-input matvec vs Accelerate sgemm ===");
    println!("    aarch64 NEON path, CP-relevant CPU sizes\n");
    // CP shapes (hidden=768, q_dim=1024, kv_dim=512, inter_dim=3072):
    let shapes = [
        Shape {
            out_dim: 2048,
            in_dim: 768,
            label: "CP wqkv (Q+2KV)",
        }, // q_dim + 2*kv_dim
        Shape {
            out_dim: 768,
            in_dim: 1024,
            label: "CP wo",
        },
        Shape {
            out_dim: 6144,
            in_dim: 768,
            label: "CP gate+up",
        }, // 2*inter_dim
        Shape {
            out_dim: 768,
            in_dim: 3072,
            label: "CP down",
        },
        // Talker shapes (larger; hidden=1024 typical for 0.6B):
        Shape {
            out_dim: 2048,
            in_dim: 1024,
            label: "Talker wqkv",
        },
        Shape {
            out_dim: 1024,
            in_dim: 1024,
            label: "Talker wo",
        },
        Shape {
            out_dim: 8192,
            in_dim: 1024,
            label: "Talker gate+up",
        },
        Shape {
            out_dim: 1024,
            in_dim: 4096,
            label: "Talker down",
        },
    ];
    let iters = 200;
    for s in shapes {
        bench_one(42, s, iters);
    }
}
