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

//! Probe whether sgemm orientation (M=big,N=1 vs M=1,N=big) hits different
//! Accelerate internal paths. Both are mathematically the same FLOP count.
//!
//! Also probes: alignment, K-padding to multiples of 32/64, and the small-M
//! NEON kernel path for transposed-orientation matvec.

use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::time::Instant;

const ITERS: usize = 500;

#[derive(Clone, Copy)]
struct S {
    m: usize,
    k: usize,
    label: &'static str,
}

fn rng_vec(rng: &mut StdRng, n: usize) -> Vec<f32> {
    (0..n).map(|_| rng.r#gen::<f32>() - 0.5).collect()
}

/// Round up to a multiple of `mult`.
fn round_up(n: usize, mult: usize) -> usize {
    n.div_ceil(mult) * mult
}

/// Allocate Vec<f32> with at least 64-byte alignment via a sentinel pad.
/// Returns the aligned slice plus the owning Vec.
fn aligned_vec(n: usize) -> (Vec<f32>, usize) {
    // Over-allocate so that the data ptr+offset is 64-byte aligned.
    let pad = 64 / std::mem::size_of::<f32>(); // 16 f32 = 64 bytes
    let v = vec![0f32; n + pad];
    let addr = v.as_ptr() as usize;
    let off_bytes = (64 - (addr % 64)) % 64;
    let off = off_bytes / 4;
    assert!(off < pad);
    (v, off)
}

fn bench(shape: S) {
    let S { m, k, label } = shape;
    let mut rng = StdRng::seed_from_u64(7);
    let w = rng_vec(&mut rng, m * k);
    let x = rng_vec(&mut rng, k);

    // ── Variant 0: baseline.  m=big, n=1, A=W[m,k], B=x[k,1], C=out[m,1] ──
    let mut out0 = vec![0f32; m];
    for _ in 0..3 {
        rlx_cpu::blas::sgemm(&w, &x, &mut out0, m, k, 1);
    }
    let t = Instant::now();
    for _ in 0..ITERS {
        rlx_cpu::blas::sgemm(&w, &x, &mut out0, m, k, 1);
    }
    let t_v0 = t.elapsed().as_secs_f64() / ITERS as f64;

    // ── Variant 1: transposed orientation.  A=x[1,k], B=W^T[k,m], C=out[1,m] ──
    // W is row-major [m,k]; W^T is therefore col-major [k,m].  Use TransB.
    let mut out1 = vec![0f32; m];
    for _ in 0..3 {
        unsafe {
            cblas_sgemm_trans_b(&x, &w, &mut out1, 1, k, m);
        }
    }
    let t = Instant::now();
    for _ in 0..ITERS {
        unsafe {
            cblas_sgemm_trans_b(&x, &w, &mut out1, 1, k, m);
        }
    }
    let t_v1 = t.elapsed().as_secs_f64() / ITERS as f64;

    // ── Variant 2: K-padded baseline.  Pad K to multiple of 32 by allocating
    //    a [m, k_pad] zero-padded weight + [k_pad] zero-padded input.  Tests
    //    whether AMX likes K aligned to tile boundaries. ──
    let k_pad = round_up(k, 32);
    let mut w_pad = vec![0f32; m * k_pad];
    for i in 0..m {
        w_pad[i * k_pad..i * k_pad + k].copy_from_slice(&w[i * k..(i + 1) * k]);
    }
    let mut x_pad = vec![0f32; k_pad];
    x_pad[..k].copy_from_slice(&x);
    let mut out2 = vec![0f32; m];
    for _ in 0..3 {
        rlx_cpu::blas::sgemm(&w_pad, &x_pad, &mut out2, m, k_pad, 1);
    }
    let t = Instant::now();
    for _ in 0..ITERS {
        rlx_cpu::blas::sgemm(&w_pad, &x_pad, &mut out2, m, k_pad, 1);
    }
    let t_v2 = t.elapsed().as_secs_f64() / ITERS as f64;

    // ── Variant 3: 64-byte aligned baseline. ──
    let (mut w_aligned_buf, w_off) = aligned_vec(m * k);
    w_aligned_buf[w_off..w_off + m * k].copy_from_slice(&w);
    let (mut x_aligned_buf, x_off) = aligned_vec(k);
    x_aligned_buf[x_off..x_off + k].copy_from_slice(&x);
    let (mut out_aligned_buf, o_off) = aligned_vec(m);
    let w_aligned = &w_aligned_buf[w_off..w_off + m * k];
    let x_aligned = &x_aligned_buf[x_off..x_off + k];
    {
        let out_aligned = &mut out_aligned_buf[o_off..o_off + m];
        for _ in 0..3 {
            rlx_cpu::blas::sgemm(w_aligned, x_aligned, out_aligned, m, k, 1);
        }
    }
    let t = Instant::now();
    for _ in 0..ITERS {
        let out_aligned = &mut out_aligned_buf[o_off..o_off + m];
        rlx_cpu::blas::sgemm(w_aligned, x_aligned, out_aligned, m, k, 1);
    }
    let t_v3 = t.elapsed().as_secs_f64() / ITERS as f64;

    // ── Verify all four variants produce same result. ──
    let max_err_01: f32 = out0
        .iter()
        .zip(out1.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    let max_err_02: f32 = out0
        .iter()
        .zip(out2.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    let max_err_03: f32 = out0
        .iter()
        .zip(out_aligned_buf[o_off..o_off + m].iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);

    println!(
        "{label:>22}  M={m:>5} K={k:>5}  base={:>6.1}µs  transB={:>6.1}µs ({:>4.2}×)  Kpad={:>6.1}µs ({:>4.2}×)  align64={:>6.1}µs ({:>4.2}×)  errs={:.0e}/{:.0e}/{:.0e}",
        t_v0 * 1e6,
        t_v1 * 1e6,
        t_v0 / t_v1.max(1e-12),
        t_v2 * 1e6,
        t_v0 / t_v2.max(1e-12),
        t_v3 * 1e6,
        t_v0 / t_v3.max(1e-12),
        max_err_01,
        max_err_02,
        max_err_03,
    );
}

/// cblas_sgemm with transA = NoTrans, transB = Trans.
/// A is [m, k] row-major. B is [n, k] row-major (i.e. logical B^T is [k, n]).
/// C = A @ B^T, [m, n].
unsafe fn cblas_sgemm_trans_b(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    use std::os::raw::c_int;
    const ROW_MAJOR: c_int = 101;
    const NO_TRANS: c_int = 111;
    const TRANS: c_int = 112;
    unsafe extern "C" {
        fn cblas_sgemm(
            order: c_int,
            transa: c_int,
            transb: c_int,
            m: c_int,
            n: c_int,
            k: c_int,
            alpha: f32,
            a: *const f32,
            lda: c_int,
            b: *const f32,
            ldb: c_int,
            beta: f32,
            c: *mut f32,
            ldc: c_int,
        );
    }
    unsafe {
        cblas_sgemm(
            ROW_MAJOR,
            NO_TRANS,
            TRANS,
            m as c_int,
            n as c_int,
            k as c_int,
            1.0,
            a.as_ptr(),
            k as c_int,
            b.as_ptr(),
            k as c_int,
            0.0,
            c.as_mut_ptr(),
            n as c_int,
        );
    }
}

fn main() {
    println!("=== Accelerate sgemm orientation/padding/alignment probe ===");
    println!("    All variants computing the SAME math (out = W @ x).\n");
    let shapes = [
        S {
            m: 2048,
            k: 768,
            label: "CP wqkv",
        },
        S {
            m: 768,
            k: 1024,
            label: "CP wo",
        },
        S {
            m: 6144,
            k: 768,
            label: "CP gate+up",
        },
        S {
            m: 768,
            k: 3072,
            label: "CP down",
        },
        S {
            m: 2048,
            k: 1024,
            label: "Talker wqkv",
        },
        S {
            m: 1024,
            k: 1024,
            label: "Talker wo",
        },
        S {
            m: 8192,
            k: 1024,
            label: "Talker gate+up",
        },
        S {
            m: 1024,
            k: 4096,
            label: "Talker down",
        },
    ];
    for s in shapes {
        bench(s);
    }
}
