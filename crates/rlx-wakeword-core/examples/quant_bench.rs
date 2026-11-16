// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! f32 against int8 for the wake network's heaviest layer (a residual block: 64 → 64 channels,
//! 50 positions, 3 taps), on this CPU: the f32 path is the engine's (`ops::conv1d_dilated`,
//! NEON fused multiply-adds on aarch64); the int8 one quantizes the input per tensor and the
//! weights per output channel and accumulates in i32 with NEON widening multiplies
//! (`smull` + `sadalp`), then rescales. Prints time per call and the int8 error.
//!
//!   cargo run --release -p rlx-wakeword-core --example quant_bench

use std::time::Instant;

use rlx_wakeword_core::ops::conv1d_dilated;

const C: usize = 64;
const T: usize = 50;
const K: usize = 3;

fn val(i: usize) -> f32 {
    ((i * 2654435761) % 1000) as f32 / 500.0 - 1.0
}

/// int8 dilated conv: `x` quantized with one scale, `w` per output channel; i32 accumulate.
#[allow(clippy::too_many_arguments)]
fn conv_i8(xq: &[i8], sx: f32, wq: &[i8], sw: &[f32], b: &[f32], d: usize, out: &mut [f32]) {
    let edge = d;
    let tp = T + 2 * edge;
    // Padded input, each row then rearranged tap-major so a position's (channel, tap) values
    // are contiguous per tap: xp[(i * K + j) * T + p] = x[i, p + (j − 1)·d].
    let mut xt = vec![0i8; C * K * T];
    for i in 0..C {
        for j in 0..K {
            for p in 0..T {
                let q = p as isize + (j as isize - 1) * d as isize;
                if (0..T as isize).contains(&q) {
                    xt[(i * K + j) * T + p] = xq[i * T + q as usize];
                }
            }
        }
    }
    let _ = tp;
    for o in 0..C {
        let wrow = &wq[o * C * K..(o + 1) * C * K];
        let scale = sx * sw[o];
        let mut p = 0;
        #[cfg(target_arch = "aarch64")]
        unsafe {
            use core::arch::aarch64::*;
            while p + 8 <= T {
                let mut acc_lo = vdupq_n_s32(0);
                let mut acc_hi = vdupq_n_s32(0);
                for (ij, &wv) in wrow.iter().enumerate() {
                    let xv = vld1_s8(xt.as_ptr().add(ij * T + p));
                    let prod = vmull_s8(xv, vdup_n_s8(wv)); // 8 × i16
                    acc_lo = vaddw_s16(acc_lo, vget_low_s16(prod));
                    acc_hi = vaddw_s16(acc_hi, vget_high_s16(prod));
                }
                let o0 = out.as_mut_ptr().add(o * T + p);
                vst1q_f32(o0, vmlaq_n_f32(vdupq_n_f32(b[o]), vcvtq_f32_s32(acc_lo), scale));
                vst1q_f32(o0.add(4), vmlaq_n_f32(vdupq_n_f32(b[o]), vcvtq_f32_s32(acc_hi), scale));
                p += 8;
            }
        }
        for pp in p..T {
            let mut acc = 0i32;
            for (ij, &wv) in wrow.iter().enumerate() {
                acc += i32::from(xt[ij * T + pp]) * i32::from(wv);
            }
            out[o * T + pp] = b[o] + acc as f32 * scale;
        }
    }
}

fn main() {
    let x: Vec<f32> = (0..C * T).map(|i| val(i).max(0.0) * 3.0).collect();
    let w: Vec<f32> = (0..C * C * K).map(|i| val(i + 3) * 0.1).collect();
    let b: Vec<f32> = (0..C).map(|i| val(i + 77) * 0.1).collect();
    let mut out_f = vec![0.0; C * T];
    let mut out_q = vec![0.0; C * T];
    let iters = 20_000;
    let mut pad = Vec::new();
    for d in [1usize, 8] {
        let t = Instant::now();
        for _ in 0..iters {
            conv1d_dilated(&x, C, T, &w, C, K, d, &b, &mut out_f, 0, T, &mut pad);
        }
        let f32_us = t.elapsed().as_secs_f64() * 1e6 / iters as f64;
        // Weights per output channel, once (a deployed model); the input per tensor, per call.
        let (mut wq, mut sw) = (vec![0i8; w.len()], vec![0.0f32; C]);
        for o in 0..C {
            let row = &w[o * C * K..(o + 1) * C * K];
            sw[o] = row.iter().fold(0.0f32, |m, v| m.max(v.abs())) / 127.0;
            for (q, v) in wq[o * C * K..(o + 1) * C * K].iter_mut().zip(row) {
                *q = (v / sw[o]).round().clamp(-127.0, 127.0) as i8;
            }
        }
        let t = Instant::now();
        for _ in 0..iters {
            let sx = x.iter().fold(0.0f32, |m, v| m.max(v.abs())) / 127.0;
            let xq: Vec<i8> = x.iter().map(|v| (v / sx).round().clamp(-127.0, 127.0) as i8).collect();
            conv_i8(&xq, sx, &wq, &sw, &b, d, &mut out_q);
        }
        let i8_us = t.elapsed().as_secs_f64() * 1e6 / iters as f64;
        let err = out_f.iter().zip(&out_q).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        let mag = out_f.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        println!(
            "dilation {d}: f32 {f32_us:.1} µs, int8 {i8_us:.1} µs (input quantized per call), max error {err:.4} of {mag:.2}"
        );
    }
}
