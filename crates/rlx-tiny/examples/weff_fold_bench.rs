// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
//! Speed of the `W_eff` fold: one projection (stages=2 + LoRA), fwd+bwd on Metal.
//! (a) per-stage `Σ x.synth_matmul + (x·A)·Bᵀ` (current) vs
//! (b) `W_eff = Σ reconstruct + A·Bᵀ; x·W_eff` (fold → 1 fwd GEMM, 2 bwd GEMMs).

use rlx_tensor::{DType, Device, Func, GraphScope, shape};
use std::time::Instant;

const M: usize = 4096;
const K: usize = 192;
const N: usize = 192;
const NE: usize = 256;
const D: usize = 4;
const R: usize = 8;

fn main() {
    let nb = K / D;
    let idx0: Vec<f64> = (0..N * nb).map(|i| ((i * 7 + 1) % NE) as f64).collect();
    let idx1: Vec<f64> = (0..N * nb).map(|i| ((i * 11 + 5) % NE) as f64).collect();
    let cb0: Vec<f32> = (0..NE * D)
        .map(|i| ((i * 17 + 3) % 31) as f32 / 31.0 - 0.5)
        .collect();
    let cb1: Vec<f32> = (0..NE * D)
        .map(|i| ((i * 13 + 9) % 29) as f32 / 29.0 - 0.5)
        .collect();
    let la: Vec<f32> = (0..K * R)
        .map(|i| ((i * 5 + 2) % 19) as f32 / 19.0 - 0.4)
        .collect();
    let lb: Vec<f32> = (0..N * R)
        .map(|i| ((i * 3 + 7) % 23) as f32 / 23.0 - 0.4)
        .collect();
    let x: Vec<f32> = (0..M * K)
        .map(|i| ((i * 13 + 7) % 23) as f32 / 23.0)
        .collect();

    let bind = |f: Func| {
        f.with_param("cb0", cb0.clone())
            .with_param("cb1", cb1.clone())
            .with_param("la", la.clone())
            .with_param("lb", lb.clone())
    };

    // (a) current: per-stage synth_matmul + LoRA, summed at the output.
    let g_a = {
        let mut s = GraphScope::new("per_stage");
        let x = s.input("x", shape![M, K]);
        let cb0 = s.param("cb0", shape![NE, D]);
        let cb1 = s.param("cb1", shape![NE, D]);
        let la = s.param("la", shape![K, R]);
        let lb = s.param("lb", shape![N, R]);
        let i0 = s.constant_nd(idx0.clone(), vec![N, nb], DType::U8);
        let i1 = s.constant_nd(idx1.clone(), vec![N, nb], DType::U8);
        let q0 = x.synth_matmul(&i0, &cb0, D as u32, NE as u32);
        let q1 = x.synth_matmul(&i1, &cb1, D as u32, NE as u32);
        let lora = x.matmul(&la).matmul_t(&lb);
        let q = &(&q0 + &q1) + &lora;
        let loss = q.mean_all();
        s.set_outputs([loss]);
        s.finish()
    };

    // (b) fold: W_eff = Σ reconstruct + A·Bᵀ (all weight-scale), then one x·W_eff.
    let g_b = {
        let mut s = GraphScope::new("weff_fold");
        let x = s.input("x", shape![M, K]);
        let cb0 = s.param("cb0", shape![NE, D]);
        let cb1 = s.param("cb1", shape![NE, D]);
        let la = s.param("la", shape![K, R]);
        let lb = s.param("lb", shape![N, R]);
        let i0 = s.constant_nd(idx0.clone(), vec![N, nb], DType::U8);
        let i1 = s.constant_nd(idx1.clone(), vec![N, nb], DType::U8);
        let w0 = i0.synth_reconstruct(&cb0, D as u32);
        let w1 = i1.synth_reconstruct(&cb1, D as u32);
        let wl = la.matmul_t(&lb); // A·Bᵀ = [k,n]
        let w_eff = &(&w0 + &w1) + &wl;
        let q = x.matmul(&w_eff);
        let loss = q.mean_all();
        s.set_outputs([loss]);
        s.finish()
    };

    let feed: &[(&str, &[f32])] = &[("x", &x)];
    let time = |g: rlx_tensor::Graph| -> f64 {
        let vg = bind(Func::from_graph(g)).value_and_grad_all();
        for _ in 0..5 {
            let _ = vg.run_on(Device::Metal, feed);
        }
        let n = 30;
        let t0 = Instant::now();
        for _ in 0..n {
            let _ = vg.run_on(Device::Metal, feed);
        }
        t0.elapsed().as_secs_f64() * 1e3 / n as f64
    };

    println!("projection {M}×{K}×{N}, stages=2 + LoRA(r={R}) — fwd+bwd on Metal\n");
    let a_ms = time(g_a);
    let b_ms = time(g_b);
    println!("  (a) per-stage synth_matmul + LoRA : {a_ms:.3} ms");
    println!("  (b) W_eff fold (1 fwd, 2 bwd GEMM): {b_ms:.3} ms");
    println!("  → fold speedup: {:.2}×", a_ms / b_ms);
}
