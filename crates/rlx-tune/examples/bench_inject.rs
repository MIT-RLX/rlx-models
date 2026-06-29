// RLX models — fine-tuning.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! Benchmark: fused (`LoraMatMul`) vs unfused (explicit matmuls) LoRA-injected
//! forward latency.
//!
//! Run: `cargo run -p rlx-tune --example bench_inject --release`

use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{CompileOptions, Device, Session};
use rlx_tune::{FuseMode, LoraSpec, inject_lora};
use std::time::Instant;

fn pseudo(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(2654435761).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            ((s >> 8) as f32 / u32::MAX as f32 - 0.5) * 0.1
        })
        .collect()
}

fn build_base(batch: usize, k: usize, n: usize) -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("bench");
    let x = g.input("x", Shape::new(&[batch, k], f));
    let w = g.param("W", Shape::new(&[k, n], f));
    let y = g.matmul(x, w, Shape::new(&[batch, n], f));
    g.set_outputs(vec![y]);
    g
}

/// Mean ms/iter of the LoRA-injected forward in the given fuse mode.
fn bench(mode: FuseMode, batch: usize, k: usize, n: usize, r: usize, iters: usize) -> f64 {
    let spec = LoraSpec::new(r, r as f32, vec!["W".into()]);
    let (graph, _) = inject_lora(&build_base(batch, k, n), &spec, mode);
    let mut c = Session::new(Device::Cpu).compile_with(graph, &CompileOptions::new());
    c.set_param("W", &pseudo(k * n, 1));
    c.set_param("lora.W.a", &pseudo(k * r, 2));
    c.set_param("lora.W.b", &pseudo(r * n, 3));
    let xd = pseudo(batch * k, 4);

    for _ in 0..5 {
        let _ = c.run(&[("x", &xd)]);
    }
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = c.run(&[("x", &xd)]);
    }
    t0.elapsed().as_secs_f64() * 1e3 / iters as f64
}

fn main() {
    let iters = 50usize;
    println!("LoRA-injected forward — fused (LoraMatMul) vs unfused (CPU, {iters} iters)\n");
    println!(
        "{:>6} {:>6} {:>5} {:>5} | {:>11} {:>11} {:>8}",
        "batch", "k", "n", "rank", "unfused ms", "fused ms", "speedup"
    );
    for &(batch, k, n, r) in &[
        (64usize, 1024usize, 1024usize, 8usize),
        (128, 2048, 2048, 16),
        (256, 2048, 2048, 16),
        (256, 4096, 4096, 32),
    ] {
        let unfused = bench(FuseMode::Unfused, batch, k, n, r, iters);
        let fused = bench(FuseMode::Fused, batch, k, n, r, iters);
        println!(
            "{batch:>6} {k:>6} {n:>5} {r:>5} | {unfused:>11.3} {fused:>11.3} {:>7.2}x",
            unfused / fused
        );
    }
    println!(
        "\nFused emits one `LoraMatMul` op (fused kernel on CPU/Metal/MLX); unfused emits\n\
         base + (x·A)·B matmuls + add. Both differentiate (autodiff unfuses the fused op)."
    );
}
