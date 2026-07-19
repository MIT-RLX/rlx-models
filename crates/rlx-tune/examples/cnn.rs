// RLX models — fine-tuning.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! A small **CNN image classifier**, trained end-to-end through the crate's
//! data-parallel trainer — same [`Trainer`] / [`DpConfig`] / `--nnodes` launcher
//! as the LoRA examples, just a different forward graph.
//!
//! Architecture (all-convolutional, strided downsampling — no pooling op
//! needed): `conv3x3/s2 → relu → conv3x3/s2 → relu → flatten → linear →
//! softmax-cross-entropy`. Every op autodiffs and runs on the CPU backend the
//! trainer compiles to.
//!
//! The task is synthetic **template classification**: `num_classes` fixed random
//! 16×16 templates (identical on every rank); each sample is `template[c] +
//! noise` with label `c`. Each rank streams its own shard of fresh batches, so
//! the averaged gradient is the larger effective batch of data-parallel SGD.
//!
//! **Throughput** (the point of this example) is reported per step and in
//! aggregate as `samples/s`; add ranks / `--accum` / `--batch` and watch it scale.
//!
//! ```bash
//! cargo run --release -p rlx-tune --example cnn                       # single process
//! cargo run --release -p rlx-tune --example cnn -- --nnodes 4         # 4-way DP
//! cargo run --release -p rlx-tune --example cnn -- --nnodes 4 --shard --overlap --bf16 --accum 4
//! cargo run --release -p rlx-tune --example cnn -- --batch 256 --steps 300
//! ```

use rlx_ir::infer::GraphExt;
use rlx_ir::op::Activation;
use rlx_ir::{DType, Graph, Shape};
use rlx_tune::cluster::{Role, launch_or_join};
use rlx_tune::{DpConfig, ParamSlot, StepMetrics, Trainer};
use std::collections::HashMap;
use std::time::Instant;

// --- model dims (fixed) ----------------------------------------------------
const IMG: usize = 16; // 16×16 grayscale
const C1: usize = 8; // conv1 out channels
const C2: usize = 16; // conv2 out channels
const CLASSES: usize = 4;
const H2: usize = 4; // spatial size after two stride-2 convs: 16→8→4
const FLAT: usize = C2 * H2 * H2; // 16·4·4 = 256

/// Deterministic pseudo-random in [-0.1, 0.1]. Uses the high 24 bits of an LCG
/// (÷ 2²⁴ for a proper [0,1) — dividing by `u32::MAX` would collapse to ≈ 0).
fn pseudo(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(2654435761).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            ((s >> 8) as f32 / 16_777_216.0 - 0.5) * 0.2
        })
        .collect()
}

/// Kaiming-ish weight init: uniform scaled by `1/√fan_in`.
fn winit(n: usize, fan_in: usize, seed: u32) -> Vec<f32> {
    let s = 1.0 / (fan_in as f32).sqrt() / 0.1;
    pseudo(n, seed).into_iter().map(|v| v * s).collect()
}

/// Build the CNN forward graph for batch `nb`; returns it plus the trainable
/// param nodes `(conv1, conv2, fc, bias)`.
fn build_cnn(nb: usize) -> (Graph, [ParamSlot; 4]) {
    let f = DType::F32;
    let mut g = Graph::new("cnn");
    let x = g.input("x", Shape::new(&[nb, 1, IMG, IMG], f));
    let labels = g.input("labels", Shape::new(&[nb], f));

    let c1 = g.param("c1", Shape::new(&[C1, 1, 3, 3], f));
    let c2 = g.param("c2", Shape::new(&[C2, C1, 3, 3], f));
    let fc = g.param("fc", Shape::new(&[FLAT, CLASSES], f));
    let fb = g.param("fb", Shape::new(&[CLASSES], f));

    // conv3x3 stride2 pad1: 16→8→4.
    let h = g.conv2d(x, c1, [3, 3], [2, 2], [1, 1], [1, 1], 1);
    let h = g.activation(Activation::Relu, h, Shape::new(&[nb, C1, 8, 8], f));
    let h = g.conv2d(h, c2, [3, 3], [2, 2], [1, 1], [1, 1], 1);
    let h = g.activation(Activation::Relu, h, Shape::new(&[nb, C2, H2, H2], f));

    let flat = g.reshape_(h, vec![nb as i64, FLAT as i64]);
    let logits = g.matmul(flat, fc, Shape::new(&[nb, CLASSES], f));
    let logits = g.add(logits, fb); // bias broadcast [C] → [N, C]

    let ce = g.softmax_cross_entropy_with_logits(logits, labels); // [N]
    let loss = g.mean(ce, vec![0], false); // scalar
    g.set_outputs(vec![loss]);

    let wrt = [
        ParamSlot {
            name: "c1".into(),
            node: c1,
        },
        ParamSlot {
            name: "c2".into(),
            node: c2,
        },
        ParamSlot {
            name: "fc".into(),
            node: fc,
        },
        ParamSlot {
            name: "fb".into(),
            node: fb,
        },
    ];
    (g, wrt)
}

/// One batch of `nb` labelled images: `image = template[c] + noise`, label `c`.
/// Returns `(x [nb·1·16·16], labels [nb])`.
fn gen_batch(seed: u32, nb: usize, templates: &[Vec<f32>]) -> (Vec<f32>, Vec<f32>) {
    let px = IMG * IMG;
    let mut x = Vec::with_capacity(nb * px);
    let mut labels = Vec::with_capacity(nb);
    for j in 0..nb {
        let c = (seed as usize + j) % CLASSES;
        labels.push(c as f32);
        let noise = pseudo(px, seed.wrapping_mul(2654435761).wrapping_add(j as u32));
        for i in 0..px {
            x.push(templates[c][i] + 0.3 * noise[i]);
        }
    }
    (x, labels)
}

fn main() -> anyhow::Result<()> {
    // Fast CPU conv (im2col + BLAS) by default — ~10× over the naive reference
    // kernel, same result. Override with `RLX_FAST_CONV=0`.
    if std::env::var_os("RLX_FAST_CONV").is_none() {
        rlx_ir::env::set("RLX_FAST_CONV", "1");
    }
    let args: Vec<String> = std::env::args().collect();
    let flag = |n: &str| args.iter().any(|a| a == n);
    let val = |n: &str| {
        args.iter()
            .position(|a| a == n)
            .and_then(|i| args.get(i + 1)?.parse().ok())
    };

    let (rank, world, comm) = match launch_or_join()? {
        Role::Launcher => return Ok(()),
        Role::Worker { rank, world, comm } => (rank, world as usize, comm),
    };

    let batch: usize = val("--batch").unwrap_or(64);
    let steps: usize = val("--steps").unwrap_or(200);

    // Fixed class templates — identical on every rank (seed 42), so all ranks
    // learn the *same* task.
    let templates: Vec<Vec<f32>> = (0..CLASSES)
        .map(|c| winit(IMG * IMG, 1, 42 + c as u32))
        .collect();

    let (g, wrt) = build_cnn(batch);
    let mut params = HashMap::new();
    params.insert("c1".to_string(), winit(C1 * 9, 9, 1)); // C1·1·3·3
    params.insert("c2".to_string(), winit(C2 * C1 * 9, C1 * 9, 2));
    params.insert("fc".to_string(), winit(FLAT * CLASSES, FLAT, 3));
    params.insert("fb".to_string(), vec![0.0; CLASSES]);

    let mut cfg = DpConfig::new(3e-3).log_every(25);
    if flag("--shard") {
        cfg = cfg.shard();
    }
    if flag("--overlap") {
        cfg = cfg.overlap();
    }
    if flag("--bf16") {
        cfg = cfg.bf16();
    }
    if let Some(a) = val("--accum") {
        cfg = cfg.grad_accum(a);
    }
    let ga = cfg.grad_accum.max(1);
    if rank == 0 {
        eprintln!(
            "CNN {IMG}×{IMG}→conv{C1}→conv{C2}→fc{CLASSES} | {} params | batch {batch} × accum {ga} × {world} ranks | {}",
            C1 * 9 + C2 * C1 * 9 + FLAT * CLASSES + CLASSES,
            cfg.describe(),
        );
    }

    // Each rank streams its own shard: seed varies by (rank, step, micro).
    let templates_p = templates.clone();
    let next_batch = move |step: usize, micro: usize| {
        let seed = (rank as usize * 1_000_003 + step * 131 + micro * 977) as u32;
        let (x, labels) = gen_batch(seed, batch, &templates_p);
        vec![("x".to_string(), x), ("labels".to_string(), labels)]
    };

    let on_step = |m: &StepMetrics| {
        if rank == 0 {
            let sps = (batch * ga * world) as f64 / (m.step_ms / 1e3);
            eprintln!("{m} | {sps:>7.0} samples/s");
        }
    };

    let mut trainer = Trainer::new(g, &wrt, &params, steps, comm.as_deref(), &cfg)?;
    let t0 = Instant::now();
    let losses = trainer.run(next_batch, on_step)?;
    let wall = t0.elapsed().as_secs_f64();

    if rank == 0 {
        let total = batch * ga * world * steps;
        println!(
            "done: loss {:.4} → {:.4} | {:.0} samples/s ({total} samples over {world} rank(s) in {wall:.2}s)",
            losses.first().copied().unwrap_or(f32::NAN),
            losses.last().copied().unwrap_or(f32::NAN),
            total as f64 / wall,
        );
    }
    Ok(())
}
