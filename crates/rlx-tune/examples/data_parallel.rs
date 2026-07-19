// RLX models — fine-tuning.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! Data-parallel LoRA fit, driven by the `--nnodes` self-spawning launcher.
//! Each rank trains on its own shard of a shared synthetic problem; because the
//! gradients are averaged, the result matches single-process training on the
//! union of the shards.
//!
//! ```bash
//! cargo run -p rlx-tune --example data_parallel                    # single process
//! cargo run -p rlx-tune --example data_parallel -- --nnodes 3      # 3-way DP, one command
//! # every knob (all optional, all compose):
//! cargo run -p rlx-tune --example data_parallel -- --nnodes 4 \
//!     --shard --overlap --bf16 --clip --warmup --cosine --accum 2
//! # checkpoint / resume (run once to save, again to resume):
//! cargo run -p rlx-tune --example data_parallel -- --nnodes 3 --shard --ckpt run.ckpt
//! ```
//!
//! See [`rlx_tune::DpConfig`] for the builder used below and [`rlx_tune::Trainer`]
//! for the checkpoint/resume API.

use rlx_ir::infer::GraphExt;
use rlx_ir::{DType, Graph, Shape};
use rlx_tune::cluster::{Role, launch_or_join};
use rlx_tune::{Checkpoint, DpConfig, ParamSlot, StepMetrics, Trainer, lora_linear};
use std::collections::HashMap;

fn pseudo(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(2654435761).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            ((s >> 8) as f32 / 16_777_216.0 - 0.5) * 0.2 // ÷2²⁴ for a proper [0,1)
        })
        .collect()
}

fn host_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0;
            for t in 0..k {
                acc += a[i * k + t] * b[t * n + j];
            }
            out[i * n + j] = acc;
        }
    }
    out
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let flag = |name: &str| args.iter().any(|a| a == name);
    let val = |name: &str| -> Option<usize> {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .and_then(|v| v.parse().ok())
    };

    let (rank, world, comm) = match launch_or_join()? {
        Role::Launcher => return Ok(()), // parent: workers spawned + awaited
        Role::Worker { rank, world, comm } => (rank, world, comm),
    };

    // A tiny LoRA fit: y = (x·A)·B → target x·M_true, frozen zero base.
    let (rows, k, n, r) = (8usize, 4usize, 3usize, 4usize);
    let f = DType::F32;
    let x_full = pseudo(rows * k, 1);
    let m_true = pseudo(k * n, 2);
    let t_full = host_matmul(&x_full, &m_true, rows, k, n);

    // This rank's shard of the rows.
    let per = rows / world as usize;
    let (r0, r1) = (rank as usize * per, (rank as usize + 1) * per);
    let x_shard = x_full[r0 * k..r1 * k].to_vec();
    let t_shard = t_full[r0 * n..r1 * n].to_vec();
    let m = r1 - r0;

    let mut g = Graph::new("dp_example");
    let x = g.input("x", Shape::new(&[m, k], f));
    let w = g.param("w", Shape::new(&[k, n], f));
    let a = g.param("a", Shape::new(&[k, r], f));
    let b = g.param("b", Shape::new(&[r, n], f));
    let y = lora_linear(&mut g, x, w, a, b, m, n, r);
    let t = g.input("t", Shape::new(&[m, n], f));
    let diff = g.sub(y, t);
    let sq = g.mul(diff, diff);
    let flat = g.reshape_(sq, vec![(m * n) as i64]);
    let loss = g.mean(flat, vec![0], false);
    g.set_outputs(vec![loss]);

    let mut params = HashMap::new();
    params.insert("w".to_string(), vec![0.0; k * n]);
    params.insert("a".to_string(), pseudo(k * r, 3));
    params.insert("b".to_string(), pseudo(r * n, 4));
    let wrt = vec![
        ParamSlot {
            name: "a".into(),
            node: a,
        },
        ParamSlot {
            name: "b".into(),
            node: b,
        },
    ];
    let inputs = vec![("x".to_string(), x_shard), ("t".to_string(), t_shard)];

    // Config from CLI flags — the fluent builder reads like the flags.
    let mut cfg = DpConfig::new(0.05).log_every(50);
    if flag("--shard") {
        cfg = cfg.shard();
    }
    if flag("--overlap") {
        cfg = cfg.overlap();
    }
    if flag("--bf16") {
        cfg = cfg.bf16();
    }
    if flag("--clip") {
        cfg = cfg.clip(0.02);
    }
    if flag("--warmup") {
        cfg = cfg.warmup(20);
    }
    if flag("--cosine") {
        cfg = cfg.cosine(0.1);
    }
    if let Some(g) = val("--accum") {
        cfg = cfg.grad_accum(g);
    }
    if rank == 0 {
        eprintln!("config: {}", cfg.describe());
    }

    // `StepMetrics: Display` gives the standard progress row for free.
    let on_step = |m: &StepMetrics| {
        if rank == 0 {
            eprintln!("{m}");
        }
    };

    // Drive the loop via a `Trainer` so we can checkpoint / resume. `--ckpt
    // <path>` resumes from the file if it exists, then saves at the end.
    let ckpt = args
        .iter()
        .position(|a| a == "--ckpt")
        .and_then(|i| args.get(i + 1))
        .cloned();
    let mut trainer = Trainer::new(g, &wrt, &params, 300, comm.as_deref(), &cfg)?;
    if let Some(path) = &ckpt {
        if std::path::Path::new(path).exists() {
            trainer.restore(&Checkpoint::load(path)?);
            if rank == 0 {
                eprintln!("resumed from {path} at step {}", trainer.step_index());
            }
        }
    }

    let losses = trainer.run(|_, _| inputs.clone(), on_step)?;

    if let Some(path) = &ckpt {
        let ck = trainer.checkpoint(); // collective when sharded — all ranks
        if rank == 0 {
            ck.save(path)?;
            eprintln!("saved checkpoint to {path} (step {})", ck.step);
        }
    }

    if rank == 0 {
        match (losses.first(), losses.last()) {
            (Some(&first), Some(&last)) => println!(
                "rank 0: loss {first:.6} -> {last:.6} over {} step(s) run ({} rank(s), shard={}, overlap={}, {:?})",
                losses.len(),
                world,
                cfg.shard_optimizer,
                cfg.overlap,
                cfg.reduce_dtype,
            ),
            _ => println!(
                "rank 0: nothing to do — already at step {} of 300 (resume complete)",
                trainer.step_index()
            ),
        }
    }
    Ok(())
}
