// RLX models — fine-tuning.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! The minimal data-parallel fine-tune — the zero-config happy path.
//!
//! Runs single-process as-is. Set `RANK`/`WORLD`/`PEERS` to go N-way data
//! parallel (torchrun style) with **no code change** — `from_env` returns the
//! collective and `train_dp` averages gradients across ranks:
//!
//! ```bash
//! cargo run -p rlx-tune --example data_parallel_min           # single process
//! RANK=0 WORLD=2 PEERS=127.0.0.1:29500,127.0.0.1:29501 \
//!     cargo run -p rlx-tune --example data_parallel_min &     # rank 0
//! RANK=1 WORLD=2 PEERS=127.0.0.1:29500,127.0.0.1:29501 \
//!     cargo run -p rlx-tune --example data_parallel_min       # rank 1
//! ```
//!
//! For a one-command launcher (no env vars) and every training knob, see the
//! `data_parallel` example.

use rlx_ir::infer::GraphExt;
use rlx_ir::{DType, Graph, Shape};
use rlx_tune::{DpConfig, ParamSlot, lora_linear, train_dp};
use std::collections::HashMap;

/// Deterministic pseudo-random values in [-0.1, 0.1].
fn pseudo(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(2654435761).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            ((s >> 8) as f32 / 16_777_216.0 - 0.5) * 0.2 // ÷2²⁴ for a proper [0,1)
        })
        .collect()
}

fn main() -> anyhow::Result<()> {
    // Zero-config: `Some(..)` only when WORLD > 1, else `None` (single process,
    // no sockets). The caller code below is identical either way.
    let comm = rlx_tune::from_env()?;

    // A tiny LoRA fit: y = (x·A)·B → target x·M, frozen zero base.
    let (m, k, n, r) = (4, 4, 3, 4);
    let f = DType::F32;
    let mut g = Graph::new("min");
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

    // Data: target = x · M_true.
    let xd = pseudo(m * k, 1);
    let m_true = pseudo(k * n, 2);
    let mut td = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            td[i * n + j] = (0..k).map(|l| xd[i * k + l] * m_true[l * n + j]).sum();
        }
    }

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
    let inputs = vec![("x".to_string(), xd), ("t".to_string(), td)];

    // One config, one call. `StepMetrics: Display` handles the progress line.
    let cfg = DpConfig::new(0.05).log_every(50);
    let losses = train_dp(
        g,
        &wrt,
        &mut params,
        &inputs,
        200,
        comm.as_deref(),
        &cfg,
        |m| {
            println!("{m}");
        },
    )?;

    println!("loss {:.6} -> {:.6}", losses[0], losses.last().unwrap());
    Ok(())
}
