// RLX models — distributed inference.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! Decode throughput: recompile-each-step vs bucketed compile cache.
//!
//! Generates tokens through the KV-cached pipeline (`Qwen3PipelineDecodeStage`)
//! with the per-block bucketed compile cache OFF then ON, and reports
//! per-token latency. The cache compiles O(log N) graphs over a generation
//! instead of one per step.
//!
//! Run: `cargo run -p rlx-qwen3 --example decode_bench --release`

#[path = "common/mod.rs"]
mod common;

use common::{Shape, Tensors, argmax, qwen3, synth};
use rlx_distributed::{NetTransport, PipelineCoordinator, ProcessGroup};
use rlx_qwen3::Qwen3Config;
use rlx_qwen3::pipeline_decode::Qwen3PipelineDecodeStage;
use rlx_runtime::Device;
use std::net::{Ipv4Addr, SocketAddr, TcpListener};
use std::sync::Arc;
use std::thread;
use std::time::Instant;

/// Generate `n` tokens through a `world`-rank decode pipeline; return
/// rank-0's per-token time (s). `cached` toggles the bucketed compile cache.
fn run(world: u32, cached: bool, cfg: &Qwen3Config, w: &Tensors, prompt: &[u32], n: usize) -> f64 {
    let listeners: Vec<TcpListener> = (0..world)
        .map(|_| TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap())
        .collect();
    let addrs: Vec<SocketAddr> = listeners.iter().map(|l| l.local_addr().unwrap()).collect();
    let cfg = Arc::new(cfg.clone());
    let w = Arc::new(w.clone());
    let prompt = Arc::new(prompt.to_vec());

    let handles: Vec<_> = listeners
        .into_iter()
        .enumerate()
        .map(|(rank, listener)| {
            let (addrs, cfg, w, prompt) = (addrs.clone(), cfg.clone(), w.clone(), prompt.clone());
            thread::spawn(move || {
                let t = NetTransport::from_listener(rank as u32, world, listener, addrs, 4 << 20)
                    .unwrap();
                let stage = Qwen3PipelineDecodeStage::new(
                    (*cfg).clone(),
                    Device::Cpu,
                    rank as u32,
                    world,
                    (*w).clone(),
                );
                let mut stage = if cached {
                    stage.with_decode_cache(128)
                } else {
                    stage
                };
                let coord = PipelineCoordinator::new(ProcessGroup::new(Arc::new(t)));
                let mut tokens = (*prompt).clone();
                let t0 = Instant::now();
                coord
                    .generate(&mut stage, &mut tokens, n, |l| argmax(l), |_| false)
                    .unwrap();
                t0.elapsed().as_secs_f64()
            })
        })
        .collect();
    let mut r0 = 0.0;
    for (rank, h) in handles.into_iter().enumerate() {
        let v = h.join().unwrap();
        if rank == 0 {
            r0 = v;
        }
    }
    r0 / n as f64
}

fn main() {
    let c = qwen3(Shape::SMALL);
    let w = synth(&c);
    let prompt = vec![1u32, 7, 3, 9, 2, 5, 4, 8];
    let n = 24usize;

    println!(
        "model d={} layers={} ffn={} vocab={} | prompt {} tok, decode {} tokens (CPU, loopback TCP)\n",
        c.hidden_size,
        c.num_hidden_layers,
        c.intermediate_size,
        c.vocab_size,
        prompt.len(),
        n
    );

    for world in [1u32, 2] {
        let off = run(world, false, &c, &w, &prompt, n) * 1e3;
        let on = run(world, true, &c, &w, &prompt, n) * 1e3;
        println!(
            "world={world}:  recompile {off:7.1} ms/tok   |   bucketed-cache {on:7.1} ms/tok   ({:.1}x faster)",
            off / on
        );
    }

    println!(
        "\nThe cache compiles O(log N) decode graphs over the run instead of one\n\
         per token; the gap is the per-step compile that recompile pays every token."
    );
}
