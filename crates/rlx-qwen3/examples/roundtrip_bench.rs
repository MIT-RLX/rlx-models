// RLX models — distributed inference.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! Round-trip latency of a *split* tiny model over loopback TCP.
//!
//! A tiny Qwen3 (fits trivially in RAM) is intentionally split across N
//! pipeline ranks. Each rank's block graph is **compiled once** up front,
//! then we time the steady-state per-token relay (compute + send/recv +
//! token broadcast). Because the model is tiny, compute is small, so the
//! gap between 1 rank (no link) and N ranks isolates the cost the link
//! round-trips add inside a real compute loop.
//!
//! Run: `cargo run -p rlx-qwen3 --example roundtrip_bench --release`

#[path = "common/mod.rs"]
mod common;

use common::{Shape, Tensors, argmax, qwen3, synth};
use rlx_core::flow_bridge::compile_options_from_profile;
use rlx_core::weight_map::WeightMap;
use rlx_distributed::{NetTransport, ProcessGroup, block_role, pipeline_layer_range};
use rlx_flow::CompileProfile;
use rlx_ir::logical_kernel::KernelDispatchConfig;
use rlx_qwen3::Qwen3Config;
use rlx_qwen3::pipeline::{BlockSpec, block_weight_filter, build_qwen3_block_graph};
use rlx_runtime::{Device, Session};
use std::net::{Ipv4Addr, SocketAddr, TcpListener};
use std::sync::Arc;
use std::thread;
use std::time::Instant;

const TAG: u32 = 100;

/// The roundtrip bench's bespoke shape (sweeps the layer count).
fn tiny_cfg(layers: usize) -> Qwen3Config {
    qwen3(Shape {
        vocab: 256,
        hidden: 128,
        intermediate: 256,
        layers,
        heads: 4,
        kv_heads: 2,
        head_dim: 32,
        max_pos: 64,
    })
}

/// One rank: compile its block once, then time `iters` steady-state relays.
/// Returns the per-token latency (s) measured on this rank.
fn run_rank(
    rank: u32,
    world: u32,
    cfg: &Qwen3Config,
    weights: &Tensors,
    prompt: &[u32],
    warmup: usize,
    iters: usize,
    group: ProcessGroup,
) -> f64 {
    let range = pipeline_layer_range(cfg.num_hidden_layers, rank, world);
    let role = block_role(rank, world);
    let spec = BlockSpec::for_role(role, range);
    let embed = spec.embed_input;
    let logits = spec.produce_logits;

    // Filter to this block's weights, build + compile ONCE.
    let mut wfilt = Tensors::new();
    for (k, v) in weights {
        if block_weight_filter(k, cfg, &spec) {
            wfilt.insert(k.clone(), v.clone());
        }
    }
    let seq = prompt.len();
    let mut wm = WeightMap::from_tensors(wfilt);
    let (graph, params) = build_qwen3_block_graph(cfg, &mut wm, 1, seq, &spec).unwrap();
    let opts = compile_options_from_profile(
        &CompileProfile::qwen3_prefill(),
        Device::Cpu,
        KernelDispatchConfig::default(),
    );
    let mut compiled = Session::new(Device::Cpu).compile_with(graph, &opts);
    for (n, d) in &params {
        compiled.set_param(n, d);
    }
    let ids: Vec<f32> = prompt.iter().map(|&t| t as f32).collect();

    let one = |compiled: &mut rlx_runtime::CompiledGraph| {
        let out = if embed {
            compiled.run(&[("input_ids", ids.as_slice())])
        } else {
            let h = group.recv_f32(rank + 1, TAG).unwrap();
            compiled.run(&[("hidden_states", h.as_slice())])
        };
        let out0 = out.into_iter().next().unwrap();
        if !logits {
            group.send_f32(rank - 1, TAG, &out0).unwrap();
        }
        let mut tok = [if logits { argmax(&out0) as f32 } else { 0.0 }];
        group.broadcast(0, &mut tok).unwrap();
    };

    group.barrier().unwrap();
    for _ in 0..warmup {
        one(&mut compiled);
    }
    group.barrier().unwrap();
    let t0 = Instant::now();
    for _ in 0..iters {
        one(&mut compiled);
    }
    let dt = t0.elapsed().as_secs_f64();
    group.barrier().unwrap();
    dt / iters as f64
}

/// Spin up `world` loopback ranks and return rank-0's per-token latency (s).
fn measure(world: u32, cfg: &Qwen3Config, weights: &Tensors, prompt: &[u32]) -> f64 {
    let warmup = 30usize;
    let iters = 400usize;
    let listeners: Vec<TcpListener> = (0..world)
        .map(|_| TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap())
        .collect();
    let addrs: Vec<SocketAddr> = listeners.iter().map(|l| l.local_addr().unwrap()).collect();
    let cfg = Arc::new(cfg.clone());
    let weights = Arc::new(weights.clone());
    let prompt = Arc::new(prompt.to_vec());

    let handles: Vec<_> = listeners
        .into_iter()
        .enumerate()
        .map(|(rank, listener)| {
            let (addrs, cfg, weights, prompt) =
                (addrs.clone(), cfg.clone(), weights.clone(), prompt.clone());
            thread::spawn(move || {
                let t = NetTransport::from_listener(rank as u32, world, listener, addrs, 1 << 20)
                    .unwrap();
                let g = ProcessGroup::new(Arc::new(t));
                run_rank(
                    rank as u32,
                    world,
                    &cfg,
                    &weights,
                    &prompt,
                    warmup,
                    iters,
                    g,
                )
            })
        })
        .collect();
    let mut rank0 = 0.0;
    for (rank, h) in handles.into_iter().enumerate() {
        let v = h.join().unwrap();
        if rank == 0 {
            rank0 = v;
        }
    }
    rank0
}

fn main() {
    let layers = 8usize;
    let cfg = tiny_cfg(layers);
    let weights = synth(&cfg);
    let prompt = vec![1u32, 5, 3, 2];
    let seq = prompt.len();
    let hop_bytes = seq * cfg.hidden_size * 4;

    println!(
        "tiny model: d={} layers={} vocab={} | prompt seq={} -> hidden hop = {} bytes",
        cfg.hidden_size, layers, cfg.vocab_size, seq, hop_bytes
    );
    println!("graphs compiled once per rank; steady-state per-token latency, loopback TCP\n");

    let base = measure(1, &cfg, &weights, &prompt);
    println!(
        "world=1 (no link)      : {:8.1} µs/token   (pure compute baseline)",
        base * 1e6
    );
    for world in [2u32, 4, 8] {
        let lat = measure(world, &cfg, &weights, &prompt);
        let overhead = (lat - base).max(0.0);
        let hops = (world - 1) as f64; // hidden-state hops toward rank 0
        println!(
            "world={world} ({hops_n} hops + bcast): {:8.1} µs/token   (+{:6.1} µs link, ~{:5.1} µs/round-trip)",
            lat * 1e6,
            overhead * 1e6,
            overhead * 1e6 / hops,
            hops_n = world - 1,
        );
    }

    println!(
        "\nEach extra rank inserts one compute→send→recv hop; the per-round-trip\n\
         figure is the link cost added to the token's critical path on this box\n\
         (loopback ~14 µs one-way — a real Thunderbolt/Ethernet link adds its own)."
    );
}
