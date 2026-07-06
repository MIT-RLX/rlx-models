// RLX models — distributed inference.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! End-to-end pipeline-parallel timing on one machine (loopback TCP).
//!
//! Times greedy generation of a few tokens for a moderate synthetic Qwen3
//! on CPU: single-node (monolithic) vs a 2- and 4-rank pipeline. Same
//! machine, so this measures *overhead*, not speedup — pipeline
//! parallelism's win is fitting a bigger model across more machines'
//! memory, which can't show as throughput on one box.
//!
//! Run: `cargo run -p rlx-qwen3 --example pipeline_bench --release`

#[path = "common/mod.rs"]
mod common;

use common::{Shape, Tensors as Weights, argmax, qwen3, synth};
use rlx_distributed::{NetTransport, PipelineCoordinator, ProcessGroup};
use rlx_qwen3::Qwen3Config;
use rlx_qwen3::pipeline::Qwen3PipelineStage;
use rlx_runtime::{Device, Session};
use std::net::{Ipv4Addr, SocketAddr, TcpListener};
use std::sync::Arc;
use std::thread;
use std::time::Instant;

/// Single-node greedy: recompute the full prefill graph each step.
fn run_monolithic(c: &Qwen3Config, w: &Weights, prompt: &[u32], n: usize) -> (Vec<u32>, f64) {
    use rlx_core::weight_map::WeightMap;
    let opts = rlx_core::flow_bridge::compile_options_from_profile(
        &rlx_flow::CompileProfile::qwen3_prefill(),
        Device::Cpu,
        rlx_ir::logical_kernel::KernelDispatchConfig::default(),
    );
    let mut tokens = prompt.to_vec();
    let mut out = Vec::new();
    let t0 = Instant::now();
    for _ in 0..n {
        let mut wm = WeightMap::from_tensors(w.clone());
        let (g, p) =
            rlx_qwen3::build_qwen3_graph_sized_last_logits(c, &mut wm, 1, tokens.len(), false)
                .unwrap();
        let mut compiled = Session::new(Device::Cpu).compile_with(g, &opts);
        for (n, d) in &p {
            compiled.set_param(n, d);
        }
        let ids: Vec<f32> = tokens.iter().map(|&t| t as f32).collect();
        let logits = compiled
            .run(&[("input_ids", ids.as_slice())])
            .into_iter()
            .next()
            .unwrap();
        let tok = argmax(&logits);
        tokens.push(tok);
        out.push(tok);
    }
    (out, t0.elapsed().as_secs_f64())
}

/// Pipeline greedy over loopback TCP, `world` ranks. Returns rank-0's
/// generated tokens and the wall-clock seconds for the generate loop
/// (connection setup excluded via a pre-barrier).
fn run_pipeline(
    c: &Qwen3Config,
    w: &Weights,
    prompt: &[u32],
    n: usize,
    world: u32,
) -> (Vec<u32>, f64) {
    let listeners: Vec<TcpListener> = (0..world)
        .map(|_| TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap())
        .collect();
    let addrs: Vec<SocketAddr> = listeners.iter().map(|l| l.local_addr().unwrap()).collect();
    let c = Arc::new(c.clone());
    let w = Arc::new(w.clone());
    let prompt = Arc::new(prompt.to_vec());

    let handles: Vec<_> = listeners
        .into_iter()
        .enumerate()
        .map(|(rank, listener)| {
            let (addrs, c, w, prompt) = (addrs.clone(), c.clone(), w.clone(), prompt.clone());
            thread::spawn(move || {
                let t = NetTransport::from_listener(rank as u32, world, listener, addrs, 8 << 20)
                    .unwrap();
                let mut stage = Qwen3PipelineStage::new(
                    (*c).clone(),
                    Device::Cpu,
                    rank as u32,
                    world,
                    (*w).clone(),
                );
                let coord = PipelineCoordinator::new(ProcessGroup::new(Arc::new(t)));
                coord.group().barrier().unwrap(); // exclude setup
                let t0 = Instant::now();
                let mut tokens = (*prompt).clone();
                let toks = coord
                    .generate(&mut stage, &mut tokens, n, argmax, |_| false)
                    .unwrap();
                (toks, t0.elapsed().as_secs_f64())
            })
        })
        .collect();
    let mut leader = (Vec::new(), 0.0);
    for (rank, h) in handles.into_iter().enumerate() {
        let r = h.join().unwrap();
        if rank == 0 {
            leader = r;
        }
    }
    leader
}

fn main() {
    let c = qwen3(Shape::MEDIUM);
    let w = synth(&c);
    let prompt = vec![1u32, 7, 3, 9, 2, 5, 4, 8];
    let n = 4usize;

    println!(
        "model: d={} layers={} heads={}/{} ffn={} vocab={} | prompt {} tok, generate {}",
        c.hidden_size,
        c.num_hidden_layers,
        c.num_attention_heads,
        c.num_key_value_heads,
        c.intermediate_size,
        c.vocab_size,
        prompt.len(),
        n
    );
    println!("(single machine, CPU, loopback TCP — measures overhead, not speedup)\n");

    let (ref_tok, t_mono) = run_monolithic(&c, &w, &prompt, n);
    println!(
        "single-node : {:7.1} ms total  ({:6.1} ms/token)   tokens={:?}",
        t_mono * 1e3,
        t_mono * 1e3 / n as f64,
        ref_tok
    );

    for world in [2u32, 4] {
        let (toks, t) = run_pipeline(&c, &w, &prompt, n, world);
        let ok = if toks == ref_tok { "==ref" } else { "!!DIFF" };
        println!(
            "pipeline x{world}: {:7.1} ms total  ({:6.1} ms/token)   {ok}",
            t * 1e3,
            t * 1e3 / n as f64
        );
    }

    println!(
        "\nnote: each step recompiles its graph (no per-block compile cache yet) — \n\
         that dominates per-token time; the cross-rank transport is ~tens of µs (see transport_bench)."
    );
}
