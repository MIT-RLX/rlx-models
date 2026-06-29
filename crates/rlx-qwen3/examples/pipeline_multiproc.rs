// RLX models — distributed inference.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! Multi-*process* pipeline-parallel inference (not threads).
//!
//! With no `--rank`, this is the **launcher**: [`LocalCluster`] picks
//! loopback ports, writes a `hosts.json`, and spawns N copies of *itself* as
//! separate OS processes (each a pipeline rank). It captures rank 0's output
//! and checks it against the single-node greedy reference. With `--rank`, it
//! is a **worker**: it forms the process group from the hostfile and runs its
//! block. This is the genuine deployment shape — separate address spaces,
//! real TCP sockets, the actual `DistConfig::connect` path — minus the wire.
//!
//! ```text
//! cargo run -p rlx-qwen3 --example pipeline_multiproc --release
//! cargo run -p rlx-qwen3 --example pipeline_multiproc --release --features mlx -- --device mlx
//! ```

#[path = "common/mod.rs"]
mod common;

use common::{Shape, argmax, qwen3, synth};
use rlx_core::weight_map::WeightMap;
use rlx_distributed::launch::worker_args;
use rlx_distributed::{DistConfig, LocalCluster, ParallelMode, PipelineCoordinator, WorkerArgs};
use rlx_qwen3::pipeline_decode::Qwen3PipelineDecodeStage;
use rlx_runtime::Device;
use std::str::FromStr;

const PROMPT: [u32; 5] = [1, 5, 3, 2, 7];
const N_GEN: usize = 6;
const WORLD: u32 = 3;

/// Read `--device NAME` from argv (defaults to CPU). Uses `Device`'s own
/// `FromStr`, so `cpu|metal|mlx|cuda|…` and aliases like `mps` all work.
fn device_arg() -> Device {
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        if a == "--device" {
            return it
                .next()
                .and_then(|v| Device::from_str(&v).ok())
                .unwrap_or(Device::Cpu);
        }
    }
    Device::Cpu
}

/// One pipeline rank: form the group from the hostfile, run its block, and
/// (on rank 0) print a machine-readable `RESULT` line for the launcher.
fn worker(w: WorkerArgs, device: Device) {
    let c = qwen3(Shape::TINY);
    let weights = synth(&c);
    let dist = DistConfig::load(&w.hostfile, Some(w.rank), ParallelMode::Pipeline).unwrap();
    let group = dist.connect().expect("connect");
    let mut stage = Qwen3PipelineDecodeStage::new(c, device, dist.rank, dist.world_size, weights)
        .with_decode_cache(64);
    let coord = PipelineCoordinator::new(group);
    let mut tokens = PROMPT.to_vec();
    let produced = coord
        .generate(&mut stage, &mut tokens, N_GEN, argmax, |_| false)
        .unwrap();
    if dist.rank == 0 {
        println!(
            "RESULT {}",
            produced
                .iter()
                .map(u32::to_string)
                .collect::<Vec<_>>()
                .join(",")
        );
    }
}

/// Top-level launcher: compute the single-node reference, fan out WORLD
/// worker processes via [`LocalCluster`], and compare.
fn launcher(device: Device) {
    // Single-node greedy reference on the SAME device as the workers, so the
    // comparison is device-consistent.
    let c = qwen3(Shape::TINY);
    let reference: Vec<u32> = {
        let mut wm = WeightMap::from_tensors(synth(&c));
        let mut g = rlx_qwen3::Qwen3Generator::from_loader(c.clone(), &mut wm, device).unwrap();
        g.prefill(&PROMPT);
        g.generate_cached(N_GEN, rlx_qwen3::SampleOpts::greedy())
            .unwrap()
    };

    // Fan out: each rank re-runs this binary with `--device <dev>` forwarded.
    let lines = LocalCluster::new(WORLD)
        .arg("--device")
        .arg(device.as_arg())
        .run()
        .expect("cluster");

    let got: Option<Vec<u32>> = lines.iter().rev().find_map(|l| {
        l.strip_prefix("RESULT ")
            .map(|rest| rest.split(',').filter_map(|s| s.parse().ok()).collect())
    });

    println!("reference (single-node): {reference:?}");
    match got {
        Some(g) if g == reference => {
            println!("multi-process rank 0  : {g:?}");
            println!("PASS — {WORLD} separate processes reproduced the single-node sequence");
        }
        Some(g) => {
            println!("multi-process rank 0  : {g:?}");
            eprintln!("FAIL — mismatch");
            std::process::exit(1);
        }
        None => {
            eprintln!("FAIL — no RESULT line captured from rank 0");
            std::process::exit(1);
        }
    }
}

fn main() {
    let device = device_arg();
    match worker_args() {
        Some(w) => worker(w, device),
        None => launcher(device),
    }
}
