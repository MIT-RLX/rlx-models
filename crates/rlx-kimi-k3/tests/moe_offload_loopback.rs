//! Capstone: the FULL disaggregated MoE path (orchestrator `run_moe_offload` +
//! worker `serve_expert_worker`/`KimiExpertProvider`, over a real 2-rank
//! `TcpTransport`) must match the LOCAL `run_moe_paged` reference on a real layer.
//! Validates phase1(router+down) → dispatch → gather+sum → tail(norm+up+shared)
//! end-to-end. Skips if the checkpoint isn't mounted.
#![cfg(feature = "cluster")]

use rlx_distributed::{
    ExpertShards, TcpTransport, free_loopback_ports, serve_expert_worker, shutdown_expert_workers,
};
use rlx_ir::Philox4x32;
use rlx_kimi_k3::dist_experts::{KimiExpertProvider, run_moe_offload};
use rlx_kimi_k3::moe::MoeDims;
use rlx_kimi_k3::runner::run_moe_paged;
use rlx_runtime::Device;
use std::collections::HashSet;
use std::net::{Ipv4Addr, SocketAddr};
use std::path::Path;

const CKPT: &str = "/Volumes/FOUR/kimi";

fn dims() -> MoeDims {
    MoeDims {
        hidden: 7168,
        latent: 3584,
        moe_inter: 3072,
        num_experts: 896,
        top_k: 8,
        num_shared: 2,
        routed_scaling: 2.5,
        eps: 1e-5,
        situ_beta: 4.0,
        situ_linear_beta: Some(25.0),
        batch: 1,
        seq: 1,
    }
}

#[test]
fn disaggregated_moe_matches_local() {
    if !Path::new(CKPT).join("config.json").exists() {
        eprintln!("skip: {CKPT} not mounted");
        return;
    }
    const HEAP: usize = 64 * 1024 * 1024;
    let d = dims();
    let layer: u32 = 1; // first MoE layer
    let (rows, hidden) = (d.batch * d.seq, d.hidden);

    let mut rng = Philox4x32::new(0xB0B);
    let mut h = vec![0f32; rows * hidden];
    rng.fill_normal(&mut h);

    // local reference.
    let mut ck_ref = rlx_kimi_k3::loader::CheckpointLoader::open(CKPT).unwrap();
    let reference = run_moe_paged(
        &mut ck_ref,
        &format!("language_model.model.layers.{layer}"),
        &h,
        d,
        Device::Cpu,
    )
    .unwrap();

    // 2-rank loopback: rank 1 = worker owning ALL experts.
    let ports = free_loopback_ports(2).unwrap();
    let peers: Vec<SocketAddr> = ports
        .iter()
        .map(|&p| SocketAddr::from((Ipv4Addr::LOCALHOST, p)))
        .collect();
    let peers_w = peers.clone();
    let worker = std::thread::spawn(move || {
        let t = TcpTransport::bind(1, 2, peers_w, HEAP).unwrap();
        let all: HashSet<usize> = (0..896).collect();
        let mut provider = KimiExpertProvider::open(CKPT, dims(), all, Device::Cpu).unwrap();
        serve_expert_worker(&t, 0, &mut provider).unwrap();
    });

    // rank 0 = orchestrator.
    let t = TcpTransport::bind(0, 2, peers, HEAP).unwrap();
    let shards = ExpertShards::round_robin(d.num_experts, &[1]); // all experts → worker rank 1
    let mut ck = rlx_kimi_k3::loader::CheckpointLoader::open(CKPT).unwrap();
    let offload = run_moe_offload(&mut ck, &t, &shards, layer, &h, d, Device::Cpu, None).unwrap();

    shutdown_expert_workers(&t, &[1]).unwrap();
    worker.join().unwrap();

    assert_eq!(offload.len(), reference.len());
    let (mut max_abs, mut sum_sq_ref) = (0f32, 0f64);
    for (o, r) in offload.iter().zip(&reference) {
        max_abs = max_abs.max((o - r).abs());
        sum_sq_ref += (*r as f64) * (*r as f64);
    }
    let rel = (offload
        .iter()
        .zip(&reference)
        .map(|(o, r)| ((o - r) as f64).powi(2))
        .sum::<f64>()
        / sum_sq_ref.max(1e-30))
    .sqrt();
    eprintln!("disaggregated vs local run_moe_paged: max|Δ| {max_abs:.3e}, rel-L2 {rel:.3e}");
    assert!(rel < 1e-3, "disaggregated MoE != local (rel-L2 {rel:.3e})");
}

/// Local-overflow fallback: 1 worker owns experts `[0,448)`, the orchestrator owns
/// the `[448,896)` OVERFLOW locally (LOCAL sentinel → not dispatched). Their sum must
/// still equal the local reference — validating `run_moe_offload`'s `local` path.
#[test]
fn disaggregated_with_local_overflow_matches_local() {
    if !Path::new(CKPT).join("config.json").exists() {
        eprintln!("skip: {CKPT} not mounted");
        return;
    }
    const HEAP: usize = 64 * 1024 * 1024;
    let d = dims();
    let layer: u32 = 1;
    let (rows, hidden) = (d.batch * d.seq, d.hidden);
    let mut rng = Philox4x32::new(0xC0DE);
    let mut h = vec![0f32; rows * hidden];
    rng.fill_normal(&mut h);

    let mut ck_ref = rlx_kimi_k3::loader::CheckpointLoader::open(CKPT).unwrap();
    let reference = run_moe_paged(
        &mut ck_ref,
        &format!("language_model.model.layers.{layer}"),
        &h,
        d,
        Device::Cpu,
    )
    .unwrap();

    // worker (rank 1) owns [0,448); orchestrator owns [448,896) LOCAL.
    let ports = free_loopback_ports(2).unwrap();
    let peers: Vec<SocketAddr> = ports
        .iter()
        .map(|&p| SocketAddr::from((Ipv4Addr::LOCALHOST, p)))
        .collect();
    let peers_w = peers.clone();
    let worker = std::thread::spawn(move || {
        let t = TcpTransport::bind(1, 2, peers_w, HEAP).unwrap();
        let owned: HashSet<usize> = (0..448).collect();
        let mut p = KimiExpertProvider::open(CKPT, dims(), owned, Device::Cpu).unwrap();
        serve_expert_worker(&t, 0, &mut p).unwrap();
    });

    let t = TcpTransport::bind(0, 2, peers, HEAP).unwrap();
    let mut rank_of = vec![1u32; d.num_experts]; // [0,448) → worker rank 1
    for e in 448..d.num_experts {
        rank_of[e] = ExpertShards::LOCAL; // [448,896) → orchestrator-local (skip dispatch)
    }
    let shards = ExpertShards { rank_of };
    let local_owned: HashSet<usize> = (448..d.num_experts).collect();
    let mut local = KimiExpertProvider::open(CKPT, dims(), local_owned, Device::Cpu).unwrap();
    let mut ck = rlx_kimi_k3::loader::CheckpointLoader::open(CKPT).unwrap();
    let offload = run_moe_offload(
        &mut ck,
        &t,
        &shards,
        layer,
        &h,
        d,
        Device::Cpu,
        Some(&mut local),
    )
    .unwrap();

    shutdown_expert_workers(&t, &[1]).unwrap();
    worker.join().unwrap();

    let rel = (offload
        .iter()
        .zip(&reference)
        .map(|(o, r)| ((o - r) as f64).powi(2))
        .sum::<f64>()
        / reference
            .iter()
            .map(|r| (*r as f64).powi(2))
            .sum::<f64>()
            .max(1e-30))
    .sqrt();
    eprintln!("disaggregated+local-overflow vs local: rel-L2 {rel:.3e}");
    assert!(rel < 1e-3, "worker+local != whole (rel-L2 {rel:.3e})");
}
