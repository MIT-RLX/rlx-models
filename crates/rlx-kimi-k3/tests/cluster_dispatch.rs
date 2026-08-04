//! Validates the decode-loop dispatch plumbing: with a `ClusterMoe` installed,
//! `try_offload` (what `run_moe_layer` calls per MoE layer) routes to the workers
//! over a real loopback transport and matches the local `run_moe_paged`. Skips if
//! the checkpoint isn't mounted.
#![cfg(feature = "cluster")]

use rlx_distributed::{
    ExpertShards, TcpTransport, free_loopback_ports, serve_expert_worker, shutdown_expert_workers,
};
use rlx_ir::Philox4x32;
use rlx_kimi_k3::dist_experts::{
    ClusterMoe, KimiExpertProvider, install_cluster_moe, take_cluster_moe, try_offload,
};
use rlx_kimi_k3::moe::MoeDims;
use rlx_kimi_k3::runner::run_moe_paged;
use rlx_runtime::Device;
use std::collections::HashSet;
use std::net::{Ipv4Addr, SocketAddr};
use std::path::Path;
use std::sync::Arc;

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
fn installed_cluster_dispatch_matches_local() {
    if !Path::new(CKPT).join("config.json").exists() {
        eprintln!("skip: {CKPT} not mounted");
        return;
    }
    const HEAP: usize = 64 * 1024 * 1024;
    let d = dims();
    let layer: u32 = 1;
    let (rows, hidden) = (d.batch * d.seq, d.hidden);
    let mut rng = Philox4x32::new(0xD15);
    let mut mn = vec![0f32; rows * hidden];
    rng.fill_normal(&mut mn); // stand-in MoE FFN input

    // local reference (no cluster installed).
    let mut ck_ref = rlx_kimi_k3::loader::CheckpointLoader::open(CKPT).unwrap();
    let reference = run_moe_paged(
        &mut ck_ref,
        &format!("language_model.model.layers.{layer}"),
        &mn,
        d,
        Device::Cpu,
    )
    .unwrap();

    // loopback worker owning all experts.
    let ports = free_loopback_ports(2).unwrap();
    let peers: Vec<SocketAddr> = ports
        .iter()
        .map(|&p| SocketAddr::from((Ipv4Addr::LOCALHOST, p)))
        .collect();
    let peers_w = peers.clone();
    let worker = std::thread::spawn(move || {
        let t = TcpTransport::bind(1, 2, peers_w, HEAP).unwrap();
        let owned: HashSet<usize> = (0..896).collect();
        let mut p = KimiExpertProvider::open(CKPT, dims(), owned, Device::Cpu).unwrap();
        serve_expert_worker(&t, 0, &mut p).unwrap();
    });

    let t: Arc<dyn rlx_distributed::Transport> =
        Arc::new(TcpTransport::bind(0, 2, peers, HEAP).unwrap());
    install_cluster_moe(ClusterMoe {
        transport: t.clone(),
        shards: ExpertShards::round_robin(d.num_experts, &[1]),
        local: None,
    });

    // this is exactly what the decode loop's `run_moe_layer` calls per MoE layer.
    let mut ck = rlx_kimi_k3::loader::CheckpointLoader::open(CKPT).unwrap();
    let dispatched = try_offload(&mut ck, layer, &mn, d, Device::Cpu).unwrap();

    shutdown_expert_workers(&*t, &[1]).unwrap();
    worker.join().unwrap();
    let _ = take_cluster_moe();

    let out = dispatched.expect("cluster installed → try_offload must return Some");
    let rel = (out
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
    eprintln!("installed-cluster try_offload vs local run_moe_paged: rel-L2 {rel:.3e}");
    assert!(
        rel < 1e-3,
        "cluster-dispatched MoE != local (rel-L2 {rel:.3e})"
    );
}

#[test]
fn no_cluster_installed_is_passthrough() {
    // Without an installed ClusterMoe, try_offload must return None (→ local path).
    let mut ck = match rlx_kimi_k3::loader::CheckpointLoader::open(CKPT) {
        Ok(c) => c,
        Err(_) => {
            eprintln!("skip: not mounted");
            return;
        }
    };
    let d = dims();
    let mn = vec![0f32; d.hidden];
    let r = try_offload(&mut ck, 1, &mn, d, Device::Cpu).unwrap();
    assert!(
        r.is_none(),
        "no cluster installed → try_offload must be None"
    );
}
