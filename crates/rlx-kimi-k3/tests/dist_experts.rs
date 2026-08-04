//! Kimi `ExpertProvider` correctness via the sharding invariant: the routed-latent
//! partials of two DISJOINT shards (even + odd experts) must sum to the partial of
//! a single shard owning ALL experts — i.e. `dispatch_experts`' gather+sum is
//! order-/shard-invariant. Runs on the real checkpoint; skips if unmounted.
#![cfg(feature = "cluster")]

use rlx_distributed::ExpertProvider;
use rlx_ir::Philox4x32;
use rlx_kimi_k3::dist_experts::KimiExpertProvider;
use rlx_kimi_k3::moe::MoeDims;
use rlx_runtime::Device;
use std::collections::HashSet;
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
fn shard_partials_sum_to_whole() {
    if !Path::new(CKPT).join("config.json").exists() {
        eprintln!("skip: {CKPT} not mounted");
        return;
    }
    let d = dims();
    let (rows, l) = (1usize, d.latent);
    // 8 distinct fired experts for the single token.
    let ids: Vec<u32> = vec![5, 12, 33, 40, 101, 200, 355, 700];
    let probs: Vec<f32> = vec![0.2, 0.15, 0.1, 0.05, 0.25, 0.1, 0.1, 0.05];
    let mut rng = Philox4x32::new(0x9E3);
    let mut h_lat = vec![0f32; rows * l];
    rng.fill_normal(&mut h_lat);

    let all: HashSet<usize> = (0..d.num_experts).collect();
    let even: HashSet<usize> = (0..d.num_experts).step_by(2).collect();
    let odd: HashSet<usize> = (1..d.num_experts).step_by(2).collect();

    let mut pa = KimiExpertProvider::open(CKPT, d, all, Device::Cpu).unwrap();
    let mut pe = KimiExpertProvider::open(CKPT, d, even, Device::Cpu).unwrap();
    let mut po = KimiExpertProvider::open(CKPT, d, odd, Device::Cpu).unwrap();

    let whole = pa.compute(1, &h_lat, rows, l, &ids, &probs).unwrap();
    let part_e = pe.compute(1, &h_lat, rows, l, &ids, &probs).unwrap();
    let part_o = po.compute(1, &h_lat, rows, l, &ids, &probs).unwrap();

    assert_eq!(whole.len(), rows * l);
    assert!(whole.iter().all(|v| v.is_finite()));
    assert!(whole.iter().any(|&v| v != 0.0), "whole partial is all-zero");

    let mut max_abs = 0f32;
    let mut sum_abs = 0f64;
    for i in 0..rows * l {
        let e = (part_e[i] + part_o[i] - whole[i]).abs();
        max_abs = max_abs.max(e);
        sum_abs += (whole[i] as f64).abs();
    }
    let rel = max_abs as f64 / (sum_abs / (rows * l) as f64).max(1e-30);
    eprintln!("shard-sum vs whole: max|Δ| {max_abs:.3e}, rel {rel:.3e}");
    assert!(
        max_abs < 1e-3,
        "even+odd shard partials != whole (max|Δ| {max_abs:.3e})"
    );
}
