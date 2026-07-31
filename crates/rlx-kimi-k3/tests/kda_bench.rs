// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! Throughput benchmark for one KDA (Kimi Delta Attention) layer at the real
//! Kimi-K3 head config (96 heads × 128 head_dim, conv 4). Ignored by default;
//! run explicitly, e.g.
//!   RLX_TEST_DEVICE=metal cargo test -p rlx-kimi-k3 --features metal \
//!     --test kda_bench -- --ignored --nocapture
//! Sequence lengths come from RLX_BENCH_SEQ (comma-separated, default "64,256").
//! `hidden` is reduced to 2048 (env RLX_BENCH_HIDDEN) so the projection matmuls
//! don't dominate — the scan cost (the KDA-specific bottleneck) is set by
//! heads × head_dim, which are kept real.

use rlx_core::flow_util::{built_from_hir, compile_built};
use rlx_ir::hir::{HirModule, HirMut};
use rlx_ir::{DType, Shape};
use rlx_kimi_k3::kda::{KdaDims, KdaWeights, build_kda_layer};
use rlx_runtime::Device;
use std::collections::HashMap;
use std::time::Instant;

fn dev() -> Device {
    match std::env::var("RLX_TEST_DEVICE").ok().as_deref() {
        Some("metal") | Some("mtl") => Device::Metal,
        Some("mlx") => Device::Mlx,
        Some("gpu") | Some("wgpu") => Device::Gpu,
        Some("coreml") | Some("ane") => Device::Ane,
        Some("cuda") => Device::Cuda,
        Some("vulkan") | Some("vk") => Device::Vulkan,
        _ => Device::Cpu,
    }
}

fn fill(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (((s >> 33) as f32) / (u32::MAX as f32) - 0.5) * 0.2
        })
        .collect()
}

fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(d)
}

#[test]
#[ignore = "benchmark; run with --ignored --nocapture"]
fn kda_layer_throughput() {
    let hidden = env_usize("RLX_BENCH_HIDDEN", 2048);
    let seqs: Vec<usize> = std::env::var("RLX_BENCH_SEQ")
        .unwrap_or_else(|_| "64,256".into())
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let iters = env_usize("RLX_BENCH_ITERS", 5);
    let device = dev();

    println!(
        "\nKDA layer throughput — device={device:?} hidden={hidden} heads=96 head_dim=128 conv=4 (iters={iters})"
    );
    println!(
        "  {:>6}  {:>10}  {:>12}  {:>14}",
        "seq", "ms/layer", "tok/s/layer", "est 93-layer"
    );

    for &seq in &seqs {
        let d = KdaDims {
            hidden,
            num_heads: 96,
            head_dim: 128,
            conv_kernel: 4,
            gate_lower_bound: Some(-5.0),
            eps: 1e-5,
            batch: 1,
            seq,
        };
        let (h, hd, proj, k) = (d.num_heads, d.head_dim, d.proj(), d.conv_kernel);
        let w = KdaWeights {
            q_proj: fill(hidden * proj, 1),
            k_proj: fill(hidden * proj, 2),
            v_proj: fill(hidden * proj, 3),
            q_conv: fill(proj * k, 4),
            k_conv: fill(proj * k, 5),
            v_conv: fill(proj * k, 6),
            f_a: fill(hidden * hd, 7),
            f_b: fill(hd * proj, 8),
            dt_bias: fill(proj, 9),
            a_log: fill(hd, 10),
            b_proj: fill(hidden * h, 11),
            g_proj: fill(hidden * proj, 12),
            o_norm: vec![1.0; hd],
            o_proj: fill(proj * hidden, 13),
        };

        let mut hir = HirModule::new("kda_bench");
        let mut g = HirMut::new(&mut hir);
        let h_in = g.input("h", Shape::new(&[d.batch, seq, hidden], DType::F32));
        let mut params = HashMap::new();
        let out = build_kda_layer(&mut g, &mut params, "kda", h_in, &w, d).expect("build kda");
        g.set_outputs(vec![out]);
        let built = built_from_hir(hir, params).expect("build model");
        let mut compiled = compile_built(built, device).expect("compile kda");

        let hin = fill(d.batch * seq * hidden, 100);
        // warm up (compile-time unroll / kernel JIT happens on first run)
        let _ = compiled.run(&[("h", hin.as_slice())]);

        let t0 = Instant::now();
        for _ in 0..iters {
            let y = compiled.run(&[("h", hin.as_slice())]);
            std::hint::black_box(&y);
        }
        let per = t0.elapsed().as_secs_f64() / iters as f64;
        let tok_s = seq as f64 / per;
        // 93-layer whole-decoder estimate if every layer cost like this KDA layer
        // (upper bound on speed — MLA + MoE layers are extra work).
        let full_tok_s = tok_s / 93.0;
        println!(
            "  {seq:>6}  {:>10.2}  {:>12.1}  {:>10.1} tok/s",
            per * 1e3,
            tok_s,
            full_tok_s
        );
    }
    println!();
}
