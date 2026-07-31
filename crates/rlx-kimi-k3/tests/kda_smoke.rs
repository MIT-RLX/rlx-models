// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! Smoke test: one KDA (Kimi Delta Attention) layer with tiny synthetic weights,
//! compiled + run on CPU; finite output. KDA uses the per-channel
//! `Op::GatedDeltaNet` which currently runs on CPU only (GPU kernels pending).

use rlx_core::flow_util::{built_from_hir, compile_built};
use rlx_ir::hir::{HirModule, HirMut};
use rlx_ir::{DType, Shape};
use rlx_kimi_k3::kda::{KdaDims, KdaWeights, build_kda_layer};
use rlx_runtime::Device;
use std::collections::HashMap;

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

#[test]
fn kda_layer_compiles_and_runs() {
    let d = KdaDims {
        hidden: 16,
        num_heads: 2,
        head_dim: 8,
        conv_kernel: 4,
        gate_lower_bound: Some(-5.0),
        eps: 1e-5,
        batch: 1,
        seq: 3,
    };
    let (hidden, h, hd, proj, k) = (d.hidden, d.num_heads, d.head_dim, d.proj(), d.conv_kernel);

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

    let mut hir = HirModule::new("kda_smoke");
    let mut g = HirMut::new(&mut hir);
    let h_in = g.input("h", Shape::new(&[d.batch, d.seq, hidden], DType::F32));
    let mut params = HashMap::new();
    let out = build_kda_layer(&mut g, &mut params, "kda", h_in, &w, d).expect("build kda");
    g.set_outputs(vec![out]);

    let built = built_from_hir(hir, params).expect("build model");
    let mut compiled = compile_built(built, dev()).expect("compile kda");

    let hin = fill(d.batch * d.seq * hidden, 100);
    let y = compiled
        .run(&[("h", hin.as_slice())])
        .into_iter()
        .next()
        .expect("kda output");
    assert_eq!(y.len(), d.batch * d.seq * hidden);
    assert!(y.iter().all(|v| v.is_finite()), "KDA output must be finite");
}
