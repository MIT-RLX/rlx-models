// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! Smoke test: the LatentMoE FFN with tiny synthetic weights, compiled on the
//! `RLX_TEST_DEVICE` backend (default CPU); finite output.

use rlx_core::flow_util::{built_from_hir, compile_built};
use rlx_ir::hir::{HirModule, HirMut};
use rlx_ir::{DType, Shape};
use rlx_kimi_k3::moe::{MoeDims, MoeWeights, build_latent_moe};
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
fn latent_moe_compiles_and_runs() {
    let d = MoeDims {
        hidden: 16,
        latent: 12,
        moe_inter: 8,
        num_experts: 4,
        top_k: 2,
        num_shared: 1,
        routed_scaling: 1.0,
        eps: 1e-5,
        situ_beta: 4.0,
        situ_linear_beta: Some(25.0),
        batch: 1,
        seq: 3,
    };
    let (hidden, l, mi, e, ns) = (d.hidden, d.latent, d.moe_inter, d.num_experts, d.num_shared);
    let si = ns * mi;

    let w = MoeWeights {
        router: fill(hidden * e, 1),
        e_score_bias: fill(e, 2),
        down_latent: fill(hidden * l, 3),
        up_latent: fill(l * hidden, 4),
        routed_norm: vec![1.0; l],
        experts_gate_up: fill(e * l * 2 * mi, 5),
        experts_down: fill(e * mi * l, 6),
        shared_gate: fill(hidden * si, 7),
        shared_up: fill(hidden * si, 8),
        shared_down: fill(si * hidden, 9),
    };

    let mut hir = HirModule::new("moe_smoke");
    let mut g = HirMut::new(&mut hir);
    let h_in = g.input("h", Shape::new(&[d.batch, d.seq, hidden], DType::F32));
    let mut params = HashMap::new();
    let out = build_latent_moe(&mut g, &mut params, "moe", h_in, &w, d).expect("build moe");
    g.set_outputs(vec![out]);

    let built = built_from_hir(hir, params).expect("build model");
    let mut compiled = compile_built(built, dev()).expect("compile moe");

    let hin = fill(d.batch * d.seq * hidden, 100);
    let y = compiled
        .run(&[("h", hin.as_slice())])
        .into_iter()
        .next()
        .expect("moe output");
    assert_eq!(y.len(), d.batch * d.seq * hidden);
    assert!(y.iter().all(|v| v.is_finite()), "MoE output must be finite");
}
