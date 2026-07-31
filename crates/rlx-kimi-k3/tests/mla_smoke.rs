// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! Smoke test: one NoPE MLA layer with tiny synthetic weights, compiled on the
//! `RLX_TEST_DEVICE` backend (default CPU); finite output. MLA is standard
//! softmax attention, so it runs on every backend.

use rlx_core::flow_util::{built_from_hir, compile_built};
use rlx_ir::hir::{HirModule, HirMut};
use rlx_ir::{DType, Shape};
use rlx_kimi_k3::mla::{MlaDims, MlaWeights, build_mla_layer};
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
fn mla_layer_compiles_and_runs() {
    let d = MlaDims {
        hidden: 16,
        num_heads: 2,
        q_lora_rank: 8,
        kv_lora_rank: 6,
        qk_nope_head_dim: 4,
        qk_rope_head_dim: 2,
        v_head_dim: 4,
        eps: 1e-5,
        batch: 1,
        seq: 3,
    };
    let (hidden, h, ql, kvl, nope, rope, vd, qk) = (
        d.hidden,
        d.num_heads,
        d.q_lora_rank,
        d.kv_lora_rank,
        d.qk_nope_head_dim,
        d.qk_rope_head_dim,
        d.v_head_dim,
        d.qk(),
    );

    let w = MlaWeights {
        q_a_proj: fill(hidden * ql, 1),
        q_a_layernorm: vec![1.0; ql],
        q_b_proj: fill(ql * h * qk, 2),
        kv_a_proj_with_mqa: fill(hidden * (kvl + rope), 3),
        kv_a_layernorm: vec![1.0; kvl],
        kv_b_proj: fill(kvl * h * (nope + vd), 4),
        g_proj: fill(hidden * h * vd, 5),
        o_proj: fill(h * vd * hidden, 6),
    };

    let mut hir = HirModule::new("mla_smoke");
    let mut g = HirMut::new(&mut hir);
    let h_in = g.input("h", Shape::new(&[d.batch, d.seq, hidden], DType::F32));
    let mut params = HashMap::new();
    let out = build_mla_layer(&mut g, &mut params, "mla", h_in, &w, d).expect("build mla");
    g.set_outputs(vec![out]);

    let built = built_from_hir(hir, params).expect("build model");
    let mut compiled = compile_built(built, dev()).expect("compile mla");

    let hin = fill(d.batch * d.seq * hidden, 100);
    let y = compiled
        .run(&[("h", hin.as_slice())])
        .into_iter()
        .next()
        .expect("mla output");
    assert_eq!(y.len(), d.batch * d.seq * hidden);
    assert!(y.iter().all(|v| v.is_finite()), "MLA output must be finite");
}
