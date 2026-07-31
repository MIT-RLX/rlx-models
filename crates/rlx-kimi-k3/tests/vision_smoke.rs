// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! Smoke test: the MoonViT vision tower + patchmergerv2 projector with tiny
//! synthetic weights, compiled on `RLX_TEST_DEVICE` (default CPU); finite tokens.

use rlx_core::flow_util::{built_from_hir, compile_built};
use rlx_ir::hir::{HirModule, HirMut};
use rlx_ir::{DType, Shape};
use rlx_kimi_k3::vision::{VisionBlockWeights, VisionDims, VisionWeights, build_vision};
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
            (((s >> 33) as f32) / (u32::MAX as f32) - 0.5) * 0.15
        })
        .collect()
}

#[test]
fn vision_tower_compiles_and_runs() {
    let d = VisionDims {
        hidden: 8,
        qkv_hidden: 12,
        num_heads: 3,
        head_dim: 4,
        inter: 16,
        merge: 2,
        text_hidden: 14,
        proj_mid: 16,
        eps: 1e-5,
        grid_h: 4,
        grid_w: 4,
    };
    let l = d.seq_len();
    let (hid, qh, hd) = (d.hidden, d.qkv_hidden, d.head_dim);

    let blocks: Vec<VisionBlockWeights> = (0..2)
        .map(|i| {
            let sd = 100 + i as u64 * 50;
            VisionBlockWeights {
                norm0: vec![1.0; hid],
                wqkv: fill(hid * 3 * qh, sd + 1),
                wo: fill(qh * hid, sd + 2),
                norm1: vec![1.0; hid],
                fc0: fill(hid * d.inter, sd + 3),
                fc1: fill(d.inter * hid, sd + 4),
            }
        })
        .collect();
    let w = VisionWeights {
        blocks,
        final_norm: vec![1.0; hid],
        proj0: fill(d.merge_in() * d.proj_mid, 700),
        proj2: fill(d.proj_mid * d.text_hidden, 701),
        post_norm: vec![1.0; d.text_hidden],
    };

    let mut hir = HirModule::new("vision_smoke");
    let mut g = HirMut::new(&mut hir);
    let hidden = g.input("hidden", Shape::new(&[1, l, hid], DType::F32));
    let cos = g.input("cos", Shape::new(&[l, hd / 2], DType::F32));
    let sin = g.input("sin", Shape::new(&[l, hd / 2], DType::F32));
    let mut params = HashMap::new();
    let out = build_vision(&mut g, &mut params, hidden, cos, sin, &w, d).expect("build vision");
    g.set_outputs(vec![out]);

    let built = built_from_hir(hir, params).expect("build model");
    let mut compiled = compile_built(built, dev()).expect("compile vision");

    let hin = fill(l * hid, 1);
    let cosv = fill(l * (hd / 2), 2);
    let sinv = fill(l * (hd / 2), 3);
    let y = compiled
        .run(&[
            ("hidden", hin.as_slice()),
            ("cos", cosv.as_slice()),
            ("sin", sinv.as_slice()),
        ])
        .into_iter()
        .next()
        .expect("vision output");
    let n_merged = (d.grid_h / d.merge) * (d.grid_w / d.merge);
    assert_eq!(y.len(), n_merged * d.text_hidden);
    assert!(
        y.iter().all(|v| v.is_finite()),
        "vision tokens must be finite"
    );
}
