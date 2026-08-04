// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! O(1) KDA **decode step** parity: running a sequence in two chunks through
//! `build_kda_decode_step` — carrying the short-conv state AND the recurrent scan
//! state across the boundary — must reproduce the tail of the full-sequence
//! `build_kda_layer`. This is the O(seq)→O(1) decode path for Kimi-K3's 69 KDA
//! layers (vs re-scanning the whole prefix each token). Runs on `RLX_TEST_DEVICE`.

use rlx_core::flow_util::{built_from_hir, compile_built};
use rlx_ir::hir::{HirModule, HirMut};
use rlx_ir::{DType, Shape};
use rlx_kimi_k3::kda::{KdaDims, KdaWeights, build_kda_decode_step, build_kda_layer};
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

fn weights(d: KdaDims) -> KdaWeights {
    let (hidden, h, hd, proj, k) = (d.hidden, d.num_heads, d.head_dim, d.proj(), d.conv_kernel);
    KdaWeights {
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
    }
}

fn dims(seq: usize) -> KdaDims {
    KdaDims {
        hidden: 16,
        num_heads: 2,
        head_dim: 8,
        conv_kernel: 4,
        gate_lower_bound: Some(-5.0),
        eps: 1e-5,
        batch: 1,
        seq,
    }
}

#[test]
fn kda_decode_step_matches_full_layer() {
    let (s1, s2) = (2usize, 3usize);
    let s = s1 + s2;
    let d_full = dims(s);
    let (hidden, h, hd, proj, kk) = (
        d_full.hidden,
        d_full.num_heads,
        d_full.head_dim,
        d_full.proj(),
        d_full.conv_kernel,
    );
    let w = weights(d_full);
    let h_full = fill(s * hidden, 100);

    // ── full-sequence reference ──
    let mut hir = HirModule::new("kda_full");
    let mut g = HirMut::new(&mut hir);
    let h_in = g.input("h", Shape::new(&[1, s, hidden], DType::F32));
    let mut params = HashMap::new();
    let out = build_kda_layer(&mut g, &mut params, "kda", h_in, &w, d_full).expect("full");
    g.set_outputs(vec![out]);
    let built = built_from_hir(hir, params).expect("full built");
    let mut compiled = compile_built(built, dev()).expect("full compile");
    let full_out = compiled.run(&[("h", h_full.as_slice())]).remove(0);

    // Build one decode-step graph for `s_new` tokens; outputs
    // [out, new_conv_q, new_conv_k, new_conv_v, scan_state(written-back)].
    let build_step = |s_new: usize| {
        let mut hir = HirModule::new("kda_step");
        let mut g = HirMut::new(&mut hir);
        let h_in = g.input("h", Shape::new(&[1, s_new, hidden], DType::F32));
        let csq = g.input("csq", Shape::new(&[1, kk - 1, proj], DType::F32));
        let csk = g.input("csk", Shape::new(&[1, kk - 1, proj], DType::F32));
        let csv = g.input("csv", Shape::new(&[1, kk - 1, proj], DType::F32));
        let state = g.input("state", Shape::new(&[1, h, hd, hd], DType::F32));
        let mut params = HashMap::new();
        let (out, ncq, nck, ncv) = build_kda_decode_step(
            &mut g,
            &mut params,
            "kda",
            h_in,
            csq,
            csk,
            csv,
            state,
            &w,
            dims(s_new),
        )
        .expect("step");
        g.set_outputs(vec![out, ncq, nck, ncv, state]);
        let built = built_from_hir(hir, params).expect("step built");
        compile_built(built, dev()).expect("step compile")
    };

    // ── chunk 1 from zero conv/scan state ──
    let zero_cs = vec![0f32; (kk - 1) * proj];
    let zero_state = vec![0f32; h * hd * hd];
    let mut c1 = build_step(s1);
    let o1 = c1.run(&[
        ("h", &h_full[..s1 * hidden]),
        ("csq", zero_cs.as_slice()),
        ("csk", zero_cs.as_slice()),
        ("csv", zero_cs.as_slice()),
        ("state", zero_state.as_slice()),
    ]);
    let (ncq1, nck1, ncv1, state1) = (o1[1].clone(), o1[2].clone(), o1[3].clone(), o1[4].clone());

    // ── chunk 2 resuming from chunk 1's carried states ──
    let mut c2 = build_step(s2);
    let o2 = c2.run(&[
        ("h", &h_full[s1 * hidden..]),
        ("csq", ncq1.as_slice()),
        ("csk", nck1.as_slice()),
        ("csv", ncv1.as_slice()),
        ("state", state1.as_slice()),
    ]);
    let out2 = &o2[0];

    // chunk 2's output must equal the tail of the full-sequence KDA layer.
    let want = &full_out[s1 * hidden..];
    assert_eq!(out2.len(), want.len());
    let worst = out2
        .iter()
        .zip(want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        worst < 1e-5,
        "KDA decode-step tail vs full layer worst diff {worst} > 1e-5"
    );
}
