// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! Runner-level KDA decode parity: generate a sequence ONE TOKEN AT A TIME through
//! [`rlx_kimi_k3::runner::run_kda_decode_step`] (carrying the [`KdaState`] host-side,
//! the real generation pattern) and assert the last token's attention output matches
//! `build_kda_layer` prefilling the whole sequence at once. This proves the runner
//! decode API + `O(1)` state threading — not just the op (covered by `kda_decode_step`).

use rlx_core::flow_util::{built_from_hir, compile_built};
use rlx_ir::hir::{HirModule, HirMut};
use rlx_ir::{DType, Shape};
use rlx_kimi_k3::kda::{KdaDims, KdaWeights, build_kda_layer};
use rlx_kimi_k3::runner::{KdaState, run_kda_decode_step};
use rlx_runtime::Device;
use std::collections::HashMap;

fn dev() -> Device {
    match std::env::var("RLX_TEST_DEVICE").ok().as_deref() {
        Some("metal") | Some("mtl") => Device::Metal,
        Some("mlx") => Device::Mlx,
        Some("gpu") | Some("wgpu") => Device::Gpu,
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
fn kda_decode_one_at_a_time_matches_prefill() {
    let d = dev();
    let seq = 5;
    let hidden = dims(1).hidden;
    let w = weights(dims(1));
    let h_full = fill(seq * hidden, 7);

    // ── reference: prefill the whole sequence in one graph ──
    let mut hir = HirModule::new("kda_full");
    let mut g = HirMut::new(&mut hir);
    let h_node = g.input("h", Shape::new(&[1, seq, hidden], DType::F32));
    let mut params = HashMap::new();
    let out = build_kda_layer(&mut g, &mut params, "kda", h_node, &w, dims(seq)).expect("full");
    g.set_outputs(vec![out]);
    let built = built_from_hir(hir, params).expect("full built");
    let mut compiled = compile_built(built, d).expect("full compile");
    let full = compiled.run(&[("h", h_full.as_slice())]).remove(0);
    let want_last = &full[(seq - 1) * hidden..]; // last token's attention output

    // ── decode: one token at a time, carrying KdaState ──
    let mut state = KdaState::zeros(dims(1));
    let mut got_last = Vec::new();
    for t in 0..seq {
        let (out_t, next) = run_kda_decode_step(
            &w,
            &h_full[t * hidden..(t + 1) * hidden],
            1,
            &state,
            dims(1),
            d,
        )
        .expect("decode step");
        state = next;
        got_last = out_t;
    }

    let worst = want_last
        .iter()
        .zip(&got_last)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    eprintln!("KDA decode(1-at-a-time) vs prefill {d:?}: worst |Δ| = {worst:.3e}");
    assert!(
        worst < 1e-4,
        "KDA decode diverges from prefill: {worst:.3e}"
    );
}
