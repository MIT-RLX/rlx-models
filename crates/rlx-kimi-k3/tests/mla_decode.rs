// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! MLA **KV-cache decode** parity: generate a sequence one token at a time through
//! [`build_mla_decode_step`] (growing the key/value cache each step, starting from
//! an empty cache) and assert the last token's output matches `build_mla_layer`
//! prefilling the whole sequence. This is the O(1) decode path for Kimi-K3's 24 MLA
//! layers — the piece (with KDA decode) that unlocks a full O(1)/token generation.

use rlx_core::flow_util::{built_from_hir, compile_built};
use rlx_ir::hir::{HirModule, HirMut};
use rlx_ir::{DType, Shape};
use rlx_kimi_k3::mla::{MlaDims, MlaWeights, build_mla_decode_step, build_mla_layer};
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

fn dims(seq: usize) -> MlaDims {
    MlaDims {
        hidden: 16,
        num_heads: 2,
        q_lora_rank: 8,
        kv_lora_rank: 6,
        qk_nope_head_dim: 4,
        qk_rope_head_dim: 2,
        v_head_dim: 4,
        eps: 1e-5,
        batch: 1,
        seq,
    }
}

fn weights(d: MlaDims) -> MlaWeights {
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
    MlaWeights {
        q_a_proj: fill(hidden * ql, 1),
        q_a_layernorm: vec![1.0; ql],
        q_b_proj: fill(ql * h * qk, 2),
        kv_a_proj_with_mqa: fill(hidden * (kvl + rope), 3),
        kv_a_layernorm: vec![1.0; kvl],
        kv_b_proj: fill(kvl * h * (nope + vd), 4),
        g_proj: fill(hidden * h * vd, 5),
        o_proj: fill(h * vd * hidden, 6),
    }
}

#[test]
fn mla_decode_kv_cache_matches_prefill() {
    let d = dev();
    let seq = 4;
    let cfg = dims(1);
    let (hidden, hq) = (cfg.hidden, cfg.num_heads * cfg.qk());
    let w = weights(cfg);
    let h_full = fill(seq * hidden, 7);

    // ── reference: prefill the whole sequence ──
    let mut hir = HirModule::new("mla_full");
    let mut g = HirMut::new(&mut hir);
    let h_node = g.input("h", Shape::new(&[1, seq, hidden], DType::F32));
    let mut params = HashMap::new();
    let out = build_mla_layer(&mut g, &mut params, "mla", h_node, &w, dims(seq)).expect("full");
    g.set_outputs(vec![out]);
    let built = built_from_hir(hir, params).expect("full built");
    let mut compiled = compile_built(built, d).expect("full compile");
    let full = compiled.run(&[("h", h_full.as_slice())]).remove(0);
    let want_last = &full[(seq - 1) * hidden..];

    // ── decode: one token at a time, growing the KV cache from empty ──
    let (mut cache_k, mut cache_v) = (Vec::<f32>::new(), Vec::<f32>::new());
    let mut got_last = Vec::new();
    for t in 0..seq {
        let s_past = t; // cache holds t tokens so far
        let mut hir = HirModule::new("mla_step");
        let mut g = HirMut::new(&mut hir);
        let h_node = g.input("h", Shape::new(&[1, 1, hidden], DType::F32));
        let ck = g.input("ck", Shape::new(&[1, s_past, hq], DType::F32));
        let cv = g.input("cv", Shape::new(&[1, s_past, hq], DType::F32));
        let mut params = HashMap::new();
        let (out, nk, nv) =
            build_mla_decode_step(&mut g, &mut params, "mla", h_node, ck, cv, &w, dims(1))
                .expect("decode step");
        g.set_outputs(vec![out, nk, nv]);
        let built = built_from_hir(hir, params).expect("step built");
        let mut c = compile_built(built, d).expect("step compile");
        let mut r = c.run(&[
            ("h", &h_full[t * hidden..(t + 1) * hidden]),
            ("ck", cache_k.as_slice()),
            ("cv", cache_v.as_slice()),
        ]);
        got_last = r.remove(0);
        cache_k = r.remove(0);
        cache_v = r.remove(0);
    }

    let worst = want_last
        .iter()
        .zip(&got_last)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    eprintln!("MLA KV-cache decode vs prefill {d:?}: worst |Δ| = {worst:.3e}");
    assert!(
        worst < 1e-4,
        "MLA decode diverges from prefill: {worst:.3e}"
    );
}
