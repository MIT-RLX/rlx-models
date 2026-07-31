// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! Verifies the new **per-channel** `Op::GatedDeltaNet` variant added to rlx-ir /
//! rlx-cpu for Kimi-K3 KDA: the state decays per key-row `S[i,j] *= exp(g[i])`
//! (vs the scalar per-head `S *= exp(g)`). Builds a tiny graph via
//! `HirMut::gated_delta_net_pc`, runs it on CPU, and compares against a reference
//! implementation of the recurrence.

use rlx_core::flow_util::{built_from_hir, compile_built};
use rlx_ir::hir::{HirModule, HirMut};
use rlx_ir::{DType, HirGraphExt, Shape};
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
            (((s >> 33) as f32) / (u32::MAX as f32) - 0.5) * 0.6
        })
        .collect()
}

/// Reference per-channel gated delta-net (no carry; state reset per batch).
/// q,k,v: [b,s,h,n]; g: [b,s,h,n] (per key channel); beta: [b,s,h]. out: [b,s,h,n].
#[allow(clippy::too_many_arguments)]
fn reference(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: &[f32],
    beta: &[f32],
    b: usize,
    s: usize,
    h: usize,
    n: usize,
) -> Vec<f32> {
    let scale = 1.0f32 / (n as f32).sqrt();
    let mut out = vec![0f32; b * s * h * n];
    let hn = h * n;
    for bi in 0..b {
        for hi in 0..h {
            let mut smat = vec![0f32; n * n]; // S[i,j] at i*n+j
            for ti in 0..s {
                let base = bi * s * hn + ti * hn + hi * n;
                let gbase = base; // g is per-channel, same layout as q/k/v
                let betabase = bi * s * h + ti * h + hi;
                let (qr, kr, vr) = (&q[base..base + n], &k[base..base + n], &v[base..base + n]);
                let gr = &g[gbase..gbase + n];
                let bt = beta[betabase];
                // S[i,j] *= exp(g[i])
                for i in 0..n {
                    let a = gr[i].exp();
                    for j in 0..n {
                        smat[i * n + j] *= a;
                    }
                }
                // sk[j] = sum_i S[i,j] * k[i]
                let mut sk = vec![0f32; n];
                for i in 0..n {
                    for j in 0..n {
                        sk[j] += smat[i * n + j] * kr[i];
                    }
                }
                // d[j] = (v[j] - sk[j]) * beta
                for j in 0..n {
                    sk[j] = (vr[j] - sk[j]) * bt;
                }
                // S[i,j] += k[i] * d[j]
                for i in 0..n {
                    for j in 0..n {
                        smat[i * n + j] += kr[i] * sk[j];
                    }
                }
                // o[j] = (sum_i S[i,j] * q[i]) * scale
                for j in 0..n {
                    let mut acc = 0f32;
                    for i in 0..n {
                        acc += smat[i * n + j] * qr[i];
                    }
                    out[base + j] = acc * scale;
                }
            }
        }
    }
    out
}

#[test]
fn per_channel_gated_delta_net_matches_reference() {
    let (b, s, h, n) = (1usize, 3usize, 2usize, 4usize);
    let bshn = b * s * h * n;

    let q = fill(bshn, 1);
    let k = fill(bshn, 2);
    let v = fill(bshn, 3);
    let g = fill(bshn, 4); // per-channel log-gate
    let beta = fill(b * s * h, 5);

    let mut hir = HirModule::new("gdn_pc_test");
    let mut gb = HirMut::new(&mut hir);
    let f = DType::F32;
    let q_in = gb.input("q", Shape::new(&[b, s, h, n], f));
    let k_in = gb.input("k", Shape::new(&[b, s, h, n], f));
    let v_in = gb.input("v", Shape::new(&[b, s, h, n], f));
    let g_in = gb.input("g", Shape::new(&[b, s, h, n], f)); // rank-4 per-channel gate
    let beta_in = gb.input("beta", Shape::new(&[b, s, h], f));
    let out = gb.gated_delta_net_pc(
        q_in,
        k_in,
        v_in,
        g_in,
        beta_in,
        n,
        Shape::new(&[b, s, h, n], f),
    );
    gb.set_outputs(vec![out]);

    let built = built_from_hir(hir, HashMap::new()).expect("build gdn_pc graph");
    let mut compiled = compile_built(built, dev()).expect("compile gdn_pc");
    let y = compiled
        .run(&[
            ("q", q.as_slice()),
            ("k", k.as_slice()),
            ("v", v.as_slice()),
            ("g", g.as_slice()),
            ("beta", beta.as_slice()),
        ])
        .into_iter()
        .next()
        .expect("gdn_pc output");

    let want = reference(&q, &k, &v, &g, &beta, b, s, h, n);
    assert_eq!(y.len(), want.len());
    let mut worst = 0f32;
    for i in 0..y.len() {
        worst = worst.max((y[i] - want[i]).abs());
    }
    assert!(
        worst < 1e-5,
        "per-channel GatedDeltaNet worst diff {worst} > 1e-5"
    );
}
