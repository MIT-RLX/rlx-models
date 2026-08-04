// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! Verifies the **carry + per-channel** `Op::GatedDeltaNet` variant (Kimi-K3 KDA
//! *decode*): a scan resumed from a non-zero recurrent state must produce the
//! same output as the corresponding tail of the full-sequence scan. This is the
//! property O(1) stateful decode relies on — process one chunk/step from the
//! prior state instead of re-scanning the whole prefix. Runs via
//! `HirMut::gated_delta_net_carry_pc` on `RLX_TEST_DEVICE`, checked against a
//! reference recurrence.

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

/// Reference per-channel gated delta-net starting from `init_state` `[b,h,n,n]`.
/// Returns `(out [b,s,h,n], final_state [b,h,n,n])`. `b == 1`.
#[allow(clippy::too_many_arguments)]
fn reference(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: &[f32],
    beta: &[f32],
    init_state: &[f32],
    s: usize,
    h: usize,
    n: usize,
) -> (Vec<f32>, Vec<f32>) {
    let scale = 1.0f32 / (n as f32).sqrt();
    let mut out = vec![0f32; s * h * n];
    let mut fstate = vec![0f32; h * n * n];
    let hn = h * n;
    for hi in 0..h {
        let soff = hi * n * n;
        let mut smat = init_state[soff..soff + n * n].to_vec(); // S[i,j] at i*n+j
        for ti in 0..s {
            let base = ti * hn + hi * n;
            let betabase = ti * h + hi;
            let (qr, kr, vr) = (&q[base..base + n], &k[base..base + n], &v[base..base + n]);
            let gr = &g[base..base + n];
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
        fstate[soff..soff + n * n].copy_from_slice(&smat);
    }
    (out, fstate)
}

#[test]
fn carry_pc_resumes_scan_from_state() {
    // Full sequence split as chunk1 [0..s1) then chunk2 [s1..s).
    let (h, n) = (2usize, 4usize);
    let (s1, s2) = (2usize, 3usize);
    let s = s1 + s2;
    let hn = h * n;

    let q = fill(s * hn, 1);
    let k = fill(s * hn, 2);
    let v = fill(s * hn, 3);
    let g = fill(s * hn, 4); // per-channel log-gate
    let beta = fill(s * h, 5);

    // Reference: full scan, and the state after chunk1 (the decode "cache").
    let zero = vec![0f32; h * n * n];
    let (full_out, _) = reference(&q, &k, &v, &g, &beta, &zero, s, h, n);
    let (_, state_after1) = reference(
        &q[..s1 * hn],
        &k[..s1 * hn],
        &v[..s1 * hn],
        &g[..s1 * hn],
        &beta[..s1 * h],
        &zero,
        s1,
        h,
        n,
    );

    // Graph: run ONLY chunk2 via carry_pc, resuming from `state_after1`.
    let mut hir = HirModule::new("kda_carry_pc");
    let mut gb = HirMut::new(&mut hir);
    let f = DType::F32;
    let q_in = gb.input("q", Shape::new(&[1, s2, h, n], f));
    let k_in = gb.input("k", Shape::new(&[1, s2, h, n], f));
    let v_in = gb.input("v", Shape::new(&[1, s2, h, n], f));
    let g_in = gb.input("g", Shape::new(&[1, s2, h, n], f));
    let beta_in = gb.input("beta", Shape::new(&[1, s2, h], f));
    let state_in = gb.input("state", Shape::new(&[1, h, n, n], f));
    let out = gb.gated_delta_net_carry_pc(
        q_in,
        k_in,
        v_in,
        g_in,
        beta_in,
        state_in,
        n,
        Shape::new(&[1, s2, h, n], f),
    );
    gb.set_outputs(vec![out]);

    let built = built_from_hir(hir, HashMap::new()).expect("build carry_pc graph");
    let mut compiled = compile_built(built, dev()).expect("compile carry_pc");
    let y = compiled
        .run(&[
            ("q", &q[s1 * hn..]),
            ("k", &k[s1 * hn..]),
            ("v", &v[s1 * hn..]),
            ("g", &g[s1 * hn..]),
            ("beta", &beta[s1 * h..]),
            ("state", state_after1.as_slice()),
        ])
        .into_iter()
        .next()
        .expect("carry_pc output");

    // chunk2's output must equal the tail of the full-sequence scan.
    let want = &full_out[s1 * hn..];
    assert_eq!(y.len(), want.len());
    let mut worst = 0f32;
    for i in 0..y.len() {
        worst = worst.max((y[i] - want[i]).abs());
    }
    assert!(
        worst < 1e-5,
        "carry_pc resumed scan worst diff {worst} > 1e-5 (state not consumed correctly?)"
    );
}

/// The recurrent state must be READABLE after a run so a decode loop can thread
/// it: output the state node and check it equals the reference's final state.
#[test]
fn carry_pc_state_is_read_back() {
    let (h, n) = (2usize, 4usize);
    let s = 3usize;
    let hn = h * n;

    let q = fill(s * hn, 11);
    let k = fill(s * hn, 12);
    let v = fill(s * hn, 13);
    let g = fill(s * hn, 14);
    let beta = fill(s * h, 15);

    let zero = vec![0f32; h * n * n];
    let (_, want_state) = reference(&q, &k, &v, &g, &beta, &zero, s, h, n);

    let mut hir = HirModule::new("kda_carry_readback");
    let mut gb = HirMut::new(&mut hir);
    let f = DType::F32;
    let q_in = gb.input("q", Shape::new(&[1, s, h, n], f));
    let k_in = gb.input("k", Shape::new(&[1, s, h, n], f));
    let v_in = gb.input("v", Shape::new(&[1, s, h, n], f));
    let g_in = gb.input("g", Shape::new(&[1, s, h, n], f));
    let beta_in = gb.input("beta", Shape::new(&[1, s, h], f));
    let state_in = gb.input("state", Shape::new(&[1, h, n, n], f));
    let out = gb.gated_delta_net_carry_pc(
        q_in,
        k_in,
        v_in,
        g_in,
        beta_in,
        state_in,
        n,
        Shape::new(&[1, s, h, n], f),
    );
    // Output BOTH the scan result and the (written-back) state.
    gb.set_outputs(vec![out, state_in]);

    let built = built_from_hir(hir, HashMap::new()).expect("build readback graph");
    let mut compiled = compile_built(built, dev()).expect("compile readback");
    let outs = compiled.run(&[
        ("q", q.as_slice()),
        ("k", k.as_slice()),
        ("v", v.as_slice()),
        ("g", g.as_slice()),
        ("beta", beta.as_slice()),
        ("state", zero.as_slice()),
    ]);
    let state_out = &outs[1];

    assert_eq!(state_out.len(), want_state.len());
    let mut worst = 0f32;
    for i in 0..state_out.len() {
        worst = worst.max((state_out[i] - want_state[i]).abs());
    }
    assert!(
        worst < 1e-5,
        "carry_pc state read-back worst diff {worst} > 1e-5 (writeback not visible?)"
    );
}
