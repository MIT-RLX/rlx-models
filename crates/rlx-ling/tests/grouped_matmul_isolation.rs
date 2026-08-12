// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! Minimal isolation of [`rlx_ir::Op::GroupedMatMul`] — the one op that makes
//! Ling's MoE disagree across backends.
//!
//! Bisecting the full model narrowed a CPU↔MLX divergence to the routed expert
//! path (`moe_shared_expert_only_matches_cpu` is bit-exact, everything with live
//! routed experts is not). This reduces that to the single op, on the two layouts
//! the MoE builder actually emits:
//!
//! * **contiguous** `[E, K, N]` — a plain expert bank.
//! * **transposed** `[E, N, K] → transpose(0,2,1)` — what
//!   [`rlx_deepseek::moe::emit_deepseek_moe`] feeds it, since HF stores experts
//!   `[E, N, K]`.
//!
//! Reference is computed on the host, so this is an absolute correctness check
//! rather than a backend-vs-backend one.

use rlx_core::flow_util::compile_graph_profile;
use rlx_flow::CompileProfile;
use rlx_ir::infer::GraphExt;
use rlx_ir::{DType, Graph, NodeId, Op, Shape};
use rlx_runtime::Device;
use std::collections::HashMap;

const M: usize = 6; // rows / tokens
const K: usize = 12; // reduction dim
const N: usize = 8; // output dim
const E: usize = 4; // experts

fn dev() -> Device {
    std::env::var("RLX_TEST_DEVICE")
        .ok()
        .filter(|s| !s.is_empty())
        .map(|s| rlx_cli::parse_device(&s).expect("bad RLX_TEST_DEVICE"))
        .unwrap_or(Device::Cpu)
}

fn fill(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f32) / (u32::MAX as f32) - 0.5
        })
        .collect()
}

/// `out[r, :] = x[r, :] @ w_ekn[idx[r]]`, computed on the host in f64.
fn reference(x: &[f32], w_ekn: &[f32], idx: &[usize]) -> Vec<f32> {
    let mut out = vec![0f32; M * N];
    for (r, &e) in idx.iter().enumerate() {
        for j in 0..N {
            let mut acc = 0f64;
            for i in 0..K {
                acc += x[r * K + i] as f64 * w_ekn[e * K * N + i * N + j] as f64;
            }
            out[r * N + j] = acc as f32;
        }
    }
    out
}

/// Build `x @ w[idx]`; `transposed` feeds the op a `[E,N,K]` param transposed
/// in-graph to `[E,K,N]` (the MoE builder's layout) instead of a native `[E,K,N]`.
fn run_case(
    device: Device,
    transposed: bool,
    x: &[f32],
    w_ekn: &[f32],
    idx: &[f32],
    profile: &CompileProfile,
) -> Vec<f32> {
    let f = DType::F32;
    let mut g = Graph::new("grouped_matmul_isolation");
    let mut params: HashMap<String, Vec<f32>> = HashMap::new();

    let x_id = g.add_node(
        Op::Param { name: "x".into() },
        vec![],
        Shape::new(&[M, K], f),
    );
    params.insert("x".into(), x.to_vec());

    let w_id: NodeId = if transposed {
        // Store [E, N, K] (HF order) and transpose in-graph, as the MoE does.
        let mut w_enk = vec![0f32; E * N * K];
        for e in 0..E {
            for i in 0..K {
                for j in 0..N {
                    w_enk[e * N * K + j * K + i] = w_ekn[e * K * N + i * N + j];
                }
            }
        }
        let raw = g.add_node(
            Op::Param { name: "w".into() },
            vec![],
            Shape::new(&[E, N, K], f),
        );
        params.insert("w".into(), w_enk);
        g.transpose_(raw, vec![0, 2, 1])
    } else {
        let raw = g.add_node(
            Op::Param { name: "w".into() },
            vec![],
            Shape::new(&[E, K, N], f),
        );
        params.insert("w".into(), w_ekn.to_vec());
        raw
    };

    let idx_id = g.add_node(
        Op::Param { name: "idx".into() },
        vec![],
        Shape::new(&[M], f),
    );
    params.insert("idx".into(), idx.to_vec());

    let out = g.add_node(
        Op::GroupedMatMul,
        vec![x_id, w_id, idx_id],
        Shape::new(&[M, N], f),
    );
    g.set_outputs(vec![out]);

    let mut compiled =
        compile_graph_profile(device, g, params, profile).expect("compile grouped matmul");
    compiled.run(&[]).into_iter().next().expect("output")
}

fn check(label: &str, device: Device, transposed: bool, profile: &CompileProfile, pname: &str) {
    let x = fill(M * K, 11);
    let w = fill(E * K * N, 23);
    let idx_u: Vec<usize> = (0..M).map(|r| (r * 3 + 1) % E).collect();
    let idx: Vec<f32> = idx_u.iter().map(|&e| e as f32).collect();

    let want = reference(&x, &w, &idx_u);
    let got = run_case(device, transposed, &x, &w, &idx, profile);
    assert_eq!(got.len(), want.len());

    let max_abs = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let scale = want.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
    eprintln!(
        "{label} [{device:?}, transposed={transposed}, profile={pname}]: max |Δ| {max_abs:.3e} (rel {:.2e})",
        max_abs / scale
    );
    assert!(
        max_abs / scale < 1e-5,
        "{label}: GroupedMatMul on {device:?} (transposed={transposed}) is wrong — \
         max |Δ| {max_abs:.3e}, rel {:.2e}",
        max_abs / scale
    );
}

#[test]
fn grouped_matmul_contiguous_weights() {
    check(
        "grouped-matmul",
        dev(),
        false,
        &CompileProfile::default(),
        "default",
    );
    check(
        "grouped-matmul",
        dev(),
        false,
        &CompileProfile::llama32_prefill(),
        "llama32_prefill",
    );
}

#[test]
fn grouped_matmul_transposed_weights() {
    check(
        "grouped-matmul",
        dev(),
        true,
        &CompileProfile::default(),
        "default",
    );
    check(
        "grouped-matmul",
        dev(),
        true,
        &CompileProfile::llama32_prefill(),
        "llama32_prefill",
    );
}

// ── Composed expert-MLP: the exact shape `emit_deepseek_moe` emits ──
//
// Two GroupedMatMuls chained through a narrow+SwiGLU, repeated per top-k slot,
// with both slots sharing one transposed expert bank. Each piece passes in
// isolation; this checks them together.

const I2: usize = 5; // expert intermediate

fn composed_reference(
    x: &[f32],
    gate_up: &[f32], // [E, 2*I2, K] (HF order)
    down: &[f32],    // [E, K, I2]  (HF order)
    slots: &[(Vec<usize>, Vec<f32>)],
) -> Vec<f32> {
    let mut out = vec![0f64; M * K];
    for (idx, prob) in slots {
        for r in 0..M {
            let e = idx[r];
            let mut hid = [0f64; I2];
            for j in 0..I2 {
                let (mut g, mut u) = (0f64, 0f64);
                for i in 0..K {
                    g += x[r * K + i] as f64 * gate_up[e * 2 * I2 * K + j * K + i] as f64;
                    u += x[r * K + i] as f64 * gate_up[e * 2 * I2 * K + (I2 + j) * K + i] as f64;
                }
                hid[j] = (g / (1.0 + (-g).exp())) * u;
            }
            for o in 0..K {
                let mut acc = 0f64;
                for j in 0..I2 {
                    acc += hid[j] * down[e * K * I2 + o * I2 + j] as f64;
                }
                out[r * K + o] += acc * prob[r] as f64;
            }
        }
    }
    out.iter().map(|v| *v as f32).collect()
}

fn composed_run(
    device: Device,
    x: &[f32],
    gate_up: &[f32],
    down: &[f32],
    slots: &[(Vec<usize>, Vec<f32>)],
) -> Vec<f32> {
    let f = DType::F32;
    let mut g = Graph::new("composed_moe");
    let mut params: HashMap<String, Vec<f32>> = HashMap::new();

    let x_id = g.add_node(
        Op::Param { name: "x".into() },
        vec![],
        Shape::new(&[M, K], f),
    );
    params.insert("x".into(), x.to_vec());
    let gu = g.add_node(
        Op::Param { name: "gu".into() },
        vec![],
        Shape::new(&[E, 2 * I2, K], f),
    );
    params.insert("gu".into(), gate_up.to_vec());
    let dw = g.add_node(
        Op::Param { name: "dw".into() },
        vec![],
        Shape::new(&[E, K, I2], f),
    );
    params.insert("dw".into(), down.to_vec());
    // Same in-graph transposes the MoE builder does, shared across slots.
    let gu_t = g.transpose_(gu, vec![0, 2, 1]); // [E, K, 2*I2]
    let dw_t = g.transpose_(dw, vec![0, 2, 1]); // [E, I2, K]

    let mut acc: Option<NodeId> = None;
    for (si, (idx, prob)) in slots.iter().enumerate() {
        let iname = format!("idx{si}");
        let pname = format!("prob{si}");
        let idx_id = g.add_node(
            Op::Param {
                name: iname.clone(),
            },
            vec![],
            Shape::new(&[M], f),
        );
        params.insert(iname, idx.iter().map(|&e| e as f32).collect());
        let p_id = g.add_node(
            Op::Param {
                name: pname.clone(),
            },
            vec![],
            Shape::new(&[M, 1], f),
        );
        params.insert(pname, prob.clone());

        let gate_up_o = g.add_node(
            Op::GroupedMatMul,
            vec![x_id, gu_t, idx_id],
            Shape::new(&[M, 2 * I2], f),
        );
        let gg = g.narrow_(gate_up_o, 1, 0, I2);
        let uu = g.narrow_(gate_up_o, 1, I2, I2);
        let act = g.silu(gg);
        let hx = g.mul(act, uu);
        let down_o = g.add_node(
            Op::GroupedMatMul,
            vec![hx, dw_t, idx_id],
            Shape::new(&[M, K], f),
        );
        let w = g.mul(down_o, p_id);
        acc = Some(match acc {
            Some(a) => g.add(a, w),
            None => w,
        });
    }
    g.set_outputs(vec![acc.unwrap()]);
    let mut compiled = compile_graph_profile(device, g, params, &CompileProfile::llama32_prefill())
        .expect("compile composed moe");
    compiled.run(&[]).into_iter().next().expect("output")
}

#[test]
fn grouped_matmul_composed_expert_mlp() {
    let device = dev();
    let x = fill(M * K, 31);
    let gate_up = fill(E * 2 * I2 * K, 41);
    let down = fill(E * K * I2, 51);
    let slots: Vec<(Vec<usize>, Vec<f32>)> = (0..2)
        .map(|s| {
            let idx: Vec<usize> = (0..M).map(|r| (r + s * 2 + 1) % E).collect();
            let prob: Vec<f32> = (0..M).map(|r| 0.3 + 0.1 * (r + s) as f32).collect();
            (idx, prob)
        })
        .collect();

    // bisect: 1 slot vs 2 slots
    for n in [1usize, 2] {
        let sub = &slots[..n];
        let w2 = composed_reference(&x, &gate_up, &down, sub);
        let g2 = composed_run(device, &x, &gate_up, &down, sub);
        let m2 = g2
            .iter()
            .zip(&w2)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let s2 = w2.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
        eprintln!("  slots={n}: max |Δ| {m2:.3e} (rel {:.2e})", m2 / s2);
    }
    let want = composed_reference(&x, &gate_up, &down, &slots);
    let got = composed_run(device, &x, &gate_up, &down, &slots);
    let max_abs = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let scale = want.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
    eprintln!(
        "composed-expert-mlp [{device:?}]: max |Δ| {max_abs:.3e} (rel {:.2e})",
        max_abs / scale
    );
    assert!(
        max_abs / scale < 1e-5,
        "composed expert MLP on {device:?} is wrong — max |Δ| {max_abs:.3e}, rel {:.2e}",
        max_abs / scale
    );
}
