// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0-only

//! Where the time goes in one Pestle projection, at the real 27B dimensions.
//!
//! `emit_linear` expands each projection into five nodes —
//! `mul → DequantMatMul → mul → DequantMatMul → mul` — so a 63-block trunk
//! issues ~5× the dispatches a dense trunk does. This isolates the cost of
//! the three per-channel scale multiplies from the two matmuls they wrap, so
//! kernel work can be aimed at whichever actually dominates rather than at
//! whichever is easier to fuse.
//!
//! Not a correctness test; ignored by default because it wants a GPU and
//! allocates real-size weights.
//!
//! ```sh
//! cargo test -p rlx-qwen35 --release --features metal \
//!   --test pestle_chain_bench -- --ignored --nocapture
//! ```

use rlx_ir::quant::QuantScheme;
use rlx_ir::*;
use rlx_runtime::{Device, Session};
use std::time::Instant;

/// A square 27B-scale slot (`attn_output` / `ssm_out` shape) so projections
/// chain end-to-end, and `REPS` of them per graph so the fixed command-buffer
/// submit — ~0.5 ms on Metal, which swamps a 5-node graph — amortizes away and
/// the per-dispatch cost is what is actually being measured.
const N_EMBD: usize = 5120;
const RANK: usize = 3968;
const REPS: usize = 32;
const ITERS: usize = 20;
const TRIALS: usize = 5;

fn q2_0_weight(rows: usize, cols: usize) -> Vec<u8> {
    let w: Vec<f32> = (0..rows * cols)
        .map(|i| [-0.5f32, 0.0, 0.5, 1.0][i % 4])
        .collect();
    rlx_gguf::q2_dequant::quantize_q2_0(&w).expect("quantize Q2_0")
}

struct Case {
    name: &'static str,
    dispatches: usize,
    ms: f64,
}

#[allow(clippy::too_many_arguments)]
fn time_graph(
    device: Device,
    name: &'static str,
    per_rep_dispatches: usize,
    build: &dyn Fn(&mut Graph, NodeId, usize) -> NodeId,
    u8_params: &[(&str, &[u8])],
    f32_params: &[(&str, &[f32])],
) -> Case {
    let mut g = Graph::new(name);
    let mut cur = g.input("x", Shape::new(&[1, N_EMBD], DType::F32));
    for r in 0..REPS {
        cur = build(&mut g, cur, r);
    }
    g.set_outputs(vec![cur]);
    let mut c = Session::new(device).compile(g);
    for (n, b) in u8_params {
        c.set_param_typed(n, b, DType::U8);
    }
    for (n, v) in f32_params {
        c.set_param(n, v);
    }
    let xs = vec![0.01f32; N_EMBD];
    for _ in 0..3 {
        c.run(&[("x", xs.as_slice())]); // warm
    }
    // Min over trials, not mean: GPU microbenchmarks are contaminated upward
    // by scheduler noise and thermal/clock excursions, never downward, so the
    // minimum is the stable estimator. Mean-of-N here swung 40-120 GB/s run
    // to run on the same kernel.
    let mut best = f64::INFINITY;
    for _ in 0..TRIALS {
        let t = Instant::now();
        for _ in 0..ITERS {
            c.run(&[("x", xs.as_slice())]);
        }
        best = best.min(t.elapsed().as_secs_f64() * 1000.0 / ITERS as f64);
    }
    Case {
        name,
        dispatches: per_rep_dispatches * REPS,
        ms: best,
    }
}

#[test]
#[ignore = "benchmark: needs a GPU and allocates 27B-scale weights"]
fn pestle_chain_cost_breakdown() {
    let device = if rlx_runtime::is_available(Device::Metal) {
        Device::Metal
    } else if rlx_runtime::is_available(Device::Cuda) {
        Device::Cuda
    } else {
        eprintln!("skip: no GPU");
        return;
    };

    let v_bytes = q2_0_weight(RANK, N_EMBD); // [rank, in]
    let u_bytes = q2_0_weight(N_EMBD, RANK); // [out, rank]
    let dense_bytes = q2_0_weight(N_EMBD, N_EMBD); // the linear it replaces
    let pre = vec![1.0001f32; N_EMBD];
    let mid = vec![1.0001f32; RANK];
    let post = vec![1.0001f32; N_EMBD];

    fn dq(g: &mut Graph, x: NodeId, w: NodeId, n: usize) -> NodeId {
        g.add_node(
            Op::DequantMatMul {
                scheme: QuantScheme::GgufQ2_0,
            },
            vec![x, w],
            Shape::new(&[1, n], DType::F32),
        )
    }
    fn mul(g: &mut Graph, a: NodeId, b: NodeId, n: usize) -> NodeId {
        g.add_node(
            Op::Binary(rlx_ir::op::BinaryOp::Mul),
            vec![a, b],
            Shape::new(&[1, n], DType::F32),
        )
    }

    let cases = vec![
        // The dense linear Pestle replaces — the floor.
        time_graph(
            device,
            "dense Q2_0 [5120->5120]",
            1,
            &|g, x, _| {
                let w = g.param("w", Shape::new(&[dense_bytes.len()], DType::U8));
                dq(g, x, w, N_EMBD)
            },
            &[("w", &dense_bytes)],
            &[],
        ),
        // The two factor matmuls alone.
        time_graph(
            device,
            "V,U matmuls only",
            2,
            &|g, x, _| {
                let v = g.param("v", Shape::new(&[v_bytes.len()], DType::U8));
                let u = g.param("u", Shape::new(&[u_bytes.len()], DType::U8));
                let h = dq(g, x, v, RANK);
                dq(g, h, u, N_EMBD)
            },
            &[("v", &v_bytes), ("u", &u_bytes)],
            &[],
        ),
        // The full chain as `emit_linear` builds it today.
        time_graph(
            device,
            "full pestle chain",
            5,
            &|g, x, _| {
                let v = g.param("v", Shape::new(&[v_bytes.len()], DType::U8));
                let u = g.param("u", Shape::new(&[u_bytes.len()], DType::U8));
                let sp = g.param("sp", Shape::new(&[N_EMBD], DType::F32));
                let sm = g.param("sm", Shape::new(&[RANK], DType::F32));
                let so = g.param("so", Shape::new(&[N_EMBD], DType::F32));
                let x1 = mul(g, sp, x, N_EMBD);
                let h = dq(g, x1, v, RANK);
                let h2 = mul(g, sm, h, RANK);
                let y = dq(g, h2, u, N_EMBD);
                mul(g, so, y, N_EMBD)
            },
            &[("v", &v_bytes), ("u", &u_bytes)],
            &[("sp", &pre), ("sm", &mid), ("so", &post)],
        ),
    ];

    eprintln!("\n{device:?}, m=1 (decode), {REPS} chained projections x {ITERS} iters");
    eprintln!(
        "{:<28} {:>6} {:>10} {:>12}  vs dense",
        "case", "disp", "ms", "us/disp"
    );
    let base = cases[0].ms;
    for c in &cases {
        eprintln!(
            "{:<28} {:>6} {:>10.3} {:>12.1}  {:.2}x",
            c.name,
            c.dispatches,
            c.ms,
            c.ms * 1000.0 / c.dispatches as f64,
            c.ms / base
        );
    }
    let matmuls = cases[1].ms;
    let full = cases[2].ms;
    eprintln!(
        "\nthe 3 scale multiplies cost {:.3} ms ({:.0}% of the chain), \
         {:.1} us each",
        full - matmuls,
        100.0 * (full - matmuls) / full,
        (full - matmuls) * 1000.0 / (3 * REPS) as f64
    );
}

/// Effective GEMV bandwidth vs shape, to locate the cliff the factor
/// matmuls fall off.
///
/// Pestle's ranks are not round: 3968 = 31×128, an **odd** number of Q2_0
/// blocks, where 5120 = 40×128 is even. If the simdgroup GEMV loses a
/// vectorized path on odd block counts that would explain the factors
/// running at half the dense rate despite moving only 1.55× the bytes.
#[test]
#[ignore = "benchmark: needs a GPU"]
fn q2_0_gemv_bandwidth_vs_shape() {
    let device = if rlx_runtime::is_available(Device::Metal) {
        Device::Metal
    } else if rlx_runtime::is_available(Device::Cuda) {
        Device::Cuda
    } else {
        eprintln!("skip: no GPU");
        return;
    };

    // (k, n) — k is the reduction length, n the output width.
    const SHAPES: &[(usize, usize)] = &[
        (5120, 5120), // 40 blocks, the dense reference
        (5120, 3968), // V: even k-blocks, ragged n
        (3968, 5120), // U: 31 k-blocks (odd)
        (4096, 5120), // 32 k-blocks (even), nearest round rank
        (3968, 3968), // odd k-blocks and ragged n
        (4096, 4096), // fully round
    ];

    eprintln!("\n{device:?} Q2_0 GEMV, m=1, {REPS} chained x {ITERS} iters");
    eprintln!(
        "{:>6} {:>6} {:>8} {:>10} {:>10} {:>10}",
        "k", "n", "k/128", "MB", "us/disp", "GB/s"
    );
    for &(k_dim, n_dim) in SHAPES {
        // Chain by alternating k->n and n->k so shapes compose.
        let w_a = q2_0_weight(n_dim, k_dim);
        let _w_b = q2_0_weight(k_dim, n_dim);
        let bytes = w_a.len();
        let c = time_graph(
            device,
            "shape",
            2,
            &|g, x, r| {
                let wa = g.param(format!("wa{r}"), Shape::new(&[bytes], DType::U8));
                let wb = g.param(format!("wb{r}"), Shape::new(&[bytes], DType::U8));
                let h = g.add_node(
                    Op::DequantMatMul {
                        scheme: QuantScheme::GgufQ2_0,
                    },
                    vec![x, wa],
                    Shape::new(&[1, n_dim], DType::F32),
                );
                g.add_node(
                    Op::DequantMatMul {
                        scheme: QuantScheme::GgufQ2_0,
                    },
                    vec![h, wb],
                    Shape::new(&[1, k_dim], DType::F32),
                )
            },
            &[],
            &[],
        );
        // Params are per-rep here; set them after compile is not possible with
        // the shared helper, so this measures the same weights uploaded once —
        // fine, the point is per-dispatch timing, not numerics.
        let us = c.ms * 1000.0 / c.dispatches as f64;
        eprintln!(
            "{:>6} {:>6} {:>8} {:>10.2} {:>10.1} {:>10.1}",
            k_dim,
            n_dim,
            k_dim / 128,
            bytes as f64 / 1e6,
            us,
            bytes as f64 / (us * 1e-6) / 1e9
        );
    }
}
