// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Every parameter gradient of the trunk must agree across devices.
//!
//! A backend can get the whole FORWARD bit-identical and still return garbage
//! gradients, and nothing in a training run will say so: the loss looks fine,
//! it just stops moving. Metal did exactly that here — 32 of 54 parameters came
//! back NaN or zero while the loss matched CPU to the last bit — and it cost a
//! 1529-step `rlx-kev` reproduction, which completed with its loss sitting on
//! `ln(n_options)`, the chance floor.
//!
//! Two causes, both invisible to a per-op test because every op was correct in
//! isolation (`GatedDeltaNetBackward`, the RmsNorm backwards and
//! `AttentionBackward` all matched CPU to ~1e-7 at these dims):
//!
//! 1. Metal's **thunk-level** fusions (`fuse_gdn_gated_norm`, `fuse_l2_norm`,
//!    the decode-MLP ones) dropped the producer of an intermediate after
//!    checking only that it was not a graph output. Their liveness scan stops
//!    at the fused window — deliberately, since a full scan hits an opaque
//!    thunk and rejects — so a reader AFTER the window was invisible. A forward
//!    graph has none; a backward graph re-reads the recomputed forward values,
//!    which then read an unwritten slot.
//! 2. The conv weight gradient took a per-group im2col+GEMM path at `N == 1`
//!    that returned a history-dependent answer (right on a fresh arena, wrong
//!    once the surrounding memory was dirty).
//!
//! That combination is why this has to be a whole-trunk test rather than a set
//! of per-op ones.
//!
//! Scaled down to `seq = 4` on `synth::tiny_cfg`, so it runs in ~2 s.

#![cfg(feature = "metal")]

use rlx_ir::NodeId;
use rlx_ir::infer::GraphExt;
use rlx_opt::rlx_autodiff::grad_with_loss;
use rlx_runtime::{Device, Session};

/// Trunk forward reduced to a scalar, differentiated w.r.t. every `Param`.
type Params = std::collections::HashMap<String, Vec<f32>>;

fn backward_graph(seq: usize) -> (rlx_ir::Graph, Params, Vec<String>) {
    let cfg = rlx_qwen35::synth::tiny_cfg();
    let w = rlx_qwen35::synth::synth_weights(&cfg);
    let (fwd, params, _) = rlx_qwen35::build_qwen35_graph_sized_ext(
        &cfg, w, 1, seq, false, false, false, false, None, true, false, true, false,
    )
    .expect("forward");
    let mut g = fwd;
    let out = g.outputs[0];
    let flat = g.reshape_(out, vec![-1]);
    let loss = g.mean(flat, vec![0], false);
    g.set_outputs(vec![loss]);
    let wrt: Vec<NodeId> = g
        .nodes()
        .iter()
        .filter(|n| matches!(n.op, rlx_ir::Op::Param { .. }))
        .map(|n| n.id)
        .collect();
    let names: Vec<String> = wrt
        .iter()
        .map(|id| match &g.node(*id).op {
            rlx_ir::Op::Param { name } => name.clone(),
            _ => unreachable!(),
        })
        .collect();
    assert!(!wrt.is_empty(), "the trunk has no parameters");
    (grad_with_loss(&g, &wrt), params, names)
}

#[test]
fn metal_trunk_gradients_match_cpu() {
    if !rlx_runtime::is_available(Device::Metal) {
        eprintln!("no Metal device; skipping");
        return;
    }
    let seq = 4usize;
    let cfg = rlx_qwen35::synth::tiny_cfg();
    let (bwd, params, names) = backward_graph(seq);
    let ids: Vec<f32> = (0..seq).map(|i| (i % 20 + 1) as f32).collect();
    let (cos, sin) = rlx_qwen35::mrope_prefill_feeds(&cfg, seq, None, cfg.rope_dim_count / 2);
    let run = |d: Device| {
        let mut c = Session::new(d).compile(bwd.clone());
        for (n, v) in &params {
            c.set_param(n, v);
        }
        c.run(&[
            ("input_ids", &ids[..]),
            ("rope_cos", &cos[..]),
            ("rope_sin", &sin[..]),
            ("d_output", &[1.0f32][..]),
        ])
    };
    let cpu = run(Device::Cpu);
    let met = run(Device::Metal);

    // The forward is the control: if it already differs, the gradient
    // comparison below says nothing about the backward.
    let dl = (cpu[0][0] - met[0][0]).abs();
    assert!(
        dl <= 1e-5 * cpu[0][0].abs().max(1.0),
        "the forward already differs (cpu {} vs metal {}), so this test cannot \
         attribute anything to the backward",
        cpu[0][0],
        met[0][0]
    );

    let mut bad = Vec::new();
    for (i, name) in names.iter().enumerate() {
        let (a, b) = (&cpu[i + 1], &met[i + 1]);
        let err = a
            .iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        let scale = a.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-12);
        let nonfinite = b.iter().filter(|v| !v.is_finite()).count();
        if nonfinite > 0 || err / scale > 1e-3 {
            bad.push(format!(
                "{name} (rel {:.2e}, {nonfinite}/{} non-finite)",
                err / scale,
                b.len()
            ));
        }
    }
    assert!(
        bad.is_empty(),
        "forward matches to {dl:.2e} but {} of {} gradients disagree on Metal:\n  {}",
        bad.len(),
        names.len(),
        bad.join("\n  ")
    );
}
