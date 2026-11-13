// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! A Qwen3.5 forward and its backward w.r.t. the input hidden states, compiled
//! under one sharing scope on Metal, must hold one copy of the weights: the
//! backward's weights all mapped from the forward's, and none of them — nor
//! their transposes for `dX = dY·Wᵀ` — in its activation arena.
//!
//! The backward used to keep every `Wᵀ` in its arena, which also stopped the
//! weight split there: 0.8B's backward arena was 10.4 GB against 0.84 GB now,
//! and 4B did not fit. Nothing fails when that comes back — it only costs
//! memory — so this checks [`rlx_runtime::CompiledGraph::memory_stats`].

#![cfg(feature = "metal")]

use rlx_opt::rlx_autodiff::{GradWithLossOptions, Wrt, grad_with_loss_wrt};
use rlx_runtime::{CompileOptions, Device, MemoryStats, Session};

fn graphs(seq: usize) -> (rlx_ir::Graph, rlx_ir::Graph) {
    // Wide enough that the FFN weights (256×512, 512 KiB) pass the size at
    // which a weight leaves the activation arena.
    let mut cfg = rlx_qwen35::synth::tiny_cfg();
    cfg.hidden_size = 256;
    cfg.intermediate_size = 512;
    let w = std::sync::Arc::new(rlx_qwen35::synth::synth_weights(&cfg));
    let (built, _) = rlx_qwen35::Qwen35Flow::new(&cfg)
        .shared_weights(w.clone())
        .batch(1)
        .seq(seq)
        .runtime_mrope()
        .lm_head()
        .build_with_weights(&w)
        .expect("forward");
    let (fwd, _) = rlx_core::flow_util::graph_from_built(built).expect("graph");
    let bwd = grad_with_loss_wrt(
        &fwd,
        &[Wrt::Leaf("hidden".into())],
        GradWithLossOptions::STRICT.with_aux(false),
    );
    (fwd, bwd)
}

/// Forward and backward stats, compiled the way rlx-reconstruct does.
fn compile(scope: &str) -> (MemoryStats, MemoryStats) {
    let (fwd, bwd) = graphs(8);
    let mut opts = CompileOptions::default();
    opts.fusion_opts.no_weight_concat = true;
    let (f, b) = rlx_runtime::with_param_sharing_scope(scope, || {
        (
            Session::new(Device::Metal).compile_with(fwd, &opts),
            Session::new(Device::Metal).compile_with(bwd, &opts),
        )
    });
    (
        f.memory_stats().expect("stats"),
        b.memory_stats().expect("stats"),
    )
}

#[test]
fn backward_holds_no_copy_of_the_weights() {
    if !rlx_runtime::is_available(Device::Metal) {
        eprintln!("no Metal device; skipping");
        return;
    }
    // Split however few bytes leave the arena: this model is small.
    rlx_ir::env::set("RLX_METAL_WEIGHT_SPLIT_MIN", "1");
    let pid = std::process::id();
    let (fwd, bwd) = compile(&format!("qwen35-bwd:{pid}"));
    rlx_ir::env::set("RLX_METAL_NO_WEIGHT_TRANSPOSE_FOLD", "1");
    let (_, copied) = compile(&format!("qwen35-bwd-copied:{pid}"));
    rlx_ir::env::unset("RLX_METAL_NO_WEIGHT_TRANSPOSE_FOLD");
    rlx_ir::env::unset("RLX_METAL_WEIGHT_SPLIT_MIN");
    eprintln!("forward {fwd:?}\nbackward {bwd:?}\nbackward, transposes copied {copied:?}");

    assert!(
        fwd.weight_params > 0,
        "the forward's weights never left its arena"
    );
    assert!(
        bwd.folded_weight_transposes > 0,
        "no weight transpose was folded"
    );
    assert!(
        bwd.weight_params >= fwd.weight_params,
        "backward keeps weights in its arena: {} in its weight buffer, {} in the forward's",
        bwd.weight_params,
        fwd.weight_params
    );
    assert_eq!(
        bwd.shared_weight_params, fwd.weight_params,
        "backward has weights of its own instead of the forward's"
    );
    assert!(
        bwd.arena_bytes < copied.arena_bytes,
        "folding saved no arena ({} vs {} B)",
        bwd.arena_bytes,
        copied.arena_bytes
    );
}
