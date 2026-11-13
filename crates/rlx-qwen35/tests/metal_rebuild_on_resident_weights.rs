// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! A second shape of a Qwen3.5 graph, built from weights whose large
//! projections were cleared (`clear_dense_f32_projections_at_least`), compiled
//! under the sharing scope of a live first executable: the cleared params map
//! onto that executable's pages, so it needs neither the checkpoint again nor
//! an upload of them — and computes what a build from the full weights does.

#![cfg(feature = "metal")]

use std::collections::HashMap;
use std::sync::Arc;

use rlx_qwen35::{Qwen35Config, Qwen35Weights};
use rlx_runtime::{CompileOptions, CompiledGraph, Device, Session};

fn build(cfg: &Qwen35Config, w: &Arc<Qwen35Weights>, seq: usize) -> (rlx_ir::Graph, HashMap<String, Vec<f32>>) {
    let (built, _) = rlx_qwen35::Qwen35Flow::new(cfg)
        .shared_weights(w.clone())
        .batch(1)
        .seq(seq)
        .runtime_mrope()
        .lm_head()
        .build_with_weights(w)
        .expect("build");
    rlx_core::flow_util::graph_from_built(built).expect("graph")
}

fn compile(g: rlx_ir::Graph, scope: &str) -> CompiledGraph {
    let mut opts = CompileOptions::default();
    opts.fusion_opts.no_weight_concat = true;
    rlx_runtime::with_param_sharing_scope(scope, || Session::new(Device::Metal).compile_with(g, &opts))
}

fn run(c: &mut CompiledGraph, g_inputs: &[&str], cfg: &Qwen35Config, seq: usize) -> Vec<f32> {
    let hidden: Vec<f32> = (0..seq * cfg.hidden_size).map(|i| ((i % 13) as f32 - 6.0) * 0.05).collect();
    let ids: Vec<f32> = (0..seq).map(|i| i as f32).collect();
    let (cos, sin) = rlx_qwen35::mrope_prefill_feeds(cfg, seq, None, cfg.rope_dim_count / 2);
    let feeds: Vec<(&str, &[f32])> = g_inputs
        .iter()
        .map(|&n| {
            let v: &[f32] = match n {
                "hidden" => &hidden,
                "input_ids" => &ids,
                "rope_cos" => &cos,
                "rope_sin" => &sin,
                other => panic!("unexpected input {other}"),
            };
            (n, v)
        })
        .collect();
    c.run(&feeds).swap_remove(0)
}

fn inputs(g: &rlx_ir::Graph) -> Vec<&'static str> {
    ["hidden", "input_ids", "rope_cos", "rope_sin"].into_iter().filter(|n| g.input_id(n).is_some()).collect()
}

#[test]
fn rebuild_on_resident_weights_matches_a_full_build() {
    if !rlx_runtime::is_available(Device::Metal) {
        eprintln!("no Metal device; skipping");
        return;
    }
    // Split however few bytes leave the arena: this model is small.
    rlx_ir::env::set("RLX_METAL_WEIGHT_SPLIT_MIN", "1");
    let mut cfg = rlx_qwen35::synth::tiny_cfg();
    cfg.hidden_size = 256;
    cfg.intermediate_size = 512;
    let full = rlx_qwen35::synth::synth_weights(&cfg);
    // The FFN matrices: large enough to leave the arena, so they are shared.
    let mut cleared = full.clone();
    cleared.clear_dense_f32_projections_at_least(cfg.hidden_size * cfg.intermediate_size);
    let (full, cleared) = (Arc::new(full), Arc::new(cleared));
    let scope = format!("qwen35-rebuild:{}", std::process::id());

    let (g, params) = build(&cfg, &full, 8);
    let g_in = inputs(&g);
    let mut first = compile(g, &scope);
    for (n, v) in &params {
        first.set_param(n, v);
    }

    // Another shape, as rlx-reconstruct recompiles per case length.
    let seq = 12;
    let (g, params) = build(&cfg, &cleared, seq);
    let (g_full, params_full) = build(&cfg, &full, seq);
    let mut names: Vec<&String> = params.keys().collect();
    let mut names_full: Vec<&String> = params_full.keys().collect();
    names.sort();
    names_full.sort();
    assert_eq!(names, names_full, "a cleared build declares other params");
    let shells: Vec<&str> = params.iter().filter(|(_, v)| v.is_empty()).map(|(n, _)| n.as_str()).collect();
    assert!(!shells.is_empty(), "nothing was cleared");

    let mut again = compile(g, &scope);
    for (n, v) in &params {
        if v.is_empty() {
            assert!(again.param_already_set(n), "cleared `{n}` is not resident");
        } else {
            again.set_param(n, v);
        }
    }
    again.allow_unbound(&shells);
    // The mapped pages outlive the executable that made them.
    drop(first);
    let got = run(&mut again, &g_in, &cfg, seq);

    let mut fresh = compile(g_full, &format!("qwen35-rebuild-fresh:{}", std::process::id()));
    for (n, v) in &params_full {
        fresh.set_param(n, v);
    }
    let want = run(&mut fresh, &g_in, &cfg, seq);
    rlx_ir::env::unset("RLX_METAL_WEIGHT_SPLIT_MIN");

    assert_eq!(got.len(), want.len());
    let d = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    assert!(d < 1e-4, "{} cleared params; logits off by {d}", shells.len());
    assert!(want.iter().any(|v| v.abs() > 1e-3), "logits are all zero");
}
