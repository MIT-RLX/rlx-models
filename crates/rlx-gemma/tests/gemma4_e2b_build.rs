// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// GPLv3 — see repository LICENSE.

//! Graph-build integration test for the Gemma 4 E2B mobile QAT checkpoint.
//!
//! Exercises the full path that turns the real safetensors checkpoint into an
//! rlx graph: the `GemmaQatLoader` (name remap + 2/4/8-bit dequant + grouped
//! embedding scale) feeding the Gemma builder with Per-Layer Embeddings,
//! KV-shared layers, double-wide MLP and the proportional RoPE fix. Confirms
//! every weight the builder requests resolves and all shapes line up — no
//! numeric parity here (that needs the per_layer_inputs precompute + a run).
//!
//! Skips when the checkpoint isn't present. Point at it with
//! `RLX_GEMMA4_E2B_DIR=/path/to/snapshot` or rely on the HF hub cache.

use std::collections::HashMap;
use std::path::PathBuf;

use rlx_gemma::config::GemmaConfig;
use rlx_gemma::qat_loader::GemmaQatLoader;

fn fixture_dir() -> Option<PathBuf> {
    if let Some(d) = std::env::var_os("RLX_GEMMA4_E2B_DIR") {
        let p = PathBuf::from(d);
        return p.join("config.json").is_file().then_some(p);
    }
    let home = std::env::var_os("HOME")?;
    let base = std::path::Path::new(&home).join(
        ".cache/huggingface/hub/\
         models--google--gemma-4-E2B-it-qat-mobile-transformers/snapshots",
    );
    let snap = std::fs::read_dir(&base).ok()?.flatten().next()?.path();
    snap.join("config.json").is_file().then_some(snap)
}

#[test]
fn gemma4_e2b_prefill_graph_builds_from_real_checkpoint() {
    let Some(dir) = fixture_dir() else {
        eprintln!("[gemma4 e2b build] checkpoint not found — skipping");
        return;
    };

    let cfg = GemmaConfig::from_file(&dir.join("config.json")).expect("parse config.json");
    // Sanity: this is the E2B mobile shape.
    assert!(cfg.has_ple(), "expected Per-Layer Embeddings");
    assert_eq!(cfg.num_hidden_layers, 35);
    assert_eq!(cfg.first_kv_shared_layer(), 15);
    assert!(cfg.use_double_wide_mlp);

    let mut loader = GemmaQatLoader::open(&dir).expect("open qat loader");

    let mut packed = HashMap::new();
    let (graph, params) = rlx_gemma::builder::build_gemma_graph_sized_packed_ext(
        &cfg,
        &mut loader,
        /*batch*/ 1,
        /*seq*/ 5,
        /*with_lm_head*/ true,
        /*last_token_from_input*/ false,
        /*with_kv_outputs*/ false,
        &mut packed,
        None,
        None,
    )
    .expect("build E2B prefill graph");

    assert!(
        graph.nodes().len() > 100,
        "graph too small: {}",
        graph.nodes().len()
    );
    assert!(!params.is_empty(), "no F32 params bound");
    // The per_layer_inputs graph input must exist for an E2B (PLE) build.
    assert!(
        graph.nodes().iter().any(|n| {
            matches!(&n.op, rlx_ir::op::Op::Input { name } if name == "per_layer_inputs")
        }),
        "per_layer_inputs input missing from PLE graph"
    );
    eprintln!(
        "[gemma4 e2b build] ok: {} nodes, {} f32 params",
        graph.nodes().len(),
        params.len()
    );
}

#[test]
fn gemma4_e2b_decode_graph_builds_from_real_checkpoint() {
    let Some(dir) = fixture_dir() else {
        eprintln!("[gemma4 e2b decode build] checkpoint not found — skipping");
        return;
    };
    let cfg = GemmaConfig::from_file(&dir.join("config.json")).expect("parse config.json");
    assert!(cfg.has_ple());
    let mut loader = GemmaQatLoader::open(&dir).expect("open qat loader");
    let mut packed = HashMap::new();
    // Decode bucket with some past context (past_seq=8).
    let (graph, params) = rlx_gemma::builder::build_gemma_decode_graph_sized_packed_ext(
        &cfg,
        &mut loader,
        /*batch*/ 1,
        /*past_seq*/ 8,
        /*use_custom_mask*/ false,
        &mut packed,
        None,
        None,
    )
    .expect("build E2B decode graph");

    assert!(graph.nodes().len() > 100);
    assert!(!params.is_empty());
    // The decode graph must expose per_layer_inputs (PLE) for E2B and NOT have
    // tried to load the absent k_norm for shared layers (would have errored).
    assert!(
        graph.nodes().iter().any(|n| {
            matches!(&n.op, rlx_ir::op::Op::Input { name } if name == "per_layer_inputs")
        }),
        "per_layer_inputs input missing from E2B decode graph"
    );
    eprintln!(
        "[gemma4 e2b decode build] ok: {} nodes, {} f32 params",
        graph.nodes().len(),
        params.len()
    );
}
