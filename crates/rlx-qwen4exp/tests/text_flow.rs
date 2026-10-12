// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! The text flow end to end on a tiny synthetic model, plus the two pieces
//! whose contracts are easiest to get wrong in isolation.
//!
//! The fixture is [`rlx_qwen4exp::synth`], which is also where
//! `tests/tensor_manifest.rs` checks the shapes against the real 1224-tensor
//! list. Using it here rather than a second hand-written fixture is deliberate:
//! two independent statements of the same tensor shapes drift, and when they do
//! the test fixture is the one that silently becomes wrong.

use rlx_core::flow_util::{WeightMapSource, compile_built};
use rlx_core::weight_map::WeightMap;
use rlx_flow::{MapWeights, ModelFlow};
use rlx_ir::{DType, Shape};
use rlx_qwen4exp::hc::{HcDims, emit_hc_combine, emit_hc_mix};
use rlx_qwen4exp::{
    DensePleTable, PleHash, Qwen4ExpConfig, build_qwen4exp_text_flow_with_source, ngram_rows,
    ple_embed_input, prepare_ple_embeddings, synth,
};
use rlx_runtime::Device;

fn dev() -> Device {
    std::env::var("RLX_TEST_DEVICE")
        .ok()
        .filter(|s| !s.is_empty())
        .map(|s| rlx_cli::parse_device(&s).expect("bad RLX_TEST_DEVICE"))
        .unwrap_or(Device::Cpu)
}

/// The tiny model's weights, minus the one tensor that is never a graph param.
fn weights(cfg: &Qwen4ExpConfig) -> WeightMap {
    let mut t = synth::synth_tensors(cfg, 7);
    t.remove("per_layer_token_embd.weight");
    WeightMap::from_tensors(t)
}

/// A table whose rows are a function of the row index, so a gathered row
/// identifies which row was gathered.
fn table(cfg: &Qwen4ExpConfig) -> DensePleTable {
    let rows = PleHash::derive(cfg, 0).padded_rows as usize;
    DensePleTable {
        data: (0..rows * cfg.ple_head_dim)
            .map(|i| i as f32 * 1e-4)
            .collect(),
        rows,
        head_dim: cfg.ple_head_dim,
    }
}

/// An EOS truncates the window behind it, and `prepare_ple_embeddings` turns
/// the resulting rows into the buffer the graph input wants.
#[test]
fn ple_hash_resets_at_eos_and_gathers_rows() {
    let c = synth::tiny_config(2048);
    let hash = PleHash::derive(&c, 0);
    let ids = [1, 2, c.ple_eos_token_id, 3];
    let rows = ngram_rows(&hash, &ids);
    assert_eq!(rows.len(), ids.len() * hash.n_heads);

    // Token 3's predecessor is the EOS, so it and everything older read as EOS.
    let eos = c.ple_eos_token_id as u64;
    let last = hash.rows(&[3, eos, eos]);
    assert_eq!(&rows[3 * hash.n_heads..], last.as_slice());

    let mut tbl = table(&c);
    let gathered = prepare_ple_embeddings(&c, 0, &ids, &mut tbl).unwrap();
    assert_eq!(gathered.len(), ids.len() * c.ple_embed_dim());
}

/// The whole stack: three GatedDeltaNet layers (one carrying the PLE module)
/// and one gated-attention layer, with the MoE at every site.
#[test]
fn dense_and_indexed_layers_build_and_run_together() {
    let c = synth::tiny_config(2048);
    let seq = 4;
    assert!(c.qsa_is_dense(seq), "this test exercises the short-circuit");

    let mut w = weights(&c);
    let built = build_qwen4exp_text_flow_with_source(&c, &mut WeightMapSource(&mut w), seq, true)
        .expect("build text flow");
    let leftovers: Vec<String> = w.keys().map(str::to_string).collect();
    assert!(
        leftovers.iter().all(|n| n.contains(".indexer.")),
        "the flow left model weights unread: {leftovers:?}"
    );

    let ids: Vec<f32> = (0..seq).map(|t| (t + 1) as f32).collect();
    let ple: Vec<f32> = (0..seq * c.ple_embed_dim())
        .map(|i| (i as f32 * 0.03).sin() * 0.1)
        .collect();
    let out = compile_built(built, dev())
        .expect("compile text flow")
        .run(&[
            ("input_ids", ids.as_slice()),
            (&ple_embed_input(0), ple.as_slice()),
        ])
        .pop()
        .expect("logits");
    assert_eq!(out.len(), seq * c.vocab_size);
    assert!(out.iter().all(|v| v.is_finite()));
}

/// Past the budget the indexer genuinely selects, so the layer takes the bias
/// path instead of the fused causal one — and must still run.
#[test]
fn indexed_selection_path_builds_past_the_dense_budget() {
    // budget 4, ratio 4 → one block fits; seq 8 has two.
    let c = synth::tiny_config(4);
    let seq = 8;
    assert!(!c.qsa_is_dense(seq));

    let mut w = weights(&c);
    let built = build_qwen4exp_text_flow_with_source(&c, &mut WeightMapSource(&mut w), seq, false)
        .expect("build indexed flow");
    // With the selection live, the indexer weights ARE read — unlike the dense
    // case above. Only the LM head is left, because this build has none.
    let leftovers: Vec<String> = w.keys().map(str::to_string).collect();
    assert_eq!(leftovers, vec!["output.weight".to_string()]);

    let ids: Vec<f32> = (0..seq).map(|t| (t + 1) as f32).collect();
    let ple = vec![0.01; seq * c.ple_embed_dim()];
    let out = compile_built(built, dev())
        .expect("compile indexed flow")
        .run(&[
            ("input_ids", ids.as_slice()),
            (&ple_embed_input(0), ple.as_slice()),
        ])
        .pop()
        .expect("hidden");
    assert_eq!(out.len(), seq * c.hidden_size);
    assert!(out.iter().all(|v| v.is_finite()));
}

/// One hyper-connection site in isolation, on hand-written weights — the mix
/// and the scatter are a few dozen tiny nodes and worth exercising without the
/// rest of the stack in the way.
#[test]
fn hyper_connection_mix_and_injection_execute() {
    let (seq, hidden, streams) = (2usize, 4usize, 2usize);
    let width = hidden * streams;
    let dims = HcDims {
        hidden,
        streams,
        lowrank: 3,
        norm_eps: 1e-5,
        norm_offset: false,
        seq,
    };
    let mut w = MapWeights::default();
    w.insert("site_norm.weight", vec![1.0; width], vec![width]);
    w.insert("site_down.weight", vec![0.02; 3 * width], vec![3, width]);
    w.insert("site_up.weight", vec![0.03; width * 3], vec![width, 3]);
    w.insert(
        "site_inject.weight",
        vec![0.01; streams * width],
        vec![streams, width],
    );

    let out_shape = Shape::new(&[1, seq, streams, hidden], DType::F32);
    let flow = ModelFlow::new("qwen4exp_hc_test")
        .input("x", out_shape.clone())
        .plugin_named("hc", move |emit, _| {
            let x = emit.flow_input("x")?.hir_id();
            let gates = emit_hc_mix(emit, "site", x, dims, true)?;
            let y = emit_hc_combine(emit, gates.mixed, x, gates, dims);
            Ok(Some(emit.wrap(y, out_shape.clone())))
        })
        .output("y");

    let built = flow.build_with(&mut w, None).unwrap();
    let input: Vec<f32> = (0..seq * width).map(|i| i as f32 * 0.05).collect();
    let out = compile_built(built, dev())
        .unwrap()
        .run(&[("x", input.as_slice())])
        .pop()
        .unwrap();
    assert_eq!(out.len(), input.len());
    assert!(out.iter().all(|v| v.is_finite()));
}

/// A site built without a scatter has no `inject` to scatter with, and must say
/// so rather than invent one.
#[test]
fn combining_a_site_with_no_injection_panics() {
    let dims = HcDims {
        hidden: 4,
        streams: 2,
        lowrank: 3,
        norm_eps: 1e-5,
        norm_offset: false,
        seq: 1,
    };
    let width = dims.width();
    let mut w = MapWeights::default();
    w.insert("site_norm.weight", vec![1.0; width], vec![width]);
    w.insert("site_down.weight", vec![0.02; 3 * width], vec![3, width]);
    w.insert("site_up.weight", vec![0.03; width * 3], vec![width, 3]);

    let shape = Shape::new(&[1, 1, dims.streams, dims.hidden], DType::F32);
    let flow = ModelFlow::new("qwen4exp_hc_no_inject")
        .input("x", shape.clone())
        .plugin_named("hc", move |emit, _| {
            let x = emit.flow_input("x")?.hir_id();
            let gates = emit_hc_mix(emit, "site", x, dims, false)?;
            assert!(gates.inject.is_none(), "no `_inject` tensor was read");
            // The final mixer is the only `with_inject = false` site, and it
            // never combines; doing so is a programming error, not a bad input.
            let y = emit_hc_combine(emit, gates.mixed, x, gates, dims);
            Ok(Some(emit.wrap(y, shape.clone())))
        })
        .output("y");
    let err = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        flow.build_with(&mut w, None)
    }));
    assert!(err.is_err(), "combining without an injection must panic");
}

/// The flow refuses what it cannot build instead of emitting something close.
#[test]
fn unsupported_mtp_and_invalid_lengths_are_rejected() {
    let mut c = synth::tiny_config(2048);
    c.num_nextn_predict_layers = 1;
    c.with_mtp = true;
    let mut w = weights(&c);
    let err = build_qwen4exp_text_flow_with_source(&c, &mut WeightMapSource(&mut w), 4, false)
        .unwrap_err()
        .to_string();
    assert!(err.contains("MTP"), "{err}");

    c.with_mtp = false;
    let mut w = weights(&c);
    assert!(
        build_qwen4exp_text_flow_with_source(&c, &mut WeightMapSource(&mut w), 0, false).is_err(),
        "seq = 0 must be rejected"
    );
}
