// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! End-to-end `glm5next` text flow on a tiny synthetic checkpoint.
//!
//! GLM-5.3-Flash is 320 B parameters and its smallest published quantization is
//! 93 GB, so the whole model cannot be run here. What *can* be checked is that
//! the graph closes over the real architecture: both attention kinds, both FFN
//! kinds, mHC at every site, and the DSA indexer in both its dense and its
//! actually-selecting regime. The fixture lives in [`common`].

mod common;

use common::{
    HEADS, HIDDEN, IDX_HD, IDX_HEADS, KPOOL, KV_LORA, NOPE, Q_LORA, VOCAB, cfg, dev, weights,
};
use rlx_core::flow_util::{WeightMapSource, compile_built};
use rlx_flow::{CompileProfile, ModelFlow};
use rlx_glm5next::IndexerDims;
use rlx_glm5next::mla::{MlaDims, emit_mla_attention};
use rlx_glm5next::{Glm5NextConfig, build_glm5next_text_flow};
use rlx_ir::{DType, Shape};

fn run(c: &Glm5NextConfig, seq: usize) -> Vec<f32> {
    let mut w = weights(c);
    let built = build_glm5next_text_flow(c, &mut w, seq, true).expect("build flow");
    let mut compiled = compile_built(built, dev()).expect("compile");
    let ids: Vec<f32> = (0..seq).map(|i| (i % VOCAB) as f32).collect();
    let mut outs = compiled.run(&[("input_ids", ids.as_slice())]);
    assert_eq!(outs.len(), 1, "the flow declares exactly one output");
    outs.pop().expect("logits")
}

/// `index_topk` well above `seq`: the DSA selection is the identity, so the
/// layer takes the fused causal-attention path.
#[test]
fn prefill_is_finite_with_dense_dsa() {
    let c = cfg(2048);
    let seq = 12;
    let logits = run(&c, seq);
    assert_eq!(logits.len(), seq * VOCAB);
    assert!(
        logits.iter().all(|v| v.is_finite()),
        "logits must be finite"
    );
    assert!(
        logits.iter().any(|v| v.abs() > 1e-6),
        "logits must not be all-zero"
    );
}

/// `index_topk` below `seq`: the indexer actually scores, selects and scatters
/// a sparse mask, so the whole k-pool path is exercised.
#[test]
fn prefill_is_finite_with_sparse_dsa() {
    let c = cfg(8); // budget = 8/4 = 2 pools out of 3
    let seq = 12;
    let logits = run(&c, seq);
    assert_eq!(logits.len(), seq * VOCAB);
    assert!(
        logits.iter().all(|v| v.is_finite()),
        "a sparse DSA mask must not produce NaN — every query keeps at least \
         its own tail visible"
    );
}

/// A sequence that is not a multiple of `index_kpool` leaves an incomplete tail
/// pool, which is the branch `index_kpool_always_select_tail` exists for.
#[test]
fn prefill_handles_a_ragged_tail() {
    let c = cfg(8);
    let seq = 11; // 2 complete pools + 3 tail tokens
    let logits = run(&c, seq);
    assert_eq!(logits.len(), seq * VOCAB);
    assert!(logits.iter().all(|v| v.is_finite()));
}

/// The dense-DSA short-circuit must be a pure optimization: running the
/// indexer must give the same answer as skipping it.
///
/// This needs `IndexerDims::force_emit`, and that is the whole point. Selection
/// is the identity *exactly when* `is_dense()` holds, so no choice of
/// `index_topk` puts the emitter in a state where the machinery runs and
/// provably selects everything — comparing two configs that both short-circuit
/// compares nothing. Forcing the emission is the only way to actually exercise
/// the pooling, scoring, top-k and scatter against the causal mask they are
/// supposed to reproduce.
#[test]
fn dsa_selection_is_the_identity_below_the_budget() {
    let c = cfg(2048);
    let seq = 12;
    let mla = MlaDims {
        hidden: HIDDEN,
        num_heads: HEADS,
        q_lora_rank: Q_LORA,
        kv_lora_rank: KV_LORA,
        qk_nope_head_dim: NOPE,
        v_head_dim: NOPE,
        eps: c.rms_norm_eps,
        seq,
    };
    let base = IndexerDims {
        hidden: HIDDEN,
        q_lora_rank: Q_LORA,
        n_heads: IDX_HEADS,
        head_dim: IDX_HD,
        topk: c.index_topk,
        kpool: KPOOL,
        always_select_tail: true,
        seq,
        force_emit: false,
    };
    assert!(
        base.is_dense(),
        "the budget must not bind for this comparison"
    );

    let run = |force: bool| -> Vec<f32> {
        let mut w = weights(&c);
        let idx = IndexerDims {
            force_emit: force,
            ..base
        };
        let hs = Shape::new(&[1, seq, HIDDEN], DType::F32);
        let flow = ModelFlow::new("mla")
            .with_profile(CompileProfile::llama32_prefill())
            .input("x", hs.clone())
            .plugin_named("blk", move |emit, _p| {
                let x = emit.flow_input("x")?.hir_id();
                let out = emit_mla_attention(emit, "blk.3", x, mla, idx)?;
                Ok(Some(emit.wrap(out, hs.clone())))
            })
            .output("out");
        let built = flow
            .build_with(&mut WeightMapSource(&mut w), None)
            .expect("build MLA block");
        let mut compiled = compile_built(built, dev()).expect("compile");
        let x: Vec<f32> = (0..seq * HIDDEN)
            .map(|i| ((i as f32) * 0.017).sin() * 0.5)
            .collect();
        compiled.run(&[("x", x.as_slice())]).pop().expect("out")
    };

    let short_circuit = run(false);
    let via_indexer = run(true);
    assert_eq!(short_circuit.len(), via_indexer.len());
    assert!(
        short_circuit.iter().all(|v| v.is_finite()) && via_indexer.iter().all(|v| v.is_finite())
    );
    let worst = short_circuit
        .iter()
        .zip(&via_indexer)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let scale = short_circuit.iter().map(|v| v.abs()).fold(1e-3, f32::max);
    assert!(
        worst / scale < 1e-5,
        "running the indexer must reproduce the causal mask; max |Δ| = {worst} over {scale}"
    );
}

#[test]
fn mtp_is_rejected_rather_than_silently_skipped() {
    let mut c = cfg(2048);
    c.with_mtp = true;
    let mut w = weights(&c);
    let err = build_glm5next_text_flow(&c, &mut w, 8, true).unwrap_err();
    assert!(err.to_string().contains("MTP"), "unexpected error: {err}");
}
