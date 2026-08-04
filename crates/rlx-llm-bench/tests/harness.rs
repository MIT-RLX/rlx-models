// RLX models — LLM benchmark harness.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! End-to-end harness tests driven by the weightless [`MockRunner`], so the
//! whole pipeline (MC scoring, perplexity bridge, parity, speed) is exercised
//! with no checkpoint and no backend beyond CPU.

use rlx_llm_bench::McItem;
use rlx_llm_bench::PerplexityConfig;
use rlx_llm_bench::mock::MockRunner;
use rlx_llm_bench::model::BenchModel;
use rlx_llm_bench::parity::{ReferenceDump, run_parity};
use rlx_llm_bench::quality::run_perplexity;
use rlx_llm_bench::speed::{SpeedConfig, run_speed};

fn mock_model(vocab: usize) -> BenchModel {
    BenchModel::new(
        "mock",
        "cpu",
        Box::new(MockRunner::new(vocab)),
        None,
        Vec::new(),
    )
}

#[test]
fn mc_scoring_prefers_smaller_ids() {
    // The mock ranks smaller ids higher, so the [3,3] choice beats [9,9].
    let mut model = mock_model(32);
    let item = McItem {
        context: vec![1, 2],
        choices: vec![vec![3, 3], vec![9, 9]],
    };
    let res = model.score_mc(&item).unwrap();
    assert_eq!(res.best, 0, "smaller-id choice should win");
    assert_eq!(res.best_norm, 0);
    assert!(res.scores[0] > res.scores[1]);
    assert_eq!(res.scores.len(), 2);
}

#[test]
fn mc_scoring_rejects_empty_context() {
    let mut model = mock_model(16);
    let item = McItem {
        context: vec![],
        choices: vec![vec![1]],
    };
    assert!(model.score_mc(&item).is_err());
}

#[test]
fn perplexity_bridge_is_finite_and_positive() {
    let mut model = mock_model(64);
    let toks: Vec<u32> = (0..40).map(|i| (i % 20) as u32).collect();
    let ppl = run_perplexity(
        &mut model,
        &toks,
        PerplexityConfig {
            seq_len: 16,
            stride: 8,
        },
    )
    .unwrap();
    assert!(ppl.is_finite() && ppl > 0.0, "ppl = {ppl}");
}

#[test]
fn self_parity_matches_exactly() {
    let mut model = mock_model(48);
    let prompt = vec![1u32, 2, 3, 4];
    let logits = model.runner.prefill_logits(&prompt).unwrap();
    let dump = ReferenceDump::from_logits(prompt, logits);
    let r = run_parity(&mut model, &dump).unwrap();
    assert_eq!(r.argmax_match, Some(true));
    let cos = r.cosine.expect("cosine present when reference has logits");
    assert!((cos - 1.0).abs() < 1e-5, "cosine = {cos}");
}

#[test]
fn speed_generates_requested_tokens() {
    let mut model = mock_model(128);
    let cfg = SpeedConfig {
        prompt_ids: Vec::new(),
        prompt_len: 8,
        decode_tokens: 5,
        warmup: false,
    };
    let r = run_speed(&mut model, &cfg).unwrap();
    assert_eq!(r.prompt_tokens, 8);
    assert_eq!(r.decode_tokens, 5);
    // The mock supports a dedicated prefill, so this should be non-zero.
    assert!(r.prefill_toks_s > 0.0);
}

#[test]
fn parity_reference_roundtrips_json() {
    let dump = ReferenceDump::from_logits(vec![5, 6, 7], vec![0.1, 0.2, 0.3, 0.9]);
    let dir = std::env::temp_dir();
    let path = dir.join("rlx_llm_bench_ref_roundtrip.json");
    dump.save(&path).unwrap();
    let loaded = ReferenceDump::load(&path).unwrap();
    assert_eq!(loaded.prompt_ids, vec![5, 6, 7]);
    assert_eq!(loaded.argmax, Some(3));
    let _ = std::fs::remove_file(&path);
}
