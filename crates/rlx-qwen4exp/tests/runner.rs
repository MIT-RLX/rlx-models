// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! The high-level runner against the hand-driven loop it replaces.
//!
//! `tests/decode_equivalence.rs` already pins that stepping the decode flow
//! reproduces a single-shot prefill. This file pins the layer above it: that
//! [`Qwen4ExpRunner`] drives that same flow the same way. The oracle is the
//! explicit loop — M-RoPE tables, n-gram rows hashed against the history,
//! `set_real_len`, `commit`, `push_token` — so a runner that threads any one of
//! those wrongly disagrees here.

use rlx_core::flow_util::compile_built;
use rlx_core::weight_map::WeightMap;
use rlx_qwen4exp::config::{PleHash, Qwen4ExpConfig};
use rlx_qwen4exp::decode::{
    DecodeNames, DecodeSession, ScanState, build_qwen4exp_decode_flow_with,
};
use rlx_qwen4exp::flow::{OutputKind, ROPE_COS_INPUT, ROPE_SIN_INPUT};
use rlx_qwen4exp::ple::{DensePleTable, ple_embed_input};
use rlx_qwen4exp::rope::{mrope_tables, text_section_pos};
use rlx_qwen4exp::runner::{Qwen4ExpRunner, RunnerOptions, scan_state_for};
use rlx_qwen4exp::synth;
use rlx_runtime::Device;

const TOKENS: [u32; 5] = [11, 2048, 7, 909, 31];

fn dev() -> Device {
    rlx_ir::env::var_result("RLX_TEST_DEVICE")
        .ok()
        .filter(|s| !s.is_empty())
        .map(|s| rlx_cli::parse_device(&s).expect("bad RLX_TEST_DEVICE"))
        .unwrap_or(Device::Cpu)
}

fn cfg() -> Qwen4ExpConfig {
    synth::tiny_config(2048)
}

fn weights(cfg: &Qwen4ExpConfig) -> WeightMap {
    let mut t = synth::synth_tensors(cfg, 13);
    // Never a graph parameter: the rows are gathered host-side.
    t.remove("per_layer_token_embd.weight");
    WeightMap::from_tensors(t)
}

fn table(cfg: &Qwen4ExpConfig) -> DensePleTable {
    let rows = PleHash::derive(cfg, 0).padded_rows as usize;
    let hd = cfg.ple_head_dim;
    DensePleTable {
        data: (0..rows * hd)
            .map(|i| ((i % 89) as f32 - 44.0) * 0.003)
            .collect(),
        rows,
        head_dim: hd,
    }
}

fn runner(cfg: &Qwen4ExpConfig, opts: RunnerOptions) -> Qwen4ExpRunner {
    let mut w = weights(cfg);
    Qwen4ExpRunner::from_source(
        cfg.clone(),
        &mut rlx_core::flow_util::WeightMapSource(&mut w),
        opts,
    )
    .expect("build runner")
    .with_ple_rows(Box::new(table(cfg)))
}

/// Logits for every position, from the loop the runner is meant to replace.
fn hand_driven(cfg: &Qwen4ExpConfig, tokens: &[u32], cap: usize) -> Vec<Vec<f32>> {
    let mut tbl = table(cfg);
    let mut w = weights(cfg);
    let (built, layout) = build_qwen4exp_decode_flow_with(
        cfg,
        &mut rlx_core::flow_util::WeightMapSource(&mut w),
        cap,
        OutputKind::Logits,
        ScanState::Portable,
    )
    .expect("build decode");
    let names = DecodeNames::new(cfg);
    let mut session = DecodeSession::new(cfg, layout, cap);
    let mut compiled = compile_built(built, dev()).expect("compile decode");

    let mut out = Vec::new();
    for (t, &tok) in tokens.iter().enumerate() {
        let emb = session
            .ple_embeddings(cfg, 0, tok, &mut tbl)
            .expect("ple rows");
        let (cos, sin) = mrope_tables(cfg, 1, Some(&[text_section_pos(t)]));
        let ids = [tok as f32];
        let ple_name = ple_embed_input(0);
        let mut inputs: Vec<(&str, &[f32])> = vec![
            ("input_ids", &ids),
            (ROPE_COS_INPUT, cos.as_slice()),
            (ROPE_SIN_INPUT, sin.as_slice()),
            (ple_name.as_str(), emb.as_slice()),
        ];
        inputs.extend(session.inputs(cfg, &names));
        let packed = compiled.run(&inputs).pop().expect("packed");
        out.push(session.commit(&packed).expect("commit").to_vec());
        session.push_token(tok);
    }
    out
}

/// The runner's prefill + decode agree with the explicit loop, token for token.
#[test]
fn the_runner_reproduces_the_hand_driven_loop() {
    let cfg = cfg();
    let probe = runner(&cfg, RunnerOptions::new(dev()));
    let want = hand_driven(&cfg, &TOKENS, probe.capacity());
    drop(probe);
    assert_eq!(want.len(), TOKENS.len());
    assert_eq!(want[0].len(), cfg.vocab_size);

    let mut run = runner(&cfg, RunnerOptions::new(dev()));
    // Prefill the first two, then step the rest — the seam the runner has to
    // get right is that decode continues the prefill's state.
    let got0 = run.prefill_logits(&TOKENS[..2]).expect("prefill");
    let mut got = vec![got0];
    for &tok in &TOKENS[2..] {
        got.push(run.decode_logits(tok).expect("decode"));
    }

    // Prefill returns only its last position, so compare that against the
    // matching entry and every decode step after it.
    // Bit-identical, not merely close: the runner compiles the same graph and
    // feeds it the same values, so any difference is a threading mistake rather
    // than arithmetic. Asserting equality is what makes a one-position M-RoPE
    // slip visible — at a 2e-3 relative tolerance it would scrape through.
    let pairs = [(1usize, 0usize), (2, 1), (3, 2), (4, 3)];
    for (w, g) in pairs {
        let (a, b) = (&want[w], &got[g]);
        let max_abs = a
            .iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert_eq!(
            a, b,
            "position {w}: runner and hand-driven loop differ by {max_abs}"
        );
    }
    assert_eq!(run.position(), TOKENS.len());
}

/// `generate_greedy` returns the argmax of the logits it decoded, in order.
#[test]
fn generate_greedy_is_the_argmax_chain() {
    let cfg = cfg();
    let mut run = runner(&cfg, RunnerOptions::new(dev()));
    let mut seen: Vec<u32> = Vec::new();
    let out = run
        .generate_greedy(&TOKENS[..2], 3, |t| {
            seen.push(t);
            true
        })
        .expect("generate");
    assert_eq!(out, seen, "the callback sees exactly what is returned");
    assert_eq!(out.len(), 3);

    // Same prompt through the explicit calls must give the same chain.
    let mut run2 = runner(&cfg, RunnerOptions::new(dev()));
    let mut logits = run2.prefill_logits(&TOKENS[..2]).expect("prefill");
    for (i, &tok) in out.iter().enumerate() {
        let want = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap();
        assert_eq!(tok, want, "token {i}");
        if i + 1 < out.len() {
            logits = run2.decode_logits(tok).expect("decode");
        }
    }
}

/// Past the QSA budget the runner refuses, with the reason.
#[test]
fn the_runner_refuses_to_outrun_its_capacity() {
    let cfg = cfg();
    let mut opts = RunnerOptions::new(dev());
    opts.cap = 3;
    let mut run = runner(&cfg, opts);
    let err = run
        .prefill_logits(&TOKENS)
        .expect_err("5 tokens must not fit a capacity of 3");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("capacity") && msg.contains("indexer"),
        "unhelpful refusal: {msg}"
    );
    assert_eq!(run.capacity(), 3);
}

/// A config with a PLE module cannot run without the row source, and says so.
#[test]
fn a_ple_model_without_its_table_is_refused() {
    let cfg = cfg();
    let mut w = weights(&cfg);
    let mut run = Qwen4ExpRunner::from_source(
        cfg.clone(),
        &mut rlx_core::flow_util::WeightMapSource(&mut w),
        RunnerOptions::new(dev()),
    )
    .expect("build runner");
    let err = run.prefill_logits(&TOKENS[..1]).expect_err("no row source");
    assert!(
        format!("{err:#}").contains("with_ple_rows"),
        "the refusal should name the fix: {err:#}"
    );
}

/// `reset` re-runs a sequence from scratch — and refuses when the scan state
/// lives on the device, where it cannot be re-zeroed.
#[test]
fn reset_restarts_a_portable_runner_and_refuses_an_in_place_one() {
    let cfg = cfg();
    let mut run = runner(&cfg, RunnerOptions::new(dev()));
    let first = run.prefill_logits(&TOKENS[..2]).expect("prefill");
    run.reset().expect("portable reset");
    assert_eq!(run.position(), 0);
    let again = run.prefill_logits(&TOKENS[..2]).expect("re-prefill");
    assert_eq!(first, again, "a reset runner must replay identically");

    let mut single = runner(&cfg, RunnerOptions::new(dev()).single_sequence());
    if scan_state_for(dev()) == ScanState::InPlace {
        let err = single.reset().expect_err("in-place state cannot be reset");
        assert!(format!("{err:#}").contains("rebuilding"), "{err:#}");
    } else {
        single.reset().expect("portable on this device");
    }
}

/// The scan-state policy is the one the backends actually support.
#[test]
fn the_scan_state_policy_matches_the_backends() {
    for d in [Device::Cpu, Device::Metal, Device::Gpu] {
        assert_eq!(scan_state_for(d), ScanState::InPlace, "{d:?}");
    }
    // MLX and CoreML substitute state into a per-eval env: it freezes at zero.
    for d in [Device::Mlx, Device::Ane, Device::Cuda, Device::Vulkan] {
        assert_eq!(scan_state_for(d), ScanState::Portable, "{d:?}");
    }
}

/// A chunked prefill then a one-token decode, against the same runner driven
/// one token throughout.
///
/// This is the seam `DecodeSession::set_chunk` exists for: the two graphs pack
/// their output differently, but the recurrent state they thread does not, so
/// the decode after a chunked prefill has to continue it exactly. Getting the
/// switch wrong shows up here and nowhere else — the within-prefill logits
/// would still agree.
#[test]
fn a_chunked_prefill_continues_into_decode() {
    let cfg = cfg();
    let chunk = 3;
    let cap = 16;

    let mut opts_c = RunnerOptions::new(dev()).prefill_chunk(chunk);
    opts_c.cap = cap;
    let mut wd = weights(&cfg);
    let mut wp = weights(&cfg);
    let mut chunked = Qwen4ExpRunner::from_sources(
        cfg.clone(),
        &mut rlx_core::flow_util::WeightMapSource(&mut wd),
        &mut rlx_core::flow_util::WeightMapSource(&mut wp),
        opts_c,
    )
    .expect("build chunked runner")
    .with_ple_rows(Box::new(table(&cfg)));

    let mut opts_1 = RunnerOptions::new(dev());
    opts_1.cap = cap;
    let mut single = runner(&cfg, opts_1);

    // Five tokens: one full chunk of three, then a short chunk of two.
    let a = chunked.prefill_logits(&TOKENS).expect("chunked prefill");
    let b = single.prefill_logits(&TOKENS).expect("single prefill");
    close(&a, &b, "prefill");
    assert_eq!(chunked.position(), TOKENS.len());

    // And the state carried: two decode steps on each.
    for (i, tok) in [4u32, 77u32].into_iter().enumerate() {
        let a = chunked.decode_logits(tok).expect("chunked decode");
        let b = single.decode_logits(tok).expect("single decode");
        close(&a, &b, &format!("decode step {i}"));
    }
}

/// `from_sources` is for the chunked case only, and says so.
#[test]
fn from_sources_rejects_an_unchunked_request() {
    let cfg = cfg();
    let mut w1 = weights(&cfg);
    let mut w2 = weights(&cfg);
    let err = Qwen4ExpRunner::from_sources(
        cfg.clone(),
        &mut rlx_core::flow_util::WeightMapSource(&mut w1),
        &mut rlx_core::flow_util::WeightMapSource(&mut w2),
        RunnerOptions::new(dev()),
    )
    .err()
    .expect("prefill_chunk of 1 must be rejected");
    assert!(format!("{err:#}").contains("from_source"), "{err:#}");
}

/// A capacity that cannot hold one prefill chunk is refused at build time.
#[test]
fn a_capacity_that_cannot_hold_a_chunk_is_refused() {
    let cfg = cfg();
    let mut opts = RunnerOptions::new(dev()).prefill_chunk(8);
    opts.cap = cfg.indexer_budget;
    let mut w1 = weights(&cfg);
    let mut w2 = weights(&cfg);
    let err = Qwen4ExpRunner::from_sources(
        cfg.clone(),
        &mut rlx_core::flow_util::WeightMapSource(&mut w1),
        &mut rlx_core::flow_util::WeightMapSource(&mut w2),
        opts,
    )
    .err()
    .expect("cap + chunk past the budget must be rejected");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("indexer_budget") && msg.contains("prefill chunk"),
        "{msg}"
    );
}

/// Two logit rows agree, at a tolerance picked from both sides of the gap.
///
/// A chunk-3 and a chunk-1 graph differ only in reduction order, which measures
/// 4.9e-8 relative here. Losing the session's live-position row across the
/// chunk switch — the mistake this test exists to catch — measures 5.2e-4. The
/// 2e-3 the sibling prefill test uses would let that through, so this sits at
/// 1e-5: 200x the legitimate noise, 50x under the defect.
fn close(a: &[f32], b: &[f32], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: width");
    let max_abs = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    let scale = a.iter().map(|v| v.abs()).fold(1e-3f32, f32::max);
    assert!(
        max_abs / scale < 1e-5,
        "{what}: chunked and single-token runners differ by {max_abs} \
         ({:.3e} relative to a {scale} scale)",
        max_abs / scale
    );
}
