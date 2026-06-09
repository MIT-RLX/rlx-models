// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! basic test for the qwen35 forward graph builder.
//!
//! Builds a tiny `Qwen35Weights` struct by hand (3 trunk layers
//! — `linear, linear, full_attn` with full_attention_interval=3
//! — plus 1 MTP layer), runs `build_qwen35_graph_sized` + a
//! compiled prefill, and verifies the output shape is the expected
//! `[1, 1, n_vocab]` last-token logits (with `last_logits_only=true`)
//! plus the MTP head's `[1, 1, n_vocab]` next-token logits.
//!
//! This is a *plumbing* test — it doesn't verify numerical
//! correctness (that needs a llama-cpp-rs reference on a real file).
//! It does verify:
//!   - Every shape inference path is valid for the layer mix.
//!   - The `Op::GatedDeltaNet` kernel slots into the trunk graph
//!     and produces a tensor that the rest of the graph accepts.
//!   - The `set_param` upload + `run` pipeline works end-to-end
//!     for the qwen35-shaped IR.

mod compile_support;

use rlx_models::qwen3::SampleOpts;
use rlx_models::qwen35::execution::{
    Qwen35CompileCache, decode_config, get_or_specialize_hir, prefill_config,
};
use rlx_models::qwen35::synth;
use rlx_models::{
    Qwen35Runner, Qwen35RunnerBuilder, build_qwen35_decode_graph, build_qwen35_graph_sized,
    build_qwen35_graph_sized_ext, build_qwen35_prefill_cache_graph,
    build_qwen35_prefill_cache_hir_dynamic_ext, decode_step_feeds, mrope_slice_at_pos,
    mtp_draft_vocab_size, pack_input_ids, recurrent_output_count, seed_cache_from_outputs,
    zero_recurrent_inputs,
};
use rlx_runtime::Device;

#[test]
fn qwen35_forward_graph_builds_and_runs_with_mtp() {
    let cfg = synth::tiny_cfg();
    let weights = synth::synth_weights(&cfg);

    let seq = 4;
    let (graph, params, packed) = build_qwen35_graph_sized(
        &cfg, weights, /*batch*/ 1, seq, /*with_lm_head*/ true,
        /*last_logits_only*/ true, /*enable_mtp_head*/ true,
    )
    .expect("build qwen35 graph");
    assert!(
        packed.is_empty(),
        "synth weights all F32; packed map must be empty (got {} entries)",
        packed.len()
    );

    let mut compiled = compile_support::compile_qwen35_prefill(Device::Cpu, graph, params);
    // Token ids as F32 input (the embed gather kernel does the
    // implicit cast — host I/O surface is F32-only).
    let ids: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let last_idx = vec![(seq - 1) as f32];

    // RoPE tables are baked into the graph (`qwen35.rope.cos/sin`).
    let feeds: Vec<(&str, &[f32])> = vec![("input_ids", &ids), ("last_token_idx", &last_idx)];

    let outs = compiled.run(&feeds);
    assert!(
        outs.len() >= 2,
        "expected trunk + MTP outputs, got {}",
        outs.len()
    );
    let n_vocab = cfg.vocab_size;
    assert_eq!(
        outs[0].len(),
        n_vocab,
        "trunk last-token logits len = {} (want {n_vocab})",
        outs[0].len()
    );
    assert_eq!(
        outs[1].len(),
        n_vocab,
        "MTP head logits len = {} (want {n_vocab})",
        outs[1].len()
    );
    // Preflight: all logits must be finite. NaN / Inf would mean the
    // graph has a numerical blowup (e.g. unguarded division, bad
    // L2 norm denom).
    for (i, v) in outs[0].iter().enumerate() {
        assert!(
            v.is_finite(),
            "trunk logits[{i}] = {v} (non-finite — math blew up)"
        );
    }
    for (i, v) in outs[1].iter().enumerate() {
        assert!(
            v.is_finite(),
            "MTP logits[{i}] = {v} (non-finite — math blew up)"
        );
    }
}

#[test]
fn qwen35_runner_builder_works_without_mtp() {
    // check: same as above but with `enable_mtp=false`, and only
    // verifying the trunk path.
    let cfg = synth::tiny_cfg();
    let weights = synth::synth_weights(&cfg);

    let (graph, params, _packed) = build_qwen35_graph_sized(&cfg, weights, 1, 4, true, true, false)
        .expect("build qwen35 graph (no mtp)");
    let mut compiled = compile_support::compile_qwen35_prefill(Device::Cpu, graph, params);
    let ids: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let last_idx = vec![3.0f32];
    let feeds: Vec<(&str, &[f32])> = vec![("input_ids", &ids), ("last_token_idx", &last_idx)];
    let outs = compiled.run(&feeds);
    assert_eq!(outs.len(), 1);
    assert_eq!(outs[0].len(), cfg.vocab_size);
    for v in &outs[0] {
        assert!(v.is_finite(), "logit = {v}");
    }
    let _ = Qwen35Runner::builder(); // just verifies the type exists
}

/// Larger-dims check: realistic ratios (hidden = n_head * head_dim,
/// GQA group = 2, dt_rank = 4×group, 5 trunk layers with two
/// full-attn slots at intervals matching the production config).
/// Catches shape-mismatch bugs (e.g. reshape rank mismatches) that
/// the tiny test misses because `1 * 1 * h * n == h * n` looks
/// right even when the rank is wrong.
#[test]
fn qwen35_forward_medium_dims_runs_with_mtp() {
    let cfg = synth::medium_cfg();
    let weights = synth::synth_weights(&cfg);
    let seq = 8;
    let (graph, params, _packed) =
        build_qwen35_graph_sized(&cfg, weights, 1, seq, true, true, true)
            .expect("build qwen35 medium graph");

    let mut compiled = compile_support::compile_qwen35_prefill(Device::Cpu, graph, params);

    let ids: Vec<f32> = (0..seq as u32).map(|i| (i + 1) as f32).collect();
    let last_idx = vec![(seq - 1) as f32];
    let feeds: Vec<(&str, &[f32])> = vec![("input_ids", &ids), ("last_token_idx", &last_idx)];
    let outs = compiled.run(&feeds);
    assert_eq!(outs.len(), 2, "trunk + MTP outputs");

    let trunk = &outs[0];
    let mtp = &outs[1];
    assert_eq!(trunk.len(), cfg.vocab_size);
    assert_eq!(mtp.len(), cfg.vocab_size);

    let nan_trunk = trunk.iter().filter(|v| !v.is_finite()).count();
    let nan_mtp = mtp.iter().filter(|v| !v.is_finite()).count();
    assert_eq!(nan_trunk, 0, "trunk has {nan_trunk} non-finite logits");
    assert_eq!(nan_mtp, 0, "MTP has {nan_mtp} non-finite logits");

    // Preflight: logits must not be all-zero (would indicate a
    // collapsed forward — e.g. RMS-norm divide-by-zero, or all
    // matmuls producing zero from a broken transpose).
    let trunk_nonzero = trunk.iter().filter(|v| **v != 0.0).count();
    assert!(
        trunk_nonzero > 0,
        "trunk logits collapsed to all-zero (broken forward)"
    );

    // Preflight: argmax should be a unique-ish token, not a degenerate
    // tie across vocab.
    let max_val = trunk.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let ties = trunk.iter().filter(|&&v| v == max_val).count();
    assert!(
        ties < cfg.vocab_size / 4,
        "trunk argmax tied across {ties}/{} tokens — likely all-equal logits",
        cfg.vocab_size
    );
}

#[test]
fn qwen35_prefill_cache_and_decode_graphs_run() {
    let cfg = synth::tiny_cfg();
    let weights = synth::synth_weights(&cfg);
    let seq = 4;

    let (cache_graph, cache_params, _) =
        build_qwen35_prefill_cache_graph(&cfg, weights.clone(), 1, seq).expect("prefill-cache");
    let mut prefill =
        compile_support::compile_qwen35_prefill(Device::Cpu, cache_graph, cache_params);

    let mut ids = vec![0f32; seq];
    for (i, v) in [1.0, 2.0, 3.0, 4.0].into_iter().enumerate() {
        ids[i] = v;
    }
    let zero_in = zero_recurrent_inputs(&cfg, 1);
    let last_idx = vec![(seq - 1) as f32];
    let mut feeds: Vec<(&str, &[f32])> = vec![("input_ids", &ids), ("last_token_idx", &last_idx)];
    for (name, data) in &zero_in {
        feeds.push((name, data.as_slice()));
    }
    let seed_outs = prefill.run(&feeds);
    let n_extra = recurrent_output_count(&cfg);
    assert_eq!(seed_outs.len(), 1 + n_extra);

    let (logits, cache, _) = seed_cache_from_outputs(&cfg, 1, seq, &[seq], seed_outs, false, false)
        .expect("parse cache");
    assert_eq!(logits.len(), cfg.vocab_size);
    assert_eq!(cache.past_seq, seq);

    let past_seq = cache.past_seq;
    let (dec_graph, dec_params, _) =
        build_qwen35_decode_graph(&cfg, weights, 1, past_seq).expect("decode graph");
    let mut decode = compile_support::compile_qwen35_decode(Device::Cpu, dec_graph, dec_params);

    let head_half = cfg.key_length / 2;
    let (cos, sin) = mrope_slice_at_pos(&cfg, past_seq, head_half);
    let token = 5.0f32;
    let token_ids = [token];
    let mut dec_feeds: Vec<(&str, &[f32])> = vec![
        ("input_ids", token_ids.as_slice()),
        ("rope_cos", cos.as_slice()),
        ("rope_sin", sin.as_slice()),
    ];
    let owned = decode_step_feeds(&cfg, &cache, &[token as u32], &cos, &sin, None, &[0usize])
        .expect("feeds");
    for (name, data) in &owned {
        dec_feeds.push((name, data.as_slice()));
    }
    let dec_outs = decode.run(&dec_feeds);
    assert_eq!(dec_outs.len(), 1 + n_extra);
    assert_eq!(dec_outs[0].len(), cfg.vocab_size);
    for v in &dec_outs[0] {
        assert!(v.is_finite(), "decode logit = {v}");
    }
}

#[test]
fn qwen35_prefill_cache_batch2_decode_runs() {
    let cfg = synth::tiny_cfg();
    let weights = synth::synth_weights(&cfg);
    let batch = 2usize;
    let seq = 4;

    let (cache_graph, cache_params, _) =
        build_qwen35_prefill_cache_graph(&cfg, weights.clone(), batch, seq).expect("prefill-cache");
    let mut prefill =
        compile_support::compile_qwen35_prefill(Device::Cpu, cache_graph, cache_params);

    let prompts = vec![vec![1, 2, 3, 4], vec![5, 6, 7, 8]];
    let prompt_lens: Vec<usize> = prompts.iter().map(|p| p.len()).collect();
    let ids = pack_input_ids(&prompts, seq).expect("pack");
    let zero_in = zero_recurrent_inputs(&cfg, batch);
    let last_idx = vec![(seq - 1) as f32; batch];
    let mut feeds: Vec<(&str, &[f32])> = vec![("input_ids", &ids), ("last_token_idx", &last_idx)];
    for (name, data) in &zero_in {
        feeds.push((name, data.as_slice()));
    }
    let seed_outs = prefill.run(&feeds);
    let n_extra = recurrent_output_count(&cfg);
    assert_eq!(seed_outs.len(), 1 + n_extra);

    let (logits, cache, _) =
        seed_cache_from_outputs(&cfg, batch, seq, &prompt_lens, seed_outs, false, false)
            .expect("parse cache");
    assert_eq!(cache.batch, batch);
    assert_eq!(cache.past_seq, seq);
    assert_eq!(logits.len(), batch * cfg.vocab_size);

    let past_seq = cache.past_seq;
    let (dec_graph, dec_params, _) =
        build_qwen35_decode_graph(&cfg, weights, batch, past_seq).expect("decode graph");
    let mut decode = compile_support::compile_qwen35_decode(Device::Cpu, dec_graph, dec_params);

    let head_half = cfg.key_length / 2;
    let (cos, sin) = mrope_slice_at_pos(&cfg, past_seq, head_half);
    let tokens = vec![9u32, 10u32];
    let owned = decode_step_feeds(&cfg, &cache, &tokens, &cos, &sin, None, &[0usize, 0usize])
        .expect("feeds");
    let dec_feeds: Vec<(&str, &[f32])> = owned
        .iter()
        .map(|(name, data)| (name.as_str(), data.as_slice()))
        .collect();
    let dec_outs = decode.run(&dec_feeds);
    assert_eq!(dec_outs.len(), 1 + n_extra);
    assert_eq!(dec_outs[0].len(), batch * cfg.vocab_size);
    for v in &dec_outs[0] {
        assert!(v.is_finite(), "batch decode logit = {v}");
    }
}

#[test]
fn bucketed_decode_matches_oneshot() {
    let cfg = synth::tiny_cfg();
    let prompt: Vec<u32> = vec![1, 2, 3, 5];
    let steps = 6;
    let max_seq = 32;

    let mut one = Qwen35RunnerBuilder::default()
        .inline_weights(cfg.clone(), synth::synth_weights(&cfg))
        .device(Device::Cpu)
        .max_seq(max_seq)
        .bucketed_decode(false)
        .last_logits_only(true)
        .build()
        .expect("oneshot runner");
    let oneshot = one
        .generate_with_opts(&prompt, steps, SampleOpts::greedy(), |_| true)
        .expect("oneshot generate");

    let mut buc = Qwen35RunnerBuilder::default()
        .inline_weights(cfg.clone(), synth::synth_weights(&cfg))
        .device(Device::Cpu)
        .max_seq(max_seq)
        .bucketed_decode(true)
        .last_logits_only(true)
        .build()
        .expect("bucketed runner");
    let bucketed = buc
        .generate_with_opts(&prompt, steps, SampleOpts::greedy(), |_| true)
        .expect("bucketed generate");

    assert_eq!(
        bucketed, oneshot,
        "bucketed-cache decode diverged from one-shot decode — \
         mask, padding, or output-slice bug"
    );
}

#[test]
fn fast_mtp_emits_narrower_logits() {
    let cfg = synth::tiny_cfg();
    let weights = synth::synth_weights(&cfg);
    let batch = 1;
    let seq = 4;
    let (graph, params, _) = build_qwen35_graph_sized_ext(
        &cfg, weights, batch, seq, true, true, true, false, None, false, true, false, false,
    )
    .expect("build fast_mtp graph");
    let mut compiled = compile_support::compile_qwen35_prefill(Device::Cpu, graph, params);
    let outs = compiled.run(&[
        ("input_ids", &[1.0, 2.0, 3.0, 4.0]),
        ("last_token_idx", &[3.0]),
    ]);
    assert!(outs.len() >= 2, "expected trunk + mtp outputs");
    let mtp = &outs[1];
    let want = mtp_draft_vocab_size(cfg.vocab_size, true);
    assert_eq!(
        mtp.len(),
        want,
        "FastMTP logits len {} != expected {want}",
        mtp.len()
    );
}

#[test]
fn mtp_spec_decode_round_runs_on_synthetic() {
    use rlx_models::qwen35::{Qwen35MtpDraft, Qwen35TrunkTarget, speculative_decode_round};
    use rlx_runtime::spec_decode::SpecDecoder;

    let cfg = synth::tiny_cfg();
    let prompt = vec![1u32, 2, 3];
    let max_seq = 32;

    let draft_runner = Qwen35RunnerBuilder::default()
        .inline_weights(cfg.clone(), synth::synth_weights(&cfg))
        .device(Device::Cpu)
        .max_seq(max_seq)
        .bucketed_decode(false)
        .enable_mtp(true)
        .mtp_logits_path(true)
        .last_logits_only(true)
        .build()
        .expect("draft runner");
    let target_runner = Qwen35RunnerBuilder::default()
        .inline_weights(cfg.clone(), synth::synth_weights(&cfg))
        .device(Device::Cpu)
        .max_seq(max_seq)
        .bucketed_decode(false)
        .enable_mtp(false)
        .mtp_logits_path(false)
        .last_logits_only(true)
        .build()
        .expect("target runner");

    let draft = Qwen35MtpDraft::new(draft_runner);
    let target = Qwen35TrunkTarget::new(target_runner);
    let mut dec = SpecDecoder::new(draft, target, 2, 0);
    let out = dec.step(&prompt);
    assert!(
        !out.is_empty(),
        "spec decode round must emit at least one token"
    );

    let round = speculative_decode_round(
        Qwen35MtpDraft::new(
            Qwen35RunnerBuilder::default()
                .inline_weights(synth::tiny_cfg(), synth::synth_weights(&synth::tiny_cfg()))
                .device(Device::Cpu)
                .max_seq(max_seq)
                .bucketed_decode(false)
                .enable_mtp(true)
                .mtp_logits_path(true)
                .last_logits_only(true)
                .build()
                .unwrap(),
        ),
        Qwen35TrunkTarget::new(
            Qwen35RunnerBuilder::default()
                .inline_weights(synth::tiny_cfg(), synth::synth_weights(&synth::tiny_cfg()))
                .device(Device::Cpu)
                .max_seq(max_seq)
                .bucketed_decode(false)
                .enable_mtp(false)
                .mtp_logits_path(false)
                .last_logits_only(true)
                .build()
                .unwrap(),
        ),
        &prompt,
        2,
        0,
    );
    assert!(!round.is_empty());
}

/// Dynamic prefill: compile HIR once with `sym::SEQ`, specialize per prompt length.
#[test]
fn qwen35_dynamic_prefill_specializes_per_seq() {
    let cfg = synth::tiny_cfg();
    let weights = synth::synth_weights(&cfg);
    let max_seq = 8;
    let mut cache = Qwen35CompileCache::new(Device::Cpu, 4);

    let (template_params, template_packed) = {
        let (_, p, packed) = build_qwen35_prefill_cache_hir_dynamic_ext(
            &cfg,
            weights.clone(),
            1,
            max_seq,
            false,
            false,
            false,
            false,
        )
        .expect("dynamic prefill HIR");
        (p, packed)
    };

    let mut template_loaded = false;
    for (seq, ids) in [
        (4usize, vec![1.0f32, 2.0, 3.0, 4.0]),
        (6usize, vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]),
    ] {
        let config = prefill_config(1, seq);
        let cfg_c = cfg.clone();
        let weights_c = weights.clone();
        let template_params_ref = &template_params;
        let compiled = get_or_specialize_hir(
            &mut cache,
            &config,
            || {
                if template_loaded {
                    panic!("dynamic HIR builder must run only once");
                }
                template_loaded = true;
                build_qwen35_prefill_cache_hir_dynamic_ext(
                    &cfg_c, weights_c, 1, max_seq, false, false, false, false,
                )
                .expect("dynamic prefill HIR")
                .0
            },
            |c| {
                for (name, data) in template_params_ref {
                    c.set_param(name, data);
                }
                Ok(())
            },
        )
        .expect("specialize dynamic prefill");
        let _ = template_packed;

        let last_idx = vec![(seq - 1) as f32];
        let zero_in = zero_recurrent_inputs(&cfg, 1);
        let mut feeds: Vec<(&str, &[f32])> =
            vec![("input_ids", &ids), ("last_token_idx", &last_idx)];
        for (name, data) in &zero_in {
            feeds.push((name, data.as_slice()));
        }
        let outs = compiled.run(&feeds);
        // Prefill-cache with `last_token_idx` returns last-position logits `[vocab]`.
        assert_eq!(outs[0].len(), cfg.vocab_size, "seq={seq}");
        for v in &outs[0] {
            assert!(v.is_finite(), "seq={seq} logit={v}");
        }
    }
    assert_eq!(cache.len(), 2);
}

/// Dynamic decode: compile HIR once with `sym::PAST_SEQ`, specialize per prefix length.
#[test]
fn qwen35_dynamic_decode_specializes_per_past_seq() {
    use rlx_models::build_qwen35_decode_hir_dynamic_ext;

    let cfg = synth::tiny_cfg();
    let weights = synth::synth_weights(&cfg);
    let max_seq = 8;
    let mut cache = Qwen35CompileCache::new(Device::Cpu, 4);

    let built =
        build_qwen35_decode_hir_dynamic_ext(&cfg, weights.clone(), 1, max_seq, false, false, false)
            .expect("dynamic decode HIR");
    let template_params = built.1;

    let mut template_loaded = false;
    for past_seq in [3usize, 5] {
        let config = decode_config(1, past_seq);
        let cfg_c = cfg.clone();
        let weights_c = weights.clone();
        let template_params_ref = &template_params;
        let compiled = get_or_specialize_hir(
            &mut cache,
            &config,
            || {
                if template_loaded {
                    panic!("dynamic decode HIR builder must run only once");
                }
                template_loaded = true;
                build_qwen35_decode_hir_dynamic_ext(
                    &cfg_c, weights_c, 1, max_seq, false, false, false,
                )
                .expect("dynamic decode HIR")
                .0
            },
            |c| {
                for (name, data) in template_params_ref {
                    c.set_param(name, data);
                }
                Ok(())
            },
        )
        .expect("specialize dynamic decode");
        let _ = past_seq;
        let _ = compiled;
    }
    assert_eq!(cache.len(), 2);
}

/// Runner: dynamic prefill + dynamic decode end-to-end.
#[test]
fn qwen35_runner_dynamic_prefill_and_decode() {
    let cfg = synth::tiny_cfg();
    let weights = synth::synth_weights(&cfg);
    let mut runner = Qwen35RunnerBuilder::default()
        .inline_weights(cfg.clone(), weights)
        .device(Device::Cpu)
        .max_seq(8)
        .dynamic_prefill(true)
        .dynamic_decode(true)
        .bucketed_decode(false)
        .last_logits_only(true)
        .build()
        .expect("dynamic runner");

    runner
        .prefill_get_last_logits(&[1, 2, 3])
        .expect("dynamic prefill");
    let logits = runner.decode_get_logits(4).expect("dynamic decode step");
    assert_eq!(logits.len(), cfg.vocab_size);
}

/// Dynamic prefill + MTP head logits path.
#[test]
fn qwen35_dynamic_prefill_with_mtp() {
    let cfg = synth::tiny_cfg();
    let weights = synth::synth_weights(&cfg);
    let mut runner = Qwen35RunnerBuilder::default()
        .inline_weights(cfg.clone(), weights)
        .device(Device::Cpu)
        .max_seq(8)
        .enable_mtp(true)
        .mtp_logits_path(true)
        .dynamic_prefill(true)
        .last_logits_only(true)
        .build()
        .expect("mtp dynamic runner");

    let seed = runner
        .prefill_seed_for_decode(&[1, 2, 3, 4])
        .expect("mtp dynamic prefill seed");
    assert_eq!(seed.trunk_logits.len(), cfg.vocab_size);
    assert!(seed.mtp_logits.is_some());
}

/// Runner API: `dynamic_prefill(true)` seeds decode cache at two prompt lengths.
#[test]
fn qwen35_runner_dynamic_prefill_seeds_decode() {
    let cfg = synth::tiny_cfg();
    let weights = synth::synth_weights(&cfg);
    let mut runner = Qwen35RunnerBuilder::default()
        .inline_weights(cfg.clone(), weights)
        .device(Device::Cpu)
        .max_seq(8)
        .enable_mtp(false)
        .dynamic_prefill(true)
        .last_logits_only(true)
        .build()
        .expect("dynamic prefill runner");

    for prompt in [&[1u32, 2, 3][..], &[1u32, 2, 3, 4, 5][..]] {
        let logits = runner
            .prefill_get_last_logits(prompt)
            .expect("prefill at varying seq");
        assert_eq!(logits.len(), cfg.vocab_size);
        for v in &logits {
            assert!(v.is_finite());
        }
        runner.reset_decode_cache();
    }
}

fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i as u32)
        .unwrap()
}

/// Batched generation (batch=2) on synthetic weights.
#[test]
fn qwen35_batch2_generate_runs() {
    let cfg = synth::tiny_cfg();
    let weights = synth::synth_weights(&cfg);
    let mut runner = Qwen35RunnerBuilder::default()
        .inline_weights(cfg.clone(), weights)
        .device(Device::Cpu)
        .batch(2)
        .max_seq(8)
        .last_logits_only(true)
        .build()
        .expect("batch=2 runner");

    let generated = runner
        .generate_batch_with_opts(
            &[vec![1, 2, 3], vec![4, 5, 6, 7]],
            2,
            None,
            SampleOpts::greedy(),
            |_, _| true,
        )
        .expect("batch generate");
    assert_eq!(generated.len(), 2);
    assert_eq!(generated[0].len(), 2);
    assert_eq!(generated[1].len(), 2);
    for row in &generated {
        for &tok in row {
            assert!((tok as usize) < cfg.vocab_size);
        }
    }
}

/// Cached decode must match full reprefill greedy tokens (synthetic).
#[test]
fn qwen35_generate_matches_repredict_greedy() {
    let cfg = synth::tiny_cfg();
    let weights = synth::synth_weights(&cfg);
    let prompt = vec![1u32, 2, 3];
    let steps = 4usize;

    let mut cached_runner = Qwen35RunnerBuilder::default()
        .inline_weights(cfg.clone(), weights.clone())
        .device(Device::Cpu)
        .max_seq(16)
        .bucketed_decode(true)
        .last_logits_only(true)
        .build()
        .expect("cached runner");
    let cached = cached_runner
        .generate_with_opts(&prompt, steps, SampleOpts::greedy(), |_| true)
        .expect("cached generate");

    let mut repredict_runner = Qwen35RunnerBuilder::default()
        .inline_weights(cfg, weights)
        .device(Device::Cpu)
        .max_seq(16)
        .last_logits_only(true)
        .build()
        .expect("repredict runner");
    let mut repredict = Vec::with_capacity(steps);
    let mut context = prompt;
    for _ in 0..steps {
        let logits = repredict_runner
            .predict_logits(&context)
            .expect("predict")
            .logits;
        let tok = argmax(&logits);
        repredict.push(tok);
        context.push(tok);
    }

    assert_eq!(
        cached, repredict,
        "cached decode diverged from full reprefill greedy baseline"
    );
}

/// Batch=2: bucketed decode cache must match non-bucketed path.
#[test]
fn qwen35_batch2_bucketed_matches_oneshot() {
    let cfg = synth::tiny_cfg();
    let weights = synth::synth_weights(&cfg);
    let prompts = [vec![1u32, 2, 3], vec![4u32, 5, 6, 7]];
    let steps = 4;
    let max_seq = 16;

    let mut one = Qwen35RunnerBuilder::default()
        .inline_weights(cfg.clone(), weights.clone())
        .device(Device::Cpu)
        .batch(2)
        .max_seq(max_seq)
        .bucketed_decode(false)
        .last_logits_only(true)
        .build()
        .expect("oneshot batch runner");
    let oneshot = one
        .generate_batch_with_opts(&prompts, steps, None, SampleOpts::greedy(), |_, _| true)
        .expect("oneshot batch generate");

    let mut buc = Qwen35RunnerBuilder::default()
        .inline_weights(cfg, weights)
        .device(Device::Cpu)
        .batch(2)
        .max_seq(max_seq)
        .bucketed_decode(true)
        .last_logits_only(true)
        .build()
        .expect("bucketed batch runner");
    let bucketed = buc
        .generate_batch_with_opts(&prompts, steps, None, SampleOpts::greedy(), |_, _| true)
        .expect("bucketed batch generate");

    assert_eq!(
        bucketed, oneshot,
        "batch=2 bucketed decode diverged from one-shot — padding/mask bug"
    );
}

/// AOT cache persists prefill LIR; second runner reuses warm-start artifacts.
#[test]
fn qwen35_aot_cache_warm_start() {
    use std::fs;

    let dir = std::env::temp_dir().join(format!("rlx_qwen35_aot_{}", std::process::id()));
    fs::remove_dir_all(&dir).ok();

    let cfg = synth::tiny_cfg();
    let weights = synth::synth_weights(&cfg);
    let build = || {
        Qwen35RunnerBuilder::default()
            .inline_weights(cfg.clone(), weights.clone())
            .device(Device::Cpu)
            .max_seq(8)
            .aot_cache_dir(&dir)
            .last_logits_only(true)
            .build()
            .expect("aot runner")
    };

    let mut first = build();
    first
        .prefill_get_last_logits(&[1, 2, 3])
        .expect("first prefill");
    let lir_files: Vec<_> = fs::read_dir(&dir)
        .expect("read AOT dir")
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path().extension().is_some_and(|x| x == "json")
                && e.file_name().to_string_lossy().ends_with(".lir.json")
        })
        .collect();
    assert!(
        !lir_files.is_empty(),
        "AOT cache should persist at least one *.lir.json under {:?}",
        dir
    );

    let mut second = build();
    let logits = second
        .prefill_get_last_logits(&[1, 2, 3, 4])
        .expect("second prefill from warm AOT");
    assert_eq!(logits.len(), cfg.vocab_size);

    fs::remove_dir_all(&dir).ok();
}
