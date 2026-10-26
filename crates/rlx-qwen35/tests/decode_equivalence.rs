// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0-only

//! The decode graph, stepped one token at a time, must reproduce what the
//! prefill graph computes for the same context.
//!
//! Qwen3.5's trunk is a 3:1 hybrid, so a decode step has to resume three
//! separate pieces of carried state rather than just read a KV cache:
//!
//! * the GatedDeltaNet recurrent state, resumed instead of scanned from zero;
//! * the `ssm_conv1d` short-conv window, whose left-pad becomes a carried
//!   `[k-1, channels]` state;
//! * the full-attention KV cache on every `full_attention_interval`-th layer.
//!
//! Prefill computes all three in one pass over the prompt; decode reads them
//! back. Nothing checks that the hand-off is lossless, and a small leak is
//! invisible at a glance: the model keeps emitting fluent text, it just
//! answers as though it had read a slightly different prompt. It shows up as
//! a greedy run that tracks a reference for a few tokens and then walks off.
//!
//! So this pins greedy decode against teacher forcing. Generate `N` tokens the
//! normal way (token 1 from prefill, the rest from decode), then re-run each
//! prefix through prefill alone and require the same argmax at every step.
//! Those are the same function; only the path differs.

use rlx_qwen35::synth::{synth_weights, tiny_cfg};
use rlx_qwen35::{Qwen35Config, Qwen35RunnerBuilder, Qwen35TrunkLayer, Qwen35Weights};
use rlx_runtime::Device;

const PROMPT: [u32; 4] = [5, 11, 2, 19];
const N_NEW: usize = 6;

fn dev() -> Device {
    rlx_ir::env::var_result("RLX_TEST_DEVICE")
        .ok()
        .filter(|s| !s.is_empty())
        .map(|s| rlx_cli::parse_device(&s).expect("bad RLX_TEST_DEVICE"))
        .unwrap_or(Device::Cpu)
}

/// Deterministic pseudo-random scalar.
fn hashed(seed: u64, i: usize) -> f32 {
    let mut x = seed ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x ^= x >> 29;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 32;
    ((x >> 40) as f32) / 8_388_608.0 - 1.0
}

/// Give the synthetic model full-rank weights.
///
/// `synth_weights` builds everything from `ramp()`, a strictly linear
/// sequence, which leaves the model so nearly position-invariant that a
/// decode step reading slightly stale state still lands on the same argmax.
/// That would make this test pass while blind.
fn randomize(w: &mut Qwen35Weights, cfg: &Qwen35Config, conv_identity: bool) {
    let fill = |v: &mut [f32], tag: u64| {
        for (i, x) in v.iter_mut().enumerate() {
            *x = 0.3 * hashed(tag, i);
        }
    };
    let mut embd = w.token_embd().to_vec();
    fill(&mut embd, 0x1234);
    w.set_token_embd(std::sync::Arc::from(embd));

    for (il, layer) in w.trunk_layers.iter_mut().enumerate() {
        let tag = 0x5000 + il as u64 * 97;
        let ffn = match layer {
            Qwen35TrunkLayer::Linear(l) => {
                for (j, p) in [
                    &mut l.attn_qkv,
                    &mut l.attn_gate,
                    &mut l.ssm_beta,
                    &mut l.ssm_alpha,
                    &mut l.ssm_out,
                ]
                .into_iter()
                .enumerate()
                {
                    for m in p.mats_mut() {
                        if let rlx_qwen35::MatWeight::F32(v) = m {
                            fill(v, tag + j as u64);
                        }
                    }
                }
                // Keep the GDN decay in a sane range; `ssm_a` is exponentiated.
                for (i, a) in l.ssm_a.iter_mut().enumerate() {
                    *a = -1.0 - 0.1 * (i % 3) as f32;
                }
                if conv_identity {
                    // Only the current timestep's tap: the conv output stops
                    // depending on history, so a wrong carried window cannot
                    // affect the result. Weight layout is [channels, 1, k, 1].
                    let k = cfg.ssm_conv_kernel;
                    for (i, v) in l.ssm_conv1d.iter_mut().enumerate() {
                        *v = if i % k == k - 1 { 1.0 } else { 0.0 };
                    }
                } else {
                    fill(&mut l.ssm_conv1d, tag + 11);
                }
                &mut l.ffn
            }
            Qwen35TrunkLayer::FullAttn(l) => {
                for (j, p) in [
                    &mut l.attn_q_gate,
                    &mut l.attn_k,
                    &mut l.attn_v,
                    &mut l.attn_output,
                ]
                .into_iter()
                .enumerate()
                {
                    for m in p.mats_mut() {
                        if let rlx_qwen35::MatWeight::F32(v) = m {
                            fill(v, tag + 20 + j as u64);
                        }
                    }
                }
                &mut l.ffn
            }
        };
        if let rlx_qwen35::Qwen35LayerFfn::Dense { gate, up, down } = ffn {
            for (j, p) in [gate, up, down].into_iter().enumerate() {
                for m in p.mats_mut() {
                    if let rlx_qwen35::MatWeight::F32(v) = m {
                        fill(v, tag + 40 + j as u64);
                    }
                }
            }
        }
    }
}

fn model(full_attention_interval: usize, conv_identity: bool) -> (Qwen35Config, Qwen35Weights) {
    let mut cfg = tiny_cfg();
    cfg.full_attention_interval = full_attention_interval;
    let mut w = synth_weights(&cfg);
    randomize(&mut w, &cfg, conv_identity);
    (cfg, w)
}

fn runner(cfg: &Qwen35Config, w: &Qwen35Weights) -> rlx_qwen35::Qwen35Runner {
    Qwen35RunnerBuilder::default()
        .inline_weights(cfg.clone(), w.clone())
        .device(dev())
        .fast_greedy_lm_head(false)
        .max_seq(PROMPT.len() + N_NEW + 2)
        .build()
        .expect("build runner")
}

fn argmax(v: &[f32]) -> u32 {
    v.iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &x)| {
            if x > bv { (i, x) } else { (bi, bv) }
        })
        .0 as u32
}

/// Returns the mismatch report for a trunk with the given layer mix.
fn check(full_attention_interval: usize, conv_identity: bool) -> Vec<String> {
    let (cfg, w) = model(full_attention_interval, conv_identity);

    let mut gen_runner = runner(&cfg, &w);
    let generated = gen_runner
        .generate(&PROMPT, N_NEW, |_| true)
        .expect("generate");
    assert_eq!(generated.len(), N_NEW, "short generation");

    let mut ctx: Vec<u32> = PROMPT.to_vec();
    let mut mismatches = Vec::new();
    let mut prefill_sums = Vec::new();
    for (step, &tok) in generated.iter().enumerate() {
        // Fresh runner per call: `predict_logits` on a reused runner could
        // otherwise answer from carried state and silently make the whole
        // comparison vacuous.
        let mut tf_runner = runner(&cfg, &w);
        let out = tf_runner.predict_logits(&ctx).expect("predict_logits");
        prefill_sums.push(out.logits.iter().map(|&x| x as f64).sum::<f64>());
        let want = argmax(&out.logits);
        if want != tok {
            // Report the margin too: a 1e-6 tie flipping is float
            // reassociation, a wide gap is a state hand-off bug.
            let mut sorted: Vec<f32> = out.logits.clone();
            sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
            let margin = sorted[0] - sorted[1];
            mismatches.push(format!(
                "step {step} (ctx len {}): decode={tok} prefill={want} (prefill top-2 margin {margin:.4})",
                ctx.len()
            ));
        }
        ctx.push(tok);
    }

    // Guard the guard: if prefill returned the same logits for every context
    // length, "decode disagrees with prefill" would be about prefill.
    let distinct = prefill_sums
        .iter()
        .map(|v| format!("{v:.3}"))
        .collect::<std::collections::HashSet<_>>()
        .len();
    assert!(
        distinct > 1,
        "prefill returned identical logits for all {} context lengths — the \
         teacher-forced side is not seeing the growing context, so this test \
         cannot say anything about decode",
        prefill_sums.len()
    );

    mismatches
}

/// All-GatedDeltaNet trunk: isolates the recurrent scan state and the
/// `ssm_conv1d` window hand-off, with no KV cache in play.
#[test]
fn gdn_only_decode_matches_prefill() {
    let m = check(99, false);
    assert!(
        m.is_empty(),
        "GDN decode diverges from prefill:\n  {}",
        m.join("\n  ")
    );
}

/// All-full-attention trunk: isolates the KV cache.
#[test]
fn full_attention_only_decode_matches_prefill() {
    let m = check(1, false);
    assert!(
        m.is_empty(),
        "full-attention decode diverges from prefill:\n  {}",
        m.join("\n  ")
    );
}

/// The shipped 3:1 mix.
#[test]
fn hybrid_decode_matches_prefill() {
    let m = check(3, false);
    assert!(
        m.is_empty(),
        "hybrid decode diverges from prefill:\n  {}",
        m.join("\n  ")
    );
}

/// GDN with the short conv made time-local: if this passes while
/// `gdn_only_decode_matches_prefill` fails, the fault is the carried conv
/// window; if it also fails, it is the recurrent scan state.
#[test]
fn gdn_with_time_local_conv_decode_matches_prefill() {
    let m = check(99, true);
    assert!(
        m.is_empty(),
        "GDN (time-local conv) decode diverges from prefill:\n  {}",
        m.join("\n  ")
    );
}

/// Prefill logits must not depend on how much zero padding the runner adds
/// after the prompt.
///
/// Full attention masks padding out, but a GatedDeltaNet layer is a
/// recurrent scan with no mask — every padded position still advances the
/// state. If that is happening, the exported state describes the prompt plus
/// a run of zero tokens, and every decode step after it reads a cache the
/// prefill path never produced.
#[test]
fn prefill_logits_are_independent_of_padding() {
    let (cfg, w) = model(99, false);
    let ctx: Vec<u32> = PROMPT.to_vec();

    let logits_for = |max_seq: usize| -> Vec<f32> {
        Qwen35RunnerBuilder::default()
            .inline_weights(cfg.clone(), w.clone())
            .device(dev())
            .fast_greedy_lm_head(false)
            .max_seq(max_seq)
            .build()
            .expect("build runner")
            .predict_logits(&ctx)
            .expect("predict_logits")
            .logits
    };

    let tight = logits_for(ctx.len());
    let padded = logits_for(ctx.len() + 8);
    let worst = tight
        .iter()
        .zip(&padded)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        worst < 1e-4,
        "prefill logits move by {worst} when the prompt is padded from {} to {} \
         positions — padding is leaking into the recurrent state",
        ctx.len(),
        ctx.len() + 8
    );
}

/// Generation must not depend on `max_seq`.
///
/// `prefill_logits_are_independent_of_padding` only covers the logits, which
/// are read at the last *real* position and so never see the padding. The
/// recurrent state is different: it is exported after the scan has run over
/// every compiled position, padding included. If padding advances it, the
/// state handed to decode depends on how much room the caller asked for.
#[test]
fn generation_is_independent_of_max_seq() {
    let (cfg, w) = model(99, false);
    let gen_with = |max_seq: usize| -> Vec<u32> {
        Qwen35RunnerBuilder::default()
            .inline_weights(cfg.clone(), w.clone())
            .device(dev())
            .fast_greedy_lm_head(false)
            .max_seq(max_seq)
            .build()
            .expect("build runner")
            .generate(&PROMPT, 3, |_| true)
            .expect("generate")
    };
    let tight = gen_with(PROMPT.len() + 3);
    let roomy = gen_with(PROMPT.len() + 3 + 8);
    assert_eq!(
        tight, roomy,
        "generation changed with max_seq ({:?} vs {:?}) — the decode state \
         depends on how many padded positions prefill scanned",
        tight, roomy
    );
}

/// Same weights, same prompt, same runner settings — twice.
///
/// If this ever fails, the decode path is reading something it did not
/// write, and every other comparison in this file is measuring noise.
#[test]
fn generation_is_deterministic() {
    let (cfg, w) = model(99, false);
    let run = || -> Vec<u32> {
        Qwen35RunnerBuilder::default()
            .inline_weights(cfg.clone(), w.clone())
            .device(dev())
            .fast_greedy_lm_head(false)
            .max_seq(PROMPT.len() + 4)
            .build()
            .expect("build runner")
            .generate(&PROMPT, 4, |_| true)
            .expect("generate")
    };
    assert_eq!(run(), run(), "decode is not deterministic");
}

/// The host LM head (`fast_greedy_lm_head`, the builder default) must score
/// decode steps exactly as the in-graph head does.
///
/// Every other test here turns the host head off. With it on, the decode
/// graph exported the hidden state *before* `output_norm`, which the host
/// head does not apply, so decode logits came out unnormalized — about 10×
/// too small on Qwen3.5-0.8B. With a unit `output_norm` (what `synth_weights`
/// builds) a missing norm is a uniform rescale and greedy argmax cannot see
/// it, so this uses a non-unit gamma and compares values, not argmax.
#[test]
fn host_lm_head_decode_matches_graph_lm_head() {
    let (cfg, mut w) = model(4, false);
    for (i, g) in w.output_norm.iter_mut().enumerate() {
        *g = 1.5 + hashed(0x77, i);
    }
    let decode_logits = |host_head: bool| -> Vec<Vec<f32>> {
        let mut r = Qwen35RunnerBuilder::default()
            .inline_weights(cfg.clone(), w.clone())
            .device(dev())
            .fast_greedy_lm_head(host_head)
            .max_seq(PROMPT.len() + N_NEW + 2)
            .build()
            .expect("build runner");
        let mut rows = vec![r.prefill_get_last_logits(&PROMPT).expect("prefill")];
        for t in [3u32, 7, 1] {
            rows.push(r.decode_get_logits(t).expect("decode"));
        }
        rows
    };
    let host = decode_logits(true);
    let graph = decode_logits(false);
    for (step, (h, g)) in host.iter().zip(&graph).enumerate() {
        let worst = h
            .iter()
            .zip(g)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            worst < 1e-4,
            "step {step}: host-head logits differ from graph-head logits by {worst}"
        );
    }
}
