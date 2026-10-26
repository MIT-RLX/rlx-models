// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! The MTP draft block, and the residual seam it hangs off.
//!
//! Two claims get checked, because each is silent when wrong:
//!
//! 1. **`HcResidual` + the head flow == `Logits`.** A speculative loop wants
//!    the trunk's residual (for the draft block) *and* the trunk's logits. If
//!    splitting the trunk at that seam changed the logits, every acceptance
//!    decision would be made against a model the trunk is not.
//! 2. **The MTP block's tensor contract is the published `mtp-*.gguf`'s.** The
//!    draft block is a separate artifact with its own `block_count = 49`, so
//!    `block_tensor_names` has to predict exactly its 32 tensors — no more, no
//!    fewer.

use rlx_core::flow_util::{WeightMapSource, compile_built};
use rlx_core::weight_map::WeightMap;
use rlx_qwen4exp::mtp::{
    HC_RESIDUAL_INPUT, build_qwen4exp_head_flow_with, build_qwen4exp_mtp_flow_with,
};
use rlx_qwen4exp::rope::{mrope_tables, table_width};
use rlx_qwen4exp::{
    FlowOptions, OutputKind, Qwen4ExpConfig, build_qwen4exp_text_flow_with_options,
    ngram_embeddings, ple_embed_input, synth,
};
use rlx_runtime::Device;
use std::collections::BTreeSet;

const TOKENS: [f32; 4] = [3.0, 11.0, 7.0, 2.0];

/// The `blk.48` tensor list of `unsloth/Qwen3.8-Flash-Next-GGUF/MTP`, read out
/// of that file's header.
const PUBLISHED_MTP_BLOCK: &[&str] = &[
    "attn_k.weight",
    "attn_k_norm.weight",
    "attn_output.weight",
    "attn_q.weight",
    "attn_q_norm.weight",
    "attn_v.weight",
    "ffn_down_exps.weight",
    "ffn_down_shexp.weight",
    "ffn_gate_exps.weight",
    "ffn_gate_inp.weight",
    "ffn_gate_inp_shexp.weight",
    "ffn_gate_shexp.weight",
    "ffn_up_exps.weight",
    "ffn_up_shexp.weight",
    "hc_attn_down.weight",
    "hc_attn_inject.weight",
    "hc_attn_norm.weight",
    "hc_attn_up.weight",
    "hc_ffn_down.weight",
    "hc_ffn_inject.weight",
    "hc_ffn_norm.weight",
    "hc_ffn_up.weight",
    "indexer.k_norm.weight",
    "indexer.k_proj.weight",
    "indexer.q_norm.weight",
    "indexer.q_proj.weight",
    "nextn.eh_proj.weight",
    "nextn.enorm.weight",
    "nextn.hc_head_down.weight",
    "nextn.hc_head_norm.weight",
    "nextn.hc_head_up.weight",
    "nextn.hnorm.weight",
];

fn dev() -> Device {
    rlx_ir::env::var_result("RLX_TEST_DEVICE")
        .ok()
        .filter(|s| !s.is_empty())
        .map(|s| rlx_cli::parse_device(&s).expect("bad RLX_TEST_DEVICE"))
        .unwrap_or(Device::Cpu)
}

/// The tiny config with an MTP block declared, and a 5th layer to carry it.
fn cfg_with_mtp() -> Qwen4ExpConfig {
    let mut c = synth::tiny_config(2048);
    c.num_nextn_predict_layers = 1;
    c
}

/// Tensors for the trunk plus `blk.{num_hidden_layers}`, the MTP block.
fn weights(cfg: &Qwen4ExpConfig) -> WeightMap {
    let mut with_mtp = cfg.clone();
    with_mtp.with_mtp = true;
    let mut t = synth::synth_tensors(&with_mtp, 5);
    t.remove("per_layer_token_embd.weight");
    WeightMap::from_tensors(t)
}

#[test]
fn the_mtp_block_contract_is_the_published_one() {
    let cfg = Qwen4ExpConfig::qwen38_flash_next();
    let got: BTreeSet<String> = cfg
        .block_tensor_names(cfg.num_hidden_layers)
        .into_iter()
        .collect();
    let want: BTreeSet<String> = PUBLISHED_MTP_BLOCK.iter().map(|s| s.to_string()).collect();
    let missing: Vec<_> = want.difference(&got).collect();
    let extra: Vec<_> = got.difference(&want).collect();
    assert!(
        missing.is_empty() && extra.is_empty(),
        "blk.{}: predicted-but-absent {missing:?}, present-but-unpredicted {extra:?}",
        cfg.num_hidden_layers
    );
    assert_eq!(got.len(), 32);

    // The MTP block is a QSA attention layer whatever the trunk's schedule
    // says at that index — layer 48 would be `linear` by the interval rule.
    assert!(got.contains("indexer.q_proj.weight"));
    assert!(!got.contains("ssm_a"));
    // And it carries no `output_hc_*`: the draft head has its own mixer.
    assert!(got.iter().all(|n| !n.starts_with("output_hc")));
}

#[test]
fn an_mtp_block_needs_a_config_that_declares_one() {
    let cfg = synth::tiny_config(2048);
    assert_eq!(cfg.num_nextn_predict_layers, 0);
    let mut w = weights(&cfg);
    let err =
        build_qwen4exp_mtp_flow_with(&cfg, &mut WeightMapSource(&mut w), 2, OutputKind::Logits)
            .unwrap_err()
            .to_string();
    assert!(err.contains("nextn_predict_layers"), "{err}");
}

/// Splitting the trunk at the residual must not change its logits.
#[test]
fn the_residual_seam_reproduces_the_trunk_logits() {
    let cfg = cfg_with_mtp();
    let seq = TOKENS.len();
    let vocab = cfg.vocab_size;
    let hash = rlx_qwen4exp::PleHash::derive(&cfg, 0);
    let ids: Vec<u32> = TOKENS.iter().map(|t| *t as u32).collect();
    let emb = ngram_embeddings(&hash, &ids, &[], cfg.ple_head_dim, |r| {
        Ok((0..cfg.ple_head_dim)
            .map(|j| (((r as usize + j) % 83) as f32 - 41.0) * 0.003)
            .collect())
    })
    .expect("ple rows");

    let run_trunk = |output: OutputKind| -> Vec<f32> {
        let mut w = weights(&cfg);
        let mut opts = FlowOptions::prefill(seq);
        opts.output = output;
        let built = build_qwen4exp_text_flow_with_options(&cfg, &mut WeightMapSource(&mut w), opts)
            .expect("build trunk");
        compile_built(built, dev())
            .expect("compile trunk")
            .run(&[
                ("input_ids", TOKENS.as_slice()),
                (&ple_embed_input(0), emb.as_slice()),
            ])
            .pop()
            .expect("trunk output")
    };

    let direct = run_trunk(OutputKind::Logits);
    let residual = run_trunk(OutputKind::HcResidual);
    assert_eq!(direct.len(), seq * vocab);
    assert_eq!(residual.len(), seq * cfg.hc_dim());

    let mut w = weights(&cfg);
    let head =
        build_qwen4exp_head_flow_with(&cfg, &mut WeightMapSource(&mut w), seq).expect("build head");
    let via_seam = compile_built(head, dev())
        .expect("compile head")
        .run(&[(HC_RESIDUAL_INPUT, residual.as_slice())])
        .pop()
        .expect("head output");

    assert_eq!(via_seam.len(), direct.len());
    let max_abs = direct
        .iter()
        .zip(&via_seam)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let scale = direct.iter().map(|v| v.abs()).fold(1e-3f32, f32::max);
    assert!(
        max_abs / scale < 1e-5,
        "the residual seam changed the trunk's logits by {max_abs} ({:.2e} relative)",
        max_abs / scale
    );
}

/// The draft block runs, produces finite logits, and hands back a residual a
/// second draft step could consume.
#[test]
fn the_draft_block_runs_and_chains() {
    let cfg = cfg_with_mtp();
    let seq = TOKENS.len();
    let hc_dim = cfg.hc_dim();
    // The MTP flow bakes its own M-RoPE tables; this is the shape contract they
    // must satisfy.
    let (cos, _sin) = mrope_tables(&cfg, seq, None);
    assert_eq!(cos.len(), seq * table_width(&cfg));
    let residual: Vec<f32> = (0..seq * hc_dim)
        .map(|i| ((i as f32) * 0.021).sin() * 0.2)
        .collect();

    let run = |output: OutputKind| -> Vec<f32> {
        let mut w = weights(&cfg);
        let built = build_qwen4exp_mtp_flow_with(&cfg, &mut WeightMapSource(&mut w), seq, output)
            .expect("build mtp");
        compile_built(built, dev())
            .expect("compile mtp")
            .run(&[
                ("input_ids", TOKENS.as_slice()),
                (HC_RESIDUAL_INPUT, residual.as_slice()),
            ])
            .pop()
            .expect("mtp output")
    };

    let logits = run(OutputKind::Logits);
    assert_eq!(logits.len(), seq * cfg.vocab_size);
    assert!(logits.iter().all(|v| v.is_finite()));

    let next_h = run(OutputKind::HcResidual);
    assert_eq!(next_h.len(), seq * hc_dim);
    assert!(next_h.iter().all(|v| v.is_finite()));
    // The draft residual must actually differ from its input — an eh_proj or a
    // hyper-connection wired to a no-op would return the input unchanged.
    let delta = next_h
        .iter()
        .zip(&residual)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(delta > 1e-4, "the draft block did not change the residual");

    // Chaining: the second step reads the first's residual and stays finite.
    let mut w = weights(&cfg);
    let built =
        build_qwen4exp_mtp_flow_with(&cfg, &mut WeightMapSource(&mut w), seq, OutputKind::Logits)
            .expect("build mtp");
    let step2 = compile_built(built, dev())
        .expect("compile mtp")
        .run(&[
            ("input_ids", TOKENS.as_slice()),
            (HC_RESIDUAL_INPUT, next_h.as_slice()),
        ])
        .pop()
        .expect("mtp step 2");
    assert!(step2.iter().all(|v| v.is_finite()));
}
