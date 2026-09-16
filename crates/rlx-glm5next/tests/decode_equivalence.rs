// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! The decode graph, stepped one token at a time, must reproduce the prefill
//! graph's logits for the same prompt.
//!
//! This is the load-bearing test for the whole decode path, because it closes
//! over every piece of carried state at once:
//!
//! * the KDA short conv's left-pad, which becomes a carried `[k-1, proj]` state
//!   instead of zeros;
//! * the KDA delta-net scan state, resumed rather than started from zero;
//! * the MLA latent KV cache and its validity mask.
//!
//! It also checks something subtler for free: prefill runs MLA **un-absorbed**
//! (expand the latent to per-head keys/values, fused `Op::Attention`) while
//! decode runs it **absorbed** (project the query into latent space, attend
//! against the cached latent directly). Those are algebraically identical but
//! textually share no code, so agreement here means both readings of GGUF's
//! `attn_k_b` / `attn_v_b` orientations are right.
//!
//! mHC carries nothing across tokens — it mixes across depth, not time — so a
//! divergence here cannot be blamed on it.

use rlx_core::flow_util::compile_built;
use rlx_core::weight_map::WeightMap;
use rlx_glm5next::config::{AttnKind, IndexerKind};
use rlx_glm5next::{
    DecodeNames, DecodeSession, Glm5NextConfig, ScanState, build_glm5next_decode_flow_with,
    build_glm5next_text_flow,
};
use rlx_runtime::Device;
use std::collections::HashMap;

const HIDDEN: usize = 32;
const HEADS: usize = 2;
const NOPE: usize = 16;
const KV_LORA: usize = 8;
const Q_LORA: usize = 12;
const KDA_HEADS: usize = 2;
const KDA_HD: usize = 16;
const CONV_K: usize = 4;
const IDX_HEADS: usize = 2;
const IDX_HD: usize = 8;
const KPOOL: usize = 4;
const EXPERTS: usize = 6;
const TOPK: usize = 2;
const MOE_INTER: usize = 12;
const INTER: usize = 24;
const VOCAB: usize = 16;
const HC: usize = 4;
const SEQ: usize = 6;

fn dev() -> Device {
    std::env::var("RLX_TEST_DEVICE")
        .ok()
        .filter(|s| !s.is_empty())
        .map(|s| rlx_cli::parse_device(&s).expect("bad RLX_TEST_DEVICE"))
        .unwrap_or(Device::Cpu)
}

fn fill(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (((s >> 33) as f32) / (u32::MAX as f32) - 0.5) * 0.5
        })
        .collect()
}

fn cfg_with(layer_types: Vec<AttnKind>, first_dense: usize) -> Glm5NextConfig {
    let n = layer_types.len();
    Glm5NextConfig {
        vocab_size: VOCAB,
        hidden_size: HIDDEN,
        intermediate_size: INTER,
        num_hidden_layers: n,
        num_attention_heads: HEADS,
        rms_norm_eps: 1e-5,
        max_position_embeddings: 1024,
        tie_word_embeddings: false,
        layer_types,
        indexer_types: vec![IndexerKind::Full; n],
        first_k_dense_replace: first_dense,
        q_lora_rank: Q_LORA,
        kv_lora_rank: KV_LORA,
        qk_nope_head_dim: NOPE,
        qk_rope_head_dim: 0,
        v_head_dim: NOPE,
        index_n_heads: IDX_HEADS,
        index_head_dim: IDX_HD,
        // Well above SEQ, so DSA is the identity and prefill takes the fused
        // causal path — the regime decode is defined for.
        index_topk: 2048,
        index_kpool: KPOOL,
        index_kpool_always_select_tail: true,
        linear_num_heads: KDA_HEADS,
        linear_head_dim: KDA_HD,
        linear_conv_kernel_dim: CONV_K,
        linear_lower_bound: Some(-5.0),
        hc_mult: HC,
        hc_sinkhorn_iters: 20,
        hc_eps: 1e-6,
        n_routed_experts: EXPERTS,
        num_experts_per_tok: TOPK,
        n_shared_experts: 1,
        moe_intermediate_size: MOE_INTER,
        n_group: 1,
        topk_group: 1,
        routed_scaling_factor: 2.5,
        norm_topk_prob: true,
        swiglu_limit: 10.0,
        num_nextn_predict_layers: 1,
        with_mtp: false,
    }
}

fn cfg() -> Glm5NextConfig {
    cfg_with(
        vec![
            AttnKind::Kda,
            AttnKind::Kda,
            AttnKind::Kda,
            AttnKind::MlaDsa,
        ],
        3,
    )
}

fn weights(c: &Glm5NextConfig) -> WeightMap {
    let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    let mut seed = 1u64;
    let mut put = |t: &mut HashMap<_, _>, k: String, shape: Vec<usize>| {
        let n: usize = shape.iter().product();
        seed += 7;
        t.insert(k, (fill(n, seed), shape));
    };

    put(&mut t, "token_embd.weight".into(), vec![VOCAB, HIDDEN]);
    put(&mut t, "output.weight".into(), vec![VOCAB, HIDDEN]);
    t.insert(
        "output_norm.weight".into(),
        (vec![1.0; HIDDEN], vec![HIDDEN]),
    );

    let kda_proj = KDA_HEADS * KDA_HD;
    let qk = HEADS * NOPE;
    let hc_mix = (2 + HC) * HC;

    for i in 0..c.num_hidden_layers {
        let b = format!("blk.{i}");
        for n in ["attn_norm", "ffn_norm"] {
            t.insert(format!("{b}.{n}.weight"), (vec![1.0; HIDDEN], vec![HIDDEN]));
        }
        for site in ["hc_attn", "hc_ffn"] {
            put(
                &mut t,
                format!("{b}.{site}_fn.weight"),
                vec![hc_mix, HC * HIDDEN],
            );
            put(&mut t, format!("{b}.{site}_base.weight"), vec![hc_mix]);
            t.insert(
                format!("{b}.{site}_scale.weight"),
                (vec![0.8, 1.2, -0.5], vec![3]),
            );
        }

        match c.attn_kind(i) {
            AttnKind::Kda => {
                for p in ["attn_q", "attn_k", "attn_v"] {
                    put(&mut t, format!("{b}.{p}.weight"), vec![kda_proj, HIDDEN]);
                }
                put(
                    &mut t,
                    format!("{b}.attn_output.weight"),
                    vec![HIDDEN, kda_proj],
                );
                for cv in ["ssm_conv1d_q", "ssm_conv1d_k", "ssm_conv1d_v"] {
                    put(
                        &mut t,
                        format!("{b}.{cv}.weight"),
                        vec![kda_proj, 1, CONV_K],
                    );
                }
                put(&mut t, format!("{b}.ssm_a"), vec![KDA_HEADS]);
                put(&mut t, format!("{b}.ssm_dt.bias"), vec![kda_proj]);
                put(
                    &mut t,
                    format!("{b}.ssm_beta.weight"),
                    vec![KDA_HEADS, HIDDEN],
                );
                put(&mut t, format!("{b}.ssm_f_a.weight"), vec![KDA_HD, HIDDEN]);
                put(
                    &mut t,
                    format!("{b}.ssm_f_b.weight"),
                    vec![kda_proj, KDA_HD],
                );
                put(&mut t, format!("{b}.ssm_g_a.weight"), vec![KDA_HD, HIDDEN]);
                put(
                    &mut t,
                    format!("{b}.ssm_g_b.weight"),
                    vec![kda_proj, KDA_HD],
                );
                t.insert(
                    format!("{b}.ssm_norm.weight"),
                    (vec![1.0; KDA_HD], vec![KDA_HD]),
                );
            }
            AttnKind::MlaDsa => {
                put(&mut t, format!("{b}.attn_q_a.weight"), vec![Q_LORA, HIDDEN]);
                t.insert(
                    format!("{b}.attn_q_a_norm.weight"),
                    (vec![1.0; Q_LORA], vec![Q_LORA]),
                );
                put(&mut t, format!("{b}.attn_q_b.weight"), vec![qk, Q_LORA]);
                put(
                    &mut t,
                    format!("{b}.attn_kv_a_mqa.weight"),
                    vec![KV_LORA, HIDDEN],
                );
                t.insert(
                    format!("{b}.attn_kv_a_norm.weight"),
                    (vec![1.0; KV_LORA], vec![KV_LORA]),
                );
                put(
                    &mut t,
                    format!("{b}.attn_k_b.weight"),
                    vec![HEADS, KV_LORA, NOPE],
                );
                put(
                    &mut t,
                    format!("{b}.attn_v_b.weight"),
                    vec![HEADS, NOPE, KV_LORA],
                );
                put(&mut t, format!("{b}.attn_output.weight"), vec![HIDDEN, qk]);
                put(
                    &mut t,
                    format!("{b}.indexer.attn_q_b.weight"),
                    vec![IDX_HEADS * IDX_HD, Q_LORA],
                );
                put(
                    &mut t,
                    format!("{b}.indexer.attn_k.weight"),
                    vec![IDX_HD, HIDDEN],
                );
                t.insert(
                    format!("{b}.indexer.k_norm.weight"),
                    (vec![1.0; IDX_HD], vec![IDX_HD]),
                );
                t.insert(
                    format!("{b}.indexer.k_norm.bias"),
                    (vec![0.0; IDX_HD], vec![IDX_HD]),
                );
                put(
                    &mut t,
                    format!("{b}.indexer.proj.weight"),
                    vec![IDX_HEADS, HIDDEN],
                );
                put(
                    &mut t,
                    format!("{b}.indexer_compressor_gate.weight"),
                    vec![IDX_HD, HIDDEN],
                );
                put(
                    &mut t,
                    format!("{b}.indexer_compressor_ape.weight"),
                    vec![KPOOL, IDX_HD],
                );
            }
        }

        if c.is_moe_layer(i) {
            put(
                &mut t,
                format!("{b}.ffn_gate_inp.weight"),
                vec![EXPERTS, HIDDEN],
            );
            put(&mut t, format!("{b}.exp_probs_b.bias"), vec![EXPERTS]);
            for n in ["ffn_gate_exps", "ffn_up_exps"] {
                put(
                    &mut t,
                    format!("{b}.{n}.weight"),
                    vec![EXPERTS, MOE_INTER, HIDDEN],
                );
            }
            put(
                &mut t,
                format!("{b}.ffn_down_exps.weight"),
                vec![EXPERTS, HIDDEN, MOE_INTER],
            );
            put(
                &mut t,
                format!("{b}.ffn_gate_shexp.weight"),
                vec![MOE_INTER, HIDDEN],
            );
            put(
                &mut t,
                format!("{b}.ffn_up_shexp.weight"),
                vec![MOE_INTER, HIDDEN],
            );
            put(
                &mut t,
                format!("{b}.ffn_down_shexp.weight"),
                vec![HIDDEN, MOE_INTER],
            );
        } else {
            put(&mut t, format!("{b}.ffn_gate.weight"), vec![INTER, HIDDEN]);
            put(&mut t, format!("{b}.ffn_up.weight"), vec![INTER, HIDDEN]);
            put(&mut t, format!("{b}.ffn_down.weight"), vec![HIDDEN, INTER]);
        }
    }
    WeightMap::from_tensors(t)
}

fn prompt() -> Vec<f32> {
    (0..SEQ).map(|i| ((i * 5 + 3) % VOCAB) as f32).collect()
}

/// Prefill the whole prompt in one graph; returns `[SEQ, VOCAB]`.
fn prefill_logits(c: &Glm5NextConfig) -> Vec<f32> {
    let mut w = weights(c);
    let built = build_glm5next_text_flow(c, &mut w, SEQ, true).expect("build prefill");
    let mut compiled = compile_built(built, dev()).expect("compile prefill");
    let ids = prompt();
    let mut outs = compiled.run(&[("input_ids", ids.as_slice())]);
    outs.pop().expect("logits")
}

/// Step the decode graph over the same prompt; returns `[SEQ, VOCAB]`.
fn decode_logits(c: &Glm5NextConfig, scan_mode: ScanState) -> Vec<f32> {
    let mut w = weights(c);
    let cap = SEQ;
    let (built, layout) =
        build_glm5next_decode_flow_with(c, &mut w, cap, true, scan_mode).expect("build decode");
    let mut compiled = compile_built(built, dev()).expect("compile decode");

    let names = DecodeNames::new(c);
    let mut session = DecodeSession::new(c, layout, cap);
    let ids = prompt();
    let mut all = Vec::with_capacity(SEQ * VOCAB);
    for &id in &ids {
        let tok = [id];
        let mut binds: Vec<(&str, &[f32])> = vec![("input_ids", &tok[..])];
        binds.extend(session.inputs(c, &names));
        let mut outs = compiled.run(&binds);
        let packed = outs.pop().expect("packed");
        let logits = session.commit(&packed).expect("commit");
        all.extend_from_slice(logits);
    }
    all
}

fn compare(a: &[f32], b: &[f32], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: length mismatch");
    let worst = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max);
    let scale = a.iter().map(|v| v.abs()).fold(1e-3, f32::max);
    assert!(
        worst / scale < 5e-4,
        "{what}: decode diverges from prefill; max |Δ| = {worst} over scale {scale}"
    );
}

/// The whole point: stepping the decode graph reproduces prefill.
#[test]
fn decode_matches_prefill_portable_scan() {
    let c = cfg();
    let want = prefill_logits(&c);
    let got = decode_logits(&c, ScanState::Portable);
    compare(&got, &want, "portable scan");
}

/// `ScanState::InPlace` keeps the delta-net state in a param that the op mutates
/// instead of round-tripping it through graph I/O. It must produce the same
/// answer — a backend where the in-place update does not survive to the next
/// `run()` fails here loudly instead of drifting in production.
#[test]
fn decode_matches_prefill_in_place_scan() {
    let c = cfg();
    let want = prefill_logits(&c);
    let got = decode_logits(&c, ScanState::InPlace);
    compare(&got, &want, "in-place scan");
}

/// The two scan modes must agree with each other, not merely both be close to
/// prefill.
#[test]
fn scan_modes_agree() {
    let c = cfg();
    let portable = decode_logits(&c, ScanState::Portable);
    let in_place = decode_logits(&c, ScanState::InPlace);
    compare(&portable, &in_place, "portable vs in-place");
}

/// `InPlace` drops the scan state from the packed output; `Portable` carries it.
/// That difference is the bulk of decode's host traffic.
#[test]
fn in_place_scan_removes_the_state_round_trip() {
    let c = cfg();
    let mut w = weights(&c);
    let (_, portable) =
        build_glm5next_decode_flow_with(&c, &mut w, SEQ, true, ScanState::Portable).unwrap();
    let mut w = weights(&c);
    let (_, in_place) =
        build_glm5next_decode_flow_with(&c, &mut w, SEQ, true, ScanState::InPlace).unwrap();

    let scan_w = KDA_HEADS * KDA_HD * KDA_HD;
    let n_kda = c
        .layer_types
        .iter()
        .filter(|k| **k == AttnKind::Kda)
        .count();
    assert_eq!(portable.total - in_place.total, n_kda * scan_w);

    let sess_p = DecodeSession::new(&c, portable, SEQ);
    let sess_i = DecodeSession::new(&c, in_place, SEQ);
    assert!(
        sess_i.state_bytes() < sess_p.state_bytes(),
        "in-place must move less state per token"
    );
}

/// Decode is only defined where DSA selection is the identity. Past the budget
/// it must refuse rather than quietly run dense attention, which would be a
/// different model from the trained one.
#[test]
fn decode_refuses_a_capacity_past_the_dsa_budget() {
    let mut c = cfg();
    c.index_topk = 8; // budget now smaller than the requested cache
    let mut w = weights(&c);
    let err = build_glm5next_decode_flow_with(&c, &mut w, 32, true, ScanState::Portable)
        .unwrap_err()
        .to_string();
    assert!(err.contains("index_topk"), "unexpected error: {err}");
}

/// Running past the compiled capacity is an error, not a silently dropped token.
#[test]
fn decode_refuses_to_run_past_capacity() {
    let c = cfg();
    let mut w = weights(&c);
    let cap = 2;
    let (built, layout) =
        build_glm5next_decode_flow_with(&c, &mut w, cap, true, ScanState::Portable).unwrap();
    let mut compiled = compile_built(built, dev()).expect("compile");
    let names = DecodeNames::new(&c);
    let mut session = DecodeSession::new(&c, layout, cap);

    let mut last = Ok(());
    for step in 0..(cap + 1) {
        let tok = [1.0f32];
        let mut binds: Vec<(&str, &[f32])> = vec![("input_ids", &tok[..])];
        binds.extend(session.inputs(&c, &names));
        let mut outs = compiled.run(&binds);
        let packed = outs.pop().expect("packed");
        last = session.commit(&packed).map(|_| ());
        if step < cap {
            assert!(last.is_ok(), "step {step} should fit the cache");
        }
    }
    let err = last.unwrap_err().to_string();
    assert!(err.contains("capacity"), "unexpected error: {err}");
}

/// One KDA layer, dense FFN: isolates the short-conv carry and the delta-net
/// scan resume from everything else.
#[test]
fn decode_matches_prefill_kda_only() {
    let c = cfg_with(vec![AttnKind::Kda], 1);
    let want = prefill_logits(&c);
    let got = decode_logits(&c, ScanState::Portable);
    compare(&got, &want, "kda only");
}

/// One MLA layer, dense FFN: isolates the latent KV cache and the absorbed vs
/// un-absorbed reading of `attn_k_b` / `attn_v_b`.
#[test]
fn decode_matches_prefill_mla_only() {
    let c = cfg_with(vec![AttnKind::MlaDsa], 1);
    let want = prefill_logits(&c);
    let got = decode_logits(&c, ScanState::Portable);
    compare(&got, &want, "mla only");
}

/// One MoE layer on top of a KDA attention, to separate the FFN from the
/// attention paths.
#[test]
fn decode_matches_prefill_moe_layer() {
    let c = cfg_with(vec![AttnKind::Kda], 0);
    let want = prefill_logits(&c);
    let got = decode_logits(&c, ScanState::Portable);
    compare(&got, &want, "kda + moe");
}
