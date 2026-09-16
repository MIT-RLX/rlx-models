// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! The checkpoint contract, checked against the real published tensor list —
//! **all 46 blocks**, from 52 KB of header data and no weights at all.
//!
//! `tests/real_weights.rs` runs two blocks on real values but needs a 337 MB
//! download to do it. This is the complement: it can say nothing about the
//! numbers, but it covers every block in the model for free, because a GGUF's
//! tensor *index* is header data. The fixture is the name/dims/type of all 1412
//! tensors in `unsloth/GLM-5.3-Flash-GGUF` UD-IQ1_S, read out of the two data
//! shards' headers.
//!
//! Three things get triangulated here, and it is the combination that matters:
//!
//! 1. [`Glm5NextConfig::block_tensor_names`] states the contract in config
//!    terms — "a KDA layer has these tensors, an MoE layer has those".
//! 2. Against the fixture, that prediction must reproduce the real file
//!    **exactly**: no missing tensor, and no invented one, for every block.
//! 3. Against the emitters, building a flow over a weight map keyed by exactly
//!    those names must consume all of them and ask for nothing else.
//!
//! (1)+(2) catches a wrong layer schedule; (2)+(3) catches the contract and the
//! emitters drifting apart. Either alone would let a whole class of error
//! through — the manifest could be a fiction that happens to match the emitters,
//! or the emitters could agree with a manifest that does not match reality.

use rlx_core::weight_map::WeightMap;
use rlx_glm5next::config::AttnKind;
use rlx_glm5next::{Glm5NextConfig, build_glm5next_text_flow};
use std::collections::{BTreeMap, BTreeSet};

const MANIFEST: &str = include_str!("fixtures/glm5_3_flash_tensors.tsv");
const HF_CONFIG: &str = include_str!("fixtures/glm5_3_flash_config.json");

/// `name -> (dims in GGML order, ggml type)`.
fn published() -> BTreeMap<String, (Vec<usize>, String)> {
    MANIFEST
        .lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
        .map(|l| {
            let mut f = l.split('\t');
            let name = f.next().expect("name").to_string();
            let dims = f
                .next()
                .expect("dims")
                .split(',')
                .map(|d| d.parse().expect("dim"))
                .collect();
            let ty = f.next().expect("type").to_string();
            (name, (dims, ty))
        })
        .collect()
}

fn cfg() -> Glm5NextConfig {
    Glm5NextConfig::from_hf_json(HF_CONFIG).expect("parse the published config")
}

#[test]
fn fixture_is_the_whole_checkpoint() {
    let p = published();
    assert_eq!(p.len(), 1412, "the published UD-IQ1_S set is 1412 tensors");
}

/// The contract must reproduce the real tensor list exactly, block by block.
///
/// Reported per block rather than as one set difference, so a failure names the
/// layer and the direction (missing vs invented) instead of dumping 1412 names.
#[test]
fn predicted_tensors_match_the_published_checkpoint() {
    let cfg = cfg();
    let p = published();

    // Group the real names by block.
    let mut real: BTreeMap<usize, BTreeSet<String>> = BTreeMap::new();
    let mut globals: BTreeSet<String> = BTreeSet::new();
    for name in p.keys() {
        match name.strip_prefix("blk.") {
            Some(rest) => {
                let (idx, suffix) = rest.split_once('.').expect("blk.N.suffix");
                real.entry(idx.parse().expect("block index"))
                    .or_default()
                    .insert(suffix.to_string());
            }
            None => {
                globals.insert(name.clone());
            }
        }
    }

    assert_eq!(
        globals,
        ["output.weight", "output_norm.weight", "token_embd.weight"]
            .map(String::from)
            .into_iter()
            .collect::<BTreeSet<_>>(),
        "unexpected non-block tensors"
    );
    assert_eq!(
        real.len(),
        cfg.block_count(),
        "config predicts {} blocks, the checkpoint has {}",
        cfg.block_count(),
        real.len()
    );

    for (block, actual) in &real {
        let want: BTreeSet<String> = cfg.block_tensor_names(*block).into_iter().collect();
        let missing: Vec<_> = want.difference(actual).collect();
        let extra: Vec<_> = actual.difference(&want).collect();
        assert!(
            missing.is_empty() && extra.is_empty(),
            "blk.{block}: predicted-but-absent {missing:?}, present-but-unpredicted {extra:?}"
        );
    }
}

/// The layer schedule the config derives must be the one the checkpoint's
/// tensors imply — `ssm_*` marks a KDA layer, `attn_kv_a_mqa` an MLA one.
///
/// This is the same fact `config_parsing` gets from `head_count_kv`, but read
/// off the *weights* instead of the metadata, so a converter that disagreed
/// with itself would show up.
#[test]
fn layer_schedule_agrees_with_the_tensors_present() {
    let cfg = cfg();
    let p = published();
    let has = |b: usize, s: &str| p.contains_key(&format!("blk.{b}.{s}"));

    let (mut kda, mut mla, mut moe, mut dense) = (0, 0, 0, 0);
    for i in 0..cfg.num_hidden_layers {
        let looks_kda = has(i, "ssm_a");
        let looks_mla = has(i, "attn_kv_a_mqa.weight");
        assert!(
            looks_kda ^ looks_mla,
            "blk.{i} is neither cleanly KDA nor cleanly MLA"
        );
        match cfg.attn_kind(i) {
            AttnKind::Kda => {
                assert!(looks_kda, "blk.{i}: config says KDA, tensors say MLA");
                kda += 1;
            }
            AttnKind::MlaDsa => {
                assert!(looks_mla, "blk.{i}: config says MLA, tensors say KDA");
                mla += 1;
            }
        }
        let looks_moe = has(i, "ffn_gate_exps.weight");
        assert_eq!(
            cfg.is_moe_layer(i),
            looks_moe,
            "blk.{i}: MoE disagreement between config and tensors"
        );
        if looks_moe { moe += 1 } else { dense += 1 }
    }
    assert_eq!((kda, mla), (34, 11), "GLM-5.3-Flash is 34 KDA + 11 MLA");
    assert_eq!((dense, moe), (3, 42), "3 leading dense layers, 42 MoE");
}

/// Shapes, for every block — not just the two that `real_weights` downloads.
///
/// The fixture keeps GGML's innermost-first order, which the loader reverses,
/// so these are written the way the file stores them.
#[test]
fn published_shapes_match_the_config() {
    let cfg = cfg();
    let p = published();
    let dims = |n: &str| p.get(n).map(|(d, _)| d.clone());
    let (h, kvl, nope, vd) = (
        cfg.hidden_size,
        cfg.kv_lora_rank,
        cfg.qk_nope_head_dim,
        cfg.v_head_dim,
    );

    assert_eq!(dims("token_embd.weight"), Some(vec![h, cfg.vocab_size]));
    assert_eq!(dims("output.weight"), Some(vec![h, cfg.vocab_size]));
    assert_eq!(dims("output_norm.weight"), Some(vec![h]));

    for i in 0..cfg.block_count() {
        let d = |s: &str| dims(&format!("blk.{i}.{s}"));
        assert_eq!(d("attn_norm.weight"), Some(vec![h]), "blk.{i}");

        if i < cfg.num_hidden_layers {
            // mHC: `fn` is [(2+H)·H, H·hidden] in torch terms, so GGML stores it
            // innermost-first as [H·hidden, (2+H)·H].
            assert_eq!(
                d("hc_attn_fn.weight"),
                Some(vec![cfg.hc_mult * h, cfg.hc_mix()]),
                "blk.{i} hc_attn_fn"
            );
            assert_eq!(d("hc_ffn_base.weight"), Some(vec![cfg.hc_mix()]), "blk.{i}");
            assert_eq!(d("hc_ffn_scale.weight"), Some(vec![3]), "blk.{i}");
        }

        let is_mla = i >= cfg.num_hidden_layers || cfg.attn_kind(i) == AttnKind::MlaDsa;
        if is_mla {
            // The two orientations that are not the same way round: GGML's
            // contraction axis is `nope` for k_b but `kv_lora` for v_b.
            assert_eq!(
                d("attn_k_b.weight"),
                Some(vec![nope, kvl, cfg.num_attention_heads]),
                "blk.{i} attn_k_b"
            );
            assert_eq!(
                d("attn_v_b.weight"),
                Some(vec![kvl, vd, cfg.num_attention_heads]),
                "blk.{i} attn_v_b"
            );
            assert_eq!(d("attn_kv_a_mqa.weight"), Some(vec![h, kvl]), "blk.{i}");
            assert_eq!(
                d("indexer.attn_k.weight"),
                Some(vec![h, cfg.index_head_dim]),
                "blk.{i}"
            );
            assert_eq!(
                d("indexer_compressor_ape.weight"),
                Some(vec![cfg.index_head_dim, cfg.index_kpool]),
                "blk.{i}"
            );
        } else {
            let proj = cfg.kda_proj();
            assert_eq!(d("attn_q.weight"), Some(vec![h, proj]), "blk.{i}");
            assert_eq!(
                d("ssm_conv1d_q.weight"),
                Some(vec![cfg.linear_conv_kernel_dim, 1, proj]),
                "blk.{i}"
            );
            assert_eq!(d("ssm_a"), Some(vec![cfg.linear_num_heads]), "blk.{i}");
            assert_eq!(
                d("ssm_norm.weight"),
                Some(vec![cfg.linear_head_dim]),
                "blk.{i}"
            );
        }

        if i >= cfg.num_hidden_layers || cfg.is_moe_layer(i) {
            assert_eq!(
                d("ffn_gate_exps.weight"),
                Some(vec![h, cfg.moe_intermediate_size, cfg.n_routed_experts]),
                "blk.{i} gate_exps"
            );
            assert_eq!(
                d("ffn_down_exps.weight"),
                Some(vec![cfg.moe_intermediate_size, h, cfg.n_routed_experts]),
                "blk.{i} down_exps"
            );
            assert_eq!(
                d("ffn_gate_inp.weight"),
                Some(vec![h, cfg.n_routed_experts]),
                "blk.{i}"
            );
        } else {
            assert_eq!(
                d("ffn_gate.weight"),
                Some(vec![h, cfg.intermediate_size]),
                "blk.{i}"
            );
            assert_eq!(
                d("ffn_down.weight"),
                Some(vec![cfg.intermediate_size, h]),
                "blk.{i}"
            );
        }
    }
}

/// The other side of the triangle: the **emitters** must ask for exactly the
/// names the contract predicts.
///
/// A weight map is built holding precisely `tensor_manifest()` (at toy widths,
/// with the real layer schedule); after building the flow, nothing may be left
/// over. Leftovers mean the manifest promises a tensor no emitter reads; a build
/// failure means an emitter reads one the manifest does not promise.
///
/// With one documented exception, asserted rather than waived: below the
/// `index_topk` budget the DSA short-circuit is the identity and the indexer is
/// never emitted, so its seven tensors per MLA layer go unread. Run past the
/// budget and they are all consumed.
#[test]
fn the_emitters_consume_exactly_the_predicted_tensors() {
    // Toy widths, real structure: one dense-KDA layer, one MoE-KDA layer, one
    // MLA+DSA layer — every block kind the flow emits.
    let mut c = cfg();
    c.hidden_size = 32;
    c.intermediate_size = 24;
    c.vocab_size = 16;
    c.num_attention_heads = 2;
    c.qk_nope_head_dim = 16;
    c.v_head_dim = 16;
    c.kv_lora_rank = 8;
    c.q_lora_rank = 12;
    c.index_n_heads = 2;
    c.index_head_dim = 8;
    c.linear_num_heads = 2;
    c.linear_head_dim = 16;
    c.n_routed_experts = 6;
    c.num_experts_per_tok = 2;
    c.moe_intermediate_size = 12;
    c.num_hidden_layers = 4;
    c.layer_types.truncate(4);
    c.indexer_types.truncate(4);
    // The MTP block is not emitted by the text flow, so leave it out of the map.
    c.num_nextn_predict_layers = 0;
    c.validate().expect("toy config");

    let shape_of = |name: &str| -> Vec<usize> {
        let s = name.rsplit('.').nth(1).unwrap_or("");
        let (h, proj) = (c.hidden_size, c.kda_proj());
        let qk = c.num_attention_heads * c.qk_nope_head_dim;
        match () {
            _ if name.ends_with("hc_attn_fn.weight") || name.ends_with("hc_ffn_fn.weight") => {
                vec![c.hc_mix(), c.hc_mult * h]
            }
            _ if name.ends_with("_base.weight") => vec![c.hc_mix()],
            _ if name.ends_with("_scale.weight") => vec![3],
            _ if name.ends_with("attn_norm.weight") || name.ends_with("ffn_norm.weight") => vec![h],
            _ if name.ends_with("output_norm.weight") => vec![h],
            _ if name.ends_with("token_embd.weight") || name.ends_with("output.weight") => {
                vec![c.vocab_size, h]
            }
            _ if name.ends_with("attn_q.weight")
                || name.ends_with("attn_k.weight")
                || name.ends_with("attn_v.weight") =>
            {
                vec![proj, h]
            }
            _ if name.ends_with("attn_output.weight") => {
                // KDA layers project from `proj`, MLA layers from `heads*v_dim`.
                let blk: usize = name
                    .strip_prefix("blk.")
                    .and_then(|r| r.split('.').next())
                    .and_then(|n| n.parse().ok())
                    .unwrap_or(0);
                if c.attn_kind(blk) == AttnKind::Kda {
                    vec![h, proj]
                } else {
                    vec![h, qk]
                }
            }
            _ if name.ends_with("ssm_conv1d_q.weight")
                || name.ends_with("ssm_conv1d_k.weight")
                || name.ends_with("ssm_conv1d_v.weight") =>
            {
                vec![proj, 1, c.linear_conv_kernel_dim]
            }
            _ if name.ends_with("ssm_a") => vec![c.linear_num_heads],
            _ if name.ends_with("ssm_dt.bias") => vec![proj],
            _ if name.ends_with("ssm_beta.weight") => vec![c.linear_num_heads, h],
            _ if name.ends_with("ssm_f_a.weight") || name.ends_with("ssm_g_a.weight") => {
                vec![c.linear_head_dim, h]
            }
            _ if name.ends_with("ssm_f_b.weight") || name.ends_with("ssm_g_b.weight") => {
                vec![proj, c.linear_head_dim]
            }
            _ if name.ends_with("ssm_norm.weight") => vec![c.linear_head_dim],
            _ if name.ends_with("attn_q_a.weight") => vec![c.q_lora_rank, h],
            _ if name.ends_with("attn_q_a_norm.weight") => vec![c.q_lora_rank],
            _ if name.ends_with("attn_q_b.weight") => {
                if name.contains("indexer") {
                    vec![c.index_n_heads * c.index_head_dim, c.q_lora_rank]
                } else {
                    vec![qk, c.q_lora_rank]
                }
            }
            _ if name.ends_with("attn_kv_a_mqa.weight") => vec![c.kv_lora_rank, h],
            _ if name.ends_with("attn_kv_a_norm.weight") => vec![c.kv_lora_rank],
            _ if name.ends_with("attn_k_b.weight") => {
                vec![c.num_attention_heads, c.kv_lora_rank, c.qk_nope_head_dim]
            }
            _ if name.ends_with("attn_v_b.weight") => {
                vec![c.num_attention_heads, c.v_head_dim, c.kv_lora_rank]
            }
            _ if name.ends_with("indexer.attn_k.weight") => vec![c.index_head_dim, h],
            _ if name.ends_with("indexer.k_norm.weight")
                || name.ends_with("indexer.k_norm.bias") =>
            {
                vec![c.index_head_dim]
            }
            _ if name.ends_with("indexer.proj.weight") => vec![c.index_n_heads, h],
            _ if name.ends_with("indexer_compressor_gate.weight") => vec![c.index_head_dim, h],
            _ if name.ends_with("indexer_compressor_ape.weight") => {
                vec![c.index_kpool, c.index_head_dim]
            }
            _ if name.ends_with("ffn_gate_inp.weight") => vec![c.n_routed_experts, h],
            _ if name.ends_with("exp_probs_b.bias") => vec![c.n_routed_experts],
            _ if name.ends_with("ffn_gate_exps.weight") || name.ends_with("ffn_up_exps.weight") => {
                vec![c.n_routed_experts, c.moe_intermediate_size, h]
            }
            _ if name.ends_with("ffn_down_exps.weight") => {
                vec![c.n_routed_experts, h, c.moe_intermediate_size]
            }
            _ if name.ends_with("ffn_gate_shexp.weight")
                || name.ends_with("ffn_up_shexp.weight") =>
            {
                vec![c.moe_intermediate_size, h]
            }
            _ if name.ends_with("ffn_down_shexp.weight") => vec![h, c.moe_intermediate_size],
            _ if name.ends_with("ffn_gate.weight") || name.ends_with("ffn_up.weight") => {
                vec![c.intermediate_size, h]
            }
            _ if name.ends_with("ffn_down.weight") => vec![h, c.intermediate_size],
            _ => panic!("no toy shape for {name} (suffix {s})"),
        }
    };

    let names = c.tensor_manifest();
    let build = |c: &Glm5NextConfig, seq: usize| -> BTreeSet<String> {
        let mut wm = WeightMap::from_tensors(
            names
                .iter()
                .map(|n| {
                    let shape = shape_of(n);
                    let len: usize = shape.iter().product();
                    // 0.02 keeps the MoE router's sigmoid off its saturated ends.
                    (n.clone(), (vec![0.02f32; len], shape))
                })
                .collect(),
        );
        build_glm5next_text_flow(c, &mut wm, seq, true).expect("build over the predicted manifest");
        wm.keys().map(|s| s.to_string()).collect()
    };

    // ── below the budget: DSA is the identity, so no indexer is emitted ──
    let dense_leftover = build(&c, 8);
    let indexer_tensors: BTreeSet<String> = names
        .iter()
        .filter(|n| n.contains("indexer"))
        .cloned()
        .collect();
    assert!(
        !indexer_tensors.is_empty(),
        "the toy config should still have an MLA layer"
    );
    assert_eq!(
        dense_leftover, indexer_tensors,
        "below index_topk exactly the indexer tensors should go unread"
    );

    // ── past the budget: the indexer runs, so everything is consumed ──
    let mut sparse = c.clone();
    sparse.index_topk = 8; // 2 pools of 4 out of 4 → selection actually binds
    let sparse_leftover = build(&sparse, 16);
    assert!(
        sparse_leftover.is_empty(),
        "the manifest promises tensors no emitter reads: {sparse_leftover:?}"
    );
}
