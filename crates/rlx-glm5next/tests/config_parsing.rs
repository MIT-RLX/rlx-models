// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! `Glm5NextConfig` against the two shapes GLM-5.3-Flash actually ships in.
//!
//! The fixture is `zai-org/GLM-5.3-Flash`'s real `config.json` (only the
//! 1509-entry fp8 `modules_to_not_convert` list, which says nothing about the
//! architecture, is stripped). The GGUF case is built from the metadata keys
//! read out of `unsloth/GLM-5.3-Flash-GGUF`'s first shard, which carries all 72
//! KV pairs and zero tensors.
//!
//! Both readers must land on the same 45-layer model, which is the point: the
//! GGUF converter renames nearly every field and folds `layer_types` into an
//! integer `head_count_kv` array.

use rlx_gguf::{GgufFile, MetaValue};
use rlx_glm5next::Glm5NextConfig;
use rlx_glm5next::config::{AttnKind, IndexerKind};

const HF_CONFIG: &str = include_str!("fixtures/glm5_3_flash_config.json");

/// The published `unsloth/GLM-5.3-Flash-GGUF` metadata, verbatim.
fn published_gguf() -> GgufFile {
    let mut g = GgufFile::empty();
    let m = &mut g.metadata;
    let mut s = |k: &str, v: &str| {
        m.insert(k.into(), MetaValue::String(v.into()));
    };
    s("general.architecture", "glm5next");
    s("general.name", "GLM 5.3 Flash");

    let mut u = |k: &str, v: u32| {
        g.metadata.insert(k.into(), MetaValue::U32(v));
    };
    u("glm5next.block_count", 46);
    u("glm5next.context_length", 1_048_576);
    u("glm5next.embedding_length", 4096);
    u("glm5next.feed_forward_length", 12288);
    u("glm5next.attention.head_count", 64);
    u("glm5next.expert_count", 288);
    u("glm5next.expert_used_count", 8);
    u("glm5next.expert_group_count", 1);
    u("glm5next.expert_group_used_count", 1);
    u("glm5next.expert_gating_func", 2);
    u("glm5next.vocab_size", 154_880);
    u("glm5next.attention.q_lora_rank", 1536);
    u("glm5next.attention.kv_lora_rank", 512);
    u("glm5next.rope.dimension_count", 0);
    u("glm5next.attention.key_length", 512);
    u("glm5next.attention.value_length", 512);
    u("glm5next.attention.key_length_mla", 256);
    u("glm5next.attention.value_length_mla", 256);
    u("glm5next.ssm.conv_kernel", 4);
    u("glm5next.kda.head_dim", 128);
    u("glm5next.attention.indexer.head_count", 32);
    u("glm5next.attention.indexer.key_length", 128);
    u("glm5next.attention.indexer.top_k", 2048);
    u("glm5next.attention.indexer.kpool", 4);
    u("glm5next.hyper_connection.count", 4);
    u("glm5next.hyper_connection.sinkhorn_iterations", 20);
    u("glm5next.expert_feed_forward_length", 2048);
    u("glm5next.expert_shared_feed_forward_length", 2048);
    u("glm5next.expert_shared_count", 1);
    u("glm5next.leading_dense_block_count", 3);
    u("glm5next.nextn_predict_layers", 1);

    let mut f = |k: &str, v: f32| {
        g.metadata.insert(k.into(), MetaValue::F32(v));
    };
    f("glm5next.attention.layer_norm_rms_epsilon", 1e-5);
    f("glm5next.attention.layer_norm_epsilon", 1e-6);
    f("glm5next.hyper_connection.epsilon", 1e-6);
    f("glm5next.kda.gate_lower_bound", -5.0);
    f("glm5next.expert_weights_scale", 2.5);

    g.metadata
        .insert("glm5next.expert_weights_norm".into(), MetaValue::Bool(true));

    // 0 = KDA, 1 = MLA, on the shipped 3-then-1 cycle, for all 46 blocks.
    let kv: Vec<MetaValue> = (0..46)
        .map(|i| MetaValue::U32(u32::from(i % 4 == 3)))
        .collect();
    g.metadata.insert(
        "glm5next.attention.head_count_kv".into(),
        MetaValue::Array(kv),
    );
    g.metadata.insert(
        "glm5next.swiglu_clamp_exp".into(),
        MetaValue::Array(vec![MetaValue::F32(10.0); 46]),
    );
    g
}

fn assert_is_glm_5_3_flash(c: &Glm5NextConfig) {
    assert_eq!(c.vocab_size, 154_880);
    assert_eq!(c.hidden_size, 4096);
    assert_eq!(c.intermediate_size, 12288);
    assert_eq!(c.num_attention_heads, 64);
    assert_eq!(
        c.num_hidden_layers, 45,
        "the MTP block is not a decoder layer"
    );
    assert_eq!(c.num_nextn_predict_layers, 1);
    assert_eq!(c.block_count(), 46, "…but it is a block in the checkpoint");

    assert_eq!(c.q_lora_rank, 1536);
    assert_eq!(c.kv_lora_rank, 512);
    assert_eq!(c.qk_nope_head_dim, 256);
    assert_eq!(c.v_head_dim, 256);
    assert_eq!(c.qk_rope_head_dim, 0, "glm5next is NoPE");

    assert_eq!(c.index_n_heads, 32);
    assert_eq!(c.index_head_dim, 128);
    assert_eq!(c.index_topk, 2048);
    assert_eq!(c.index_kpool, 4);

    assert_eq!(c.linear_num_heads, 64);
    assert_eq!(c.linear_head_dim, 128);
    assert_eq!(c.linear_conv_kernel_dim, 4);
    assert_eq!(c.linear_lower_bound, Some(-5.0), "the safe-gate KDA form");

    assert_eq!(c.hc_mult, 4);
    assert_eq!(c.hc_sinkhorn_iters, 20);

    assert_eq!(c.n_routed_experts, 288);
    assert_eq!(c.num_experts_per_tok, 8);
    assert_eq!(c.n_shared_experts, 1);
    assert_eq!(c.moe_intermediate_size, 2048);
    assert_eq!(c.first_k_dense_replace, 3);
    assert_eq!(c.routed_scaling_factor, 2.5);
    assert_eq!(c.swiglu_limit, 10.0);

    // 34 KDA + 11 MLA, MLA every 4th layer starting at 3.
    assert_eq!(
        c.layer_types
            .iter()
            .filter(|k| **k == AttnKind::MlaDsa)
            .count(),
        11
    );
    assert_eq!(
        c.layer_types
            .iter()
            .filter(|k| **k == AttnKind::Kda)
            .count(),
        34
    );
    for i in 0..c.num_hidden_layers {
        let want = if i % 4 == 3 {
            AttnKind::MlaDsa
        } else {
            AttnKind::Kda
        };
        assert_eq!(c.attn_kind(i), want, "layer {i}");
        assert_eq!(c.indexer_kind(i), IndexerKind::Full, "layer {i}");
    }
    assert!(!c.is_moe_layer(2));
    assert!(c.is_moe_layer(3));
}

#[test]
fn parses_the_published_hf_config() {
    let c = Glm5NextConfig::from_hf_json(HF_CONFIG).expect("parse config.json");
    assert_is_glm_5_3_flash(&c);
}

#[test]
fn parses_the_published_gguf_metadata() {
    let c = Glm5NextConfig::from_gguf(&published_gguf()).expect("parse GGUF metadata");
    assert_is_glm_5_3_flash(&c);
}

/// The two readers rename nearly every field; they must still agree.
#[test]
fn gguf_and_hf_agree() {
    let hf = Glm5NextConfig::from_hf_json(HF_CONFIG).expect("hf");
    let gg = Glm5NextConfig::from_gguf(&published_gguf()).expect("gguf");
    assert_eq!(hf.layer_types, gg.layer_types);
    assert_eq!(hf.num_hidden_layers, gg.num_hidden_layers);
    assert_eq!(hf.hidden_size, gg.hidden_size);
    assert_eq!(hf.n_routed_experts, gg.n_routed_experts);
    assert_eq!(hf.index_topk, gg.index_topk);
    assert_eq!(hf.hc_mult, gg.hc_mult);
    assert_eq!(hf.swiglu_limit, gg.swiglu_limit);
    assert_eq!(hf.linear_lower_bound, gg.linear_lower_bound);
}

/// At the shipped `index_topk = 2048` and `kpool = 4`, DSA is exactly dense
/// causal attention for any prompt up to 2048 tokens.
#[test]
fn dsa_is_dense_up_to_the_topk_budget() {
    let c = Glm5NextConfig::from_gguf(&published_gguf()).expect("gguf");
    assert!(c.dsa_is_dense(1));
    assert!(c.dsa_is_dense(2048));
    assert!(!c.dsa_is_dense(2049));
}

#[test]
fn rejects_a_foreign_architecture() {
    let mut g = published_gguf();
    g.metadata.insert(
        "general.architecture".into(),
        MetaValue::String("glm4moe".into()),
    );
    let err = Glm5NextConfig::from_gguf(&g).unwrap_err().to_string();
    assert!(err.contains("glm4moe"), "unexpected error: {err}");
}

/// `head_count_kv` is the only thing distinguishing a KDA layer from an MLA
/// layer in GGUF. Losing it must be an error, not a model of 45 MLA layers.
#[test]
fn rejects_a_missing_layer_schedule() {
    let mut g = published_gguf();
    g.metadata.remove("glm5next.attention.head_count_kv");
    let err = Glm5NextConfig::from_gguf(&g).unwrap_err().to_string();
    assert!(err.contains("head_count_kv"), "unexpected error: {err}");
}

/// Softmax routing (`expert_gating_func = 1`) is a different gate; a
/// glm5next-shaped checkpoint carrying it must not be run through the sigmoid
/// path.
#[test]
fn rejects_non_sigmoid_routing() {
    let mut g = published_gguf();
    g.metadata
        .insert("glm5next.expert_gating_func".into(), MetaValue::U32(1));
    let err = Glm5NextConfig::from_gguf(&g).unwrap_err().to_string();
    assert!(
        err.contains("expert_gating_func"),
        "unexpected error: {err}"
    );
}
