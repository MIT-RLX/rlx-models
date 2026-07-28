// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! MiniMax-M3 GGUF path: config parse from synthetic `minimax-m3.*` / `minimax.*`
//! metadata, and the ggml→flow tensor-name mapping. Best-effort per llama.cpp
//! ggml-org/llama.cpp#24908 (not validated against a real M3 GGUF).

use rlx_gguf::MetaValue;
use rlx_minimax::m3::config::MiniMaxM3Config;
use rlx_minimax::m3::weights::gguf_to_flow_name;
use std::collections::HashMap;

fn meta() -> HashMap<String, MetaValue> {
    let mut m = HashMap::new();
    let put = |m: &mut HashMap<String, MetaValue>, k: &str, v: MetaValue| {
        m.insert(k.to_string(), v);
    };
    put(
        &mut m,
        "general.architecture",
        MetaValue::String("minimax-m3".into()),
    );
    put(&mut m, "minimax-m3.block_count", MetaValue::U32(6));
    put(&mut m, "minimax-m3.embedding_length", MetaValue::U32(64));
    put(&mut m, "minimax-m3.attention.head_count", MetaValue::U32(8));
    put(
        &mut m,
        "minimax-m3.attention.head_count_kv",
        MetaValue::U32(2),
    );
    put(
        &mut m,
        "minimax-m3.attention.key_length",
        MetaValue::U32(16),
    );
    put(
        &mut m,
        "minimax-m3.feed_forward_length",
        MetaValue::U32(128),
    );
    put(
        &mut m,
        "minimax-m3.expert_feed_forward_length",
        MetaValue::U32(32),
    );
    put(&mut m, "minimax-m3.expert_count", MetaValue::U32(16));
    put(&mut m, "minimax-m3.expert_used_count", MetaValue::U32(4));
    put(
        &mut m,
        "minimax-m3.leading_dense_block_count",
        MetaValue::U32(3),
    );
    put(
        &mut m,
        "minimax-m3.attention.layer_norm_rms_epsilon",
        MetaValue::F32(1e-6),
    );
    put(
        &mut m,
        "minimax-m3.rope.freq_base",
        MetaValue::F32(5_000_000.0),
    );
    // MSA keys under the `minimax.*` namespace.
    put(&mut m, "minimax.block_size", MetaValue::U32(128));
    put(&mut m, "minimax.indexer_head_count", MetaValue::U32(2));
    put(&mut m, "minimax.indexer_head_dim", MetaValue::U32(128));
    put(&mut m, "minimax.top_k_blocks", MetaValue::U32(16));
    put(&mut m, "minimax.local_blocks", MetaValue::U32(1));
    put(&mut m, "minimax.partial_rotary_factor", MetaValue::F32(0.5));
    m
}

#[test]
fn gguf_config_parses() {
    let cfg = MiniMaxM3Config::from_gguf_meta(&meta()).expect("parse gguf meta");
    assert_eq!(cfg.num_hidden_layers, 6);
    assert_eq!(cfg.hidden_size, 64);
    assert_eq!(cfg.num_attention_heads, 8);
    assert_eq!(cfg.num_key_value_heads, 2);
    assert_eq!(cfg.head_dim(), 16);
    assert_eq!(cfg.rotary_dim, 8); // 16 · 0.5
    assert_eq!(cfg.num_local_experts, 16);
    assert_eq!(cfg.num_experts_per_tok, 4);
    assert_eq!(cfg.sparse.block_size, 128);
    assert_eq!(cfg.sparse.index_n_heads, 2);
    assert_eq!(cfg.sparse.index_head_dim, 128);
    // First 3 layers dense + full attention; the rest MoE + MSA sparse.
    assert!(!cfg.is_moe_layer(0) && !cfg.is_moe_layer(2) && cfg.is_moe_layer(3));
    assert!(!cfg.is_sparse_layer(2) && cfg.is_sparse_layer(3) && cfg.is_sparse_layer(5));
}

#[test]
fn gguf_rejects_non_m3_arch() {
    let mut m = meta();
    m.insert(
        "general.architecture".into(),
        MetaValue::String("llama".into()),
    );
    assert!(MiniMaxM3Config::from_gguf_meta(&m).is_err());
}

#[test]
fn gguf_name_mapping() {
    let n = |s: &str| gguf_to_flow_name(s);
    assert_eq!(
        n("token_embd.weight").as_deref(),
        Some("model.embed_tokens.weight")
    );
    assert_eq!(
        n("output_norm.weight").as_deref(),
        Some("model.norm.weight")
    );
    assert_eq!(n("output.weight").as_deref(), Some("lm_head.weight"));
    assert_eq!(
        n("blk.5.attn_q.weight").as_deref(),
        Some("model.layers.5.self_attn.q_proj.weight")
    );
    assert_eq!(
        n("blk.5.attn_k_norm.weight").as_deref(),
        Some("model.layers.5.self_attn.k_norm.weight")
    );
    assert_eq!(
        n("blk.7.attn_index_q_norm.weight").as_deref(),
        Some("model.layers.7.self_attn.index_q_norm.weight")
    );
    assert_eq!(
        n("blk.3.ffn_gate_inp.weight").as_deref(),
        Some("model.layers.3.block_sparse_moe.gate.weight")
    );
    assert_eq!(
        n("blk.3.ffn_down_shexp.weight").as_deref(),
        Some("model.layers.3.block_sparse_moe.shared_experts.down_proj.weight")
    );
    assert_eq!(
        n("blk.2.ffn_gate.weight").as_deref(),
        Some("model.layers.2.mlp.gate_proj.weight")
    );
    // Stacked expert tensors need gate/up combining → not a 1:1 rename.
    assert!(n("blk.3.ffn_gate_exps.weight").is_none());
}
