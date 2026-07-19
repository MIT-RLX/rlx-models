// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! Env-gated real-weights check for
//! [`prism-ml/Ternary-Bonsai-27B`](https://huggingface.co/prism-ml/Ternary-Bonsai-27B-gguf).
//!
//! Same qwen35 hybrid as 1-bit Bonsai-27B, but projections use PrismML
//! `Q2_0` (`Q2_0_g128`, ggml type 42) — ternary `{−1,0,+1}` in 2-bit
//! slots (~2.125 bpw deployed).
//!
//! Run with: `RLX_TERNARY_BONSAI27B_GGUF=/path/Ternary-Bonsai-27B-Q2_0.gguf \`
//!   `cargo test -p rlx-models --features "bonsai,qwen35" \`
//!   `--test real_weights_ternary_bonsai27b`

use std::path::PathBuf;

fn ternary_bonsai27b_path() -> Option<PathBuf> {
    std::env::var("RLX_TERNARY_BONSAI27B_GGUF")
        .ok()
        .map(PathBuf::from)
}

#[test]
fn ternary_bonsai27b_dispatches_qwen35_and_parses_config() {
    let Some(path) = ternary_bonsai27b_path() else {
        eprintln!("skip: RLX_TERNARY_BONSAI27B_GGUF not set");
        return;
    };

    use rlx_models::bonsai::{BonsaiArch, detect_arch};
    assert_eq!(
        detect_arch(&path).expect("detect_arch"),
        BonsaiArch::Qwen35Hybrid,
        "Ternary-Bonsai-27B must route to the qwen35 runner"
    );

    let raw = rlx_gguf::GgufFile::header_from_path(&path).expect("read GGUF header");
    assert_eq!(
        raw.metadata
            .get("general.architecture")
            .and_then(rlx_gguf::MetaValue::as_str),
        Some("qwen35")
    );
    let cfg = rlx_models::Qwen35Config::from_gguf(&raw).expect("parse Qwen35Config");
    assert_eq!(cfg.num_hidden_layers, 64, "block_count");
    assert_eq!(cfg.hidden_size, 5120, "embedding_length");
    assert_eq!(cfg.num_attention_heads, 24, "head_count");
    assert_eq!(cfg.num_key_value_heads, 4, "head_count_kv");
    assert_eq!(cfg.full_attention_interval, 4, "1-in-4 full attention");
    assert_eq!(cfg.ssm_state_size, 128, "gated-DeltaNet state size");
    assert!(!cfg.is_moe(), "Ternary-Bonsai-27B is dense, not MoE");

    let qkv = raw
        .get("blk.0.attn_qkv.weight")
        .expect("blk.0.attn_qkv.weight");
    assert_eq!(
        qkv.dtype,
        rlx_gguf::GgmlType::Q2_0,
        "attn_qkv must be the ternary Q2_0 format"
    );

    eprintln!(
        "Ternary-Bonsai-27B OK: arch=qwen35 layers={} hidden={} heads={}/{} \
         full_attn_interval={} attn_qkv dtype={:?}",
        cfg.num_hidden_layers,
        cfg.hidden_size,
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.full_attention_interval,
        qkv.dtype,
    );
}
