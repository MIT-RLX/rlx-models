// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! Env-gated real-weights check for [`prism-ml/Bonsai-27B`](https://huggingface.co/prism-ml/Bonsai-27B-gguf).
//!
//! Bonsai-27B is a Qwen3.6-27B derivative shipped as
//! `general.architecture = qwen35` with the big projections stored in the
//! custom 1-bit `Q1_0` (`Q1_0_g128`) format — **not** the Llama-shaped
//! Bonsai the small-family runner handles. This test validates that
//! `rlx-bonsai` dispatches it to the qwen35 runner and that the config
//! parses, using the cheap GGUF header reader (the file is ~3.8 GB, so we
//! never slurp the data segment).
//!
//! Run with: `RLX_BONSAI27B_GGUF=/path/Bonsai-27B-Q1_0.gguf \`
//!   `cargo test -p rlx-models --features "bonsai,qwen35" \`
//!   `--test real_weights_bonsai27b`

use std::path::PathBuf;

fn bonsai27b_path() -> Option<PathBuf> {
    std::env::var("RLX_BONSAI27B_GGUF").ok().map(PathBuf::from)
}

#[test]
fn bonsai27b_dispatches_qwen35_and_parses_config() {
    let Some(path) = bonsai27b_path() else {
        eprintln!("skip: RLX_BONSAI27B_GGUF not set");
        return;
    };

    // Arch dispatch: Bonsai-27B is the qwen35 hybrid, not the llama small family.
    use rlx_models::bonsai::{BonsaiArch, detect_arch};
    assert_eq!(
        detect_arch(&path).expect("detect_arch"),
        BonsaiArch::Qwen35Hybrid,
        "Bonsai-27B must route to the qwen35 runner"
    );

    // Config parse from the header alone (no 3.8 GB data slurp).
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
    assert!(!cfg.is_moe(), "Bonsai-27B is dense, not MoE");

    // The big projections carry the custom 1-bit Q1_0 scheme.
    let qkv = raw
        .get("blk.0.attn_qkv.weight")
        .expect("blk.0.attn_qkv.weight");
    assert_eq!(
        qkv.dtype,
        rlx_gguf::GgmlType::Q1_0,
        "attn_qkv must be the custom 1-bit Q1_0 format"
    );

    eprintln!(
        "Bonsai-27B OK: arch=qwen35 layers={} hidden={} heads={}/{} full_attn_interval={} \
         attn_qkv dtype={:?}",
        cfg.num_hidden_layers,
        cfg.hidden_size,
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.full_attention_interval,
        qkv.dtype,
    );
}
