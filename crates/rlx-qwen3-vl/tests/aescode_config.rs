// RLX models — Qwen3-VL.
// Copyright (C) 2026 Eugene Hauptmann.
// SPDX-License-Identifier: GPL-3.0-only

//! `microsoft/AesCode-8B` and `microsoft/AesCode-32B` against this crate's
//! config reader.
//!
//! AesCode generates infographics as HTML/CSS from a prompt plus an
//! image-generated aesthetic reference. Architecturally it is **stock
//! Qwen3-VL**: every field of `AesCode-8B`'s config matches
//! `Qwen/Qwen3-VL-8B-Instruct` and `AesCode-32B`'s matches
//! `Qwen/Qwen3-VL-32B-Instruct` — text dims, `rope_theta` 5e6,
//! `mrope_section [24, 20, 20]` with `mrope_interleaved`, and a vision tower
//! of depth 27 / hidden 1152 / patch 16 / merge 2 / deepstack `[8, 16, 24]`.
//! Both READMEs name the base model, and the GGUF conversions record it in
//! `general.base_model.0.name`. So there is no new architecture here, and
//! this file's job is to prove the reader agrees rather than to port
//! anything.
//!
//! The reason it is worth a test: the reader used to **ignore**
//! `spatial_merge_size` and `deepstack_visual_indexes` (serde skips unknown
//! fields), so a Qwen3-VL config parsed "successfully" into a geometry this
//! crate's vision graph could not compute — one embedding per patch, from
//! the final layer only. That is 4x too many vision tokens drawn from the
//! wrong features: the model answers confidently about an image it did not
//! really see. They are read now, and `validate_supported` refuses.

use rlx_qwen3_vl::Qwen3VlVisionConfig;

const AESCODE_8B: &str = include_str!("fixtures/aescode_8b_config.json");
const AESCODE_32B: &str = include_str!("fixtures/aescode_32b_config.json");

/// Parse a fixture through the real path (which takes a file).
///
/// The directory name carries a process-wide counter, not a caller-supplied
/// label. A label is unique per *call site*, and two tests here legitimately
/// parse the same fixture — so they shared a directory, raced, and each
/// deleted the other's file: both passed alone and one failed in the suite.
fn parse(json: &str, _tag: &str) -> Qwen3VlVisionConfig {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static N: AtomicUsize = AtomicUsize::new(0);
    let dir = std::env::temp_dir().join(format!(
        "rlx-qwen3vl-aescode-{}-{}",
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let path = dir.join("config.json");
    std::fs::write(&path, json).expect("write fixture");
    let cfg = Qwen3VlVisionConfig::from_hf_config_json(&path)
        .expect("AesCode should parse as a Qwen3-VL config");
    std::fs::remove_dir_all(&dir).ok();
    cfg
}

#[test]
fn both_aescode_sizes_parse_as_qwen3_vl() {
    for (json, tag, lm_hidden) in [(AESCODE_8B, "8b", 4096), (AESCODE_32B, "32b", 5120)] {
        let cfg = parse(json, tag);
        // The tower is identical across both sizes; only the projector's
        // output follows the language model.
        assert_eq!(cfg.hidden_size, 1152, "{tag} vision hidden");
        assert_eq!(cfg.num_hidden_layers, 27, "{tag} depth (read via `depth`)");
        assert_eq!(cfg.num_attention_heads, 16, "{tag} heads (via `num_heads`)");
        assert_eq!(cfg.intermediate_size, 4304, "{tag} ffn");
        assert_eq!(cfg.patch_size, 16, "{tag} patch");
        assert_eq!(cfg.num_channels, 3, "{tag} channels (via `in_channels`)");
        assert_eq!(
            cfg.projector_output_dim, lm_hidden,
            "{tag} projector must land on the LM's hidden size"
        );
    }
}

/// The geometry that used to be dropped on the floor.
#[test]
fn the_merge_and_deepstack_geometry_is_actually_read() {
    for (json, tag) in [(AESCODE_8B, "8b"), (AESCODE_32B, "32b")] {
        let cfg = parse(json, tag);
        assert_eq!(cfg.spatial_merge_size, 2, "{tag}: Qwen3-VL merges 2x2");
        assert_eq!(cfg.temporal_patch_size, 2, "{tag}: two frames per patch");
        assert_eq!(
            cfg.num_position_embeddings,
            Some(2304),
            "{tag}: the learned table's row count"
        );
        assert_eq!(
            cfg.deepstack_visual_indexes,
            vec![8, 16, 24],
            "{tag}: three layers feed the LM in addition to the last"
        );
    }
}

/// A checkpoint this crate's vision graph cannot compute must be refused,
/// not approximated.
#[test]
fn a_real_qwen3_vl_tower_is_refused_with_a_reason() {
    let cfg = parse(AESCODE_8B, "refuse");
    let err = cfg
        .validate_supported()
        .expect_err("2x2 merging + deepstack are not implemented here");
    let msg = err.to_string();
    // Name the number of tokens it would get wrong, and where the real
    // implementation lives — an error that only says "unsupported" sends
    // the reader back to the source.
    assert!(msg.contains("2x2"), "should name the merge factor: {msg}");
    assert!(msg.contains("4x too many"), "should quantify it: {msg}");
    assert!(
        msg.contains("rlx-qwen35"),
        "should point at the tower to port from: {msg}"
    );
}

/// An older export that merges nothing still builds.
#[test]
fn a_pre_qwen3_vl_export_without_the_new_fields_is_still_supported() {
    // Qwen2.5-VL-shaped: transformers-wide names, no merge, no deepstack.
    let json = r#"{
        "vision_config": {
            "hidden_size": 1152, "num_hidden_layers": 27, "num_attention_heads": 16,
            "intermediate_size": 4304, "image_size": 384, "patch_size": 14,
            "num_channels": 3
        },
        "text_config": { "hidden_size": 2048 }
    }"#;
    let cfg = parse(json, "legacy");
    assert_eq!(cfg.spatial_merge_size, 1, "absent means no merging");
    assert_eq!(cfg.temporal_patch_size, 1);
    assert!(cfg.num_position_embeddings.is_none());
    assert!(cfg.deepstack_visual_indexes.is_empty());
    assert_eq!(cfg.projector_output_dim, 2048, "falls back to LM hidden");
    cfg.validate_supported()
        .expect("this is exactly the shape the graph implements");
}

/// The *text* trunk parses too, which is what "the text path works" rests on.
///
/// Worth a test because the config is nested: a Qwen3-VL export states the
/// language model under `text_config`, and a reader that parses the top level
/// finds no `vocab_size` and no `hidden_size`. `rlx-qwen3` handles the
/// nesting; this pins that it keeps doing so for these two checkpoints,
/// since that is the half of AesCode that runs today.
///
/// `rope_theta` is the field to watch: 5e6 here against stock Qwen3's 1e6,
/// and a silently-defaulted base is a model that degrades with context
/// rather than one that fails.
#[test]
fn the_text_trunk_parses_out_of_the_nested_config() {
    use rlx_qwen3::Qwen3Config;

    for (json, tag, hidden, layers, heads, ffn) in [
        (AESCODE_8B, "8b", 4096usize, 36usize, 32usize, 12288usize),
        (AESCODE_32B, "32b", 5120, 64, 64, 25600),
    ] {
        let dir = std::env::temp_dir().join(format!(
            "rlx-qwen3vl-aescode-text-{}-{tag}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let path = dir.join("config.json");
        std::fs::write(&path, json).expect("write");
        let cfg = Qwen3Config::from_file(&path).expect("AesCode text config should parse");
        std::fs::remove_dir_all(&dir).ok();

        assert_eq!(cfg.hidden_size, hidden, "{tag} hidden");
        assert_eq!(cfg.num_hidden_layers, layers, "{tag} layers");
        assert_eq!(cfg.num_attention_heads, heads, "{tag} heads");
        assert_eq!(cfg.num_key_value_heads, 8, "{tag} kv heads");
        assert_eq!(cfg.intermediate_size, ffn, "{tag} ffn");
        assert_eq!(cfg.head_dim, 128, "{tag} head_dim");
        assert_eq!(cfg.vocab_size, 151936, "{tag} vocab");
        assert_eq!(
            cfg.rope_theta, 5_000_000.0,
            "{tag}: Qwen3-VL's base is 5e6, not stock Qwen3's 1e6"
        );
        assert!(cfg.qk_norm, "{tag}: Qwen3 normalizes Q and K");
    }
}
