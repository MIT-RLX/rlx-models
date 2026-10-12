// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! `Cloudflare/clef` against this crate's config reader.
//!
//! clef is a `qwen3_5` checkpoint — Cloudflare's typed-output decision model on
//! a Qwen3.5 trunk — so the question is not whether to port an architecture but
//! whether the one here already describes it. The fixture is the published
//! `config.json`, verbatim, so a reader that drifts from the real file fails
//! here rather than on someone's first load.

use rlx_qwen35::Qwen35Config;
use std::path::Path;

const CLEF: &str = include_str!("fixtures/clef_config.json");

fn parse() -> Qwen35Config {
    let v: serde_json::Value = serde_json::from_str(CLEF).expect("fixture is JSON");
    Qwen35Config::from_hf_config_value(&v, Path::new("clef_config.json"))
        .expect("clef should parse as a Qwen3.5 config")
}

/// The published hyper-parameters, read off the real file.
#[test]
fn clef_parses_as_a_qwen35_checkpoint() {
    let c = parse();
    assert_eq!(c.hidden_size, 5120);
    assert_eq!(c.num_hidden_layers, 64);
    assert_eq!(c.vocab_size, 248320);
    assert_eq!(c.num_attention_heads, 24);
    assert_eq!(c.num_key_value_heads, 4);
    assert_eq!(c.key_length, 256, "head_dim");
    assert_eq!(c.value_length, 256);
    assert_eq!(c.intermediate_size, 17408);
}

/// The GatedDeltaNet half: clef is a hybrid, not a dense transformer.
///
/// `full_attention_interval = 4` over 64 layers means 16 gated-GQA layers and
/// 48 linear-attention ones. Reading the schedule wrong is the failure that
/// still produces finite output.
#[test]
fn the_hybrid_layer_schedule_is_read() {
    let c = parse();
    // HF spells these `linear_*`; this crate carries the GGUF names.
    assert_eq!(c.ssm_group_count, 16, "linear_num_key_heads");
    assert_eq!(c.ssm_state_size, 128, "linear_key_head_dim");
    assert_eq!(c.ssm_conv_kernel, 4, "linear_conv_kernel_dim");
    // 48 value heads x 128 — the GDN inner width, and the length of A_log.
    assert_eq!(c.ssm_inner_size, 48 * 128);
    assert_eq!(
        c.ssm_time_step_rank, 48,
        "A_log / dt_bias are per value head"
    );

    assert_eq!(c.full_attention_interval, 4);
    let full = (1..=c.num_hidden_layers)
        .filter(|i| i % c.full_attention_interval == 0)
        .count();
    assert_eq!(full, 16, "every 4th of 64 layers is full attention");
    assert_eq!(c.num_hidden_layers - full, 48, "the rest are linear");
}

/// Partial RoPE and the output gate — both on, both easy to miss.
#[test]
fn the_attention_options_are_read() {
    let c = parse();
    // partial_rotary_factor 0.25 of head_dim 256 = 64 rotated dims.
    assert_eq!(c.rope_dim_count, 64);
    assert_eq!(c.rope_theta, 10_000_000.0);
    assert!(c.mrope_interleaved, "clef's M-RoPE is interleaved");
    assert_eq!(c.rope_dim_sections, vec![11, 11, 10]);
}
