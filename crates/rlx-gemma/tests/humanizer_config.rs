// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! `jialinyyzz/humanizer` against this crate's config reader.
//!
//! humanizer is a `gemma4_unified` checkpoint — a text-rewriting post-train on
//! the Gemma 4 flagship — so nothing new has to be built for it. What has to be
//! true is that the reader here describes it, including the three options the
//! flagship turns on that a plainer Gemma does not: `attention_k_eq_v`, a
//! wider `global_head_dim` than `head_dim`, and logit softcapping. The fixture
//! is the published `config.json`, verbatim.

use rlx_gemma::config::GemmaLayerType;
use rlx_gemma::{GemmaArch, GemmaConfig};

const HUMANIZER: &str = include_str!("fixtures/humanizer_config.json");

/// `GemmaConfig` reads from a path, so the fixture is written out first — which
/// also exercises the `text_config` unwrapping the real file needs.
///
/// One directory per call, not one per process: the tests run in parallel and a
/// shared path means whichever finishes first deletes the file the others are
/// still reading.
fn parse() -> GemmaConfig {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static N: AtomicUsize = AtomicUsize::new(0);
    let dir = std::env::temp_dir().join(format!(
        "rlx-gemma-humanizer-{}-{}",
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).expect("tmpdir");
    let path = dir.join("config.json");
    std::fs::write(&path, HUMANIZER).expect("write fixture");
    let cfg = GemmaConfig::from_file(&path).expect("humanizer should parse as a Gemma 4 config");
    let _ = std::fs::remove_dir_all(&dir);
    cfg
}

#[test]
fn humanizer_parses_as_gemma4() {
    let c = parse();
    assert_eq!(c.arch, GemmaArch::Gemma4);
    assert_eq!(c.hidden_size, 3840);
    assert_eq!(c.num_hidden_layers, 48);
    assert_eq!(c.vocab_size, 262144);
    assert_eq!(c.intermediate_size, 15360);
    assert_eq!(c.num_attention_heads, 16);
    assert_eq!(c.num_key_value_heads, 8);
    assert!(c.tie_word_embeddings, "the LM head is tied");
}

/// The three flagship-only options, each of which changes the graph.
///
/// `attention_k_eq_v` is the sharp one: the full-attention layers ship **no**
/// `v_proj` tensor, so a reader that misses it looks for a weight the
/// checkpoint does not contain.
#[test]
fn the_flagship_attention_options_are_read() {
    let c = parse();
    assert!(
        c.attention_k_eq_v,
        "the flag the full-attention layers act on"
    );
    assert_eq!(
        c.global_head_dim,
        Some(512),
        "full-attention layers are wider"
    );
    assert_eq!(c.head_dim, Some(256), "sliding layers keep the narrow head");
    assert_eq!(c.final_logit_softcapping, Some(30.0));
    assert_eq!(c.sliding_window, Some(1024));
}

/// 48 layers alternating sliding/full, each with its own RoPE base.
///
/// The two attention kinds do not share a rotary base — 10 000 for sliding,
/// 1 000 000 for full — so a single `rope_theta` would be wrong for half the
/// stack.
#[test]
fn the_layer_schedule_and_per_kind_rope_are_read() {
    let c = parse();
    assert_eq!(c.layer_types.len(), 48, "one entry per layer");
    let full = c
        .layer_types
        .iter()
        .filter(|t| **t == GemmaLayerType::FullAttention)
        .count();
    assert!(
        full > 0 && full < 48,
        "the stack alternates, got {full} full"
    );
    // Full-attention layers rotate at 1e6, sliding ones at 1e4, and the base
    // is selected *per layer* — a single `rope_theta` would be wrong for one
    // of the two kinds. (The flat field is the sliding/fallback base; the
    // per-kind values live in `rope_parameters`.)
    for i in 0..c.num_hidden_layers {
        let want = if c.layer_types[i] == GemmaLayerType::FullAttention {
            1_000_000.0
        } else {
            10_000.0
        };
        assert_eq!(c.layer_rope_theta(i), want, "layer {i}");
    }

    // V is aliased to K on full-attention layers only; sliding layers ship an
    // independent `v_proj`. Asserting it uniformly would describe a different
    // checkpoint.
    for i in 0..c.num_hidden_layers {
        let full = c.layer_types[i] == GemmaLayerType::FullAttention;
        assert_eq!(c.layer_k_eq_v(i), full, "layer {i}");
        let want_dim = if full { 512 } else { 256 };
        assert_eq!(c.layer_head_dim(i), want_dim, "layer {i} head_dim");
    }
}
