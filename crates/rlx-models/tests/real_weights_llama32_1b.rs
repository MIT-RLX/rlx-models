// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! Real-weight tests against the model `rlx-llama32` is named for:
//! **Llama 3.2 1B Instruct** (official Meta weights via bartowski).
//!
//! Bigger than the SmolLM2 135M sibling test (1B params, 131 K context,
//! Llama-3 RoPE scaling). Together with `real_weights_smollm2.rs` this
//! verifies the pipeline scales to a real Meta release and that the
//! `rlx-llama-base` config reader handles Llama-3-style RoPE scaling
//! (`factor`, `low_freq_factor`, `high_freq_factor`,
//! `original_max_position_embeddings`).
//!
//! Run:
//!   ```sh
//!   curl -L -o /tmp/rlx-weights/Llama-3.2-1B.gguf \
//!     https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf
//!   curl -L -o /tmp/rlx-weights/llama-3.2-tokenizer.json \
//!     https://huggingface.co/unsloth/Llama-3.2-1B-Instruct/resolve/main/tokenizer.json
//!   RLX_LLAMA32_GGUF=/tmp/rlx-weights/Llama-3.2-1B.gguf \
//!   RLX_LLAMA32_TOKENIZER=/tmp/rlx-weights/llama-3.2-tokenizer.json \
//!   RLX_LLAMA32_RUN_INFERENCE=1 \
//!     cargo test -p rlx-models --test real_weights_llama32_1b --release --features qwen35-tokenizer -- --nocapture
//!   ```

use rlx_llama_base::{LlamaBaseConfig, RopeScaling};
use rlx_models::run::{ChatTemplate, CompatibilityStatus, check_path};
use std::path::PathBuf;

fn weights_path() -> Option<PathBuf> {
    std::env::var("RLX_LLAMA32_GGUF").ok().map(PathBuf::from)
}

fn tokenizer_path() -> Option<PathBuf> {
    std::env::var("RLX_LLAMA32_TOKENIZER")
        .ok()
        .map(PathBuf::from)
        .or_else(|| {
            weights_path().and_then(|p| {
                let sib = p.parent()?.join("tokenizer.json");
                sib.is_file().then_some(sib)
            })
        })
}

#[test]
fn config_from_real_llama32_1b_gguf() {
    let Some(path) = weights_path() else {
        eprintln!("skip: set RLX_LLAMA32_GGUF");
        return;
    };
    let cfg = LlamaBaseConfig::from_gguf_path(&path).expect("LlamaBaseConfig parse");
    assert_eq!(cfg.arch, "llama");
    // Llama 3.2 1B official shape: 16 layers, 2048 hidden, 32 heads, 8 KV heads, 128K ctx, 128k vocab.
    assert_eq!(cfg.num_hidden_layers, 16, "Llama 3.2 1B has 16 layers");
    assert_eq!(cfg.hidden_size, 2048);
    assert_eq!(cfg.num_attention_heads, 32);
    assert_eq!(cfg.num_key_value_heads, 8);
    assert_eq!(cfg.gqa_groups(), 4);
    assert_eq!(cfg.effective_head_dim(), 64);
    assert_eq!(cfg.vocab_size, 128_256, "Llama 3 tokenizer is 128K");
    assert_eq!(cfg.max_position_embeddings, 131_072);

    // Llama 3 uses extended-RoPE with factor=32.0, low_freq=1.0,
    // high_freq=4.0, original_ctx=8192. The bartowski GGUF includes
    // the rope.scaling.* keys; verify they parse.
    match &cfg.rope_scaling {
        Some(RopeScaling::Llama3 {
            factor,
            low_freq_factor,
            high_freq_factor,
            original_max_position_embeddings,
        }) => {
            assert!(*factor > 0.0, "non-zero rope factor");
            assert!(*low_freq_factor > 0.0);
            assert!(*high_freq_factor > 0.0);
            assert_eq!(*original_max_position_embeddings, 8192);
            eprintln!(
                "Llama 3.2 RoPE: factor={factor} low={low_freq_factor} high={high_freq_factor} orig_ctx={original_max_position_embeddings}"
            );
        }
        other => {
            // Not all GGUF converters emit the scaling keys — accept None too,
            // but assert it didn't misparse as a wrong variant.
            assert!(
                matches!(other, None | Some(RopeScaling::Linear { .. })),
                "expected Llama3 / None / Linear rope_scaling, got {other:?}"
            );
            eprintln!(
                "Llama 3.2 GGUF has no rope.scaling.* keys (or compatible variant): {other:?}"
            );
        }
    }
    eprintln!(
        "Llama 3.2 1B config: layers={} hidden={} heads={} kv={} ctx={} vocab={} rope_theta={}",
        cfg.num_hidden_layers,
        cfg.hidden_size,
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.max_position_embeddings,
        cfg.vocab_size,
        cfg.rope_theta,
    );
}

#[test]
fn compat_check_for_llama32_1b() {
    let Some(path) = weights_path() else {
        eprintln!("skip: set RLX_LLAMA32_GGUF");
        return;
    };
    let report = check_path(&path).expect("check_path");
    match &report.status {
        CompatibilityStatus::Supported { runner } => {
            assert_eq!(*runner, "llama32");
        }
        other => panic!("expected Supported, got {other:?}\n{report}"),
    }
    let fields = report.gguf_fields.as_ref().expect("GGUF fields");
    assert!(fields.is_complete(), "missing: {:?}", fields.missing());
    // 128K context window is much larger than SmolLM2's 8K — important
    // to verify the field reader handles values > u32::MAX gracefully
    // (131072 is well within range; this is forward-looking).
    assert_eq!(fields.context_length, Some(131_072));
}

#[test]
fn chat_template_on_llama32_1b() {
    let Some(path) = weights_path() else {
        eprintln!("skip: set RLX_LLAMA32_GGUF");
        return;
    };
    let Ok(template) = ChatTemplate::from_gguf(&path) else {
        eprintln!("skip: GGUF has no tokenizer.chat_template");
        return;
    };
    let msgs = vec![
        rlx_models::run::ChatMessage::system("be terse"),
        rlx_models::run::ChatMessage::user("hi"),
    ];
    let rendered = template
        .render(&msgs, true)
        .expect("Llama 3.2 chat template should render");
    assert!(rendered.contains("hi"));
    assert!(rendered.contains("be terse"));
    // Llama 3 chat template uses <|begin_of_text|>, <|start_header_id|>,
    // <|end_header_id|>, <|eot_id|>. At least one should appear.
    assert!(
        ["<|begin_of_text|>", "<|start_header_id|>", "<|eot_id|>"]
            .iter()
            .any(|tok| rendered.contains(tok)),
        "expected Llama 3 chat tokens in: {rendered}"
    );
    eprintln!(
        "Llama 3.2 rendered prompt ({} bytes):\n{}",
        rendered.len(),
        rendered
    );
}

/// End-to-end forward inference at 1 B params. Gated on
/// `RLX_LLAMA32_RUN_INFERENCE=1` — the packed-decode path costs
/// ~3-5 s per token on CPU at this size, so the full 4-token test
/// is roughly 20 s wall clock.
#[test]
fn forward_inference_llama32_1b() {
    if std::env::var("RLX_LLAMA32_RUN_INFERENCE").ok().as_deref() != Some("1") {
        eprintln!("skip: set RLX_LLAMA32_RUN_INFERENCE=1");
        return;
    }
    let Some(weights) = weights_path() else {
        eprintln!("skip: set RLX_LLAMA32_GGUF");
        return;
    };
    let Some(tokenizer) = tokenizer_path() else {
        eprintln!("skip: tokenizer.json not found");
        return;
    };

    use rlx_models::Llama32Runner;
    let mut runner = Llama32Runner::builder()
        .weights(&weights)
        .packed_weights(true)
        .max_seq(64)
        .build()
        .expect("Llama32Runner::build");

    let prompt_ids = rlx_models::llama32::encode_prompt_auto(
        &weights,
        Some(&tokenizer),
        "The capital of France is",
    )
    .expect("encode_prompt");
    assert!(!prompt_ids.is_empty());

    // Single token. At 1 B params on CPU, the packed-decode path
    // re-runs the full prefill graph for each new token (~minutes per
    // token), so we only verify the round-trip works end-to-end and
    // exits without garbage. Increase locally for longer generation
    // when wallclock isn't a concern.
    let n_new = 1;
    let mut emitted = Vec::new();
    let generated = runner
        .generate_packed(&prompt_ids, n_new, |t| emitted.push(t))
        .expect("generate_packed");

    assert_eq!(generated.len(), n_new);
    assert_eq!(emitted.len(), n_new);
    assert!(
        generated.iter().all(|t| (*t as usize) < 128_256),
        "tokens out of Llama-3 vocab: {generated:?}"
    );
    eprintln!("Llama 3.2 1B inference (n_new={n_new}): prompt_ids={prompt_ids:?} → {generated:?}");
}
