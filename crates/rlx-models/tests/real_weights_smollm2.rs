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

//! Real-weight integration tests against a downloaded GGUF.
//!
//! Set `RLX_SMOLLM2_GGUF` to the path of any `llama`-arch GGUF
//! (verified against bartowski/SmolLM2-135M-Instruct-GGUF Q4_K_M, 100 MB).
//! Tests are silently skipped when the env var isn't set, matching the
//! `QWEN35_GGUF_PATH` / `LLAMA32_GGUF_PATH` convention used elsewhere
//! in this test directory.
//!
//! Run:
//!   ```sh
//!   curl -L -o /tmp/smollm2.gguf \
//!     https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-Q4_K_M.gguf
//!   RLX_SMOLLM2_GGUF=/tmp/smollm2.gguf \
//!     cargo test -p rlx-models --test real_weights_smollm2
//!   ```

use rlx_llama_base::{LlamaBaseConfig, RopeScaling};
use rlx_models::run::{ChatTemplate, CompatibilityStatus, check_path};
use std::path::PathBuf;

fn weights_path() -> Option<PathBuf> {
    std::env::var("RLX_SMOLLM2_GGUF").ok().map(PathBuf::from)
}

fn tokenizer_path() -> Option<PathBuf> {
    std::env::var("RLX_SMOLLM2_TOKENIZER")
        .ok()
        .map(PathBuf::from)
        // Default: sidecar next to the GGUF (the convention rlx-llama32 uses).
        .or_else(|| {
            weights_path().and_then(|p| {
                let sib = p.parent()?.join("tokenizer.json");
                sib.is_file().then_some(sib)
            })
        })
}

#[test]
fn config_from_real_gguf_parses_llama_arch() {
    let Some(path) = weights_path() else {
        eprintln!("skip: set RLX_SMOLLM2_GGUF to a llama-arch .gguf path");
        return;
    };
    let cfg = LlamaBaseConfig::from_gguf_path(&path)
        .expect("LlamaBaseConfig::from_gguf_path should succeed on real llama GGUF");

    // Architectural preflight — these match the reference SmolLM2-135M-Instruct
    // GGUF metadata. The asserts are deliberately permissive about field
    // values that vary across distillations (rope_theta, eps) but strict
    // about shape so the test catches accidental schema mismatch.
    assert_eq!(cfg.arch, "llama");
    assert!(cfg.num_hidden_layers >= 8, "block_count too low: {:?}", cfg);
    assert!(cfg.hidden_size >= 64, "hidden_size too low: {:?}", cfg);
    assert!(cfg.num_attention_heads > 0);
    assert!(cfg.num_key_value_heads > 0);
    assert!(cfg.gqa_groups() >= 1);
    assert!(cfg.effective_head_dim() > 0);
    assert!(cfg.vocab_size > 1024, "vocab_size suspiciously small");
    assert!(cfg.max_position_embeddings >= 2048);
    assert!(cfg.rope_theta > 0.0);
    assert!(cfg.rms_norm_eps > 0.0 && cfg.rms_norm_eps < 1e-2);
    // SmolLM2 doesn't use sliding window or rope scaling.
    assert!(matches!(
        cfg.rope_scaling,
        None | Some(RopeScaling::Linear { .. })
    ));

    eprintln!(
        "SmolLM2 config: layers={} hidden={} heads={} kv={} ctx={} vocab={} rope_theta={}",
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
fn compat_check_reports_supported_for_real_gguf() {
    let Some(path) = weights_path() else {
        eprintln!("skip: set RLX_SMOLLM2_GGUF");
        return;
    };
    let report = check_path(&path).expect("check_path should succeed");
    match &report.status {
        CompatibilityStatus::Supported { runner } => {
            assert_eq!(
                *runner, "llama32",
                "SmolLM2 (llama arch) should dispatch to llama32 runner"
            );
        }
        other => panic!("expected Supported, got {other:?}\nreport:\n{report}"),
    }
    let fields = report
        .gguf_fields
        .as_ref()
        .expect("GGUF fields should be present");
    assert!(
        fields.is_complete(),
        "all required fields should be present: missing {:?}",
        fields.missing()
    );
    assert!(fields.context_length.is_some());
    assert!(fields.embedding_length.is_some());
    assert!(fields.block_count.is_some());
    assert_eq!(
        fields.tokenizer_model.as_deref(),
        Some("gpt2"),
        "SmolLM2 uses GPT-2 BPE"
    );
    assert!(fields.has_tokens);

    // JSON round-trip — proves the output is consumable downstream.
    let j = report.to_json();
    let v: serde_json::Value = serde_json::from_str(&j).expect("JSON round-trip");
    assert_eq!(v["status"], "supported");
    assert_eq!(v["source"], "gguf");
    assert_eq!(v["arch"], "llama");
}

#[test]
fn chat_template_round_trip_on_real_gguf() {
    let Some(path) = weights_path() else {
        eprintln!("skip: set RLX_SMOLLM2_GGUF");
        return;
    };
    let Ok(template) = ChatTemplate::from_gguf(&path) else {
        // Not all GGUFs ship a chat template — that's a valid skip,
        // not a failure. SmolLM2-Instruct does.
        eprintln!("skip: GGUF has no tokenizer.chat_template");
        return;
    };

    // BOS/EOS should resolve via tokenizer.ggml.tokens table.
    assert!(
        template.bos_token().is_some() || template.eos_token().is_some(),
        "expected at least one special token to resolve from tokenizer.ggml.tokens"
    );

    let msgs = vec![
        rlx_models::run::ChatMessage::system("be brief"),
        rlx_models::run::ChatMessage::user("hi"),
    ];
    let rendered = template
        .render(&msgs, true)
        .expect("chat template should render on real SmolLM2 template");
    assert!(!rendered.is_empty(), "rendered prompt was empty");
    // SmolLM2's template uses ChatML-style <|im_start|>/<|im_end|>.
    // Don't hard-code those — but the user content must appear somewhere
    // and the prompt must end with something resembling an assistant cue.
    assert!(rendered.contains("hi"), "user content missing: {rendered}");
    assert!(
        rendered.contains("be brief"),
        "system content missing: {rendered}"
    );
    eprintln!(
        "SmolLM2 rendered prompt ({} bytes):\n{}",
        rendered.len(),
        rendered
    );
}

/// End-to-end forward inference through the existing `rlx-llama32` runner
/// on a real GGUF. Gated on `RLX_SMOLLM2_RUN_INFERENCE=1` because the
/// packed-decode path is ~5 seconds per token on CPU for 135M parameters
/// — too slow for default test runs.
#[test]
fn forward_inference_real_weights() {
    if std::env::var("RLX_SMOLLM2_RUN_INFERENCE").ok().as_deref() != Some("1") {
        eprintln!("skip: set RLX_SMOLLM2_RUN_INFERENCE=1 to run forward inference");
        return;
    }
    let Some(weights) = weights_path() else {
        eprintln!("skip: set RLX_SMOLLM2_GGUF");
        return;
    };
    let Some(tokenizer) = tokenizer_path() else {
        eprintln!("skip: tokenizer.json not found next to weights (set RLX_SMOLLM2_TOKENIZER)");
        return;
    };

    use rlx_models::Llama32Runner;
    let mut runner = Llama32Runner::builder()
        .weights(&weights)
        .packed_weights(true)
        .max_seq(64)
        .build()
        .expect("Llama32Runner::build");

    // Encode prompt directly to ids — bypasses chat-template path so the
    // test is portable across SmolLM2 / TinyLlama / any other llama-arch.
    let prompt_ids = rlx_models::llama32::encode_prompt_auto(
        &weights,
        Some(&tokenizer),
        "The capital of France is",
    )
    .expect("encode_prompt");
    assert!(!prompt_ids.is_empty(), "tokenizer produced empty ids");

    let n_new = 4;
    let mut emitted = Vec::new();
    let generated = runner
        .generate_packed(&prompt_ids, n_new, |tok| emitted.push(tok))
        .expect("generate_packed");

    assert_eq!(generated.len(), n_new, "expected {n_new} new tokens");
    assert_eq!(emitted.len(), n_new, "callback should fire per token");
    assert!(
        generated.iter().all(|t| (*t as usize) < 49152),
        "all token ids must fit in SmolLM2 vocab; got {generated:?}"
    );
    eprintln!("SmolLM2 inference: prompt_ids={prompt_ids:?} → {generated:?}");
}
