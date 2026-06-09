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

//! Second real-weight family: Qwen 2.5 0.5B Instruct (qwen2 arch).
//!
//! Mirrors `real_weights_smollm2.rs` but on a different arch tag with
//! a different runner — proves the rlx-llama-base config reader,
//! compat surface, chat template engine, and the full inference path
//! all handle arch variants, not just the one Llama checkpoint.
//!
//! Run:
//!   ```sh
//!   curl -L -o /tmp/rlx-weights/Qwen2.5-0.5B.gguf \
//!     https://huggingface.co/bartowski/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf
//!   curl -L -o /tmp/rlx-weights/qwen2.5-tokenizer.json \
//!     https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/tokenizer.json
//!   RLX_QWEN25_GGUF=/tmp/rlx-weights/Qwen2.5-0.5B.gguf \
//!   RLX_QWEN25_TOKENIZER=/tmp/rlx-weights/qwen2.5-tokenizer.json \
//!   RLX_QWEN25_RUN_INFERENCE=1 \
//!     cargo test -p rlx-models --test real_weights_qwen25 --release --features qwen35-tokenizer -- --nocapture
//!   ```

use rlx_llama_base::LlamaBaseConfig;
use rlx_models::run::{ChatTemplate, CompatibilityStatus, check_path};
use std::path::PathBuf;

fn weights_path() -> Option<PathBuf> {
    std::env::var("RLX_QWEN25_GGUF").ok().map(PathBuf::from)
}

fn tokenizer_path() -> Option<PathBuf> {
    std::env::var("RLX_QWEN25_TOKENIZER")
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
fn config_from_real_qwen2_gguf() {
    let Some(path) = weights_path() else {
        eprintln!("skip: set RLX_QWEN25_GGUF");
        return;
    };
    // LlamaBaseConfig::from_gguf is arch-agnostic — it derives keys
    // from `general.architecture`. For Qwen2.5 0.5B this should
    // surface arch=qwen2 with the right dims.
    let cfg = LlamaBaseConfig::from_gguf_path(&path).expect("LlamaBaseConfig parse");
    assert_eq!(cfg.arch, "qwen2", "Qwen 2.5 ships as `qwen2` arch tag");
    assert!(cfg.num_hidden_layers >= 8);
    assert!(cfg.hidden_size >= 64);
    assert!(cfg.num_attention_heads > 0);
    assert!(cfg.num_key_value_heads > 0);
    assert!(cfg.vocab_size > 1024);
    assert!(cfg.max_position_embeddings >= 8192);
    assert!(cfg.rope_theta > 0.0);
    assert!(cfg.rms_norm_eps > 0.0 && cfg.rms_norm_eps < 1e-2);
    eprintln!(
        "Qwen2.5-0.5B config: arch={} layers={} hidden={} heads={} kv={} ctx={} vocab={} rope_theta={}",
        cfg.arch,
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
fn compat_check_routes_qwen2_to_qwen3_runner() {
    let Some(path) = weights_path() else {
        eprintln!("skip: set RLX_QWEN25_GGUF");
        return;
    };
    let report = check_path(&path).expect("check_path");
    // qwen2 arch dispatches to the qwen3 runner; the runner now reads
    // `attention_bias` + `qk_norm` from the GGUF arch tag and emits the
    // right per-layer math (Qwen 2 = biases + no QK-norm).
    match &report.status {
        CompatibilityStatus::Supported { runner } => {
            assert_eq!(*runner, "qwen3");
        }
        other => panic!("expected Supported, got {other:?}\n{report}"),
    }
    let fields = report.gguf_fields.as_ref().expect("GGUF fields");
    assert!(fields.is_complete(), "missing: {:?}", fields.missing());
}

#[test]
fn chat_template_on_qwen2_5() {
    let Some(path) = weights_path() else {
        eprintln!("skip: set RLX_QWEN25_GGUF");
        return;
    };
    let Ok(template) = ChatTemplate::from_gguf(&path) else {
        eprintln!("skip: GGUF has no tokenizer.chat_template");
        return;
    };
    let msgs = vec![
        rlx_models::run::ChatMessage::system("answer in one word"),
        rlx_models::run::ChatMessage::user("hi"),
    ];
    let rendered = template.render(&msgs, true).expect("render chat");
    assert!(rendered.contains("hi"), "missing user content: {rendered}");
    assert!(
        rendered.contains("answer in one word") || rendered.to_lowercase().contains("system"),
        "expected system content or system tag: {rendered}"
    );
    eprintln!(
        "Qwen2.5 rendered prompt ({} bytes):\n{}",
        rendered.len(),
        rendered
    );
}

/// End-to-end forward inference via the existing `rlx-qwen3` runner.
/// Gated on `RLX_QWEN25_RUN_INFERENCE=1` — the packed path is slow on
/// CPU even at 0.5B.
#[test]
fn forward_inference_real_qwen2_5() {
    if std::env::var("RLX_QWEN25_RUN_INFERENCE").ok().as_deref() != Some("1") {
        eprintln!("skip: set RLX_QWEN25_RUN_INFERENCE=1");
        return;
    }
    let Some(weights) = weights_path() else {
        eprintln!("skip: set RLX_QWEN25_GGUF");
        return;
    };
    let Some(tokenizer) = tokenizer_path() else {
        eprintln!("skip: tokenizer.json not found");
        return;
    };

    let mut runner = rlx_models::Qwen3Runner::builder()
        .weights(&weights)
        .packed_weights(true)
        .max_seq(64)
        .build()
        .expect("Qwen3Runner::build (Qwen 2 path = attention_bias + no QK-norm)");

    // Qwen 2.5 ships its own tokenizer; the qwen35 BPE module is
    // arch-neutral at this layer (any tokenizers.json works).
    let prompt_ids = rlx_models::qwen35::encode_prompt_auto(
        &weights,
        Some(&tokenizer),
        "The capital of France is",
    )
    .expect("encode_prompt");
    assert!(!prompt_ids.is_empty(), "tokenizer produced empty ids");

    // n_new=1 — packed-decode at 0.5 B is much faster than 1 B but
    // still ~1 min per token on CPU. Single-token round-trip is
    // enough to verify the Qwen 2 arch path (biases + skipped
    // QK-norm) runs end-to-end and produces a valid vocab token.
    let n_new = 1;
    let mut emitted = Vec::new();
    let generated = runner
        .generate_packed(&prompt_ids, n_new, |t| emitted.push(t))
        .expect("generate_packed");
    assert_eq!(generated.len(), n_new);
    assert_eq!(emitted.len(), n_new);
    assert!(
        generated.iter().all(|t| *t < 200_000_u32),
        "tokens out of plausible Qwen 2 vocab: {generated:?}"
    );
    eprintln!("Qwen2.5-0.5B inference (n_new={n_new}): prompt_ids={prompt_ids:?} → {generated:?}");
}
