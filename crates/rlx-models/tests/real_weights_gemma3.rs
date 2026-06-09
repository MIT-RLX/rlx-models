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

//! Third real-weight family: Gemma 3 270M Instruct (gemma3 arch).
//!
//! Smallest model in the PLAN.md catalog (~250 MB Q4_K_M). Verifies the
//! harness covers a non-Llama, non-Qwen family and that the
//! `gemma3 → PLAN.md M2` mapping in `known_unimplemented_arch` matches
//! what `check_path` reports.
//!
//! Run:
//!   ```sh
//!   curl -L -o /tmp/rlx-weights/gemma-3-270m.gguf \
//!     https://huggingface.co/unsloth/gemma-3-270m-it-GGUF/resolve/main/gemma-3-270m-it-Q4_K_M.gguf
//!   RLX_GEMMA3_GGUF=/tmp/rlx-weights/gemma-3-270m.gguf \
//!     cargo test -p rlx-models --test real_weights_gemma3 -- --nocapture
//!   ```
//!
//! No inference test: `rlx-gemma` targets gemma/gemma2 only; the
//! Gemma 3 runner is M2 work (per-layer sliding window + new RoPE).

use rlx_llama_base::LlamaBaseConfig;
use rlx_models::run::{ChatTemplate, CompatibilityStatus, check_path};
use std::path::PathBuf;

fn weights_path() -> Option<PathBuf> {
    std::env::var("RLX_GEMMA3_GGUF").ok().map(PathBuf::from)
}

#[test]
fn config_from_real_gemma3_gguf() {
    let Some(path) = weights_path() else {
        eprintln!("skip: set RLX_GEMMA3_GGUF");
        return;
    };
    let cfg = LlamaBaseConfig::from_gguf_path(&path).expect("LlamaBaseConfig parse");
    assert_eq!(cfg.arch, "gemma3", "Gemma 3 ships as `gemma3` arch tag");
    assert!(cfg.num_hidden_layers >= 8, "block_count too low: {cfg:?}");
    assert!(cfg.hidden_size >= 64);
    assert!(cfg.num_attention_heads > 0);
    assert!(cfg.num_key_value_heads > 0);
    assert!(cfg.vocab_size > 1024);
    assert!(cfg.max_position_embeddings >= 2048);
    eprintln!(
        "Gemma 3 270M config: arch={} layers={} hidden={} heads={} kv={} ctx={} vocab={} rope_theta={}",
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
fn compat_check_reports_gemma3_as_m2() {
    let Some(path) = weights_path() else {
        eprintln!("skip: set RLX_GEMMA3_GGUF");
        return;
    };
    let report = check_path(&path).expect("check_path should succeed");
    match &report.status {
        CompatibilityStatus::KnownUnimplemented(u) => {
            assert_eq!(u.milestone, "M2", "Gemma 3 is M2 work");
            assert!(u.family.contains("Gemma 3"), "got family: {}", u.family);
            eprintln!("Gemma 3 270M: {} ({}) — {}", u.family, u.milestone, u.note);
        }
        other => panic!("expected KnownUnimplemented(Gemma 3 M2), got {other:?}\n{report}"),
    }
    // The required-GGUF-field check should still report all fields
    // present — the gap is the runner, not the metadata.
    let fields = report.gguf_fields.as_ref().expect("GGUF fields");
    assert!(
        fields.is_complete(),
        "Gemma 3 metadata complete; missing: {:?}",
        fields.missing()
    );
}

#[test]
fn chat_template_on_real_gemma3() {
    let Some(path) = weights_path() else {
        eprintln!("skip: set RLX_GEMMA3_GGUF");
        return;
    };
    let Ok(template) = ChatTemplate::from_gguf(&path) else {
        eprintln!("skip: GGUF has no tokenizer.chat_template");
        return;
    };
    let msgs = vec![rlx_models::run::ChatMessage::user("hi")];
    let rendered = template
        .render(&msgs, true)
        .expect("Gemma 3 chat template should render with minijinja");
    assert!(rendered.contains("hi"), "missing user content: {rendered}");
    // Gemma uses <start_of_turn>/<end_of_turn>, not ChatML.
    assert!(
        rendered.contains("start_of_turn") || rendered.contains("<bos>") || rendered.contains("<|"),
        "expected Gemma-style or other recognized tokens: {rendered}"
    );
    eprintln!(
        "Gemma 3 rendered prompt ({} bytes):\n{}",
        rendered.len(),
        rendered
    );
}
