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

//! End-to-end test for the **`rlx-bonsai` stub fix**.
//!
//! Bonsai (the small-reasoning family per PLAN.md M4) ships as
//! `general.architecture = llama` in its GGUF converters, so the
//! crate is a thin delegating wrapper over [`rlx_llama32::Llama32Runner`]
//! with arch-tag validation. Any llama-arch GGUF works as a stand-in
//! for the actual Bonsai checkpoints — we use the already-downloaded
//! SmolLM2 135M here because it's the smallest llama-tagged GGUF in
//! the test fixture set. Real Bonsai 2B/4B/8B drops in unchanged.
//!
//! Run:
//!   ```sh
//!   RLX_SMOLLM2_GGUF=/tmp/rlx-weights/SmolLM2-135M.gguf \
//!   RLX_SMOLLM2_TOKENIZER=/tmp/rlx-weights/tokenizer.json \
//!   RLX_BONSAI_RUN_INFERENCE=1 \
//!     cargo test -p rlx-models --test real_weights_bonsai_stub --release \
//!       --features qwen35-tokenizer -- --nocapture
//!   ```

use rlx_models::bonsai::{BonsaiRunner, FAMILY, PLAN_MILESTONE};
use std::path::PathBuf;

fn llama_gguf_path() -> Option<PathBuf> {
    std::env::var("RLX_SMOLLM2_GGUF").ok().map(PathBuf::from)
}

fn tokenizer_path() -> Option<PathBuf> {
    std::env::var("RLX_SMOLLM2_TOKENIZER")
        .ok()
        .map(PathBuf::from)
        .or_else(|| {
            llama_gguf_path().and_then(|p| {
                let sib = p.parent()?.join("tokenizer.json");
                sib.is_file().then_some(sib)
            })
        })
}

#[test]
fn bonsai_runner_builds_against_llama_arch_gguf() {
    let Some(path) = llama_gguf_path() else {
        eprintln!("skip: set RLX_SMOLLM2_GGUF to any llama-arch .gguf");
        return;
    };
    let runner = BonsaiRunner::builder()
        .weights(&path)
        .packed_weights(true)
        .max_seq(64)
        .build()
        .expect("BonsaiRunner should build on a llama-arch GGUF");
    let cfg = runner.config();
    assert_eq!(cfg.arch, "llama", "Bonsai stub validates arch tag");
    assert!(cfg.num_hidden_layers > 0);
    assert!(cfg.hidden_size > 0);
    eprintln!(
        "BonsaiRunner built: family={FAMILY} milestone={PLAN_MILESTONE} \
         arch={} layers={} hidden={} heads={}",
        cfg.arch, cfg.num_hidden_layers, cfg.hidden_size, cfg.num_attention_heads
    );
}

#[test]
fn bonsai_runner_rejects_non_llama_arch() {
    // Pointing the Bonsai runner at a Qwen 2.5 / Gemma 3 / etc. GGUF
    // should fail at build() with an arch-mismatch error, not silently
    // produce garbage. We use whichever non-llama GGUF the test env
    // has handy.
    let candidates = [
        std::env::var("RLX_QWEN25_GGUF").ok(),
        std::env::var("RLX_GEMMA3_GGUF").ok(),
    ];
    let Some(path) = candidates.into_iter().flatten().next() else {
        eprintln!("skip: need RLX_QWEN25_GGUF or RLX_GEMMA3_GGUF for negative test");
        return;
    };
    let err = BonsaiRunner::builder()
        .weights(PathBuf::from(&path))
        .packed_weights(true)
        .max_seq(64)
        .build()
        .err()
        .expect("non-llama GGUF should be rejected");
    let s = format!("{err:#}");
    assert!(
        s.contains("expected `general.architecture = llama`") || s.contains("expected"),
        "expected arch-mismatch message, got: {s}"
    );
    eprintln!("BonsaiRunner correctly rejected non-llama GGUF: {s}");
}

/// End-to-end forward inference through `BonsaiRunner`. Gated on
/// `RLX_BONSAI_RUN_INFERENCE=1` — the underlying packed-decode path is
/// the same as `Llama32Runner::generate_packed`, but worth verifying
/// the delegation didn't break anything.
#[test]
fn forward_inference_via_bonsai_stub() {
    if std::env::var("RLX_BONSAI_RUN_INFERENCE").ok().as_deref() != Some("1") {
        eprintln!("skip: set RLX_BONSAI_RUN_INFERENCE=1");
        return;
    }
    let Some(weights) = llama_gguf_path() else {
        eprintln!("skip: set RLX_SMOLLM2_GGUF");
        return;
    };
    let Some(tokenizer) = tokenizer_path() else {
        eprintln!("skip: tokenizer.json not found");
        return;
    };

    let mut runner = BonsaiRunner::builder()
        .weights(&weights)
        .packed_weights(true)
        .max_seq(64)
        .build()
        .expect("BonsaiRunner build");

    let prompt_ids = rlx_models::llama32::encode_prompt_auto(&weights, Some(&tokenizer), "hello")
        .expect("encode_prompt");
    assert!(!prompt_ids.is_empty());

    let mut emitted = Vec::new();
    let generated = runner
        .generate_packed(&prompt_ids, 1, |t| emitted.push(t))
        .expect("generate_packed");
    assert_eq!(generated.len(), 1);
    assert_eq!(emitted.len(), 1);
    eprintln!(
        "Bonsai stub inference (substrate=SmolLM2 135M, n_new=1): {prompt_ids:?} → {generated:?}"
    );
}
