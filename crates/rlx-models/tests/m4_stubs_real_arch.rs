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

//! Verifies every M4 stub crate (`rlx-bonsai`, `rlx-omnicoder`,
//! `rlx-mistral`, `rlx-phi`, `rlx-granite`, `rlx-cohere`) wires its
//! arch-validation correctly against real downloaded GGUFs.
//!
//! The substrate-availability rule:
//!   * SmolLM2 (`llama` arch) — accepted by `rlx-bonsai`
//!   * Qwen 2.5 (`qwen2` arch) — accepted by `rlx-omnicoder`
//!   * Gemma 3 (`gemma3` arch) — rejected by **all** stubs (their
//!     accept-lists are llama / qwen3 / mistral3 / phi3 / granite /
//!     command-r — no gemma3 in any)
//!
//! We don't yet download real Mistral 3 / Phi 3 / Granite / Cohere
//! GGUFs (each multi-GB), but every stub's *rejection* path is
//! exercised here against the real arch tags we do have.

use rlx_models::{
    bonsai::BonsaiRunner, cohere::CohereRunner, granite::GraniteRunner, mistral::MistralRunner,
    omnicoder::OmniCoderRunner, phi::PhiRunner,
};
use std::path::{Path, PathBuf};

fn smollm_path() -> Option<PathBuf> {
    std::env::var("RLX_SMOLLM2_GGUF").ok().map(PathBuf::from)
}
fn qwen25_path() -> Option<PathBuf> {
    std::env::var("RLX_QWEN25_GGUF").ok().map(PathBuf::from)
}
fn gemma3_path() -> Option<PathBuf> {
    std::env::var("RLX_GEMMA3_GGUF").ok().map(PathBuf::from)
}

/// Bonsai stub accepts `llama` arch.
#[test]
fn bonsai_accepts_llama_rejects_other() {
    let Some(llama) = smollm_path() else {
        eprintln!("skip: RLX_SMOLLM2_GGUF");
        return;
    };
    BonsaiRunner::builder()
        .weights(&llama)
        .packed_weights(true)
        .max_seq(64)
        .build()
        .expect("Bonsai accepts llama-arch GGUF");

    if let Some(qwen) = qwen25_path() {
        let err = BonsaiRunner::builder()
            .weights(&qwen)
            .packed_weights(true)
            .max_seq(64)
            .build()
            .err()
            .expect("Bonsai rejects qwen2-arch GGUF");
        assert!(format!("{err:#}").contains("expected"));
    }
}

/// OmniCoder stub accepts `qwen3` / `qwen2` arch.
#[test]
fn omnicoder_accepts_qwen_rejects_other() {
    let Some(qwen) = qwen25_path() else {
        eprintln!("skip: RLX_QWEN25_GGUF");
        return;
    };
    OmniCoderRunner::builder()
        .weights(&qwen)
        .packed_weights(true)
        .max_seq(64)
        .build()
        .expect("OmniCoder accepts qwen-arch GGUF");

    if let Some(llama) = smollm_path() {
        let err = OmniCoderRunner::builder()
            .weights(&llama)
            .packed_weights(true)
            .max_seq(64)
            .build()
            .err()
            .expect("OmniCoder rejects llama-arch GGUF");
        assert!(format!("{err:#}").contains("expected"));
    }
}

/// Mistral / Phi / Granite / Cohere stubs all reject the wrong arch.
/// We test rejection rather than acceptance because we don't yet ship
/// real Mistral / Phi / Granite / Cohere GGUFs in the test fixture.
#[test]
fn mistral_phi_granite_cohere_reject_wrong_arch() {
    let Some(any) = smollm_path().or_else(qwen25_path).or_else(gemma3_path) else {
        eprintln!("skip: need at least one real GGUF env var set");
        return;
    };

    // Pointing each at the wrong arch must produce a structured
    // arch-mismatch error rather than panicking.
    for (name, build_fn) in [
        (
            "rlx-mistral",
            Box::new(|p: &Path| {
                MistralRunner::builder()
                    .weights(p)
                    .packed_weights(true)
                    .max_seq(64)
                    .build()
                    .map(|_| ())
            }) as Box<dyn Fn(&Path) -> anyhow::Result<()>>,
        ),
        (
            "rlx-phi",
            Box::new(|p: &Path| {
                PhiRunner::builder()
                    .weights(p)
                    .packed_weights(true)
                    .max_seq(64)
                    .build()
                    .map(|_| ())
            }),
        ),
        (
            "rlx-granite",
            Box::new(|p: &Path| {
                GraniteRunner::builder()
                    .weights(p)
                    .packed_weights(true)
                    .max_seq(64)
                    .build()
                    .map(|_| ())
            }),
        ),
        (
            "rlx-cohere",
            Box::new(|p: &Path| {
                CohereRunner::builder()
                    .weights(p)
                    .packed_weights(true)
                    .max_seq(64)
                    .build()
                    .map(|_| ())
            }),
        ),
    ] {
        let err = build_fn(&any)
            .err()
            .unwrap_or_else(|| panic!("{name} should reject {any:?}"));
        let msg = format!("{err:#}");
        assert!(
            msg.contains("expected"),
            "{name} error should mention `expected` arches, got: {msg}"
        );
        eprintln!("{name}: correctly rejected {any:?}: {msg}");
    }
}

/// End-to-end forward inference through `OmniCoderRunner` (the only
/// stub with a real-weight substrate available — Qwen 2.5 0.5 B). The
/// delegation chain is OmniCoderRunner → Qwen3Runner →
/// `build_qwen3_graph_sized_packed` with `attention_bias + qk_norm=false`
/// (the Qwen 2 path I just fixed).
#[test]
fn omnicoder_runs_end_to_end() {
    if std::env::var("RLX_OMNICODER_RUN_INFERENCE").ok().as_deref() != Some("1") {
        eprintln!("skip: set RLX_OMNICODER_RUN_INFERENCE=1");
        return;
    }
    let Some(weights) = qwen25_path() else {
        eprintln!("skip: RLX_QWEN25_GGUF");
        return;
    };
    let Some(tokenizer) = std::env::var("RLX_QWEN25_TOKENIZER")
        .ok()
        .map(PathBuf::from)
    else {
        eprintln!("skip: RLX_QWEN25_TOKENIZER");
        return;
    };
    let mut runner = OmniCoderRunner::builder()
        .weights(&weights)
        .packed_weights(true)
        .max_seq(64)
        .build()
        .expect("OmniCoderRunner build");
    let prompt_ids =
        rlx_models::qwen35::encode_prompt_auto(&weights, Some(&tokenizer), "hi").expect("encode");
    let mut emitted = Vec::new();
    let generated = runner
        .generate_packed(&prompt_ids, 1, |t| emitted.push(t))
        .expect("generate_packed");
    assert_eq!(generated.len(), 1);
    eprintln!("OmniCoder stub inference (substrate=Qwen 2.5 0.5B): {prompt_ids:?} → {generated:?}");
}
