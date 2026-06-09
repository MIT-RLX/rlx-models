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

//! Parity-check harness for rlx-gemma vs llama.cpp.
//!
//! PLAN.md M2 definition-of-done: ≥99% top-1 token match over the
//! first 32 tokens of a fixed prompt + seed. This file is the
//! harness; the fixture itself is external, so the test self-skips
//! when no fixture path is supplied.
//!
//! Today the harness only covers `gemma` / `gemma2` arch tags. Gemma
//! 3 / 4 land once `rlx_flow::blocks::GemmaLayerStyle` gains V3 / V4
//! variants (sibling `rlx/` workspace work) — the harness itself
//! needs no changes when those tags become routable.
//!
//! Usage:
//!
//! ```sh
//! cargo test -p rlx-gemma --features parity-llama \
//!     --test gguf_parity -- --nocapture
//! ```
//!
//! Required env vars:
//!   * `RLX_GEMMA_PARITY_GGUF` — path to a `.gguf` file with
//!     `general.architecture` in `{gemma, gemma2}`. Without this, the
//!     test prints a skip line and returns success.
//!
//! Optional env vars:
//!   * `RLX_GEMMA_PARITY_PROMPT_IDS` — comma-separated token ids.
//!     Default: `1,2,3,4,5,6,7,8`.
//!   * `RLX_GEMMA_PARITY_MAX_SEQ` — prefill bucket size (default 64).

#![cfg(feature = "parity-llama")]

use std::path::PathBuf;

use rlx_gemma::{GemmaRunner, llama_reference};
use rlx_runtime::Device;

fn parity_fixture() -> Option<PathBuf> {
    std::env::var_os("RLX_GEMMA_PARITY_GGUF").map(PathBuf::from)
}

fn parity_prompt_ids() -> Vec<u32> {
    std::env::var("RLX_GEMMA_PARITY_PROMPT_IDS")
        .ok()
        .map(|s| {
            s.split(',')
                .filter_map(|t| t.trim().parse::<u32>().ok())
                .collect::<Vec<_>>()
        })
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| (1u32..=8).collect())
}

fn parity_max_seq() -> usize {
    std::env::var("RLX_GEMMA_PARITY_MAX_SEQ")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|n| *n > 0)
        .unwrap_or(64)
}

fn argmax_u32(logits: &[f32]) -> Option<(u32, f32)> {
    logits
        .iter()
        .copied()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, v)| (i as u32, v))
}

#[test]
fn top1_token_matches_llama_cpp_on_fixture() {
    let Some(fixture) = parity_fixture() else {
        eprintln!(
            "skip: RLX_GEMMA_PARITY_GGUF not set — supply a Gemma / Gemma 2 \
             GGUF to run the rlx-gemma ↔ llama.cpp parity check (PLAN.md M2)"
        );
        return;
    };
    if !fixture.is_file() {
        panic!("RLX_GEMMA_PARITY_GGUF points at {fixture:?} which is not a file");
    }

    let prompt_ids = parity_prompt_ids();
    let max_seq = parity_max_seq();
    assert!(
        prompt_ids.len() <= max_seq,
        "prompt_ids.len()={} exceeds RLX_GEMMA_PARITY_MAX_SEQ={}",
        prompt_ids.len(),
        max_seq
    );

    eprintln!(
        "# parity fixture: {fixture:?} (prompt_ids.len={}, max_seq={})",
        prompt_ids.len(),
        max_seq
    );

    // 1) rlx-gemma prediction.
    let mut runner = GemmaRunner::builder()
        .weights(&fixture)
        .device(Device::Cpu)
        .max_seq(max_seq)
        .build()
        .expect("GemmaRunner::build()");
    let rlx_logits = runner
        .predict_logits(&prompt_ids)
        .expect("GemmaRunner::predict_logits()");
    let (rlx_top, rlx_top_logit) = argmax_u32(&rlx_logits).expect("non-empty rlx logits");

    // 2) llama.cpp reference.
    let llama_logits = llama_reference::last_token_logits(&fixture, &prompt_ids)
        .expect("llama_reference::last_token_logits()");
    let (llama_top, llama_top_logit) = argmax_u32(&llama_logits).expect("non-empty llama logits");

    eprintln!(
        "# rlx top1 = {rlx_top} ({rlx_top_logit:+.4}) | llama top1 = {llama_top} ({llama_top_logit:+.4})"
    );
    assert_eq!(
        rlx_logits.len(),
        llama_logits.len(),
        "rlx vocab size ({}) != llama vocab size ({})",
        rlx_logits.len(),
        llama_logits.len()
    );
    assert_eq!(
        rlx_top, llama_top,
        "top-1 token mismatch: rlx={rlx_top} llama={llama_top}"
    );
}
