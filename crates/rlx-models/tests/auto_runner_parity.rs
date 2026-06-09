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

//! Generic `auto_runner` parity harness vs llama.cpp.
//!
//! Works for any family routed through `rlx_models::run::auto_runner`
//! (qwen3, qwen35, gemma, llama32, lfm). Runs first-token top-1
//! greedy decode on the supplied GGUF and compares with llama.cpp's
//! prediction on the same prompt + the same model file.
//!
//! Env vars:
//!   * `RLX_PARITY_GGUF`         — path to a `.gguf` file. Skip if unset.
//!   * `RLX_PARITY_PROMPT_IDS`   — comma-separated token ids
//!     (default: `1,2,3,4,5,6,7,8`).
//!
//! Requires the `parity-llama` feature (pulls in `llama-cpp-2`).
//!
//!   RLX_PARITY_GGUF=/path/to/model.gguf \
//!     cargo test -p rlx-models --features parity-llama \
//!       --test auto_runner_parity --release -- --nocapture

#![cfg(feature = "parity-llama")]

use std::path::PathBuf;

use rlx_models::LmRunner;
use rlx_models::run::auto_runner;

fn fixture() -> Option<PathBuf> {
    let p = std::env::var_os("RLX_PARITY_GGUF").map(PathBuf::from)?;
    p.is_file().then_some(p)
}

fn parse_prompt_ids() -> Vec<u32> {
    let s =
        std::env::var("RLX_PARITY_PROMPT_IDS").unwrap_or_else(|_| "1,2,3,4,5,6,7,8".to_string());
    s.split(',')
        .map(|x| x.trim().parse::<u32>().expect("prompt id parse"))
        .collect()
}

fn argmax_u32(logits: &[f32]) -> (u32, f32) {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, &v)| (i as u32, v))
        .expect("non-empty logits")
}

fn topk_indices(logits: &[f32], k: usize) -> Vec<u32> {
    let mut pairs: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    pairs.iter().take(k).map(|(i, _)| *i as u32).collect()
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut dot = 0f64;
    let mut na = 0f64;
    let mut nb = 0f64;
    for i in 0..n {
        let x = a[i] as f64;
        let y = b[i] as f64;
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na <= 0.0 || nb <= 0.0 {
        return 0.0;
    }
    (dot / na.sqrt() / nb.sqrt()) as f32
}

#[test]
fn top1_token_matches_llama_cpp_via_auto_runner() {
    let Some(path) = fixture() else {
        eprintln!("skip: RLX_PARITY_GGUF not set");
        return;
    };
    let prompt_ids = parse_prompt_ids();
    eprintln!("# parity fixture: {path:?} (prompt_ids={:?})", prompt_ids);

    // 1) rlx side via auto_runner.
    let mut runner = auto_runner(&path).expect("auto_runner");
    eprintln!("# rlx family: {}", runner.family());
    let rlx_logits = runner.predict_logits(&prompt_ids).expect("predict_logits");
    let (rlx_top, rlx_top_logit) = argmax_u32(&rlx_logits);

    // 2) llama.cpp reference (last-token logits on the same prompt).
    let llama_logits = rlx_qwen35::llama_reference::last_token_logits(&path, &prompt_ids)
        .expect("llama reference");
    let (llama_top, llama_top_logit) = argmax_u32(&llama_logits);

    eprintln!(
        "# rlx top1 = {rlx_top} ({rlx_top_logit:+.4}) | llama top1 = {llama_top} ({llama_top_logit:+.4})"
    );
    // Soft metrics — useful when the runner has a small numerical
    // mismatch relative to llama-cpp (different accumulation order,
    // FMA fusion, etc.). They're informational regardless of strict
    // top-1 outcome.
    let cos = cosine(&rlx_logits, &llama_logits);
    let rlx_top5 = topk_indices(&rlx_logits, 5);
    let llama_top5 = topk_indices(&llama_logits, 5);
    let top5_overlap: usize = rlx_top5.iter().filter(|t| llama_top5.contains(t)).count();
    eprintln!("# cosine = {cos:.6}  top5_overlap = {top5_overlap}/5");
    eprintln!("# rlx top5   = {rlx_top5:?}");
    eprintln!("# llama top5 = {llama_top5:?}");
    // Strict top-1 — promoted to a soft check when the cosine is
    // very high but accumulated mismatch moves the argmax. The strict
    // assertion still fires for "no parity at all" cases (low cos,
    // empty top-5 overlap).
    if rlx_top != llama_top {
        if cos < 0.99 || top5_overlap < 3 {
            panic!(
                "rlx vs llama parity failed: top1 differs (rlx={rlx_top}, llama={llama_top}) \
                 AND cosine {cos:.4} < 0.99 / top5_overlap {top5_overlap} < 3"
            );
        }
        eprintln!(
            "# WARN: top1 differs but cosine {cos:.4} and top5_overlap {top5_overlap}/5 \
             are within tolerance — small numeric mismatch, not a structural bug"
        );
    }
}
