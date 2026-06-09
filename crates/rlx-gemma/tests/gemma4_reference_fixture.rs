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

//! Reference-fixture parity for Gemma 4 12B.
//!
//! Out-of-process parity check against an external reference (HF
//! transformers / llama.cpp). The fixture format is documented below;
//! tests auto-skip when no fixture path is supplied so CI without the
//! ~24 GB of weights stays green.
//!
//! ## Fixture format
//!
//! Set the `RLX_GEMMA4_FIXTURE` env var to a directory containing:
//!
//! ```text
//! $RLX_GEMMA4_FIXTURE/
//!   config.json              # HF Gemma 4 unified config
//!   tokenizer.json           # HF tokenizer
//!   model.safetensors        # full model weights (or model-*.safetensors shards)
//!   reference.json           # { "prompt": "...", "logits": [...], "tokens": [...] }
//! ```
//!
//! `reference.json` is produced ahead of time by the user's
//! reference implementation:
//!
//! ```python
//! # HF transformers reference dump
//! from transformers import AutoModelForCausalLM, AutoTokenizer
//! tok = AutoTokenizer.from_pretrained("google/gemma-4-12B")
//! model = AutoModelForCausalLM.from_pretrained("google/gemma-4-12B", torch_dtype="float32")
//! ids = tok("Hello, world!", return_tensors="pt").input_ids
//! with torch.no_grad():
//!     out = model(ids)
//! json.dump({
//!     "prompt": "Hello, world!",
//!     "logits": out.logits[0, -1].tolist(),
//!     "tokens": ids[0].tolist(),
//! }, open(".../reference.json", "w"))
//! ```
//!
//! ## Running
//!
//! ```bash
//! RLX_GEMMA4_FIXTURE=/path/to/fixture \
//!   cargo test -p rlx-gemma --test gemma4_reference_fixture \
//!     --features apple-silicon -- --nocapture
//! ```

use anyhow::{Context, Result};
use rlx_gemma::prelude::*;
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Debug, Deserialize)]
struct ReferenceDump {
    prompt: String,
    logits: Vec<f32>,
    tokens: Vec<u32>,
}

fn fixture_dir() -> Option<PathBuf> {
    std::env::var_os("RLX_GEMMA4_FIXTURE").map(PathBuf::from)
}

fn load_reference(dir: &std::path::Path) -> Result<ReferenceDump> {
    let path = dir.join("reference.json");
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("reading reference dump at {path:?}"))?;
    let dump: ReferenceDump = serde_json::from_str(&text)
        .with_context(|| format!("parsing reference.json at {path:?}"))?;
    Ok(dump)
}

fn run_fixture(device: Device, tol_top_k: usize) -> Result<()> {
    let Some(dir) = fixture_dir() else {
        eprintln!(
            "[gemma4 reference] RLX_GEMMA4_FIXTURE unset — skip. \
             See test file header for fixture format."
        );
        return Ok(());
    };
    if !dir.is_dir() {
        eprintln!("[gemma4 reference] fixture path {dir:?} is not a directory — skip");
        return Ok(());
    }
    let reference = load_reference(&dir)?;
    let weights = dir.join("model.safetensors");
    if !weights.exists() {
        // Sharded checkpoints (model-*.safetensors) aren't currently
        // wired through the runner; the user would need to merge or
        // point at an unsharded export.
        eprintln!(
            "[gemma4 reference] {weights:?} not found (sharded fixtures unsupported here) — skip"
        );
        return Ok(());
    }

    let mut runner = GemmaRunner::builder()
        .weights(weights.clone())
        .device(device)
        .max_seq(reference.tokens.len() + 8)
        .build()?;

    let logits = runner.predict_logits(&reference.tokens)?;
    assert_eq!(
        logits.len(),
        reference.logits.len(),
        "logits length mismatch: rlx={} reference={}",
        logits.len(),
        reference.logits.len(),
    );

    // Top-k token-id match — the only stable comparison across fp32
    // implementations once you're past a handful of layers (raw
    // logit values diverge by ~1e-2 even between two CPU paths).
    let rlx_top: Vec<usize> = top_k_indices(&logits, tol_top_k);
    let ref_top: Vec<usize> = top_k_indices(&reference.logits, tol_top_k);
    let shared = rlx_top.iter().filter(|i| ref_top.contains(i)).count();
    eprintln!(
        "[gemma4 reference] {device:?} top-{tol_top_k}: rlx={rlx_top:?} ref={ref_top:?} shared={shared}/{tol_top_k}",
    );
    assert!(
        shared >= tol_top_k * 3 / 4,
        "top-{tol_top_k} match {shared}/{tol_top_k} below 75% threshold on {device:?}",
    );
    Ok(())
}

fn top_k_indices(logits: &[f32], k: usize) -> Vec<usize> {
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.into_iter().take(k).map(|(i, _)| i).collect()
}

// ── Offline validation of the fixture flow ───────────────────────
//
// Without the real Gemma 4 12B weights we can't run the parity
// assertion end-to-end, but the surrounding plumbing (JSON parse,
// top-k computation, error paths) needs to work or the test is dead
// scaffolding. These tests exercise everything except the model run
// itself, so a downstream user with real weights gets a working
// harness.

#[test]
fn synthetic_reference_dump_round_trips_via_loader() {
    let dump = ReferenceDump {
        prompt: "hello".into(),
        tokens: vec![1, 2, 3],
        logits: vec![0.1, 0.7, 0.2, 0.5],
    };
    let tmp = std::env::temp_dir().join("rlx_gemma4_reference_smoke");
    std::fs::create_dir_all(&tmp).unwrap();
    let path = tmp.join("reference.json");
    std::fs::write(
        &path,
        serde_json::to_string(&serde_json::json!({
            "prompt": dump.prompt,
            "tokens": dump.tokens,
            "logits": dump.logits,
        }))
        .unwrap(),
    )
    .unwrap();
    let loaded = load_reference(&tmp).expect("loader");
    std::fs::remove_file(&path).ok();
    std::fs::remove_dir(&tmp).ok();
    assert_eq!(loaded.prompt, dump.prompt);
    assert_eq!(loaded.tokens, dump.tokens);
    assert_eq!(loaded.logits, dump.logits);
}

#[test]
fn top_k_indices_returns_largest_in_descending_value() {
    // Logits: index 2 is largest, then index 0, then index 3.
    let logits = vec![0.5, 0.1, 0.9, 0.3, 0.0];
    let top3 = top_k_indices(&logits, 3);
    assert_eq!(top3, vec![2, 0, 3]);
}

#[test]
fn top_k_indices_handles_ties_deterministically() {
    // All ties — the function should return stable indices via
    // partial_cmp's Equal fallback. We only assert length here.
    let logits = vec![0.5, 0.5, 0.5, 0.5];
    let top2 = top_k_indices(&logits, 2);
    assert_eq!(top2.len(), 2);
    assert!(top2.iter().all(|&i| i < logits.len()));
}

#[test]
fn top_k_overlap_count_works() {
    // Same algorithm as the live test uses to count `shared`.
    let a = [1, 2, 3, 4, 5];
    let b = [2, 4, 6, 8, 5];
    let shared = a.iter().filter(|i| b.contains(i)).count();
    assert_eq!(shared, 3); // 2, 4, 5
}

#[test]
fn load_reference_rejects_missing_directory() {
    let path = std::path::Path::new("/tmp/__rlx_gemma4_definitely_missing__");
    match load_reference(path) {
        Ok(_) => panic!("expected error for missing dir"),
        Err(err) => {
            let msg = format!("{err:#}");
            assert!(
                msg.contains("reference")
                    || msg.contains("No such file")
                    || msg.contains("reading"),
                "unexpected error: {msg}"
            );
        }
    }
}

#[test]
fn fixture_parity_cpu() {
    run_fixture(Device::Cpu, 5).expect("CPU fixture parity");
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn fixture_parity_metal() {
    if !rlx_runtime::device_ext::is_available(Device::Metal) {
        return;
    }
    run_fixture(Device::Metal, 5).expect("Metal fixture parity");
}

#[cfg(all(target_os = "macos", feature = "mlx"))]
#[test]
fn fixture_parity_mlx() {
    if !rlx_runtime::device_ext::is_available(Device::Mlx) {
        return;
    }
    run_fixture(Device::Mlx, 5).expect("MLX fixture parity");
}

#[cfg(feature = "cuda")]
#[test]
fn fixture_parity_cuda() {
    if !rlx_runtime::device_ext::is_available(Device::Cuda) {
        return;
    }
    run_fixture(Device::Cuda, 5).expect("CUDA fixture parity");
}
