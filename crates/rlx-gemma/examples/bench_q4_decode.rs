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

//! rlx-gemma vs llama.cpp throughput comparison on the **same**
//! Q4_K_M GGUF (`unsloth/gemma-4-31B-it-GGUF`). Splits **load**,
//! **prefill**, and **decode** into three independent wall-clock
//! measurements — matching how `llama-bench` reports `pp t/s` and
//! `tg t/s` separately.
//!
//! Method: one `generate()` call with `n_new = N_NEW`, streaming
//! callback timestamps every token. Wall clock to the first
//! emitted token = prefill + 1 decode step. Wall clock between
//! later tokens = mean steady-state decode-step latency.
//!
//! Run:
//! ```bash
//! cargo run -p rlx-gemma --release --features "metal mlx" \
//!     --example bench_q4_decode -- \
//!     /Users/Shared/rlx-models/.eagle3-bench/weights/verifier/gemma-4-31B-it-Q4_K_M.gguf
//! ```
//!
//! llama.cpp b9606 baseline (`-ngl all`, `--seed 42 --temp 0`):
//!   - Metal: Prompt **34.7 t/s** | Generation **12.0 t/s**
//!   - CPU:   Prompt  4.5 t/s     | Generation  6.0 t/s

use anyhow::{Context, Result};
use rlx_gemma::{GemmaConfigSource, GemmaRunnerBuilder};
use rlx_runtime::Device;
use rlx_runtime::is_available;
use std::path::{Path, PathBuf};
use std::time::Instant;
use tokenizers::Tokenizer;

const PROMPT: &str =
    "Explain in one short paragraph what speculative decoding is and why EAGLE3 is fast.";
fn n_new() -> usize {
    std::env::var("RLX_BENCH_N_NEW")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(4) // default small enough to complete even if decode is very slow
}

/// Prefer the real Gemma tokenizer at
/// `$RLX_BENCH_TOKENIZER` (or the default 31B-metadata path) so the
/// model sees a real embedding sequence rather than random IDs (the
/// random-IDs path collapses every step to PAD=0 — see #8). Falls
/// back to a `[<bos>, 100..]` synthetic sequence if the tokenizer
/// file isn't available — useful for perf-only runs.
fn prompt_ids() -> Vec<u32> {
    let tok_path = std::env::var("RLX_BENCH_TOKENIZER").unwrap_or_else(|_| {
        "/Users/Shared/rlx-models/.eagle3-bench/weights/target_metadata/tokenizer.json".into()
    });
    if Path::new(&tok_path).exists()
        && let Ok(tok) = Tokenizer::from_file(&tok_path)
        && let Ok(enc) = tok.encode(PROMPT, true)
    {
        return enc.get_ids().to_vec();
    }
    let n_words = PROMPT.split_whitespace().count();
    let mut ids = vec![2u32];
    ids.extend((100..(100 + n_words as u32)).map(|t| t));
    ids
}

struct Phases {
    load_s: f32,
    prefill_s: f32,
    prefill_toks: usize,
    decode_s: f32,
    decode_toks: usize,
}

fn bench_one(device: Device, label: &str, weights: &str) -> Result<Phases> {
    use std::io::Write;
    // ── 1. Load ────────────────────────────────────────────────────
    let t_load = Instant::now();
    let mut runner = GemmaRunnerBuilder::default()
        .weights(weights)
        .device(device)
        .config(GemmaConfigSource::Embedded)
        .packed_weights(true)
        .build()
        .with_context(|| format!("{label}: building runner"))?;
    let load_s = t_load.elapsed().as_secs_f32();
    println!("   [{label}] LOAD     {load_s:7.2} s");
    std::io::stdout().flush().ok();

    // ── 2. Prefill + decode in one generate() with token timestamps.
    let prompt_ids = prompt_ids();
    let n_prompt = prompt_ids.len();
    let n_new_v = n_new();

    // Diagnostic: get logits from a fresh predict_logits call so we can
    // tell if the broken `[0,0,…]` output (#50) is from degenerate logits
    // (all-zero or NaN) vs greedy collapse to the actually-best token.
    if std::env::var("RLX_BENCH_LOGIT_DIAG").is_ok() {
        let t0 = Instant::now();
        let logits = runner
            .predict_logits(&prompt_ids)
            .with_context(|| format!("{label}: predict_logits"))?;
        let n = logits.len();
        let mut n_nan = 0usize;
        let mut n_zero = 0usize;
        let mut n_finite = 0usize;
        let mut max = f32::NEG_INFINITY;
        let mut max_idx = 0usize;
        let mut min = f32::INFINITY;
        let mut sum = 0.0f64;
        for (i, &v) in logits.iter().enumerate() {
            if v.is_nan() {
                n_nan += 1;
                continue;
            }
            n_finite += 1;
            if v == 0.0 {
                n_zero += 1;
            }
            if v > max {
                max = v;
                max_idx = i;
            }
            if v < min {
                min = v;
            }
            sum += v as f64;
        }
        let mean = if n_finite > 0 {
            sum / n_finite as f64
        } else {
            0.0
        };
        println!(
            "   [{label}] LOGITS   n={n} nan={n_nan} zero={n_zero} finite={n_finite}  min={min:.4e} mean={mean:.4e} max={max:.4e}  argmax={max_idx}  ({:.2}s)",
            t0.elapsed().as_secs_f32()
        );
        std::io::stdout().flush().ok();
    }

    let mut token_times: Vec<f32> = Vec::with_capacity(n_new_v + 1);
    let t_gen = Instant::now();
    let toks = runner
        .generate(&prompt_ids, n_new_v, |_t| {
            let now = t_gen.elapsed().as_secs_f32();
            token_times.push(now);
            // Stream progress so a slow decode isn't a silent stare.
            let idx = token_times.len();
            if idx == 1 {
                println!(
                    "   [{label}] PREFILL  {now:7.2} s  ({n_prompt} prompt toks · {:6.2} t/s prompt)",
                    n_prompt as f32 / now.max(1e-6)
                );
            } else {
                let prev = token_times[idx - 2];
                println!(
                    "   [{label}]   tok#{idx:>2}   +{:6.2} s  (instant {:5.2} t/s)",
                    now - prev,
                    1.0 / (now - prev).max(1e-6)
                );
            }
            std::io::stdout().flush().ok();
        })
        .with_context(|| format!("{label}: generate"))?;
    if toks.is_empty() {
        anyhow::bail!("{label}: generate returned zero tokens");
    }

    let first_t = token_times.first().copied().unwrap_or(0.0);
    let prefill_s = first_t;
    let decode_s = if token_times.len() > 1 {
        token_times.last().unwrap() - token_times.first().unwrap()
    } else {
        0.0
    };
    let decode_count = token_times.len().saturating_sub(1);
    let decode_tps = decode_count as f32 / decode_s.max(1e-6);
    println!(
        "   [{label}] DECODE   {decode_s:7.2} s  ({decode_count} new toks · {decode_tps:6.2} t/s gen)"
    );
    println!("   [{label}] TOKS     {:?}", toks);
    std::io::stdout().flush().ok();

    Ok(Phases {
        load_s,
        prefill_s,
        prefill_toks: n_prompt,
        decode_s,
        decode_toks: decode_count,
    })
}

fn try_run(device: Device, label: &str, weights: &str) {
    if !is_available(device) {
        println!("   [{label}] not available — skipped\n");
        return;
    }
    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        bench_one(device, label, weights)
    }));
    match res {
        Ok(Ok(_)) => {}
        Ok(Err(e)) => println!("   [{label}] FAILED: {e:?}\n"),
        Err(p) => {
            let msg = p
                .downcast_ref::<String>()
                .map(|s| s.as_str())
                .or_else(|| p.downcast_ref::<&str>().copied())
                .unwrap_or("(non-string panic)");
            println!("   [{label}] PANIC: {msg}\n");
        }
    }
}

fn main() -> Result<()> {
    let weights: PathBuf = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .context("usage: bench_q4_decode <gguf> [metal|mlx|cpu|all]")?;
    let weights_str = weights
        .to_str()
        .context("non-utf8 weights path")?
        .to_string();
    let only = std::env::args().nth(2).unwrap_or_else(|| "all".into());

    println!("→ rlx-gemma decode bench on {weights_str}");
    println!("   prompt = {PROMPT:?}");
    println!("   n_new  = {} (override with RLX_BENCH_N_NEW=…)", n_new());
    println!("   only   = {only}\n");

    println!("  Comparison from llama.cpp b9606 (-ngl all, same model):");
    println!("    Metal: Prompt 34.7 t/s · Generation 12.0 t/s");
    println!("    CPU:   Prompt  4.5 t/s · Generation  6.0 t/s\n");

    let all = [
        (Device::Metal, "Metal"),
        (Device::Mlx, "MLX  "),
        (Device::Cpu, "CPU  "),
    ];
    let filter = only.to_ascii_lowercase();
    for (dev, label) in all {
        if filter != "all" && !label.trim().to_ascii_lowercase().starts_with(&filter) {
            continue;
        }
        println!("[{label}] starting…");
        try_run(dev, label, &weights_str);
    }

    println!("✓ DONE.");
    Ok(())
}
