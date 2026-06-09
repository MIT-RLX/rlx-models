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

//! End-to-end Qwen3.5 / Qwen3.6 (qwen35 arch) prefill example.
//!
//! ```text
//! cargo run --release -p rlx-models --example run_qwen35 -- \
//!     <path/to/Qwen3.5-0.8B-MTP-GGUF/...gguf> \
//!     [--mtp] [--prompt-ids 1,2,3] [--max-tokens N]
//! ```
//!
//! Loads the hybrid gated-DeltaNet + attention forward graph,
//! runs a single prefill pass on the supplied prompt ids (default
//! `[1, 2, 3]`), and prints the top-5 next-token logits from the
//! trunk LM head + (when `--mtp`) the MTP head.
//!
//! With `--max-tokens N > 0`, runs an autoregressive greedy
//! generation loop instead and prints each new token id as it's
//! sampled. Each new token costs one full prefill (no decode-state
//! cache yet), so generation scales as `O(N · seq · n_state²)`.
//!
//! Memory: this path dequants every K-quant weight to F32 at load
//! time. Qwen3.5-0.8B Q4_K_M fits (~1.5 GB peak); Qwen3.6-27B
//! Q4_K_M does **not** fit on commodity Macs — extend the packed-
//! weights (`Op::DequantMatMul`) path to qwen35 to run the 27B
//! file.

use anyhow::{Context, Result, bail};
use rlx_models::{Qwen35Runner, Qwen35RunnerBuilder};

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .context("usage: run_qwen35 <weights.gguf> [--mtp] [--prompt-ids 1,2,3]")?;

    let mut enable_mtp = false;
    let mut packed_weights = false;
    let mut prompt_ids: Vec<u32> = vec![1, 2, 3];
    let mut max_tokens: usize = 0;
    let mut max_seq_override: Option<usize> = None;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--mtp" => enable_mtp = true,
            "--packed" => packed_weights = true,
            "--max-tokens" => {
                max_tokens = args
                    .next()
                    .context("--max-tokens requires a number")?
                    .parse::<usize>()
                    .context("--max-tokens: invalid integer")?;
            }
            "--max-seq" => {
                max_seq_override = Some(
                    args.next()
                        .context("--max-seq requires a number")?
                        .parse::<usize>()
                        .context("--max-seq: invalid integer")?,
                );
            }
            "--prompt-ids" => {
                let raw = args
                    .next()
                    .context("--prompt-ids requires a comma-separated list")?;
                prompt_ids = raw
                    .split(',')
                    .map(|s| s.trim().parse::<u32>())
                    .collect::<std::result::Result<_, _>>()
                    .context("--prompt-ids: invalid integer")?;
            }
            other => bail!("unknown flag: {other}"),
        }
    }

    let max_seq = max_seq_override.unwrap_or_else(|| (prompt_ids.len() + max_tokens).max(8));
    println!(
        "[run_qwen35] loading {path:?} (prompt_len={}, max_seq={max_seq}, mtp={enable_mtp})",
        prompt_ids.len()
    );

    let mut runner: Qwen35Runner = Qwen35RunnerBuilder::default()
        .weights(&path)
        .max_seq(max_seq)
        .enable_mtp(enable_mtp)
        .packed_weights(packed_weights)
        .last_logits_only(true)
        .build()?;

    println!(
        "[run_qwen35] compiled — cfg: hidden={}, layers={} ({} MTP), \
         ssm_state={}, ssm_inner={}, dt_rank={}",
        runner.cfg().hidden_size,
        runner.cfg().num_hidden_layers,
        runner.cfg().nextn_predict_layers,
        runner.cfg().ssm_state_size,
        runner.cfg().ssm_inner_size,
        runner.cfg().ssm_time_step_rank,
    );

    if max_tokens == 0 {
        // Single prefill — print top-5.
        let out = runner.predict_logits(&prompt_ids)?;
        println!(
            "[run_qwen35] trunk logits: {} values (vocab≈{})",
            out.logits.len(),
            out.vocab_size
        );
        print_top5("trunk LM head", &out.logits, out.vocab_size);
        if let Some(mtp) = &out.mtp_logits {
            print_top5("MTP head", mtp, out.vocab_size);
        }
    } else {
        // Autoregressive greedy loop. Stream each id as it's
        // sampled — useful for preflight-checking that generation is
        // making progress on long files.
        println!(
            "[run_qwen35] generating {max_tokens} tokens (greedy, \
             repeated-prefill at O(seq · n_state²) per token)…"
        );
        let new_ids = runner.generate(&prompt_ids, max_tokens, |t| {
            print!("{t} ");
            std::io::Write::flush(&mut std::io::stdout()).ok();
            true
        })?;
        println!("\n[run_qwen35] generated: {new_ids:?}");
    }
    Ok(())
}

fn print_top5(label: &str, logits: &[f32], vocab: usize) {
    let take = vocab.min(logits.len());
    let mut idx: Vec<usize> = (0..take).collect();
    idx.sort_by(|&a, &b| {
        logits[b]
            .partial_cmp(&logits[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    println!("  [{label}] top-5:");
    for &i in idx.iter().take(5) {
        println!("    token {i:6}  logit {:>12.5}", logits[i]);
    }
}
