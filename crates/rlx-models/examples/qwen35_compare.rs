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

//! Parity-helper for `qwen35` GGUFs.
//!
//! Runs the RLX `Qwen35Runner` on a prompt-id sequence and prints
//! a logits header (vector length, top-K ids, top-K values) in a
//! format easy to diff against `llama-cli`'s `--seed 1 --logit-bias
//! --no-warmup --n-predict 1 --temp 0` output, or against a custom
//! `llama-cpp-rs` driver.
//!
//! Usage:
//! ```text
//! cargo run --release -p rlx-models --example qwen35_compare -- \
//!     <weights.gguf> [--packed] [--prompt-ids 1,2,3] [--top-k 16]
//! ```
//!
//! Output format (one line per top-K entry):
//! ```text
//! RLX_LOGIT idx=<rank> token=<id> value=<f32>
//! ```
//!
//! Companion reference script (Python or llama-cpp-rs example) should
//! emit:
//! ```text
//! REF_LOGIT idx=<rank> token=<id> value=<f32>
//! ```
//!
//! Then `diff` or compute cosine in your tool of choice. A future
//! slice should integrate `llama-cpp-rs` directly so this becomes
//! a single-command parity test rather than a two-process diff.

use anyhow::{Context, Result, bail};
use rlx_models::{Qwen35Runner, Qwen35RunnerBuilder};

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let path = args.next().context(
        "usage: qwen35_compare <weights.gguf> [--packed] [--prompt-ids ...] [--top-k N]",
    )?;

    let mut packed = false;
    let mut prompt_ids: Vec<u32> = vec![1, 2, 3];
    let mut top_k: usize = 16;
    let mut max_seq_override: Option<usize> = None;

    while let Some(a) = args.next() {
        match a.as_str() {
            "--packed" => packed = true,
            "--prompt-ids" => {
                let raw = args.next().context("--prompt-ids")?;
                prompt_ids = raw
                    .split(',')
                    .map(|s| s.trim().parse::<u32>())
                    .collect::<std::result::Result<_, _>>()
                    .context("--prompt-ids")?;
            }
            "--top-k" => {
                top_k = args.next().context("--top-k")?.parse()?;
            }
            "--max-seq" => {
                max_seq_override = Some(args.next().context("--max-seq")?.parse()?);
            }
            other => bail!("unknown flag: {other}"),
        }
    }

    let max_seq = max_seq_override.unwrap_or_else(|| prompt_ids.len().max(8));
    let mut runner: Qwen35Runner = Qwen35RunnerBuilder::default()
        .weights(&path)
        .max_seq(max_seq)
        .packed_weights(packed)
        .last_logits_only(true)
        .build()?;

    let cfg = runner.cfg();
    eprintln!(
        "# RLX qwen35: hidden={}, layers={} ({} MTP), heads={}/{} kv, \
         state={}, dt_rank={}",
        cfg.hidden_size,
        cfg.num_hidden_layers,
        cfg.nextn_predict_layers,
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.ssm_state_size,
        cfg.ssm_time_step_rank,
    );
    eprintln!("# prompt_ids: {prompt_ids:?}, packed={packed}, top_k={top_k}");

    let out = runner.predict_logits(&prompt_ids)?;
    eprintln!(
        "# RLX logits: len={} vocab≈{}",
        out.logits.len(),
        out.vocab_size
    );

    // Sort + print top-K. Use eprintln! for headers so stdout
    // contains only the RLX_LOGIT lines (= grep-friendly diff
    // against `REF_LOGIT` from the reference).
    let mut idx: Vec<usize> = (0..out.logits.len()).collect();
    idx.sort_by(|&a, &b| {
        out.logits[b]
            .partial_cmp(&out.logits[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for (rank, &i) in idx.iter().take(top_k).enumerate() {
        println!("RLX_LOGIT idx={rank} token={i} value={:.6}", out.logits[i]);
    }
    Ok(())
}
