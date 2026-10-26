// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! Perplexity of an LM's full-sequence graph on fixed token ids, in f32 and in
//! each number format. Forward only, so it needs a fraction of the memory
//! `rlx-run qat` takes to report the same number.
//!
//! ```text
//! cargo run --release -p rlx-models --features train,qwen3,gemma,llama32 \
//!     --example lm_format_ppl -- DIR ids.txt bf16 f8 f4
//! ```
//!
//! `ids.txt` holds comma-separated token ids. `RLX_NUMBER_FORMAT_KEEP` and
//! `RLX_NUMBER_FORMAT_ROUND_RESIDUAL` apply; `DEVICE` (default `cpu`) picks the
//! backend and `PARAMS_IN_GRAPH=1` rounds weights inside the graph (QAT's
//! path) instead of on upload.

use anyhow::{Context, Result};
use rlx_runtime::{CompileOptions, Device, NumberFormat, Session};

/// Mean next-token cross-entropy of `logits` `[seq, vocab]` against `ids`.
fn nll(logits: &[f32], ids: &[u32]) -> f64 {
    let vocab = logits.len() / ids.len();
    let mut total = 0f64;
    for (t, row) in logits.chunks_exact(vocab).take(ids.len() - 1).enumerate() {
        let max = row.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v)) as f64;
        let lse = max
            + row
                .iter()
                .map(|&v| (v as f64 - max).exp())
                .sum::<f64>()
                .ln();
        total += lse - row[ids[t + 1] as usize] as f64;
    }
    total / (ids.len() - 1) as f64
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let dir = std::path::Path::new(args.first().context("weights dir")?);
    let ids: Vec<u32> = std::fs::read_to_string(args.get(1).context("ids file")?)?
        .split(',')
        .map(|s| s.trim().parse())
        .collect::<Result<_, _>>()?;
    let lm = rlx_models::lm_train::trainable_lm(dir, 1, ids.len())?;
    let mut inputs = vec![(
        lm.token_input.clone(),
        ids.iter().map(|&x| x as f32).collect::<Vec<f32>>(),
    )];
    inputs.extend(lm.extra_inputs.iter().cloned());
    let refs: Vec<(&str, &[f32])> = inputs
        .iter()
        .map(|(n, v)| (n.as_str(), v.as_slice()))
        .collect();
    let formats = std::iter::once(None).chain(args[2..].iter().map(Some));
    let mut base = None;
    println!("{} tokens", ids.len());
    for name in formats {
        let mut opts = CompileOptions::new();
        opts.number_format_options.params_in_graph =
            std::env::var("PARAMS_IN_GRAPH").is_ok_and(|v| v == "1");
        if let Some(n) = name {
            opts.number_format = Some(n.parse::<NumberFormat>().map_err(anyhow::Error::msg)?);
        }
        let device = std::env::var("DEVICE").map_or(Device::Cpu, |d| {
            rlx_cli::parse_device(&d).unwrap_or_else(|e| panic!("DEVICE: {e}"))
        });
        let mut cg = Session::new(device).compile_with(lm.graph.clone(), &opts);
        for (k, v) in &lm.params {
            cg.set_param(k, v);
        }
        let ppl = nll(&cg.run(&refs).remove(0), &ids).exp();
        let b = *base.get_or_insert(ppl);
        println!(
            "{:<8} ppl {ppl:>9.3} {:>+8.1}%",
            name.map_or("f32", |n| n.as_str()),
            100.0 * (ppl / b - 1.0)
        );
    }
    Ok(())
}
