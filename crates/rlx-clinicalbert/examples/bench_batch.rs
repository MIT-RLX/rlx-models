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

//! Latency sweep over batch sizes for encoder + optional MLM head.

#[path = "support/common.rs"]
mod common;

use anyhow::{Context, Result, bail};
use rlx_clinicalbert::{ClinicalBertRunner, ClinicalBertTokenizer, MlmExecMode, Pooling};
use std::path::PathBuf;
use std::time::Instant;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let weights = PathBuf::from(common::require_flag(&args, "--weights")?);
    let device = common::parse_device(
        &common::parse_flag(&args, "--device")?.unwrap_or_else(|| "cpu".into()),
    )?;
    let seq: usize = common::parse_flag(&args, "--seq")?
        .unwrap_or_else(|| "32".into())
        .parse()
        .context("--seq")?;
    let batches: Vec<usize> = common::parse_flag(&args, "--batches")?
        .unwrap_or_else(|| "1,4,8".into())
        .split(',')
        .map(str::trim)
        .map(|s| s.parse().context("--batches"))
        .collect::<Result<_>>()?;
    let iters: usize = common::parse_flag(&args, "--iters")?
        .unwrap_or_else(|| "10".into())
        .parse()
        .context("--iters")?;
    let mlm_mode = match common::parse_flag(&args, "--mlm-mode")?
        .unwrap_or_else(|| "auto".into())
        .to_ascii_lowercase()
        .as_str()
    {
        "cpu" => MlmExecMode::Cpu,
        "ingraph" | "in-graph" | "graph" => MlmExecMode::InGraph,
        "auto" | "default" => MlmExecMode::Auto,
        other => bail!("unknown --mlm-mode {other:?}"),
    };

    let tok = ClinicalBertTokenizer::from_dir_or_sibling(&weights)?;
    let sentence = "The patient was admitted with chest pain and shortness of breath.";

    println!("bench_batch device={device:?} seq={seq} iters={iters} mlm_mode={mlm_mode:?}");
    for batch in batches {
        let texts: Vec<&str> = (0..batch).map(|_| sentence).collect();
        let enc = tok.encode_batch(&texts, seq)?;
        let mut runner = ClinicalBertRunner::builder()
            .weights(&weights)
            .device(device)
            .batch(batch)
            .max_seq(seq)
            .pooling(Pooling::Cls)
            .with_pooler()
            .mlm_mode(mlm_mode)
            .build()?;

        // Warmup
        let hidden = runner.forward(
            &enc.input_ids,
            &enc.attention_mask,
            &enc.token_type_ids,
            &enc.position_ids,
        )?;
        let _ = runner.pooler_output(&hidden)?;
        let _ = runner.mlm_logits(&hidden)?;

        let t0 = Instant::now();
        for _ in 0..iters {
            let hidden = runner.forward(
                &enc.input_ids,
                &enc.attention_mask,
                &enc.token_type_ids,
                &enc.position_ids,
            )?;
            let _ = runner.pooler_output(&hidden)?;
            let _ = runner.mlm_logits(&hidden)?;
        }
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;
        println!(
            "batch={batch:>3}  {ms:>8.2} ms/iter  resolved_mlm={:?}",
            runner.mlm_mode()
        );
    }
    Ok(())
}
