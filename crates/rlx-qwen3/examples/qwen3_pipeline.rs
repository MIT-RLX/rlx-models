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

//! Pipeline-parallel Qwen3 generation across nodes.
//!
//! Launch one process per rank, each with the same `hosts.json` and its
//! own `--rank`:
//!
//! ```text
//! # hosts.json:  { "backend": "tcp", "hosts": ["10.0.0.1:9000", "10.0.0.2:9000"] }
//!
//! # on host 0:
//! cargo run -p rlx-qwen3 --example qwen3_pipeline -- \
//!     --rank 0 --hostfile hosts.json --model /path/to/qwen3 --prompt-ids 9707,11
//! # on host 1:
//! cargo run -p rlx-qwen3 --example qwen3_pipeline -- \
//!     --rank 1 --hostfile hosts.json --model /path/to/qwen3 --prompt-ids 9707,11
//! ```
//!
//! Each rank loads only the weights for its layer block. Rank 0 owns the
//! last layers + LM head and samples; the chosen token is broadcast so all
//! ranks stay in lockstep. Only rank 0 prints.
//!
//! Tokenization is out of scope here — pass prompt token ids directly via
//! `--prompt-ids`. Wire a tokenizer (see `rlx_qwen3::high_level_runner`)
//! for text in/out.

#[path = "common/mod.rs"]
mod common;

use anyhow::{Context, Result, bail};
use common::argmax;
use rlx_distributed::{DistConfig, ParallelMode, PipelineCoordinator};
use rlx_qwen3::Qwen3Config;
use rlx_qwen3::pipeline::Qwen3PipelineStage;
use rlx_qwen3::pipeline_decode::Qwen3PipelineDecodeStage;
use rlx_runtime::Device;
use std::collections::HashMap;

struct Args {
    rank: u32,
    hostfile: String,
    model: String,
    prompt_ids: Vec<u32>,
    max_tokens: usize,
    device: Device,
    decode: bool,
}

fn parse_args() -> Result<Args> {
    let mut rank = None;
    let mut hostfile = None;
    let mut model = None;
    let mut prompt_ids = Vec::new();
    let mut max_tokens = 32usize;
    let mut device = Device::Cpu;
    let mut decode = false;

    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        match flag.as_str() {
            "--rank" => rank = Some(it.next().context("--rank value")?.parse()?),
            "--hostfile" => hostfile = Some(it.next().context("--hostfile value")?),
            "--model" => model = Some(it.next().context("--model value")?),
            "--max-tokens" => max_tokens = it.next().context("--max-tokens value")?.parse()?,
            // cpu | metal | mlx | cuda | rocm | gpu (needs the matching cargo
            // feature built into rlx-qwen3, e.g. `--features metal`).
            "--device" => {
                device = it
                    .next()
                    .context("--device value")?
                    .parse()
                    .map_err(|e| anyhow::anyhow!("bad --device: {e}"))?;
            }
            // Use the KV-cached decode stage (O(layers)/token) instead of
            // the prefill-recompute stage.
            "--decode" => decode = true,
            "--prompt-ids" => {
                prompt_ids = it
                    .next()
                    .context("--prompt-ids value")?
                    .split(',')
                    .map(|s| s.trim().parse::<u32>())
                    .collect::<Result<_, _>>()?;
            }
            other => bail!("unknown flag {other}"),
        }
    }
    Ok(Args {
        rank: rank.context("--rank is required")?,
        hostfile: hostfile.context("--hostfile is required")?,
        model: model.context("--model is required")?,
        prompt_ids,
        max_tokens,
        device,
        decode,
    })
}

/// Load `config.json` + drain all weights into the in-memory map the
/// pipeline stage filters down to its block.
fn load_config_and_weights(
    model_dir: &str,
) -> Result<(Qwen3Config, HashMap<String, (Vec<f32>, Vec<usize>)>)> {
    let cfg_path = std::path::Path::new(model_dir).join("config.json");
    let cfg: Qwen3Config = serde_json::from_str(
        &std::fs::read_to_string(&cfg_path)
            .with_context(|| format!("reading {}", cfg_path.display()))?,
    )
    .context("parsing Qwen3 config.json")?;

    let mut loader = rlx_core::weight_loader::load_from_path(model_dir)?;
    let mut weights = HashMap::new();
    for k in loader.remaining_keys() {
        let v = loader.take(&k).with_context(|| format!("draining {k}"))?;
        let canonical = rlx_core::weight_loader::gguf_to_hf_name(&k).unwrap_or_else(|| k.clone());
        weights.insert(canonical, v);
    }
    Ok((cfg, weights))
}

fn main() -> Result<()> {
    let args = parse_args()?;

    // 1. Form the process group from the hostfile.
    let dist = DistConfig::load(&args.hostfile, Some(args.rank), ParallelMode::Pipeline)?;
    let group = dist.connect().context("forming process group")?;
    let leader = group.is_leader();
    let rprintln = |s: &str| {
        if leader {
            println!("{s}");
        }
    };

    // 2. Load config + weights for this rank's block.
    let (cfg, weights) = load_config_and_weights(&args.model)?;
    rprintln(&format!(
        "rank {}/{} on {:?}, {} mode",
        dist.rank,
        dist.world_size,
        args.device,
        if args.decode {
            "decode (KV-cached)"
        } else {
            "prefill"
        }
    ));

    // 3. Generate. Every rank calls forward_step in lockstep; only the
    //    First/Single rank reads `tokens`, only rank 0 samples, and the
    //    token is broadcast so all ranks advance together. Both stage types
    //    implement BlockRunner, so the loop is identical.
    let coord = PipelineCoordinator::new(group);
    let mut tokens = args.prompt_ids.clone();
    if tokens.is_empty() {
        bail!("--prompt-ids must contain at least one token id");
    }

    let generated = if args.decode {
        let mut stage =
            Qwen3PipelineDecodeStage::new(cfg, args.device, dist.rank, dist.world_size, weights);
        coord.generate(&mut stage, &mut tokens, args.max_tokens, argmax, |_| false)?
    } else {
        let mut stage =
            Qwen3PipelineStage::new(cfg, args.device, dist.rank, dist.world_size, weights);
        coord.generate(&mut stage, &mut tokens, args.max_tokens, argmax, |_| false)?
    };

    rprintln(&format!("generated ids: {generated:?}"));
    Ok(())
}
