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

//! MiniMax-M3 CLI entry (`rlx-minimax --m3 …` / dispatched as `minimax-m3`).
//!
//! Token-id driven for now (the 200k-vocab tiktoken tokenizer wiring is a
//! follow-up): pass `--prompt-ids` as comma-separated ids and it runs the
//! prefill runner's greedy generate loop, printing the produced ids.

use anyhow::{Result, anyhow, bail};
use rlx_cli::LmRunner;
use rlx_runtime::Device;
use std::path::Path;

use super::runner::MiniMaxM3Runner;

/// CLI entry point. Recognized flags: `--weights <path>` (required),
/// `--config <config.json>`, `--prompt-ids <a,b,c>`, `--max-tokens <n>`,
/// `--device <cpu|metal|mlx|wgpu|cuda|vulkan>`.
pub fn cli_run(args: &[String]) -> Result<()> {
    let mut weights: Option<String> = None;
    let mut config: Option<String> = None;
    let mut prompt_ids: Vec<u32> = Vec::new();
    let mut max_tokens: usize = 16;
    let mut device = Device::Cpu;

    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--weights" => weights = it.next().cloned(),
            "--config" => config = it.next().cloned(),
            "--prompt-ids" => {
                let s = it
                    .next()
                    .ok_or_else(|| anyhow!("--prompt-ids needs a comma-separated value"))?;
                prompt_ids = s
                    .split(',')
                    .filter(|t| !t.is_empty())
                    .map(|t| t.trim().parse::<u32>())
                    .collect::<std::result::Result<_, _>>()
                    .map_err(|e| anyhow!("bad --prompt-ids: {e}"))?;
            }
            "--max-tokens" => {
                max_tokens = it
                    .next()
                    .ok_or_else(|| anyhow!("--max-tokens needs a value"))?
                    .parse()
                    .map_err(|e| anyhow!("bad --max-tokens: {e}"))?;
            }
            "--device" => {
                let d = it.next().ok_or_else(|| anyhow!("--device needs a value"))?;
                device = rlx_cli::parse_device(d).map_err(|e| anyhow!("bad --device: {e}"))?;
            }
            "--m3" | "minimax-m3" => {} // dispatch tokens, ignore
            other => bail!("rlx-minimax m3: unknown argument `{other}`"),
        }
    }

    let weights = weights.ok_or_else(|| anyhow!("--weights <path> is required"))?;
    if prompt_ids.is_empty() {
        bail!("--prompt-ids <a,b,c> is required (tokenizer wiring is a follow-up)");
    }

    let mut runner =
        MiniMaxM3Runner::from_pretrained(&weights, config.as_deref().map(Path::new), device)?;
    eprintln!(
        "rlx-minimax m3: loaded {} layers, vocab {}",
        runner.config().num_hidden_layers,
        runner.vocab_size()
    );

    // KV-cache incremental decode (falls back internally to per-length compile).
    let out = runner.decode_generate(&prompt_ids, max_tokens, |t| {
        print!("{t} ");
        use std::io::Write;
        let _ = std::io::stdout().flush();
        true
    })?;
    println!();
    eprintln!("rlx-minimax m3: generated {} tokens", out.len());
    Ok(())
}
