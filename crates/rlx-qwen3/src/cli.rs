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

use crate::{Precision, Qwen3ConfigSource, Qwen3Runner, SampleOpts};
use anyhow::{Context, Result, anyhow, bail};
use rlx_cli::{WeightFormat, WeightsResolveCli, parse_standard_device, req};
use std::io::Write;
use std::path::PathBuf;

const USAGE: &str = "\
rlx-qwen3 — run a Qwen3 LM (safetensors or gguf)
USAGE:
  rlx-qwen3 --weights <PATH> [flags]

FLAGS:
  --weights <PATH>       required
  --device <NAME>        cpu|metal|mlx|cuda|rocm|gpu|vulkan (default cpu)
  --config <PATH>        override config.json
  --format <FMT>         safetensors|gguf
  --prompt <TEXT>        comma-separated token ids (no tokenizer yet)
  --prompt-ids <I,I,I>
  --max-tokens <N>       default 32
  --max-seq <N>          default 128
  --precision <f32|f16-lm>
  --max-memory-gb <F>
  --no-stream
  --use-mtp
  --packed
  --temperature <F>
  --top-p <F>
  --prefer-quant <SUBSTR>  when --weights is a dir, pick a .gguf whose name contains SUBSTR (default Q4_K_M)
  --gguf-index <N>         pick the Nth .gguf after sorting (0-based; overrides --prefer-quant)
";

pub fn run(args: &[String]) -> Result<()> {
    let mut weights: Option<PathBuf> = None;
    let mut device = "cpu".to_string();
    let mut config: Option<PathBuf> = None;
    let mut format: Option<String> = None;
    let mut prompt: Option<String> = None;
    let mut prompt_ids: Option<Vec<u32>> = None;
    let mut max_tokens = 32usize;
    let mut max_seq = 128usize;
    let mut precision = "f32".to_string();
    let mut max_memory_gb: Option<f32> = None;
    let mut stream = true;
    let mut use_mtp = false;
    let mut packed = false;
    let mut temperature = 0f32;
    let mut top_p = 1f32;
    let mut resolve_cli = WeightsResolveCli::default();

    let mut i = 0;
    while i < args.len() {
        let a = &args[i];
        match a.as_str() {
            "--weights" => weights = Some(req(args, &mut i)?.into()),
            "--device" => device = req(args, &mut i)?,
            "--config" => config = Some(req(args, &mut i)?.into()),
            "--format" => format = Some(req(args, &mut i)?),
            "--prompt" => prompt = Some(req(args, &mut i)?),
            "--prompt-ids" => {
                prompt_ids = Some(
                    req(args, &mut i)?
                        .split(',')
                        .map(|s| s.trim().parse::<u32>())
                        .collect::<Result<_, _>>()
                        .context("--prompt-ids: comma-separated u32 list")?,
                );
            }
            "--max-tokens" => {
                max_tokens = req(args, &mut i)?.parse().context("--max-tokens: usize")?;
            }
            "--max-seq" => max_seq = req(args, &mut i)?.parse().context("--max-seq: usize")?,
            "--precision" => precision = req(args, &mut i)?,
            "--max-memory-gb" => {
                max_memory_gb = Some(req(args, &mut i)?.parse().context("--max-memory-gb: f32")?);
            }
            "--no-stream" => {
                stream = false;
                i += 1;
            }
            "--use-mtp" => {
                use_mtp = true;
                i += 1;
            }
            "--packed" => {
                packed = true;
                i += 1;
            }
            "--temperature" => {
                temperature = req(args, &mut i)?.parse().context("--temperature: f32")?;
            }
            "--top-p" => top_p = req(args, &mut i)?.parse().context("--top-p: f32")?,
            "--prefer-quant" | "--prefer" | "-p" => {
                resolve_cli.prefer_gguf = Some(req(args, &mut i)?);
            }
            "--gguf-index" => {
                resolve_cli.gguf_index =
                    Some(req(args, &mut i)?.parse().context("--gguf-index: usize")?);
            }
            "--help" | "-h" => {
                print!("{USAGE}");
                return Ok(());
            }
            other => bail!("unknown flag: {other}"),
        }
    }

    let weights_in = weights.ok_or_else(|| anyhow!("--weights is required"))?;
    let weights = rlx_cli::resolve_weights_cli(&weights_in, &resolve_cli)?;
    let device = parse_standard_device("qwen3", &device)?;
    let precision = match precision.as_str() {
        "f32" => Precision::F32,
        "f16-lm" | "f16_lm" => Precision::F16LmHead,
        other => bail!("--precision: expected f32|f16-lm, got {other}"),
    };
    let format = format.as_deref().map(WeightFormat::parse).transpose()?;
    let sample = SampleOpts {
        temperature,
        top_p,
        ..SampleOpts::greedy()
    };

    let mut b = Qwen3Runner::builder()
        .weights(weights.clone())
        .prefer_gguf_quant(
            resolve_cli
                .prefer_gguf
                .clone()
                .unwrap_or_else(|| "Q4_K_M".into()),
        )
        .device(device)
        .max_seq(max_seq)
        .precision(precision)
        .stream(stream)
        .use_mtp(use_mtp)
        .packed_weights(packed)
        .sample(sample);
    if let Some(fmt) = format {
        b = b.format(fmt);
    }
    if let Some(p) = config {
        b = b.config(Qwen3ConfigSource::JsonFile(p));
    }
    if let Some(g) = max_memory_gb {
        b = b.max_memory_gb(g);
    }

    let ids = match (prompt_ids, prompt) {
        (Some(ids), _) => ids,
        (None, Some(p)) => p
            .split(',')
            .map(|s| s.trim().parse::<u32>())
            .collect::<Result<_, _>>()
            .context("--prompt must be comma-separated u32 ids; use --prompt-ids")?,
        (None, None) => vec![1u32, 17, 42, 314, 2718, 9001, 27182, 8128],
    };

    eprintln!(
        "[rlx-qwen3] weights={weights:?} device={device:?} max_seq={max_seq} \
         precision={precision:?} stream={stream}"
    );
    let mut runner = b.build()?;
    eprintln!(
        "[rlx-qwen3] compiled — vocab={} hidden={} layers={}",
        runner.config().vocab_size,
        runner.config().hidden_size,
        runner.config().num_hidden_layers
    );

    let t0 = std::time::Instant::now();
    let mut printed = 0;
    if packed {
        eprintln!(
            "[rlx-qwen3] packed streaming: each token costs ~one full prefill \
             (low-memory path for large Q4_K_M GGUFs)"
        );
    }
    runner.generate(&ids, max_tokens, |tok| {
        if stream {
            print!("{tok} ");
            let _ = std::io::stdout().flush();
        }
        printed += 1;
    })?;
    let dt = t0.elapsed();
    println!();
    eprintln!(
        "[rlx-qwen3] generated {printed} tokens in {:.2?} ({:.1} tok/s)",
        dt,
        printed as f64 / dt.as_secs_f64()
    );
    Ok(())
}
