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

// RLX CLI for gemma
use crate::{GemmaConfigSource, GemmaRunner};
use anyhow::{Context, Result, anyhow, bail};
use rlx_cli::{WeightFormat, WeightsResolveCli, parse_gemma_device, req, resolve_weights_cli};
use rlx_core::LM_DEVICE_NAMES;
use rlx_qwen3::SampleOpts;
use std::io::Write;
use std::path::PathBuf;

pub fn run(args: &[String]) -> Result<()> {
    let mut weights: Option<PathBuf> = None;
    let mut device = "auto".to_string();
    let mut config: Option<PathBuf> = None;
    let mut format: Option<String> = None;
    let mut prompt: Option<String> = None;
    let mut system: Option<String> = None;
    let mut raw_prompt = false;
    let mut tokenizer: Option<PathBuf> = None;
    let mut prompt_ids: Option<Vec<u32>> = None;
    let mut max_tokens = 32usize;
    let mut max_seq = 128usize;
    let mut max_memory_gb: Option<f32> = None;
    let mut stream = true;
    let mut packed = false;
    let mut temperature = 0f32;
    let mut top_p = 1f32;
    let mut resolve_cli = WeightsResolveCli::default();

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--weights" => weights = Some(req(args, &mut i)?.into()),
            "--device" => device = req(args, &mut i)?,
            "--config" => config = Some(req(args, &mut i)?.into()),
            "--format" => format = Some(req(args, &mut i)?),
            "--prompt" => prompt = Some(req(args, &mut i)?),
            "--system" => system = Some(req(args, &mut i)?),
            "--raw-prompt" => {
                raw_prompt = true;
                i += 1;
            }
            "--tokenizer" => tokenizer = Some(req(args, &mut i)?.into()),
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
            "--max-memory-gb" => {
                max_memory_gb = Some(req(args, &mut i)?.parse().context("--max-memory-gb: f32")?);
            }
            "--no-stream" => {
                stream = false;
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
                eprintln!("rlx-gemma — see README for flags; --device accepts {LM_DEVICE_NAMES}");
                return Ok(());
            }
            other => bail!("unknown flag: {other}"),
        }
    }

    let weights = resolve_weights_cli(
        &weights.ok_or_else(|| anyhow!("--weights is required"))?,
        &resolve_cli,
    )?;
    let device = parse_gemma_device(&device)?;
    let format = format.as_deref().map(WeightFormat::parse).transpose()?;
    let fmt = match format {
        Some(f) => f,
        None => WeightFormat::from_path(&weights)?,
    };
    let use_packed = packed
        || (matches!(fmt, WeightFormat::Gguf)
            && std::fs::metadata(&weights)
                .map(|m| m.len() >= 256 * 1024 * 1024)
                .unwrap_or(false));
    let sample = SampleOpts {
        temperature,
        top_p,
        ..SampleOpts::greedy()
    };

    let mut b = GemmaRunner::builder()
        .weights(weights.clone())
        .device(device)
        .max_seq(max_seq)
        .stream(stream)
        .sample(sample)
        .packed_weights(use_packed)
        .format(fmt);
    if let Some(p) = config {
        b = b.config(GemmaConfigSource::JsonFile(p));
    }
    if let Some(g) = max_memory_gb {
        b = b.max_memory_gb(g);
    }

    let ids = if let Some(ids) = prompt_ids {
        ids
    } else if let Some(text) = prompt {
        crate::encode_chat_prompt_auto(
            &weights,
            tokenizer.as_deref(),
            system.as_deref(),
            &text,
            !raw_prompt,
        )?
    } else {
        vec![128000, 128006, 882, 128007, 271, 9906, 128009]
    };

    eprintln!(
        "[rlx-gemma] gemma: weights={weights:?} device={device:?} max_seq={max_seq} \
         stream={stream} packed={use_packed}"
    );
    let mut runner = b.build()?;
    eprintln!(
        "[rlx-gemma] compiled — vocab={} hidden={} layers={}",
        runner.config().vocab_size,
        runner.config().hidden_size,
        runner.config().num_hidden_layers
    );

    let t0 = std::time::Instant::now();
    let mut printed = 0;
    if use_packed {
        eprintln!(
            "[rlx-gemma] packed streaming: each token costs ~one full prefill (low-memory path)"
        );
    }
    runner.generate(&ids, max_tokens, |tok| {
        if stream {
            match crate::decode_token_auto(&weights, tokenizer.as_deref(), tok) {
                Ok(piece) => {
                    print!("{piece}");
                }
                Err(_) => print!("{tok} "),
            }
            std::io::stdout().flush().ok();
        }
        printed += 1;
    })?;
    let dt = t0.elapsed();
    println!();
    eprintln!(
        "[rlx-gemma] generated {printed} tokens in {:.2?} ({:.1} tok/s)",
        dt,
        printed as f64 / dt.as_secs_f64()
    );
    Ok(())
}
