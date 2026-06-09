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

// RLX CLI for llama32
use crate::{Llama32ConfigSource, Llama32Runner};
use anyhow::{Context, Result, anyhow, bail};
use rlx_cli::{WeightFormat, parse_llama32_device, req};
use rlx_qwen3::SampleOpts;
use std::io::Write;
use std::path::PathBuf;

pub fn run(args: &[String]) -> Result<()> {
    let mut weights: Option<PathBuf> = None;
    let mut device = "cpu".to_string();
    let mut config: Option<PathBuf> = None;
    let mut format: Option<String> = None;
    let mut prompt: Option<String> = None;
    let mut tokenizer: Option<PathBuf> = None;
    let mut prompt_ids: Option<Vec<u32>> = None;
    let mut max_tokens = 32usize;
    let mut max_seq = 512usize;
    let mut max_memory_gb: Option<f32> = None;
    let mut stream = true;
    let mut packed: Option<bool> = None;
    let mut bucketed_decode = true;
    let mut temperature = 0f32;
    let mut top_p = 1f32;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--weights" => weights = Some(req(args, &mut i)?.into()),
            "--device" => device = req(args, &mut i)?,
            "--config" => config = Some(req(args, &mut i)?.into()),
            "--format" => format = Some(req(args, &mut i)?),
            "--prompt" => prompt = Some(req(args, &mut i)?),
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
                packed = Some(true);
                i += 1;
            }
            "--no-packed" => {
                packed = Some(false);
                i += 1;
            }
            "--no-bucketed-decode" => {
                bucketed_decode = false;
                i += 1;
            }
            "--temperature" => {
                temperature = req(args, &mut i)?.parse().context("--temperature: f32")?;
            }
            "--top-p" => top_p = req(args, &mut i)?.parse().context("--top-p: f32")?,
            "--help" | "-h" => {
                eprintln!("rlx-llama32 — see README for flags");
                return Ok(());
            }
            other => bail!("unknown flag: {other}"),
        }
    }

    let weights = weights.ok_or_else(|| anyhow!("--weights is required"))?;
    let device = parse_llama32_device(&device)?;
    let weight_fmt = match format.as_deref() {
        Some("safetensors") => WeightFormat::Safetensors,
        Some("gguf") => WeightFormat::Gguf,
        Some(other) => bail!("--format: expected safetensors|gguf, got {other}"),
        None => WeightFormat::from_path(&weights)?,
    };
    let use_packed = packed.unwrap_or_else(|| {
        matches!(weight_fmt, WeightFormat::Gguf)
            && std::fs::metadata(&weights)
                .map(|m| m.len() >= 256 * 1024 * 1024)
                .unwrap_or(false)
    });
    let sample = SampleOpts {
        temperature,
        top_p,
        ..SampleOpts::greedy()
    };

    // MLX bucketed decode can diverge on some graphs; one-shot decode is slower
    // but matches CPU. Metal may still hit MPS limits on long contexts.
    let bucketed_decode = bucketed_decode && !matches!(device, rlx_runtime::Device::Mlx);

    let mut b = Llama32Runner::builder()
        .weights(weights.clone())
        .device(device)
        .max_seq(max_seq)
        .stream(stream)
        .sample(sample)
        .bucketed_decode_cache(bucketed_decode);
    if let Some(p) = packed {
        b = b.packed_weights(p);
    }
    if format.is_some() {
        b = b.format(weight_fmt);
    }
    if let Some(p) = config {
        b = b.config(Llama32ConfigSource::JsonFile(p));
    }
    if let Some(g) = max_memory_gb {
        b = b.max_memory_gb(g);
    }

    let ids = if let Some(ids) = prompt_ids {
        ids
    } else if let Some(text) = prompt {
        crate::encode_prompt_auto(&weights, tokenizer.as_deref(), &text)?
    } else {
        vec![128000, 128006, 882, 128007, 271, 9906, 128009]
    };

    eprintln!(
        "[rlx-llama32] llama32: weights={weights:?} device={device:?} max_seq={max_seq} \
         stream={stream} packed={use_packed}"
    );
    let mut runner = b.build()?;
    eprintln!(
        "[rlx-llama32] compiled — vocab={} hidden={} layers={}",
        runner.config().vocab_size,
        runner.config().hidden_size,
        runner.config().num_hidden_layers
    );

    let t0 = std::time::Instant::now();
    let mut generated = Vec::new();
    if use_packed {
        eprintln!(
            "[rlx-llama32] packed streaming: each token costs ~one full prefill (low-memory path)"
        );
    }
    runner.generate(&ids, max_tokens, |tok| {
        generated.push(tok);
        if stream {
            #[cfg(feature = "tokenizer")]
            {
                if let Ok(text) =
                    rlx_qwen35::decode_ids_auto(&weights, tokenizer.as_deref(), &generated, true)
                {
                    print!("\r{text}");
                    std::io::stdout().flush().ok();
                    return;
                }
            }
            print!("{tok} ");
            std::io::stdout().flush().ok();
        }
    })?;
    let dt = t0.elapsed();
    println!();
    eprintln!(
        "[rlx-llama32] generated {} tokens in {:.2?} ({:.1} tok/s)",
        generated.len(),
        dt,
        generated.len() as f64 / dt.as_secs_f64()
    );
    #[cfg(feature = "tokenizer")]
    if !generated.is_empty() {
        match rlx_qwen35::decode_ids_auto(&weights, tokenizer.as_deref(), &generated, true) {
            Ok(response) => {
                println!("[rlx-llama32] response:\n{response}");
            }
            Err(e) => eprintln!("[rlx-llama32] decode failed: {e:#}"),
        }
    }
    Ok(())
}
