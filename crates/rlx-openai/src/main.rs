// RLX models — OpenAI-compatible multi-model server.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! Canonical OpenAI HTTP server for RLX chat LMs (`RegistryBackend`).
//!
//! ```sh
//! cargo run -p rlx-openai --release --features laguna,apple-silicon -- \
//!   --host 127.0.0.1 --port 8080 \
//!   --engine qwen3 --weights /path/qwen3 --model-id qwen3 --device metal \
//!   --engine laguna --weights /path/Laguna.gguf --tokenizer-dir /path/tok \
//!     --model-id laguna --device metal
//! ```

mod load;

use anyhow::{Context, Result, bail};
use load::{EngineSpec, enabled_engine_kinds, load_engine};
use rlx_serve::{ModelBackend, RegistryBackend, build_router_backend, serve_http};
use std::sync::Arc;

fn print_help() {
    eprintln!(
        "rlx-openai — central OpenAI-compatible server for RLX chat LMs

Usage:
  rlx-openai [--host HOST] [--port N] [--max-tokens N] \\
    --engine KIND --weights PATH [--tokenizer-dir DIR] [--model-id ID] \\
      [--device cpu|metal|mlx|…] [--eos IDS] \\
      [--continuous-batching] [--fused] [--max-batch N] [--batch-tokens N] [--kv-bits N] \\
      [--prompt-cache-mb N] \\
    [--engine KIND …]

Kinds enabled in this build: {}

Shared flags may appear anywhere. Per-engine flags apply to the preceding
`--engine` until the next `--engine`.

Prefer this binary over per-crate `--serve` / the Qwen3-only `rlx-serve` bin.
",
        enabled_engine_kinds().join(", ")
    );
}

fn parse_args(args: &[String]) -> Result<(String, u16, usize, Vec<EngineSpec>)> {
    let mut host = "127.0.0.1".to_string();
    let mut port = 8080u16;
    let mut max_tokens = 256usize;
    let mut engines: Vec<EngineSpec> = Vec::new();
    let mut cur: Option<EngineSpec> = None;

    let flush = |engines: &mut Vec<EngineSpec>, cur: &mut Option<EngineSpec>| {
        if let Some(spec) = cur.take() {
            engines.push(spec);
        }
    };

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            "--host" => {
                i += 1;
                host = args
                    .get(i)
                    .cloned()
                    .context("--host requires a value")?;
            }
            "--port" => {
                i += 1;
                port = args
                    .get(i)
                    .ok_or_else(|| anyhow::anyhow!("--port requires a value"))?
                    .parse()
                    .context("--port")?;
            }
            "--max-tokens" => {
                i += 1;
                max_tokens = args
                    .get(i)
                    .ok_or_else(|| anyhow::anyhow!("--max-tokens requires a value"))?
                    .parse()
                    .context("--max-tokens")?;
            }
            "--engine" => {
                flush(&mut engines, &mut cur);
                i += 1;
                let kind = args
                    .get(i)
                    .cloned()
                    .context("--engine requires KIND")?;
                cur = Some(EngineSpec::new(kind));
            }
            "--weights" | "--model" => {
                i += 1;
                let path = args
                    .get(i)
                    .cloned()
                    .context("--weights requires PATH")?;
                let spec = cur
                    .as_mut()
                    .ok_or_else(|| anyhow::anyhow!("--weights must follow --engine KIND"))?;
                spec.weights = path.into();
            }
            "--tokenizer-dir" | "--tokenizer" => {
                i += 1;
                let path = args
                    .get(i)
                    .cloned()
                    .context("--tokenizer-dir requires DIR")?;
                let spec = cur.as_mut().ok_or_else(|| {
                    anyhow::anyhow!("--tokenizer-dir must follow --engine KIND")
                })?;
                spec.tokenizer_dir = Some(path.into());
            }
            "--model-id" => {
                i += 1;
                let id = args
                    .get(i)
                    .cloned()
                    .context("--model-id requires a value")?;
                let spec = cur
                    .as_mut()
                    .ok_or_else(|| anyhow::anyhow!("--model-id must follow --engine KIND"))?;
                spec.model_id = id;
            }
            "--device" => {
                i += 1;
                let d = args
                    .get(i)
                    .cloned()
                    .context("--device requires a value")?;
                let spec = cur
                    .as_mut()
                    .ok_or_else(|| anyhow::anyhow!("--device must follow --engine KIND"))?;
                spec.device = d;
            }
            "--eos" => {
                i += 1;
                let raw = args.get(i).context("--eos requires IDS")?;
                let spec = cur
                    .as_mut()
                    .ok_or_else(|| anyhow::anyhow!("--eos must follow --engine KIND"))?;
                spec.eos = raw
                    .split(',')
                    .filter_map(|s| s.trim().parse().ok())
                    .collect();
            }
            "--prompt-cache-mb" => {
                i += 1;
                let n: usize = args
                    .get(i)
                    .ok_or_else(|| anyhow::anyhow!("--prompt-cache-mb requires a value"))?
                    .parse()
                    .context("--prompt-cache-mb")?;
                let spec = cur.as_mut().ok_or_else(|| {
                    anyhow::anyhow!("--prompt-cache-mb must follow --engine KIND")
                })?;
                spec.prompt_cache_mb = n;
            }
            "--continuous-batching" => {
                let spec = cur.as_mut().ok_or_else(|| {
                    anyhow::anyhow!("--continuous-batching must follow --engine KIND")
                })?;
                spec.continuous = true;
            }
            "--fused" => {
                let spec = cur
                    .as_mut()
                    .ok_or_else(|| anyhow::anyhow!("--fused must follow --engine KIND"))?;
                spec.fused = true;
            }
            "--max-batch" => {
                i += 1;
                let n: usize = args
                    .get(i)
                    .ok_or_else(|| anyhow::anyhow!("--max-batch requires a value"))?
                    .parse()
                    .context("--max-batch")?;
                let spec = cur
                    .as_mut()
                    .ok_or_else(|| anyhow::anyhow!("--max-batch must follow --engine KIND"))?;
                spec.max_batch = n;
            }
            "--batch-tokens" => {
                i += 1;
                let n: usize = args
                    .get(i)
                    .ok_or_else(|| anyhow::anyhow!("--batch-tokens requires a value"))?
                    .parse()
                    .context("--batch-tokens")?;
                let spec = cur.as_mut().ok_or_else(|| {
                    anyhow::anyhow!("--batch-tokens must follow --engine KIND")
                })?;
                spec.batch_tokens = n;
            }
            "--kv-bits" => {
                i += 1;
                let n: u32 = args
                    .get(i)
                    .ok_or_else(|| anyhow::anyhow!("--kv-bits requires a value"))?
                    .parse()
                    .context("--kv-bits")?;
                let spec = cur
                    .as_mut()
                    .ok_or_else(|| anyhow::anyhow!("--kv-bits must follow --engine KIND"))?;
                spec.kv_bits = n;
            }
            other => bail!("unknown arg {other} (see --help)"),
        }
        i += 1;
    }
    flush(&mut engines, &mut cur);

    if engines.is_empty() {
        bail!("pass at least one `--engine KIND --weights PATH` (see --help)");
    }
    Ok((host, port, max_tokens, engines))
}

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        print_help();
        std::process::exit(2);
    }
    let (host, port, max_tokens, specs) = parse_args(&args)?;

    let mut registry = RegistryBackend::new();
    for spec in &specs {
        let engine = load_engine(spec)?;
        registry = registry.register(engine);
    }
    let cards = registry.model_cards();
    eprintln!(
        "[rlx-openai] registered {} engine(s): {}",
        registry.len(),
        cards
            .iter()
            .map(|c| c.id.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    );

    let app = build_router_backend(Arc::new(registry), max_tokens);
    serve_http(app, &host, port).await
}
