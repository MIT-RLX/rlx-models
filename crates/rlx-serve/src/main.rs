// RLX models — OpenAI-compatible server.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! `rlx-serve` binary: load a Qwen3 model and serve the OpenAI API.
//!
//! ```sh
//! cargo run -p rlx-serve --release -- \
//!   --model /path/to/qwen3 --host 127.0.0.1 --port 8080 --device cpu --eos 151645
//! # fused continuous batching (shared-matmul throughput across the batch),
//! # with q8_0 KV-cache storage (~2× more concurrent contexts):
//! cargo run -p rlx-serve --release -- \
//!   --model /path/to/qwen3 --continuous-batching --fused --max-batch 16 \
//!   --batch-tokens 4096 --kv-bits 8
//! # then:
//! curl localhost:8080/v1/models
//! curl localhost:8080/v1/chat/completions -d \
//!   '{"model":"rlx","messages":[{"role":"user","content":"hi"}],"stream":true}'
//! ```

use anyhow::{Context, Result};
use rlx_qwen3::Qwen3Runner;
use rlx_runtime::Device;
use rlx_runtime::quantized_kv::KvQuant;
use rlx_serve::build_router;
use rlx_serve::engine::SingleEngine;
use rlx_serve::{BatchedEngine, Engine, FusedBatchRunner, RunnerBatchRunner};
use rlx_text::{auto_chat_template, load_tokenizer};
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<()> {
    let mut model: Option<String> = None;
    let mut tokenizer_path: Option<String> = None;
    let mut host = "127.0.0.1".to_string();
    let mut port = 8080u16;
    let mut device = Device::Cpu;
    let mut max_tokens = 256usize;
    let mut eos: Vec<u32> = Vec::new();
    let mut model_id = "rlx".to_string();
    let mut prompt_cache_mb: usize = 0;
    let mut continuous = false;
    let mut fused = false;
    let mut max_batch: usize = 8;
    let mut batch_tokens: usize = 2048;
    let mut kv_bits: u32 = 0; // 0=off, 16=f16, 8=q8_0, 5=q5_0, 4=q4_0

    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                i += 1;
                model = Some(args[i].clone());
            }
            "--tokenizer" => {
                i += 1;
                tokenizer_path = Some(args[i].clone());
            }
            "--host" => {
                i += 1;
                host = args[i].clone();
            }
            "--port" => {
                i += 1;
                port = args[i].parse().context("--port")?;
            }
            "--device" => {
                i += 1;
                device = Device::from_str(&args[i]).map_err(|e| anyhow::anyhow!("{e}"))?;
            }
            "--max-tokens" => {
                i += 1;
                max_tokens = args[i].parse().context("--max-tokens")?;
            }
            "--eos" => {
                i += 1;
                eos = args[i]
                    .split(',')
                    .filter_map(|s| s.trim().parse().ok())
                    .collect();
            }
            "--model-id" => {
                i += 1;
                model_id = args[i].clone();
            }
            "--prompt-cache-mb" => {
                i += 1;
                prompt_cache_mb = args[i].parse().context("--prompt-cache-mb")?;
            }
            "--continuous-batching" => continuous = true,
            "--fused" => fused = true,
            "--kv-bits" => {
                i += 1;
                kv_bits = args[i].parse().context("--kv-bits")?;
            }
            "--max-batch" => {
                i += 1;
                max_batch = args[i].parse().context("--max-batch")?;
            }
            "--batch-tokens" => {
                i += 1;
                batch_tokens = args[i].parse().context("--batch-tokens")?;
            }
            other => anyhow::bail!("unknown arg {other}"),
        }
        i += 1;
    }

    let model = model.context("--model <dir-or-weights> is required")?;
    let model_path = PathBuf::from(&model);
    let tok_path = tokenizer_path
        .map(PathBuf::from)
        .unwrap_or_else(|| model_path.join("tokenizer.json"));

    eprintln!("[rlx-serve] loading model {model} on {device:?}");
    let runner = Qwen3Runner::builder()
        .weights(model_path.clone())
        .device(device)
        .build()
        .context("building Qwen3Runner")?;
    let tokenizer =
        load_tokenizer(&tok_path).with_context(|| format!("loading tokenizer {tok_path:?}"))?;
    let chat_template = auto_chat_template(&model_path).ok();
    if chat_template.is_none() {
        eprintln!("[rlx-serve] no chat template found; using plain role concatenation");
    }

    let engine: Arc<dyn Engine> = if continuous && fused {
        // Fused continuous batching: same-length decodes are folded into ONE
        // batched forward, so weight matmuls are shared across the batch — the
        // throughput multiplier. Needs the F32 generator (not packed weights).
        let generator = runner
            .into_generator()
            .context("--fused requires F32 weights (not packed/quantized)")?;
        let kv_quant = match kv_bits {
            0 => None,
            16 => Some(KvQuant::F16),
            8 => Some(KvQuant::Q8_0),
            5 => Some(KvQuant::Q5_0),
            4 => Some(KvQuant::Q4_0),
            other => anyhow::bail!("--kv-bits must be one of 0,4,5,8,16 (got {other})"),
        };
        eprintln!(
            "[rlx-serve] continuous batching (FUSED): max_batch={max_batch} batch_tokens={batch_tokens} kv_bits={kv_bits}"
        );
        Arc::new(BatchedEngine::new(
            Box::new(FusedBatchRunner::with_kv_quant(generator, kv_quant)),
            tokenizer,
            chat_template,
            eos,
            model_id,
            batch_tokens,
            max_batch,
        ))
    } else if continuous {
        // Continuous batching, no fusion: concurrent requests' decode steps are
        // time-sliced through one runner via KV swap. Eliminates head-of-line
        // blocking; `--fused` adds the shared-matmul throughput win on top.
        let br = RunnerBatchRunner::new(Box::new(runner)).context("RunnerBatchRunner")?;
        eprintln!(
            "[rlx-serve] continuous batching: max_batch={max_batch} batch_tokens={batch_tokens}"
        );
        Arc::new(BatchedEngine::new(
            Box::new(br),
            tokenizer,
            chat_template,
            eos,
            model_id,
            batch_tokens,
            max_batch,
        ))
    } else {
        let mut engine =
            SingleEngine::new(Box::new(runner), tokenizer, chat_template, eos, model_id);
        if prompt_cache_mb > 0 {
            engine = engine.with_prompt_cache(prompt_cache_mb << 20);
            eprintln!("[rlx-serve] prompt cache enabled: {prompt_cache_mb} MiB");
        }
        Arc::new(engine)
    };
    let app = build_router(engine, max_tokens);

    let addr = format!("{host}:{port}");
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .with_context(|| format!("binding {addr}"))?;
    eprintln!("[rlx-serve] listening on http://{addr}");
    axum::serve(listener, app).await?;
    Ok(())
}
