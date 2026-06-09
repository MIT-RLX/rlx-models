// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Run a Qwen3 model from a HuggingFace `.safetensors` checkpoint via
// the high-level `Qwen3Runner` builder. Streams tokens to stdout.
//
// Usage:
//   RLX_QWEN3_WEIGHTS=/path/to/model.safetensors \
//   RLX_QWEN3_CONFIG=/path/to/config.json \
//       cargo run --release -p rlx-models --features metal \
//           --example run_qwen3_safetensors
//
// If RLX_QWEN3_CONFIG is unset, the builder looks for `config.json`
// next to the safetensors file.
//
// Equivalent CLI call:
//   rlx-qwen3 (or `rlx-run qwen3`) --weights model.safetensors --device metal \
//                 --prompt-ids 1,17,42 --max-tokens 16

use anyhow::{Context, Result};
use rlx_models::run::{ConfigSource, Precision, Qwen3Runner};
use rlx_runtime::Device;

fn main() -> Result<()> {
    let weights = rlx_ir::env::var("RLX_QWEN3_WEIGHTS")
        .context("set RLX_QWEN3_WEIGHTS to /path/to/model.safetensors")?;
    let config = rlx_ir::env::var("RLX_QWEN3_CONFIG");

    let mut builder = Qwen3Runner::builder()
        .weights(weights)
        .device(Device::Metal) // change to Cpu / Mlx / Gpu as needed
        .max_seq(128) // prefill bucket size
        .precision(Precision::F32)
        .max_memory_gb(8.0) // soft cap; errors out if exceeded
        .stream(true);
    if let Some(cfg_path) = config {
        builder = builder.config(ConfigSource::JsonFile(cfg_path.into()));
    }
    let mut runner = builder.build()?;
    let cfg = runner.config();
    eprintln!(
        "[qwen3] compiled — vocab={} hidden={} layers={} heads={}/{} on {:?}",
        cfg.vocab_size,
        cfg.hidden_size,
        cfg.num_hidden_layers,
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        runner.device(),
    );

    // Token ids — replace with real tokenizer output for production
    // use; the runner intentionally stays tokenizer-agnostic.
    let prompt = [1u32, 17, 42, 314, 2718, 9001, 27182, 8128];
    eprintln!("[qwen3] prompt ids: {prompt:?}");

    let t0 = std::time::Instant::now();
    let tokens = runner.generate(&prompt, 16, |tok| {
        eprint!(" {tok}");
    })?;
    eprintln!();
    eprintln!(
        "[qwen3] generated {} tokens in {:?}",
        tokens.len(),
        t0.elapsed()
    );
    Ok(())
}
