// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Run MiniCPM5-1B from Hugging Face safetensors via `MiniCpm5Runner`.
//
// Usage:
//   just fetch-minicpm5
//   RLX_MINICPM5_WEIGHTS=/tmp/rlx-weights/MiniCPM5-1B/model-00000-of-00001.safetensors \
//       cargo run -p rlx-models --example run_minicpm5 --release
//
// Equivalent CLI:
//   just minicpm5 -- --weights "$RLX_MINICPM5_WEIGHTS" --device cpu \
//       --prompt-ids 1,42,314 --max-tokens 16

use anyhow::{Context, Result};
use rlx_minicpm5::MiniCpm5Runner;
use rlx_runtime::Device;
use std::path::PathBuf;

fn weights_path() -> Result<PathBuf> {
    if let Ok(p) = std::env::var("RLX_MINICPM5_WEIGHTS") {
        return Ok(PathBuf::from(p));
    }
    let default = PathBuf::from("/tmp/rlx-weights/MiniCPM5-1B/model-00000-of-00001.safetensors");
    if default.is_file() {
        return Ok(default);
    }
    anyhow::bail!(
        "set RLX_MINICPM5_WEIGHTS or run `just fetch-minicpm5` (expected {})",
        default.display()
    )
}

fn main() -> Result<()> {
    let weights = weights_path()?;
    let device = std::env::var("RLX_MINICPM5_DEVICE").unwrap_or_else(|_| "cpu".into());
    let device = match device.as_str() {
        "cpu" => Device::Cpu,
        "metal" | "mps" => Device::Metal,
        "mlx" => Device::Mlx,
        "cuda" => Device::Cuda,
        other => anyhow::bail!("RLX_MINICPM5_DEVICE: unsupported {other:?}"),
    };

    let max_seq: usize = std::env::var("RLX_MINICPM5_MAX_SEQ")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(512);

    let mut runner = MiniCpm5Runner::builder()
        .weights(&weights)
        .device(device)
        .max_seq(max_seq)
        .build()
        .context("MiniCpm5Runner::build")?;

    let cfg = runner.llama_config();
    eprintln!(
        "[minicpm5] vocab={} hidden={} layers={} heads={}/{} device={device:?}",
        cfg.vocab_size,
        cfg.hidden_size,
        cfg.num_hidden_layers,
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
    );

    let prompt: Vec<u32> = match std::env::var("RLX_MINICPM5_PROMPT_IDS") {
        Ok(s) => s
            .split(',')
            .map(|t| t.trim().parse::<u32>())
            .collect::<Result<Vec<_>, _>>()
            .context("RLX_MINICPM5_PROMPT_IDS")?,
        Err(_) => vec![1, 42, 314, 2718],
    };

    let n_new: usize = std::env::var("RLX_MINICPM5_MAX_TOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(16);

    eprintln!(
        "[minicpm5] prompt_len={} generating {n_new} tokens",
        prompt.len()
    );

    let t0 = std::time::Instant::now();
    let _logits = runner.predict_logits(&prompt)?;
    eprintln!(
        "[minicpm5] prefill logits n_vocab={} ({:.1} ms)",
        _logits.len(),
        t0.elapsed().as_secs_f64() * 1000.0
    );

    let t1 = std::time::Instant::now();
    let out = runner.generate(&prompt, n_new, |tok| eprint!(" {tok}"))?;
    eprintln!();
    eprintln!(
        "[minicpm5] generated {} tokens ({:.1} ms)",
        out.len().saturating_sub(prompt.len()),
        t1.elapsed().as_secs_f64() * 1000.0
    );

    Ok(())
}
