// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Does a family's time-to-first-token scale with the PROMPT or with the
//! CONTEXT SETTING? Driven through `LmRunner`, so it covers any family
//! `auto_runner_with` can build.
//!
//! The bug class this detects: a family that compiles one prefill graph at
//! `max_seq` computes the entire context for every prompt. TTFT then tracks a
//! setting rather than the work — Qwen3.5-0.8B/Metal took 1.7-2.7s for a
//! 15-token prompt at max_seq 2048 and ~11.5s at 8192. Nothing fails, no test
//! goes red, and raising "context size" in a product quietly quadruples
//! first-token latency. Only a ratio across prompt lengths at a FIXED context
//! shows it.
//!
//! Reported per length: cold TTFT (includes any compile for that bucket) and
//! warm TTFT (second run, compile excluded). The verdict uses warm, because
//! that isolates "how much does it COMPUTE" from "did it compile".
//!
//! ```sh
//! cargo run --release -p rlx-models --features runner,metal,qwen3,qwen35 \
//!   --example prefill_scaling -- model.gguf metal 8192
//! ```

use anyhow::{Context, Result, bail};
use rlx_runtime::{Device, is_available};
use std::path::PathBuf;
use std::time::Instant;

/// Fractions of the context to probe. The shortest must be well under the
/// narrowest sane prefill bucket, or a bucketed family looks width-pinned too.
const PROMPT_LENS: [usize; 4] = [16, 128, 1024, 4096];

fn parse_device(s: &str) -> Result<Device> {
    Ok(match s.trim() {
        "cpu" => Device::Cpu,
        "metal" => Device::Metal,
        "mlx" => Device::Mlx,
        "cuda" => Device::Cuda,
        "rocm" => Device::Rocm,
        "wgpu" | "gpu" => Device::Gpu,
        "vulkan" => Device::Vulkan,
        other => bail!("unknown device `{other}`"),
    })
}

/// TTFT: time until `generate` hands back its first token.
fn ttft(runner: &mut dyn rlx_runtime::lm::LmRunner, prompt: &[u32]) -> Result<f64> {
    let t = Instant::now();
    let mut first: Option<f64> = None;
    runner.generate(prompt, 1, &mut |_| {
        if first.is_none() {
            first = Some(t.elapsed().as_secs_f64());
        }
        false
    })?;
    Ok(first.unwrap_or_else(|| t.elapsed().as_secs_f64()))
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let path = PathBuf::from(
        args.next()
            .context("usage: prefill_scaling <checkpoint> [device] [ctx]")?,
    );
    let device = match args.next() {
        Some(s) => parse_device(&s)?,
        None => Device::Cpu,
    };
    let ctx: usize = match args.next() {
        Some(s) => s.parse().context("ctx")?,
        None => 8192,
    };
    if !is_available(device) {
        bail!("device {device:?} is not available in this build");
    }

    // ONE runner at a FIXED context, reused for every length — that is the whole
    // point. Rebuilding per length would size the graph to each prompt and hide
    // exactly the behaviour under test.
    let opts = rlx_models::run::AutoOptions::new(device).max_seq(ctx);
    let mut runner = rlx_models::run::auto_runner_with(&path, &opts).context("auto_runner_with")?;
    let family = runner.family();
    println!(
        "[prefill] {} | family={family} | device={device:?} | ctx={ctx}",
        path.display()
    );
    println!(
        "[prefill] {:>8} {:>10} {:>10}",
        "prompt", "cold s", "warm s"
    );

    let lens: Vec<usize> = PROMPT_LENS
        .iter()
        .copied()
        .filter(|n| *n + 8 < ctx)
        .collect();
    let mut warm: Vec<(usize, f64)> = Vec::new();
    for &n in &lens {
        let prompt: Vec<u32> = (0..n).map(|i| 16 + (i % 512) as u32).collect();
        let cold = ttft(runner.as_mut(), &prompt)?;
        let hot = ttft(runner.as_mut(), &prompt)?;
        println!("[prefill] {n:>8} {cold:>10.3} {hot:>10.3}");
        warm.push((n, hot));
    }
    if warm.len() < 2 {
        bail!("ctx {ctx} is too small to probe more than one prompt length");
    }

    let (short_n, short_t) = warm[0];
    let (long_n, long_t) = *warm.last().unwrap();
    let ratio = if short_t > 0.0 {
        long_t / short_t
    } else {
        f64::NAN
    };
    println!(
        "[prefill] {long_n} tok / {short_n} tok = {ratio:.2}x warm TTFT \
         ({long_t:.3}s vs {short_t:.3}s)"
    );

    // A prompt 64-256x longer that costs about the same means the graph is
    // sized by the context, not the prompt. 1.5x is deliberately lenient: a
    // bucketed family still pays a fixed per-call overhead that dominates at
    // 16 tokens, so the honest signal is "barely moved", not "moved less than
    // linearly".
    if ratio < 1.5 {
        println!(
            "[prefill] WIDTH-PINNED: family `{family}` prefills at a fixed width, so TTFT \
             tracks ctx={ctx} rather than the prompt. Bucket the prefill graph with \
             `rlx_core::prefill_bucket_for` (see rlx-qwen35's `ensure_static_prefill_cache`)."
        );
        bail!("prefill does not scale with prompt length (ratio {ratio:.2}x)");
    }
    println!("[prefill] PASS ✓ TTFT scales with the prompt, not with ctx");
    Ok(())
}
