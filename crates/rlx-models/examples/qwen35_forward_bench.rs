// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Real-model Qwen3.5 timing harness.
//
// ```bash
// cargo run --release -p rlx-models --example qwen35_forward_bench --features "metal,mlx" -- \
//   /path/to/model.gguf [--device cpu|metal|mlx] [--packed] [--tokens 16] [--predict]
// ```

use rlx_models::Qwen35RunnerBuilder;
use rlx_models::qwen3::SampleOpts;
use rlx_runtime::Device;
use std::env;
use std::time::Instant;

fn parse_device(s: &str) -> anyhow::Result<Device> {
    match s.to_ascii_lowercase().as_str() {
        "cpu" => Ok(Device::Cpu),
        "metal" => Ok(Device::Metal),
        "mlx" => Ok(Device::Mlx),
        other => anyhow::bail!("unknown device {other:?} (expected cpu|metal|mlx)"),
    }
}

fn main() -> anyhow::Result<()> {
    let mut args = env::args().skip(1);
    let weights = args
        .next()
        .ok_or_else(|| anyhow::anyhow!("usage: qwen35_forward_bench <weights.gguf> [--device cpu|metal|mlx] [--packed] [--tokens N] [--predict]"))?;
    let mut device = Device::Cpu;
    let mut packed = false;
    let mut tokens = 16usize;
    let mut predict = false;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--device" => {
                device = parse_device(
                    &args
                        .next()
                        .ok_or_else(|| anyhow::anyhow!("--device requires a value"))?,
                )?
            }
            other if other.starts_with("--device=") => {
                device = parse_device(other.trim_start_matches("--device="))?;
            }
            "--packed" => packed = true,
            "--predict" => predict = true,
            "--tokens" => {
                tokens = args
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("--tokens requires a value"))?
                    .parse()?
            }
            other if other.starts_with("--tokens=") => {
                tokens = other.trim_start_matches("--tokens=").parse()?;
            }
            other => anyhow::bail!("unknown flag: {other}"),
        }
    }

    let prompt: Vec<u32> = (1..=8).collect();
    let max_seq = (prompt.len() + tokens).max(16);

    let t0 = Instant::now();
    let mut runner = Qwen35RunnerBuilder::default()
        .weights(&weights)
        .device(device)
        .packed_weights(packed)
        .max_seq(max_seq)
        .last_logits_only(true)
        .build()?;
    let build_ms = t0.elapsed().as_secs_f64() * 1000.0;

    if predict {
        let t = Instant::now();
        let out = runner.predict_logits(&prompt)?;
        let predict_ms = t.elapsed().as_secs_f64() * 1000.0;
        let t = Instant::now();
        let _ = runner.predict_logits(&prompt)?;
        let predict_steady_ms = t.elapsed().as_secs_f64() * 1000.0;
        println!(
            "predict_logits: first={predict_ms:.1}ms steady={predict_steady_ms:.1}ms vocab={}",
            out.vocab_size
        );
    }

    let t = Instant::now();
    let generated = runner.generate_with_opts(&prompt, tokens, SampleOpts::greedy(), |_| true)?;
    let gen_ms = t.elapsed().as_secs_f64() * 1000.0;
    let tok_per_s = tokens as f64 / (gen_ms / 1000.0);

    println!("# qwen35_forward_bench");
    println!("  weights     : {weights}");
    println!("  device      : {device:?}");
    println!("  packed      : {packed}");
    println!("  build       : {build_ms:.1} ms");
    println!("  prompt_len  : {}", prompt.len());
    println!("  new_tokens  : {tokens}");
    println!("  generate    : {gen_ms:.1} ms ({tok_per_s:.2} tok/s)");
    println!("  generated   : {generated:?}");
    Ok(())
}
