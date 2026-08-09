//! TTFT + steady-state TPS microbench for the LFM2/LFM2.5 GGUF runner.
//!
//!   cargo run -p rlx-lfm --example lfm2_bench --features tokenizer --release -- \
//!       --weights FILE.gguf --device cpu --max-tokens 64
//!
//! For Metal/MLX add `--features tokenizer,apple-silicon`; CUDA/ROCm/Vulkan add
//! the matching feature. Warms the compiled graph + dequant cache first, then
//! reports:  TTFT (prefill → first token) and steady-state decode tok/s.
use anyhow::{Result, anyhow};
use rlx_cli::parse_standard_device;
use rlx_lfm::{Lfm2GgufRunner, resolve_gguf};
use std::path::PathBuf;
use std::time::Instant;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut weights: Option<String> = std::env::var("RLX_LFM_WEIGHTS").ok();
    let mut device = "cpu".to_string();
    let mut prompt = "The capital of France is".to_string();
    let mut max_tokens = 64usize;
    let mut repeat = 1usize;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--weights" => {
                i += 1;
                weights = Some(
                    args.get(i)
                        .ok_or_else(|| anyhow!("--weights needs a value"))?
                        .clone(),
                );
            }
            "--device" => {
                i += 1;
                device = args
                    .get(i)
                    .ok_or_else(|| anyhow!("--device needs a value"))?
                    .clone();
            }
            "--prompt" => {
                i += 1;
                prompt = args
                    .get(i)
                    .ok_or_else(|| anyhow!("--prompt needs a value"))?
                    .clone();
            }
            "--max-tokens" => {
                i += 1;
                max_tokens = args
                    .get(i)
                    .ok_or_else(|| anyhow!("--max-tokens needs a value"))?
                    .parse()?;
            }
            "--repeat" => {
                i += 1;
                repeat = args
                    .get(i)
                    .ok_or_else(|| anyhow!("--repeat needs a value"))?
                    .parse()?;
            }
            other => return Err(anyhow!("unknown arg {other}")),
        }
        i += 1;
    }
    if repeat > 1 {
        let base = prompt.clone();
        for _ in 1..repeat {
            prompt.push(' ');
            prompt.push_str(&base);
        }
    }

    let weights =
        weights.ok_or_else(|| anyhow!("pass --weights <FILE.gguf|DIR> or set RLX_LFM_WEIGHTS"))?;
    let gguf = resolve_gguf(&PathBuf::from(&weights))?;
    let dev = parse_standard_device("lfm", &device)?;

    let runner = Lfm2GgufRunner::open(&gguf, dev)?;
    let ids = rlx_qwen35::encode_prompt_from_gguf(&gguf, &prompt)?;

    // Warm: build the cached decode graph + warm the dequant cache.
    let _ = runner.generate(&ids, 8, |_| true)?;

    // Timed run: capture TTFT (first token) and steady-state decode rate.
    let t0 = Instant::now();
    let mut ttft = None;
    let mut n = 0usize;
    let _ = runner.generate(&ids, max_tokens, |_| {
        if ttft.is_none() {
            ttft = Some(t0.elapsed());
        }
        n += 1;
        true
    })?;
    let total = t0.elapsed();
    let ttft = ttft.unwrap_or(total);
    let steady = total.saturating_sub(ttft);
    let steady_tps = if n > 1 {
        (n - 1) as f64 / steady.as_secs_f64().max(1e-9)
    } else {
        0.0
    };

    println!(
        "[bench] {dev:?}  prompt_tok={}  gen={n}  TTFT={:.1}ms  steady={:.2} tok/s  overall={:.2} tok/s",
        ids.len(),
        ttft.as_secs_f64() * 1e3,
        steady_tps,
        n as f64 / total.as_secs_f64().max(1e-9),
    );
    Ok(())
}
