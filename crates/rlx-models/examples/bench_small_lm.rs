// Quick decode throughput bench for small GGUF LMs (Gemma 3 270M, Phi-3-mini, …).
//
// ```sh
// RLX_BENCH_N_NEW=16 cargo run -p rlx-models --release \
//   --features gemma,phi,llama32,metal,mlx,coreml,apple-silicon \
//   --example bench_small_lm -- \
//   gemma /tmp/rlx-weights/gemma-3-270m.gguf all
// ```

use anyhow::{Context, Result};
use rlx_runtime::{Device, is_available};
use std::path::PathBuf;
use std::time::Instant;

const PROMPT_IDS: &[u32] = &[2, 100, 101, 102, 103, 104, 105, 106];

fn n_new() -> usize {
    std::env::var("RLX_BENCH_N_NEW")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(16)
}

struct ResultRow {
    load_s: f32,
    prefill_tps: f32,
    decode_tps: f32,
}

fn bench_gemma(device: Device, weights: &str) -> Result<ResultRow> {
    use rlx_gemma::GemmaRunnerBuilder;
    let t_load = Instant::now();
    let mut runner = GemmaRunnerBuilder::default()
        .weights(weights)
        .device(device)
        .packed_weights(true)
        .build()?;
    let load_s = t_load.elapsed().as_secs_f32();

    let mut token_times = Vec::new();
    let t_gen = Instant::now();
    let n_prompt = PROMPT_IDS.len();
    let n_new_v = n_new();
    runner.generate(PROMPT_IDS, n_new_v, |_| {
        token_times.push(t_gen.elapsed().as_secs_f32());
    })?;
    let (prefill_tps, decode_tps) = phase_tps(&token_times, n_prompt);
    Ok(ResultRow {
        load_s,
        prefill_tps,
        decode_tps,
    })
}

fn bench_phi(device: Device, weights: &str) -> Result<ResultRow> {
    use rlx_phi::PhiRunner;
    let t_load = Instant::now();
    let mut runner = PhiRunner::builder()
        .weights(weights)
        .device(device)
        .packed_weights(true)
        .max_seq(128)
        .build()?;
    let load_s = t_load.elapsed().as_secs_f32();

    let mut token_times = Vec::new();
    let t_gen = Instant::now();
    let n_prompt = PROMPT_IDS.len();
    let n_new_v = n_new();
    runner.generate_packed(PROMPT_IDS, n_new_v, |_| {
        token_times.push(t_gen.elapsed().as_secs_f32());
    })?;
    let (prefill_tps, decode_tps) = phase_tps(&token_times, n_prompt);
    Ok(ResultRow {
        load_s,
        prefill_tps,
        decode_tps,
    })
}

fn phase_tps(token_times: &[f32], n_prompt: usize) -> (f32, f32) {
    if token_times.is_empty() {
        return (0.0, 0.0);
    }
    let first = token_times[0];
    let prefill_tps = n_prompt as f32 / first.max(1e-6);
    let decode_count = token_times.len().saturating_sub(1);
    let decode_s = if token_times.len() > 1 {
        token_times.last().unwrap() - first
    } else {
        0.0
    };
    let decode_tps = decode_count as f32 / decode_s.max(1e-6);
    (prefill_tps, decode_tps)
}

fn try_bench(model: &str, device: Device, label: &str, weights: &str) -> Option<ResultRow> {
    if !is_available(device) {
        eprintln!("[{model}/{label}] not available — skip");
        return None;
    }
    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| match model {
        "gemma" => bench_gemma(device, weights),
        "phi" => bench_phi(device, weights),
        other => Err(anyhow::anyhow!("unknown model {other}")),
    }));
    match res {
        Ok(Ok(row)) => {
            println!(
                "[{model}/{label}] load={:.2}s prefill={:.1} t/s decode={:.1} t/s",
                row.load_s, row.prefill_tps, row.decode_tps
            );
            Some(row)
        }
        Ok(Err(e)) => {
            eprintln!("[{model}/{label}] FAILED: {e:#}");
            None
        }
        Err(p) => {
            let msg = p
                .downcast_ref::<String>()
                .map(|s| s.as_str())
                .or_else(|| p.downcast_ref::<&str>().copied())
                .unwrap_or("(panic)");
            eprintln!("[{model}/{label}] PANIC: {msg}");
            None
        }
    }
}

fn main() -> Result<()> {
    let model = std::env::args()
        .nth(1)
        .context("usage: bench_small_lm <gemma|phi> <gguf> [cpu|metal|mlx|coreml|ane|all]")?;
    let weights: PathBuf = std::env::args()
        .nth(2)
        .map(PathBuf::from)
        .context("missing gguf path")?;
    let weights_str = weights.to_str().context("non-utf8 path")?.to_string();
    let only = std::env::args()
        .nth(3)
        .unwrap_or_else(|| "all".into())
        .to_ascii_lowercase();

    println!(
        "→ bench_small_lm model={model} weights={weights_str} n_new={}",
        n_new()
    );
    println!("   prompt_ids={PROMPT_IDS:?}\n");

    let backends: &[(&str, Device)] = &[
        ("cpu", Device::Cpu),
        ("metal", Device::Metal),
        ("mlx", Device::Mlx),
        ("coreml", Device::Ane),
    ];
    for (name, dev) in backends {
        if only != "all" && only != *name && !(only == "ane" && *name == "coreml") {
            continue;
        }
        try_bench(&model, *dev, name, &weights_str);
    }
    Ok(())
}
