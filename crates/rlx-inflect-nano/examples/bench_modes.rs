//! Benchmark each ExecutionMode (steady-state, warmed).
//!
//! cargo run -p rlx-inflect-nano --features gpu --release --example bench_modes -- \
//!     --data weights/inflect-nano-rlx
//!
//! Modes that target a GPU (Latency/Hybrid) only differ from the host modes when
//! a GPU backend is compiled in (e.g. --features metal|mlx|gpu).

use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;
use rlx_inflect_nano::{ExecutionMode, InferOpts, InflectNano};

const TEXT: &str = "The weather is nice today, and I feel very relaxed. \
Doctor Smith paid forty two dollars at a quarter past three. \
In nineteen ninety nine we found a thousand reasons to keep going, and then a few more.";

fn arg(name: &str, default: &str) -> String {
    let a: Vec<String> = std::env::args().collect();
    a.iter()
        .position(|x| x == name)
        .and_then(|i| a.get(i + 1))
        .cloned()
        .unwrap_or_else(|| default.to_string())
}

fn main() -> Result<()> {
    let data = arg("--data", "weights/inflect-nano-rlx");
    let iters: usize = arg("--iters", "8").parse().unwrap_or(8);
    let model = InflectNano::load_from_dir(&PathBuf::from(&data))?;
    let opts = InferOpts::default();

    let modes = [
        ("precision", ExecutionMode::Precision),
        ("memory", ExecutionMode::MemoryFootprint),
        ("latency", ExecutionMode::Latency),
        ("hybrid", ExecutionMode::Hybrid),
    ];

    #[cfg(feature = "rlx-graph")]
    println!(
        "preferred accelerator: {:?}",
        InflectNano::preferred_accelerator()
    );
    let audio_secs = {
        let w = model.synthesize_mode(TEXT, &opts, ExecutionMode::Precision)?;
        w.samples.len() as f32 / w.sample_rate as f32
    };
    println!("audio = {audio_secs:.2}s, {iters} timed iters/mode\n");
    println!(
        "{:<12} {:>10} {:>10} {:>9}",
        "mode", "median_ms", "best_ms", "RTF"
    );

    for (name, mode) in modes {
        // warm up (pays first-time graph compile / AOT cache fill)
        for _ in 0..2 {
            let _ = model.synthesize_mode(TEXT, &opts, mode)?;
        }
        let mut times = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            let _ = model.synthesize_mode(TEXT, &opts, mode)?;
            times.push(t0.elapsed().as_secs_f32());
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = times[times.len() / 2];
        let best = times[0];
        println!(
            "{name:<12} {:>10.1} {:>10.1} {:>8.1}x",
            median * 1e3,
            best * 1e3,
            audio_secs / median
        );
    }
    Ok(())
}
