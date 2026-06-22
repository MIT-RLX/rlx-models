//! Warm steady-state CoreML benchmark (cached session) vs host.
//!
//! cargo run -p rlx-inflect-nano --features coreml --release --example bench_coreml -- \
//!     --data weights/inflect-nano-rlx
#![cfg(feature = "onnx")]

use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;
use rlx_inflect_nano::{InferOpts, InflectNano};

const TEXT: &str = "The weather is nice today, and I feel very relaxed. \
In nineteen ninety nine we found a thousand reasons to keep going, and then a few more.";

fn arg(name: &str, default: &str) -> String {
    let a: Vec<String> = std::env::args().collect();
    a.iter()
        .position(|x| x == name)
        .and_then(|i| a.get(i + 1))
        .cloned()
        .unwrap_or_else(|| default.to_string())
}

fn time_n(
    label: &str,
    audio_secs: f32,
    iters: usize,
    mut f: impl FnMut() -> Result<()>,
) -> Result<()> {
    for _ in 0..iters {
        let t = Instant::now();
        f()?;
        let _ = t;
    }
    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        f()?;
        times.push(t.elapsed().as_secs_f32());
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = times[times.len() / 2];
    println!(
        "{label:<22} median {:>7.1} ms  RTF {:>6.1}x",
        med * 1e3,
        audio_secs / med
    );
    Ok(())
}

fn main() -> Result<()> {
    let data = arg("--data", "weights/inflect-nano-rlx");
    let iters: usize = arg("--iters", "8").parse().unwrap_or(8);
    let model = InflectNano::load_from_dir(&PathBuf::from(&data))?;
    let opts = InferOpts::default();

    let audio = model.synthesize(TEXT, &opts)?;
    let audio_secs = audio.samples.len() as f32 / audio.sample_rate as f32;
    println!("audio = {audio_secs:.2}s, {iters} warm iters\n");

    // First CoreML call pays the one-time model compile; warm-up loop absorbs it.
    let t0 = Instant::now();
    let _ = model.synthesize_coreml(TEXT, &opts)?;
    println!(
        "CoreML first-call (compile+run): {:.0} ms\n",
        t0.elapsed().as_secs_f32() * 1e3
    );

    time_n("host CPU (f32)", audio_secs, iters, || {
        model.synthesize(TEXT, &opts).map(|_| ())
    })?;
    time_n("CoreML (cached, warm)", audio_secs, iters, || {
        model.synthesize_coreml(TEXT, &opts).map(|_| ())
    })?;
    Ok(())
}
