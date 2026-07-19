//! Gepard timing bench — prefill, AR decode, NanoCodec on selected backend.
//!
//! ```bash
//! cargo run -p rlx-gepard --release --example bench_timing --features apple-silicon -- \
//!   --device metal --text "The quick brown fox jumps over the lazy dog."
//! ```

use std::path::PathBuf;
use std::time::Instant;

use rlx_gepard::{GepardSynthesizer, InferOpts, default_seed_for_text};
use rlx_runtime::{Device, is_available};

fn main() -> anyhow::Result<()> {
    let dir = std::env::var("RLX_GEPARD_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("weights/tts/gepard"));
    anyhow::ensure!(
        dir.join("model.safetensors").is_file(),
        "missing Gepard weights"
    );

    let text = std::env::var("RLX_TEXT")
        .unwrap_or_else(|_| "The quick brown fox jumps over the lazy dog.".to_string());
    let device = std::env::args()
        .skip_while(|a| a != "--device")
        .nth(1)
        .unwrap_or_else(|| "metal".to_string());
    let dev = parse_device(&device);
    if dev != Device::Cpu && !is_available(dev) {
        anyhow::bail!("backend {device} not available on this host");
    }

    let t0 = Instant::now();
    let synth = GepardSynthesizer::open(&dir, dev)?.with_opts(InferOpts {
        seed: default_seed_for_text(&text),
        ..Default::default()
    });
    let pcm = synth.synthesize(&text, "")?;
    let total_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let dur_s = pcm.len() as f64 / 22_050.0;
    let rtf = if dur_s > 0.0 {
        total_ms / 1000.0 / dur_s
    } else {
        0.0
    };

    println!("== Gepard timing ==");
    println!("device: {device}");
    println!("text:   {text:?}");
    println!("samples: {}", pcm.len());
    println!("duration: {dur_s:.2}s @ 22050 Hz");
    println!("total: {:.0} ms", total_ms);
    println!("RTF: {rtf:.3}");
    Ok(())
}

fn parse_device(s: &str) -> Device {
    match s.trim().to_ascii_lowercase().as_str() {
        "metal" => Device::Metal,
        "mlx" => Device::Mlx,
        "cuda" => Device::Cuda,
        "rocm" => Device::Rocm,
        "gpu" | "wgpu" => Device::Gpu,
        "vulkan" => Device::Vulkan,
        _ => Device::Cpu,
    }
}
