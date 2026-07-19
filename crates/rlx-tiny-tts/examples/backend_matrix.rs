//! TinyTTS / MeloTTS cross-backend parity (cosine vs CPU) + audibility.
//!
//! ```bash
//! cargo run -p rlx-tiny-tts --release --example backend_matrix --features apple-silicon \
//!   -- weights/tts/melotts
//! RLX_DEVICES=cpu,gpu,vulkan cargo run -p rlx-tiny-tts --release --example backend_matrix \
//!   --features all-backends -- weights/tiny-tts-rlx
//! ```
//!
//! Pass when peak ≥ 1e-3, samples ≥ 1600, and cosine vs CPU ≥ 0.95 (env
//! `RLX_COS_MIN` overrides).

use std::path::PathBuf;
use std::time::Instant;

use rlx_runtime::{Device, is_available};
use rlx_tiny_tts::{InferOpts, KernelVariant, TinyTts, peak_amplitude};

const DEFAULT_TEXT: &str = "The quick brown fox jumps over the lazy dog near the river bank.";

fn main() -> anyhow::Result<()> {
    let dir = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("weights/tts/melotts"));
    let text = std::env::var("RLX_TEXT").unwrap_or_else(|_| DEFAULT_TEXT.into());
    let cos_min: f32 = std::env::var("RLX_COS_MIN")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.95);
    let devices = parse_devices();

    let model = TinyTts::load(&dir)?;
    let mut opts = InferOpts::from_config(model.config());
    opts.seed = 1234;
    opts.kernel = KernelVariant::Precise;

    println!("== TinyTTS/MeloTTS cross-backend ==");
    println!("dir={}  text={text:?}", dir.display());
    println!(
        "{:<8} {:>8} {:>7} {:>8} {:>8}",
        "backend", "ms", "peak", "samples", "cos"
    );

    let mut cpu_wav: Option<Vec<f32>> = None;
    let mut failed = false;

    for (dev, label) in &devices {
        if *dev != Device::Cpu && !is_available(*dev) {
            println!(
                "{label:<8} {:>8} {:>7} {:>8} {:>8}  (n/a)",
                "-", "-", "-", "-"
            );
            continue;
        }
        let t0 = Instant::now();
        let wav = match model.synthesize_on(&text, *dev, &opts) {
            Ok(w) => w,
            Err(e) => {
                println!("{label:<8}  synth err: {e:#}");
                failed = true;
                continue;
            }
        };
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        let peak = peak_amplitude(&wav.samples);
        let cos = cpu_wav
            .as_ref()
            .map(|r| cosine(r, &wav.samples))
            .unwrap_or(1.0);
        if cpu_wav.is_none() {
            cpu_wav = Some(wav.samples.clone());
        }
        println!(
            "{label:<8} {ms:>8.1} {peak:>7.3} {:>8} {cos:>8.5}",
            wav.samples.len()
        );
        if peak < 1e-3 || wav.samples.len() < 1600 {
            failed = true;
        }
        if cos < cos_min {
            failed = true;
        }
    }

    if failed {
        anyhow::bail!("TinyTTS backend matrix failed (need peak≥1e-3, n≥1600, cos≥{cos_min})");
    }
    Ok(())
}

fn parse_devices() -> Vec<(Device, &'static str)> {
    if let Ok(raw) = std::env::var("RLX_DEVICES") {
        return raw
            .split(',')
            .filter_map(|s| {
                let s = s.trim();
                let d = match s.to_ascii_lowercase().as_str() {
                    "cpu" => Device::Cpu,
                    "metal" => Device::Metal,
                    "mlx" => Device::Mlx,
                    "gpu" | "wgpu" => Device::Gpu,
                    "cuda" => Device::Cuda,
                    "vulkan" => Device::Vulkan,
                    "ane" | "coreml" => Device::Ane,
                    _ => return None,
                };
                Some((d, Box::leak(s.to_string().into_boxed_str()) as &str))
            })
            .collect();
    }
    vec![
        (Device::Cpu, "cpu"),
        (Device::Metal, "metal"),
        (Device::Mlx, "mlx"),
        (Device::Gpu, "gpu"),
        (Device::Cuda, "cuda"),
        (Device::Vulkan, "vulkan"),
        (Device::Ane, "CoreML"),
    ]
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for i in 0..n {
        let x = a[i] as f64;
        let y = b[i] as f64;
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    (dot / (na.sqrt() * nb.sqrt())) as f32
}
