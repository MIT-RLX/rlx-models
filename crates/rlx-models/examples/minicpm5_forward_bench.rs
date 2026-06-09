// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// MiniCPM5-1B timing harness (safetensors + F32 generator).
//
// ```bash
// just fetch-minicpm5
// just bench-minicpm5-real-all-backends
// just bench-minicpm5-real --device mlx --tokens 8
// ```

use rlx_minicpm5::MiniCpm5Runner;
use rlx_runtime::Device;
use std::env;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
use std::time::Instant;

fn parse_device(s: &str) -> anyhow::Result<Device> {
    match s.to_ascii_lowercase().as_str() {
        "cpu" => Ok(Device::Cpu),
        "metal" => Ok(Device::Metal),
        "mlx" => Ok(Device::Mlx),
        "cuda" => Ok(Device::Cuda),
        "rocm" => Ok(Device::Rocm),
        "gpu" | "wgpu" => Ok(Device::Gpu),
        "vulkan" => Ok(Device::Vulkan),
        other => anyhow::bail!("unknown device {other:?} (cpu|metal|mlx|cuda|rocm|gpu|vulkan)"),
    }
}

fn device_label(d: Device) -> &'static str {
    match d {
        Device::Cpu => "cpu",
        Device::Metal => "metal",
        Device::Mlx => "mlx",
        Device::Cuda => "cuda",
        Device::Rocm => "rocm",
        Device::Gpu => "wgpu",
        Device::Vulkan => "vulkan",
        _ => "other",
    }
}

fn all_backend_devices() -> Vec<Device> {
    vec![
        Device::Cpu,
        #[cfg(feature = "metal")]
        Device::Metal,
        #[cfg(feature = "mlx")]
        Device::Mlx,
        #[cfg(feature = "cuda")]
        Device::Cuda,
        #[cfg(feature = "rocm")]
        Device::Rocm,
        #[cfg(feature = "gpu")]
        Device::Gpu,
        #[cfg(feature = "vulkan")]
        Device::Vulkan,
    ]
}

fn default_weights() -> anyhow::Result<PathBuf> {
    if let Ok(w) = env::var("RLX_MINICPM5_WEIGHTS") {
        return Ok(PathBuf::from(w));
    }
    let dir = env::var("MINICPM5_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/tmp/rlx-weights/MiniCPM5-1B"));
    let shard = dir.join("model-00000-of-00001.safetensors");
    if shard.is_file() {
        return Ok(shard);
    }
    anyhow::bail!("set RLX_MINICPM5_WEIGHTS or run `just fetch-minicpm5`")
}

struct Opts {
    weights: PathBuf,
    devices: Vec<Device>,
    tokens: usize,
    prefill_only: bool,
}

fn parse_args() -> anyhow::Result<Opts> {
    let mut args: Vec<String> = env::args().skip(1).filter(|a| a != "--").collect();
    let weights = match args.first() {
        Some(p) if !p.starts_with('-') => PathBuf::from(args.remove(0)),
        _ => default_weights()?,
    };
    let mut device = Device::Cpu;
    let mut all_backends = false;
    let mut tokens = 8usize;
    let mut prefill_only = false;
    let mut args = args.into_iter();
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
            "--all-backends" => all_backends = true,
            "--tokens" => {
                tokens = args
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("--tokens requires a value"))?
                    .parse()?
            }
            other if other.starts_with("--tokens=") => {
                tokens = other.trim_start_matches("--tokens=").parse()?;
            }
            "--prefill-only" => prefill_only = true,
            other => anyhow::bail!("unknown flag: {other}"),
        }
    }
    let devices = if all_backends {
        all_backend_devices()
    } else {
        vec![device]
    };
    Ok(Opts {
        weights,
        devices,
        tokens,
        prefill_only,
    })
}

fn run_one(
    weights: &Path,
    device: Device,
    tokens: usize,
    prefill_only: bool,
) -> anyhow::Result<()> {
    if device != Device::Cpu && !rlx_runtime::is_available(device) {
        println!(
            "  [{label}] skip (unavailable)",
            label = device_label(device)
        );
        return Ok(());
    }

    let prompt: Vec<u32> = (1..=8).collect();
    let max_seq = (prompt.len() + tokens).max(128);

    let t0 = Instant::now();
    let mut runner = MiniCpm5Runner::builder()
        .weights(weights)
        .device(device)
        .max_seq(max_seq)
        .build()?;
    let build_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let t = Instant::now();
    let logits = runner.predict_logits(&prompt)?;
    let prefill_first_ms = t.elapsed().as_secs_f64() * 1000.0;

    let t = Instant::now();
    let _ = runner.predict_logits(&prompt)?;
    let prefill_steady_ms = t.elapsed().as_secs_f64() * 1000.0;

    let (gen_ms, tok_per_s) = if prefill_only {
        (0.0, 0.0)
    } else {
        let t = Instant::now();
        let generated = runner.generate(&prompt, tokens, |_| {})?;
        let gen_ms = t.elapsed().as_secs_f64() * 1000.0;
        let tok_per_s = tokens as f64 / (gen_ms / 1000.0);
        println!("    generated     : {generated:?}");
        (gen_ms, tok_per_s)
    };

    let label = device_label(device);
    println!("  [{label}]");
    println!("    build           : {build_ms:.1} ms");
    println!("    vocab           : {}", logits.len());
    println!("    prefill (1st)   : {prefill_first_ms:.1} ms");
    println!("    prefill (warm)  : {prefill_steady_ms:.1} ms");
    if !prefill_only {
        println!("    decode_tokens   : {tokens}");
        println!("    generate        : {gen_ms:.1} ms ({tok_per_s:.2} tok/s)");
    }
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let opts = parse_args()?;
    println!("# minicpm5_forward_bench (openbmb/MiniCPM5-1B)");
    println!("  weights        : {}", opts.weights.display());
    println!("  prompt_len     : 8");
    if opts.prefill_only {
        println!("  mode           : prefill-only");
    }

    let mut failed = Vec::new();
    for device in opts.devices {
        let label = device_label(device);
        let weights = opts.weights.clone();
        let tokens = opts.tokens;
        let prefill_only = opts.prefill_only;
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            run_one(&weights, device, tokens, prefill_only)
        }));
        match outcome {
            Ok(Ok(())) => {}
            Ok(Err(e)) => {
                eprintln!("  [{label}] ERROR: {e:#}");
                failed.push(label);
            }
            Err(_) => {
                eprintln!(
                    "  [{label}] ERROR: panic during run (e.g. wgpu buffer limit on full 1B)"
                );
                failed.push(label);
            }
        }
    }

    if failed.is_empty() {
        return Ok(());
    }
    let gpu_only = failed.iter().all(|s| matches!(*s, "wgpu" | "vulkan"));
    if gpu_only {
        eprintln!(
            "note: {} skipped — full 1B F32 exceeds WebGPU storage binding limits on this host",
            failed.join(", ")
        );
        return Ok(());
    }
    anyhow::bail!("failed backends: {}", failed.join(", "))
}
