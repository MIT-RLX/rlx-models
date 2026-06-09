// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! LocateAnything-3B timing across RLX backends (ground-single, bundled sample).
//!
//! ```bash
//! just bench-locateanything-backends
//! cargo run -p rlx-models --example locateanything_bench --release --features all-backends -- --all-backends
//! ```
//!
//! Multi-backend runs use a subprocess per device so GPU/WGPU arenas and mmap caches
//! are released between backends (avoids OOM after Metal/MLX on unified memory).

use anyhow::{Context, Result};
use rlx_locateanything::{
    GenerateProfile, GenerationMode, InferenceOptions, LocateAnythingSession, PromptStyle,
    default_model_dir, fixtures,
};
use rlx_runtime::{Device, is_available};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::process::{Command, Stdio};
use std::time::Instant;

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

#[allow(clippy::vec_init_then_push)]
fn all_backend_devices() -> Vec<Device> {
    // GPU backends first — full 3B on CPU can take many minutes.
    let mut out = Vec::new();
    #[cfg(feature = "cuda")]
    out.push(Device::Cuda);
    #[cfg(feature = "metal")]
    out.push(Device::Metal);
    #[cfg(feature = "mlx")]
    out.push(Device::Mlx);
    #[cfg(feature = "rocm")]
    out.push(Device::Rocm);
    #[cfg(feature = "gpu")]
    out.push(Device::Gpu);
    #[cfg(feature = "vulkan")]
    out.push(Device::Vulkan);
    out.push(Device::Cpu);
    out
}

struct Opts {
    devices: Vec<Device>,
    max_image_side: u32,
    max_tokens: usize,
    phrase: String,
    /// Re-exec one process per backend (default when more than one device is requested).
    isolate_backends: bool,
}

fn parse_args() -> Result<Opts> {
    let mut devices = Vec::new();
    let mut all_backends = false;
    let mut isolate_backends = false;
    let mut no_isolate = false;
    let mut max_image_side = 480u32;
    let mut max_tokens = 32usize;
    let mut phrase = "person".to_string();
    let mut args = std::env::args().skip(1).peekable();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--all-backends" => all_backends = true,
            "--isolate-backends" => isolate_backends = true,
            "--no-isolate" => no_isolate = true,
            "--device" => {
                let s = args
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("--device requires a value"))?;
                devices.push(rlx_locateanything::resolve_device(Some(&s))?);
            }
            "--max-image-side" => {
                max_image_side = args
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("--max-image-side requires a value"))?
                    .parse()?;
            }
            "--max-tokens" => {
                max_tokens = args
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("--max-tokens requires a value"))?
                    .parse()?;
            }
            "--phrase" => {
                phrase = args
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("--phrase requires a value"))?;
            }
            "--help" | "-h" => {
                eprintln!(
                    "locateanything_bench — [--all-backends] [--device NAME] [--isolate-backends]\n\
                     [--no-isolate] [--max-image-side N] [--max-tokens N] [--phrase TEXT]"
                );
                std::process::exit(0);
            }
            other => anyhow::bail!("unknown flag: {other}"),
        }
    }
    let devices = if all_backends || devices.is_empty() {
        all_backend_devices()
    } else {
        devices
    };
    let isolate_backends = if no_isolate {
        false
    } else if isolate_backends {
        true
    } else {
        devices.len() > 1
    };
    Ok(Opts {
        devices,
        max_image_side,
        max_tokens,
        phrase,
        isolate_backends,
    })
}

struct Timings {
    label: &'static str,
    open_ms: f64,
    preprocess_ms: f64,
    warmup_ms: f64,
    ground1_ms: f64,
    ground2_ms: f64,
    new_tokens: usize,
    warm_profile: GenerateProfile,
}

fn print_generate_profile(p: &GenerateProfile) {
    println!(
        "    profile (warm)  : vision={:.0} ms (cache {})  prefill={:.0} ms (cache {})  decode={:.0} ms  fuse={:.0} ms  gpu_kv={}",
        p.vision_ms,
        if p.vision_cache_hit { "hit" } else { "miss" },
        p.prefill_ms,
        if p.prefill_cache_hit { "hit" } else { "miss" },
        p.decode_mtp_ms,
        p.fuse_embed_ms,
        if p.gpu_kv_resident { "yes" } else { "no" },
    );
}

fn run_one(opts: &Opts, device: Device) -> Result<Option<Timings>> {
    let label = device_label(device);
    if device != Device::Cpu && !is_available(device) {
        println!("  [{label}] skip (backend not available in this binary)");
        return Ok(None);
    }

    let session_opts = InferenceOptions::for_grounding()
        .device(device)
        .max_image_side(opts.max_image_side)
        .max_new_tokens(opts.max_tokens)
        .generation_mode(GenerationMode::Fast)
        .prompt_style(PromptStyle::Processor);

    let t0 = Instant::now();
    let mut session = LocateAnythingSession::open_with_options(default_model_dir()?, session_opts)?;
    let open_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let image = fixtures::sample_image_path();
    let t0 = Instant::now();
    let prep = session.preprocess_file(&image)?;
    let preprocess_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let t0 = Instant::now();
    session.warmup(&prep, &opts.phrase)?;
    let warmup_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let t0 = Instant::now();
    let out1 = session.ground(&prep, &opts.phrase)?;
    let ground1_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let t0 = Instant::now();
    let (out2, warm_profile) = session.ground_with_profile(&prep, &opts.phrase)?;
    let ground2_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let e2e_ms = open_ms + preprocess_ms + warmup_ms + ground1_ms;
    let tok_per_s = out2.new_tokens as f64 / (ground2_ms / 1000.0);

    println!("  [{label}]");
    println!("    open            : {open_ms:.0} ms");
    println!("    preprocess      : {preprocess_ms:.0} ms");
    println!("    warmup (compile): {warmup_ms:.0} ms");
    println!(
        "    ground (1st)    : {ground1_ms:.0} ms  ({} new tokens)",
        out1.new_tokens
    );
    println!(
        "    ground (warm)   : {ground2_ms:.0} ms  ({} new tokens, {tok_per_s:.2} tok/s)",
        out2.new_tokens
    );
    print_generate_profile(&warm_profile);
    println!("    e2e cold        : {e2e_ms:.0} ms  (open→first ground)");

    Ok(Some(Timings {
        label,
        open_ms,
        preprocess_ms,
        warmup_ms,
        ground1_ms,
        ground2_ms,
        new_tokens: out2.new_tokens,
        warm_profile,
    }))
}

fn run_one_isolated(opts: &Opts, device: Device) -> Result<()> {
    let label = device_label(device);
    let exe = std::env::current_exe().context("current_exe")?;
    let model_dir = default_model_dir()?;
    let mut cmd = Command::new(&exe);
    cmd.arg("--no-isolate")
        .arg("--device")
        .arg(label)
        .arg("--max-image-side")
        .arg(opts.max_image_side.to_string())
        .arg("--max-tokens")
        .arg(opts.max_tokens.to_string())
        .arg("--phrase")
        .arg(&opts.phrase)
        .env("RLX_LOCATEANYTHING_DIR", model_dir)
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit());
    let status = cmd
        .status()
        .with_context(|| format!("spawn bench subprocess for {label}"))?;
    if !status.success() {
        anyhow::bail!("subprocess for {label} exited with {status}");
    }
    Ok(())
}

fn main() -> Result<()> {
    let opts = parse_args()?;

    if opts.isolate_backends {
        println!("# locateanything_bench (nvidia/LocateAnything-3B) — one subprocess per backend");
        println!(
            "  image          : {}",
            fixtures::sample_image_path().display()
        );
        println!("  phrase         : {}", opts.phrase);
        println!("  max_image_side : {}", opts.max_image_side);
        println!("  max_tokens     : {}", opts.max_tokens);
        println!("  generation     : fast");
        println!();

        let mut failed = Vec::new();
        for device in &opts.devices {
            let label = device_label(*device).to_string();
            if let Err(e) = run_one_isolated(&opts, *device) {
                eprintln!("  [{label}] ERROR: {e:#}");
                failed.push(label);
            }
        }
        if !failed.is_empty() {
            eprintln!("\nfailed backends: {}", failed.join(", "));
            anyhow::bail!("one or more isolated backend runs failed");
        }
        return Ok(());
    }

    println!("# locateanything_bench (nvidia/LocateAnything-3B)");
    println!(
        "  image          : {}",
        fixtures::sample_image_path().display()
    );
    println!("  phrase         : {}", opts.phrase);
    println!("  max_image_side : {}", opts.max_image_side);
    println!("  max_tokens     : {}", opts.max_tokens);
    println!("  generation     : fast");
    println!();

    let mut results = Vec::new();
    let mut failed = Vec::new();

    for device in opts.devices {
        let label = device_label(device).to_string();
        let bench_opts = Opts {
            devices: vec![device],
            max_image_side: opts.max_image_side,
            max_tokens: opts.max_tokens,
            phrase: opts.phrase.clone(),
            isolate_backends: false,
        };
        let capture = label.clone();
        let outcome = catch_unwind(AssertUnwindSafe(|| run_one(&bench_opts, device)));
        match outcome {
            Ok(Ok(Some(t))) => results.push(t),
            Ok(Ok(None)) => {}
            Ok(Err(e)) => {
                eprintln!("  [{capture}] ERROR: {e:#}");
                failed.push(capture);
            }
            Err(_) => {
                eprintln!("  [{capture}] ERROR: panic during run");
                failed.push(capture);
            }
        }
    }

    if !results.is_empty() {
        println!("\n# ranking (steady-state ground, lower is faster)");
        results.sort_by(|a, b| {
            a.ground2_ms
                .partial_cmp(&b.ground2_ms)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for (i, t) in results.iter().enumerate() {
            let e2e = t.open_ms + t.preprocess_ms + t.warmup_ms + t.ground1_ms;
            let p = &t.warm_profile;
            println!(
                "  {}. {:6}  warm_ground={:7.0} ms  e2e_cold={:7.0} ms  warmup={:7.0} ms  prefill_cache={}",
                i + 1,
                t.label,
                t.ground2_ms,
                e2e,
                t.warmup_ms,
                if p.prefill_cache_hit { "hit" } else { "miss" },
            );
        }
        let fastest = &results[0];
        println!(
            "\nfastest: {} ({:.0} ms warm ground, {:.2} tok/s)",
            fastest.label,
            fastest.ground2_ms,
            fastest.new_tokens as f64 / (fastest.ground2_ms / 1000.0)
        );
    }

    if !failed.is_empty() {
        eprintln!("\nfailed backends: {}", failed.join(", "));
    }

    Ok(())
}
