// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Released under GPL-3.0; see crate-level header for the full notice.

//! Timing + parity harness for the SAM3 ViT-L vision trunk.
//!
//! Loads a SAM3 checkpoint (safetensors or GGUF), extracts the vision encoder
//! weights, then for each requested backend:
//!   1. Builds the device-side encoder once.
//!   2. Runs a warm-up pass (this absorbs HIR build + device pipeline compile).
//!   3. Reports the best of `RLX_SAM3_VISION_ITERS` steady-state `run_tokens`
//!      calls — i.e. graph reuse is amortized, so the number is inference
//!      only (no rebuild per iter).
//!   4. Compares its output against the host reference and reports
//!      max-abs / mean-abs / cosine.
//!
//! Skips neck / detector / segmentation extraction so the harness still works
//! when those layers' weights use a different layout convention.
//!
//! Usage:
//!   cargo run --release -p rlx-sam3 --example vision_timing -- <weights>
//!
//! Env:
//!   RLX_SAM3_VISION_ITERS   number of timed steady-state iterations (default 3)
//!   RLX_SAM3_VISION_DEVICES csv of host,cpu,metal,mlx,cuda (default host,metal)

use anyhow::{Context, Result};
use rlx_flow::CompileProfile;
use rlx_runtime::Device;
use rlx_sam3::config::Sam3Config;
use rlx_sam3::preprocess::preprocess_image;
use rlx_sam3::sam3_profile_near_weights;
use rlx_sam3::vision_encoder::{
    Sam3VisionEncoderWeights, encode_image_native, extract_vision_encoder_weights,
};
use rlx_sam3::vision_encoder_ir::{Sam3CompiledVisionEncoder, host_preroll};
use std::path::Path;
use std::time::Instant;

#[derive(Debug)]
struct Row {
    label: String,
    warm: f64,
    best: f64,
    max_abs: Option<f32>,
    cos: Option<f32>,
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let weights_path = args
        .next()
        .context("usage: vision_timing <weights.gguf|weights.safetensors>")?;

    let iters: usize = std::env::var("RLX_SAM3_VISION_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);

    let devices_csv =
        std::env::var("RLX_SAM3_VISION_DEVICES").unwrap_or_else(|_| "host,metal".into());
    let devices: Vec<String> = devices_csv
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();

    let cfg = Sam3Config::base();

    let path = Path::new(&weights_path);
    let is_gguf = path.extension().is_some_and(|e| e == "gguf");
    if is_gguf {
        rlx_core::gguf_validate_arch(path, rlx_core::SAM3_GGUF_ARCHES)?;
    }
    eprintln!("[load] reading checkpoint from {}", weights_path);
    let t0 = Instant::now();

    // Any SAM3 GGUF that ships a `*.tensor_map.json` sidecar uses the SAM3
    // loader, even when no tensor is quantized — the sidecar carries the
    // opaque-key → friendly-name remap the generic loader doesn't know about.
    let has_sidecar = is_gguf && {
        let mut s = path.as_os_str().to_owned();
        s.push(".tensor_map.json");
        std::path::Path::new(&s).exists()
    };
    let (mut wm, gguf_packed) =
        if is_gguf && (rlx_sam3::gguf_has_packed_linears(path)? || has_sidecar) {
            let (wm, packed) = rlx_sam3::load_sam3_from_gguf(path)?;
            (wm, Some(packed))
        } else {
            (
                rlx_core::load_weight_map(path, rlx_core::SAM3_GGUF_ARCHES)?,
                None,
            )
        };
    eprintln!("[load] checkpoint mmap+parse: {:?}", t0.elapsed());

    let t1 = Instant::now();
    let vision: Sam3VisionEncoderWeights =
        extract_vision_encoder_weights(&mut wm, &cfg.vit, gguf_packed.as_ref())
            .context("extract_vision_encoder_weights")?;
    eprintln!(
        "[load] vision encoder weights extracted: {:?}",
        t1.elapsed()
    );

    let profile = sam3_profile_near_weights(path);

    // 1008×1008 mid-gray image — matches the SAM3 base canvas exactly so no
    // resize happens inside `preprocess_image`.
    let side = cfg.vit.img_size;
    let rgb = vec![128u8; side * side * 3];
    let (image_nchw, _resized_hw) = preprocess_image(&rgb, side, side);

    // Pre-compute the host post-`ln_pre` tokens once. Cheap and identical for
    // every backend, so we feed it directly into the compiled encoder's
    // `run_tokens` instead of timing `host_preroll` repeatedly.
    let tokens_in = host_preroll(&vision, &cfg.vit, &image_nchw)?;

    // Compute the host reference output once, for parity comparison.
    eprintln!("[ref] computing host reference output (single forward) …");
    let t_ref = Instant::now();
    let host_ref = encode_image_native(&vision, gguf_packed.as_ref(), &cfg.vit, &image_nchw)?;
    eprintln!("[ref] host reference: {:?}", t_ref.elapsed());
    let ref_tokens = host_ref.tokens.clone();

    let mut rows: Vec<Row> = Vec::new();
    for dev_name in &devices {
        let label = format!("vision[{dev_name}]");
        let result = match dev_name.as_str() {
            "host" => run_host(&vision, gguf_packed.as_ref(), &cfg, &image_nchw, iters),
            other => match parse_device(other) {
                Some(dev) => run_ir(
                    &vision,
                    gguf_packed.as_ref(),
                    &cfg,
                    &tokens_in,
                    dev,
                    &profile,
                    iters,
                ),
                None => {
                    eprintln!("[skip] unknown device '{other}'");
                    continue;
                }
            },
        };
        match result {
            Ok((warm, best, out)) => {
                let (max_abs, cos) = parity_stats(&ref_tokens, &out);
                rows.push(Row {
                    label,
                    warm,
                    best,
                    max_abs: Some(max_abs),
                    cos: Some(cos),
                });
            }
            Err(e) => eprintln!("[fail] {label}: {e:#}"),
        }
    }

    println!();
    println!(
        "{:<20} {:>12} {:>12} {:>14} {:>12}",
        "backend", "warm-up s", "best s", "max-abs Δ", "cosine"
    );
    println!(
        "{:-<20} {:->12} {:->12} {:->14} {:->12}",
        "", "", "", "", ""
    );
    for r in &rows {
        let max_abs = r.max_abs.map(|v| format!("{v:.3e}")).unwrap_or("?".into());
        let cos = r.cos.map(|v| format!("{v:.6}")).unwrap_or("?".into());
        println!(
            "{:<20} {:>12.3} {:>12.3} {:>14} {:>12}",
            r.label, r.warm, r.best, max_abs, cos
        );
    }
    Ok(())
}

fn parse_device(name: &str) -> Option<Device> {
    Some(match name {
        "cpu" => Device::Cpu,
        "metal" => Device::Metal,
        "mlx" => Device::Mlx,
        "cuda" => Device::Cuda,
        _ => return None,
    })
}

fn run_host(
    weights: &Sam3VisionEncoderWeights,
    gguf_packed: Option<&rlx_flow::GgufPackedParams>,
    cfg: &Sam3Config,
    image_nchw: &[f32],
    iters: usize,
) -> Result<(f64, f64, Vec<f32>)> {
    let t0 = Instant::now();
    let first = encode_image_native(weights, gguf_packed, &cfg.vit, image_nchw)?;
    let warm = t0.elapsed().as_secs_f64();
    let mut best = f64::INFINITY;
    let mut last = first.tokens.clone();
    for _ in 0..iters.max(1) {
        let t = Instant::now();
        let out = encode_image_native(weights, gguf_packed, &cfg.vit, image_nchw)?;
        let dt = t.elapsed().as_secs_f64();
        if dt < best {
            best = dt;
        }
        last = out.tokens;
    }
    Ok((warm, best, last))
}

#[allow(clippy::too_many_arguments)]
fn run_ir(
    weights: &Sam3VisionEncoderWeights,
    gguf_packed: Option<&rlx_flow::GgufPackedParams>,
    cfg: &Sam3Config,
    tokens_in: &[f32],
    device: Device,
    profile: &CompileProfile,
    iters: usize,
) -> Result<(f64, f64, Vec<f32>)> {
    // Build the compiled encoder ONCE — HIR construction + device pipeline
    // compile both happen here. Subsequent `run_tokens` calls only execute
    // the graph; that's the steady-state cost we want to measure.
    let mut compiled = Sam3CompiledVisionEncoder::new_with_profile_and_gguf(
        weights,
        &cfg.vit,
        1,
        device,
        profile,
        gguf_packed,
    )?;
    let t0 = Instant::now();
    let first = compiled.run_tokens(tokens_in)?;
    let warm = t0.elapsed().as_secs_f64();
    let mut best = f64::INFINITY;
    let mut last = first;
    for _ in 0..iters.max(1) {
        let t = Instant::now();
        let out = compiled.run_tokens(tokens_in)?;
        let dt = t.elapsed().as_secs_f64();
        if dt < best {
            best = dt;
        }
        last = out;
    }
    Ok((warm, best, last))
}

fn parity_stats(reference: &[f32], candidate: &[f32]) -> (f32, f32) {
    if reference.len() != candidate.len() {
        return (f32::INFINITY, 0.0);
    }
    let max_abs = reference
        .iter()
        .zip(candidate.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let dot: f32 = reference
        .iter()
        .zip(candidate.iter())
        .map(|(a, b)| a * b)
        .sum();
    let na: f32 = reference.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = candidate.iter().map(|x| x * x).sum::<f32>().sqrt();
    let cos = dot / (na * nb + 1e-12);
    (max_abs, cos)
}
