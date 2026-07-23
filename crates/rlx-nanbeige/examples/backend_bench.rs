// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//!
//! Nanbeige looped-Transformer backend bench (OOM-aware).
//!
//! Default is a **synthetic** looped graph sized per device so Metal/MLX/CUDA
//! get a heavier workload while wgpu/Vulkan stay under storage-buffer limits.
//! Pass `--weights DIR` to time the real 3B checkpoint (skipped on backends
//! that [`BackendPlan::allow_full_f32`] rejects).
//!
//! ```sh
//! cargo run -p rlx-nanbeige --example backend_bench --features all-backends --release
//! cargo run -p rlx-nanbeige --example backend_bench --features apple-silicon --release -- \
//!   --weights /tmp/rlx-weights/Nanbeige4.2-3B --device mlx
//! just bench-nanbeige-backends
//! ```

use anyhow::{Context, Result, bail};
use rlx_core::flow_bridge::compile_graph_with_profile;
use rlx_core::weight_map::WeightMap;
use rlx_flow::CompileProfile;
use rlx_llama32::{
    STANDARD_DEVICES, build_llama32_decode_graph_sized_ext, build_llama32_graph_sized_last_logits,
    validate_device,
};
use rlx_nanbeige::{
    BackendPlan, Llama32Config, NanbeigeRunner, approx_param_bytes_f32, kv_cache_bytes,
    nanbeige42_3b_preset, prepare_device, working_set_bytes,
};
use rlx_runtime::{CompiledGraph, Device};
use std::collections::HashMap;
use std::env;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::PathBuf;
use std::time::Instant;

#[derive(Clone, Copy)]
enum SynthTier {
    /// Fits wgpu / Vulkan storage binds.
    Portable,
    /// Heavier graph for Metal / MLX / CUDA / ROCm / CPU.
    Accelerator,
}

struct Opts {
    devices: Vec<Device>,
    weights: Option<PathBuf>,
    warm: usize,
    iters: usize,
    ab_bucket: bool,
}

fn parse_device(s: &str) -> Result<Device> {
    match s.to_ascii_lowercase().as_str() {
        "cpu" => Ok(Device::Cpu),
        "metal" => Ok(Device::Metal),
        "mlx" => Ok(Device::Mlx),
        "cuda" => Ok(Device::Cuda),
        "rocm" => Ok(Device::Rocm),
        "gpu" | "wgpu" => Ok(Device::Gpu),
        "vulkan" => Ok(Device::Vulkan),
        other => bail!("unknown device {other:?}"),
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

fn soft_budget_gb(device: Device) -> f32 {
    let _ = device;
    rlx_runtime::memory_estimate::soft_memory_budget_bytes()
        .map(|b| (b as f64 / (1024.0 * 1024.0 * 1024.0)) as f32)
        .unwrap_or(24.0)
        .min(28.0)
}

fn parse_args() -> Result<Opts> {
    let mut args = env::args().skip(1).filter(|a| a != "--").peekable();
    let mut device: Option<Device> = None;
    let mut all = true;
    let mut weights = None;
    let mut warm = 2usize;
    let mut iters = 5usize;
    let mut ab_bucket = true;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--device" => {
                device = Some(parse_device(
                    &args.next().ok_or_else(|| anyhow::anyhow!("--device needs value"))?,
                )?);
                all = false;
            }
            "--all-backends" => all = true,
            "--weights" => {
                weights = Some(PathBuf::from(
                    args.next()
                        .ok_or_else(|| anyhow::anyhow!("--weights needs path"))?,
                ));
            }
            "--warm" => {
                warm = args
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("--warm needs int"))?
                    .parse()?;
            }
            "--iters" => {
                iters = args
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("--iters needs int"))?
                    .parse()?;
            }
            "--no-ab-bucket" => ab_bucket = false,
            other => bail!("unknown flag {other}"),
        }
    }
    let devices = if all {
        STANDARD_DEVICES.to_vec()
    } else {
        vec![device.unwrap_or(Device::Cpu)]
    };
    Ok(Opts {
        devices,
        weights,
        warm,
        iters,
        ab_bucket,
    })
}

fn synth_cfg(tier: SynthTier) -> Llama32Config {
    let mut c = nanbeige42_3b_preset();
    c.num_loops = 2;
    c.skip_loop_final_norm = false;
    match tier {
        SynthTier::Portable => {
            c.vocab_size = 128;
            c.hidden_size = 64;
            c.intermediate_size = 128;
            c.num_hidden_layers = 2;
            c.num_attention_heads = 4;
            c.num_key_value_heads = 2;
            c.head_dim = Some(16);
            c.max_position_embeddings = 256;
        }
        SynthTier::Accelerator => {
            c.vocab_size = 512;
            c.hidden_size = 256;
            c.intermediate_size = 512;
            c.num_hidden_layers = 4;
            c.num_attention_heads = 8;
            c.num_key_value_heads = 2;
            c.head_dim = Some(32);
            c.max_position_embeddings = 1024;
        }
    }
    c
}

fn tier_for(device: Device) -> SynthTier {
    match device {
        Device::Gpu | Device::Vulkan | Device::Ane => SynthTier::Portable,
        _ => SynthTier::Accelerator,
    }
}

fn ramp(n: usize, scale: f32) -> Vec<f32> {
    (0..n).map(|i| 0.001 + scale * (i as f32) * 0.01).collect()
}

fn synth_weights(cfg: &Llama32Config) -> WeightMap {
    let h = cfg.hidden_size;
    let q = cfg.q_proj_dim();
    let kv = cfg.kv_proj_dim();
    let ff = cfg.intermediate_size;
    let mut t = HashMap::new();
    t.insert(
        "model.embed_tokens.weight".into(),
        (ramp(cfg.vocab_size * h, 0.001), vec![cfg.vocab_size, h]),
    );
    for i in 0..cfg.physical_layers() {
        let lp = format!("model.layers.{i}");
        t.insert(format!("{lp}.input_layernorm.weight"), (vec![1.0; h], vec![h]));
        t.insert(
            format!("{lp}.post_attention_layernorm.weight"),
            (vec![1.0; h], vec![h]),
        );
        t.insert(
            format!("{lp}.self_attn.q_proj.weight"),
            (ramp(q * h, 0.01), vec![q, h]),
        );
        t.insert(
            format!("{lp}.self_attn.k_proj.weight"),
            (ramp(kv * h, 0.01), vec![kv, h]),
        );
        t.insert(
            format!("{lp}.self_attn.v_proj.weight"),
            (ramp(kv * h, 0.01), vec![kv, h]),
        );
        t.insert(
            format!("{lp}.self_attn.o_proj.weight"),
            (ramp(h * q, 0.01), vec![h, q]),
        );
        t.insert(
            format!("{lp}.mlp.gate_proj.weight"),
            (ramp(ff * h, 0.01), vec![ff, h]),
        );
        t.insert(
            format!("{lp}.mlp.up_proj.weight"),
            (ramp(ff * h, 0.01), vec![ff, h]),
        );
        t.insert(
            format!("{lp}.mlp.down_proj.weight"),
            (ramp(h * ff, 0.01), vec![h, ff]),
        );
    }
    t.insert("model.norm.weight".into(), (vec![1.0; h], vec![h]));
    t.insert(
        "lm_head.weight".into(),
        (ramp(cfg.vocab_size * h, 0.001), vec![cfg.vocab_size, h]),
    );
    WeightMap::from_tensors(t)
}

fn attach(compiled: &mut CompiledGraph, params: &HashMap<String, Vec<f32>>) {
    for (n, d) in params {
        compiled.set_param(n, d);
    }
}

fn timed_ms<F: FnMut()>(warm: usize, iters: usize, mut f: F) -> f64 {
    for _ in 0..warm {
        f();
    }
    let t0 = Instant::now();
    for _ in 0..iters {
        f();
    }
    t0.elapsed().as_secs_f64() * 1000.0 / iters.max(1) as f64
}

struct SynthRow {
    device: String,
    tier: String,
    max_seq: usize,
    prompt: usize,
    prefill_ms: f64,
    decode_ms: f64,
    decode_tok_s: f64,
    kv_mib: f64,
    note: String,
}

fn run_synth(device: Device, warm: usize, iters: usize) -> Result<SynthRow> {
    prepare_device(device);
    if device != Device::Cpu && !rlx_runtime::is_available(device) {
        bail!("unavailable");
    }
    let cfg = synth_cfg(tier_for(device));
    validate_device(&cfg, device, false)?;
    let plan = BackendPlan::for_device(&cfg, device);
    let seq = plan.prompt_len.min(plan.max_seq).max(4);
    let past = seq;

    let mut wm = synth_weights(&cfg);
    let (prefill_g, prefill_p) =
        build_llama32_graph_sized_last_logits(&cfg, &mut wm, 1, seq, false)?;
    let profile_p = CompileProfile::llama32_prefill();
    let mut prefill = compile_graph_with_profile(device, prefill_g, &profile_p)?;
    attach(&mut prefill, &prefill_p);

    let ids: Vec<f32> = (0..seq).map(|i| (i as f32) + 1.0).collect();
    let last = [(seq - 1) as f32];
    let prefill_ms = timed_ms(warm, iters, || {
        let _ = prefill.run(&[("input_ids", ids.as_slice()), ("last_token_idx", &last)]);
    });

    let mut wm2 = synth_weights(&cfg);
    let (decode_g, decode_p) =
        build_llama32_decode_graph_sized_ext(&cfg, &mut wm2, 1, past, false)?;
    let profile_d = CompileProfile::llama32_decode();
    let mut decode = compile_graph_with_profile(device, decode_g, &profile_d)?;
    attach(&mut decode, &decode_p);

    let kv_dim = cfg.kv_proj_dim();
    let zeros = vec![0.0f32; past * kv_dim];
    let mut named: Vec<(String, Vec<f32>)> = Vec::with_capacity(1 + 2 * cfg.kv_layers());
    named.push(("input_ids".into(), vec![7.0f32]));
    for i in 0..cfg.kv_layers() {
        named.push((format!("past_k_{i}"), zeros.clone()));
        named.push((format!("past_v_{i}"), zeros.clone()));
    }
    let decode_ms = timed_ms(warm, iters, || {
        let refs: Vec<(&str, &[f32])> = named
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_slice()))
            .collect();
        let _ = decode.run(&refs);
    });

    Ok(SynthRow {
        device: device_label(device).into(),
        tier: match tier_for(device) {
            SynthTier::Portable => "portable".into(),
            SynthTier::Accelerator => "accel".into(),
        },
        max_seq: plan.max_seq,
        prompt: seq,
        prefill_ms,
        decode_ms,
        decode_tok_s: 1000.0 / decode_ms.max(1e-9),
        kv_mib: kv_cache_bytes(&cfg, plan.max_seq) as f64 / (1024.0 * 1024.0),
        note: plan.note.into(),
    })
}

struct RealRow {
    device: String,
    max_seq: usize,
    prompt: usize,
    tokens: usize,
    build_ms: f64,
    prefill_ms: f64,
    gen_ms: f64,
    tok_s: f64,
    note: String,
}

fn run_real(weights: &PathBuf, device: Device, warm: usize) -> Result<RealRow> {
    prepare_device(device);
    if device != Device::Cpu && !rlx_runtime::is_available(device) {
        bail!("unavailable");
    }
    let cfg = nanbeige42_3b_preset();
    let plan = BackendPlan::for_device(&cfg, device);
    if !plan.allow_full_f32 {
        bail!("{}", plan.note);
    }
    rlx_nanbeige::assert_full_model_fits(&cfg, device, plan.max_seq)?;

    let t0 = Instant::now();
    let budget_gb = soft_budget_gb(device);
    let mut runner = NanbeigeRunner::builder()
        .weights(weights)
        .with_device_plan(&cfg, device)
        .max_memory_gb(budget_gb)
        .build()
        .context("build runner")?;
    let build_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let prompt: Vec<u32> = (1..=plan.prompt_len as u32).collect();
    for _ in 0..warm {
        let _ = runner.predict_logits(&prompt)?;
    }
    let t = Instant::now();
    let _ = runner.predict_logits(&prompt)?;
    let prefill_ms = t.elapsed().as_secs_f64() * 1000.0;

    let t = Instant::now();
    let _ = runner.generate(&prompt, plan.decode_tokens, |_| {})?;
    let gen_ms = t.elapsed().as_secs_f64() * 1000.0;
    let tok_s = plan.decode_tokens as f64 / (gen_ms / 1000.0).max(1e-9);

    Ok(RealRow {
        device: device_label(device).into(),
        max_seq: plan.max_seq,
        prompt: plan.prompt_len,
        tokens: plan.decode_tokens,
        build_ms,
        prefill_ms,
        gen_ms,
        tok_s,
        note: plan.note.into(),
    })
}

fn ab_bucket_metal_mlx(warm: usize, iters: usize) -> Result<()> {
    println!("\n## A/B bucketed decode (synth accel, Metal+MLX when available)");
    let cfg = synth_cfg(SynthTier::Accelerator);
    let past = 32usize;
    for device in [Device::Metal, Device::Mlx] {
        if !rlx_runtime::is_available(device) {
            println!("  {device:?}: skip");
            continue;
        }
        let mut wm = synth_weights(&cfg);
        let (g, p) = build_llama32_decode_graph_sized_ext(&cfg, &mut wm, 1, past, false)?;
        let profile = CompileProfile::llama32_decode();
        let mut compiled = compile_graph_with_profile(device, g, &profile)?;
        attach(&mut compiled, &p);
        let kv_dim = cfg.kv_proj_dim();
        let zeros = vec![0.0f32; past * kv_dim];
        let mut named: Vec<(String, Vec<f32>)> = Vec::new();
        named.push(("input_ids".into(), vec![3.0]));
        for i in 0..cfg.kv_layers() {
            named.push((format!("past_k_{i}"), zeros.clone()));
            named.push((format!("past_v_{i}"), zeros.clone()));
        }
        let ms = timed_ms(warm, iters, || {
            let refs: Vec<(&str, &[f32])> = named
                .iter()
                .map(|(k, v)| (k.as_str(), v.as_slice()))
                .collect();
            let _ = compiled.run(&refs);
        });
        println!(
            "  {device:?}: decode {ms:.2} ms/tok  ({:.1} tok/s)  kv_layers={}  (plan bucketed=true)",
            1000.0 / ms,
            cfg.kv_layers()
        );
    }
    Ok(())
}

fn main() -> Result<()> {
    let opts = parse_args()?;
    // Raise MLX compile node cap before any synth/real graph touches MLX
    // (OnceLock config is fixed after first compile otherwise).
    for &d in &opts.devices {
        prepare_device(d);
    }
    let preset = nanbeige42_3b_preset();
    println!("# nanbeige backend_bench");
    println!(
        "  full-3B est params≈{:.2} GiB F32  kv@512≈{:.1} MiB (loops={})",
        approx_param_bytes_f32(&preset) as f64 / (1024.0 * 1024.0 * 1024.0),
        kv_cache_bytes(&preset, 512) as f64 / (1024.0 * 1024.0),
        preset.num_loops
    );
    println!(
        "  working_set@512≈{:.2} GiB",
        working_set_bytes(&preset, 512) as f64 / (1024.0 * 1024.0 * 1024.0)
    );
    println!("  warm={} iters={}", opts.warm, opts.iters);

    println!("\n## Synth looped prefill + decode");
    println!(
        "| {:<8} | {:<8} | {:>7} | {:>6} | {:>10} | {:>10} | {:>10} | {:>7} | note |",
        "device", "tier", "max_seq", "prompt", "prefill_ms", "decode_ms", "tok/s", "kv_MiB"
    );
    println!(
        "|----------|----------|---------|--------|------------|------------|------------|---------|------|"
    );

    let mut failed = Vec::new();
    let mut best_accel: Option<(String, f64)> = None;
    let mut best_accel_gpu: Option<(String, f64)> = None;
    let mut best_portable: Option<(String, f64)> = None;
    for &device in &opts.devices {
        let label = device_label(device);
        let outcome = catch_unwind(AssertUnwindSafe(|| run_synth(device, opts.warm, opts.iters)));
        match outcome {
            Ok(Ok(row)) => {
                println!(
                    "| {:<8} | {:<8} | {:>7} | {:>6} | {:>10.2} | {:>10.2} | {:>10.1} | {:>7.2} | {} |",
                    row.device,
                    row.tier,
                    row.max_seq,
                    row.prompt,
                    row.prefill_ms,
                    row.decode_ms,
                    row.decode_tok_s,
                    row.kv_mib,
                    row.note
                );
                if row.tier == "accel" {
                    if best_accel
                        .as_ref()
                        .map(|(_, t)| row.decode_tok_s > *t)
                        .unwrap_or(true)
                    {
                        best_accel = Some((row.device.clone(), row.decode_tok_s));
                    }
                    if row.device != "cpu"
                        && best_accel_gpu
                            .as_ref()
                            .map(|(_, t)| row.decode_tok_s > *t)
                            .unwrap_or(true)
                    {
                        best_accel_gpu = Some((row.device.clone(), row.decode_tok_s));
                    }
                } else if best_portable
                    .as_ref()
                    .map(|(_, t)| row.decode_tok_s > *t)
                    .unwrap_or(true)
                {
                    best_portable = Some((row.device.clone(), row.decode_tok_s));
                }
            }
            Ok(Err(e)) => {
                println!("| {label:<8} | skip     |         |        |            |            |            |         | {e:#} |");
            }
            Err(_) => {
                eprintln!("| {label:<8} | PANIC/OOM | — |");
                failed.push(label);
            }
        }
    }
    if let Some((d, t)) = best_accel_gpu.or(best_accel) {
        println!("\nRecommended accel backend: **{d}** ({t:.1} tok/s synth decode)");
    }
    if let Some((d, t)) = best_portable {
        println!("Recommended portable backend: **{d}** ({t:.1} tok/s synth decode)");
    }

    if opts.ab_bucket {
        let _ = ab_bucket_metal_mlx(opts.warm, opts.iters);
    }

    if let Some(weights) = &opts.weights {
        println!("\n## Real weights ({})", weights.display());
        println!(
            "| {:<8} | {:>7} | {:>6} | {:>5} | {:>10} | {:>10} | {:>10} | {:>8} | note |",
            "device", "max_seq", "prompt", "tok", "build_ms", "prefill_ms", "gen_ms", "tok/s"
        );
        println!(
            "|----------|---------|--------|-------|------------|------------|------------|----------|------|"
        );
        for &device in &opts.devices {
            let label = device_label(device);
            let outcome =
                catch_unwind(AssertUnwindSafe(|| run_real(weights, device, opts.warm.min(1))));
            match outcome {
                Ok(Ok(row)) => {
                    println!(
                        "| {:<8} | {:>7} | {:>6} | {:>5} | {:>10.0} | {:>10.1} | {:>10.1} | {:>8.2} | {} |",
                        row.device,
                        row.max_seq,
                        row.prompt,
                        row.tokens,
                        row.build_ms,
                        row.prefill_ms,
                        row.gen_ms,
                        row.tok_s,
                        row.note
                    );
                }
                Ok(Err(e)) => {
                    println!(
                        "| {label:<8} | skip    |        |       |            |            |            |          | {e:#} |"
                    );
                }
                Err(_) => {
                    eprintln!("| {label:<8} | PANIC/OOM during real run |");
                    failed.push(label);
                }
            }
        }
    } else {
        println!(
            "\n(real 3B skipped — pass `--weights /tmp/rlx-weights/Nanbeige4.2-3B` after `just fetch-nanbeige`)"
        );
    }

    if !failed.is_empty() {
        bail!("panic/OOM on: {}", failed.join(", "));
    }
    Ok(())
}
