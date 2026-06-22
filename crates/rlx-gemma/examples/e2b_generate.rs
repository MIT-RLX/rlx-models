// RLX — versatile ML compiler + runtime. GPLv3.
//! Greedy generation runner for Gemma 4 E2B mobile QAT (safetensors).
//!
//! Picks the device from the first CLI arg (`cpu` / `metal` / `mlx`) and
//! greedy-decodes 10 tokens from the test prompt "The capital of France is",
//! verifying the output matches the HF transformers reference exactly
//! (10/10 tokens) — same check `tests/gemma4_e2b_generate.rs` runs on CPU.
//!
//! Run:
//! ```bash
//! cargo run --release -p rlx-gemma --features "metal mlx" --example e2b_generate -- metal
//! ```
//! Expects the checkpoint at
//! `~/.cache/huggingface/hub/models--google--gemma-4-E2B-it-qat-mobile-transformers/snapshots/<rev>/`
//! (downloaded via `hf download google/gemma-4-E2B-it-qat-mobile-transformers`).

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use rlx_core::autoregressive::run_packed_prefill;
use rlx_gemma::config::GemmaConfig;
use rlx_gemma::gemma_e2b::{compile_e2b_prefill, resolve_e2b_device};
use rlx_gemma::qat_loader::GemmaQatLoader;
use rlx_runtime::Device;

const PROMPT_IDS: &[u32] = &[818, 5279, 529, 7001, 563]; // "The capital of France is"
const HF_REFERENCE: &[u32] = &[7001, 563, 7001, 563, 7001, 563, 7001, 563, 7001, 563];

fn ckpt_dir() -> Result<PathBuf> {
    if let Some(d) = std::env::var_os("RLX_GEMMA4_E2B_DIR") {
        let p = PathBuf::from(d);
        if p.join("config.json").is_file() {
            return Ok(p);
        }
        bail!("RLX_GEMMA4_E2B_DIR={p:?} has no config.json");
    }
    let home = std::env::var_os("HOME").context("HOME not set")?;
    let base = std::path::Path::new(&home).join(
        ".cache/huggingface/hub/\
         models--google--gemma-4-E2B-it-qat-mobile-transformers/snapshots",
    );
    let snap = std::fs::read_dir(&base)
        .with_context(|| format!("read_dir {base:?} — checkpoint downloaded?"))?
        .flatten()
        .next()
        .with_context(|| format!("{base:?} is empty"))?
        .path();
    if !snap.join("config.json").is_file() {
        bail!("{snap:?} has no config.json");
    }
    Ok(snap)
}

fn pick_device(arg: Option<&str>) -> Result<Device> {
    let raw = arg.unwrap_or("cpu").to_ascii_lowercase();
    match raw.as_str() {
        "cpu" => Ok(Device::Cpu),
        "metal" => Ok(Device::Metal),
        "mlx" => Ok(Device::Mlx),
        "gpu" | "wgpu" => Ok(Device::Gpu),
        "ane" | "coreml" => Ok(Device::Ane),
        "cuda" => Ok(Device::Cuda),
        "rocm" => Ok(Device::Rocm),
        "vulkan" => Ok(Device::Vulkan),
        "tpu" => Ok(Device::Tpu),
        other => bail!(
            "unknown device {other:?} — try cpu / metal / mlx / gpu / ane / cuda / rocm / vulkan / tpu"
        ),
    }
}

fn argmax(v: &[f32]) -> u32 {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

fn main() -> Result<()> {
    let requested = pick_device(std::env::args().nth(1).as_deref())?;
    let device = resolve_e2b_device(requested);
    let label = format!("{device:?}");
    if device != requested {
        eprintln!("   (requested {requested:?}, running on {device:?})");
    }
    println!("→ Gemma 4 E2B QAT greedy generation on {label}");

    let dir = ckpt_dir()?;
    let cfg = GemmaConfig::from_file(&dir.join("config.json"))?;
    let vocab = cfg.vocab_size;
    let bucket = 16usize;
    let n_new = HF_REFERENCE.len();

    let t_load = Instant::now();
    let loader = GemmaQatLoader::open(&dir)?;
    let mut bld = GemmaQatLoader::open(&dir)?;
    let mut packed = HashMap::new();
    let (graph, params) = rlx_gemma::builder::build_gemma_graph_sized_packed_ext(
        &cfg,
        &mut bld,
        1,
        bucket,
        true,
        false,
        false,
        &mut packed,
        None,
        None,
    )?;
    println!(
        "   [{label}] LOAD+BUILD  {:6.2} s",
        t_load.elapsed().as_secs_f32()
    );

    let t_compile = Instant::now();
    let mut compiled = compile_e2b_prefill(device, graph, params)?;
    println!(
        "   [{label}] COMPILE     {:6.2} s",
        t_compile.elapsed().as_secs_f32()
    );

    let mut ids = vec![0u32; bucket];
    for (i, &t) in PROMPT_IDS.iter().enumerate() {
        ids[i] = t;
    }

    let t_gen = Instant::now();
    let mut generated: Vec<u32> = Vec::with_capacity(n_new);
    for step in 0..n_new {
        let cur = PROMPT_IDS.len() + step;
        let ple = loader.compute_per_layer_inputs(&cfg, &ids)?;
        let ids_f32: Vec<f32> = ids.iter().map(|&i| i as f32).collect();
        let outs = run_packed_prefill(
            &mut compiled,
            device,
            cur,
            bucket,
            &[
                ("input_ids", ids_f32.as_slice()),
                ("per_layer_inputs", ple.as_slice()),
            ],
        );
        let logits = &outs[0];
        let next = argmax(&logits[(cur - 1) * vocab..cur * vocab]);
        generated.push(next);
        if cur < bucket {
            ids[cur] = next;
        }
        let elapsed = t_gen.elapsed().as_secs_f32();
        let last = generated.len();
        let prev = if last > 1 {
            // crude per-token: total / count
            elapsed / (last as f32)
        } else {
            elapsed
        };
        println!("   [{label}]   tok#{last:>2}  = {next:>7}  ({prev:6.2} s avg/tok)");
    }

    let matches = generated
        .iter()
        .zip(HF_REFERENCE)
        .take_while(|(a, b)| a == b)
        .count();
    println!(
        "   [{label}] GENERATE   {:6.2} s  ({} tokens · {:5.2} t/s)",
        t_gen.elapsed().as_secs_f32(),
        n_new,
        n_new as f32 / t_gen.elapsed().as_secs_f32().max(1e-6)
    );
    println!("   [{label}] rlx  : {generated:?}");
    println!("   [{label}] hf   : {HF_REFERENCE:?}");
    println!("   [{label}] match: {matches}/{n_new}");

    if matches < HF_REFERENCE.len() {
        bail!("diverged from HF after {matches}/{n_new} tokens (greedy should be bit-exact)");
    }
    println!("✓ matches HF reference end-to-end");
    Ok(())
}
