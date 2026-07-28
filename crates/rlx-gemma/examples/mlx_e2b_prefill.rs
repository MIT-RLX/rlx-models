// RLX — versatile ML compiler + runtime. GPLv3.
//! Validate mlx-community gemma-3n E2B **text** prefill against the mlx-lm oracle.
//!
//! Loads an `mlx-community/gemma-3n-E2B-it-4bit`-style directory (config.json
//! `quantization` affine block + packed uint32 `.weight` / `.scales` / `.biases`),
//! builds the E2B LM prefill graph via the shared Gemma builder, runs one prefill
//! on the requested device, and compares the **last-token logits** to the oracle
//! written by `scripts/mlx_oracle_dump.py`:
//!   - `oracle.json`                    → prompt_ids, prefill_argmax, vocab
//!   - `oracle_prefill_last_logits.npy` → f32 logits `[vocab]`
//!
//! Run:
//! ```bash
//! PYTHONPATH=/Users/Shared/mlx-lm python3 scripts/mlx_oracle_dump.py \
//!     .mlx-test/gemma3n-e2b-4bit --prompt "The capital of France is" --ngen 3
//! cargo run --release -p rlx-gemma --example mlx_e2b_prefill -- .mlx-test/gemma3n-e2b-4bit cpu
//! ```

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use rlx_core::autoregressive::run_packed_prefill;
use rlx_gemma::config::GemmaConfig;
use rlx_gemma::gemma_e2b::{compile_e2b_prefill, resolve_e2b_device};
use rlx_gemma::qat_loader::GemmaQatLoader;
use rlx_runtime::Device;

fn pick_device(arg: Option<&str>) -> Result<Device> {
    match arg.unwrap_or("cpu").to_ascii_lowercase().as_str() {
        "cpu" => Ok(Device::Cpu),
        "metal" => Ok(Device::Metal),
        "mlx" => Ok(Device::Mlx),
        "gpu" | "wgpu" => Ok(Device::Gpu),
        "cuda" => Ok(Device::Cuda),
        "vulkan" => Ok(Device::Vulkan),
        other => bail!("unknown device {other:?}"),
    }
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let mut dot = 0f64;
    let mut na = 0f64;
    let mut nb = 0f64;
    for (x, y) in a.iter().zip(b) {
        dot += *x as f64 * *y as f64;
        na += (*x as f64).powi(2);
        nb += (*y as f64).powi(2);
    }
    dot / (na.sqrt() * nb.sqrt())
}

/// Minimal `.npy` reader for a C-contiguous little-endian `float32` array.
fn read_npy_f32(path: &Path) -> Result<Vec<f32>> {
    let b = std::fs::read(path).with_context(|| format!("read {path:?}"))?;
    anyhow::ensure!(&b[..6] == b"\x93NUMPY", "{path:?}: not a .npy file");
    let data_start = match b[6] {
        1 => 10 + u16::from_le_bytes([b[8], b[9]]) as usize,
        _ => 12 + u32::from_le_bytes([b[8], b[9], b[10], b[11]]) as usize,
    };
    let hdr = String::from_utf8_lossy(&b[..data_start]);
    anyhow::ensure!(
        hdr.contains("<f4") || hdr.contains("float32"),
        "{path:?}: expected <f4 dtype, header={hdr}"
    );
    Ok(b[data_start..]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let dir = PathBuf::from(
        args.next()
            .unwrap_or_else(|| ".mlx-test/gemma3n-e2b-4bit".to_string()),
    );
    let requested = pick_device(args.next().as_deref())?;
    let device = resolve_e2b_device(requested);

    let cfg = GemmaConfig::from_file(&dir.join("config.json"))?;
    let vocab = cfg.vocab_size;
    println!(
        "→ gemma-3n E2B mlx-affine prefill on {device:?}  (arch={:?}, layers={}, kv_shared={}, hidden={})",
        cfg.arch,
        cfg.active_num_layers(),
        cfg.num_kv_shared_layers,
        cfg.hidden_size
    );

    // Oracle (prompt ids + reference logits).
    let oracle: serde_json::Value = serde_json::from_slice(
        &std::fs::read(dir.join("oracle.json"))
            .context("read oracle.json — run scripts/mlx_oracle_dump.py first")?,
    )?;
    let prompt_ids: Vec<u32> = oracle["prompt_ids"]
        .as_array()
        .context("oracle.prompt_ids")?
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    let oracle_argmax = oracle["prefill_argmax"].as_u64().unwrap() as usize;
    let oracle_logits = read_npy_f32(&dir.join("oracle_prefill_last_logits.npy"))?;
    anyhow::ensure!(
        oracle_logits.len() == vocab,
        "oracle logits len {} != vocab {vocab}",
        oracle_logits.len()
    );
    let seq = prompt_ids.len();
    println!("   prompt_ids ({seq}): {prompt_ids:?}   oracle argmax={oracle_argmax}");

    // Build + compile the E2B prefill graph (F32 dequant, exact) at seq = prompt len.
    let t0 = Instant::now();
    let loader = GemmaQatLoader::open(&dir)?;
    let mut bld = GemmaQatLoader::open(&dir)?;
    let mut packed = HashMap::new();
    let (graph, params) = rlx_gemma::builder::build_gemma_graph_sized_packed_ext(
        &cfg,
        &mut bld,
        1,
        seq,
        true,
        false,
        false,
        &mut packed,
        None,
        None,
    )?;
    println!("   LOAD+BUILD  {:6.2} s", t0.elapsed().as_secs_f32());

    let t1 = Instant::now();
    let mut compiled = compile_e2b_prefill(device, graph, params)?;
    println!("   COMPILE     {:6.2} s", t1.elapsed().as_secs_f32());

    // Per-layer embedding inputs (out-of-graph gather + project + combine).
    let ple = loader.compute_per_layer_inputs(&cfg, &prompt_ids)?;
    let ids_f32: Vec<f32> = prompt_ids.iter().map(|&i| i as f32).collect();

    let t2 = Instant::now();
    let outs = run_packed_prefill(
        &mut compiled,
        device,
        seq,
        seq,
        &[
            ("input_ids", ids_f32.as_slice()),
            ("per_layer_inputs", ple.as_slice()),
        ],
    );
    println!("   PREFILL     {:6.2} s", t2.elapsed().as_secs_f32());

    let logits = &outs[0];
    anyhow::ensure!(
        logits.len() >= seq * vocab,
        "logits len {} < seq*vocab {}",
        logits.len(),
        seq * vocab
    );
    let last = &logits[(seq - 1) * vocab..seq * vocab];
    let rlx_argmax = argmax(last);
    let cos = cosine(last, &oracle_logits);

    println!("\n── result ──────────────────────────────────────────────");
    println!("   rlx argmax    = {rlx_argmax}");
    println!("   oracle argmax = {oracle_argmax}");
    println!("   cosine        = {cos:.6}");
    let ok_finite = last.iter().all(|v| v.is_finite());
    println!("   finite        = {ok_finite}");
    // Peek top-5 for context.
    let mut idx: Vec<usize> = (0..vocab).collect();
    idx.sort_by(|&a, &b| last[b].partial_cmp(&last[a]).unwrap());
    let top5: Vec<(usize, f32)> = idx[..5].iter().map(|&i| (i, last[i])).collect();
    println!("   rlx top5      = {top5:?}");

    if !ok_finite {
        bail!("non-finite logits");
    }
    if rlx_argmax == oracle_argmax && cos > 0.999 {
        println!("\n✓ PASS  argmax exact + cosine {cos:.6} ≈ 1.0 vs mlx-lm oracle");
        Ok(())
    } else {
        bail!("MISMATCH argmax {rlx_argmax} vs {oracle_argmax}, cosine {cos:.6}");
    }
}
