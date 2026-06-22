// RLX — versatile ML compiler + runtime. GPLv3.
//! Per-layer hidden-state trajectory: compare CPU / Metal / MLX on the same
//! E2B QAT forward. CPU is the in-repo reference backend.
//!
//! Run:
//! ```bash
//! RLX_TAP_ALL=1 cargo run --release -p rlx-gemma --features apple-silicon --example e2b_backend_traj
//! ```

use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use rlx_core::flow_util::compile_graph_gemma_prefill_with_params;
use rlx_gemma::config::GemmaConfig;
use rlx_gemma::qat_loader::GemmaQatLoader;
use rlx_runtime::Device;

const PROMPT_IDS: &[u32] = &[818, 5279, 529, 7001, 563];

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
    let snap = std::fs::read_dir(&base)?
        .flatten()
        .next()
        .with_context(|| format!("{base:?} empty"))?
        .path();
    if !snap.join("config.json").is_file() {
        bail!("{snap:?} has no config.json");
    }
    Ok(snap)
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
    let na: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    (dot / (na * nb + 1e-12)) as f32
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    a[..n]
        .iter()
        .zip(&b[..n])
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}

fn run(
    device: Device,
    cfg: &GemmaConfig,
    dir: &std::path::Path,
    ple: &[f32],
    ids_f32: &[f32],
) -> Result<Vec<Vec<f32>>> {
    let mut loader = GemmaQatLoader::open(dir)?;
    let mut packed = HashMap::new();
    let (graph, params) = rlx_gemma::builder::build_gemma_graph_sized_packed_ext(
        cfg,
        &mut loader,
        1,
        ids_f32.len(),
        true,
        false,
        false,
        &mut packed,
        None,
        None,
    )?;
    let mut compiled = compile_graph_gemma_prefill_with_params(device, graph, params)?;
    Ok(compiled.run(&[("input_ids", ids_f32), ("per_layer_inputs", ple)]))
}

fn main() -> Result<()> {
    if std::env::var("RLX_TAP_ALL").ok().is_none() {
        bail!("set RLX_TAP_ALL=1");
    }
    let dir = ckpt_dir()?;
    let cfg = GemmaConfig::from_file(&dir.join("config.json"))?;
    let loader = GemmaQatLoader::open(&dir)?;
    let ids: Vec<u32> = PROMPT_IDS.to_vec();
    let ple = loader.compute_per_layer_inputs(&cfg, &ids)?;
    let ids_f32: Vec<f32> = ids.iter().map(|&i| i as f32).collect();
    let h = cfg.hidden_size;
    let nl = cfg.num_hidden_layers;
    let vocab = cfg.vocab_size;
    let seq = ids.len();

    let devices: Vec<(Device, &str)> = [
        (Device::Cpu, "Cpu"),
        (Device::Metal, "Metal"),
        (Device::Mlx, "Mlx"),
    ]
    .into_iter()
    .filter(|(d, _)| rlx_runtime::is_available(*d))
    .collect();

    let mut runs: Vec<(String, Vec<Vec<f32>>)> = Vec::new();
    for &(d, label) in &devices {
        let t = std::time::Instant::now();
        let outs = run(d, &cfg, &dir, &ple, &ids_f32)?;
        println!(
            "   [{label}] forward {:.2}s, {} outputs",
            t.elapsed().as_secs_f32(),
            outs.len()
        );
        runs.push((label.into(), outs));
    }

    let cpu_idx = runs
        .iter()
        .position(|(l, _)| l == "Cpu")
        .context("Cpu run missing")?;
    let cpu = &runs[cpu_idx].1;
    let last = seq - 1;
    let cpu_ax = argmax(&cpu[0][last * vocab..(last + 1) * vocab]);
    println!("\n── Last-token argmax (CPU ref = {cpu_ax}) ──");
    for (label, outs) in &runs {
        let ax = argmax(&outs[0][last * vocab..(last + 1) * vocab]);
        let tag = if ax == cpu_ax { "✓" } else { "✗ DIVERGED" };
        println!("   [{label:>5}] argmax={ax:>6} {tag}");
    }

    println!("\n── Per-layer hidden cos / maxdiff vs CPU (tok0 + all tokens) ──");
    println!("   layer │  Metal cos(all) maxdiff │   MLX cos(all) maxdiff");
    println!("   ──────┼─────────────────────────┼────────────────────────");
    let mut first_metal: Option<(usize, f32, f32)> = None;
    let mut first_mlx: Option<(usize, f32, f32)> = None;
    for layer in 0..nl {
        let cpu_h = &cpu[1 + layer];
        let row = format!("   {layer:>5} │");
        let mut line = row;
        for (label, outs) in &runs {
            if label == "Cpu" {
                continue;
            }
            let other = &outs[1 + layer];
            let all_n = seq * h;
            let cos_all = cosine(&cpu_h[..all_n], &other[..all_n]);
            let md = max_abs(&cpu_h[..all_n], &other[..all_n]);
            line.push_str(&format!(" {label:>5} {cos_all:>6.4} {md:>8.3e} │"));
            if label == "Metal" && first_metal.is_none() && (cos_all < 0.9999 || md > 0.05) {
                first_metal = Some((layer, cos_all, md));
            }
            if label == "Mlx" && first_mlx.is_none() && (cos_all < 0.9999 || md > 0.05) {
                first_mlx = Some((layer, cos_all, md));
            }
        }
        println!("{line}");
    }

    println!("\n── First layer where backend diverges from CPU (cos<0.9999 or maxdiff>0.05) ──");
    match first_metal {
        Some((l, c, d)) => println!("   Metal first diverge: layer {l} cos={c:.4} maxdiff={d:.3e}"),
        None => println!("   Metal matches CPU within tolerance on all layers"),
    }
    match first_mlx {
        Some((l, c, d)) => println!("   Mlx first diverge: layer {l} cos={c:.4} maxdiff={d:.3e}"),
        None => println!("   Mlx matches CPU within tolerance on all layers"),
    }

    Ok(())
}
