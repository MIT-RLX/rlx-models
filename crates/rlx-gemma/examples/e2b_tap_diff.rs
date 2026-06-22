// RLX — versatile ML compiler + runtime. GPLv3.
//! Bisect Metal/MLX-vs-CPU divergence on Gemma 4 E2B QAT by enabling
//! `RLX_TAP_L0=1` to surface layer-0 intermediates as extra graph outputs,
//! then comparing each tap's statistics across backends.
//!
//! Run:
//! ```bash
//! RLX_TAP_L0=1 cargo run --release -p rlx-gemma --features "metal mlx" --example e2b_tap_diff
//! ```
//! The taps (defined by the prefill builder around `l0_taps.push(...)`):
//!   1: embed*scale (layer-0 input)
//!   2: input_layernorm(x)
//!   A: Q post-projection, pre per-head norm
//!   B: K post-projection
//!   C: V post-projection
//!   D: Q reshape-only (4D pre-norm)
//!   E: Q after per-head q_norm (4D)
//!   3: Q post-norm (back to 3D)
//!   4: K post-norm (3D)
//!   5: V post-norm (3D)
//!   6: Q post-RoPE
//!   7: K post-RoPE
//!   F: K_rep (post-repeat to Q heads)
//!   G: V_rep
//!   8: attention output (pre-o_proj)
//!   9: attn_out after post_attention_norm
//!  10: residual h + attn_out
//!  11: layer 0 final h (post-FFN add)
//!
//! Compare any tap's stats. First divergent tap pinpoints the buggy op.

use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use rlx_core::flow_util::compile_graph_gemma_prefill_with_params;
use rlx_gemma::config::GemmaConfig;
use rlx_gemma::qat_loader::GemmaQatLoader;
use rlx_runtime::Device;

const PROMPT_IDS: &[u32] = &[818, 5279, 529, 7001, 563, 7001];
const BUCKET: usize = 16;

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

#[derive(Clone)]
struct TapStats {
    len: usize,
    nan: usize,
    min: f32,
    max: f32,
    mean: f64,
    first4: [f32; 4],
}

fn stats(v: &[f32]) -> TapStats {
    let mut nan = 0;
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut sum = 0f64;
    let mut nf = 0u64;
    for &x in v {
        if x.is_nan() {
            nan += 1;
        } else {
            if x < min {
                min = x;
            }
            if x > max {
                max = x;
            }
            sum += x as f64;
            nf += 1;
        }
    }
    let mean = if nf > 0 { sum / nf as f64 } else { 0.0 };
    let mut first4 = [0f32; 4];
    for (i, &x) in v.iter().take(4).enumerate() {
        first4[i] = x;
    }
    TapStats {
        len: v.len(),
        nan,
        min,
        max,
        mean,
        first4,
    }
}

fn run_on(device: Device, ple: &[f32], ids_f32: &[f32]) -> Result<Vec<Vec<f32>>> {
    let dir = ckpt_dir()?;
    let cfg = GemmaConfig::from_file(&dir.join("config.json"))?;
    let mut bld = GemmaQatLoader::open(&dir)?;
    let mut packed = HashMap::new();
    let (graph, params) = rlx_gemma::builder::build_gemma_graph_sized_packed_ext(
        &cfg,
        &mut bld,
        1,
        BUCKET,
        true,
        false,
        false,
        &mut packed,
        None,
        None,
    )?;
    let mut compiled = compile_graph_gemma_prefill_with_params(device, graph, params)?;
    let outs = compiled.run(&[("input_ids", ids_f32), ("per_layer_inputs", ple)]);
    Ok(outs)
}

const TAP_NAMES: &[&str] = &[
    "tap01_embed_scale",
    "tap02_input_layernorm",
    "tapA_Q_post_proj",
    "tapB_K_post_proj",
    "tapC_V_post_proj",
    "tapD_Q_4d_pre_norm",
    "tapE_Q_4d_post_norm",
    "tap03_Q_post_norm_3d",
    "tap04_K_post_norm_3d",
    "tap05_V_post_norm_3d",
    "tap06_Q_post_rope",
    "tap07_K_post_rope",
    "tapF_K_rep",
    "tapG_V_rep",
    "tap08_attn_out",
    "tap09_attn_out_post_norm",
    "tap10_residual_post_attn",
    "tap11_layer0_final_h",
];

fn main() -> Result<()> {
    if std::env::var("RLX_TAP_L0").ok().is_none() {
        bail!("set RLX_TAP_L0=1 to enable the layer-0 taps");
    }
    let dir = ckpt_dir()?;
    let cfg = GemmaConfig::from_file(&dir.join("config.json"))?;
    let loader = GemmaQatLoader::open(&dir)?;
    let mut ids = vec![0u32; BUCKET];
    for (i, &t) in PROMPT_IDS.iter().enumerate() {
        ids[i] = t;
    }
    let ple = loader.compute_per_layer_inputs(&cfg, &ids)?;
    let ids_f32: Vec<f32> = ids.iter().map(|&i| i as f32).collect();

    println!("→ tap diff on prompt {PROMPT_IDS:?} (bucket={BUCKET})");
    println!(
        "→ PLE len={}, expected={}",
        ple.len(),
        BUCKET * cfg.num_hidden_layers * cfg.ple_width()
    );

    let devices: Vec<(Device, &str)> = [
        (Device::Cpu, "Cpu"),
        (Device::Metal, "Metal"),
        (Device::Mlx, "Mlx"),
    ]
    .into_iter()
    .filter(|(d, _)| rlx_runtime::is_available(*d))
    .collect();
    let mut all_outs: Vec<(String, Vec<Vec<f32>>)> = Vec::new();
    for &(d, label) in &devices {
        let t = std::time::Instant::now();
        match run_on(d, &ple, &ids_f32) {
            Ok(outs) => {
                println!(
                    "   [{label}] ran in {:.2}s, {} outputs",
                    t.elapsed().as_secs_f32(),
                    outs.len()
                );
                all_outs.push((label.into(), outs));
            }
            Err(e) => {
                println!("   [{label}] FAILED: {e}");
            }
        }
    }
    if all_outs.is_empty() {
        bail!("no successful runs");
    }

    let n_outs = all_outs[0].1.len();
    println!(
        "→ {} outputs total (1 logits + {} taps)",
        n_outs,
        n_outs - 1
    );

    // Per-position argmax across the whole bucket: pinpoints a position-dependent
    // backend divergence (e.g. the MLX generation bug shows up at pos >= prompt).
    {
        let vocab = cfg.vocab_size;
        println!("\n── Per-position argmax (all {BUCKET} positions) ──");
        let amx = |row: &[f32]| {
            row.iter()
                .enumerate()
                .filter(|(_, v)| v.is_finite())
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0)
        };
        for pos in 0..BUCKET {
            let mut line = format!("   pos {pos:>2}:");
            let mut cpu_ax = None;
            for (label, outs) in &all_outs {
                let row = &outs[0][pos * vocab..(pos + 1) * vocab];
                let ax = amx(row);
                if label == "Cpu" {
                    cpu_ax = Some(ax);
                }
                let flag = if cpu_ax.is_some() && Some(ax) != cpu_ax {
                    " ✗"
                } else {
                    ""
                };
                line.push_str(&format!("  {label}={ax:>6}{flag}"));
            }
            println!("{line}");
        }
    }

    // Output[0] is logits, output[1..] are taps in the order TAP_NAMES.
    // The first 5 prompt-token logits matter for greedy decoding.
    println!(
        "\n── Last-token logits (tok#{}, argmax) ──",
        PROMPT_IDS.len() - 1
    );
    let vocab = cfg.vocab_size;
    for (label, outs) in &all_outs {
        let logits = &outs[0];
        let last_pos = PROMPT_IDS.len() - 1;
        let row = &logits[last_pos * vocab..(last_pos + 1) * vocab];
        let s = stats(row);
        let ax = row
            .iter()
            .enumerate()
            .filter(|(_, v)| v.is_finite())
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        println!(
            "   [{label:>5}] argmax={ax:>6}  min={:.3e} mean={:.3e} max={:.3e}  nan={}  first4={:?}",
            s.min, s.mean, s.max, s.nan, s.first4
        );
    }

    println!("\n── Tap-by-tap stats (length, NaN, min, mean, max, first 4 vals) ──");
    let n_taps = n_outs.saturating_sub(1);
    for i in 0..n_taps {
        let name = TAP_NAMES.get(i).copied().unwrap_or("<extra>");
        println!("\n   {name} (output[{}])", i + 1);
        for (label, outs) in &all_outs {
            if let Some(t) = outs.get(i + 1) {
                let s = stats(t);
                println!(
                    "      [{label:>5}] len={:>7}  nan={:>3}  min={:+.3e}  mean={:+.3e}  max={:+.3e}  first4={:?}",
                    s.len, s.nan, s.min, s.mean, s.max, s.first4
                );
            }
        }
    }

    println!("\n── Tap full-tensor L2 vs CPU ──");
    if let Some(ci) = all_outs.iter().position(|(l, _)| l == "Cpu") {
        let cpu_outs = &all_outs[ci].1;
        for tap_i in 0..n_taps {
            let name = TAP_NAMES.get(tap_i).copied().unwrap_or("<extra>");
            let Some(c) = cpu_outs.get(tap_i + 1) else {
                continue;
            };
            for (label, outs) in &all_outs {
                if label == "Cpu" {
                    continue;
                }
                let Some(g) = outs.get(tap_i + 1) else {
                    continue;
                };
                let n = c.len().min(g.len());
                let l2: f64 = c[..n]
                    .iter()
                    .zip(&g[..n])
                    .map(|(a, b)| (*a as f64 - *b as f64).powi(2))
                    .sum::<f64>()
                    .sqrt();
                println!("   {name:>24} [{label:>5}] L2={l2:.3e}");
            }
        }
    }

    // Numerical diff for first 4 entries — useful when stats agree but values diverge.
    println!("\n── First divergent tap (where Metal or Mlx first differs from Cpu) ──");
    let cpu_idx = all_outs.iter().position(|(l, _)| l == "Cpu");
    if let Some(ci) = cpu_idx {
        let cpu_outs = &all_outs[ci].1;
        for (label, outs) in &all_outs {
            if label == "Cpu" {
                continue;
            }
            let mut first_diverge: Option<(usize, &str, f32, f32)> = None;
            for tap_i in 0..n_taps {
                let Some(c) = cpu_outs.get(tap_i + 1) else {
                    continue;
                };
                let Some(g) = outs.get(tap_i + 1) else {
                    continue;
                };
                let n = c.len().min(g.len());
                for j in 0..n.min(4096) {
                    let a = c[j];
                    let b = g[j];
                    let diff = (a - b).abs();
                    let scale = a.abs().max(b.abs()).max(1e-6);
                    if diff / scale > 0.01 && diff > 1e-3 {
                        first_diverge =
                            Some((tap_i, TAP_NAMES.get(tap_i).copied().unwrap_or("?"), a, b));
                        break;
                    }
                }
                if first_diverge.is_some() {
                    break;
                }
            }
            match first_diverge {
                Some((i, name, cv, gv)) => {
                    println!(
                        "   [{label}] first diverges at tap #{i} ({name}): cpu={cv} vs {label}={gv}"
                    );
                }
                None => {
                    println!("   [{label}] matches Cpu within tolerance on all checked taps");
                }
            }
        }
    }

    Ok(())
}
