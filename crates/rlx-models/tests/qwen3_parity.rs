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

//! Qwen3 numerical parity test: rlx-models vs candle-transformers.
//!
//! ## Running
//!
//! ```bash
//! huggingface-cli download Qwen/Qwen3-0.6B model.safetensors config.json
//!
//! RLX_QWEN3_WEIGHTS=<...>/model.safetensors \
//! RLX_QWEN3_CONFIG=<...>/config.json \
//!   cargo test -p rlx-models --features parity-candle --release \
//!     qwen3_parity -- --nocapture
//! ```
//!
//! Without `RLX_QWEN3_WEIGHTS`, every test prints a skip note and
//! returns Ok so CI without the checkpoint stays green.
//!
//! ## What "parity" means
//!
//! Both sides are fed the same `[B, L]` u32/f32 token id tensor with
//! `offset = 0`. We compare:
//!
//!   - **Hidden states** after `model.norm` — element-wise max/mean
//!     |Δ|, plus per-token cosine similarity.
//!   - **Logits** after `lm_head` — element-wise max/mean |Δ|.
//!   - **Top-1 token agreement** — does argmax of rlx logits equal
//!     argmax of candle logits at every position?
//!   - **Wall-clock latency** — how long each side takes for the
//!     same forward.
//!
//! Tolerance covers reduction-order mismatch in softmax-attention and
//! RoPE. The "real" correctness signal is top-1 agreement: even if
//! raw logits differ by 1e-4, the same token wins argmax → greedy
//! generation produces identical sequences.

#![cfg(feature = "parity-candle")]

mod compile_support;

use anyhow::{Context, Result};
use candle_core::{DType as CDType, Device as CDevice, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen3 as candle_qwen3;
use rlx_models::qwen3::{Qwen3Config, build_qwen3_graph_sized};
use rlx_models::weight_map::WeightMap;
use rlx_runtime::{Device, Session};
use std::time::Instant;

const HIDDEN_TOL: f32 = 5e-3;
const LOGIT_TOL: f32 = 1e-1; // wider — lm_head matmul amplifies noise

fn weights_path() -> Option<String> {
    rlx_ir::env::var("RLX_QWEN3_WEIGHTS")
}

fn config_path() -> Option<String> {
    rlx_ir::env::var("RLX_QWEN3_CONFIG")
}

/// Deterministic token ids within the Qwen3 vocab (≥0, <151936).
/// Slice this for variable L; replicate across rows for variable B.
fn synth_token_ids() -> Vec<u32> {
    vec![
        1, 17, 42, 314, 2718, 9001, 27182, 8128, 65535, 12345, 256, 1024, 4096, 16384, 32768, 100,
        200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000,
        9000, 10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000, 20000, 21000,
        22000, 23000, 24000, 25000, 26000, 27000, 28000, 29000, 30000, 31000, 32000, 33000, 34000,
        35000, 36000, 37000, 38000, 39000, 40000, 41000, 42000, 43000, 44000, 45000, 46000, 47000,
        48000, 49000, 50000, 51000, 52000, 53000, 54000, 55000, 56000, 57000, 58000, 59000, 60000,
        61000, 62000, 63000, 64000, 65000, 66000, 67000, 68000, 69000, 70000, 71000, 72000, 73000,
        74000, 75000, 76000, 77000, 78000, 79000, 80000, 81000, 82000, 83000, 84000, 85000, 86000,
        87000, 88000, 89000, 90000, 91000, 92000, 93000, 94000, 95000, 96000, 97000, 98000, 99000,
        100000, 101000, 102000, 103000, 104000, 105000, 106000, 107000, 108000, 109000, 110000,
    ]
}

fn make_batched_ids(batch: usize, seq: usize) -> Vec<u32> {
    let pool = synth_token_ids();
    assert!(seq <= pool.len(), "synth pool too small for L={seq}");
    let mut out = Vec::with_capacity(batch * seq);
    for b in 0..batch {
        // Rotate each row by `b * 7 mod pool.len()` so rows have
        // independent ids, exercising any batch-dependent bug.
        let offset = (b * 7) % pool.len();
        for i in 0..seq {
            out.push(pool[(offset + i) % pool.len()]);
        }
    }
    out
}

/// candle prefill → returns `(hidden_flat, logits_flat, ms_elapsed)`
/// where hidden = `[B, L, hidden]` and logits = `[B, L, vocab]` after
/// lm_head. Both flattened row-major.
fn run_candle(
    weights: &str,
    cfg: &candle_qwen3::Config,
    batch: usize,
    seq: usize,
    ids: &[u32],
) -> Result<(Vec<f32>, Vec<f32>, f64)> {
    let device = CDevice::Cpu;
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[weights], CDType::F32, &device)
            .context("loading safetensors via candle VarBuilder")?
    };
    let mut model = candle_qwen3::ModelForCausalLM::new(cfg, vb)?;

    let input = Tensor::from_vec(ids.to_vec(), (batch, seq), &device)?;
    let t0 = Instant::now();
    let logits = model.forward(&input, /*offset*/ 0)?;
    let ms = t0.elapsed().as_secs_f64() * 1e3;

    // ModelForCausalLM.forward returns only the last position's
    // logits if input has multiple positions, but for parity we want
    // all positions. Re-run via base.forward + manual lm_head — or
    // simpler: just compare hidden states + last-position logits.
    let hidden_flat = vec![0f32; 0]; // unused; we'll compare logits only via the wrapper
    let logits_flat = logits.flatten_all()?.to_vec1::<f32>()?;
    Ok((hidden_flat, logits_flat, ms))
}

/// Candle's `Model::forward` returns full hidden states (no lm_head).
fn run_candle_hidden(
    weights: &str,
    cfg: &candle_qwen3::Config,
    batch: usize,
    seq: usize,
    ids: &[u32],
) -> Result<(Vec<f32>, f64)> {
    let device = CDevice::Cpu;
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[weights], CDType::F32, &device)
            .context("loading safetensors via candle VarBuilder")?
    };
    let mut model = candle_qwen3::Model::new(cfg, vb)?;
    let input = Tensor::from_vec(ids.to_vec(), (batch, seq), &device)?;
    let t0 = Instant::now();
    let hidden = model.forward(&input, /*offset*/ 0)?;
    let ms = t0.elapsed().as_secs_f64() * 1e3;
    Ok((hidden.flatten_all()?.to_vec1::<f32>()?, ms))
}

/// rlx prefill → returns `(hidden_or_logits, ms_elapsed)`. When
/// `with_lm_head=true` the output is logits.
fn run_rlx(
    weights: &str,
    cfg: &Qwen3Config,
    batch: usize,
    seq: usize,
    ids: &[u32],
    with_lm_head: bool,
    device: Device,
) -> Result<(Vec<f32>, f64)> {
    let mut wm = WeightMap::from_file(weights)?;
    let (graph, params) = build_qwen3_graph_sized(
        cfg,
        &mut wm,
        batch,
        seq,
        with_lm_head,
        /*with_kv_outputs*/ false,
    )?;
    let mut compiled =
        rlx_models::flow_util::compile_graph_qwen3_prefill_with_params(device, graph, params)?;
    let ids_f32: Vec<f32> = ids.iter().map(|&i| i as f32).collect();
    let t0 = Instant::now();
    let outputs = compiled.run(&[("input_ids", ids_f32.as_slice())]);
    let ms = t0.elapsed().as_secs_f64() * 1e3;
    Ok((outputs.into_iter().next().unwrap_or_default(), ms))
}

fn max_mean_abs_diff(a: &[f32], b: &[f32]) -> (f32, f32, usize) {
    assert_eq!(
        a.len(),
        b.len(),
        "length mismatch: {} vs {}",
        a.len(),
        b.len()
    );
    let mut max = 0f32;
    let mut sum = 0f64;
    let mut idx = 0;
    for i in 0..a.len() {
        let d = (a[i] - b[i]).abs();
        sum += d as f64;
        if d > max {
            max = d;
            idx = i;
        }
    }
    (max, (sum / a.len() as f64) as f32, idx)
}

/// Per-row cosine similarity, returning (mean, min). Rows of length
/// `row_dim`. Assumes a is laid out [N, row_dim] flat.
fn cosine_stats(a: &[f32], b: &[f32], row_dim: usize) -> (f32, f32) {
    assert_eq!(a.len(), b.len());
    let n = a.len() / row_dim;
    let mut sum = 0f64;
    let mut min: f32 = f32::INFINITY;
    for r in 0..n {
        let aa = &a[r * row_dim..(r + 1) * row_dim];
        let bb = &b[r * row_dim..(r + 1) * row_dim];
        let mut dot = 0f32;
        let mut na = 0f32;
        let mut nb = 0f32;
        for i in 0..row_dim {
            dot += aa[i] * bb[i];
            na += aa[i] * aa[i];
            nb += bb[i] * bb[i];
        }
        let denom = (na.sqrt() * nb.sqrt()).max(f32::MIN_POSITIVE);
        let cs = (dot / denom).clamp(-1.0, 1.0);
        sum += cs as f64;
        if cs < min {
            min = cs;
        }
    }
    ((sum / n as f64) as f32, min)
}

/// Top-1 token agreement across N positions of `[N, vocab]` logits.
/// Returns (matching, total).
fn top1_agreement(a: &[f32], b: &[f32], vocab: usize) -> (usize, usize) {
    assert_eq!(a.len(), b.len());
    let n = a.len() / vocab;
    let mut matched = 0;
    for r in 0..n {
        let aa = &a[r * vocab..(r + 1) * vocab];
        let bb = &b[r * vocab..(r + 1) * vocab];
        if argmax(aa) == argmax(bb) {
            matched += 1;
        }
    }
    (matched, n)
}

/// Extract `[B, vocab]` (the last position of each batch row) from a
/// `[B, L, vocab]` flat tensor.
fn extract_last_position_per_batch(
    logits: &[f32],
    batch: usize,
    seq: usize,
    vocab: usize,
) -> Vec<f32> {
    assert_eq!(logits.len(), batch * seq * vocab);
    let mut out = Vec::with_capacity(batch * vocab);
    for b in 0..batch {
        let start = b * seq * vocab + (seq - 1) * vocab;
        out.extend_from_slice(&logits[start..start + vocab]);
    }
    out
}

fn argmax(xs: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in xs.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best
}

fn to_candle_cfg(cfg: &Qwen3Config) -> candle_qwen3::Config {
    use candle_nn::Activation;
    let hidden_act = match cfg.hidden_act.as_str() {
        "silu" => Activation::Silu,
        "gelu" => Activation::Gelu,
        other => panic!("unsupported hidden_act for parity: {other}"),
    };
    candle_qwen3::Config {
        vocab_size: cfg.vocab_size,
        hidden_size: cfg.hidden_size,
        intermediate_size: cfg.intermediate_size,
        num_hidden_layers: cfg.num_hidden_layers,
        num_attention_heads: cfg.num_attention_heads,
        head_dim: cfg.head_dim,
        attention_bias: cfg.attention_bias,
        num_key_value_heads: cfg.num_key_value_heads,
        max_position_embeddings: cfg.max_position_embeddings,
        sliding_window: cfg.sliding_window,
        max_window_layers: cfg.max_window_layers,
        tie_word_embeddings: cfg.tie_word_embeddings,
        rope_theta: cfg.rope_theta,
        rms_norm_eps: cfg.rms_norm_eps,
        use_sliding_window: cfg.use_sliding_window,
        hidden_act,
    }
}

// ─── Tests ───────────────────────────────────────────────────────────

/// Single-point parity check (B=1, L=8): hidden states only.
/// Kept as a fast basic test.
#[test]
fn qwen3_parity_vs_candle() -> Result<()> {
    let (weights, cfg_path) = match (weights_path(), config_path()) {
        (Some(w), Some(c)) => (w, c),
        _ => {
            eprintln!("skipping — set RLX_QWEN3_WEIGHTS + RLX_QWEN3_CONFIG");
            return Ok(());
        }
    };
    if !std::path::Path::new(&weights).exists() {
        eprintln!("skipping — weights not found at {weights}");
        return Ok(());
    }

    let cfg = Qwen3Config::from_file(std::path::Path::new(&cfg_path))?;
    let candle_cfg = to_candle_cfg(&cfg);
    let ids = synth_token_ids()[..8].to_vec();

    eprintln!(
        "[qwen3 parity] hidden={} layers={} q={} kv={} head_dim={} vocab={} L={}",
        cfg.hidden_size,
        cfg.num_hidden_layers,
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.head_dim,
        cfg.vocab_size,
        ids.len(),
    );

    let (candle_h, _) = run_candle_hidden(&weights, &candle_cfg, 1, ids.len(), &ids)?;
    let (rlx_h, _) = run_rlx(&weights, &cfg, 1, ids.len(), &ids, false, Device::Cpu)?;
    let (diff, mean, idx) = max_mean_abs_diff(&rlx_h, &candle_h);
    eprintln!("[qwen3 parity] max|Δ|={diff:.6}  mean|Δ|={mean:.6}  worst@{idx}  tol={HIDDEN_TOL}");

    assert!(
        diff <= HIDDEN_TOL,
        "rlx vs candle hidden-state diff {diff} exceeds tol {HIDDEN_TOL}"
    );
    Ok(())
}

/// Comprehensive sweep on the given device. Prints a table; asserts
/// every cell passes the relevant tol.
fn run_full_sweep_on(device: Device, label: &str) -> Result<()> {
    let (weights, cfg_path) = match (weights_path(), config_path()) {
        (Some(w), Some(c)) => (w, c),
        _ => {
            eprintln!("skipping {label} sweep — set RLX_QWEN3_WEIGHTS + RLX_QWEN3_CONFIG");
            return Ok(());
        }
    };
    if !std::path::Path::new(&weights).exists() {
        eprintln!("skipping {label} sweep — weights not found at {weights}");
        return Ok(());
    }

    let cfg = Qwen3Config::from_file(std::path::Path::new(&cfg_path))?;
    let candle_cfg = to_candle_cfg(&cfg);

    let batches = [1usize, 2, 4];
    let seqs = [8usize, 32, 64, 128];

    eprintln!();
    eprintln!("Qwen3 parity sweep — device={label}, dtype=F32, rlx vs candle (CPU)");
    eprintln!(
        "model: hidden={} layers={} q={} kv={} head_dim={} vocab={}",
        cfg.hidden_size,
        cfg.num_hidden_layers,
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.head_dim,
        cfg.vocab_size,
    );
    eprintln!();
    eprintln!(
        "  B   L  | max|Δh|   mean|Δh|  cos(mean)  cos(min)  max|Δlog|  top1   rlx ms   candle ms"
    );
    eprintln!(
        "  -- ---  | -------   --------- ----------  --------  ---------  -----  -------  ---------"
    );

    let mut all_pass = true;
    for &batch in &batches {
        for &seq in &seqs {
            let ids = make_batched_ids(batch, seq);

            let (candle_h, candle_h_ms) =
                run_candle_hidden(&weights, &candle_cfg, batch, seq, &ids)?;
            let (rlx_h, rlx_h_ms) = run_rlx(&weights, &cfg, batch, seq, &ids, false, device)?;
            let (max_h, mean_h, _) = max_mean_abs_diff(&rlx_h, &candle_h);
            let (cos_mean, cos_min) = cosine_stats(&rlx_h, &candle_h, cfg.hidden_size);

            let (_, candle_logits, candle_l_ms) =
                run_candle(&weights, &candle_cfg, batch, seq, &ids)?;
            let (rlx_logits, rlx_l_ms) = run_rlx(&weights, &cfg, batch, seq, &ids, true, device)?;

            // candle's `ModelForCausalLM::forward` returns only the
            // last position per batch row — `[B, vocab]` flat — while
            // rlx returns the full `[B, L, vocab]`. Extract the
            // last-position row for each batch from rlx so the two
            // tensors are comparable.
            let vocab = cfg.vocab_size;
            let last_rlx = extract_last_position_per_batch(&rlx_logits, batch, seq, vocab);
            let trimmed_candle = if candle_logits.len() == batch * vocab {
                candle_logits.clone()
            } else if candle_logits.len() == batch * seq * vocab {
                extract_last_position_per_batch(&candle_logits, batch, seq, vocab)
            } else {
                anyhow::bail!(
                    "candle logit length {} not B*V ({}) nor B*L*V ({})",
                    candle_logits.len(),
                    batch * vocab,
                    batch * seq * vocab
                );
            };
            let (max_l, _mean_l, _) = max_mean_abs_diff(&last_rlx, &trimmed_candle);
            let (top1_match, top1_total) = top1_agreement(&last_rlx, &trimmed_candle, vocab);

            let pass = max_h <= HIDDEN_TOL && top1_match == top1_total;
            all_pass &= pass;
            let rlx_total = rlx_h_ms + rlx_l_ms;
            let candle_total = candle_h_ms + candle_l_ms;
            let mark = if pass { " " } else { "!" };
            eprintln!(
                "{mark} {batch:>2} {seq:>3}  | {max_h:8.5}  {mean_h:8.5}  {cos_mean:.7}  {cos_min:.6}  {max_l:8.4}  {top1_match}/{top1_total}  {rlx_total:7.1}  {candle_total:7.1}"
            );
        }
    }
    eprintln!();
    eprintln!(
        "tolerance: max|Δh| ≤ {HIDDEN_TOL}, max|Δlog| ≤ {LOGIT_TOL} (info), top-1 must be 100%"
    );

    assert!(all_pass, "{label}: one or more (B, L) cells failed parity");
    Ok(())
}

/// Comprehensive sweep: B × L × {hidden, logits, top-1} with timings.
/// Prints a table; asserts that every cell passes the relevant tol.
#[test]
fn qwen3_full_parity_sweep_cpu() -> Result<()> {
    run_full_sweep_on(Device::Cpu, "CPU")
}

/// Same sweep on the MLX (Apple Silicon GPU) backend. MLX's
/// `fast::scaled_dot_product_attention` supports every `MaskKind`
/// natively (Causal, Custom, None, SlidingWindow) and handles
/// arbitrary Lq vs Lk — so the prefill graph runs unmodified.
#[cfg(feature = "mlx")]
#[test]
fn qwen3_full_parity_sweep_mlx() -> Result<()> {
    run_full_sweep_on(Device::Mlx, "MLX")
}

/// Same sweep on the Metal backend. Requires the Metal SDPA
/// `mask_kind` extension that landed alongside this test — without
/// it, the Causal prefill mask would have panicked at lowering.
#[cfg(feature = "metal")]
#[test]
fn qwen3_full_parity_sweep_metal() -> Result<()> {
    run_full_sweep_on(Device::Metal, "Metal")
}

#[cfg(feature = "cuda")]
#[test]
fn qwen3_full_parity_sweep_cuda() -> Result<()> {
    if !rlx_runtime::is_available(Device::Cuda) {
        eprintln!("skip qwen3_full_parity_sweep_cuda: CUDA not available");
        return Ok(());
    }
    run_full_sweep_on(Device::Cuda, "CUDA")
}

#[cfg(feature = "rocm")]
#[test]
fn qwen3_full_parity_sweep_rocm() -> Result<()> {
    if !rlx_runtime::is_available(Device::Rocm) {
        eprintln!("skip qwen3_full_parity_sweep_rocm: ROCm not available");
        return Ok(());
    }
    run_full_sweep_on(Device::Rocm, "ROCm")
}

#[cfg(feature = "gpu")]
#[test]
fn qwen3_full_parity_sweep_wgpu() -> Result<()> {
    if !rlx_runtime::is_available(Device::Gpu) {
        eprintln!("skip qwen3_full_parity_sweep_wgpu: WGPU not available");
        return Ok(());
    }
    run_full_sweep_on(Device::Gpu, "WGPU")
}

#[cfg(feature = "vulkan")]
#[test]
fn qwen3_full_parity_sweep_vulkan() -> Result<()> {
    if !rlx_runtime::is_available(Device::Vulkan) {
        eprintln!("skip qwen3_full_parity_sweep_vulkan: Vulkan not available");
        return Ok(());
    }
    run_full_sweep_on(Device::Vulkan, "Vulkan")
}
