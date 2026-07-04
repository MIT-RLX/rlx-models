//! Per-backend timing + precision matrix for packed Gemma 3 270M.
//!
//!   RLX_GEMMA3_GGUF=/tmp/rlx-weights/gemma-3-270m.gguf \
//!   cargo run -p rlx-gemma --features "<backends>" --release --example gemma_bench
//!
//! Precision = last-token prefill logits vs the CPU reference (cosine + max_abs).
//! Timing    = warmed prefill latency (predict_logits) and greedy decode tok/s.

use anyhow::Result;
use rlx_gemma::GemmaRunner;
use rlx_qwen3::SampleOpts;
use rlx_runtime::{Device, is_available};
use std::path::PathBuf;
use std::time::Instant;

const HF_CHAT_IDS: &[u32] = &[
    2, 105, 2364, 107, 3689, 563, 1156, 2915, 1156, 236881, 25685, 528, 886, 2822, 13315, 236761,
    106, 107, 105, 4368, 107,
];
const DECODE_STEPS: usize = 32;

fn weights() -> PathBuf {
    std::env::var("RLX_GEMMA3_GGUF")
        .expect("set RLX_GEMMA3_GGUF")
        .into()
}

fn build(dev: Device) -> Result<GemmaRunner> {
    Ok(GemmaRunner::builder()
        .weights(&weights())
        .packed_weights(true)
        .device(dev)
        .max_seq(512)
        .sample(SampleOpts::greedy())
        .build()?)
}

fn cos_maxabs(a: &[f32], b: &[f32]) -> (f32, f32) {
    let n = a.len().min(b.len());
    let (mut dot, mut na, mut nb, mut mx) = (0f64, 0f64, 0f64, 0f32);
    for i in 0..n {
        let (x, y) = (a[i] as f64, b[i] as f64);
        dot += x * y;
        na += x * x;
        nb += y * y;
        mx = mx.max((a[i] - b[i]).abs());
    }
    let cos = if na > 0.0 && nb > 0.0 {
        (dot / (na.sqrt() * nb.sqrt())) as f32
    } else {
        0.0
    };
    (cos, mx)
}

struct Row {
    label: &'static str,
    prefill_ms: f64,
    dec_tokens: usize,
    dec_tok_s: f64,
    cos: f32,
    maxabs: f32,
    hcos: f32,
    toks: Vec<u32>,
}

fn bench(dev: Device, label: &'static str, cpu_logits: Option<&[f32]>, cpu_hidden: Option<&[f32]>) -> Result<Row> {
    // Phase 1 — precision + prefill timing (one runner, dropped before phase 2).
    let mut best = f64::INFINITY;
    let mut logits = Vec::new();
    let mut hidden = Vec::new();
    {
        let mut r = build(dev)?;
        let _ = r.predict_logits(HF_CHAT_IDS)?; // warm compile
        for _ in 0..3 {
            let t = Instant::now();
            logits = r.predict_logits(HF_CHAT_IDS)?;
            best = best.min(t.elapsed().as_secs_f64() * 1000.0);
        }
        hidden = r.predict_last_hidden(HF_CHAT_IDS).unwrap_or_default();
    }
    let (cos, maxabs) = match cpu_logits {
        Some(c) => cos_maxabs(c, &logits),
        None => (1.0, 0.0),
    };
    let hcos = match cpu_hidden {
        Some(c) if !hidden.is_empty() => cos_maxabs(c, &hidden).0,
        _ => 1.0,
    };

    // Phase 2 — decode timing (one runner). generate() re-prefills then decodes;
    // subtract the phase-1 prefill latency to isolate the decode span.
    let (dec_ms, toks) = {
        let mut r = build(dev)?;
        // Warm with the SAME step count so every decode bucket the timed run
        // touches is already compiled (NVRTC/shader JIT out of the hot path).
        let _ = r.generate(HF_CHAT_IDS, DECODE_STEPS, |_| {})?;
        let t = Instant::now();
        let toks = r.generate(HF_CHAT_IDS, DECODE_STEPS, |_| {})?;
        let gen_ms = t.elapsed().as_secs_f64() * 1000.0;
        ((gen_ms - best).max(0.001), toks)
    };
    let dec_tok_s = toks.len() as f64 / (dec_ms / 1000.0);

    Ok(Row {
        label,
        prefill_ms: best,
        dec_tokens: toks.len(),
        dec_tok_s,
        cos,
        maxabs,
        hcos,
        toks,
    })
}

fn main() -> Result<()> {
    let candidates: &[(Device, &str)] = &[
        (Device::Cpu, "CPU"),
        (Device::Cuda, "CUDA"),
        (Device::Metal, "Metal"),
        (Device::Mlx, "MLX"),
        (Device::Gpu, "wgpu"),
        (Device::Vulkan, "Vulkan"),
        (Device::Ane, "CoreML"),
        (Device::Rocm, "ROCm"),
    ];

    // CPU reference logits + hidden first.
    let cpu_row = bench(Device::Cpu, "CPU", None, None)?;
    let (cpu_logits, cpu_hidden) = {
        let mut r = build(Device::Cpu)?;
        let l = r.predict_logits(HF_CHAT_IDS)?;
        let h = r.predict_last_hidden(HF_CHAT_IDS).unwrap_or_default();
        (l, h)
    };

    let mut rows = vec![cpu_row];
    for &(dev, label) in candidates.iter().filter(|(d, _)| *d != Device::Cpu) {
        if !is_available(dev) {
            continue;
        }
        // Isolate each backend: a wgpu OOM (etc.) panics — don't lose the matrix.
        let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            bench(dev, label, Some(&cpu_logits), Some(&cpu_hidden))
        }));
        match res {
            Ok(Ok(r)) => rows.push(r),
            Ok(Err(e)) => eprintln!("{label}: FAILED — {e}"),
            Err(_) => eprintln!("{label}: PANIC (likely OOM) — skipped"),
        }
    }

    println!("\n=== Gemma 3 270M — per-backend timing + precision (vs CPU) ===");
    println!("{:<8} {:>11} {:>10} {:>8} {:>11} {:>12} {:>11}", "backend", "prefill_ms", "dec_tok/s", "dec_tok", "hidden_cos", "logit_cos", "logit_maxabs");
    for r in &rows {
        println!(
            "{:<8} {:>11.2} {:>10.1} {:>8} {:>11.6} {:>12.6} {:>11.4}",
            r.label, r.prefill_ms, r.dec_tok_s, r.dec_tokens, r.hcos, r.cos, r.maxabs
        );
    }
    println!("\nfirst greedy tokens (should all equal CPU):");
    for r in &rows {
        println!("  {:<8} {:?}", r.label, &r.toks[..r.toks.len().min(8)]);
    }
    Ok(())
}
