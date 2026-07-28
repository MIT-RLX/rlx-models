// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Cross-backend parity sweep: run the same GGUF + prompt on every available
//! rlx backend (CPU / Metal / MLX / wgpu-Gpu / CUDA / ROCm) and report the
//! decoded text + how far greedy decoding agrees with the CPU reference.
//!
//! ```sh
//! cargo run --example backend_sweep --features runner,metal,mlx,gpu -- \
//!     --weights /path/to/Qwen3-0.6B-Q4_K_M.gguf \
//!     --prompt "What is the capital of France? Answer in one sentence." \
//!     --n-new 24
//! ```

use anyhow::{Result, bail};
use rlx_models::run::{
    ChatMessage, LmRunner, Qwen3Runner, auto_chat_template, auto_detokenize, auto_tokenize,
};
use rlx_runtime::Device;
use rlx_runtime::device_ext::is_available;
use std::path::PathBuf;

fn main() -> Result<()> {
    let mut weights: Option<PathBuf> = None;
    let mut prompt: Option<String> = None;
    let mut n_new = 24usize;
    let mut only: Option<String> = None;
    let mut naive = false;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--weights" => weights = it.next().map(PathBuf::from),
            "--prompt" => prompt = it.next(),
            "--n-new" => n_new = it.next().and_then(|s| s.parse().ok()).unwrap_or(24),
            // Restrict the sweep to one backend, e.g. `--only gpu` / `--only metal`.
            "--only" => only = it.next().map(|s| s.to_lowercase()),
            // Force the naive (re-prefill, no KV-cache decode) path.
            "--naive" => naive = true,
            other => bail!("unknown arg: {other}"),
        }
    }
    let weights = weights.expect("--weights required");
    let prompt = prompt.expect("--prompt required");

    // Device-independent: render chat template (falls back to the raw prompt for
    // weight formats without embedded template metadata) + tokenize once.
    let text_in = match auto_chat_template(&weights) {
        Ok(t) => t
            .render(&[ChatMessage::user(prompt.clone())], true)
            .unwrap_or(prompt.clone()),
        Err(e) => {
            eprintln!("(no chat template: {e}; using raw prompt)");
            prompt.clone()
        }
    };
    let prompt_ids = auto_tokenize(&weights, &text_in, None)?;
    eprintln!(
        "{} prompt tokens; generating {n_new} greedily\n",
        prompt_ids.len()
    );

    let candidates = [
        Device::Cpu,
        Device::Metal,
        Device::Mlx,
        Device::Gpu,
        Device::Cuda,
        Device::Rocm,
    ];
    let mut ref_ids: Option<Vec<u32>> = None;

    for dev in candidates {
        if let Some(o) = &only {
            if !format!("{dev:?}").to_lowercase().contains(o.as_str()) {
                continue;
            }
        }
        // CPU is always present; skip accelerators that don't probe available.
        if dev != Device::Cpu && !is_available(dev) {
            continue;
        }
        eprint!("── {dev:?}: ");
        let build = Qwen3Runner::builder()
            .weights(&weights)
            .device(dev)
            .max_seq(256)
            .build();
        let mut runner = match build {
            Ok(r) => r,
            Err(e) => {
                eprintln!("BUILD FAILED — {e}");
                continue;
            }
        };
        if naive {
            runner.disable_decode_compile_cache();
        }
        // Prefill logits fingerprint — isolates prefill correctness from decode.
        if let Ok(logits) = LmRunner::predict_logits(&mut runner, &prompt_ids) {
            let mut argmax = 0usize;
            let (mut mx, mut sumsq, mut nonfinite) = (f32::NEG_INFINITY, 0f64, 0usize);
            for (i, &v) in logits.iter().enumerate() {
                if !v.is_finite() {
                    nonfinite += 1;
                }
                if v > logits[argmax] {
                    argmax = i;
                }
                mx = mx.max(v.abs());
                sumsq += (v as f64) * (v as f64);
            }
            eprint!(
                "prefill argmax={argmax} max|logit|={mx:.2} l2={:.1} nonfinite={nonfinite} | ",
                sumsq.sqrt()
            );
        }
        let mut ids = Vec::new();
        let gen_res = LmRunner::generate(&mut runner, &prompt_ids, n_new, &mut |t| {
            ids.push(t);
            true
        });
        if let Err(e) = gen_res {
            eprintln!("GENERATE FAILED — {e}");
            continue;
        }
        let text = auto_detokenize(&weights, &ids, None, true).unwrap_or_default();
        let one_line = text.replace('\n', " ⏎ ");
        let parity = match &ref_ids {
            None => {
                ref_ids = Some(ids.clone());
                "reference".to_string()
            }
            Some(r) => {
                let n = ids.iter().zip(r).take_while(|(a, b)| a == b).count();
                format!("{n}/{} greedy tokens match CPU", r.len())
            }
        };
        eprintln!("[{parity}]\n     \"{}\"", one_line.trim());
    }
    Ok(())
}
