// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Run a Qwen3 model from a llama.cpp `.gguf` file (Q4_K_M / Q5_K_M /
// Q6_K supported out of the box). The runner auto-detects the
// format, pulls the architecture config from GGUF metadata, and
// hides any MTP heads from non-MTP builders.
//
// Usage:
//   cargo run --release -p rlx-models --features metal \
//       --example run_qwen3_gguf -- /path/to/Qwen3-X-Q4_K_M.gguf
//
// Memory budget:
//   ~16 GB unified is enough for Qwen3-8B Q4_K_M at F32-dequant.
//   For 27B+, wait for Op::DequantMatMul Metal lowering (the IR is
//   already there; the kernel is still TBD).
//
// Equivalent CLI call:
//   rlx-qwen3 (or `rlx-run qwen3`) --weights model.gguf --device metal \
//                 --prompt-ids 1,17,42 --max-tokens 16

use anyhow::{Result, anyhow};
use rlx_models::run::{Qwen3Runner, list_mtp_keys};
use rlx_runtime::Device;
use std::path::PathBuf;

fn main() -> Result<()> {
    let path: PathBuf = std::env::args()
        .nth(1)
        .ok_or_else(|| anyhow!("usage: run_qwen3_gguf <file.gguf>"))?
        .into();

    // Pre-flight: surface any MTP heads before they're silently
    // dropped (they're not used by the base inference path today).
    let mtp = list_mtp_keys(&path)?;
    if !mtp.is_empty() {
        eprintln!(
            "[qwen3-gguf] note: {} MTP heads detected and will be ignored: {:?}",
            mtp.len(),
            &mtp[..mtp.len().min(3)]
        );
    }

    let mut runner = Qwen3Runner::builder()
        .weights(&path)
        .device(Device::Metal)
        .max_seq(128)
        .stream(true)
        .max_memory_gb(16.0)
        .build()?;
    let cfg = runner.config();
    eprintln!(
        "[qwen3-gguf] compiled — vocab={} hidden={} layers={} on {:?}",
        cfg.vocab_size,
        cfg.hidden_size,
        cfg.num_hidden_layers,
        runner.device(),
    );

    let prompt = [1u32, 17, 42, 314, 2718, 9001, 27182, 8128];
    eprintln!("[qwen3-gguf] prompt ids: {prompt:?}");

    let t0 = std::time::Instant::now();
    runner.generate(&prompt, 16, |tok| eprint!(" {tok}"))?;
    eprintln!();
    eprintln!("[qwen3-gguf] finished in {:?}", t0.elapsed());
    Ok(())
}
