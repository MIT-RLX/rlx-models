// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! Token-level KV-decode parity: CPU vs native Vulkan, bucket decode.
//!
//! Isolates the LM (no SNAC) so a divergence shows up as the exact decode step
//! and token id where Vulkan drifts from the CPU reference. Greedy decode is
//! deterministic, so ANY valid prompt must produce identical token streams on a
//! correct backend. Skips when no GGUF (`LLAMA32_GGUF`/`ORPHEUS_GGUF_PATH`) or
//! no Vulkan device is present.
//!
//! ```bash
//! ORPHEUS_GGUF_PATH=/tmp/rlx-weights/orpheus/orpheus-3b-0.1-ft-Q4_K_M.gguf \
//!   cargo test -p rlx-llama32 --features vulkan --release \
//!   --test vulkan_kv_parity -- --nocapture
//! ```

use rlx_core::weight_loader::GgufLoader;
use rlx_llama32::{Llama32Generator, MetalGgufPrefillMode, llama32_cfg_from_gguf};
use rlx_qwen3::sampling::SampleOpts;
use rlx_runtime::{Device, is_available};

fn gguf_path() -> Option<String> {
    for k in ["LLAMA32_GGUF", "ORPHEUS_GGUF_PATH"] {
        if let Ok(p) = std::env::var(k) {
            if std::path::Path::new(&p).is_file() {
                return Some(p);
            }
        }
    }
    let def = "/tmp/rlx-weights/orpheus/orpheus-3b-0.1-ft-Q4_K_M.gguf";
    std::path::Path::new(def).is_file().then(|| def.to_string())
}

/// Greedy-decode `n` tokens on `device` from `prompt`, bucket decode.
fn gen_tokens(device: Device, gguf: &str, prompt: &[u32], n: usize) -> anyhow::Result<Vec<u32>> {
    let mut loader = GgufLoader::from_file(gguf)?;
    let cfg = llama32_cfg_from_gguf(loader.file())?;
    let path = std::path::Path::new(gguf);
    let mut g = Llama32Generator::from_loader_at_mode(
        cfg,
        &mut loader,
        device,
        path,
        MetalGgufPrefillMode::CpuF32,
    )?
    .with_compile_seq_cap(96)
    .with_decode_cache(96);
    g.prefill(prompt);
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        out.push(g.step_cached(SampleOpts::greedy())?);
    }
    Ok(out)
}

#[test]
fn vulkan_kv_decode_matches_cpu_token_level() -> anyhow::Result<()> {
    let Some(gguf) = gguf_path() else {
        eprintln!("skip: no GGUF (set ORPHEUS_GGUF_PATH)");
        return Ok(());
    };
    if !is_available(Device::Vulkan) {
        eprintln!("skip: no Vulkan device");
        return Ok(());
    }
    // Native on-device Vulkan decode (else it falls back to CPU and parity is trivial).
    unsafe { std::env::set_var("ORPHEUS_VULKAN_NATIVE", "1") };

    // Arbitrary but fixed valid prompt — greedy is deterministic, so a correct
    // backend must match CPU regardless of the prompt's meaning. ~15 tokens to
    // mirror the Orpheus bench prompt length.
    let prompt: Vec<u32> = (0..15).map(|i| (i * 211 + 3) as u32).collect();
    let n = 8usize;

    let cpu = gen_tokens(Device::Cpu, &gguf, &prompt, n)?;
    let vk = gen_tokens(Device::Vulkan, &gguf, &prompt, n)?;
    eprintln!("CPU    tokens: {cpu:?}");
    eprintln!("Vulkan tokens: {vk:?}");

    let diverge = cpu.iter().zip(vk.iter()).position(|(a, b)| a != b);
    match diverge {
        None if cpu == vk => {
            eprintln!("[rlx-llama32] Vulkan KV decode matches CPU for all {n} tokens")
        }
        _ => {}
    }
    assert_eq!(
        cpu, vk,
        "Vulkan KV decode diverges from CPU at step {:?} (CPU={cpu:?}, Vulkan={vk:?})",
        diverge
    );
    Ok(())
}
