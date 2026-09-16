// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! Drive the full DFlash propose/verify loop and report acceptance.
//!
//! The target here is a **stub**: it wants a fixed arithmetic sequence and
//! emits constant taps. That is enough to show the loop running end to end and
//! to print the stats a real run reports — but the acceptance rate it prints is
//! meaningless, because a stub target agrees with nothing the drafter proposes.
//! Swap [`CallbackTarget`] for a real model to get a number worth reading.
//!
//! What this *does* prove with a real checkpoint: the drafter's graphs compile,
//! the cache bookkeeping stays consistent across rounds, and every round
//! commits at least one token.
//!
//! Usage: `cargo run --release -p rlx-dflash --example dflash_loop -- <gguf> [device] [n]`

use std::cell::Cell;
use std::rc::Rc;

use anyhow::{Context, Result};
use rlx_core::weight_loader::GgufLoader;
use rlx_dflash::{
    CallbackTarget, DflashConfig, DflashDrafter, DflashLoop, DrafterOptions, TargetStep,
};
use rlx_runtime::Device;
use rlx_runtime::spec_decode::SparseDist;

fn one(id: u32) -> SparseDist {
    SparseDist {
        ids: vec![id],
        probs: vec![1.0],
    }
}

fn device_from(name: &str) -> Device {
    match name {
        "metal" => Device::Metal,
        "mlx" => Device::Mlx,
        "cuda" => Device::Cuda,
        "rocm" => Device::Rocm,
        "vulkan" => Device::Vulkan,
        "gpu" | "wgpu" => Device::Gpu,
        _ => Device::Cpu,
    }
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .context("usage: dflash_loop <dflash gguf> [device] [n_tokens]")?;
    let device = device_from(&args.next().unwrap_or_else(|| "cpu".into()));
    let n_tokens: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(64);

    let raw = rlx_gguf::GgufFile::from_path_mmap(&path)
        .or_else(|_| rlx_gguf::GgufFile::from_path(&path))
        .with_context(|| format!("opening {path}"))?;
    let cfg = DflashConfig::from_gguf(&raw)?;
    let tap_dim = cfg.fused_input_dim();
    let block = cfg.block_size;
    println!(
        "drafter: {} layers, block {}, taps {} x {} = {tap_dim}",
        cfg.num_hidden_layers,
        block,
        cfg.target_layers.len(),
        cfg.hidden_size
    );

    let vocab = cfg.vocab_size;
    let hidden = cfg.hidden_size;
    // One bucket: the target's embedding and LM head are uploaded into every
    // decoder graph, so more buckets cost proportionally more memory.
    let ctx = cfg.sliding_window.unwrap_or(1024).min(256);

    let mut loader = GgufLoader::from_file(&path)?;
    let t = std::time::Instant::now();
    let mut drafter = DflashDrafter::new(
        cfg,
        &mut loader,
        DrafterOptions {
            device,
            max_context: ctx,
            min_bucket: ctx,
            ..Default::default()
        },
    )?;
    println!("compiled every graph in {:?}", t.elapsed());

    // An Eagle-style head has no embedding and no LM head; both come from the
    // target. Filling them with a stand-in keeps the plumbing honest about
    // what it is measuring.
    let need: Vec<String> = drafter.missing_shared().to_vec();
    for name in &need {
        println!("  shared with target: {name} ({} values)", vocab * hidden);
        let mut st = 0xbeef_cafe_1234_5678u64;
        let data: Vec<f32> = (0..vocab * hidden)
            .map(|_| {
                st = st.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                (((st >> 40) as f32 / (1u32 << 23) as f32) - 1.0) * 0.02
            })
            .collect();
        drafter.set_shared_param(name, &data)?;
    }

    // Stub target: wants 1, 2, 3, …, advancing only when the loop tells it how
    // much of the speculative batch survived. Shared through a Cell because all
    // three callbacks read the same counter.
    let emitted = Rc::new(Cell::new(0u32));
    let target = CallbackTarget::new(
        tap_dim,
        {
            let e = emitted.clone();
            move |p: &[u32]| {
                e.set(1);
                Ok(TargetStep {
                    dists: vec![one(1)],
                    taps: vec![0.05f32; p.len() * tap_dim],
                })
            }
        },
        {
            let e = emitted.clone();
            move |_anchor: u32, draft: &[u32]| {
                let slots = draft.len() + 1;
                let base = e.get();
                Ok(TargetStep {
                    dists: (0..slots).map(|i| one(base + 1 + i as u32)).collect(),
                    taps: vec![0.05f32; slots * tap_dim],
                })
            }
        },
        {
            let e = emitted.clone();
            move |keep: usize| {
                e.set(e.get() + keep as u32 + 1);
                Ok(())
            }
        },
    );

    let mut l = DflashLoop::new(drafter, target, 0xd_f1a5)?;
    let t = std::time::Instant::now();
    let out = l.generate(&[7, 8, 9], n_tokens, |_| false)?;
    let el = t.elapsed();

    println!("\ngenerated {} tokens in {el:?}", out.len());
    println!("first 12: {:?}", &out[..12.min(out.len())]);
    println!(
        "rounds {} | drafted {} | accepted {} | acceptance {:.3} | tokens/target-forward {:.2}",
        l.stats.steps,
        l.stats.drafted,
        l.stats.accepted,
        l.stats.acceptance_rate(),
        l.stats.tokens_per_target_forward(),
    );
    println!(
        "\nNOTE: the target is a stub, so the acceptance above measures nothing.\n\
         Plug a real model into CallbackTarget for a number worth reading."
    );
    Ok(())
}
