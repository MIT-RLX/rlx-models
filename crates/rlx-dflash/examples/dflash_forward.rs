// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! Load the real DFlash drafter and run one forward over synthetic target taps.
//!
//! Validates config parse, tensor-name coverage, packed load and the graph
//! itself against the shipped `dflash-kquant.gguf`. Real taps require the target
//! model; this feeds deterministic pseudo-random hidden states of the right
//! shape, which is enough to prove the drafter builds, loads every weight, and
//! produces finite, non-degenerate output.
//!
//! Usage: `cargo run --release -p rlx-dflash --example dflash_forward -- <gguf> [device]`

use std::collections::HashMap;

use anyhow::{Context, Result};
use rlx_core::weight_loader::GgufLoader;
use rlx_dflash::{DflashConfig, build_dflash_graph};
use rlx_ir::DType;
use rlx_runtime::{Device, Session};

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
        .context("usage: dflash_forward <dflash gguf> [device]")?;
    let device = device_from(&args.next().unwrap_or_else(|| "cpu".into()));

    let raw = rlx_gguf::GgufFile::from_path_mmap(&path)
        .or_else(|_| rlx_gguf::GgufFile::from_path(&path))
        .with_context(|| format!("opening {path}"))?;
    let cfg = DflashConfig::from_gguf(&raw)?;
    println!(
        "dflash: layers={} hidden={} ffn={} heads={}/{} head_dim={}",
        cfg.num_hidden_layers,
        cfg.hidden_size,
        cfg.intermediate_size,
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.head_dim
    );
    println!(
        "block_size={} target_layers={:?} sliding_window={:?}",
        cfg.block_size, cfg.target_layers, cfg.sliding_window
    );
    println!(
        "fc fuses {} taps x {} = {}",
        cfg.target_layers.len(),
        cfg.hidden_size,
        cfg.fused_input_dim()
    );

    // Draft a full block at once — that is how it runs in production.
    let seq = cfg.block_size;
    let mut loader = GgufLoader::from_file(&path)?;
    let mut packed = HashMap::new();
    let (graph, params) = build_dflash_graph(&cfg, &mut loader, 1, seq, &mut packed)?;
    println!(
        "\ngraph: {} nodes, {} packed tensors",
        graph.nodes().len(),
        packed.len()
    );

    let mut compiled = Session::new(device).compile(graph);
    for (name, data) in params {
        compiled.set_param(&name, &data);
    }
    for name in packed.keys() {
        let bytes = loader
            .tensor_bytes_borrowed(name)
            .with_context(|| format!("packed bytes for {name}"))?;
        compiled.set_param_typed(name, bytes, DType::U8);
    }

    // Deterministic pseudo-random taps, O(1) magnitude like real residuals.
    let n = seq * cfg.fused_input_dim();
    let mut st = 0x243f_6a88_85a3_08d3u64;
    let taps: Vec<f32> = (0..n)
        .map(|_| {
            st = st.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            (((st >> 33) as f64 / (1u64 << 31) as f64) - 1.0) as f32 * 0.5
        })
        .collect();

    let t = std::time::Instant::now();
    let out = compiled.run(&[("dflash_taps", taps.as_slice())]);
    let el = t.elapsed();

    let h = &out[0];
    anyhow::ensure!(
        h.len() == seq * cfg.hidden_size,
        "expected [{seq}, {}] hidden, got {}",
        cfg.hidden_size,
        h.len()
    );
    anyhow::ensure!(h.iter().all(|v| v.is_finite()), "non-finite draft hidden");
    let mean = h.iter().sum::<f32>() / h.len() as f32;
    let var = h.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / h.len() as f32;
    anyhow::ensure!(
        var > 1e-8,
        "draft hidden is constant — weights not applied?"
    );

    println!(
        "\nforward: {seq} draft positions in {el:?} ({:.1} pos/s)",
        seq as f64 / el.as_secs_f64()
    );
    println!(
        "hidden[{seq}x{}]: mean {mean:.5} var {var:.5}",
        cfg.hidden_size
    );
    println!("first 6: {:?}", &h[..6.min(h.len())]);
    println!("\nOK — drafter loads and runs; needs target taps for real proposals");
    Ok(())
}
