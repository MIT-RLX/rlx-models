// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! Report what the packed GGUF prefill graph actually asks the arena for.
//!
//! Answers "why does a 15.9 GB checkpoint need 56 GB of RAM?" by separating the
//! two candidate causes:
//!   * weights entering the graph **F32-expanded** (param nodes typed F32 with
//!     `num_elements * 4` bytes), vs
//!   * weights staying **packed** (1-D `U8` params sized by packed byte length)
//!     and the blow-up living downstream in the backend / plan.
//!
//! Usage: `cargo run --release -p rlx-llama32 --example arena_report -- <gguf> [seq]`

use std::collections::HashMap;

use anyhow::{Context, Result};
use rlx_core::weight_loader::GgufLoader;
use rlx_ir::DType;
use rlx_llama32::builder::build_llama32_graph_sized_packed;
use rlx_llama32::config::llama32_cfg_from_gguf;

fn gb(bytes: u128) -> f64 {
    bytes as f64 / 1_073_741_824.0
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let path = args.next().context("usage: arena_report <gguf> [seq]")?;
    let seq: usize = args.next().map_or(Ok(128), |s| s.parse())?;

    let raw = rlx_gguf::GgufFile::from_path_mmap(&path)
        .or_else(|_| rlx_gguf::GgufFile::from_path(&path))
        .with_context(|| format!("opening {path}"))?;
    let cfg = llama32_cfg_from_gguf(&raw)?;
    let file_bytes: u128 = std::fs::metadata(&path)?.len() as u128;
    println!(
        "arch={:?} layers={} hidden={} vocab={} file={:.2} GB",
        cfg.gguf_arch.as_deref().unwrap_or("?"),
        cfg.num_hidden_layers,
        cfg.hidden_size,
        cfg.vocab_size,
        gb(file_bytes)
    );

    let mut loader = GgufLoader::from_file(&path)?;
    let mut packed = HashMap::new();
    let mut embed_host = None;
    let (graph, params) = build_llama32_graph_sized_packed(
        &cfg,
        &mut loader,
        1,
        seq,
        /*with_lm_head*/ true,
        /*last_logits_only*/ false,
        /*with_kv_outputs*/ false,
        &mut packed,
        &mut embed_host,
    )?;

    // Split every Param node by dtype: U8 params are packed weights (good),
    // F32 params of weight size are dequantized copies (the failure mode).
    let mut packed_bytes: u128 = 0;
    let mut f32_param_bytes: u128 = 0;
    let mut f32_param_count = 0usize;
    let mut packed_count = 0usize;
    let mut biggest: Vec<(String, DType, u128)> = Vec::new();
    for node in graph.nodes().iter() {
        let rlx_ir::Op::Param { name } = &node.op else {
            continue;
        };
        let elems = node.shape.num_elements().unwrap_or(0) as u128;
        let dt = node.shape.dtype();
        let bytes = node.shape.size_bytes().unwrap_or(0) as u128;
        match dt {
            DType::U8 => {
                packed_bytes += bytes;
                packed_count += 1;
            }
            DType::F32 => {
                f32_param_bytes += elems * 4;
                f32_param_count += 1;
            }
            _ => {}
        }
        biggest.push((name.clone(), dt, bytes));
    }
    biggest.sort_by_key(|(_, _, b)| std::cmp::Reverse(*b));

    let host_params: u128 = params.values().map(|v| (v.len() * 4) as u128).sum();
    println!(
        "\nparam nodes: {packed_count} packed U8 ({:.2} GB) | {f32_param_count} F32 ({:.2} GB)",
        gb(packed_bytes),
        gb(f32_param_bytes)
    );
    println!(
        "host `params` map (uploaded then dropped): {:.2} GB",
        gb(host_params)
    );
    println!(
        "packed side-table entries: {} | graph nodes: {}",
        packed.len(),
        graph.nodes().len()
    );
    println!(
        "\nverdict: {}",
        if f32_param_bytes > packed_bytes {
            "WEIGHTS ARE F32-EXPANDED in the graph — builder-side problem"
        } else {
            "weights stay packed in the graph — blow-up is downstream (plan/backend)"
        }
    );

    println!("\nlargest param slots:");
    for (name, dt, bytes) in biggest.iter().take(12) {
        println!("  {:>9.3} GB  {dt:?}  {name}", gb(*bytes));
    }

    // The arena plan itself — the number every RSS claim above is really about.
    // Compare the three width policies AND the dequant pin, so it is obvious
    // which knob (if any) actually moves the reservation.
    let a = rlx_opt::memory::plan_memory_native(&graph, 64).arena_size as u128;
    let b = rlx_opt::memory::plan_memory_native_in_order(&graph, 64).arena_size as u128;
    let c = rlx_opt::memory::plan_memory_hybrid(&graph, 64).arena_size as u128;
    let d = rlx_opt::memory::plan_memory_f32_uniform(&graph, 64).arena_size as u128;
    println!("\narena plan (packed params = {:.2} GB):", gb(packed_bytes));
    println!("  native                : {:.2} GB", gb(a));
    println!("  native_in_order       : {:.2} GB  <- CPU backend", gb(b));
    println!("  hybrid (f16 acts)     : {:.2} GB", gb(c));
    println!("  f32_uniform           : {:.2} GB", gb(d));
    println!(
        "  activations above weights: native {:.2} GB | in_order {:.2} GB",
        gb(a.saturating_sub(packed_bytes)),
        gb(b.saturating_sub(packed_bytes))
    );

    // Stage the RSS so the blow-up can be attributed: graph build vs backend
    // compile (arena reservation) vs packed upload. `ps` keeps this dependency-
    // free and works on both macOS and Linux.
    println!("\nRSS by stage:");
    println!("  after graph build : {:.2} GB", rss_gb());
    let session = rlx_runtime::Session::new(rlx_runtime::Device::Cpu);
    let mut compiled = session.compile(graph);
    println!("  after CPU compile : {:.2} GB", rss_gb());
    for (name, data) in params {
        compiled.set_param(&name, &data);
    }
    println!("  after F32 params  : {:.2} GB", rss_gb());
    for name in packed.keys() {
        if let Some(bytes) = loader.tensor_bytes_borrowed(name) {
            compiled.set_param_typed(name, bytes, DType::U8);
        }
    }
    println!("  after packed upload: {:.2} GB", rss_gb());
    Ok(())
}

fn rss_gb() -> f64 {
    let pid = std::process::id();
    std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &pid.to_string()])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.trim().parse::<f64>().ok())
        .map_or(f64::NAN, |kb| kb / 1_048_576.0)
}
