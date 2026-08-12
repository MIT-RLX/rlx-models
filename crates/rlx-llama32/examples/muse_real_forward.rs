// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! Real-weight forward pass for a packed GGUF checkpoint, bypassing
//! `Llama32Generator` entirely.
//!
//! The generator keeps several compiled graphs alive at once, and each owns a
//! contiguous arena holding a full copy of the weights — which OOMs a 30B-class
//! checkpoint on a 64 GB box. A single prefill graph is fine (~2× the packed
//! size), so this drives one directly: build → compile → upload → run, then read
//! the last position's logits.
//!
//! That is enough to validate an architecture against REAL weights: a wrong
//! block wiring (a dropped attention gate, an unapplied sliding window, a
//! swapped norm) does not produce plausible next-token predictions.
//!
//! Usage:
//!   cargo run --release -p rlx-llama32 --features tokenizer \
//!     --example muse_real_forward -- <gguf> ["prompt text"] [device]

use std::collections::HashMap;

use anyhow::{Context, Result};
use rlx_core::weight_loader::GgufLoader;
use rlx_ir::DType;
use rlx_llama32::builder::build_llama32_graph_sized_packed;
use rlx_llama32::config::llama32_cfg_from_gguf;
use rlx_runtime::{Device, Session};

fn device_from(name: &str) -> Device {
    match name {
        "metal" => Device::Metal,
        "mlx" => Device::Mlx,
        "cuda" => Device::Cuda,
        "rocm" => Device::Rocm,
        "vulkan" => Device::Vulkan,
        "gpu" | "wgpu" => Device::Gpu,
        "coreml" | "ane" => Device::Ane,
        _ => Device::Cpu,
    }
}

/// GGUF `tokenizer.ggml.tokens` as a plain id→piece table, for decoding without
/// pulling in a full tokenizer. GPT-2 byte-level pieces use `Ġ` for space.
fn vocab_pieces(raw: &rlx_gguf::GgufFile) -> Vec<String> {
    match raw.metadata.get("tokenizer.ggml.tokens") {
        Some(rlx_gguf::MetaValue::Array(a)) => a
            .iter()
            .map(|v| v.as_str().unwrap_or("<?>").to_string())
            .collect(),
        _ => Vec::new(),
    }
}

fn show(piece: &str) -> String {
    piece.replace('Ġ', " ").replace('Ċ', "\\n")
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .context("usage: muse_real_forward <gguf> [prompt] [device]")?;
    let prompt = args
        .next()
        .unwrap_or_else(|| "The capital of France is".to_string());
    let device = device_from(&args.next().unwrap_or_else(|| "cpu".into()));

    let raw = rlx_gguf::GgufFile::from_path_mmap(&path)
        .or_else(|_| rlx_gguf::GgufFile::from_path(&path))
        .with_context(|| format!("opening {path}"))?;
    let cfg = llama32_cfg_from_gguf(&raw)?;
    let pieces = vocab_pieces(&raw);
    println!(
        "arch={} layers={} hidden={} heads={}/{} head_dim={} vocab={}",
        cfg.gguf_arch.as_deref().unwrap_or("?"),
        cfg.num_hidden_layers,
        cfg.hidden_size,
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.head_dim(),
        cfg.vocab_size
    );
    println!(
        "swa: window={:?} pattern={:?} | softcap={:?} logit_scale={:?}",
        cfg.sliding_window, cfg.sliding_window_pattern, cfg.final_logit_softcap, cfg.logit_scale
    );

    // Tokenize with the GGUF's own byte-level BPE (no sidecar tokenizer.json).
    let ids = rlx_llama32::encode_prompt_auto(std::path::Path::new(&path), None, &prompt)
        .context("tokenizing prompt from GGUF vocab")?;
    println!("\nprompt: {prompt:?}\nids: {ids:?} ({} tokens)", ids.len());
    anyhow::ensure!(!ids.is_empty(), "empty prompt after tokenization");

    let seq = ids.len();
    let mut loader = GgufLoader::from_file(&path)?;
    let mut packed = HashMap::new();
    let mut embed_host = None;
    let (graph, params) = build_llama32_graph_sized_packed(
        &cfg,
        &mut loader,
        1,
        seq,
        /*with_lm_head*/ std::env::var("RLX_NO_LM_HEAD").is_err(),
        /*last_logits_only*/ false,
        /*with_kv_outputs*/ false,
        &mut packed,
        &mut embed_host,
    )?;

    let t_build = std::time::Instant::now();
    eprintln!("[stage] graph built");
    let mut compiled = Session::new(device).compile(graph);
    eprintln!("[stage] compiled  ({:?})", t_build.elapsed());
    let t_up = std::time::Instant::now();
    for (name, data) in params {
        compiled.set_param(&name, &data);
    }
    for name in packed.keys() {
        let bytes = loader
            .tensor_bytes_borrowed(name)
            .with_context(|| format!("packed bytes for {name}"))?;
        compiled.set_param_typed(name, bytes, DType::U8);
    }

    eprintln!("[stage] weights uploaded ({:?})", t_up.elapsed());
    // The packed embed is gathered host-side and fed as `input_embeddings`.
    let ids_f32: Vec<f32> = ids.iter().map(|&t| t as f32).collect();
    let out = if let Some((bytes, scheme)) = embed_host.as_ref() {
        let mut embeds = vec![0f32; seq * cfg.hidden_size];
        rlx_llama32::builder::gather_embed_rows(
            bytes,
            *scheme,
            cfg.hidden_size,
            &ids_f32,
            &mut embeds,
        )?;
        eprintln!("[stage] running forward");
        // Repeat the SAME forward: pass 1 is a cold dequant cache, later passes
        // hit it. If the model's f32 footprint exceeds the cache budget the
        // later passes do not speed up — that is LRU thrash, and it is the
        // decode-time cost too.
        let passes: usize = std::env::var("RLX_FWD_PASSES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1);
        let mut r = Vec::new();
        for i in 0..passes.max(1) {
            let t_run = std::time::Instant::now();
            r = compiled.run(&[("input_embeddings", embeds.as_slice())]);
            let el = t_run.elapsed();
            eprintln!(
                "[PREFILL pass {}] {seq} tok in {:?}  =>  {:.2} tok/s",
                i + 1,
                el,
                seq as f64 / el.as_secs_f64()
            );
        }
        r
    } else {
        compiled.run(&[("input_ids", ids_f32.as_slice())])
    };

    let logits = &out[0];
    if std::env::var("RLX_NO_LM_HEAD").is_ok() {
        eprintln!("[no-lm-head] output len {}", logits.len());
        return Ok(());
    }
    anyhow::ensure!(
        logits.len() == seq * cfg.vocab_size,
        "logits len {} != seq {seq} * vocab {}",
        logits.len(),
        cfg.vocab_size
    );
    let last = &logits[(seq - 1) * cfg.vocab_size..];
    anyhow::ensure!(
        last.iter().all(|v| v.is_finite()),
        "non-finite logits — arch or weights wrong"
    );

    let mut order: Vec<usize> = (0..last.len()).collect();
    order.sort_by(|&a, &b| last[b].partial_cmp(&last[a]).unwrap());
    println!("\ntop-10 next-token predictions:");
    for (rank, &t) in order.iter().take(10).enumerate() {
        let piece = pieces.get(t).map_or("<oov>".to_string(), |p| show(p));
        println!("  {:>2}. {:>8.4}  id={t:<7} {piece:?}", rank + 1, last[t]);
    }

    // The softcap has to bind: `logits = cap * tanh(logits / cap)` means nothing
    // may exceed the cap. A miswired final stage shows up here immediately.
    if let Some(cap) = cfg.final_logit_softcap {
        let max = last.iter().cloned().fold(f32::MIN, f32::max);
        let min = last.iter().cloned().fold(f32::MAX, f32::min);
        println!("\nlogit range [{min:.4}, {max:.4}] vs softcap ±{cap}");
        anyhow::ensure!(max <= cap + 1e-3 && min >= -cap - 1e-3, "softcap violated");
        println!("softcap holds");
    }
    Ok(())
}
