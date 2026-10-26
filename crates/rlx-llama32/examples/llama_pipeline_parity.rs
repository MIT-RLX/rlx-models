// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! Where do llama's decode paths stop agreeing, on real weights?
//!
//! There are three ways to generate from this family, and a disagreement
//! between any two is a bug in one of them. Comparing them in one process
//! over the same weights is the only way to tell which:
//!
//!   1. `generate_cached` — the generator's own loop, sampling internally.
//!   2. `predict_logits` + `decode_get_logits` — the host-driven pair, where
//!      the caller samples. This is what a server uses, because temperature,
//!      stop strings and log-probs all live above the model.
//!   3. The pipeline stage across several worlds — one rank holding
//!      everything, the layers split in half, and worlds that do *not*
//!      divide the layer count, where some ranks hold one layer more than
//!      others.
//!
//! The uneven worlds are run twice: once letting the stage split the
//! layers itself, and once handing it an explicit `LayerSplit` — the two
//! ways a caller can divide a model, which must name the same layers.
//! They disagreed about where the remainder goes until this example was
//! pointed at it, and a pipeline whose ranks disagree still loads, still
//! runs, and simply answers wrongly.
//!
//! Every run is greedy, so any difference is real and not sampling.
//!
//! ```text
//! cargo run --release -p rlx-llama32 --example llama_pipeline_parity -- \
//!     --model /path/to/model-dir --prompt-ids 504,5929,282,8043,314 --tokens 6
//! ```

use anyhow::{Context, Result, anyhow};
use rlx_core::weight_map::WeightMap;
use rlx_distributed::{LayerSplit, NetTransport, PipelineCoordinator, ProcessGroup};
use rlx_llama32::config::Llama32Config;
use rlx_llama32::generator::Llama32Generator;
use rlx_llama32::pipeline_block::{stage, stage_with_layers};
use rlx_runtime::Device;
use rlx_runtime::lm::LmRunner;
use std::collections::HashMap;
use std::net::{Ipv4Addr, SocketAddr, TcpListener};
use std::sync::Arc;
use std::thread;

type Tensors = HashMap<String, (Vec<f32>, Vec<usize>)>;

fn argmax(v: &[f32]) -> u32 {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i as u32)
        .unwrap()
}

fn load(dir: &str) -> Result<(Llama32Config, Tensors)> {
    let d = std::path::Path::new(dir);
    let cfg = Llama32Config::from_file(&d.join("config.json"))
        .with_context(|| format!("reading {dir}/config.json"))?;
    let mut files: Vec<std::path::PathBuf> = std::fs::read_dir(d)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "safetensors"))
        .collect();
    files.sort();
    let mut w = Tensors::new();
    for f in files {
        let mut loader = rlx_core::weight_loader::load_from_path(
            f.to_str().ok_or_else(|| anyhow!("non-UTF-8 path"))?,
        )?;
        for k in loader.remaining_keys() {
            let canonical =
                rlx_core::weight_loader::gguf_to_hf_name(&k).unwrap_or_else(|| k.clone());
            let t = loader.take(&k)?;
            w.insert(canonical, t);
        }
    }
    Ok((cfg, w))
}

/// (1) The generator's own greedy loop.
fn internal(c: &Llama32Config, w: &Tensors, prompt: &[u32], n: usize) -> Result<Vec<u32>> {
    let mut wm = WeightMap::from_tensors(w.clone());
    let mut g = Llama32Generator::from_loader(c.clone(), &mut wm, Device::Cpu)?;
    g.prefill(prompt);
    g.generate_cached(n, rlx_qwen3::sampling::SampleOpts::greedy())
}

/// (2) The host-driven pair, sampled by the caller — what a server does.
fn host_driven(c: &Llama32Config, w: &Tensors, prompt: &[u32], n: usize) -> Result<Vec<u32>> {
    let mut wm = WeightMap::from_tensors(w.clone());
    let mut g = Llama32Generator::from_loader(c.clone(), &mut wm, Device::Cpu)?;
    let mut logits = g.prefill_get_last_logits(prompt)?;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let tok = argmax(&logits);
        out.push(tok);
        logits = g.decode_get_logits(tok)?;
    }
    Ok(out)
}

/// (2b) The same pair through `Llama32Runner`'s `LmRunner` impl — the exact
/// path a server takes.
///
/// Separate from (2) because the trait impl is its own wiring, and a mistake
/// there would otherwise only show up in a running server.
fn through_the_trait(weights: &str, prompt: &[u32], n: usize) -> Result<Vec<u32>> {
    let mut r = rlx_llama32::Llama32Runner::builder()
        .weights(weights)
        .device(Device::Cpu)
        .build()?;
    let mut logits = LmRunner::prefill_logits(&mut r, prompt)?;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let tok = argmax(&logits);
        out.push(tok);
        logits = LmRunner::decode_logits(&mut r, tok)?;
    }
    Ok(out)
}

/// (2c) Re-prefilling the whole context each step — what
/// `rlx_runtime::HostDriven` falls back to for a family with no cached
/// pair.
///
/// Here as a control. On qwen3.5 this disagrees with the cached decode by
/// 11-14 in the logits from the first decode step, and the open question is
/// which side is wrong. Running it on a family whose cached decode is
/// already known to agree with its own generator answers that: if it
/// agrees here, re-prefill is sound and qwen3.5 has a real bug; if it
/// disagrees here too, re-prefilling a lengthened sequence is not
/// equivalent in this codebase and the qwen3.5 result says nothing about
/// qwen3.5.
fn re_prefill(c: &Llama32Config, w: &Tensors, prompt: &[u32], n: usize) -> Result<Vec<u32>> {
    let mut wm = WeightMap::from_tensors(w.clone());
    let mut g = Llama32Generator::from_loader(c.clone(), &mut wm, Device::Cpu)?;
    let mut context = prompt.to_vec();
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let logits = g.prefill_get_last_logits(&context)?;
        let tok = argmax(&logits);
        out.push(tok);
        context.push(tok);
    }
    Ok(out)
}

/// (3) The pipeline stage across `world` ranks on loopback.
fn pipelined(
    c: &Llama32Config,
    w: &Tensors,
    prompt: &[u32],
    n: usize,
    world: u32,
    by_split: bool,
) -> Result<Vec<u32>> {
    let listeners: Vec<TcpListener> = (0..world)
        .map(|_| TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap())
        .collect();
    let addrs: Vec<SocketAddr> = listeners.iter().map(|l| l.local_addr().unwrap()).collect();
    let (c, w, prompt) = (
        Arc::new(c.clone()),
        Arc::new(w.clone()),
        Arc::new(prompt.to_vec()),
    );

    let handles: Vec<_> = listeners
        .into_iter()
        .enumerate()
        .map(|(rank, listener)| {
            let (addrs, c, w, prompt) = (addrs.clone(), c.clone(), w.clone(), prompt.clone());
            thread::spawn(move || -> Result<Vec<u32>> {
                let t = NetTransport::from_listener(rank as u32, world, listener, addrs, 64 << 20)?;
                let mut st = if by_split {
                    // The division a caller makes for itself, which is what
                    // happens as soon as the machines are not alike. For a
                    // world that does not divide the layer count this has to
                    // name the same layers as the stage's own split; nothing
                    // in a running pipeline would report it if it did not.
                    let layers = LayerSplit::even(c.num_hidden_layers, world)
                        .range(rank as u32)
                        .ok_or_else(|| anyhow!("rank {rank} outside world {world}"))?;
                    stage_with_layers(
                        (*c).clone(),
                        Device::Cpu,
                        rank as u32,
                        world,
                        (*w).clone(),
                        layers,
                    )
                } else {
                    stage((*c).clone(), Device::Cpu, rank as u32, world, (*w).clone())
                }
                .with_decode_cache(c.max_position_embeddings.min(2048));
                let coord = PipelineCoordinator::new(ProcessGroup::new(Arc::new(t)));
                let mut tokens = (*prompt).clone();
                coord.generate(&mut st, &mut tokens, n, argmax, |_| false)
            })
        })
        .collect();

    let mut out = Vec::new();
    for h in handles {
        out.push(h.join().map_err(|_| anyhow!("a rank panicked"))??);
    }
    Ok(out.remove(0))
}

fn main() -> Result<()> {
    let mut model = None;
    let mut prompt_ids: Vec<u32> = Vec::new();
    let mut tokens = 6usize;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--model" => model = args.next(),
            "--prompt-ids" => {
                prompt_ids = args
                    .next()
                    .unwrap_or_default()
                    .split(',')
                    .filter(|s| !s.trim().is_empty())
                    .map(|s| s.trim().parse())
                    .collect::<Result<_, _>>()?
            }
            "--tokens" => tokens = args.next().unwrap_or_default().parse()?,
            other => return Err(anyhow!("unknown argument {other}")),
        }
    }
    let model = model.context("--model is required")?;
    if prompt_ids.is_empty() {
        anyhow::bail!("--prompt-ids is required");
    }

    let (cfg, weights) = load(&model)?;
    // The runner builder takes the weight file, not the directory.
    let weights_file = std::path::Path::new(&model)
        .join("model.safetensors")
        .to_string_lossy()
        .into_owned();
    println!(
        "{} layers, hidden {}, {} heads / {} kv, tied {}, rope_scaling {}",
        cfg.num_hidden_layers,
        cfg.hidden_size,
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.tie_word_embeddings,
        cfg.rope_scaling.is_some()
    );
    println!("prompt {prompt_ids:?}, {tokens} tokens");
    for world in 1..=4u32 {
        let split = LayerSplit::even(cfg.num_hidden_layers, world);
        let even = cfg.num_hidden_layers % world as usize == 0;
        println!(
            "  world {world}: {:?}{}",
            split.counts,
            if even { "" } else { "   ← uneven" }
        );
    }
    println!();

    let want = internal(&cfg, &weights, &prompt_ids, tokens)?;
    println!("generate_cached      : {want:?}   ← the generator's own loop");

    for (label, got) in [
        (
            "host-driven pair    ",
            host_driven(&cfg, &weights, &prompt_ids, tokens),
        ),
        (
            "same, via LmRunner  ",
            through_the_trait(&weights_file, &prompt_ids, tokens),
        ),
        (
            "re-prefill each step",
            re_prefill(&cfg, &weights, &prompt_ids, tokens),
        ),
        (
            "pipeline, world 1   ",
            pipelined(&cfg, &weights, &prompt_ids, tokens, 1, false),
        ),
        (
            "pipeline, world 2   ",
            pipelined(&cfg, &weights, &prompt_ids, tokens, 2, false),
        ),
        (
            "pipeline, world 3   ",
            pipelined(&cfg, &weights, &prompt_ids, tokens, 3, false),
        ),
        (
            "pipeline, world 4   ",
            pipelined(&cfg, &weights, &prompt_ids, tokens, 4, false),
        ),
        (
            "world 3, LayerSplit ",
            pipelined(&cfg, &weights, &prompt_ids, tokens, 3, true),
        ),
        (
            "world 4, LayerSplit ",
            pipelined(&cfg, &weights, &prompt_ids, tokens, 4, true),
        ),
    ] {
        match got {
            Ok(ids) => {
                let verdict = if ids == want { "same" } else { "DIFFERS" };
                println!("{label} : {ids:?}   {verdict}");
            }
            Err(e) => println!("{label} : failed: {e:#}"),
        }
    }
    Ok(())
}
