// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! Where does a pipeline-sharded model stop agreeing with a whole one?
//!
//! Real weights, no network, three runs over the same prompt:
//!
//!   1. the monolithic KV-cached generator — the reference,
//!   2. `Qwen3PipelineDecodeStage` at **world 1**, which is the same stage
//!      the shards use but with no split and no hand-off,
//!   3. the stage at **world 2**, across two threads on loopback.
//!
//! The point of (2) is to separate the two things that could be wrong. If
//! world 1 already disagrees with the generator, the stage's own graphs are
//! at fault and the split is innocent. If world 1 agrees and world 2 does
//! not, the fault is in the hand-off.
//!
//! ```text
//! cargo run --release -p rlx-qwen3 --example pipeline_parity -- \
//!     --model /path/to/qwen3 --prompt-ids 785,6722,315,9625,374 --tokens 4
//! ```

use anyhow::{Context, Result, anyhow};
use rlx_distributed::{NetTransport, PipelineCoordinator, ProcessGroup};
use rlx_qwen3::Qwen3Config;
use rlx_qwen3::pipeline_decode;
use rlx_runtime::Device;
use std::collections::HashMap;
use std::net::{Ipv4Addr, SocketAddr, TcpListener};
use std::sync::Arc;
use std::thread;

type Tensors = HashMap<String, (Vec<f32>, Vec<usize>)>;

fn argmax(v: &[f32]) -> u32 {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i as u32)
        .unwrap()
}

fn load(model: &str) -> Result<(Qwen3Config, Tensors)> {
    let dir = std::path::Path::new(model);
    let cfg = Qwen3Config::from_file(&dir.join("config.json"))
        .with_context(|| format!("reading {model}/config.json"))?;
    // Every weight file, so a split checkpoint works.
    let mut files: Vec<std::path::PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.extension()
                .is_some_and(|e| e == "safetensors" || e == "gguf")
        })
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

/// The reference: one graph, every layer, KV-cached decode.
fn monolithic(c: &Qwen3Config, w: &Tensors, prompt: &[u32], n: usize) -> Result<Vec<u32>> {
    use rlx_core::weight_map::WeightMap;
    let mut wm = WeightMap::from_tensors(w.clone());
    let mut g = rlx_qwen3::generator::Qwen3Generator::from_loader(c.clone(), &mut wm, Device::Cpu)?;
    g.prefill(prompt);
    g.generate_cached(n, rlx_qwen3::sampling::SampleOpts::greedy())
}

/// The stage split `world` ways, driven the way a *server* drives it: the
/// host samples from the logits and feeds the chosen token back, instead of
/// letting the coordinator sample inside the forward pass and broadcast the
/// result.
///
/// A server has to do it this way — temperature, top-p, a logit bias, stop
/// strings and log-probs all live above the pipeline — so this is the shape
/// that actually ships, and it is worth checking separately from
/// `coord.generate`.
fn sharded_host_sampled(
    c: &Qwen3Config,
    w: &Tensors,
    prompt: &[u32],
    n: usize,
    world: u32,
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
                let mut stage = pipeline_decode::stage(
                    (*c).clone(),
                    Device::Cpu,
                    rank as u32,
                    world,
                    (*w).clone(),
                )
                .with_decode_cache(c.max_position_embeddings);
                let coord = PipelineCoordinator::new(ProcessGroup::new(Arc::new(t)));
                let mut tokens = (*prompt).clone();
                let mut produced = Vec::new();

                // Rank 0 holds the head: it samples and announces the token
                // over a broadcast of its own, exactly as a server's control
                // channel does. The coordinator's own broadcast still fires
                // and is ignored.
                for _ in 0..n {
                    let mut captured: Option<Vec<f32>> = None;
                    coord.forward_step(&mut stage, &tokens, |l| {
                        captured = Some(l.to_vec());
                        0
                    })?;
                    let mut chosen = [0f32; 1];
                    if rank == 0 {
                        chosen[0] =
                            argmax(&captured.ok_or_else(|| anyhow!("rank 0 got no logits"))?)
                                as f32;
                    }
                    coord.group().broadcast(0, &mut chosen)?;
                    let tok = chosen[0] as u32;
                    produced.push(tok);
                    tokens.push(tok);
                }
                Ok(produced)
            })
        })
        .collect();

    let mut out = Vec::new();
    for h in handles {
        out.push(h.join().map_err(|_| anyhow!("a rank panicked"))??);
    }
    Ok(out.remove(0))
}

/// The stage, split `world` ways across threads on loopback.
fn sharded(
    c: &Qwen3Config,
    w: &Tensors,
    prompt: &[u32],
    n: usize,
    world: u32,
    cached: bool,
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
                let stage = pipeline_decode::stage(
                    (*c).clone(),
                    Device::Cpu,
                    rank as u32,
                    world,
                    (*w).clone(),
                );
                let mut stage = if cached {
                    stage.with_decode_cache(c.max_position_embeddings)
                } else {
                    stage
                };
                let coord = PipelineCoordinator::new(ProcessGroup::new(Arc::new(t)));
                let mut tokens = (*prompt).clone();
                coord.generate(&mut stage, &mut tokens, n, argmax, |_| false)
            })
        })
        .collect();

    let mut out = Vec::new();
    for h in handles {
        out.push(h.join().map_err(|_| anyhow!("a rank panicked"))??);
    }
    // Every rank should have the same ids; report rank 0's and complain if not.
    for (r, ids) in out.iter().enumerate().skip(1) {
        if *ids != out[0] {
            eprintln!("  ranks disagree: rank 0 {:?} vs rank {r} {ids:?}", out[0]);
        }
    }
    Ok(out.remove(0))
}

fn main() -> Result<()> {
    let mut model = None;
    let mut prompt_ids: Vec<u32> = Vec::new();
    let mut tokens = 4usize;
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
        // "The capital of France is", as Qwen3 tokenizes it.
        prompt_ids = vec![785, 6722, 315, 9625, 374];
    }

    let (cfg, weights) = load(&model)?;
    println!(
        "{} layers, hidden {}, vocab {}, tied {}",
        cfg.num_hidden_layers, cfg.hidden_size, cfg.vocab_size, cfg.tie_word_embeddings
    );
    println!("prompt ids {prompt_ids:?}, {tokens} tokens\n");

    let reference = monolithic(&cfg, &weights, &prompt_ids, tokens)?;
    println!("monolithic generator : {reference:?}   ← the reference");

    for (label, world, cached) in [
        ("stage, world 1      ", 1u32, true),
        ("stage, world 1, slow", 1, false),
        ("stage, world 2      ", 2, true),
        ("stage, world 2, slow", 2, false),
    ] {
        match sharded(&cfg, &weights, &prompt_ids, tokens, world, cached) {
            Ok(ids) => {
                let verdict = if ids == reference { "same" } else { "DIFFERS" };
                println!("{label} : {ids:?}   {verdict}");
            }
            Err(e) => println!("{label} : failed: {e:#}"),
        }
    }
    for world in [1u32, 2] {
        let label = format!("host-sampled, world {world}");
        match sharded_host_sampled(&cfg, &weights, &prompt_ids, tokens, world) {
            Ok(ids) => {
                let verdict = if ids == reference { "same" } else { "DIFFERS" };
                println!("{label} : {ids:?}   {verdict}");
            }
            Err(e) => println!("{label} : failed: {e:#}"),
        }
    }
    Ok(())
}
