// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! `rlx-vit-elastic` — SnapViT elastic pruning + GLARE continual pre-training.
//!
//! ```text
//!   rlx-vit-elastic snapvit elastic --backbone synthetic
//!   rlx-vit-elastic snapvit prune   --backbone dino-vitb16 --weights dino.safetensors \
//!                                    --data imgs/ --sparsity 0.4 --device metal
//!   rlx-vit-elastic glare  train    --backbone dino-vitb16 --weights dino.safetensors \
//!                                    --data imgs/ --steps 200 --device metal
//! ```

use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::{Result, bail};
use rlx_runtime::Device;

use rlx_vit_elastic::data::{load_images, synthetic_images};
use rlx_vit_elastic::glare::{GlareConfig, GlareTrainer};
use rlx_vit_elastic::snapvit::{self, SnapVitParams};
use rlx_vit_elastic::vit::load_backbone;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 2 {
        eprintln!(
            "usage:\n  rlx-vit-elastic snapvit prune|elastic [opts]\n  rlx-vit-elastic glare train [opts]\n\
             opts: --backbone NAME --weights PATH --data DIR --device DEV --sparsity S --steps N --rank R"
        );
        std::process::exit(2);
    }
    let (cmd, sub) = (args[0].as_str(), args[1].as_str());
    let opts = parse_opts(&args[2..]);

    match (cmd, sub) {
        ("snapvit", "prune") | ("snapvit", "elastic") => run_snapvit(sub, &opts),
        ("glare", "train") => run_glare(&opts),
        _ => bail!("unknown command '{cmd} {sub}'"),
    }
}

fn parse_opts(rest: &[String]) -> HashMap<String, String> {
    let mut m = HashMap::new();
    let mut i = 0;
    while i < rest.len() {
        if let Some(key) = rest[i].strip_prefix("--") {
            let val = rest
                .get(i + 1)
                .cloned()
                .unwrap_or_else(|| "true".to_string());
            m.insert(key.to_string(), val);
            i += 2;
        } else {
            i += 1;
        }
    }
    m
}

fn device_of(opts: &HashMap<String, String>) -> Result<Device> {
    Ok(
        match opts.get("device").map(|s| s.as_str()).unwrap_or("cpu") {
            "cpu" => Device::Cpu,
            "metal" | "mps" => Device::Metal,
            "mlx" => Device::Mlx,
            "cuda" => Device::Cuda,
            "gpu" | "wgpu" => Device::Gpu,
            "vulkan" => Device::Vulkan,
            other => bail!("unknown device '{other}'"),
        },
    )
}

fn images_of(
    opts: &HashMap<String, String>,
    want: usize,
    side: usize,
) -> Result<Vec<rlx_vit_elastic::snapvit::CalibImage>> {
    match opts.get("data") {
        Some(dir) => load_images(&PathBuf::from(dir), want),
        None => {
            eprintln!("[data] no --data directory; using {want} synthetic images");
            Ok(synthetic_images(want, side))
        }
    }
}

fn run_snapvit(sub: &str, opts: &HashMap<String, String>) -> Result<()> {
    let backbone = opts
        .get("backbone")
        .map(|s| s.as_str())
        .unwrap_or("synthetic");
    let device = device_of(opts)?;
    let (cfg, loaded) = load_backbone(backbone, opts.get("weights").map(PathBuf::from).as_deref())?;
    println!(
        "[snapvit] backbone={backbone} hidden={} depth={} heads={} device={device:?}",
        cfg.hidden_size, cfg.num_hidden_layers, cfg.num_attention_heads
    );

    let n = opts
        .get("calib")
        .and_then(|s| s.parse().ok())
        .unwrap_or(8usize);
    let all = images_of(opts, 2 * n, cfg.img_size * 2)?;
    let (calib, fit) = all.split_at(all.len() / 2);

    let mut params = SnapVitParams::new(cfg.img_size);
    params.ssl.crops.n_local = 4;
    params.xnes.iterations = opts.get("iters").and_then(|s| s.parse().ok()).unwrap_or(10);
    params.xnes.population = opts.get("pop").and_then(|s| s.parse().ok()).unwrap_or(8);
    if cfg.hidden_size <= 192 {
        params.pca_dim = 0;
    }
    if sub == "prune" {
        let s: f32 = opts
            .get("sparsity")
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.4);
        params.elastic_sparsities = vec![s];
    }

    let res = snapvit::run(&cfg, &loaded, calib, fit, &params, device)?;
    println!(
        "[snapvit] xNES fitness: baseline={:.4} best={:.4}",
        res.baseline_fitness, res.best_fitness
    );
    println!("  sparsity   fitness  heads_pruned  ffn_pruned  param_reduction");
    for e in &res.elastic {
        println!(
            "   {:>5.2}    {:>7.4}   {:>10}   {:>9}     {:>6.1}%",
            e.sparsity,
            e.fitness,
            e.heads_pruned,
            e.ffn_pruned,
            e.param_reduction * 100.0
        );
    }
    Ok(())
}

fn run_glare(opts: &HashMap<String, String>) -> Result<()> {
    let backbone = opts
        .get("backbone")
        .map(|s| s.as_str())
        .unwrap_or("synthetic");
    let device = device_of(opts)?;
    let (cfg, loaded) = load_backbone(backbone, opts.get("weights").map(PathBuf::from).as_deref())?;
    let steps = opts
        .get("steps")
        .and_then(|s| s.parse().ok())
        .unwrap_or(100usize);

    let mut gc = GlareConfig::new(cfg.hidden_size);
    if cfg.hidden_size <= 64 {
        gc = GlareConfig::small(cfg.hidden_size);
    }
    if let Some(r) = opts.get("rank").and_then(|s| s.parse().ok()) {
        gc.adapter.rank = r;
    }
    println!(
        "[glare] backbone={backbone} hidden={} adapter_rank={} K={} steps={steps} device={device:?}",
        cfg.hidden_size, gc.adapter.rank, gc.head.out_k
    );

    let images = images_of(
        opts,
        opts.get("nimg").and_then(|s| s.parse().ok()).unwrap_or(16),
        cfg.img_size * 2,
    )?;
    let mut trainer = GlareTrainer::new(&cfg, &loaded, &gc, steps, device)?;
    let losses = trainer.train(&images, steps)?;

    let win = losses.len().clamp(1, 5);
    let first: f32 = losses.iter().take(win).sum::<f32>() / win as f32;
    let last: f32 = losses.iter().rev().take(win).sum::<f32>() / win as f32;
    println!("[glare] loss: first≈{first:.4} → last≈{last:.4} over {steps} steps");

    if let Some(out) = opts.get("out") {
        let trained = trainer.trained_params();
        let json = serde_json::to_string(&trained.keys().collect::<Vec<_>>())?;
        std::fs::write(out, json)?;
        println!(
            "[glare] wrote {} trained adapter/head/ca tensors' index to {out}",
            trained.len()
        );
    }
    Ok(())
}
