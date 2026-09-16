// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! Run one full DFlash draft step against a real drafter checkpoint.
//!
//! All three passes, in the order the drafter actually uses them:
//!
//! 1. encoder — fuse the target's residual taps,
//! 2. injection — turn the fused features into per-layer KV-cache entries,
//! 3. decoder — denoise `[anchor, MASK, …]` into a block of draft tokens.
//!
//! Real taps require the target model; this feeds deterministic pseudo-random
//! hidden states of the right shape, which is enough to prove every weight
//! loads, every shape lines up, and the block comes out finite and non-constant
//! — but NOT that the proposals are any good. Acceptance rate against a real
//! target is the only thing that shows that.
//!
//! Usage: `cargo run --release -p rlx-dflash --example dflash_forward -- <gguf> [device]`

use std::collections::HashMap;

use anyhow::{Context, Result};
use rlx_core::weight_loader::GgufLoader;
use rlx_dflash::{
    DflashConfig, ReplayLoader, build_decoder_graph, build_encoder_graph, build_kv_inject_graph,
    rope_tables,
};
use rlx_ir::DType;
use rlx_runtime::{CompiledGraph, Device, Session};

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

/// Upload dense params and any packed K-quant blobs the builder deferred.
fn bind(
    compiled: &mut CompiledGraph,
    params: HashMap<String, Vec<f32>>,
    packed: &HashMap<String, (rlx_ir::quant::QuantScheme, Vec<usize>)>,
    loader: &dyn rlx_core::weight_loader::WeightLoader,
) -> Result<()> {
    for (name, data) in &params {
        compiled.set_param(name, data);
    }
    for name in packed.keys() {
        let bytes = loader
            .tensor_bytes_borrowed(name)
            .with_context(|| format!("packed bytes for {name}"))?;
        compiled.set_param_typed(name, bytes, DType::U8);
    }
    Ok(())
}

fn deterministic(n: usize, seed: u64) -> Vec<f32> {
    let mut st = seed;
    (0..n)
        .map(|_| {
            st = st.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            (((st >> 33) as f64 / (1u64 << 31) as f64) - 1.0) as f32 * 0.5
        })
        .collect()
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
        "dflash: layers={} hidden={} ffn={} heads={}/{} head_dim={} vocab={}",
        cfg.num_hidden_layers,
        cfg.hidden_size,
        cfg.intermediate_size,
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.head_dim,
        cfg.vocab_size,
    );
    println!(
        "block_size={} target_layers={:?} sliding_window={:?}",
        cfg.block_size, cfg.target_layers, cfg.sliding_window
    );
    match &cfg.dflash2 {
        Some(d2) => println!(
            "DFlash2: conv kernel={} group={} selector rank={} top_k={}",
            d2.conv_kernel_size, d2.conv_group_size, d2.selector_rank, d2.selector_top_k
        ),
        None => println!("DFlash v1 checkpoint (no selector metadata)"),
    }

    let mut gguf = GgufLoader::from_file(&path)?;
    // Three graphs out of one checkpoint: `WeightLoader::take` is destructive,
    // so reads have to be replayable or the second graph finds nothing.
    let mut loader = ReplayLoader::new(&mut gguf);
    // A short committed prefix stands in for the accepted context.
    let ctx = 8usize;
    let block = cfg.block_size;

    // ── 1. encoder ──────────────────────────────────────────────────────
    let mut packed = HashMap::new();
    let (g_enc, p_enc) = build_encoder_graph(&cfg, &mut loader, 1, ctx, &mut packed)?;
    let mut enc = Session::new(device).compile(g_enc);
    bind(&mut enc, p_enc, &packed, &loader)?;
    let taps = deterministic(ctx * cfg.fused_input_dim(), 0x243f_6a88_85a3_08d3);
    let fused = enc.run(&[("dflash_taps", taps.as_slice())]).remove(0);
    anyhow::ensure!(fused.len() == ctx * cfg.hidden_size, "encoder shape");
    println!("\n[1] encoder: {ctx} taps -> {} fused", fused.len());

    // ── 2. KV injection ─────────────────────────────────────────────────
    let mut packed = HashMap::new();
    let (g_inj, p_inj) = build_kv_inject_graph(&cfg, &mut loader, 1, ctx, &mut packed)?;
    let mut inj = Session::new(device).compile(g_inj);
    bind(&mut inj, p_inj, &packed, &loader)?;
    let (cos, sin) = rope_tables(&(0..ctx).collect::<Vec<_>>(), cfg.head_dim, cfg.rope_theta);
    let cache = inj.run(&[
        ("dflash_fused", fused.as_slice()),
        ("rope_cos", cos.as_slice()),
        ("rope_sin", sin.as_slice()),
    ]);
    anyhow::ensure!(cache.len() == cfg.num_hidden_layers * 2, "inject outputs");
    println!(
        "[2] injection: {} cache tensors ({} per layer x {} layers)",
        cache.len(),
        2,
        cfg.num_hidden_layers
    );

    // ── 3. decoder ──────────────────────────────────────────────────────
    let mut packed = HashMap::new();
    let (g_dec, p_dec, meta) = build_decoder_graph(&cfg, &mut loader, 1, block, ctx, &mut packed)?;
    println!(
        "[3] decoder: {} nodes, {} packed tensors",
        g_dec.nodes().len(),
        packed.len()
    );
    if !meta.shared_params.is_empty() {
        println!(
            "    shares with the target (not in this file): {:?}",
            meta.shared_params
        );
    }
    let mut dec = Session::new(device).compile(g_dec);
    bind(&mut dec, p_dec, &packed, &loader)?;
    // An Eagle-style head carries no embedding and no LM head. Without the
    // target's copies the graph still runs — and drafts noise — so fill them
    // with a deterministic stand-in and say plainly that the numbers below
    // measure plumbing, not proposals.
    for name in &meta.shared_params {
        let n = match name.as_str() {
            "token_embd.weight" | "output.weight" => cfg.vocab_size * cfg.hidden_size,
            other => anyhow::bail!("unexpected shared param {other}"),
        };
        println!("    filling {name} with {n} placeholder values");
        dec.set_param(name, &deterministic(n, 0xbeef_cafe_1234_5678));
    }

    // Noise block: anchor + MASK. Without a MASK id in the metadata there is
    // nothing sensible to fill with, so say so rather than guess.
    let mask = cfg
        .mask_token_id
        .context("checkpoint has no tokenizer.ggml.mask_token_id — cannot build a noise block")?;
    let anchor = 1.0f32;
    let mut tokens = vec![mask as f32; block];
    tokens[0] = anchor;

    let positions: Vec<usize> = (ctx..ctx + block).collect();
    let (cos, sin) = rope_tables(&positions, cfg.head_dim, cfg.rope_theta);
    let anchor_in = [anchor];
    let mask = vec![1.0f32; ctx + block];

    let mut inputs: Vec<(String, &[f32])> = vec![
        ("noise_tokens".into(), tokens.as_slice()),
        ("rope_cos".into(), cos.as_slice()),
        ("rope_sin".into(), sin.as_slice()),
        ("attn_mask".into(), mask.as_slice()),
    ];
    if cfg.dflash2.is_some() {
        inputs.push(("anchor_ids".into(), &anchor_in));
    }
    let names: Vec<String> = (0..cfg.num_hidden_layers)
        .flat_map(|i| [format!("past_k_{i}"), format!("past_v_{i}")])
        .collect();
    for (i, n) in names.iter().enumerate() {
        inputs.push((n.clone(), cache[i].as_slice()));
    }
    let refs: Vec<(&str, &[f32])> = inputs.iter().map(|(n, d)| (n.as_str(), *d)).collect();

    let t = std::time::Instant::now();
    let out = dec.run(&refs);
    let el = t.elapsed();

    let hidden = &out[meta.hidden];
    anyhow::ensure!(
        hidden.len() == block * cfg.hidden_size,
        "decoder hidden shape"
    );
    anyhow::ensure!(hidden.iter().all(|v| v.is_finite()), "non-finite hidden");
    let mean = hidden.iter().sum::<f32>() / hidden.len() as f32;
    let var = hidden.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / hidden.len() as f32;
    anyhow::ensure!(
        var > 1e-8,
        "draft hidden is constant — weights not applied?"
    );

    match (meta.logits, meta.selector) {
        (Some(i), _) => {
            let logits = &out[i];
            anyhow::ensure!(logits.len() == block * cfg.vocab_size, "logits shape");
            println!(
                "    logits [{block} x {}] mean {:.5}",
                cfg.vocab_size,
                logits.iter().sum::<f32>() / logits.len() as f32
            );
        }
        (None, Some((c, s))) => {
            let k = cfg
                .dflash2
                .expect("selector implies dflash2")
                .selector_top_k;
            anyhow::ensure!(out[c].len() == block * k, "candidate shape");
            anyhow::ensure!(out[s].len() == (block - 1) * k * k, "lattice shape");
            println!(
                "    selector: {} candidates + {} pair scores (vs {} logits it did NOT ship)",
                out[c].len(),
                out[s].len(),
                block * cfg.vocab_size
            );
        }
        _ => anyhow::bail!("decoder produced neither logits nor a lattice"),
    }

    println!(
        "\nforward: {block} draft positions in {el:?} ({:.1} pos/s)",
        block as f64 / el.as_secs_f64()
    );
    println!(
        "hidden[{block}x{}]: mean {mean:.5} var {var:.5}",
        cfg.hidden_size
    );
    println!("\nOK — all three passes run; acceptance rate needs the real target");
    Ok(())
}
