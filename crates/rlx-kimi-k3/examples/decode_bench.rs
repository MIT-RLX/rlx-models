// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! `decode_bench` — greedy decode speed on a synthetic Kimi-K3 checkpoint of a
//! chosen size (real tensor layout, MXFP4 experts), through the same runner the
//! real model uses. Prints prefill time and steady-state ms per decode token.
//!
//!   cargo run -p rlx-kimi-k3 --release --example decode_bench -- \
//!       [--hidden 1024] [--layers 6] [--experts 16] [--tokens 16] [--device cpu]
//!       [--expert-cache-mb 0] [--rounds 3] [--dense f32|bf16|int8]
//!   RLX_KIMI_BENCH=1 …   # per-step breakdown (router / paging / experts / body)
//!
//! Then decode steps with and without the per-layer caches (resident backbone,
//! compiled graphs, MoE weights + router / experts graphs, expert LRU),
//! interleaved, reporting median wall and CPU time per token.

use anyhow::Result;
use rlx_kimi_k3::loader::CheckpointLoader;
use rlx_kimi_k3::runner::{DecodeState, LayerCache, decode_forward_range, run_generate};
use rlx_kimi_k3::source::{CheckpointSource, EMBED};
use rlx_kimi_k3::synth::{SynthSpec, write_synthetic_checkpoint};
use std::time::Instant;

fn arg(args: &[String], k: &str, d: usize) -> usize {
    args.iter()
        .position(|a| a == k)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(d)
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let hidden = arg(&args, "--hidden", 1024);
    let layers = arg(&args, "--layers", 6);
    let experts = arg(&args, "--experts", 16);
    let n_gen = arg(&args, "--tokens", 16).max(2);
    let device = rlx_cli::parse_device(
        args.iter()
            .position(|a| a == "--device")
            .and_then(|i| args.get(i + 1))
            .map(String::as_str)
            .unwrap_or("cpu"),
    )?;
    if let Some(d) = args
        .iter()
        .position(|a| a == "--dense")
        .and_then(|i| args.get(i + 1))
    {
        rlx_kimi_k3::common::set_dense_residency(Some(d.parse()?));
    }
    let heads = (hidden / 128).max(1);
    let spec = SynthSpec {
        layers,
        hidden,
        vocab: 4096,
        kda_layers: (1..=layers).filter(|l| l % 3 != 0).collect(),
        kda_heads: heads,
        kda_head_dim: 128,
        mla_heads: heads,
        q_lora_rank: hidden / 2,
        kv_lora_rank: 128,
        qk_nope_head_dim: 64,
        qk_rope_head_dim: 32,
        v_head_dim: 64,
        num_experts: experts,
        top_k: 4.min(experts),
        num_shared: 1,
        moe_inter: (hidden / 2).div_ceil(32) * 32,
        latent: (hidden / 2).div_ceil(32) * 32,
        dense_inter: hidden * 3,
        attn_res_block_size: 3,
        layers_per_shard: 2,
        ..SynthSpec::default()
    };
    let dir = std::env::temp_dir().join(format!("rlx-kimi-bench-h{hidden}-l{layers}-e{experts}"));
    if !dir.join("config.json").exists() {
        write_synthetic_checkpoint(&dir, &spec)?;
    }
    let src = CheckpointSource::open(&dir)?;
    let tc = src.text_config().clone();
    let mut ck = CheckpointLoader::open(&dir)?;
    let prompt: Vec<u32> = (0..8).map(|i| (i * 37 + 11) % 4096).collect();
    let make = |s: usize| tc.flow_config(1, s);
    let out = run_generate(&mut ck, &tc, make, &prompt, n_gen, layers, device)?;
    println!("greedy tokens {out:?}");

    // Decode steps with and without the per-layer caches, interleaved so a
    // change in machine load hits both alike. CPU time (all threads) tracks
    // the work done; wall time is also printed.
    let expert_mb = arg(&args, "--expert-cache-mb", 0);
    let rounds = arg(&args, "--rounds", 3);
    let cfg1 = tc.flow_config(1, 1);
    let decode = |cache: LayerCache| -> Result<(f64, f64)> {
        let mut ck = CheckpointLoader::open(&dir)?;
        let mut cache = cache;
        let mut state = DecodeState::zeros(&tc, &cfg1);
        let mut step = |t: u32, cache: &mut LayerCache, state: &mut DecodeState| -> Result<()> {
            let h0 = ck.gather_embed(EMBED, &[t], cfg1.hidden)?;
            decode_forward_range(
                &mut ck,
                &tc,
                &cfg1,
                h0,
                Vec::new(),
                state,
                0,
                layers,
                cache,
                device,
            )?;
            Ok(())
        };
        step(prompt[0], &mut cache, &mut state)?; // warm-up (cache fill / first compile)
        let (w0, c0) = (Instant::now(), cpu_seconds());
        for &t in out.iter().take(n_gen - 1) {
            step(t, &mut cache, &mut state)?;
        }
        let k = (n_gen - 1) as f64;
        Ok((w0.elapsed().as_secs_f64() / k, (cpu_seconds() - c0) / k))
    };
    let configs: [(&str, Box<dyn Fn() -> LayerCache>); 5] = [
        ("no caches", Box::new(LayerCache::disabled)),
        (
            "resident layers",
            Box::new(|| {
                LayerCache::from_env()
                    .with_moe_cache(false)
                    .with_resident_kda(false)
            }),
        ),
        (
            "+ MoE cache",
            Box::new(|| {
                LayerCache::from_env()
                    .with_moe_cache(true)
                    .with_resident_kda(false)
            }),
        ),
        (
            "+ resident KDA",
            Box::new(|| {
                LayerCache::from_env()
                    .with_moe_cache(true)
                    .with_resident_kda(true)
            }),
        ),
        (
            "+ expert LRU",
            Box::new(move || {
                LayerCache::from_env()
                    .with_moe_cache(true)
                    .with_resident_kda(true)
                    .with_expert_cache_mb(expert_mb)
            }),
        ),
    ];
    let mut samples: Vec<Vec<(f64, f64)>> = vec![Vec::new(); configs.len()];
    for _ in 0..rounds {
        for (i, (_, make)) in configs.iter().enumerate() {
            samples[i].push(decode(make())?);
        }
    }
    let med = |v: &[(f64, f64)], cpu: bool| {
        let mut x: Vec<f64> = v.iter().map(|p| if cpu { p.1 } else { p.0 }).collect();
        x.sort_by(f64::total_cmp);
        x[x.len() / 2]
    };
    println!(
        "hidden {hidden}, {layers} layers, {experts} experts on {device:?}, dense {:?}, per decode token (median of {rounds}, expert LRU {expert_mb} MB):",
        rlx_kimi_k3::common::dense_residency()
    );
    let base = med(&samples[1], true);
    for ((name, _), v) in configs.iter().zip(&samples) {
        let (w, c) = (med(v, false), med(v, true));
        println!(
            "  {name:<16} wall {:>8.1} ms   cpu {:>8.1} ms   ({:.2}× CPU vs resident layers)",
            w * 1e3,
            c * 1e3,
            base / c.max(1e-12)
        );
    }
    Ok(())
}

/// User + system CPU seconds of this process (all threads).
fn cpu_seconds() -> f64 {
    let mut ru: libc::rusage = unsafe { std::mem::zeroed() };
    // SAFETY: getrusage writes one `rusage` into the pointer we own.
    unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut ru) };
    let t = |v: libc::timeval| v.tv_sec as f64 + v.tv_usec as f64 * 1e-6;
    t(ru.ru_utime) + t(ru.ru_stime)
}
