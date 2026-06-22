// RLX — GPLv3. Localize the MLX E2B session-reuse bug: does a second run on
//! the same compiled session corrupt results vs a fresh session?
use anyhow::{Context, Result, bail};
use rlx_core::flow_util::compile_graph_gemma_prefill_with_params;
use rlx_gemma::config::GemmaConfig;
use rlx_gemma::qat_loader::GemmaQatLoader;
use rlx_runtime::Device;
use std::collections::HashMap;
use std::path::PathBuf;

const BUCKET: usize = 16;

fn ckpt_dir() -> Result<PathBuf> {
    let home = std::env::var_os("HOME").context("HOME")?;
    let base = std::path::Path::new(&home).join(
        ".cache/huggingface/hub/models--google--gemma-4-E2B-it-qat-mobile-transformers/snapshots",
    );
    let snap = std::fs::read_dir(&base)?
        .flatten()
        .next()
        .context("empty")?
        .path();
    if !snap.join("config.json").is_file() {
        bail!("no config.json");
    }
    Ok(snap)
}
fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}

fn build(
    dir: &std::path::Path,
    cfg: &GemmaConfig,
    dev: Device,
) -> Result<rlx_runtime::CompiledGraph> {
    let mut bld = GemmaQatLoader::open(dir)?;
    let mut packed = HashMap::new();
    let (graph, params) = rlx_gemma::builder::build_gemma_graph_sized_packed_ext(
        cfg,
        &mut bld,
        1,
        BUCKET,
        true,
        false,
        false,
        &mut packed,
        None,
        None,
    )?;
    compile_graph_gemma_prefill_with_params(dev, graph, params)
}

fn run(
    sess: &mut rlx_runtime::CompiledGraph,
    loader: &GemmaQatLoader,
    cfg: &GemmaConfig,
    ids: &[u32],
) -> Result<usize> {
    let mut buf = vec![0u32; BUCKET];
    buf[..ids.len()].copy_from_slice(ids);
    let ple = loader.compute_per_layer_inputs(cfg, &buf)?;
    let idf: Vec<f32> = buf.iter().map(|&i| i as f32).collect();
    let outs = sess.run(&[
        ("input_ids", idf.as_slice()),
        ("per_layer_inputs", ple.as_slice()),
    ]);
    let vocab = cfg.vocab_size;
    let last = ids.len() - 1;
    Ok(argmax(&outs[0][last * vocab..(last + 1) * vocab]))
}

fn main() -> Result<()> {
    let dev = match std::env::args().nth(1).as_deref() {
        Some("mlx") => Device::Mlx,
        Some("metal") => Device::Metal,
        _ => Device::Cpu,
    };
    let dir = ckpt_dir()?;
    let cfg = GemmaConfig::from_file(&dir.join("config.json"))?;
    let loader = GemmaQatLoader::open(&dir)?;
    let a = vec![818u32, 5279, 529, 7001, 563];
    let b = vec![818u32, 5279, 529, 7001, 563, 7001];

    println!("→ device {dev:?}");
    // Reused session: A, A (same), then B.
    let mut s = build(&dir, &cfg, dev)?;
    let r_a1 = run(&mut s, &loader, &cfg, &a)?;
    let r_a2 = run(&mut s, &loader, &cfg, &a)?;
    let r_b_reused = run(&mut s, &loader, &cfg, &b)?;
    println!("   reused: A#1={r_a1}  A#2={r_a2}  B={r_b_reused}");

    // Fresh session for B.
    let mut s2 = build(&dir, &cfg, dev)?;
    let r_b_fresh = run(&mut s2, &loader, &cfg, &b)?;
    println!("   fresh : B={r_b_fresh}");

    println!(
        "→ A#1==A#2: {}   B_reused==B_fresh: {}",
        r_a1 == r_a2,
        r_b_reused == r_b_fresh
    );
    println!("   (expected last-token argmax: A→7001, B→563)");
    Ok(())
}
