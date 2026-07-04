//! Lightweight single-backend sanity check for a packed Gemma GGUF (no CPU
//! baseline — for models too big to also dequant on CPU, e.g. Gemma 4 12B).
//!
//!   RLX_GEMMA_CHECK_DEVICE=cuda RLX_GEMMA3_GGUF=<gguf> \
//!   cargo run -p rlx-gemma --features cuda --release --example gemma_cuda_check
//!
//! Reports: logits finite? argmax; top-5; greedy token stream. A working model
//! yields varied, non-degenerate tokens; the old score_scale bug produced a
//! single repeating EOG-ish token.

use anyhow::Result;
use rlx_gemma::GemmaRunner;
use rlx_qwen3::SampleOpts;
use rlx_runtime::Device;
use std::path::PathBuf;

const DEFAULT_IDS: &[u32] = &[
    2, 105, 2364, 107, 3689, 563, 1156, 2915, 1156, 236881, 25685, 528, 886, 2822, 13315, 236761,
    106, 107, 105, 4368, 107,
];

fn prompt_ids() -> Vec<u32> {
    match std::env::var("RLX_GEMMA_PROMPT_IDS") {
        Ok(s) => s.split(',').filter_map(|t| t.trim().parse().ok()).collect(),
        Err(_) => DEFAULT_IDS.to_vec(),
    }
}

fn main() -> Result<()> {
    let path: PathBuf = std::env::var("RLX_GEMMA3_GGUF").expect("RLX_GEMMA3_GGUF").into();
    let dev = match std::env::var("RLX_GEMMA_CHECK_DEVICE").as_deref() {
        Ok("cuda") => Device::Cuda,
        Ok("metal") => Device::Metal,
        Ok("mlx") => Device::Mlx,
        Ok("wgpu") | Ok("gpu") => Device::Gpu,
        Ok("vulkan") => Device::Vulkan,
        _ => Device::Cpu,
    };
    let ids = prompt_ids();
    eprintln!("device={dev:?} gguf={path:?} n_ids={}", ids.len());

    let mut r = GemmaRunner::builder()
        .weights(&path)
        .packed_weights(true)
        .device(dev)
        .max_seq(512)
        .sample(SampleOpts::greedy())
        .build()?;

    let logits = r.predict_logits(&ids)?;
    let vocab = r.config().vocab_size;
    let slice = &logits[..vocab];
    let finite = slice.iter().all(|v| v.is_finite());
    let mut ranked: Vec<(usize, f32)> = slice.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let top5: Vec<usize> = ranked.iter().take(5).map(|(i, _)| *i).collect();
    println!("prefill: finite={finite} argmax={} top5={top5:?}", ranked[0].0);

    if std::env::var("RLX_GEMMA_CHECK_NOGEN").is_ok() {
        println!("greedy skipped (RLX_GEMMA_CHECK_NOGEN)");
        return Ok(());
    }
    let toks = r.generate(&ids, 24, |_| {})?;
    let uniq: std::collections::HashSet<u32> = toks.iter().copied().collect();
    println!("greedy({} toks, {} unique) = {:?}", toks.len(), uniq.len(), toks);
    println!(
        "verdict: {}",
        if finite && uniq.len() > 3 {
            "OK — varied, finite output"
        } else {
            "SUSPECT — degenerate/non-finite (possible bug)"
        }
    );
    Ok(())
}
