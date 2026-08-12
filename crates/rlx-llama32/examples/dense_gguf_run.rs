//! Load a dense Llama-shaped GGUF (llama / granite / exaone / …) through the
//! packed `Op::DequantMatMul` path and greedily continue a prompt.
//!
//! Usage:
//!   cargo run -p rlx-llama32 --example dense_gguf_run -- <weights.gguf> ["prompt"] [n_new]
//!
//! Prints the parsed config (including any Granite scalar multipliers), the
//! prefill last-position argmax, and the greedy continuation text.

use anyhow::Result;
use rlx_llama32::{Llama32Runner, encode_prompt_auto};
use rlx_runtime::Device;
use std::path::Path;

fn main() -> Result<()> {
    let weights = std::env::args()
        .nth(1)
        .expect("usage: dense_gguf_run <weights.gguf> [prompt] [n_new] [device]");
    let prompt = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "The capital of France is".to_string());
    let n_new: usize = std::env::args()
        .nth(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(16);
    let device = match std::env::args().nth(4).as_deref() {
        Some("metal") => Device::Metal,
        Some("mlx") => Device::Mlx,
        Some("gpu") | Some("wgpu") => Device::Gpu,
        Some("cuda") => Device::Cuda,
        _ => Device::Cpu,
    };
    eprintln!("device = {device:?}");

    let wp = Path::new(&weights);
    let ids = encode_prompt_auto(wp, None, &prompt)?;
    eprintln!(
        "prompt = {prompt:?}  ->  {} token ids: {:?}",
        ids.len(),
        ids
    );

    let mut runner = Llama32Runner::builder()
        .weights(wp)
        .device(device)
        .packed_weights(true)
        .stream(true)
        .max_seq(256)
        .build()?;

    let cfg = runner.config();
    eprintln!(
        "config: arch={:?} vocab={} hidden={} layers={} heads={}/{} head_dim={} rope_theta={} rope_style={:?}",
        cfg.gguf_arch,
        cfg.vocab_size,
        cfg.hidden_size,
        cfg.num_hidden_layers,
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.head_dim(),
        cfg.rope_theta,
        cfg.rope_style,
    );
    eprintln!(
        "granite scalars: embed={:?} residual={:?} attn_scale={:?} logit_scale={:?}",
        cfg.embedding_scale, cfg.residual_scale, cfg.attention_scale, cfg.logit_scale,
    );

    // Greedy continuation (host tied-lm_head argmax on the GGUF packed path).
    //
    // TTFT is the first callback: on the packed path prefill emits the last
    // position's logits directly, so token 1 lands as soon as prefill is done.
    // Decode throughput therefore excludes prefill — measure it over tokens
    // 2..n, not 1..n, or prefill smears into the per-token figure.
    // Two passes. The FIRST includes lazily compiling and uploading the prefill
    // graph, which is a one-time cost of seconds — reporting it as TTFT would
    // overstate steady-state latency by ~10x. The SECOND is what a warm server
    // actually sees. Both are printed; neither alone is the honest number.
    let mut out_ids = Vec::new();
    for pass in 0..2 {
        let t0 = std::time::Instant::now();
        let mut ttft: Option<std::time::Duration> = None;
        let mut per_tok: Vec<std::time::Duration> = Vec::with_capacity(n_new);
        let mut last = t0;
        out_ids = runner.generate(&ids, n_new, |_t| {
            let now = std::time::Instant::now();
            if ttft.is_none() {
                ttft = Some(now - t0);
            } else {
                per_tok.push(now - last);
            }
            last = now;
        })?;
        let total = t0.elapsed();
        let ttft = ttft.unwrap_or(total);
        let label = if pass == 0 { "COLD" } else { "WARM" };
        println!(
            "\n[{label}] TTFT {:.1} ms  ({} prompt tokens => {:.1} tok/s prefill)",
            ttft.as_secs_f64() * 1e3,
            ids.len(),
            ids.len() as f64 / ttft.as_secs_f64()
        );
        if !per_tok.is_empty() {
            let mut sorted = per_tok.clone();
            sorted.sort();
            let decode: std::time::Duration = per_tok.iter().sum();
            // Median as well as mean: under memory pressure a handful of tokens
            // stall on page faults for seconds, which drags the mean far off the
            // rate the model actually sustains.
            println!(
                "[{label}] DECODE {:.2} tok/s mean | {:.2} tok/s median  ({} tokens, mean {:.1} ms, median {:.1} ms, max {:.1} ms)",
                per_tok.len() as f64 / decode.as_secs_f64(),
                1.0 / sorted[sorted.len() / 2].as_secs_f64(),
                per_tok.len(),
                decode.as_secs_f64() * 1e3 / per_tok.len() as f64,
                sorted[sorted.len() / 2].as_secs_f64() * 1e3,
                sorted[sorted.len() - 1].as_secs_f64() * 1e3,
            );
        }
        println!(
            "[{label}] TOTAL  {:.2} s for {} new tokens",
            total.as_secs_f64(),
            out_ids.len()
        );
    }
    eprintln!("greedy_gen_ids = {out_ids:?}");
    let text = rlx_llama32::decode_ids_auto(wp, None, &out_ids, true)
        .unwrap_or_else(|e| format!("<decode failed: {e}>"));
    println!("PROMPT: {prompt}");
    println!("GEN: {text}");
    Ok(())
}
