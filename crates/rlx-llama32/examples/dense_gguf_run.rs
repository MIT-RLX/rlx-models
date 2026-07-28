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
        .stream(false)
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
    let out_ids = runner.generate(&ids, n_new, |_t| {})?;
    eprintln!("greedy_gen_ids = {out_ids:?}");
    let text = rlx_llama32::decode_ids_auto(wp, None, &out_ids, true)
        .unwrap_or_else(|e| format!("<decode failed: {e}>"));
    println!("PROMPT: {prompt}");
    println!("GEN: {text}");
    Ok(())
}
