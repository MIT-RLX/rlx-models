//! Ground-truth check: llama.cpp vs rlx-CPU vs rlx-Metal for a packed Gemma GGUF.
//! Settles whether applying Gemma 2's attn logit soft-cap (rlx-Metal/CUDA) or
//! ignoring it (rlx-CPU) matches the llama.cpp reference (which applies it).
//!
//!   RLX_GEMMA3_GGUF=<gguf> cargo run -p rlx-gemma \
//!     --features "metal parity-llama" --release --example gemma_llama_parity

use anyhow::Result;
use rlx_gemma::GemmaRunner;
use rlx_qwen3::SampleOpts;
use rlx_runtime::{Device, is_available};
use std::path::PathBuf;

const PROMPT: &str = "The capital of France is Paris. The capital of Japan is";
const STEPS: usize = 16;

fn cos_l2(a: &[f32], b: &[f32]) -> (f32, f32) {
    let n = a.len().min(b.len());
    let (mut dot, mut na, mut nb, mut l2) = (0f64, 0f64, 0f64, 0f64);
    for i in 0..n {
        let (x, y) = (a[i] as f64, b[i] as f64);
        dot += x * y; na += x * x; nb += y * y; l2 += (x - y) * (x - y);
    }
    let cos = if na > 0.0 && nb > 0.0 { (dot / (na.sqrt() * nb.sqrt())) as f32 } else { 0.0 };
    (cos, l2.sqrt() as f32)
}

fn top5(l: &[f32]) -> Vec<usize> {
    let mut v: Vec<(usize, f32)> = l.iter().enumerate().map(|(i, &x)| (i, x)).collect();
    v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    v.into_iter().take(5).map(|(i, _)| i).collect()
}

fn rlx_logits_greedy(dev: Device, path: &PathBuf, ids: &[u32]) -> Result<(Vec<usize>, Vec<u32>)> {
    let mut r = GemmaRunner::builder()
        .weights(path).packed_weights(true).device(dev).max_seq(512)
        .sample(SampleOpts::greedy()).build()?;
    let logits = r.predict_logits(ids)?;
    let vocab = r.config().vocab_size;
    let t5 = top5(&logits[..vocab]);
    let mut r2 = GemmaRunner::builder()
        .weights(path).packed_weights(true).device(dev).max_seq(512)
        .sample(SampleOpts::greedy()).build()?;
    let greedy = r2.generate(ids, STEPS, |_| {})?;
    Ok((t5, greedy))
}

fn main() -> Result<()> {
    let path: PathBuf = std::env::var("RLX_GEMMA3_GGUF").expect("RLX_GEMMA3_GGUF").into();

    // Valid gemma2 tokens for a real prompt (BOS + content), via the GGUF tokenizer.
    let ids = rlx_gemma::llama_reference::tokenize(&path, PROMPT)?;
    println!("prompt ids ({}) = {ids:?}", ids.len());

    // llama.cpp reference (applies Gemma 2 softcap).
    let llama_logits = rlx_gemma::llama_reference::last_token_logits(&path, &ids)?;
    let llama_t5 = top5(&llama_logits);
    let llama_greedy = rlx_gemma::llama_reference::greedy_generation_ids(&path, &ids, STEPS as u32, 512)?;
    println!("llama.cpp  top5={llama_t5:?}");
    println!("llama.cpp  greedy={llama_greedy:?}");

    // Hidden-state parity (post-model.norm, pre-lm_head): splits forward vs lm_head.
    let llama_h = rlx_gemma::llama_reference::last_token_hidden(&path, &ids)?;
    let mut rh = GemmaRunner::builder()
        .weights(&path).packed_weights(true).device(Device::Cpu).max_seq(512).build()?;
    let rlx_h = rh.predict_last_hidden(&ids)?;
    let (hc, hl2) = cos_l2(&rlx_h, &llama_h);
    println!("hidden(CPU): cos={hc:.6} L2={hl2:.4} rlx[0..4]={:?} llama[0..4]={:?}",
        &rlx_h[..4.min(rlx_h.len())], &llama_h[..4.min(llama_h.len())]);

    let (cpu_t5, cpu_g) = rlx_logits_greedy(Device::Cpu, &path, &ids)?;
    println!("rlx-CPU    top5={cpu_t5:?}");
    println!("rlx-CPU    greedy={cpu_g:?}");

    if is_available(Device::Metal) {
        let (m_t5, m_g) = rlx_logits_greedy(Device::Metal, &path, &ids)?;
        println!("rlx-Metal  top5={m_t5:?}");
        println!("rlx-Metal  greedy={m_g:?}");
        let n = |a: &[u32], b: &[u32]| a.iter().zip(b).take_while(|(x, y)| x == y).count();
        println!("\n== match vs llama.cpp (greedy prefix len / {STEPS}) ==");
        println!("  rlx-CPU   : top1_match={} greedy_prefix={}", cpu_t5[0] == llama_t5[0], n(&cpu_g, &llama_greedy));
        println!("  rlx-Metal : top1_match={} greedy_prefix={}", m_t5[0] == llama_t5[0], n(&m_g, &llama_greedy));
    }
    Ok(())
}
