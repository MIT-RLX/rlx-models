// RLX — versatile ML compiler + runtime. GPLv3.
//! End-to-end smoke run of the Laguna **mlx-community** loader:
//! `LagunaPackedRunner::from_mlx_dir` + tokenizer → greedy generate → decode.
//! Laguna has no external reference (mlx-lm/transformers lack it), so this is a
//! self-consistency check: the affine checkpoint loads, the forward runs finite,
//! and greedy decode yields non-degenerate (coherent) text.
//!
//!   cargo run --release -p rlx-laguna --example laguna_mlx_run -- <dir> ["prompt"] [n]

use anyhow::Result;
use rlx_laguna::chat::LagunaChat;
use rlx_laguna::runner::LagunaPackedRunner;
use rlx_text::chat::ChatMessage;

fn main() -> Result<()> {
    let dir = std::env::args()
        .nth(1)
        .expect("usage: laguna_mlx_run <dir> [prompt] [n]");
    let prompt = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "What is the capital of France?".to_string());
    let n: usize = std::env::args()
        .nth(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(12);
    let raw = std::env::var("RLX_LAGUNA_RAW").is_ok();

    let t0 = std::time::Instant::now();
    let runner = LagunaPackedRunner::from_mlx_dir(&dir)?;
    eprintln!(
        "[laguna-mlx] loaded {} layers in {:.1?}",
        runner.weights().layers.len(),
        t0.elapsed()
    );
    let chat = LagunaChat::from_dir(&dir)?;

    // Default: apply the model's chat template (Laguna-2.1 is instruct-tuned);
    // RLX_LAGUNA_RAW=1 forces bare completion.
    let ids = if raw {
        chat.encode_text(&prompt)?
    } else {
        chat.encode_chat(&[ChatMessage::user(&prompt)], false)?
    };
    eprintln!("[laguna-mlx] prompt ids = {ids:?}");
    let mut new_toks: Vec<u32> = Vec::new();
    let t1 = std::time::Instant::now();
    runner.generate(&ids, n, |t| new_toks.push(t))?;
    eprintln!(
        "[laguna-mlx] generated {} tokens in {:.1?}",
        new_toks.len(),
        t1.elapsed()
    );

    let text = chat.decode(&new_toks, true).unwrap_or_default();
    let vocab = runner.config().vocab_size as u32;
    let all_valid = new_toks.iter().all(|&t| t < vocab);
    let degenerate = new_toks.len() > 2 && new_toks.windows(2).all(|w| w[0] == w[1]);

    println!("\n── Laguna mlx-community e2e ─────────────────");
    println!("prompt        : {prompt}");
    println!("generated ids : {new_toks:?}");
    println!("generated text: {text:?}");
    println!("ids valid     : {all_valid}   degenerate: {degenerate}");
    if all_valid && !degenerate && !new_toks.is_empty() {
        println!("✅ Laguna mlx checkpoint loads + runs + generates non-degenerate text");
        Ok(())
    } else {
        Err(anyhow::anyhow!(
            "laguna mlx run degenerate/invalid (all_valid={all_valid} degenerate={degenerate})"
        ))
    }
}
