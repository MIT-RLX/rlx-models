//! prefix_cache_bench — exact prefix caching (prototype 1).
//!
//! Precompute a shared prompt prefix's KV once, then reuse it across generations:
//! only the (short) suffix is prefilled. Zero quality loss (the reused rows are
//! byte-identical to a full prefill — see the `prefill_with_reuse_matches_full_prefill`
//! parity test). Measures TTFT (prefill-to-first-logits) COLD (full prefill) vs
//! WARM (prefix reused, suffix-only), across prefix lengths.
//!
//! Run:
//!   cargo run --release -p rlx-qwen3 --example prefix_cache_bench --features metal \
//!       -- /Users/Shared/weights/qwen3-0.6b

use rlx_qwen3::{Qwen3Runner, SampleOpts};
use rlx_runtime::Device;
use std::path::PathBuf;
use std::time::Instant;
use tokenizers::Tokenizer;

fn main() -> anyhow::Result<()> {
    let weights = PathBuf::from(
        std::env::args()
            .nth(1)
            .unwrap_or_else(|| "/Users/Shared/weights/qwen3-0.6b".to_string()),
    );
    // Metal decode fast path (f16-resident weights), matching the shipped default.
    for k in [
        "RLX_QWEN3_F16_WEIGHTS",
        "RLX_QWEN3_BAKE_WEIGHTS",
        "RLX_QWEN3_GQA_NATIVE",
    ] {
        if std::env::var_os(k).is_none() {
            unsafe { std::env::set_var(k, "1") };
        }
    }
    let tok = Tokenizer::from_file(weights.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
    eprintln!("[prefix-cache] loading qwen3-0.6b (safetensors, F32 path) on Metal …");
    let mut runner = Qwen3Runner::builder()
        .weights(weights.clone())
        .device(Device::Metal)
        .format(rlx_cli::WeightFormat::Safetensors)
        .packed_weights(false)
        .max_seq(8192)
        .sample(SampleOpts::greedy())
        .build()?;

    // Long shared "system prompt" prefix + a short user "query" suffix — the
    // canonical prefix-cache case (fixed system/RAG context, varying question).
    let base = "You are a careful, knowledgeable assistant. Read the provided context \
                and follow every instruction exactly, citing sources where relevant. ";
    let long = base.repeat(400);
    let all_ids = tok
        .encode(long, false)
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .get_ids()
        .to_vec();
    let suffix_ids = tok
        .encode(
            "\n\nUser: What is the capital of France?\nAssistant:",
            false,
        )
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .get_ids()
        .to_vec();
    eprintln!(
        "[prefix-cache] prefix pool={} tokens, suffix={} tokens",
        all_ids.len(),
        suffix_ids.len()
    );

    // Measure TTFT (prefill → first token). COLD = full prefill of prefix+suffix.
    // WARM = prefix already resident in the KV cache; feed only the suffix delta
    // via the fast GPU continuation path (feed_continuation), NOT the host-decode
    // reuse path. This is what a real prefix cache buys: pay the prefix prefill
    // once, then every query pays only its own (short) suffix.
    // Cold = full prefill of prefix+suffix; warm = cache_prefix once then
    // generate_with_prefix (suffix-only, fast GPU path). Generate N tokens so we
    // can verify the warm stream is TOKEN-IDENTICAL to cold (exact prefix cache).
    let n = 8usize;
    println!(
        "{:>8} {:>7} {:>11} {:>11} {:>9}  {:>6}",
        "prefix", "suffix", "cold_ms", "warm_ms", "speedup", "parity"
    );
    for &plen in &[1024usize, 2048, 4096] {
        if plen > all_ids.len() {
            continue;
        }
        let prefix: Vec<u32> = all_ids[..plen].to_vec();
        let full: Vec<u32> = prefix.iter().chain(&suffix_ids).copied().collect();

        // COLD: fresh full prefill of prefix+suffix + n tokens.
        runner.reset_cache();
        let mut cold_toks = Vec::new();
        let t = Instant::now();
        runner.generate(&full, n, |tk| cold_toks.push(tk))?;
        let cold = t.elapsed().as_secs_f64() * 1e3;

        // Precompute the prefix snapshot once (amortized across all queries).
        let snap = runner.cache_prefix(&prefix)?;

        // WARM: reuse the prefix, replay only the suffix on the fast path + n.
        let mut warm_toks = Vec::new();
        let t = Instant::now();
        runner.generate_with_prefix_stoppable(&snap, &full, n, |tk| {
            warm_toks.push(tk);
            true
        })?;
        let warm = t.elapsed().as_secs_f64() * 1e3;

        let parity = if cold_toks == warm_toks {
            "PASS"
        } else {
            "FAIL"
        };
        println!(
            "{:>8} {:>7} {:>11.1} {:>11.1} {:>8.1}x  {:>6}",
            plen,
            suffix_ids.len(),
            cold,
            warm,
            cold / warm.max(1e-6),
            parity
        );
    }
    Ok(())
}
