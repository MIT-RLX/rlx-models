//! retrieval_decode_splice — integrated HNSW store ↔ GPU-attention splice.
//!
//! Unlike context_scale_bench (which times retrieval and a *fresh* decode
//! separately), this actually GENERATES through the wired splice: with the KV
//! store enabled, `apply_retention` evicts old rows to the disk-tiered HNSW
//! store, retrieves the top-k query-relevant blocks each step, and rebinds them
//! into the decode graph's past_k/past_v inputs — so decode runs over a BOUNDED
//! resident set (sinks + recent + retrieved) regardless of total context.
//!
//! Measures steady-state decode tps WITH the store (bounded, retrieval-spliced)
//! vs WITHOUT (full O(context) attention), at growing prompt lengths — the real
//! fused per-step cost, and whether decode stays flat as context scales.
//!
//! Run (compare the two):
//!   STORE=0 cargo run --release -p rlx-qwen3 --example retrieval_decode_splice --features metal,mmap-kv
//!   STORE=1 ... (same)

use rlx_qwen3::{KvStoreConfig, Qwen3Runner, SampleOpts};
use rlx_runtime::Device;
use std::path::PathBuf;
use std::time::Instant;
use tokenizers::Tokenizer;

fn main() -> anyhow::Result<()> {
    let weights = PathBuf::from("/Users/Shared/weights/qwen3-0.6b");
    for k in [
        "RLX_QWEN3_F16_WEIGHTS",
        "RLX_QWEN3_BAKE_WEIGHTS",
        "RLX_QWEN3_GQA_NATIVE",
    ] {
        if std::env::var_os(k).is_none() {
            unsafe { std::env::set_var(k, "1") };
        }
    }
    let store_on = std::env::var("STORE").ok().as_deref() == Some("1");
    let plen: usize = std::env::var("PROMPT_LEN")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2048);
    let n_gen: usize = std::env::var("N_GEN")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(40);

    let tok = Tokenizer::from_file(weights.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
    // PACKED=1 → native-packed Q4 GGUF decode (the fast path; splice runs via
    // step_cached→apply_retention). Else F32 safetensors.
    let packed = std::env::var("PACKED").ok().as_deref() == Some("1");
    let (wpath, fmt) = if packed {
        (
            PathBuf::from("/Users/Shared/weights/qwen3-0.6b-gguf/Qwen3-0.6B-Q4_K_M.gguf"),
            rlx_cli::WeightFormat::Gguf,
        )
    } else {
        (weights.clone(), rlx_cli::WeightFormat::Safetensors)
    };
    let mut runner = Qwen3Runner::builder()
        .weights(wpath)
        .device(Device::Metal)
        .format(fmt)
        .packed_weights(packed)
        .max_seq(8192)
        .sample(SampleOpts::greedy())
        .build()?;

    if store_on {
        // Bounded resident window (sinks + recent), block-quantized disk store,
        // top-k retrieval scored by the real query (query_scoring) so the splice
        // feeds attention the most Q·K-relevant past blocks.
        let dir = std::env::temp_dir().join("rlx_splice_store");
        let _ = std::fs::remove_dir_all(&dir);
        let envn = |k: &str, d: usize| {
            std::env::var(k)
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(d)
        };
        let (block, sinks, recent, topk) = (
            envn("BLOCK", 16),
            envn("SINKS", 4),
            envn("RECENT", 64),
            envn("TOPK", 8),
        );
        eprintln!("[splice] resident≈sinks{sinks}+recent{recent}+topk{topk}×block{block}");
        runner.enable_kv_store(
            KvStoreConfig::new()
                .dir(dir)
                .capacity_tokens(1 << 20)
                .block(block)
                .sinks(sinks)
                .recent(recent)
                .topk(topk)
                .query_scoring(envn("QSCORE", 1) == 1),
        )?;
    }

    // Long prompt: filler that overflows the resident window so decode must
    // rely on the store's retrieval to see anything beyond the last ~68 tokens.
    let base = "In the archives it is recorded that the treaty was signed in the year \
                fourteen ninety two by the assembled delegates. ";
    let pool = tok
        .encode(base.repeat(400), false)
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .get_ids()
        .to_vec();
    let prompt: Vec<u32> = pool[..plen.min(pool.len())].to_vec();

    // Warm (prefill + first decode step — first step still sees full context and
    // triggers the first eviction/retrieval; time the STEADY-STATE steps after).
    let mut count = 0usize;
    let mut t0 = Instant::now();
    let mut first = true;
    let mut timed = 0usize;
    runner.generate(&prompt, n_gen, |_tok| {
        if first {
            first = false;
            t0 = Instant::now(); // reset clock after the seed step
        } else {
            timed += 1;
        }
        count += 1;
    })?;
    let secs = t0.elapsed().as_secs_f64();
    let tps = timed as f64 / secs.max(1e-9);
    let stats = runner.kv_store_stats();
    println!(
        "STORE={} prompt={} gen={} : {:.2} tok/s (steady {} steps in {:.2}s) | store={:?}",
        store_on as u8, plen, count, tps, timed, secs, stats
    );
    Ok(())
}
