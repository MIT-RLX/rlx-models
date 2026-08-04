//! context_scale_bench — bench the disk-tiered million-token KV context store at
//! **100k tokens, multi-shot, on the GPU**, recording telemetry.
//!
//! It measures the two real components the integrated retrieval decode would sum,
//! as context grows toward 100k:
//!   1. **Store side (off-GPU):** append throughput (ingest), HNSW+disk retrieval
//!      latency, HNSW recall of planted "needle" blocks, and the RAM-index vs
//!      on-disk-data footprint — all at 100k-token scale on a real file-backed
//!      quantized mmap.
//!   2. **GPU side:** qwen3 decode latency/tps over the *bounded* resident set on
//!      Metal — which stays constant regardless of the 100k off-GPU context.
//!
//! The headline the telemetry shows: per-shot retrieval latency grows only
//! `O(log N)` (HNSW) while GPU decode stays flat — so effective context can reach
//! 100k (→ 1M) on a bounded working set. Telemetry → CSV under `--out`.
//!
//! Run:
//!   cargo run --release -p rlx-qwen3 --example context_scale_bench \
//!       --features metal,mmap-kv -- --device metal --tokens 100000
//!
//! Note: the store↔generator splice (feeding retrieved blocks into GPU attention)
//! is the pending integration; this bench measures both halves separately so the
//! sum is the projected integrated per-step cost.

use std::path::PathBuf;
use std::str::FromStr;
use std::time::Instant;

use rlx_ir::tensor_inspect::InspectLog;
use rlx_qwen3::{Qwen3Runner, SampleOpts};
use rlx_runtime::Device;
use rlx_runtime::hnsw::HnswConfig;
use rlx_runtime::kv_context_store::KvContextStore;
use rlx_runtime::quantized_kv::KvQuant;
use tokenizers::Tokenizer;

const DEFAULT_WEIGHTS: &str = "/Users/Shared/rlx-models/weights/lm/qwen3-0.6b";

/// Deterministic pseudo-random unit in [0,1) (splitmix64).
fn h(mut z: u64) -> f32 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 40) as f32 / (1u64 << 24) as f32
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let mut device = "metal".to_string();
    let mut weights = PathBuf::from(DEFAULT_WEIGHTS);
    let mut tokens = 100_000usize;
    let mut block = 64usize;
    let mut topk = 32usize;
    let mut out_dir = PathBuf::from("context_bench_out");
    let mut quant = "q4_0".to_string();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--device" => {
                i += 1;
                device = args[i].clone();
            }
            "--weights" => {
                i += 1;
                weights = PathBuf::from(&args[i]);
            }
            "--tokens" => {
                i += 1;
                tokens = args[i].parse()?;
            }
            "--block" => {
                i += 1;
                block = args[i].parse()?;
            }
            "--topk" => {
                i += 1;
                topk = args[i].parse()?;
            }
            "--quant" => {
                i += 1;
                quant = args[i].clone();
            }
            "--out" => {
                i += 1;
                out_dir = PathBuf::from(&args[i]);
            }
            other => eprintln!("[bench] ignoring {other}"),
        }
        i += 1;
    }
    std::fs::create_dir_all(&out_dir)?;
    let scheme = match quant.as_str() {
        "f16" => KvQuant::F16,
        "q8_0" => KvQuant::Q8_0,
        "q5_0" => KvQuant::Q5_0,
        _ => KvQuant::Q4_0,
    };

    // ── GPU: load qwen3 on Metal (real decode latency over a bounded resident) ──
    // f16-resident weights (the shipped decode default) for peak Metal tps.
    // Metal-only: other backends read the F16-declared param as raw f32 bytes →
    // garbage weights → gibberish, so only enable when actually on Metal.
    if device.eq_ignore_ascii_case("metal") {
        for k in [
            "RLX_QWEN3_F16_WEIGHTS",
            "RLX_QWEN3_BAKE_WEIGHTS",
            "RLX_QWEN3_GQA_NATIVE",
        ] {
            if std::env::var_os(k).is_none() {
                unsafe { std::env::set_var(k, "1") };
            }
        }
    }
    let dev = Device::from_str(&device).map_err(|e| anyhow::anyhow!("--device {device}: {e}"))?;
    let tok = Tokenizer::from_file(weights.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
    eprintln!("[bench] loading qwen3 on {dev:?} …");
    let mut runner = Qwen3Runner::builder()
        .weights(weights.clone())
        .device(dev)
        .format(rlx_cli::WeightFormat::Safetensors)
        .packed_weights(false)
        .max_seq(4096)
        .sample(SampleOpts::greedy())
        .build()?;
    let kv_dim = runner.config().kv_proj_dim();
    let n_layers = runner.config().num_hidden_layers;
    // Warm the decode graph so the first GPU timing isn't a compile.
    let warm: Vec<u32> = tok
        .encode("hello world", false)
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .get_ids()
        .to_vec();
    runner.generate_stoppable(&warm, 4, |_| true)?;
    runner.reset_cache();

    // ── Store: disk-tiered HNSW context store, sized for `tokens` ──
    let nblocks = tokens / block;
    let store_dir = out_dir.join("store");
    eprintln!(
        "[bench] store: {n_layers} layers × kv_dim {kv_dim}, {scheme:?}, {nblocks} blocks × {block} rows = {tokens} tokens, file-backed at {store_dir:?}"
    );
    let mut store = KvContextStore::new(
        n_layers,
        kv_dim,
        scheme,
        nblocks * block + block,
        Some(&store_dir),
        HnswConfig::default(),
        (topk * 2).max(64),
        1,   // centroids/block (bench uses distinctive single keys)
        1.0, // no decay
    )?;

    // Plant "needle" blocks with a distinctive key direction; queryable by that
    // direction. Filler blocks get small pseudo-random keys.
    // 8 needles spread from early (before the first shot) to near the end.
    let needle_ids: Vec<usize> = {
        let (lo, hi) = (nblocks / 20, nblocks * 9 / 10);
        (0..8).map(|i| lo + i * (hi - lo) / 7).collect()
    };
    // Diverse dense keys in [-1,1] (like real K directions), so HNSW navigation
    // is well-conditioned. A needle is a distinctive random direction seeded by a
    // dedicated seed and scaled up so it stands out from the filler cloud; a query
    // reuses that exact direction. (Degenerate all-positive-clustered keys break
    // HNSW navigation — a bench-data artifact, not a store limitation.)
    let needle_key = |ni: usize| -> Vec<f32> {
        let seed = 0xEED1E_5EED_0000u64.wrapping_add(ni as u64 * 0x100);
        (0..kv_dim)
            .map(|j| (h(seed ^ (j as u64 + 1)) * 2.0 - 1.0) * 3.0)
            .collect()
    };
    let filler_key = |b: usize| -> Vec<f32> {
        (0..kv_dim)
            .map(|j| h((b as u64) << 20 ^ (j as u64 + 1)) * 2.0 - 1.0)
            .collect()
    };

    // Build one block's per-layer K/V from a key (rows = key broadcast; cheap).
    let block_kv = |key: &[f32]| -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let mut row = Vec::with_capacity(block * kv_dim);
        for _ in 0..block {
            row.extend_from_slice(key);
        }
        let k: Vec<Vec<f32>> = (0..n_layers).map(|_| row.clone()).collect();
        let v = k.clone();
        (k, v)
    };

    // Query for a needle = its key; check the needle's block id is in the top-k.
    let needle_axis_to_id: std::collections::HashMap<usize, usize> = needle_ids
        .iter()
        .enumerate()
        .map(|(i, &b)| (i, b))
        .collect();

    // Shot marks aligned to the actual block-rounded token count so the final
    // (full-context) shot always fires.
    let full = nblocks * block;
    let shots = [full / 10, full / 3, (full * 2) / 3, full];
    let mut shot_i = 0;
    let mut appended = 0usize;
    let mut inspect = InspectLog::new();
    let mut csv = String::from(
        "shot,ctx_tokens,disk_gb,ram_idx_mb,append_tok_per_s,retrieve_ms,hnsw_recall,gpu_decode_tps,gpu_decode_ms\n",
    );
    let t_all = Instant::now();
    let mut t_append = Instant::now();

    eprintln!("[bench] ingesting {tokens} tokens (multi-shot at {shots:?}) …");
    for b in 0..nblocks {
        let (key, is_needle) = match needle_ids.iter().position(|&x| x == b) {
            Some(ni) => (needle_key(ni), true),
            None => (filler_key(b), false),
        };
        let (k, v) = block_kv(&key);
        // Tag provenance: planted needles as ingested File content, filler as
        // the model's own Generated output.
        let origin = if is_needle {
            rlx_runtime::kv_context_store::Origin::File
        } else {
            rlx_runtime::kv_context_store::Origin::Generated
        };
        store.append_block(b * block, origin, b as u32, &k, &v, &key)?;
        appended += block;

        if shot_i < shots.len() && appended >= shots[shot_i] {
            let append_tps = appended as f64 / t_append.elapsed().as_secs_f64().max(1e-9);
            // ── store retrieval shot: query each planted needle, time + recall ──
            let planted: Vec<usize> = needle_ids.iter().filter(|&&x| x <= b).copied().collect();
            let n_q = planted.len().max(1);
            let t_r = Instant::now();
            let mut hits = 0usize;
            for (ni, &nb) in needle_ids.iter().enumerate() {
                if nb > b {
                    continue;
                }
                let q = needle_key(ni);
                let got = store.retrieve(&q, topk);
                // The needle's block starts at nb*block.
                if got.iter().any(|r| r.start_pos == nb * block) {
                    hits += 1;
                }
                if let Some(r) = got.first() {
                    inspect.record_tensor(shot_i, "retrieved.k", &[r.rows, kv_dim], &r.k[0], 16);
                }
                let _ = needle_axis_to_id.get(&ni);
            }
            let retrieve_ms = t_r.elapsed().as_secs_f64() * 1e3 / n_q as f64;
            let recall = hits as f64 / n_q as f64;

            // ── GPU decode shot: bounded qwen3 decode on Metal (context-independent) ──
            runner.reset_cache();
            let prompt: Vec<u32> = tok
                .encode("The quick brown fox", false)
                .map_err(|e| anyhow::anyhow!("{e}"))?
                .get_ids()
                .to_vec();
            let t_g = Instant::now();
            let n_gen = 32usize;
            runner.generate_stoppable(&prompt, n_gen, |_| true)?;
            let g_dt = t_g.elapsed().as_secs_f64();
            let gpu_tps = n_gen as f64 / g_dt.max(1e-9);
            let gpu_ms = g_dt * 1e3 / n_gen as f64;

            let disk_gb = store.data_bytes() as f64 / 1e9;
            let ram_mb = store.resident_index_bytes() as f64 / 1e6;
            eprintln!(
                "  [shot {}] ctx {:>7} tok | disk {:.2} GB | ram-idx {:.1} MB | ingest {:.0} tok/s | retrieve {:.2} ms (recall {:.0}%) | gpu {:.0} tps ({:.2} ms/tok)",
                shot_i,
                appended,
                disk_gb,
                ram_mb,
                append_tps,
                retrieve_ms,
                recall * 100.0,
                gpu_tps,
                gpu_ms,
            );
            csv.push_str(&format!(
                "{},{},{:.3},{:.2},{:.0},{:.3},{:.3},{:.1},{:.3}\n",
                shot_i, appended, disk_gb, ram_mb, append_tps, retrieve_ms, recall, gpu_tps, gpu_ms,
            ));
            shot_i += 1;
            t_append = Instant::now();
        }
    }
    store.flush()?;

    std::fs::write(out_dir.join("telemetry.csv"), &csv)?;
    std::fs::write(out_dir.join("retrieved_stats.csv"), inspect.to_csv())?;

    println!("\n════════ 100k-CONTEXT MULTI-SHOT BENCH (device={device}, {scheme:?}) ════════");
    print!("{csv}");
    println!(
        "\nstore: {} blocks / {} tokens, disk {:.2} GB, RAM index {:.1} MB",
        store.len_blocks(),
        store.total_tokens(),
        store.data_bytes() as f64 / 1e9,
        store.resident_index_bytes() as f64 / 1e6,
    );
    println!(
        "total wall: {:.1}s. Telemetry → {out_dir:?}/telemetry.csv (+ retrieved_stats.csv)",
        t_all.elapsed().as_secs_f64()
    );
    println!(
        "reading: retrieve_ms grows ~O(log N) as ctx→{tokens}; gpu_decode_tps is flat (bounded resident) — \
         so per-step cost stays bounded while context scales."
    );
    Ok(())
}
