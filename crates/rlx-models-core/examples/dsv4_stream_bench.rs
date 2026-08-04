// RLX — versatile ML compiler + runtime. GPLv3.
//! **Weight-streaming inference benchmark** for DeepSeek-V4 — compares the two
//! designs for running a model bigger than RAM by streaming each layer's weights
//! just-in-time (see the streaming-inference design):
//!
//!   A) COPY-STREAM — materialize each layer's packed weights into RAM
//!      (`take_packed_linear`), "use" them, free. Peak resident = working set;
//!      anon copies; IO = re-read per token.
//!   B) MMAP-PAGE  — fault each layer's weights straight from the mmap
//!      (`prewarm` parallel page-fault), read zero-copy (`fold_bytes`), then evict
//!      (`MADV_DONTNEED`). Peak resident = working set as file-backed (reclaimable)
//!      pages; no anon copy.
//!
//! Streams `--layers N` layers for `--passes P` (one pass ≈ one decode token's
//! weight traffic) and reports wall-time/pass (weight-delivery latency — the
//! streaming bottleneck; the per-layer dequant+matmul is equal for both and adds
//! on top) plus peak RSS. `--full` streams all 256 experts; default streams only
//! the attention + MoE proj tensors as whole stacked slabs.
//!
//!   dsv4_stream_bench --ckpt <dir> --layers 43 --passes 3 --mode both

use anyhow::{Context, Result};
use rlx_mlx_io::{LazyMlxWeights, load_path_lazy};
use std::time::Instant;

fn flag(a: &[String], k: &str) -> Option<String> {
    a.iter()
        .position(|x| x == k)
        .and_then(|i| a.get(i + 1))
        .cloned()
}
fn has(a: &[String], k: &str) -> bool {
    a.iter().any(|x| x == k)
}

fn peak_rss_gb() -> f64 {
    let mut u: libc::rusage = unsafe { std::mem::zeroed() };
    unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut u) };
    // macOS: ru_maxrss is bytes. Linux: kilobytes.
    #[cfg(target_os = "macos")]
    {
        u.ru_maxrss as f64 / 1e9
    }
    #[cfg(not(target_os = "macos"))]
    {
        u.ru_maxrss as f64 / 1e6
    }
}

/// The packed tensors a decode layer touches (biggest first). Only keys the
/// checkpoint actually holds are used.
fn layer_keys(il: usize, n_experts: usize, full: bool) -> Vec<String> {
    let mut v = vec![
        format!("model.layers.{il}.ffn.switch_mlp.gate_proj.weight"),
        format!("model.layers.{il}.ffn.switch_mlp.up_proj.weight"),
        format!("model.layers.{il}.ffn.switch_mlp.down_proj.weight"),
        format!("model.layers.{il}.attn.wq_a.weight"),
        format!("model.layers.{il}.attn.wq_b.weight"),
        format!("model.layers.{il}.attn.wkv.weight"),
        format!("model.layers.{il}.attn.wo_a.weight"),
        format!("model.layers.{il}.attn.wo_b.weight"),
    ];
    // Per-expert layout (Vontra reference): stream only active experts unless --full.
    let ne = if full { n_experts } else { 6.min(n_experts) };
    for e in 0..ne {
        for w in ["w1", "w3", "w2"] {
            v.push(format!("model.layers.{il}.ffn.experts.{e}.{w}.weight"));
        }
    }
    v
}

/// Bytes streamed per pass (measured from the loader's tensor sizes present).
fn measure_bytes(loader: &LazyMlxWeights, keys: &[String]) -> u64 {
    keys.iter().map(|k| loader.tensor_len_bytes(k)).sum()
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let ckpt = flag(&a, "--ckpt").context("--ckpt <dir>")?;
    let n_layers: usize = flag(&a, "--layers")
        .and_then(|s| s.parse().ok())
        .unwrap_or(43);
    let passes: usize = flag(&a, "--passes")
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);
    let mode = flag(&a, "--mode").unwrap_or_else(|| "both".into());
    let full = has(&a, "--full");
    let n_experts: usize = flag(&a, "--experts")
        .and_then(|s| s.parse().ok())
        .unwrap_or(256);
    // Batch: how many tokens ride through ONE weight-stream. The disk stream is paid
    // once/pass; the compute (reading resident weights ≈ RAM-bandwidth) is done ×batch.
    let batch: usize = flag(&a, "--batch")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);

    let mut loader = load_path_lazy(&ckpt).context("open checkpoint")?;
    // Keep only the keys this checkpoint actually holds (stacked vs per-expert).
    let all: Vec<Vec<String>> = (0..n_layers)
        .map(|il| {
            layer_keys(il, n_experts, full)
                .into_iter()
                .filter(|k| loader.tensor_len_bytes(k) > 0)
                .collect::<Vec<_>>()
        })
        .collect();
    let bytes_per_pass: u64 = all.iter().map(|ks| measure_bytes(&loader, ks)).sum();
    eprintln!(
        "[stream-bench] {ckpt}\n  layers={n_layers} passes={passes} full_experts={full} \
         → {:.1} GB/pass ({} tensors/layer avg)",
        bytes_per_pass as f64 / 1e9,
        all.iter().map(|k| k.len()).sum::<usize>() / n_layers.max(1)
    );

    // ── Mode A: copy-stream (materialize each layer, use, free) ──
    if mode == "a" || mode == "both" {
        let t0 = Instant::now();
        let mut acc = 0u64;
        for _ in 0..passes {
            for ks in &all {
                // Stream this layer's weights into RAM ONCE, then `batch` tokens
                // "use" them (read ≈ RAM-bandwidth) before they're dropped.
                for k in ks {
                    if let Ok(Some(p)) = loader.take_packed_linear(k) {
                        for _ in 0..batch {
                            acc = acc.wrapping_add(
                                p.w_q.iter().step_by(64).map(|&b| b as u64).sum::<u64>(),
                            );
                        }
                    }
                }
            }
        }
        std::hint::black_box(acc);
        let dt = t0.elapsed();
        let gbps = (bytes_per_pass * passes as u64) as f64 / 1e9 / dt.as_secs_f64();
        let per_tok = dt.as_secs_f64() / (passes * batch) as f64;
        eprintln!(
            "  [A copy-stream] {:.2}s total, {:.3}s/pass, batch={batch} → {:.3}s/token ({:.1} tok/s), \
             {:.2} GB/s stream, peak RSS {:.2} GB",
            dt.as_secs_f64(),
            dt.as_secs_f64() / passes as f64,
            per_tok,
            1.0 / per_tok,
            gbps,
            peak_rss_gb()
        );
    }

    // ── Mode B: mmap-page (fault, read zero-copy, evict) ──
    if mode == "b" || mode == "both" {
        let t0 = Instant::now();
        let mut acc = 0u64;
        for _ in 0..passes {
            for ks in &all {
                let refs: Vec<&str> = ks.iter().map(|s| s.as_str()).collect();
                loader.prewarm(&refs); // parallel page-fault the layer's ranges ONCE
                for k in ks {
                    for _ in 0..batch {
                        acc = acc.wrapping_add(loader.fold_bytes(k)); // zero-copy read, ×batch tokens
                    }
                }
                loader.evict(&refs); // MADV_DONTNEED → bound resident to working set
            }
        }
        std::hint::black_box(acc);
        let dt = t0.elapsed();
        let gbps = (bytes_per_pass * passes as u64) as f64 / 1e9 / dt.as_secs_f64();
        let per_tok = dt.as_secs_f64() / (passes * batch) as f64;
        eprintln!(
            "  [B mmap-page]  {:.2}s total, {:.3}s/pass, batch={batch} → {:.3}s/token ({:.1} tok/s), \
             {:.2} GB/s stream, peak RSS {:.2} GB",
            dt.as_secs_f64(),
            dt.as_secs_f64() / passes as f64,
            per_tok,
            1.0 / per_tok,
            gbps,
            peak_rss_gb()
        );
    }
    Ok(())
}
