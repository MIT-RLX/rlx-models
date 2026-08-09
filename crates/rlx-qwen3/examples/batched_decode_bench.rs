// batched_decode_bench — decode THROUGHPUT scaling curve (tokens/sec across a
// batch of concurrent sequences). Decode is weight-read-bandwidth bound, so
// batching reads the model ONCE and produces B tokens: aggregate tok/s should
// scale ~linearly with batch until compute saturates. This measures where.
//
//   cargo run --release -p rlx-qwen3 --features metal --example batched_decode_bench \
//       -- /Users/Shared/weights/qwen3-0.6b
//
// Uniform batched decode (all sequences at the same cache length) via
// `Qwen3Generator::decode_batched_uniform`. F32 weights (F16-resident on Metal
// with RLX_QWEN3_F16_WEIGHTS=1). Env: PROMPT (ctx len), STEPS (timed steps).

use rlx_qwen3::{Qwen3Config, Qwen3Generator};
use rlx_runtime::Device;
use std::path::PathBuf;
use std::time::Instant;

fn argmax(v: &[f32]) -> u32 {
    let mut bi = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > bv {
            bv = x;
            bi = i;
        }
    }
    bi as u32
}

fn main() -> anyhow::Result<()> {
    let dir = PathBuf::from(
        std::env::args()
            .nth(1)
            .unwrap_or_else(|| "/Users/Shared/weights/qwen3-0.6b".into()),
    );
    let device = Device::Metal;
    let cfg = Qwen3Config::from_file(&dir.join("config.json"))?;
    // F32 generator (dequant-at-load) from the GGUF; on Metal with
    // RLX_QWEN3_F16_WEIGHTS=1 the projections/LM-head are F16-resident.
    let gguf = dir.join("Qwen3-0.6B-Q4_K_M.gguf");
    let gguf_s = gguf
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("bad gguf path"))?;
    // PACKED=1 → native packed decode (Q4 DequantMatMul, ~half the weight read);
    // else F32 dequant-at-load (F16-resident on Metal).
    let packed = std::env::var("PACKED").is_ok();
    let mut g = if packed {
        Qwen3Generator::new_native_packed_decode(cfg.clone(), gguf_s, device)?
    } else {
        let mut loader = rlx_core::weight_loader::GgufLoader::from_file(gguf_s)?;
        Qwen3Generator::from_loader(cfg.clone(), &mut loader, device)?
    };

    let prompt_len: usize = std::env::var("PROMPT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(128);
    let steps: usize = std::env::var("STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(48);
    let prompt: Vec<u32> = (1..=prompt_len as u32).collect();
    g.prefill_get_last_logits(&prompt)?;
    let (base_kv, _) = g
        .export_cache()
        .ok_or_else(|| anyhow::anyhow!("no cache after prefill"))?;
    let past0 = prompt_len;

    // RESIDENT=1 → GPU-resident KV (bind once/bucket, fold new row on device),
    // else the stateless host path (re-upload full [B,ctx,kv_dim] each step).
    let resident = std::env::var("RESIDENT").is_ok();
    eprintln!(
        "[batched-bench] {}{} decode, Metal, prompt={prompt_len}, {steps} timed steps/batch",
        if packed { "PACKED" } else { "F32(F16)" },
        if resident {
            " +RESIDENT-KV"
        } else {
            " host-KV"
        }
    );
    eprintln!(
        "{:>6} {:>10} {:>9} {:>13} {:>9}",
        "batch", "tokens", "wall_s", "tok/s(agg)", "vs B=1"
    );
    // RAGGED=1: sequences at MIXED cache lengths (real-server fusion). Prefill a
    // pool of staggered-length caches, then run the ragged path (host or
    // +resident) and compare per-sequence token streams for correctness.
    if std::env::var("RAGGED").is_ok() {
        let maxb = 16usize;
        let step = (prompt_len / (2 * maxb)).max(1);
        eprintln!("[batched-bench] RAGGED (mixed lengths), pool of {maxb} staggered caches");
        let mut pool: Vec<_> = Vec::with_capacity(maxb);
        for i in 0..maxb {
            let plen = prompt_len.saturating_sub(i * step).max(8);
            let p: Vec<u32> = (1..=plen as u32).collect();
            g.prefill_get_last_logits(&p)?;
            pool.push(
                g.export_cache()
                    .ok_or_else(|| anyhow::anyhow!("no cache"))?
                    .0,
            );
        }
        for &b in &[4usize, 8, 16] {
            // Two independent runs (host, resident) from the SAME staggered caches.
            let run =
                |g: &mut Qwen3Generator, resident: bool| -> anyhow::Result<(Vec<Vec<u32>>, f64)> {
                    let caches: Vec<_> = pool[..b].to_vec();
                    let mut toks: Vec<u32> = (0..b as u32).map(|i| 1 + (i % 100)).collect();
                    let mut fps: Vec<Vec<u32>> = vec![Vec::new(); b];
                    let t0;
                    if resident {
                        let refs: Vec<&_> = caches.iter().collect();
                        g.decode_batched_ragged_resident_init(&refs);
                        let lg = g.decode_batched_ragged_resident_step(&toks)?; // warm
                        for (i, l) in lg.iter().enumerate() {
                            toks[i] = argmax(l);
                        }
                        t0 = Instant::now();
                        for _ in 0..steps {
                            let lg = g.decode_batched_ragged_resident_step(&toks)?;
                            for (i, l) in lg.iter().enumerate() {
                                toks[i] = argmax(l);
                                fps[i].push(toks[i]);
                            }
                        }
                    } else {
                        let mut caches = caches;
                        let out = {
                            let e: Vec<_> = toks
                                .iter()
                                .zip(caches.iter())
                                .map(|(t, k)| (*t, k))
                                .collect();
                            g.decode_batched_ragged(&e)?
                        };
                        for (i, (l, kv)) in out.into_iter().enumerate() {
                            toks[i] = argmax(&l);
                            caches[i] = kv;
                        }
                        t0 = Instant::now();
                        for _ in 0..steps {
                            let out = {
                                let e: Vec<_> = toks
                                    .iter()
                                    .zip(caches.iter())
                                    .map(|(t, k)| (*t, k))
                                    .collect();
                                g.decode_batched_ragged(&e)?
                            };
                            for (i, (l, kv)) in out.into_iter().enumerate() {
                                toks[i] = argmax(&l);
                                caches[i] = kv;
                                fps[i].push(toks[i]);
                            }
                        }
                    }
                    Ok((
                        fps,
                        (b * steps) as f64 / t0.elapsed().as_secs_f64().max(1e-9),
                    ))
                };
            let (fp_host, tps_host) = run(&mut g, false)?;
            let (fp_res, tps_res) = run(&mut g, true)?;
            let ok = fp_host == fp_res;
            eprintln!(
                "  B={b:>2}: host {tps_host:>6.1} tok/s | resident {tps_res:>6.1} tok/s ({:.2}x) | streams {}",
                tps_res / tps_host.max(1e-9),
                if ok { "MATCH ✓" } else { "MISMATCH ✗" }
            );
        }
        return Ok(());
    }

    let batches = [1usize, 2, 4, 8, 16, 32];
    let mut base_tps = 0.0f64;
    for &b in &batches {
        let mut kvs: Vec<_> = (0..b).map(|_| base_kv.clone()).collect();
        let mut toks: Vec<u32> = (0..b as u32).map(|i| 1 + (i % 100)).collect();
        let mut past = past0;
        let mut seq0: Vec<u32> = Vec::new(); // B=1 token fingerprint (correctness)
        let t0;
        if resident {
            let refs: Vec<&_> = kvs.iter().collect();
            g.decode_batched_resident_init(&refs);
            // Warm: compile + bind (not timed).
            let logits = g.decode_batched_resident_step(&toks, past, past)?;
            past += 1;
            for (i, lo) in logits.iter().enumerate() {
                toks[i] = argmax(lo);
            }
            t0 = Instant::now();
            for _ in 0..steps {
                let logits = g.decode_batched_resident_step(&toks, past, past)?;
                past += 1;
                for (i, lo) in logits.iter().enumerate() {
                    toks[i] = argmax(lo);
                }
                if b == 1 {
                    seq0.push(toks[0]);
                }
            }
        } else {
            // Warm: compile the B-shaped decode bucket (not timed).
            let out = {
                let entries: Vec<_> = toks
                    .iter()
                    .zip(kvs.iter())
                    .map(|(t, kv)| (*t, kv))
                    .collect();
                g.decode_batched_uniform(&entries, past, past)?
            };
            past += 1;
            for (i, (logits, kv)) in out.into_iter().enumerate() {
                toks[i] = argmax(&logits);
                kvs[i] = kv;
            }
            t0 = Instant::now();
            for _ in 0..steps {
                let out = {
                    let entries: Vec<_> = toks
                        .iter()
                        .zip(kvs.iter())
                        .map(|(t, kv)| (*t, kv))
                        .collect();
                    g.decode_batched_uniform(&entries, past, past)?
                };
                past += 1;
                for (i, (logits, kv)) in out.into_iter().enumerate() {
                    toks[i] = argmax(&logits);
                    kvs[i] = kv;
                }
                if b == 1 {
                    seq0.push(toks[0]);
                }
            }
        }
        if b == 1 {
            eprintln!(
                "  [B=1 token fingerprint] {:?}",
                &seq0[..seq0.len().min(16)]
            );
        }
        let secs = t0.elapsed().as_secs_f64();
        let agg = (b * steps) as f64 / secs.max(1e-9);
        if b == 1 {
            base_tps = agg;
        }
        eprintln!(
            "{:>6} {:>10} {:>9.3} {:>13.1} {:>8.2}x",
            b,
            b * steps,
            secs,
            agg,
            agg / base_tps.max(1e-9)
        );
    }
    Ok(())
}
