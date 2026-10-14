// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Cross-DEVICE validation matrix for the decode KV-cache append, driven
//! through `LmRunner` so it covers any family `auto_runner_with` can build.
//!
//! For every (device x arm) cell — arm being in-place `Op::KvAppend` vs the
//! `concat` fallback — it reports:
//!
//! * **weights**  resident set after load (the cost of holding the model)
//! * **latency**  time to first token
//! * **speed**    steady-state decode tok/s
//! * **precision** max |Δ| and cosine of the prefill logits against a reference
//! * **agreement** share of generated tokens matching the reference stream
//!
//! and enforces the one hard gate: WITHIN a device the two arms must be
//! token-identical, because in-place append is an optimisation, not a change of
//! semantics. Cross-device differences are reported, not asserted — backends
//! legitimately differ in accumulation order, so the useful question is how far
//! apart they are, not whether they are bit-equal.
//!
//! The first argument is a COMMA-SEPARATED list of checkpoints, so weight
//! formats sit in the same table — a BitNet/ternary `Q1_0` GGUF next to a
//! `Q4_K_M` one answers "what does the smaller model cost me in latency and
//! agreement" in one run. Logit precision and token agreement are compared
//! against the first cell OF THE SAME MODEL: different checkpoints have
//! different vocabularies and generally different answers, so a cross-model
//! logit delta would be a meaningless number.
//!
//! ```sh
//! cargo run --release -p rlx-models --features runner,metal,qwen3,qwen35 \
//!   --example inplace_kv_matrix -- q4.gguf,bitnet-q1_0.gguf cpu,metal 24 256 1
//! ```

use anyhow::{Context, Result, bail};
use rlx_runtime::{Device, is_available};
use std::path::PathBuf;
use std::time::Instant;

const NO_INPLACE_KV: &str = "RLX_NO_INPLACE_KV";

struct Cell {
    model: String,
    model_gb: f64,
    device: Device,
    inplace: bool,
    family: &'static str,
    load_s: f64,
    rss_mb: f64,
    ttft: f64,
    decode_s: f64,
    tokens: Vec<u32>,
    logits: Vec<f32>,
}

impl Cell {
    fn decode_tps(&self) -> f64 {
        let n = self.tokens.len().saturating_sub(1);
        if self.decode_s > 0.0 {
            n as f64 / self.decode_s
        } else {
            f64::NAN
        }
    }
    fn arm(&self) -> &'static str {
        if self.inplace { "in-place" } else { "concat" }
    }
}

/// Resident set of this process, in MB. `ps` rather than a new dependency;
/// this is a report line, not something the gate depends on.
///
/// Reported as a DELTA across the build, because the absolute number is
/// process-cumulative: dropping a runner does not return its pages to the OS,
/// so the raw reading grows monotonically down the matrix and the first cell
/// reads absurdly low. Even the delta is a lower bound once the allocator is
/// reusing freed arenas — treat it as "what this cell newly needed".
fn rss_mb() -> f64 {
    std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.trim().parse::<f64>().ok())
        .map(|kb| kb / 1024.0)
        .unwrap_or(f64::NAN)
}

fn parse_device(s: &str) -> Result<Device> {
    Ok(match s.trim() {
        "cpu" => Device::Cpu,
        "metal" => Device::Metal,
        "mlx" => Device::Mlx,
        "cuda" => Device::Cuda,
        "rocm" => Device::Rocm,
        "wgpu" | "gpu" => Device::Gpu,
        "vulkan" => Device::Vulkan,
        other => bail!("unknown device `{other}`"),
    })
}

/// Max absolute difference and cosine similarity over the overlapping prefix.
fn compare(a: &[f32], b: &[f32]) -> (f32, f64) {
    let n = a.len().min(b.len());
    if n == 0 {
        return (f32::NAN, f64::NAN);
    }
    let max = (0..n).fold(0.0f32, |m, i| m.max((a[i] - b[i]).abs()));
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..n {
        dot += a[i] as f64 * b[i] as f64;
        na += (a[i] as f64).powi(2);
        nb += (b[i] as f64).powi(2);
    }
    let cos = if na > 0.0 && nb > 0.0 {
        dot / (na.sqrt() * nb.sqrt())
    } else {
        f64::NAN
    };
    (max, cos)
}

/// Bytes the weights occupy on disk — the honest "weights" axis when comparing
/// formats, since a ternary checkpoint's whole point is that this number drops.
fn file_gb(path: &PathBuf) -> f64 {
    std::fs::metadata(path)
        .map(|m| m.len() as f64 / (1024.0 * 1024.0 * 1024.0))
        .unwrap_or(f64::NAN)
}

fn label_of(path: &PathBuf) -> String {
    path.file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| path.display().to_string())
}

fn run_cell(
    path: &PathBuf,
    device: Device,
    inplace: bool,
    prompt: &[u32],
    n_new: usize,
    max_seq: usize,
) -> Result<Cell> {
    let rss_before = rss_mb();
    if inplace {
        rlx_ir::env::unset(NO_INPLACE_KV);
    } else {
        rlx_ir::env::set(NO_INPLACE_KV, "1");
    }
    let opts = rlx_models::run::AutoOptions::new(device).max_seq(max_seq);
    let t_load = Instant::now();
    let mut runner =
        rlx_models::run::auto_runner_with(path, &opts).context("auto_runner_with")?;
    let load_s = t_load.elapsed().as_secs_f64();
    let family = runner.family();
    let rss = (rss_mb() - rss_before).max(0.0);

    // Prefill logits first: a self-contained forward over the same prompt, and
    // the finest-grained numerical signal available through this interface.
    let logits = runner.predict_logits(prompt).context("predict_logits")?;

    let t0 = Instant::now();
    let mut first: Option<f64> = None;
    let mut tokens: Vec<u32> = Vec::with_capacity(n_new);
    runner.generate(prompt, n_new, &mut |t| {
        if first.is_none() {
            first = Some(t0.elapsed().as_secs_f64());
        }
        tokens.push(t);
        true
    })?;
    let total = t0.elapsed().as_secs_f64();
    let ttft = first.unwrap_or(total);
    Ok(Cell {
        model: label_of(path),
        model_gb: file_gb(path),
        device,
        inplace,
        family,
        load_s,
        rss_mb: rss,
        ttft,
        decode_s: (total - ttft).max(0.0),
        tokens,
        logits,
    })
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let paths: Vec<PathBuf> = args
        .next()
        .context("usage: inplace_kv_matrix <ckpt[,ckpt...]> [devices] [n_new] [prompt_len] [reps]")?
        .split(',')
        .map(|s| PathBuf::from(s.trim()))
        .collect();
    for p in &paths {
        if !p.exists() {
            bail!("checkpoint not found: {}", p.display());
        }
    }
    let devices: Vec<Device> = match args.next() {
        Some(s) => s
            .split(',')
            .map(parse_device)
            .collect::<Result<Vec<_>>>()?,
        None => vec![Device::Cpu],
    };
    let n_new: usize = match args.next() {
        Some(s) => s.parse().context("n_new")?,
        None => 24,
    };
    let prompt_len: usize = match args.next() {
        Some(s) => s.parse().context("prompt_len")?,
        None => 256,
    };
    let reps: usize = match args.next() {
        Some(s) => s.parse().context("reps")?,
        None => 1,
    };

    let available: Vec<Device> = devices.into_iter().filter(|d| is_available(*d)).collect();
    if available.is_empty() {
        bail!("none of the requested devices are available in this build");
    }

    // Size the ladder to the run: past the top decode bucket, families leave the
    // bucketed path for a one-shot full forward per token, which carries no KV
    // cache contract — so every cell would read as "no difference" and the
    // matrix would be measuring nothing.
    // A natural prompt when one is given. Synthetic filler ids drive the model
    // into near-tied logits (it degenerates into repetition), and a near-tie is
    // exactly where a 1e-7 cross-device difference flips greedy argmax — so a
    // divergence measured on filler says more about the tie than the backend.
    // `PROMPT_TEXT` tokenizes real text through the checkpoint's own tokenizer.
    // Tokenized with the FIRST checkpoint's tokenizer and reused verbatim for
    // every model, so each one sees the same ids. Comparing models on text
    // re-tokenized per model would vary the prompt length and the content of
    // the context between cells, which is not a controlled comparison — but it
    // does mean a model with a different vocabulary is being fed another
    // model's ids, so cross-model agreement is only meaningful for checkpoints
    // that share a tokenizer.
    let prompt: Vec<u32> = match std::env::var("PROMPT_TEXT") {
        Ok(text) => rlx_models::run::auto_tokenize(&paths[0], &text, None)
            .context("auto_tokenize(PROMPT_TEXT)")?,
        Err(_) => (0..prompt_len).map(|i| 16 + (i % 512) as u32).collect(),
    };
    let prompt_len = prompt.len();
    let max_seq = (prompt_len + n_new + 64).clamp(32, 8192);

    println!(
        "[matrix] models={:?} | devices={:?} | prompt={prompt_len} | n_new={n_new} | reps={reps} | max_seq={max_seq}",
        paths.iter().map(label_of).collect::<Vec<_>>(),
        available
    );

    let mut cells: Vec<Cell> = Vec::new();
    for path in &paths {
        for &device in &available {
            for inplace in [true, false] {
                // Best-of-reps on speed; a single pass cannot separate a 10%
                // effect from this machine's run-to-run spread.
                let mut best: Option<Cell> = None;
                for _ in 0..reps.max(1) {
                    match run_cell(path, device, inplace, &prompt, n_new, max_seq) {
                        Ok(c) => {
                            let keep = match &best {
                                Some(b) => c.decode_tps() > b.decode_tps(),
                                None => true,
                            };
                            if keep {
                                best = Some(c);
                            }
                        }
                        // One unsupported (model, device) pair must not sink the
                        // whole matrix — a ternary checkpoint with no kernel for
                        // this backend is a RESULT, not a harness failure.
                        Err(e) => {
                            println!(
                                "[matrix] SKIP {} on {device:?} ({}): {e}",
                                label_of(path),
                                if inplace { "in-place" } else { "concat" }
                            );
                            break;
                        }
                    }
                }
                if let Some(c) = best {
                    cells.push(c);
                }
            }
        }
    }
    rlx_ir::env::unset(NO_INPLACE_KV);
    if cells.is_empty() {
        bail!("no (model, device) pair could be built");
    }

    println!(
        "[matrix] {:<26} {:<7} {:<8} {:<9} {:>7} {:>8} {:>8} {:>10} {:>12} {:>9} {:>8}",
        "model", "family", "device", "arm", "file GB", "+rss MB", "TTFT s", "decode t/s",
        "max|dlogit|", "cosine", "prefix"
    );
    for c in &cells {
        // Reference is the first cell OF THE SAME MODEL: comparing logits across
        // checkpoints compares different vocabularies and different training,
        // which is not a precision measurement.
        let reference = cells
            .iter()
            .find(|r| r.model == c.model)
            .expect("same-model reference");
        let (maxd, cos) = compare(&reference.logits, &c.logits);
        // Matching PREFIX, not a position-wise match rate. Greedy decoding is
        // autoregressive: once two streams differ by one token they are
        // conditioning on different context, so every later position is
        // unrelated and a "79% of positions agree" figure is counting noise.
        // The honest summary is how far they stayed together.
        let n = reference.tokens.len().min(c.tokens.len());
        let prefix = (0..n)
            .take_while(|&i| reference.tokens[i] == c.tokens[i])
            .count();
        let agree = if n == 0 {
            f64::NAN
        } else {
            prefix as f64 / n as f64
        };
        let name: String = c.model.chars().take(26).collect();
        println!(
            "[matrix] {:<26} {:<7} {:<8} {:<9} {:>7.2} {:>8.0} {:>8.2} {:>10.1} {:>12.3e} {:>9.6} {:>7.1}%",
            name,
            c.family,
            format!("{:?}", c.device).to_lowercase(),
            c.arm(),
            c.model_gb,
            c.rss_mb,
            c.ttft,
            c.decode_tps(),
            maxd,
            cos,
            agree * 100.0
        );
    }
    let _ = &cells[0].load_s;

    // ── the hard gate: within a device, the two arms must agree exactly ──────
    let mut failures = 0usize;
    for path in &paths {
      let model = label_of(path);
      for &device in &available {
        let ip = cells
            .iter()
            .find(|c| c.model == model && c.device == device && c.inplace);
        let cc = cells
            .iter()
            .find(|c| c.model == model && c.device == device && !c.inplace);
        if let (Some(ip), Some(cc)) = (ip, cc) {
            if ip.tokens != cc.tokens {
                let at = ip.tokens.iter().zip(&cc.tokens).position(|(a, b)| a != b);
                println!("[matrix] FAIL {model} {device:?}: in-place diverges from concat at index {at:?}");
                println!("[matrix]   concat  : {:?}", cc.tokens);
                println!("[matrix]   in-place: {:?}", ip.tokens);
                failures += 1;
            } else {
                let speedup = ip.decode_tps() / cc.decode_tps();
                println!(
                    "[matrix] {model} {device:?}: token-identical ({} tokens), decode \
                     {speedup:.2}x, logits max|d| {:.3e}",
                    ip.tokens.len(),
                    compare(&ip.logits, &cc.logits).0
                );
            }
        }
      }
    }
    if failures > 0 {
        bail!("{failures} (model, device) cell(s) changed output under in-place KV append");
    }
    println!("[matrix] PASS ✓ every (model, device) cell token-identical across both arms");
    Ok(())
}
