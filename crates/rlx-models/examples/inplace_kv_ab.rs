// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Family-agnostic A/B of the decode KV-cache append, driven through `LmRunner`.
//!
//! Builds the SAME checkpoint twice through `auto_runner_with` — once with the
//! in-place append ON (`Op::KvAppend`, the default) and once forced back to
//! `concat` (`RLX_NO_INPLACE_KV=1`) — generates greedily from an identical
//! prompt, asserts the two token streams are IDENTICAL, and reports each one's
//! decode throughput.
//!
//! The point is that it knows nothing about any family. Whatever
//! `auto_runner_with` dispatches to is what gets measured, so a model crate is
//! covered here the moment it migrates its flow to the cache contract — and a
//! family that has NOT migrated shows up as a ~0% delta rather than as silence.
//! That is the check the original in-place KV work lacked: it landed, the
//! callers were never migrated, and nothing anywhere noticed it had stopped
//! being emitted.
//!
//! ```sh
//! cargo run --release -p rlx-models --features runner,metal,qwen3 \
//!   --example inplace_kv_ab -- /path/to/model.gguf metal 48 512
//! ```

use anyhow::{Context, Result, bail};
use rlx_runtime::{Device, is_available};
use std::path::PathBuf;
use std::time::Instant;

/// The shared opt-out, honoured by every migrated decode layer.
const NO_INPLACE_KV: &str = "RLX_NO_INPLACE_KV";

struct Run {
    family: &'static str,
    tokens: Vec<u32>,
    ttft: f64,
    decode_s: f64,
}

impl Run {
    fn decode_tps(&self) -> f64 {
        let n = self.tokens.len().saturating_sub(1);
        if self.decode_s > 0.0 {
            n as f64 / self.decode_s
        } else {
            f64::NAN
        }
    }
}

fn parse_device(s: &str) -> Result<Device> {
    Ok(match s {
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

/// One build + generate. A fresh runner each time so the decode graph is
/// recompiled under the current flag rather than reused from the other arm.
fn one(path: &PathBuf, device: Device, prompt: &[u32], n_new: usize) -> Result<Run> {
    let opts = rlx_models::run::AutoOptions::new(device);
    let mut runner =
        rlx_models::run::auto_runner_with(path, &opts).context("auto_runner_with")?;
    let family = runner.family();

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
    Ok(Run {
        family,
        tokens,
        ttft,
        decode_s: (total - ttft).max(0.0),
    })
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let path = PathBuf::from(args.next().context(
        "usage: inplace_kv_ab <checkpoint> [device] [n_new] [prompt_len] [reps]",
    )?);
    let device = match args.next() {
        Some(s) => parse_device(&s)?,
        None => Device::Cpu,
    };
    let n_new: usize = match args.next() {
        Some(s) => s.parse().context("n_new")?,
        None => 32,
    };
    let prompt_len: usize = match args.next() {
        Some(s) => s.parse().context("prompt_len")?,
        None => 256,
    };
    let reps: usize = match args.next() {
        Some(s) => s.parse().context("reps")?,
        None => 3,
    };
    if !is_available(device) {
        bail!("device {device:?} is not available in this build");
    }

    // Families size their decode bucket ladder from `max_seq`, whose default is
    // short (128 for qwen3 → buckets to 256). A prompt past the top bucket does
    // not just run slower, it leaves the bucketed decode path entirely for a
    // one-shot full forward per token — which has no KV cache contract, so the
    // A/B would compare two identical `concat`-free paths and read ~1.00x for
    // every family. Size the ladder to the run unless the caller pinned it.
    if rlx_ir::env::var("RLX_LM_MAX_SEQ").is_none() {
        let need = prompt_len + n_new + 64;
        rlx_ir::env::set("RLX_LM_MAX_SEQ", need.to_string());
        println!("[ab] RLX_LM_MAX_SEQ={need} (sized to the run)");
    }

    // Deterministic filler prompt as raw ids, so the harness needs no tokenizer
    // and works for every family. Ids stay low enough for any real vocab.
    let prompt: Vec<u32> = (0..prompt_len).map(|i| 16 + (i % 512) as u32).collect();

    println!(
        "[ab] {} | device={device:?} | prompt={prompt_len} ids | n_new={n_new} | reps={reps}",
        path.display()
    );

    // Alternate the arms and keep the BEST of each. One pass each is not enough
    // to call a <20% difference: back-to-back runs of this same binary and
    // config have come out 6.8 and 14.1 tok/s, so a single ordering can invert
    // the verdict. Best-of-N also means whichever arm happens to run cold or
    // hot cannot decide the result.
    let mut best: [Option<Run>; 2] = [None, None];
    let mut reference: Option<Vec<u32>> = None;
    let mut family = "?";
    for rep in 0..reps.max(1) {
        for arm in 0..2 {
            let inplace = arm == 0;
            if inplace {
                rlx_ir::env::unset(NO_INPLACE_KV);
            } else {
                rlx_ir::env::set(NO_INPLACE_KV, "1");
            }
            let r = one(&path, device, &prompt, n_new)?;
            family = r.family;

            // Every run must agree with the first, whichever arm it came from.
            match &reference {
                None => reference = Some(r.tokens.clone()),
                Some(expect) if *expect != r.tokens => {
                    let at = expect.iter().zip(&r.tokens).position(|(a, b)| a != b);
                    println!("[ab] reference: {expect:?}");
                    println!("[ab] this run : {:?}", r.tokens);
                    bail!(
                        "FAIL: {} KV append changed the output at index {at:?} (rep {rep}) — \
                         the in-place append must be token-identical to concat",
                        if inplace { "in-place" } else { "concat" }
                    );
                }
                _ => {}
            }
            let keep = match &best[arm] {
                Some(b) => r.decode_tps() > b.decode_tps(),
                None => true,
            };
            if keep {
                best[arm] = Some(r);
            }
        }
    }
    rlx_ir::env::unset(NO_INPLACE_KV);

    let inplace = best[0].take().context("no in-place run")?;
    let concat = best[1].take().context("no concat run")?;
    println!("[ab] family={family}");
    for (label, r) in [("concat  ", &concat), ("in-place", &inplace)] {
        println!(
            "[ab]   {label}: TTFT {:.3}s | {} tokens | decode {:.1} tok/s (best of {reps})",
            r.ttft,
            r.tokens.len(),
            r.decode_tps()
        );
    }

    let speedup = inplace.decode_tps() / concat.decode_tps();
    println!(
        "[ab] PASS ✓ token-identical ({} tokens, {} runs) | decode {:.2}x",
        inplace.tokens.len(),
        reps.max(1) * 2,
        speedup
    );
    if speedup < 1.05 {
        println!(
            "[ab] NOTE: no measurable gain. Either family `{family}` has not migrated its \
             decode flow to `bind_decode_inputs_with_capacity` (the cache shape, the \
             contract and the host KV pad all have to move together — see rlx-flow \
             blocks::kv_cache), or this prompt is short enough that the O(context) concat \
             is not yet a large share of decode. The saving grows with context."
        );
    }
    Ok(())
}
