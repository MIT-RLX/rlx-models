// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! Ling 3.0 prefill benchmark: TTFT and prompt-processing throughput on real
//! weights.
//!
//! ```sh
//! cargo run --release -p rlx-ling --example bench -- \
//!     --weights /Volumes/FOUR/weights/Ling-3.0-tiny --device cpu --seq 64
//! ```
//!
//! **TTFT** here is prompt-processing + first-token argmax on an already-compiled
//! graph — it excludes weight load and graph compile, which are reported
//! separately because they are one-time and dwarf a single forward.
//!
//! Sustained generation throughput (decode TPS) is **not** reported: this crate
//! builds a fixed-length prefill graph with no KV cache, so every extra token
//! would re-run the whole prompt. Reporting that as "TPS" would understate real
//! decode by more than an order of magnitude. What is reported instead is
//! prefill throughput (prompt tokens/s), which is a real, comparable number.

use anyhow::{Context, Result};
use rlx_core::weight_map::WeightMap;
use rlx_ling::{LingConfig, build_ling_text_flow, prepare_checkpoint};
use rlx_runtime::Device;
use std::path::PathBuf;
use std::time::Instant;

/// Peak resident set (`ru_maxrss` — a high-water mark, never decreases).
fn peak_rss_gb() -> f64 {
    // maxrss is bytes on macOS, kilobytes on Linux.
    let mut u: libc::rusage = unsafe { std::mem::zeroed() };
    if unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut u) } != 0 {
        return f64::NAN;
    }
    let scale = if cfg!(target_os = "macos") { 1e9 } else { 1e6 };
    u.ru_maxrss as f64 / scale
}

/// Current resident set. Distinguishing this from the peak is the whole point:
/// a transient 2× spike and a permanently 2×-too-large footprint call for
/// completely different fixes.
fn cur_rss_gb() -> f64 {
    let pid = std::process::id();
    let out = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &pid.to_string()])
        .output();
    match out {
        Ok(o) => String::from_utf8_lossy(&o.stdout)
            .trim()
            .parse::<f64>()
            .map(|kb| kb / 1e6)
            .unwrap_or(f64::NAN),
        Err(_) => f64::NAN,
    }
}

/// `cur / peak` formatted for a progress line.
fn rss() -> String {
    format!("RSS {:.1} GB (peak {:.1})", cur_rss_gb(), peak_rss_gb())
}

struct Args {
    mxfp4: bool,
    f32_head: bool,
    f16_head: bool,
    inplace_scan: bool,
    decode: usize,
    stream: bool,
    weights: PathBuf,
    device: Device,
    seq: usize,
    reps: usize,
    prompt: Option<String>,
}

fn parse_args() -> Result<Args> {
    let mut weights = PathBuf::from("/Volumes/FOUR/weights/Ling-3.0-tiny");
    let mut device = Device::Cpu;
    let mut seq = 64usize;
    let mut reps = 3usize;
    let mut prompt = None;
    let mut stream = false;
    let mut decode = 0usize;
    let mut inplace_scan = false;
    let mut f16_head = false;
    let mut mxfp4 = false;
    let mut f32_head = false;
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        match argv[i].as_str() {
            "--weights" => {
                weights = PathBuf::from(&argv[i + 1]);
                i += 2;
            }
            "--device" => {
                device = rlx_cli::parse_device(&argv[i + 1])?;
                i += 2;
            }
            "--seq" => {
                seq = argv[i + 1].parse()?;
                i += 2;
            }
            "--reps" => {
                reps = argv[i + 1].parse()?;
                i += 2;
            }
            "--prompt" => {
                prompt = Some(argv[i + 1].clone());
                i += 2;
            }
            "--f16-head" => {
                f16_head = true;
                i += 1;
            }
            // MXFP4 implies `--stream`: the packed path never builds an f32
            // expert bank, so there is nothing for the eager loader to load.
            "--mxfp4" => {
                mxfp4 = true;
                i += 1;
            }
            // Keep the LM head f32 under --mxfp4. The head's 4-bit error lands
            // straight on the logits (3.1e-2 vs 1.9e-3 relative for the body),
            // so this trades 0.85 GB for the greedy path's stability.
            "--f32-head" => {
                f32_head = true;
                i += 1;
            }
            "--inplace-scan" => {
                inplace_scan = true;
                i += 1;
            }
            "--decode" => {
                decode = argv[i + 1].parse()?;
                i += 2;
            }
            "--stream" => {
                stream = true;
                i += 1;
            }
            other => anyhow::bail!("unknown flag {other}"),
        }
    }
    Ok(Args {
        mxfp4,
        f32_head,
        f16_head,
        inplace_scan,
        decode,
        stream,
        weights,
        device,
        seq,
        reps,
        prompt,
    })
}

fn main() -> Result<()> {
    let args = parse_args()?;
    let cfg = LingConfig::from_file(args.weights.join("config.json"))
        .with_context(|| format!("read config from {:?}", args.weights))?;
    println!(
        "Ling 3.0 — {} layers ({} MLA / {} KDA), {} experts top-{}, hidden {}, vocab {}",
        cfg.num_hidden_layers,
        (0..cfg.num_hidden_layers)
            .filter(|&i| cfg.attn_kind(i) == rlx_ling::AttnKind::Mla)
            .count(),
        (0..cfg.num_hidden_layers)
            .filter(|&i| cfg.attn_kind(i) == rlx_ling::AttnKind::Kda)
            .count(),
        cfg.num_experts,
        cfg.num_experts_per_tok,
        cfg.hidden_size,
        cfg.vocab_size,
    );
    println!(
        "device={:?} seq={} reps={}",
        args.device, args.seq, args.reps
    );

    // Tokenize if a tokenizer is present; otherwise use deterministic ids.
    let tok_path = args.weights.join("tokenizer.json");
    let tokenizer = tok_path
        .exists()
        .then(|| tokenizers::Tokenizer::from_file(&tok_path))
        .transpose()
        .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
    // A real prompt is left-padded, not right-padded: the model is causal and we
    // read logits at the last position, so trailing PADs would be what it
    // predicts from.
    let (ids, real_len): (Vec<u32>, usize) = match (&args.prompt, &tokenizer) {
        (Some(p), Some(tk)) => {
            let enc = tk
                .encode(p.as_str(), false)
                .map_err(|e| anyhow::anyhow!("encode: {e}"))?;
            let v = enc.get_ids().to_vec();
            let n = v.len();
            anyhow::ensure!(
                n <= args.seq,
                "prompt is {n} tokens but --seq is {}; raise --seq",
                args.seq
            );
            let mut padded = vec![cfg.pad_token_id; args.seq - n];
            padded.extend_from_slice(&v);
            (padded, n)
        }
        _ => (
            (0..args.seq)
                .map(|i| ((i * 1013 + 7) % cfg.vocab_size) as u32)
                .collect(),
            args.seq,
        ),
    };
    if let (Some(p), Some(_)) = (&args.prompt, &tokenizer) {
        println!(
            "prompt ({real_len} tokens, left-padded to {}): {p:?}",
            args.seq
        );
    }
    let ids_f: Vec<f32> = ids.iter().map(|&i| i as f32).collect();

    if args.decode > 0 {
        return run_decode_bench(&args, &cfg, &ids);
    }

    let t0 = Instant::now();
    let (mut compiled, t_load, t_stack, t_build, t_compile) = if args.mxfp4 {
        // Whole model MXFP4: ~4.0 GiB arena instead of 29.5 GiB. Packing is
        // per-expert, so the f32 high water mark is one expert (~3 MB), not the
        // 1.2 GB/layer the f32 streaming path stages.
        let t3 = Instant::now();
        let plan = if args.f32_head {
            rlx_ling::quant::QuantPlan::mxfp4_body()
        } else {
            rlx_ling::quant::QuantPlan::mxfp4_all()
        };
        let compiled = rlx_ling::streaming::load_and_compile_plan(
            &cfg,
            &args.weights,
            args.seq,
            args.device,
            true,
            plan,
            |msg| println!("  {msg} ({})", rss()),
        )?;
        (
            compiled,
            std::time::Duration::ZERO,
            std::time::Duration::ZERO,
            std::time::Duration::ZERO,
            t3.elapsed(),
        )
    } else if args.stream {
        // Low-peak path: expert banks never enter the build.
        let (ckpt, mut wm) = rlx_ling::load_without_experts(&args.weights)?;
        let t_load = t0.elapsed();
        println!(
            "weights loaded (experts deferred): {} tensors in {:.1}s ({})",
            wm.len(),
            t_load.as_secs_f64(),
            rss()
        );
        let t1 = Instant::now();
        rlx_ling::streaming::install_expert_placeholders(&cfg, &mut wm)?;
        let t_stack = t1.elapsed();
        let t3 = Instant::now();
        let compiled = rlx_ling::streaming::compile_deferred(
            &cfg,
            &ckpt,
            &mut wm,
            args.seq,
            args.device,
            true,
            |msg| println!("  {msg} ({})", rss()),
        )?;
        (
            compiled,
            t_load,
            t_stack,
            std::time::Duration::ZERO,
            t3.elapsed(),
        )
    } else {
        let mut wm = WeightMap::from_safetensors_dir(&args.weights)
            .with_context(|| format!("load safetensors from {:?}", args.weights))?;
        let t_load = t0.elapsed();
        println!(
            "weights loaded: {} tensors in {:.1}s ({})",
            wm.len(),
            t_load.as_secs_f64(),
            rss()
        );
        let t1 = Instant::now();
        prepare_checkpoint(&cfg, &mut wm)?;
        let t_stack = t1.elapsed();
        println!(
            "experts stacked in {:.1}s ({})",
            t_stack.as_secs_f64(),
            rss()
        );

        let t2 = Instant::now();
        let built = build_ling_text_flow(&cfg, &mut wm, args.seq, true)?;
        let t_build = t2.elapsed();
        println!(
            "graph built in {:.1}s ({}, weightmap now {} tensors)",
            t_build.as_secs_f64(),
            rss(),
            wm.len()
        );
        drop(wm);

        // Inlined `compile_built` so each stage's memory is visible.
        let t3 = Instant::now();
        let profile = built.profile().clone();
        let typed = built.typed_params.clone();
        let (graph, params) = built.into_graph_parts()?;
        println!(
            "  params extracted: {:.1} GB f32 in {} tensors ({})",
            params.values().map(|v| v.len()).sum::<usize>() as f64 * 4.0 / 1e9,
            params.len(),
            rss()
        );
        let options = rlx_core::flow_bridge::compile_options_for_profile(&profile, args.device);
        let mut compiled = rlx_runtime::Session::new(args.device).compile_with(graph, &options);
        println!("  arena compiled ({})", rss());
        rlx_core::flow_util::attach_built_params(&mut compiled, params, &typed);
        (compiled, t_load, t_stack, t_build, t3.elapsed())
    };
    println!(
        "ready: load {:.1}s + prep {:.1}s + build {:.1}s + compile {:.1}s ({})",
        t_load.as_secs_f64(),
        t_stack.as_secs_f64(),
        t_build.as_secs_f64(),
        t_compile.as_secs_f64(),
        rss()
    );

    let (cos, sin) = cfg.rope_tables(args.seq);
    let mut times = Vec::with_capacity(args.reps);
    let mut last: Vec<f32> = Vec::new();
    for r in 0..args.reps {
        let t = Instant::now();
        let out = compiled
            .run(&[
                ("input_ids", ids_f.as_slice()),
                ("rope_cos", cos.as_slice()),
                ("rope_sin", sin.as_slice()),
            ])
            .into_iter()
            .next()
            .context("forward produced no output")?;
        // Argmax of the final position = the first generated token; folding it in
        // keeps TTFT honest (a logits tensor alone is not a token).
        let v = cfg.vocab_size;
        let row = &out[(args.seq - 1) * v..];
        let next = row
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        let dt = t.elapsed();
        println!(
            "  rep {r}: {:.3}s  first_token_id={next}  finite={}",
            dt.as_secs_f64(),
            out.iter().all(|x| x.is_finite())
        );
        times.push(dt.as_secs_f64());
        last = out;
    }

    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let best = times[0];
    let median = times[times.len() / 2];
    println!("\n── results ──");
    println!("TTFT (best of {}):      {:.3} s", args.reps, best);
    println!("TTFT (median):          {:.3} s", median);
    println!(
        "prefill throughput:     {:.1} tok/s  ({} prompt tokens)",
        args.seq as f64 / best,
        args.seq
    );
    println!(
        "one-time: load {:.1}s + prep {:.1}s + build {:.1}s + compile {:.1}s",
        t_load.as_secs_f64(),
        t_stack.as_secs_f64(),
        t_build.as_secs_f64(),
        t_compile.as_secs_f64()
    );
    println!("memory:                 {}", rss());
    println!(
        "logits: min {:.3} max {:.3}",
        last.iter().cloned().fold(f32::INFINITY, f32::min),
        last.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    );

    // Top-5 continuations — the actual evidence that real weights produce sense,
    // which a finite-logits check cannot give you.
    let v = cfg.vocab_size;
    let mut row: Vec<(usize, f32)> = last[(args.seq - 1) * v..]
        .iter()
        .copied()
        .enumerate()
        .collect();
    row.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let max = row[0].1;
    let denom: f32 = row.iter().map(|(_, l)| (l - max).exp()).sum();
    println!("top-5 next tokens:");
    for (id, logit) in row.iter().take(5) {
        let text = tokenizer
            .as_ref()
            .and_then(|t| t.decode(&[*id as u32], false).ok())
            .unwrap_or_default();
        println!(
            "  {:>7}  p={:.4}  logit={:+.3}  {:?}",
            id,
            (logit - max).exp() / denom,
            logit,
            text
        );
    }
    Ok(())
}

/// Decode-throughput bench: build ONLY the single-token decode graph (a second
/// full-weight arena would not fit alongside the prefill one), walk the prompt to
/// warm the state, then time `--decode N` generated tokens.
fn run_decode_bench(args: &Args, cfg: &LingConfig, ids: &[u32]) -> Result<()> {
    use rlx_ling::flow_decode::{
        DecodeNames, DecodeSession, ScanState, build_ling_decode_flow_full,
        build_ling_decode_flow_opts,
    };
    use rlx_ling::quant::QuantPlan;

    let cap = ids.len() + args.decode + 1;
    let t0 = Instant::now();
    // MXFP4 never materializes an f32 expert bank: load without the per-expert
    // tensors, then pack + upload them per layer after compile.
    let (mut wm, packed_ckpt) = if args.mxfp4 {
        let (ckpt, mut wm) = rlx_ling::load_without_experts(&args.weights)?;
        for layer in 0..cfg.num_hidden_layers {
            if cfg.is_moe_layer(layer) {
                rlx_ling::weights::rename_router_bias_pub(
                    &format!("model.layers.{layer}.mlp"),
                    &mut wm,
                )?;
            }
        }
        (wm, Some(ckpt))
    } else {
        let mut wm = WeightMap::from_safetensors_dir(&args.weights)?;
        prepare_checkpoint(cfg, &mut wm)?;
        (wm, None)
    };
    println!(
        "weights ready in {:.1}s ({})",
        t0.elapsed().as_secs_f64(),
        rss()
    );
    let t1 = Instant::now();
    let scan_mode = if args.inplace_scan {
        ScanState::InPlace
    } else {
        ScanState::Portable
    };
    let (built, layout) = if args.mxfp4 {
        let plan = if args.f32_head {
            QuantPlan::mxfp4_body()
        } else {
            QuantPlan::mxfp4_all()
        };
        build_ling_decode_flow_full(cfg, &mut wm, cap, true, scan_mode, false, plan)?
    } else {
        build_ling_decode_flow_opts(cfg, &mut wm, cap, true, scan_mode, args.f16_head)?
    };
    drop(wm);
    let mut compiled = rlx_core::flow_util::compile_built(built, args.device)?;
    if let Some(ckpt) = &packed_ckpt {
        rlx_ling::streaming::stream_expert_banks_mxfp4(cfg, ckpt, &mut compiled, |msg| {
            println!("  {msg} ({})", rss())
        })?;
    }
    println!(
        "decode graph built+compiled in {:.1}s, cache cap {cap}, scan={scan_mode:?}, f16_head={} ({})",
        args.f16_head,
        t1.elapsed().as_secs_f64(),
        rss()
    );

    let names = DecodeNames::new(cfg);
    let mut sess = DecodeSession::new(cfg, layout, cap);
    let (cos_all, sin_all) = cfg.rope_tables(cap);
    let half = cfg.qk_rope_head_dim / 2;
    let v = cfg.vocab_size;

    // Split the per-token cost so a slow backend can be attributed: `run` is graph
    // execution, `bind` is assembling the input list; `commit` (state memcpy) is
    // timed by the caller.
    let mut run_s = 0f64;
    let mut bind_s = 0f64;
    let step = |compiled: &mut rlx_runtime::CompiledGraph,
                sess: &mut DecodeSession,
                tok: f32,
                pos: usize,
                run_s: &mut f64,
                bind_s: &mut f64|
     -> Vec<f32> {
        let t = [tok];
        let cos = &cos_all[pos * half..(pos + 1) * half];
        let sin = &sin_all[pos * half..(pos + 1) * half];
        let tb = Instant::now();
        let mut inputs: Vec<(&str, &[f32])> =
            vec![("input_ids", &t[..]), ("rope_cos", cos), ("rope_sin", sin)];
        let st = sess.inputs(cfg, &names);
        inputs.extend(st.iter().copied());
        *bind_s += tb.elapsed().as_secs_f64();
        let tr = Instant::now();
        let out = compiled
            .run(&inputs)
            .into_iter()
            .next()
            .expect("decode out");
        *run_s += tr.elapsed().as_secs_f64();
        out
    };

    // Warm the state on the prompt (this is prompt processing, not generation).
    let t_prompt = Instant::now();
    let mut last = Vec::new();
    for (pos, &id) in ids.iter().enumerate() {
        let packed = step(
            &mut compiled,
            &mut sess,
            id as f32,
            pos,
            &mut run_s,
            &mut bind_s,
        );
        last = sess.commit(&packed)?.to_vec();
    }
    let prompt_s = t_prompt.elapsed().as_secs_f64();

    let argmax = |row: &[f32]| {
        row.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap()
    };
    // `last` is the logits after the final prompt token, so its argmax IS the
    // first generated token — include it, then feed it back.
    let mut next = argmax(&last);
    let mut generated = Vec::with_capacity(args.decode + 1);
    generated.push(next as u32);

    // Timed generation.
    let t_gen = Instant::now();
    let (gen_run0, gen_bind0) = (run_s, bind_s);
    let mut commit_s = 0f64;
    let mut per_token = Vec::with_capacity(args.decode);
    for k in 0..args.decode {
        let t = Instant::now();
        let packed = step(
            &mut compiled,
            &mut sess,
            next as f32,
            ids.len() + k,
            &mut run_s,
            &mut bind_s,
        );
        let tc = Instant::now();
        let logits = sess.commit(&packed)?;
        commit_s += tc.elapsed().as_secs_f64();
        next = argmax(logits);
        per_token.push(t.elapsed().as_secs_f64());
        generated.push(next as u32);
    }
    let gen_s = t_gen.elapsed().as_secs_f64();
    let _ = v;

    per_token.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!("\n── decode results ──");
    println!(
        "prompt ({} tok, sequential):  {:.3} s  ({:.1} tok/s)",
        ids.len(),
        prompt_s,
        ids.len() as f64 / prompt_s
    );
    println!(
        "generation ({} tok):          {:.3} s  ({:.2} tok/s)",
        args.decode,
        gen_s,
        args.decode as f64 / gen_s
    );
    println!(
        "per-token: best {:.3} s, median {:.3} s",
        per_token[0],
        per_token[per_token.len() / 2]
    );
    let nd = args.decode as f64;
    println!(
        "  per-token breakdown: run {:.4} s | bind {:.4} s | commit {:.4} s",
        (run_s - gen_run0) / nd,
        (bind_s - gen_bind0) / nd,
        commit_s / nd
    );
    println!(
        "state moved per token:        {:.1} MB in+out",
        sess.state_bytes() as f64 / 1e6
    );
    println!("memory:                       {}", rss());
    if let Some(tk) = std::fs::metadata(args.weights.join("tokenizer.json"))
        .ok()
        .and_then(|_| tokenizers::Tokenizer::from_file(args.weights.join("tokenizer.json")).ok())
    {
        if let Ok(text) = tk.decode(&generated, false) {
            println!("generated: {text:?}");
        }
    }
    Ok(())
}
