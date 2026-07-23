// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! Bench Unlimited-OCR MoE LM across precision modes: **memory**, **latency**,
//! and **accuracy** (vs F32 reference logits).
//!
//! ```bash
//! just features=apple-silicon bench-unlimited-ocr-lm-precision --device metal
//!
//! just features=apple-silicon bench-unlimited-ocr-lm-precision --full \
//!   --device metal --precisions f16,q8_0,q4_0 --seq 8 --decode-steps 4
//! ```

use anyhow::{Context, Result, bail};
use rlx_core::flow_util::compile_built;
use rlx_core::weight_map::WeightMap;
use rlx_runtime::{CompiledGraph, Device};
use rlx_unlimited_ocr::CompiledLm;
use rlx_unlimited_ocr::compile_support::{lm_runtime_guard, lm_runtime_guard_for_pack};
use rlx_unlimited_ocr::config::{
    ClipTowerConfig, ProjectorConfig, SamTowerConfig, UnlimitedOcrConfig, UnlimitedOcrVisionConfig,
};
use rlx_unlimited_ocr::default_model_dir;
use rlx_unlimited_ocr::expert_pack::{
    PackedLmWeights, expert_down_exps_key, expert_gate_exps_key, expert_up_exps_key,
    pack_experts_in_map,
};
use rlx_unlimited_ocr::lm_graph::{
    build_unlimited_ocr_decode_built_from_pack, build_unlimited_ocr_prefill_built_from_pack,
    compute_rope_slice,
};
use rlx_unlimited_ocr::lm_precision::{
    LmWeightPrecision, ResolvedLmPrecision, estimate_pack_compile_need, estimate_packed_lm_bytes,
};
use rlx_unlimited_ocr::resolve_device;
use rlx_unlimited_ocr::weights::{UnlimitedOcrWeightPrefix, UnlimitedOcrWeightStore};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

#[derive(Clone, Copy)]
struct BenchArgs {
    full: bool,
    seq: usize,
    decode_steps: usize,
    warmup: usize,
    iters: usize,
    accuracy: bool,
    top_k: usize,
    /// When >0, after latency rows run greedy decode vs F32 and report match rate.
    greedy_steps: usize,
}

fn main() -> Result<()> {
    let (args, precisions, device) = parse_args()?;
    let cfg = if args.full {
        let dir = default_model_dir().context("resolve Unlimited-OCR model dir")?;
        eprintln!("[bench] full checkpoint: {}", dir.display());
        UnlimitedOcrConfig::from_model_dir(&dir)?
    } else {
        eprintln!("[bench] scale=tiny synthetic MoE");
        tiny_cfg()
    };
    cfg.validate()?;

    eprintln!(
        "[bench] device={device:?} seq={} decode_steps={} warmup={} iters={} accuracy={}",
        args.seq, args.decode_steps, args.warmup, args.iters, args.accuracy
    );

    let embeds = fill(args.seq * cfg.hidden_size, 0.5);
    let step = fill(cfg.hidden_size, 0.9);

    let ref_logits = if args.accuracy {
        eprintln!("[bench] building F32 reference logits…");
        match run_reference_logits(&cfg, device, args, &embeds, &step) {
            Ok(r) => {
                eprintln!(
                    "[bench] F32 ref ready (prefill logits={}, decode logits={})",
                    r.prefill.len(),
                    r.decode.as_ref().map(|d| d.len()).unwrap_or(0)
                );
                Some(r)
            }
            Err(e) => {
                eprintln!("[bench] WARN: F32 reference failed ({e:#}); accuracy columns skipped");
                None
            }
        }
    } else {
        None
    };

    println!();
    print_header(ref_logits.is_some());

    let mut rows = Vec::new();
    for prec in &precisions {
        match bench_one(
            &cfg,
            *prec,
            device,
            args,
            &embeds,
            &step,
            ref_logits.as_ref(),
        ) {
            Ok(row) => {
                print_row(&row, ref_logits.is_some());
                rows.push(row);
            }
            Err(e) => eprintln!("{:<8} ERROR: {e:#}", prec.as_str()),
        }
    }

    if !rows.is_empty() {
        println!();
        eprintln!("[bench] summary:");
        if let Some(best_lat) = rows
            .iter()
            .filter(|r| r.decode_ms.is_finite())
            .min_by(|a, b| a.decode_ms.partial_cmp(&b.decode_ms).unwrap())
        {
            eprintln!(
                "  fastest decode: {} ({:.1} ms, {:.1} tok/s)",
                best_lat.prec, best_lat.decode_ms, best_lat.tok_per_s
            );
        }
        if let Some(best_acc) = rows
            .iter()
            .filter(|r| r.corr.is_some())
            .max_by(|a, b| a.corr.partial_cmp(&b.corr).unwrap())
        {
            eprintln!(
                "  closest to F32: {} (corr={:.4}, max|err|={:.3e}, top1={})",
                best_acc.prec,
                best_acc.corr.unwrap_or(0.0),
                best_acc.max_abs.unwrap_or(0.0),
                best_acc
                    .top1
                    .map(|t| if t { "yes" } else { "no" })
                    .unwrap_or("?")
            );
        }
        for r in &rows {
            if let Some(c) = r.corr {
                if c < 0.9 {
                    eprintln!(
                        "  WARN: {} prefill corr={c:.4} vs F32 — treat as unusable for generation",
                        r.prec
                    );
                }
            }
        }
        if let Some(smallest) = rows
            .iter()
            .min_by(|a, b| a.host_mib.partial_cmp(&b.host_mib).unwrap())
        {
            eprintln!(
                "  smallest host pack: {} ({:.1} MiB)",
                smallest.prec, smallest.host_mib
            );
        }
    }

    if args.greedy_steps > 0 {
        if let Err(e) = run_greedy_match(&cfg, device, args, &precisions, &embeds) {
            eprintln!("[bench] greedy match FAILED: {e:#}");
        }
    }
    Ok(())
}

struct RefLogits {
    prefill: Vec<f32>,
    decode: Option<Vec<f32>>,
}

struct Row {
    prec: &'static str,
    host_mib: f64,
    f32_param_mib: f64,
    typed_param_mib: f64,
    pack_ms: f64,
    build_ms: f64,
    compile_ms: f64,
    prefill_ms: f64,
    prefill_p50_ms: f64,
    decode_ms: f64,
    decode_p50_ms: f64,
    tok_per_s: f64,
    corr: Option<f64>,
    max_abs: Option<f64>,
    rmse: Option<f64>,
    top1: Option<bool>,
    topk_overlap: Option<f64>,
    decode_corr: Option<f64>,
}

fn print_header(with_acc: bool) {
    if with_acc {
        println!(
            "{:<8} {:>9} {:>8} {:>8} {:>9} {:>9} {:>9} {:>8} {:>7} {:>8} {:>8} {:>8} {:>5} {:>6}",
            "prec",
            "host_MiB",
            "u8_MiB",
            "pack_ms",
            "pre_ms",
            "pre_p50",
            "dec_ms",
            "dec_p50",
            "tok/s",
            "corr",
            "max|e|",
            "rmse",
            "top1",
            "topK%"
        );
    } else {
        println!(
            "{:<8} {:>9} {:>8} {:>8} {:>9} {:>9} {:>9} {:>8} {:>7}",
            "prec",
            "host_MiB",
            "u8_MiB",
            "pack_ms",
            "pre_ms",
            "pre_p50",
            "dec_ms",
            "dec_p50",
            "tok/s"
        );
    }
}

fn print_row(row: &Row, with_acc: bool) {
    if with_acc {
        println!(
            "{:<8} {:>9.1} {:>8.1} {:>8.0} {:>9.1} {:>9.1} {:>9.1} {:>8.1} {:>7.1} {:>8} {:>8} {:>8} {:>5} {:>6}",
            row.prec,
            row.host_mib,
            row.typed_param_mib,
            row.pack_ms,
            row.prefill_ms,
            row.prefill_p50_ms,
            row.decode_ms,
            row.decode_p50_ms,
            row.tok_per_s,
            row.corr
                .map(|c| format!("{c:.4}"))
                .unwrap_or_else(|| "-".into()),
            row.max_abs
                .map(|e| format!("{e:.2e}"))
                .unwrap_or_else(|| "-".into()),
            row.rmse
                .map(|e| format!("{e:.2e}"))
                .unwrap_or_else(|| "-".into()),
            row.top1.map(|t| if t { "Y" } else { "N" }).unwrap_or("-"),
            row.topk_overlap
                .map(|o| format!("{:.0}", o * 100.0))
                .unwrap_or_else(|| "-".into()),
        );
        if let Some(dc) = row.decode_corr {
            eprintln!(
                "         decode-vs-F32 corr={dc:.4}  (build={:.0}ms compile={:.0}ms f32_params={:.1}MiB)",
                row.build_ms, row.compile_ms, row.f32_param_mib
            );
        } else {
            eprintln!(
                "         (build={:.0}ms compile={:.0}ms f32_params={:.1}MiB)",
                row.build_ms, row.compile_ms, row.f32_param_mib
            );
        }
    } else {
        println!(
            "{:<8} {:>9.1} {:>8.1} {:>8.0} {:>9.1} {:>9.1} {:>9.1} {:>8.1} {:>7.1}",
            row.prec,
            row.host_mib,
            row.typed_param_mib,
            row.pack_ms,
            row.prefill_ms,
            row.prefill_p50_ms,
            row.decode_ms,
            row.decode_p50_ms,
            row.tok_per_s,
        );
    }
}

fn run_reference_logits(
    cfg: &UnlimitedOcrConfig,
    device: Device,
    args: BenchArgs,
    embeds: &[f32],
    step: &[f32],
) -> Result<RefLogits> {
    let need = estimate_pack_compile_need(cfg, ResolvedLmPrecision::F32);
    // Soft guard: skip F32 ref on full if estimate is huge (> available-ish).
    if args.full && need > (48u64 << 30) {
        bail!(
            "F32 ref estimate {} too large for full-checkpoint accuracy",
            fmt_mib(need as f64 / (1024.0 * 1024.0))
        );
    }
    let pack = make_pack(cfg, ResolvedLmPrecision::F32, args.full)?;
    let (prefill_logits, kv_outs) = prefill_once(cfg, &pack, device, args.seq, embeds)?;
    let decode_logits = if args.decode_steps > 0 {
        Some(decode_once(cfg, &pack, device, args.seq, step, &kv_outs)?)
    } else {
        None
    };
    Ok(RefLogits {
        prefill: prefill_logits,
        decode: decode_logits,
    })
}

fn bench_one(
    cfg: &UnlimitedOcrConfig,
    prec: ResolvedLmPrecision,
    device: Device,
    args: BenchArgs,
    embeds: &[f32],
    step: &[f32],
    reference: Option<&RefLogits>,
) -> Result<Row> {
    eprintln!(
        "[bench] --- {} host≈{} need≈{} ---",
        prec.as_str(),
        fmt_mib(estimate_packed_lm_bytes(cfg, prec) as f64 / (1024.0 * 1024.0)),
        fmt_mib(estimate_pack_compile_need(cfg, prec) as f64 / (1024.0 * 1024.0)),
    );

    let t_pack = Instant::now();
    let pack = make_pack(cfg, prec, args.full)?;
    let pack_ms = t_pack.elapsed().as_secs_f64() * 1e3;
    let host_mib = pack.host_nbytes() as f64 / (1024.0 * 1024.0);
    eprintln!(
        "[bench] pack done in {pack_ms:.0} ms (host {:.1} MiB, keeps_ir_packed={})",
        host_mib,
        pack.keeps_quants_in_ir()
    );

    let t_build = Instant::now();
    let built = build_unlimited_ocr_prefill_built_from_pack(cfg, &pack, 1, args.seq)
        .context("prefill build")?;
    let build_ms = t_build.elapsed().as_secs_f64() * 1e3;
    let f32_param_mib = f32_param_bytes(&built) as f64 / (1024.0 * 1024.0);
    let typed_param_mib = typed_param_bytes(&built) as f64 / (1024.0 * 1024.0);

    let t_compile = Instant::now();
    let mut prefill = lm_runtime_guard_for_pack(device, &pack, || compile_built(built, device))
        .context("prefill compile")?;
    let compile_ms = t_compile.elapsed().as_secs_f64() * 1e3;

    let n_layers = cfg.num_hidden_layers;
    for _ in 0..args.warmup {
        let _ = run_prefill(&mut prefill, device, embeds)?;
    }

    let mut prefill_samples = Vec::with_capacity(args.iters);
    let mut last_outs = Vec::new();
    for _ in 0..args.iters {
        let t0 = Instant::now();
        last_outs = run_prefill(&mut prefill, device, embeds)?;
        prefill_samples.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    let prefill_ms = mean(&prefill_samples);
    let prefill_p50_ms = percentile(&mut prefill_samples.clone(), 50.0);

    let past_seq = args.seq;
    let built_d = build_unlimited_ocr_decode_built_from_pack(cfg, &pack, 1, past_seq)
        .context("decode build")?;
    let mut decode = lm_runtime_guard_for_pack(device, &pack, || compile_built(built_d, device))
        .context("decode compile")?;

    let (cos, sin) = compute_rope_slice(cfg, past_seq);
    let past_owned: Vec<(String, Vec<f32>)> = (0..n_layers)
        .flat_map(|i| {
            [
                (format!("past_k_{i}"), last_outs[1 + 2 * i].clone()),
                (format!("past_v_{i}"), last_outs[1 + 2 * i + 1].clone()),
            ]
        })
        .collect();

    for _ in 0..args.warmup {
        let _ = run_decode(&mut decode, device, step, &cos, &sin, &past_owned)?;
    }

    let steps = args.decode_steps.max(1);
    let mut decode_samples = Vec::with_capacity(args.iters);
    let mut last_decode_logits = None;
    for _ in 0..args.iters {
        let t0 = Instant::now();
        let mut outs = Vec::new();
        for _ in 0..steps {
            outs = run_decode(&mut decode, device, step, &cos, &sin, &past_owned)?;
        }
        decode_samples.push(t0.elapsed().as_secs_f64() * 1e3 / steps as f64);
        last_decode_logits = outs.first().cloned();
    }
    let decode_ms = mean(&decode_samples);
    let decode_p50_ms = percentile(&mut decode_samples.clone(), 50.0);
    let tok_per_s = 1000.0 / decode_ms;

    let prefill_logits = last_outs.first().cloned().unwrap_or_default();
    let (corr, max_abs, rmse, top1, topk_overlap, decode_corr) = if let Some(r) = reference {
        let acc = accuracy_vs(&r.prefill, &prefill_logits, args.top_k);
        let dcorr = match (&r.decode, &last_decode_logits) {
            (Some(rd), Some(ld)) => Some(pearson(rd, ld)),
            _ => None,
        };
        (
            Some(acc.corr),
            Some(acc.max_abs),
            Some(acc.rmse),
            Some(acc.top1),
            Some(acc.topk_overlap),
            dcorr,
        )
    } else {
        (None, None, None, None, None, None)
    };

    Ok(Row {
        prec: prec.as_str(),
        host_mib,
        f32_param_mib,
        typed_param_mib,
        pack_ms,
        build_ms,
        compile_ms,
        prefill_ms,
        prefill_p50_ms,
        decode_ms,
        decode_p50_ms,
        tok_per_s,
        corr,
        max_abs,
        rmse,
        top1,
        topk_overlap,
        decode_corr,
    })
}

struct Acc {
    corr: f64,
    max_abs: f64,
    rmse: f64,
    top1: bool,
    topk_overlap: f64,
}

fn accuracy_vs(reference: &[f32], cand: &[f32], top_k: usize) -> Acc {
    let n = reference.len().min(cand.len());
    let a = &reference[..n];
    let b = &cand[..n];
    let corr = pearson(a, b);
    let mut max_abs = 0.0f64;
    let mut sse = 0.0f64;
    for i in 0..n {
        let d = (a[i] as f64 - b[i] as f64).abs();
        max_abs = max_abs.max(d);
        sse += d * d;
    }
    let rmse = (sse / n as f64).sqrt();
    let top1 = argmax(a) == argmax(b);
    let topk_overlap = topk_jaccard(a, b, top_k);
    Acc {
        corr,
        max_abs,
        rmse,
        top1,
        topk_overlap,
    }
}

fn make_pack(
    cfg: &UnlimitedOcrConfig,
    prec: ResolvedLmPrecision,
    full: bool,
) -> Result<Arc<PackedLmWeights>> {
    if full {
        let dir = default_model_dir().context("model dir")?;
        let store = UnlimitedOcrWeightStore::open(&dir)?;
        Ok(Arc::new(PackedLmWeights::from_store_with_precision(
            &store,
            cfg,
            requested(prec),
        )?))
    } else {
        let mut wm = synthetic_lm_weights(cfg);
        Ok(Arc::new(PackedLmWeights::from_weight_map(
            &mut wm,
            cfg.clone(),
            prec,
        )?))
    }
}

/// Greedy argmax chain vs F32 reference (same synthetic embeds as the latency bench).
fn run_greedy_match(
    cfg: &UnlimitedOcrConfig,
    device: Device,
    args: BenchArgs,
    precisions: &[ResolvedLmPrecision],
    embeds: &[f32],
) -> Result<()> {
    let steps = args.greedy_steps;
    eprintln!("[bench] greedy match: {steps} tokens vs F32 on {device:?}…");

    let f32_pack = make_pack(cfg, ResolvedLmPrecision::F32, args.full)?;
    let ref_toks = greedy_tokens(device, Arc::clone(&f32_pack), embeds, args.seq, steps)?;
    drop(f32_pack);
    eprintln!("[bench] F32 greedy tokens: {ref_toks:?}");

    for &prec in precisions {
        if prec == ResolvedLmPrecision::F32 {
            eprintln!("[bench] greedy f32: 100% (self)");
            continue;
        }
        let pack = make_pack(cfg, prec, args.full)?;
        let toks = greedy_tokens(device, pack, embeds, args.seq, steps)?;
        let matched = ref_toks
            .iter()
            .zip(toks.iter())
            .filter(|(a, b)| a == b)
            .count();
        let pct = 100.0 * matched as f64 / steps.max(1) as f64;
        eprintln!(
            "[bench] greedy {}: {matched}/{steps} match ({pct:.0}%) tokens={toks:?}",
            prec.as_str()
        );
    }
    Ok(())
}

fn greedy_tokens(
    device: Device,
    pack: Arc<PackedLmWeights>,
    embeds: &[f32],
    seq: usize,
    steps: usize,
) -> Result<Vec<u32>> {
    let mut lm = CompiledLm::new(device, pack);
    let (mut logits, mut kv) = lm.prefill(embeds, seq)?;
    let mut out = Vec::with_capacity(steps);
    for i in 0..steps {
        let next = argmax(&logits) as u32;
        out.push(next);
        let step_embed = lm.embed_tokens(&[next])?;
        let pos = seq + i;
        logits = lm.decode_step(&step_embed, pos, &mut kv)?;
    }
    Ok(out)
}

fn prefill_once(
    cfg: &UnlimitedOcrConfig,
    pack: &Arc<PackedLmWeights>,
    device: Device,
    seq: usize,
    embeds: &[f32],
) -> Result<(Vec<f32>, Vec<Vec<f32>>)> {
    let built = build_unlimited_ocr_prefill_built_from_pack(cfg, pack, 1, seq)?;
    let mut compiled = lm_runtime_guard_for_pack(device, pack, || compile_built(built, device))?;
    let outs = run_prefill(&mut compiled, device, embeds)?;
    let logits = outs.first().cloned().context("missing prefill logits")?;
    Ok((logits, outs))
}

fn decode_once(
    cfg: &UnlimitedOcrConfig,
    pack: &Arc<PackedLmWeights>,
    device: Device,
    past_seq: usize,
    step: &[f32],
    prefill_outs: &[Vec<f32>],
) -> Result<Vec<f32>> {
    let n_layers = cfg.num_hidden_layers;
    let built = build_unlimited_ocr_decode_built_from_pack(cfg, pack, 1, past_seq)?;
    let mut compiled = lm_runtime_guard_for_pack(device, pack, || compile_built(built, device))?;
    let (cos, sin) = compute_rope_slice(cfg, past_seq);
    let past_owned: Vec<(String, Vec<f32>)> = (0..n_layers)
        .flat_map(|i| {
            [
                (format!("past_k_{i}"), prefill_outs[1 + 2 * i].clone()),
                (format!("past_v_{i}"), prefill_outs[1 + 2 * i + 1].clone()),
            ]
        })
        .collect();
    let outs = run_decode(&mut compiled, device, step, &cos, &sin, &past_owned)?;
    outs.first().cloned().context("missing decode logits")
}

fn run_prefill(
    compiled: &mut CompiledGraph,
    device: Device,
    embeds: &[f32],
) -> Result<Vec<Vec<f32>>> {
    Ok(lm_runtime_guard(device, true, || {
        compiled.run(&[("inputs_embeds", embeds)])
    }))
}

fn run_decode(
    compiled: &mut CompiledGraph,
    device: Device,
    step: &[f32],
    cos: &[f32],
    sin: &[f32],
    past: &[(String, Vec<f32>)],
) -> Result<Vec<Vec<f32>>> {
    Ok(lm_runtime_guard(device, true, || {
        let mut pairs: Vec<(&str, &[f32])> = vec![
            ("inputs_embeds", step),
            ("rope_cos", cos),
            ("rope_sin", sin),
        ];
        for (n, d) in past {
            pairs.push((n.as_str(), d.as_slice()));
        }
        compiled.run(&pairs)
    }))
}

fn f32_param_bytes(built: &rlx_flow::BuiltModel) -> usize {
    built.params.values().map(|v| v.len() * 4).sum()
}

fn typed_param_bytes(built: &rlx_flow::BuiltModel) -> usize {
    built.typed_params.iter().map(|(_, b, _)| b.len()).sum()
}

fn requested(p: ResolvedLmPrecision) -> LmWeightPrecision {
    match p {
        ResolvedLmPrecision::F32 => LmWeightPrecision::F32,
        ResolvedLmPrecision::F16 => LmWeightPrecision::F16,
        ResolvedLmPrecision::Bf16 => LmWeightPrecision::Bf16,
        ResolvedLmPrecision::Q8_0 => LmWeightPrecision::Q8_0,
        ResolvedLmPrecision::Q4_0 => LmWeightPrecision::Q4_0,
    }
}

fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }
    xs.iter().sum::<f64>() / xs.len() as f64
}

fn percentile(xs: &mut [f64], p: f64) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((p / 100.0) * (xs.len() as f64 - 1.0)).round() as usize;
    xs[idx.min(xs.len() - 1)]
}

fn pearson(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len()) as f64;
    if n < 2.0 {
        return f64::NAN;
    }
    let ma = a.iter().map(|x| *x as f64).sum::<f64>() / n;
    let mb = b.iter().map(|x| *x as f64).sum::<f64>() / n;
    let mut num = 0.0;
    let mut da = 0.0;
    let mut db = 0.0;
    for i in 0..n as usize {
        let xa = a[i] as f64 - ma;
        let xb = b[i] as f64 - mb;
        num += xa * xb;
        da += xa * xa;
        db += xb * xb;
    }
    num / (da.sqrt() * db.sqrt() + 1e-12)
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}

fn topk_indices(v: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&i, &j| v[j].partial_cmp(&v[i]).unwrap());
    idx.truncate(k.min(v.len()));
    idx.sort_unstable();
    idx
}

fn topk_jaccard(a: &[f32], b: &[f32], k: usize) -> f64 {
    let ta = topk_indices(a, k);
    let tb = topk_indices(b, k);
    let mut i = 0;
    let mut j = 0;
    let mut inter = 0usize;
    while i < ta.len() && j < tb.len() {
        match ta[i].cmp(&tb[j]) {
            std::cmp::Ordering::Equal => {
                inter += 1;
                i += 1;
                j += 1;
            }
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
        }
    }
    let union = ta.len() + tb.len() - inter;
    if union == 0 {
        1.0
    } else {
        inter as f64 / union as f64
    }
}

fn fmt_mib(mib: f64) -> String {
    if mib >= 1024.0 {
        format!("{:.1} GiB", mib / 1024.0)
    } else {
        format!("{mib:.0} MiB")
    }
}

fn parse_args() -> Result<(BenchArgs, Vec<ResolvedLmPrecision>, Device)> {
    let mut full = false;
    let mut seq = 16usize;
    let mut decode_steps = 8usize;
    let mut warmup = 1usize;
    let mut iters = 5usize;
    let mut accuracy = true;
    let mut top_k = 5usize;
    let mut greedy_steps = 0usize;
    let mut prec_s = "f32,f16,q8_0,q4_0".to_string();
    let mut device_s: Option<String> = None;

    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--full" => full = true,
            "--tiny" => full = false,
            "--no-accuracy" => accuracy = false,
            "--accuracy" => accuracy = true,
            "--seq" => {
                seq = args
                    .next()
                    .context("--seq needs a value")?
                    .parse()
                    .context("--seq")?;
            }
            "--decode-steps" => {
                decode_steps = args
                    .next()
                    .context("--decode-steps needs a value")?
                    .parse()
                    .context("--decode-steps")?;
            }
            "--greedy-steps" => {
                greedy_steps = args
                    .next()
                    .context("--greedy-steps needs a value")?
                    .parse()
                    .context("--greedy-steps")?;
            }
            "--warmup" => {
                warmup = args
                    .next()
                    .context("--warmup needs a value")?
                    .parse()
                    .context("--warmup")?;
            }
            "--iters" => {
                iters = args
                    .next()
                    .context("--iters needs a value")?
                    .parse()
                    .context("--iters")?;
            }
            "--top-k" => {
                top_k = args
                    .next()
                    .context("--top-k needs a value")?
                    .parse()
                    .context("--top-k")?;
            }
            "--precisions" => {
                prec_s = args.next().context("--precisions needs a value")?;
            }
            "--device" => {
                device_s = Some(args.next().context("--device needs a value")?);
            }
            "--help" | "-h" => {
                eprintln!(
                    "Usage: bench_lm_precision [--tiny|--full] [--device auto|cpu|metal|…] \
                     [--precisions f32,f16,q8_0,q4_0] [--seq N] [--decode-steps N] \
                     [--greedy-steps N] [--warmup N] [--iters N] [--top-k N] \
                     [--accuracy|--no-accuracy]\n\n\
                     Reports host memory, prefill/decode latency (mean + p50), and \
                     accuracy vs F32 reference (Pearson corr, max|err|, top-1 match, top-K Jaccard).\n\
                     With --greedy-steps N, also runs N greedy tokens vs F32 and reports match rate."
                );
                std::process::exit(0);
            }
            "--" => {}
            other => bail!("unknown arg {other}"),
        }
    }

    let mut precisions = Vec::new();
    for p in prec_s.split(',') {
        let p = p.trim();
        if p.is_empty() {
            continue;
        }
        precisions.push(match LmWeightPrecision::parse(p)? {
            LmWeightPrecision::F32 => ResolvedLmPrecision::F32,
            LmWeightPrecision::F16 => ResolvedLmPrecision::F16,
            LmWeightPrecision::Bf16 => ResolvedLmPrecision::Bf16,
            LmWeightPrecision::Q8_0 => ResolvedLmPrecision::Q8_0,
            LmWeightPrecision::Q4_0 => ResolvedLmPrecision::Q4_0,
            LmWeightPrecision::Auto => bail!("use concrete precisions, not auto"),
        });
    }
    if precisions.is_empty() {
        bail!("no precisions selected");
    }

    let device = resolve_device(device_s.as_deref())?;
    if !rlx_runtime::is_available(device) {
        bail!("device {device:?} not available");
    }

    Ok((
        BenchArgs {
            full,
            seq,
            decode_steps,
            warmup,
            iters,
            accuracy,
            top_k,
            greedy_steps,
        },
        precisions,
        device,
    ))
}

fn tiny_cfg() -> UnlimitedOcrConfig {
    UnlimitedOcrConfig {
        model_type: "unlimited-ocr".into(),
        hidden_size: 64,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: 4,
        n_routed_experts: 4,
        n_shared_experts: 2,
        num_experts_per_tok: 2,
        moe_intermediate_size: 32,
        intermediate_size: 64,
        first_k_dense_replace: 1,
        vocab_size: 128,
        max_position_embeddings: 256,
        sliding_window: 16,
        use_mla: false,
        rms_norm_eps: 1e-6,
        rope_theta: 10_000.0,
        hidden_act: "silu".into(),
        bos_token_id: 0,
        eos_token_id: 1,
        pad_token_id: 2,
        image_token_id: 3,
        v_head_dim: Some(16),
        vision_config: UnlimitedOcrVisionConfig {
            sam: SamTowerConfig::default(),
            clip: ClipTowerConfig::default(),
            image_size: 1024,
        },
        projector: ProjectorConfig {
            input_dim: 2048,
            n_embed: 64,
            projector_type: "linear".into(),
        },
        patch_size: 16,
        downsample_ratio: 4,
    }
}

fn fill(n: usize, seed: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32 * 0.017 + seed).sin()) * 0.02)
        .collect()
}

fn synthetic_lm_weights(cfg: &UnlimitedOcrConfig) -> WeightMap {
    let h = cfg.hidden_size;
    let v = cfg.vocab_size;
    let ff_dense = cfg.intermediate_size;
    let moe_ff = cfg.moe_intermediate_size;
    let n_e = cfg.n_routed_experts;
    let shared_ff = moe_ff * cfg.n_shared_experts;
    let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();

    t.insert(
        UnlimitedOcrWeightPrefix::embed_tokens().into(),
        (fill(v * h, 0.1), vec![v, h]),
    );
    t.insert(
        UnlimitedOcrWeightPrefix::lm_norm().into(),
        (fill(h, 0.2), vec![h]),
    );
    t.insert(
        UnlimitedOcrWeightPrefix::lm_head().into(),
        (fill(v * h, 0.3), vec![v, h]),
    );

    for layer in 0..cfg.num_hidden_layers {
        t.insert(
            UnlimitedOcrWeightPrefix::lm_input_layernorm(layer),
            (fill(h, 1.0 + layer as f32), vec![h]),
        );
        t.insert(
            UnlimitedOcrWeightPrefix::lm_post_attention_layernorm(layer),
            (fill(h, 2.0 + layer as f32), vec![h]),
        );
        for (pi, proj) in ["q_proj", "k_proj", "v_proj", "o_proj"].iter().enumerate() {
            t.insert(
                UnlimitedOcrWeightPrefix::lm_attn(layer, proj),
                (fill(h * h, 3.0 + pi as f32), vec![h, h]),
            );
        }
        if cfg.is_dense_layer(layer) {
            for (pi, proj) in ["gate_proj", "up_proj"].iter().enumerate() {
                t.insert(
                    UnlimitedOcrWeightPrefix::lm_dense_mlp(layer, proj),
                    (fill(ff_dense * h, 4.0 + pi as f32), vec![ff_dense, h]),
                );
            }
            t.insert(
                UnlimitedOcrWeightPrefix::lm_dense_mlp(layer, "down_proj"),
                (fill(h * ff_dense, 4.5), vec![h, ff_dense]),
            );
        } else {
            t.insert(
                UnlimitedOcrWeightPrefix::lm_moe_gate(layer),
                (fill(n_e * h, 5.0), vec![n_e, h]),
            );
            for (pi, proj) in ["gate_proj", "up_proj"].iter().enumerate() {
                t.insert(
                    UnlimitedOcrWeightPrefix::lm_moe_shared_expert(layer, proj),
                    (fill(shared_ff * h, 6.0 + pi as f32), vec![shared_ff, h]),
                );
            }
            t.insert(
                UnlimitedOcrWeightPrefix::lm_moe_shared_expert(layer, "down_proj"),
                (fill(h * shared_ff, 6.5), vec![h, shared_ff]),
            );
            for e in 0..n_e {
                for (pi, proj) in ["gate_proj", "up_proj"].iter().enumerate() {
                    t.insert(
                        UnlimitedOcrWeightPrefix::lm_moe_expert(layer, e, proj),
                        (
                            fill(moe_ff * h, 7.0 + e as f32 + pi as f32 * 0.1),
                            vec![moe_ff, h],
                        ),
                    );
                }
                t.insert(
                    UnlimitedOcrWeightPrefix::lm_moe_expert(layer, e, "down_proj"),
                    (fill(h * moe_ff, 8.0 + e as f32), vec![h, moe_ff]),
                );
            }
        }
    }

    let mut map = WeightMap::from_tensors(t);
    for layer in 0..cfg.num_hidden_layers {
        if !cfg.is_dense_layer(layer) {
            pack_experts_in_map(&mut map, layer, n_e, h, moe_ff).expect("pack experts");
            assert!(map.has(&expert_gate_exps_key(layer)));
            assert!(map.has(&expert_up_exps_key(layer)));
            assert!(map.has(&expert_down_exps_key(layer)));
        }
    }
    map
}
