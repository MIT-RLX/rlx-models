// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! Logits of an LM's full-sequence graph in f32 and in f64
//! (`NumberFormat::F64`, CPU), against a float64 reference.
//!
//! The reference is raw little-endian f64 logits `[seq, vocab]` for the same
//! token ids, from `scripts/lm_f64_reference.py` (transformers with every
//! step in float64). Without it the two runs are only compared with each
//! other. The f64 run is read before its final cast to f32, so it is not
//! limited by f32 output rounding, and the graph is built under the f64
//! format so its RoPE tables are exact in f64 too.
//!
//! `RUNNER=1` (or `f32` / `f64` for one format) checks the serving path
//! instead: `auto_runner` with `prefill_logits` on the first token and
//! `decode_logits` for the rest, so the KV cache and decode graphs run in
//! the format too. Its logits come back as f32.
//!
//! `DEVICE` (default `cpu`) runs both on another backend; GPUs compute f64 in
//! double-single.
//!
//! ```text
//! python3 scripts/lm_f64_reference.py DIR 785,6722,315,9625,374 /tmp/ref.f64
//! cargo run --release -p rlx-models --features train,qwen3,gemma,llama32 \
//!     --example lm_f64_check -- DIR 785,6722,315,9625,374 /tmp/ref.f64
//! ```

use anyhow::{Context, Result};
use rlx_runtime::{CompileOptions, Device, NumberFormat, Session};

fn device() -> Device {
    std::env::var("DEVICE").map_or(Device::Cpu, |d| {
        rlx_cli::parse_device(&d).unwrap_or_else(|e| panic!("DEVICE: {e}"))
    })
}

fn feed(lm: &rlx_tune::lm::TrainableLm, ids: &[u32]) -> Vec<(String, Vec<f32>)> {
    let mut inputs = vec![(
        lm.token_input.clone(),
        ids.iter().map(|&x| x as f32).collect(),
    )];
    inputs.extend(lm.extra_inputs.iter().cloned());
    inputs
}

/// The graph as compiled, in f32.
fn logits_f32(lm: &rlx_tune::lm::TrainableLm, ids: &[u32]) -> Vec<f64> {
    let mut cg = Session::new(device()).compile_with(lm.graph.clone(), &CompileOptions::new());
    for (k, v) in &lm.params {
        cg.set_param(k, v);
    }
    let inputs = feed(lm, ids);
    let refs: Vec<(&str, &[f32])> = inputs
        .iter()
        .map(|(n, v)| (n.as_str(), v.as_slice()))
        .collect();
    cg.run(&refs).remove(0).into_iter().map(f64::from).collect()
}

/// Under `NumberFormat::F64`, read before the final cast back to f32 so the
/// comparison is not limited by f32 output rounding.
fn logits_f64(lm: &rlx_tune::lm::TrainableLm, ids: &[u32]) -> Vec<f64> {
    use rlx_ir::{DType, Op};
    let mut g = rlx_opt::rlx_compile::promote_f64::promote_to_f64(&lm.graph).graph;
    let o = g.outputs[0];
    if matches!(g.node(o).op, Op::Cast { to: DType::F32 }) {
        let x = g.node(o).inputs[0];
        g.set_outputs(vec![x]);
    }
    let mut opts = CompileOptions::new();
    opts.number_format = Some(NumberFormat::F64);
    opts.fusion_opts.skip_fusion = true;
    let mut cg = Session::new(device()).compile_with(g, &opts);
    for (k, v) in &lm.params {
        let wide: Vec<u8> = v.iter().flat_map(|&x| (x as f64).to_le_bytes()).collect();
        cg.set_param_typed(k, &wide, DType::F64);
    }
    let inputs: Vec<(String, Vec<u8>)> = feed(lm, ids)
        .into_iter()
        .map(|(n, v)| (n, v.iter().flat_map(|x| x.to_le_bytes()).collect()))
        .collect();
    let refs: Vec<(&str, &[u8], DType)> = inputs
        .iter()
        .map(|(n, b)| (n.as_str(), b.as_slice(), DType::F32))
        .collect();
    let (bytes, dt) = cg.run_typed(&refs).remove(0);
    assert_eq!(dt, DType::F64);
    bytes
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// The serving path: prefill the first token, then teacher-force the rest
/// through `decode_logits` (KV cache, decode graphs), one logits row per
/// position.
fn runner_logits(
    dir: &std::path::Path,
    ids: &[u32],
    format: Option<NumberFormat>,
) -> Result<Vec<f64>> {
    let mut opts = rlx_models::run::AutoOptions::default().max_seq(64);
    opts.device = Some(device());
    opts.servable = true;
    opts.number_format = format;
    let mut r = rlx_models::run::auto_runner_with(dir, &opts)?;
    let mut out: Vec<f64> = r
        .prefill_logits(&ids[..1])?
        .into_iter()
        .map(f64::from)
        .collect();
    for &t in &ids[1..] {
        out.extend(r.decode_logits(t)?.into_iter().map(f64::from));
    }
    Ok(out)
}

fn argmax(v: &[f64]) -> usize {
    (0..v.len())
        .max_by(|&a, &b| v[a].total_cmp(&v[b]))
        .unwrap_or(0)
}

/// Max |a - b|, the (position, token) where it occurs, and how many positions
/// pick a different argmax.
fn compare(a: &[f64], b: &[f64], vocab: usize) -> (f64, (usize, usize), usize) {
    let (at, worst) = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .enumerate()
        .fold((0, 0.0), |m, (i, d)| if d > m.1 { (i, d) } else { m });
    let flips = a
        .chunks_exact(vocab)
        .zip(b.chunks_exact(vocab))
        .filter(|(x, y)| argmax(x) != argmax(y))
        .count();
    (worst, (at / vocab, at % vocab), flips)
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let dir = std::path::Path::new(args.first().context("weights dir")?);
    let ids: Vec<u32> = args
        .get(1)
        .context("comma-separated token ids")?
        .split(',')
        .map(|s| s.trim().parse())
        .collect::<Result<_, _>>()?;
    if std::env::var("RUNNER").is_ok() {
        let reference: Vec<f64> = std::fs::read(args.get(2).context("reference")?)?
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let vocab = reference.len() / ids.len();
        let formats: Vec<Option<NumberFormat>> = match std::env::var("RUNNER").as_deref() {
            Ok("f32") => vec![None],
            Ok("f64") => vec![Some(NumberFormat::F64)],
            _ => vec![None, Some(NumberFormat::F64)],
        };
        for format in formats {
            let t = std::time::Instant::now();
            let l = runner_logits(dir, &ids, format)?;
            let (d, at, flips) = compare(&l, &reference, vocab);
            println!(
                "runner {:?} vs reference: max|Δ| {d:.3e} at {at:?}, argmax differs at {flips}/{} ({:.1}s)",
                format.unwrap_or(NumberFormat::F32),
                ids.len(),
                t.elapsed().as_secs_f64()
            );
        }
        return Ok(());
    }
    // Built under the f64 format so RoPE tables carry their f64 residual
    // (`RopeTablesStage::with_f64`); unset again so the f32 run stays f32.
    // The residual rounds away in f32, so both runs share the graph.
    rlx_ir::env::set("RLX_NUMBER_FORMAT", "f64");
    let lm = rlx_models::lm_train::trainable_lm(dir, 1, ids.len());
    rlx_ir::env::unset("RLX_NUMBER_FORMAT");
    let lm = lm?;
    let t = std::time::Instant::now();
    let l32 = logits_f32(&lm, &ids);
    eprintln!("  f32 {:.1}s", t.elapsed().as_secs_f64());
    let t = std::time::Instant::now();
    let l64 = logits_f64(&lm, &ids);
    eprintln!("  f64 {:.1}s", t.elapsed().as_secs_f64());
    let vocab = l32.len() / ids.len();
    let n = ids.len();
    let (d, at, flips) = compare(&l32, &l64, vocab);
    println!("f32 vs f64: max|Δ| {d:.3e} at {at:?}, argmax differs at {flips}/{n} positions");
    if let Some(path) = args.get(2) {
        let reference: Vec<f64> = std::fs::read(path)?
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
            .collect();
        anyhow::ensure!(
            reference.len() == l32.len(),
            "reference has {} logits, graph {}",
            reference.len(),
            l32.len()
        );
        let scale = reference.iter().fold(0f64, |m, v| m.max(v.abs()));
        println!("reference: max |logit| {scale:.2}");
        for (name, l) in [("f32", &l32), ("f64", &l64)] {
            let (d, at, flips) = compare(l, &reference, vocab);
            println!(
                "{name} vs reference: max|Δ| {d:.3e} at {at:?}, argmax differs at {flips}/{n} positions"
            );
            let per: Vec<String> = l
                .chunks_exact(vocab)
                .zip(reference.chunks_exact(vocab))
                .map(|(a, b)| {
                    let d = a
                        .iter()
                        .zip(b)
                        .map(|(x, y)| (x - y).abs())
                        .fold(0.0, f64::max);
                    format!("{d:.0e}")
                })
                .collect();
            println!("  per position: {}", per.join(" "));
        }
    }
    Ok(())
}
