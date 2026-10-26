// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! Do qwen3.5's two ways of decoding agree, on real weights?
//!
//! `prefill_get_last_logits` + `decode_get_logits` is the cached pair a
//! server drives: the caller samples, so temperature, stop strings and
//! log-probs live above the model. `predict_logits` over a growing context
//! is the same thing computed the slow way. They are supposed to be the
//! same function, and a server that switched between them — which is what
//! happens when a family forwards the pair, or fails to and gets wrapped in
//! `HostDriven` — must not get different answers.
//!
//! Takes prompt ids directly, so no tokenizer is needed: whether the ids
//! spell anything is irrelevant to whether two code paths agree.
//!
//! Compares the logits rather than only the argmax. On an arbitrary prompt
//! the top two logits are often nearly tied, so an argmax that flips says
//! little; the vectors agreeing to a small tolerance says the computation
//! is the same one.
//!
//! ```text
//! cargo run --release -p rlx-qwen35 --example decode_path_parity -- \
//!     --weights /path/to/model.gguf --prompt-ids 785,6722,315,9625,374 --tokens 4
//! ```

use anyhow::{Result, anyhow};
use rlx_qwen35::Qwen35Runner;
use rlx_runtime::Device;

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// How far apart two logit vectors are — as a distribution, not a maximum.
///
/// The max alone is misleading and was, at first, misleading me: it runs
/// over the whole vocabulary, so a handful of junk rows differing wildly
/// reads the same as two genuinely different computations. What decides the
/// argmax is the spread among the *candidates*, so that is reported
/// separately.
struct Spread {
    median: f32,
    p99: f32,
    max: f32,
    /// Worst difference among the top candidates by the first vector —
    /// these are the only entries that can change the chosen token.
    among_top: f32,
}

/// Is `b` just `a` rescaled and shifted?
///
/// This is the question `|delta|` cannot answer, and it is the one that
/// matters: a difference that preserves the *order* of the logits chooses
/// the same token every time, however large it looks. Greedy decoding is
/// then unaffected and only logit values — temperature, top-p, log-probs
/// returned to a client — see it. A difference that does not preserve
/// order changes the answer.
///
/// Fits `b ~ k*a + c` by least squares and reports the worst residual. A
/// residual far below the top-two margin means the paths rank identically.
struct Affine {
    k: f32,
    c: f32,
    worst_residual: f32,
    /// Spearman-style check: does the top-8 ordering survive?
    top8_order_same: bool,
}

fn affine_fit(a: &[f32], b: &[f32]) -> Affine {
    let n = a.len() as f64;
    let (sa, sb): (f64, f64) = (
        a.iter().map(|&x| x as f64).sum(),
        b.iter().map(|&x| x as f64).sum(),
    );
    let saa: f64 = a.iter().map(|&x| (x as f64) * (x as f64)).sum();
    let sab: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x as f64) * (y as f64))
        .sum();
    let denom = n * saa - sa * sa;
    let (k, c) = if denom.abs() < 1e-12 {
        (1.0, 0.0)
    } else {
        let k = (n * sab - sa * sb) / denom;
        (k, (sb - k * sa) / n)
    };
    let worst = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| ((y as f64) - (k * x as f64 + c)).abs() as f32)
        .fold(0.0f32, f32::max);

    let order = |v: &[f32]| {
        let mut idx: Vec<usize> = (0..v.len()).collect();
        idx.sort_by(|&i, &j| v[j].total_cmp(&v[i]));
        idx.into_iter().take(8).collect::<Vec<_>>()
    };
    Affine {
        k: k as f32,
        c: c as f32,
        worst_residual: worst,
        top8_order_same: order(a) == order(b),
    }
}

fn spread(a: &[f32], b: &[f32], top: usize) -> Spread {
    let mut diffs: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).collect();
    let mut ranked: Vec<usize> = (0..a.len()).collect();
    ranked.sort_by(|&i, &j| a[j].total_cmp(&a[i]));
    let among_top = ranked
        .iter()
        .take(top.min(a.len()))
        .map(|&i| diffs[i])
        .fold(0.0f32, f32::max);
    diffs.sort_by(|x, y| x.total_cmp(y));
    let at = |q: f64| diffs[((diffs.len() - 1) as f64 * q) as usize];
    Spread {
        median: at(0.5),
        p99: at(0.99),
        max: *diffs.last().unwrap_or(&0.0),
        among_top,
    }
}

/// Top two logits, to say whether an argmax disagreement is a near-tie.
fn margin(v: &[f32]) -> f32 {
    let mut s: Vec<f32> = v.to_vec();
    s.sort_by(|a, b| b.total_cmp(a));
    if s.len() < 2 { 0.0 } else { s[0] - s[1] }
}

fn build(weights: &str, max_seq: usize) -> Result<Qwen35Runner> {
    Qwen35Runner::builder()
        .weights(weights)
        .device(Device::Cpu)
        .max_seq(max_seq)
        .build()
}

fn main() -> Result<()> {
    let mut weights = None;
    let mut prompt_ids: Vec<u32> = Vec::new();
    let mut tokens = 4usize;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--weights" => weights = args.next(),
            "--prompt-ids" => {
                prompt_ids = args
                    .next()
                    .unwrap_or_default()
                    .split(',')
                    .filter(|s| !s.trim().is_empty())
                    .map(|s| s.trim().parse())
                    .collect::<Result<_, _>>()?
            }
            "--tokens" => tokens = args.next().unwrap_or_default().parse()?,
            other => return Err(anyhow!("unknown argument {other}")),
        }
    }
    let weights = weights.ok_or_else(|| anyhow!("--weights is required"))?;
    if prompt_ids.is_empty() {
        anyhow::bail!("--prompt-ids is required");
    }
    println!("prompt {prompt_ids:?}, {tokens} tokens\n");

    // The cached pair, driven as a server drives it.
    let mut cached = build(&weights, 512)?;
    let mut logits = cached.prefill_get_last_logits(&prompt_ids)?;
    let mut cached_out: Vec<(u32, Vec<f32>)> = Vec::new();
    for _ in 0..tokens {
        let t = argmax(&logits) as u32;
        cached_out.push((t, logits.clone()));
        logits = cached.decode_get_logits(t)?;
    }

    // The same thing the slow way: re-prefill the whole context each step,
    // which is what `HostDriven` falls back to.
    let mut slow = build(&weights, 512)?;
    let mut context = prompt_ids.clone();
    let mut slow_out: Vec<(u32, Vec<f32>)> = Vec::new();
    for _ in 0..tokens {
        let l = slow.predict_logits(&context)?.logits;
        let t = argmax(&l) as u32;
        slow_out.push((t, l));
        context.push(t);
    }

    // The runner's own greedy loop, as a third opinion. If it agrees with
    // the cached pair then the whole cached decode disagrees with prefill;
    // if it agrees with re-prefill then only the pair's entry point is
    // wrong. The two cases point at different code.
    let mut own = build(&weights, 512)?;
    let own_out = own.generate(&prompt_ids, tokens, |_| true)?;

    println!("  its own generate: {own_out:?}");
    println!(
        "  cached pair     : {:?}",
        cached_out.iter().map(|(t, _)| *t).collect::<Vec<_>>()
    );
    println!(
        "  re-prefill      : {:?}",
        slow_out.iter().map(|(t, _)| *t).collect::<Vec<_>>()
    );
    println!();
    println!(
        "  step  cached  re-prefill   median|Δ|   p99|Δ|   max|Δ|   Δ@top8   margin |  b≈k·a+c: k      c   worst resid  top8 order"
    );
    let mut disagreed = false;
    for (i, ((ct, cl), (st, sl))) in cached_out.iter().zip(slow_out.iter()).enumerate() {
        let s = spread(cl, sl, 8);
        let flag = if ct == st { "" } else { "  <- different token" };
        if ct != st {
            disagreed = true;
        }
        let f = affine_fit(cl, sl);
        println!(
            "  {i:>4}  {ct:>6}  {st:>10}   {:>9.5} {:>8.4} {:>8.4} {:>8.4} {:>8.4} | {:>8.4} {:>6.3} {:>12.5}  {}{flag}",
            s.median,
            s.p99,
            s.max,
            s.among_top,
            margin(cl),
            f.k,
            f.c,
            f.worst_residual,
            if f.top8_order_same { "same" } else { "CHANGED" }
        );
    }
    println!();
    if disagreed {
        println!(
            "The two paths chose different tokens. Read `Δ@top8` against the \n\
             top-2 margin, not `max|Δ|`: the max runs over the whole \n\
             vocabulary and a few junk rows will dominate it. If Δ@top8 is \n\
             well below the margin the paths agree and a near-tie is \n\
             flipping; if it is comparable or larger they are not computing \n\
             the same function."
        );
    } else {
        println!("Both paths chose the same tokens at every step.");
    }
    Ok(())
}
