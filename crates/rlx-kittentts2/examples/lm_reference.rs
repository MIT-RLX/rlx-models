// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0

//! Speech-LM parity against `scripts/kittentts2_reference.py`:
//! first-step logits and the penalized greedy continuation.
//!
//! ```text
//! cargo run -p rlx-kittentts2 --release --example lm_reference -- \
//!     weights/tts/kitten-tts-2 weights/tts/kitten-tts-2/reference [cpu|metal|…]
//! ```

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use rlx_kittentts2::{Kitten2Config, Rng, SamplerOpts, SpeechLm, parse_device, voices::read_npz};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let repo = PathBuf::from(args.first().context("repo dir")?);
    let refdir = PathBuf::from(args.get(1).context("reference dir")?);
    let device = parse_device(args.get(2).map_or("cpu", String::as_str))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(refdir.join("reference.json"))?)?;
    let mut npz = read_npz(&refdir.join("reference.npz"))?;
    let ref_logits = npz
        .remove("first_logits")
        .context("first_logits")?
        .into_f32()?;
    let spk = npz
        .remove("speaker_embedding")
        .context("speaker")?
        .into_f32()?;
    let ids = |k: &str| -> Vec<u32> {
        meta[k]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect()
    };
    let prompt = ids("prompt_ids");
    let mut ref_greedy = ids("greedy_tokens");
    // `KT2_STEPS=N` checks only the first N decode steps (slow backends).
    if let Some(n) = std::env::var("KT2_STEPS").ok().and_then(|s| s.parse().ok()) {
        ref_greedy.truncate(n);
    }

    let cfg = Kitten2Config::from_dir(&repo)?;
    let lm_dir = repo.join(&cfg.lm_dir);
    let ternary = repo.join(
        cfg.lm_packed
            .as_deref()
            .unwrap_or("lm/model-ternary.safetensors"),
    );
    let t0 = Instant::now();
    let mut lm = SpeechLm::open(&lm_dir, &ternary, device)?;
    eprintln!("open {:.2}s", t0.elapsed().as_secs_f32());

    let t0 = Instant::now();
    let logits = lm.prefill_logits(&prompt, &spk)?;
    eprintln!(
        "prefill {} tokens {:.2}s",
        prompt.len(),
        t0.elapsed().as_secs_f32()
    );
    let (mut max_abs, mut max_ref) = (0f32, 0f32);
    for (a, b) in logits.iter().zip(&ref_logits) {
        max_abs = max_abs.max((a - b).abs());
        max_ref = max_ref.max(b.abs());
    }
    let argmax = |v: &[f32]| {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0
    };
    println!(
        "first logits: max|Δ| {max_abs:.3e} (max|ref| {max_ref:.2}), argmax {} vs {}",
        argmax(&logits),
        argmax(&ref_logits)
    );

    let mut opts =
        SamplerOpts::from_preset(cfg.preset("stable")?, cfg.generation.repetition_window);
    opts.temperature = 1.0;
    opts.top_k = 0;
    opts.top_p = 1.0;
    opts.greedy = true;
    let t0 = Instant::now();
    let (toks, stop) = lm.generate(
        &prompt,
        &spk,
        &cfg.token_map,
        &opts,
        ref_greedy.len(),
        &mut Rng::new(0),
        |_| {},
    )?;
    let dt = t0.elapsed().as_secs_f32();
    let same = toks
        .iter()
        .zip(&ref_greedy)
        .take_while(|(a, b)| a == b)
        .count();
    println!(
        "greedy: {same}/{} tokens match the reference ({stop:?}), {:.1} tok/s incl. prefill",
        ref_greedy.len(),
        toks.len() as f32 / dt
    );
    if same < ref_greedy.len() {
        println!("  rlx {:?}\n  ref {:?}", &toks[same..], &ref_greedy[same..]);
    }
    // Teacher-forced pass over the reference tokens: per-step argmax and
    // top-2 margin after the sampler's penalties (what greedy decides on),
    // plus a checksum of the raw row, written for cross-device comparison.
    if let Some(path) = args.get(3) {
        let rows = lm.forced_logits(&prompt, &spk, &ref_greedy)?;
        let mut out = String::new();
        for (i, raw) in rows.iter().enumerate() {
            let mut r = raw.clone();
            let history: Vec<u32> = prompt.iter().chain(&ref_greedy[..i]).copied().collect();
            rlx_kittentts2::sampling::apply_penalties(
                &mut r,
                &history,
                prompt.len(),
                &cfg.token_map,
                &opts,
            );
            let mut idx: Vec<usize> = (0..r.len()).collect();
            idx.select_nth_unstable_by(1, |a, b| r[*b].total_cmp(&r[*a]));
            let (a, b) = if r[idx[0]] >= r[idx[1]] {
                (idx[0], idx[1])
            } else {
                (idx[1], idx[0])
            };
            let sum: f64 = raw.iter().map(|&v| v as f64).sum();
            out.push_str(&format!("{i} {a} {:.5} {sum:.4}\n", r[a] - r[b]));
        }
        std::fs::write(path, out)?;
        println!("forced logits summary → {path}");
    }
    Ok(())
}
