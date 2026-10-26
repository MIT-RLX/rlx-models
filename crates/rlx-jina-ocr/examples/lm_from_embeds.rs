// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! Run the decoder on caller-supplied `inputs_embeds`, bypassing the vision
//! tower entirely.
//!
//! Isolates "is the LM right" from "is the vision pack right": feed the
//! reference implementation's own `inputs_embeds` and the prefill logits must
//! match its logits.
//!
//! ```bash
//! cargo run -p rlx-jina-ocr --release --example lm_from_embeds -- \
//!     --embeds /tmp/ref_inputs_embeds.f32 --steps 8
//! ```

use anyhow::{Context, Result, ensure};
use rlx_jina_ocr::config::JinaOcrConfig;
use rlx_jina_ocr::hub::default_model_dir;
use rlx_unlimited_ocr::device::resolve_device;
use rlx_unlimited_ocr::expert_pack::PackedLmWeights;
use rlx_unlimited_ocr::lm_device::CompiledLm;
use rlx_unlimited_ocr::lm_precision::LmWeightPrecision;
use rlx_unlimited_ocr::weights::UnlimitedOcrWeightStore;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut embeds_path = PathBuf::from("/tmp/ref_inputs_embeds.f32");
    let mut steps = 0usize;
    let mut truncate: Option<usize> = None;
    let mut top_k: Option<usize> = None;
    let mut device_name: Option<String> = None;
    let mut precision = LmWeightPrecision::F32;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--embeds" => {
                embeds_path = PathBuf::from(&args[i + 1]);
                i += 2;
            }
            "--tokens" => {
                truncate = Some(args[i + 1].parse()?);
                i += 2;
            }
            "--top-k" => {
                top_k = Some(args[i + 1].parse()?);
                i += 2;
            }
            "--steps" => {
                steps = args[i + 1].parse()?;
                i += 2;
            }
            "--device" => {
                device_name = Some(args[i + 1].clone());
                i += 2;
            }
            "--lm-precision" => {
                precision = LmWeightPrecision::parse(&args[i + 1]).context("--lm-precision")?;
                i += 2;
            }
            other => anyhow::bail!("unknown flag {other}"),
        }
    }

    let dir = default_model_dir()?;
    let mut cfg = JinaOcrConfig::from_model_dir(&dir)?;
    cfg.validate()?;
    if let Some(k) = top_k {
        // A/B knob, not a model setting: eager and compiled must agree for any
        // k, so a k where they agree and a k where they do not separates
        // expert *selection* from expert *arithmetic*.
        cfg.lm.num_experts_per_tok = k;
        println!("num_experts_per_tok overridden to {k}");
    }
    let hidden = cfg.lm.hidden_size;

    let raw = std::fs::read(&embeds_path).with_context(|| format!("read {embeds_path:?}"))?;
    ensure!(
        raw.len().is_multiple_of(4),
        "embeds file is not f32-aligned"
    );
    let embeds: Vec<f32> = raw
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    ensure!(
        embeds.len().is_multiple_of(hidden),
        "embeds len {} not a multiple of hidden {hidden}",
        embeds.len()
    );
    let mut embeds = embeds;
    if let Some(t) = truncate {
        embeds.truncate(t * hidden);
    }
    let n_tokens = embeds.len() / hidden;
    println!("embeds: {n_tokens} x {hidden} from {embeds_path:?}");

    let device = resolve_device(device_name.as_deref())?;
    let store = UnlimitedOcrWeightStore::open(&dir)?;

    // Eager host path: no graph compiler, no arena, no backend kernels. If this
    // agrees with the reference and the compiled path does not, the bug is in
    // lowering/execution rather than in the algorithm.
    if rlx_ir::env::var_result("RLX_JINA_EAGER").is_ok() {
        use rlx_unlimited_ocr::lm_flow::LmFlow;
        let mut flow = LmFlow::from_config(cfg.lm.clone());
        flow.load(&store)?;
        let (logits, kv) = flow.prefill(&embeds, n_tokens)?;
        report("eager prefill", &logits);
        let stem = embeds_path.with_extension("eagerkv");
        let mut f = std::fs::File::create(&stem)?;
        for l in 0..kv.num_layers() {
            for t in [kv.layer_k(l), kv.layer_v(l)] {
                for v in t.iter() {
                    f.write_all(&v.to_le_bytes())?;
                }
            }
        }
        println!("wrote {stem:?}");
        return Ok(());
    }
    let pack = Arc::new(PackedLmWeights::from_store_for_device(
        &store,
        &cfg.lm,
        precision,
        Some(device),
    )?);
    let mut lm = CompiledLm::new(device, Arc::clone(&pack));

    // Raw prefill graph: outputs are [logits, k0, v0, k1, v1, ...], so the
    // per-layer KV localizes a divergence to the layer that introduces it.
    if rlx_ir::env::var_result("RLX_JINA_DUMP_KV").is_ok() {
        use rlx_core::flow_util::compile_built;
        use rlx_unlimited_ocr::lm_graph::build_unlimited_ocr_prefill_built_from_pack;
        let built = build_unlimited_ocr_prefill_built_from_pack(&cfg.lm, &pack, 1, n_tokens)?;
        let mut compiled = compile_built(built, device)?;
        let outs = compiled.run(&[("inputs_embeds", embeds.as_slice())]);
        println!("raw prefill outputs: {}", outs.len());
        for l in 0..cfg.lm.num_hidden_layers {
            let k = &outs[1 + 2 * l];
            let v = &outs[1 + 2 * l + 1];
            println!(
                "  layer {l:2}: k mean {:+.5} std {:.5} absmax {:.3} | v mean {:+.5} std {:.5} absmax {:.3}",
                mean(k),
                std(k),
                absmax(k),
                mean(v),
                std(v),
                absmax(v)
            );
        }
        let stem = embeds_path.with_extension("rlxkv");
        let mut f = std::fs::File::create(&stem)?;
        for l in 0..cfg.lm.num_hidden_layers {
            for t in [&outs[1 + 2 * l], &outs[1 + 2 * l + 1]] {
                for v in t.iter() {
                    f.write_all(&v.to_le_bytes())?;
                }
            }
        }
        println!("wrote {stem:?}");
    }

    let (logits, mut kv) = lm.prefill(&embeds, n_tokens)?;
    report("prefill", &logits);

    let mut ids = Vec::new();
    let mut logits = logits;
    for s in 0..steps {
        let next = argmax(&logits);
        ids.push(next);
        if next == cfg.eos_token_id() {
            println!("hit EOS at step {s}");
            break;
        }
        let step_embed = lm.embed_tokens(&[next])?;
        logits = lm.decode_step(&step_embed, n_tokens + s, &mut kv)?;
    }
    if !ids.is_empty() {
        println!("greedy ids: {ids:?}");
    }
    Ok(())
}

fn report(label: &str, logits: &[f32]) {
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
    let top: Vec<(usize, f32)> = idx[..10]
        .iter()
        .map(|&i| (i, (logits[i] * 1000.0).round() / 1000.0))
        .collect();
    let finite = logits.iter().filter(|v| v.is_finite()).count();
    println!(
        "{label}: argmax={} finite={finite}/{} top10={top:?}",
        idx[0],
        logits.len()
    );
}

fn mean(x: &[f32]) -> f32 {
    x.iter().sum::<f32>() / x.len() as f32
}

fn std(x: &[f32]) -> f32 {
    let m = mean(x);
    (x.iter().map(|v| (v - m).powi(2)).sum::<f32>() / x.len() as f32).sqrt()
}

fn absmax(x: &[f32]) -> f32 {
    x.iter().fold(0f32, |a, v| a.max(v.abs()))
}

fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, &v) in logits.iter().enumerate() {
        if v > logits[best] {
            best = i;
        }
    }
    best as u32
}
