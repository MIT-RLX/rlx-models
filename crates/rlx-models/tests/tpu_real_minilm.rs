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

//! End-to-end real-model parity: download MiniLM-L6-v2 from HF Hub,
//! compile the same graph on `Device::Cpu` and `Device::Tpu`, run
//! forward, compare hidden states.
//!
//! Why MiniLM-L6 specifically: 22 MB safetensors so the HF download
//! is fast, 6 transformer layers + standard BERT architecture so the
//! whole rlx-models pipeline (config parse, safetensors load,
//! `build_bert_graph_sized`) gets exercised. Same graph on both
//! backends — the comparison is pure compiler / runtime behavior, no
//! tokenizer state.
//!
//! Gated on `tpu` + `hf-download` features and `LIBTPU_PATH`. On
//! hosts without a PJRT plugin (most macOS/Windows dev boxes) the
//! test skips cleanly.

#![cfg(all(feature = "tpu", feature = "hf-download"))]

mod compile_support;

use std::path::Path;

use anyhow::Result;
use rlx_models::{BertConfig, WeightMap, build_bert_graph_sized};
use rlx_runtime::{Device, PrecisionPolicy, Session};

fn skip_without_plugin() -> bool {
    if std::env::var("LIBTPU_PATH").is_err() {
        eprintln!("[tpu_real_minilm] LIBTPU_PATH not set — skipping");
        return true;
    }
    if rlx_ir::env::is_unset("RLX_REAL_MODEL") {
        eprintln!(
            "[tpu_real_minilm] RLX_REAL_MODEL not set — \
                   skipping (the test downloads ~22 MB from HF on \
                   first run and is slow even when cached; opt in \
                   with RLX_REAL_MODEL=1)"
        );
        return true;
    }
    false
}

fn fetch_files(repo_id: &str) -> Result<(String, String)> {
    let repo = hf_hub::api::sync::ApiBuilder::new()
        .with_progress(true)
        .build()?
        .model(repo_id.to_string());
    let config = repo.get("config.json")?;
    let weights = repo.get("model.safetensors")?;
    Ok((
        config.to_string_lossy().into_owned(),
        weights.to_string_lossy().into_owned(),
    ))
}

fn run_on(
    device: Device,
    policy: PrecisionPolicy,
    cfg: &BertConfig,
    weights_path: &str,
    batch: usize,
    seq: usize,
    inputs: &[(&str, &[f32])],
) -> Result<Vec<f32>> {
    let mut wm = WeightMap::from_file(weights_path)?;
    let (graph, params) = build_bert_graph_sized(cfg, &mut wm, batch, seq)?;
    let mut compiled = Session::new(device)
        .with_policy(policy)
        .compile_with(graph, &rlx_runtime::CompileOptions::new());
    for (name, data) in &params {
        compiled.set_param(name, data);
    }
    let outputs = compiled.run(inputs);
    Ok(outputs.into_iter().next().unwrap_or_default())
}

#[test]
fn minilm_l6_cpu_vs_tpu_f32() -> Result<()> {
    if skip_without_plugin() {
        return Ok(());
    }

    let repo_id = "sentence-transformers/all-MiniLM-L6-v2";
    let (config_path, weights_path) = fetch_files(repo_id)?;
    let cfg = BertConfig::from_file(Path::new(&config_path))?;

    // 1×8 input — tokenizer-shaped: input_ids, attention_mask,
    // token_type_ids. Use deterministic small token ids; we're not
    // checking semantic correctness, just numerical agreement.
    let batch = 1;
    let seq = 8;
    let input_ids: Vec<f32> = (0..(batch * seq)).map(|i| ((i + 1) % 100) as f32).collect();
    let attn_mask: Vec<f32> = vec![1.0; batch * seq];
    let token_type_ids: Vec<f32> = vec![0.0; batch * seq];
    let position_ids: Vec<f32> = (0..(batch * seq)).map(|i| (i % seq) as f32).collect();
    let inputs: Vec<(&str, &[f32])> = vec![
        ("input_ids", &input_ids),
        ("attention_mask", &attn_mask),
        ("token_type_ids", &token_type_ids),
        ("position_ids", &position_ids),
    ];

    let cpu_out = run_on(
        Device::Cpu,
        PrecisionPolicy::AlwaysF32,
        &cfg,
        &weights_path,
        batch,
        seq,
        &inputs,
    )?;
    let tpu_out = run_on(
        Device::Tpu,
        PrecisionPolicy::AlwaysF32,
        &cfg,
        &weights_path,
        batch,
        seq,
        &inputs,
    )?;

    assert_eq!(
        cpu_out.len(),
        tpu_out.len(),
        "output sizes diverge: cpu={} tpu={}",
        cpu_out.len(),
        tpu_out.len()
    );
    let n = cpu_out.len();
    let mut max_err = 0.0f32;
    let mut sum_abs_err = 0.0f64;
    let mut sum_abs_ref = 0.0f64;
    for i in 0..n {
        let a = cpu_out[i];
        let bv = tpu_out[i];
        max_err = max_err.max((a - bv).abs());
        sum_abs_err += (a - bv).abs() as f64;
        sum_abs_ref += a.abs() as f64;
    }
    let rel_err = if sum_abs_ref > 0.0 {
        sum_abs_err / sum_abs_ref
    } else {
        sum_abs_err
    };
    eprintln!(
        "[tpu_real_minilm] MiniLM-L6 (B={batch}, S={seq}) \
               f32 max abs = {max_err:e}, mean rel = {rel_err:.4}"
    );
    // 6 transformer layers compound rounding error; allow ~5%
    // mean relative error. Real production serving compares cosine
    // similarity downstream and tolerates far more than this.
    assert!(
        rel_err < 0.05,
        "MiniLM-L6 CPU vs TPU diverge by {rel_err:.4} (>5%)"
    );
    Ok(())
}
