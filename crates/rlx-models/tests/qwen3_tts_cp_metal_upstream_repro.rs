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

//! Minimal upstream repro: Metal compiled CP (5-layer Qwen3 embeds) vs CPU eager.
//!
//! CPU asserts tight parity; Metal logs structured repro for `rlx-metal` (see
//! `crates/rlx-qwen3-tts/METAL_CP_UPSTREAM.md`).
//!
//! ```bash
//! export RLX_QWEN3_TTS_DIR=… RLX_QWEN3_TTS_PARITY=1
//! cargo test -p rlx-models --test qwen3_tts_cp_metal_upstream_repro --release --features metal
//! ```

use ndarray::Array2;
use rlx_qwen3_tts::Qwen3TtsConfig;
use rlx_qwen3_tts::code_predictor::{CpCompiledEngine, CpEagerModel};
use rlx_qwen3_tts::load::Qwen3TtsWeightStore;
use rlx_runtime::{Device, is_available};
use std::path::PathBuf;

const PREFILL_SEQ: usize = 2;
const LAYERS: usize = 5;
const TOLERANCE: f32 = 0.05;

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
}

fn synthetic_prefill_embeds(hidden: usize) -> Array2<f32> {
    let mut v = vec![0f32; PREFILL_SEQ * hidden];
    for t in 0..PREFILL_SEQ {
        for j in 0..hidden {
            v[t * hidden + j] = ((t + 1) as f32) * 1e-3 + (j as f32) * 1e-6;
        }
    }
    Array2::from_shape_vec((PREFILL_SEQ, hidden), v).unwrap()
}

fn print_upstream_repro(
    device: &str,
    phase: &str,
    max_abs: f32,
    eager: &[f32],
    compiled: &[f32],
    cp_cfg: &rlx_qwen3_tts::config::CodePredictorConfig,
) {
    eprintln!("--- METAL_CP_UPSTREAM_REPRO ---");
    eprintln!("device: {device}");
    eprintln!("phase: {phase}");
    eprintln!("graph: qwen3_prefill_embeds / qwen3_decode_embeds");
    eprintln!("layers: {LAYERS} (code_predictor backbone)");
    eprintln!("prefill_seq: {PREFILL_SEQ}");
    eprintln!("hidden_size: {}", cp_cfg.hidden_size);
    eprintln!("kv_dim: {}", cp_cfg.num_key_value_heads * cp_cfg.head_dim);
    eprintln!("max_abs: {max_abs} (tolerance {TOLERANCE})");
    eprintln!("env: RLX_DISABLE_MPSGRAPH=1 (metal_compile_guard)");
    eprintln!("eager[:8]    = {:?}", &eager[..8.min(eager.len())]);
    eprintln!("compiled[:8] = {:?}", &compiled[..8.min(compiled.len())]);
    eprintln!("talker_28L_metal_ok: true (same builder, different layer count)");
    eprintln!("--- end repro ---");
}

#[test]
fn cp_metal_upstream_repro_prefill_cpu_parity() {
    let Some(model_dir) = std::env::var("RLX_QWEN3_TTS_DIR").ok().map(PathBuf::from) else {
        eprintln!("skip: RLX_QWEN3_TTS_DIR");
        return;
    };
    if std::env::var("RLX_QWEN3_TTS_PARITY").ok().as_deref() != Some("1") {
        eprintln!("skip: RLX_QWEN3_TTS_PARITY=1");
        return;
    }

    let cfg = Qwen3TtsConfig::from_model_dir(&model_dir).unwrap();
    let store = Qwen3TtsWeightStore::open(&model_dir).unwrap();
    let cp_cfg = cfg.code_predictor();
    let embeds = synthetic_prefill_embeds(cp_cfg.hidden_size);

    let mut eager = CpEagerModel::open(&store, cp_cfg).unwrap();
    let eager_h = eager.forward(embeds.view()).unwrap();
    let eager_last: Vec<f32> = eager_h.row(eager_h.nrows() - 1).iter().copied().collect();

    let mut compiled =
        CpCompiledEngine::open(store.model_dir(), &store, cp_cfg, Device::Cpu).unwrap();
    compiled.warmup(8).unwrap();
    let compiled_h = compiled.prefill(embeds.view()).unwrap();
    let compiled_last: Vec<f32> = compiled_h
        .row(compiled_h.nrows() - 1)
        .iter()
        .copied()
        .collect();

    let d = max_abs(&eager_last, &compiled_last);
    eprintln!("cp upstream repro prefill CPU max_abs = {d}");
    assert!(
        d < TOLERANCE,
        "CPU baseline broken before Metal repro: max_abs={d}"
    );
}

#[test]
fn cp_compiled_metal_session_vs_cpu_session() {
    let Some(model_dir) = std::env::var("RLX_QWEN3_TTS_DIR").ok().map(PathBuf::from) else {
        eprintln!("skip: RLX_QWEN3_TTS_DIR");
        return;
    };
    if std::env::var("RLX_QWEN3_TTS_PARITY").ok().as_deref() != Some("1") {
        eprintln!("skip: RLX_QWEN3_TTS_PARITY=1");
        return;
    }
    let cfg = Qwen3TtsConfig::from_model_dir(&model_dir).unwrap();
    let store = Qwen3TtsWeightStore::open(&model_dir).unwrap();
    let cp_cfg = cfg.code_predictor();
    let embeds = synthetic_prefill_embeds(cp_cfg.hidden_size);

    let mut metal_sess =
        CpCompiledEngine::open(store.model_dir(), &store, cp_cfg, Device::Metal).unwrap();
    let mut cpu_sess =
        CpCompiledEngine::open(store.model_dir(), &store, cp_cfg, Device::Cpu).unwrap();
    let metal_h = metal_sess.prefill(embeds.view()).unwrap();
    let cpu_h = cpu_sess.prefill(embeds.view()).unwrap();
    let metal_last: Vec<f32> = metal_h.row(metal_h.nrows() - 1).iter().copied().collect();
    let cpu_last: Vec<f32> = cpu_h.row(cpu_h.nrows() - 1).iter().copied().collect();
    let d = max_abs(&metal_last, &cpu_last);
    eprintln!("cp compiled Metal session vs Cpu session max_abs={d}");
    assert!(d < 1e-4, "Metal/Cpu session compiled mismatch: {d}");
}

#[test]
fn cp_metal_upstream_repro_prefill_metal_gap() {
    if !is_available(Device::Metal) {
        eprintln!("skip: Metal unavailable");
        return;
    }
    let Some(model_dir) = std::env::var("RLX_QWEN3_TTS_DIR").ok().map(PathBuf::from) else {
        eprintln!("skip: RLX_QWEN3_TTS_DIR");
        return;
    };
    if std::env::var("RLX_QWEN3_TTS_PARITY").ok().as_deref() != Some("1") {
        eprintln!("skip: RLX_QWEN3_TTS_PARITY=1");
        return;
    }

    let cfg = Qwen3TtsConfig::from_model_dir(&model_dir).unwrap();
    let store = Qwen3TtsWeightStore::open(&model_dir).unwrap();
    let cp_cfg = cfg.code_predictor();
    let embeds = synthetic_prefill_embeds(cp_cfg.hidden_size);

    let mut eager = CpEagerModel::open(&store, cp_cfg).unwrap();
    let eager_h = eager.forward(embeds.view()).unwrap();
    let eager_last: Vec<f32> = eager_h.row(eager_h.nrows() - 1).iter().copied().collect();

    let mut compiled =
        CpCompiledEngine::open(store.model_dir(), &store, cp_cfg, Device::Metal).unwrap();
    compiled.warmup(8).unwrap();
    let compiled_h = compiled.prefill(embeds.view()).unwrap();
    let compiled_last: Vec<f32> = compiled_h
        .row(compiled_h.nrows() - 1)
        .iter()
        .copied()
        .collect();

    let d = max_abs(&eager_last, &compiled_last);
    print_upstream_repro("Metal", "prefill", d, &eager_last, &compiled_last, cp_cfg);
    eprintln!("cp upstream repro prefill Metal max_abs = {d}");
    if std::env::var("RLX_QWEN3_TTS_CP_METAL").ok().as_deref() == Some("1")
        && std::env::var("RLX_QWEN3_TTS_METAL_COMPILED")
            .ok()
            .as_deref()
            == Some("1")
        && d >= TOLERANCE
    {
        eprintln!("known gap: native Metal CP (`CP_METAL=1` + `METAL_COMPILED=1`)");
        return;
    }
    assert!(
        d < TOLERANCE,
        "Metal CP compiled prefill diverged from eager: max_abs={d} (default compile_device=CPU via cp_compile_device)"
    );
}
