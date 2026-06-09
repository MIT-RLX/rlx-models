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

//! Micro-benchmark: CP `predict_groups` (eager vs CPU compiled).

use crate::code_predictor::compiled::CpCompiledEngine;
use crate::code_predictor::eager::CpEagerModel;
use crate::config::CodePredictorConfig;
use crate::load::Qwen3TtsWeightStore;
use anyhow::Result;
use ndarray::{Array2, ArrayView1};
use rlx_runtime::Device;
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpBenchBackend {
    Eager,
    CompiledCpu,
}

pub struct CpBenchReport {
    pub backend: CpBenchBackend,
    pub frames: usize,
    pub total_ms: f64,
    pub ms_per_frame: f64,
}

impl CpBenchReport {
    pub fn print_line(&self) {
        let label = match self.backend {
            CpBenchBackend::Eager => "CPU eager",
            CpBenchBackend::CompiledCpu => "CPU compiled",
        };
        eprintln!(
            "[qwen3-tts cp-bench] {label}: {frames} frames total={total:.2}ms ({per:.2}ms/frame)",
            frames = self.frames,
            total = self.total_ms,
            per = self.ms_per_frame,
        );
    }
}

fn predict_groups_loop(
    store: &Qwen3TtsWeightStore,
    cp: &CodePredictorConfig,
    talker_codec: &Array2<f32>,
    group_embeds: &[Array2<f32>],
    lm_heads: &[Array2<f32>],
    hidden: ArrayView1<f32>,
    frames: usize,
    backend: CpBenchBackend,
) -> Result<CpBenchReport> {
    let group0 = 1995u32;
    let t0 = Instant::now();
    match backend {
        CpBenchBackend::Eager => {
            let mut model = CpEagerModel::open(store, cp)?;
            for _ in 0..frames {
                let _ =
                    model.predict_groups(talker_codec, group_embeds, lm_heads, hidden, group0)?;
            }
        }
        CpBenchBackend::CompiledCpu => {
            let mut model = CpCompiledEngine::open(store.model_dir(), store, cp, Device::Cpu)?;
            model.warmup(frames.max(8))?;
            for _ in 0..frames {
                let _ =
                    model.predict_groups(talker_codec, group_embeds, lm_heads, hidden, group0)?;
            }
        }
    }
    let total_ms = t0.elapsed().as_secs_f64() * 1000.0;
    Ok(CpBenchReport {
        backend,
        frames,
        total_ms,
        ms_per_frame: total_ms / frames.max(1) as f64,
    })
}

/// Time `predict_groups` for `frames` iterations with fixed `hidden` (talker last row).
pub fn bench_cp_predict_groups(
    store: &Qwen3TtsWeightStore,
    cp: &CodePredictorConfig,
    hidden: ArrayView1<f32>,
    frames: usize,
    backend: CpBenchBackend,
) -> Result<CpBenchReport> {
    let talker_snap = store.tensor_snapshot(&["talker.model.codec_embedding.weight"])?;
    let (tc_data, tc_shape) = talker_snap
        .get("talker.model.codec_embedding.weight")
        .expect("codec_embedding");
    let talker_codec = Array2::from_shape_vec((tc_shape[0], tc_shape[1]), tc_data.clone())?;

    let mut group_embeds = Vec::with_capacity(cp.num_code_groups - 1);
    let mut lm_heads = Vec::with_capacity(cp.num_code_groups - 1);
    for i in 0..cp.num_code_groups - 1 {
        let key = format!("talker.code_predictor.model.codec_embedding.{i}.weight");
        let (data, shape) = store.tensor_snapshot(&[&key])?[&key].clone();
        group_embeds.push(Array2::from_shape_vec((shape[0], shape[1]), data)?);
        let hkey = format!("talker.code_predictor.lm_head.{i}.weight");
        let (data, shape) = store.tensor_snapshot(&[&hkey])?[&hkey].clone();
        lm_heads.push(Array2::from_shape_vec((shape[0], shape[1]), data)?);
    }

    predict_groups_loop(
        store,
        cp,
        &talker_codec,
        &group_embeds,
        &lm_heads,
        hidden,
        frames,
        backend,
    )
}

/// A/B eager vs CPU compiled; returns (eager, compiled).
pub fn bench_cp_ab(
    store: &Qwen3TtsWeightStore,
    cp: &CodePredictorConfig,
    hidden: ArrayView1<f32>,
    frames: usize,
    warmup: usize,
) -> Result<(CpBenchReport, CpBenchReport)> {
    for _ in 0..warmup {
        let _ =
            bench_cp_predict_groups(store, cp, hidden, frames.clamp(1, 4), CpBenchBackend::Eager)?;
        let _ = bench_cp_predict_groups(
            store,
            cp,
            hidden,
            frames.clamp(1, 4),
            CpBenchBackend::CompiledCpu,
        )?;
    }
    let eager = bench_cp_predict_groups(store, cp, hidden, frames, CpBenchBackend::Eager)?;
    let compiled = bench_cp_predict_groups(store, cp, hidden, frames, CpBenchBackend::CompiledCpu)?;
    Ok((eager, compiled))
}
