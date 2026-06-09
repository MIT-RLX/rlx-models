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

//! A/B: hand-tuned CPU eager vs rlx fused bucketed decode (`BucketedCompileCache` + `FusionPolicy::Fusable`).

use crate::code_predictor::{CpBenchReport, CpCompiledEngine, CpEagerModel, bench_cp_ab};
use crate::config::CodePredictorConfig;
use crate::load::Qwen3TtsWeightStore;
use crate::talker::engine::TalkerEngine;
use anyhow::Result;
use ndarray::{Array2, ArrayView1};
use rlx_runtime::Device;
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct TalkerDecodeBenchReport {
    pub label: &'static str,
    pub device: Device,
    pub eager: bool,
    pub prefill_ms: f64,
    pub decode_ms: f64,
    pub steps: usize,
    pub ms_per_step: f64,
}

impl TalkerDecodeBenchReport {
    pub fn print_line(&self) {
        eprintln!(
            "[qwen3-tts fusion-bench] talker {} (eager={}): prefill={:.2}ms decode={:.2}ms ({:.2}ms/step, {} steps)",
            self.label, self.eager, self.prefill_ms, self.decode_ms, self.ms_per_step, self.steps,
        );
    }
}

pub struct FusionBenchSummary {
    pub talker_eager: TalkerDecodeBenchReport,
    pub talker_compiled: TalkerDecodeBenchReport,
    pub cp_eager: CpBenchReport,
    pub cp_compiled: CpBenchReport,
}

impl FusionBenchSummary {
    pub fn print_summary(&self) {
        self.talker_eager.print_line();
        self.talker_compiled.print_line();
        self.cp_eager.print_line();
        self.cp_compiled.print_line();
        let talker_win = if self.talker_compiled.ms_per_step < self.talker_eager.ms_per_step {
            "rlx fused (compiled)"
        } else {
            "CPU eager"
        };
        let cp_win = if self.cp_compiled.ms_per_frame < self.cp_eager.ms_per_frame {
            "rlx fused (compiled)"
        } else {
            "CPU eager"
        };
        eprintln!(
            "[qwen3-tts fusion-bench] talker winner: {talker_win} (Δ {:.2}ms/step)",
            (self.talker_eager.ms_per_step - self.talker_compiled.ms_per_step).abs()
        );
        eprintln!(
            "[qwen3-tts fusion-bench] CP winner: {cp_win} (Δ {:.2}ms/frame)",
            (self.cp_eager.ms_per_frame - self.cp_compiled.ms_per_frame).abs()
        );
        eprintln!(
            "[qwen3-tts fusion-bench] rlx stack: CompileProfile::qwen3_decode() → Fusable fusion → BucketedCompileCache + active_extent"
        );
    }
}

fn bench_talk_decode_with_eager_flag(
    store: &Qwen3TtsWeightStore,
    talker_cfg: &crate::config::TalkerConfig,
    session_device: Device,
    force_eager: bool,
    prefill_seq: usize,
    decode_steps: usize,
    label: &'static str,
) -> Result<TalkerDecodeBenchReport> {
    unsafe {
        if force_eager {
            std::env::set_var("RLX_QWEN3_TTS_TALKER_EAGER", "1");
        } else {
            std::env::set_var("RLX_QWEN3_TTS_TALKER_EAGER", "0");
        }
    }
    let hidden = talker_cfg.hidden_size;
    let mut talker = TalkerEngine::open(store, talker_cfg, session_device)?;
    let eager = talker.is_eager();
    let mut prefill = Array2::<f32>::zeros((prefill_seq.max(1), hidden));
    for (i, v) in prefill.iter_mut().enumerate() {
        *v = ((i % 97) as f32) * 1e-4;
    }
    talker.warmup(prefill_seq.max(1))?;

    let t0 = Instant::now();
    talker.reset_kv();
    talker.prefill(prefill.view())?;
    let prefill_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let t1 = Instant::now();
    let mut steps = 0usize;
    for step in 0..decode_steps {
        let mut emb = vec![0f32; hidden];
        emb[0] = (step as f32) * 1e-3;
        talker.decode_hidden_step(ndarray::ArrayView1::from(&emb))?;
        steps += 1;
    }
    let decode_ms = t1.elapsed().as_secs_f64() * 1000.0;

    Ok(TalkerDecodeBenchReport {
        label,
        device: session_device,
        eager,
        prefill_ms,
        decode_ms,
        steps,
        ms_per_step: decode_ms / steps.max(1) as f64,
    })
}

/// Compare talker eager vs CPU-compiled fused decode and CP eager vs compiled on one hidden row.
pub fn bench_fusion_ab(
    store: &Qwen3TtsWeightStore,
    cfg: &crate::config::Qwen3TtsConfig,
    session_device: Device,
    cp_frames: usize,
    talker_prefill_seq: usize,
    talker_decode_steps: usize,
    warmup: usize,
) -> Result<FusionBenchSummary> {
    let talker_cfg = cfg.talker();
    let cp_cfg = cfg.code_predictor();

    let mut talker = TalkerEngine::open(store, talker_cfg, session_device)?;
    talker.warmup(talker_prefill_seq.max(8))?;
    let hidden =
        talker.prefill(ndarray::Array2::<f32>::zeros((8, talker_cfg.hidden_size)).view())?;
    let h_last = hidden.row(hidden.nrows() - 1);

    for _ in 0..warmup {
        let _ = bench_cp_ab(store, cp_cfg, h_last.view(), cp_frames.clamp(1, 4), 1)?;
    }

    let talker_eager = bench_talk_decode_with_eager_flag(
        store,
        talker_cfg,
        session_device,
        true,
        talker_prefill_seq,
        talker_decode_steps,
        "eager",
    )?;
    let talker_compiled = bench_talk_decode_with_eager_flag(
        store,
        talker_cfg,
        session_device,
        false,
        talker_prefill_seq,
        talker_decode_steps,
        "rlx fused",
    )?;
    unsafe {
        std::env::remove_var("RLX_QWEN3_TTS_TALKER_EAGER");
    }

    let (cp_eager, cp_compiled) = bench_cp_ab(store, cp_cfg, h_last.view(), cp_frames, warmup)?;

    Ok(FusionBenchSummary {
        talker_eager,
        talker_compiled,
        cp_eager,
        cp_compiled,
    })
}

/// Quick CP-only fused vs eager (no env mutation).
pub fn bench_cp_fused_vs_eager_one(
    store: &Qwen3TtsWeightStore,
    cp: &CodePredictorConfig,
    hidden: ArrayView1<f32>,
    group0: u32,
) -> Result<(f64, f64)> {
    let talker_snap = store.tensor_snapshot(&["talker.model.codec_embedding.weight"])?;
    let (tc_data, tc_shape) = talker_snap["talker.model.codec_embedding.weight"].clone();
    let talker_codec = Array2::from_shape_vec((tc_shape[0], tc_shape[1]), tc_data)?;
    let mut group_embeds = Vec::new();
    let mut lm_heads = Vec::new();
    for i in 0..cp.num_code_groups - 1 {
        let key = format!("talker.code_predictor.model.codec_embedding.{i}.weight");
        let (data, shape) = store.tensor_snapshot(&[&key])?[&key].clone();
        group_embeds.push(Array2::from_shape_vec((shape[0], shape[1]), data)?);
        let hkey = format!("talker.code_predictor.lm_head.{i}.weight");
        let (data, shape) = store.tensor_snapshot(&[&hkey])?[&hkey].clone();
        lm_heads.push(Array2::from_shape_vec((shape[0], shape[1]), data)?);
    }

    let t0 = Instant::now();
    let mut eager = CpEagerModel::open(store, cp)?;
    eager.predict_groups(&talker_codec, &group_embeds, &lm_heads, hidden, group0)?;
    let eager_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let t1 = Instant::now();
    let mut compiled = CpCompiledEngine::open(store.model_dir(), store, cp, Device::Cpu)?;
    compiled.warmup(16)?;
    compiled.predict_groups(&talker_codec, &group_embeds, &lm_heads, hidden, group0)?;
    let compiled_ms = t1.elapsed().as_secs_f64() * 1000.0;

    Ok((eager_ms, compiled_ms))
}
