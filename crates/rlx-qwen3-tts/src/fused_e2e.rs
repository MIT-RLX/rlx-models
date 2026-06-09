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

//! End-to-end fusion plan and unified codec-AR hot loop.
//!
//! **Target:** one warmed RLX session where talker decode, code-predictor AR, and speech
//! decode share compile profiles / GPU KV — not three independent eager backends stitched in
//! host code.
//!
//! **Today (Metal):** talker + CP stay on hand-tuned CPU eager until native Metal
//! `inputs_embeds` bucketed decode is fixed upstream; speech `pre_transformer` may compile on
//! GPU while conv/vocoder stay CPU.
//!
//! **Production fused path:** host talker lm_head + [`CodecFrameFusedEngine::run_codec_frame`]
//! (one CP backbone + talker decode graph). Full megagraph is opt-in via
//! `RLX_QWEN3_TTS_CODEC_FRAME_MEGAGRAPH=1`.

use crate::code_predictor::CodePredictorEngine;
use crate::codec_frame_fused::CodecFrameFusedEngine;
use crate::compile_opts::{talker_decode_compile_device, talker_metal_native_compile};
use crate::config::TalkerConfig;
use crate::talker::engine::TalkerEngine;
use crate::talker::math::{
    apply_repetition_penalty, linear_logits_flat_into, sample_greedy_talker_codec,
};
use anyhow::{Context, Result};
use rlx_runtime::Device;
use std::time::Instant;

/// How each synthesis stage is executed under the e2e fusion policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StageBackend {
    /// Hand-tuned CPU micro-kernels (parity reference, fast on 0.6B today).
    CpuEager,
    /// RLX tier-1 graph + `Fusable` fusion (`BucketedCompileCache` / `CompileCache`).
    RlxFusedGraph,
    /// Hybrid: eager prefill + compiled decode (Metal `METAL_COMPILED=1` workaround).
    RlxHybrid,
}

/// Per-stage execution for the current device + env.
#[derive(Debug, Clone)]
pub struct E2EPipelinePlan {
    pub device: Device,
    pub talker: StageBackend,
    pub code_predictor: StageBackend,
    pub speech_pre_transformer: StageBackend,
    /// Conv / vocoder tail (always CPU eager today).
    pub speech_conv: StageBackend,
    /// True when every AR stage uses `RlxFusedGraph` on the session device.
    pub fully_fused_ar: bool,
    /// Blockers preventing full graph fusion (empty when fully fused).
    pub blockers: Vec<&'static str>,
}

impl E2EPipelinePlan {
    pub fn for_device(device: Device) -> Self {
        let talker = talker_stage(device);
        let cp = cp_stage(device);
        let speech_pt = speech_pt_stage(device);
        let mut blockers = Vec::new();

        if talker != StageBackend::RlxFusedGraph {
            if device == Device::Metal && !talker_metal_native_compile(device) {
                blockers.push("Metal talker: CPU eager default (set RLX_QWEN3_TTS_METAL_COMPILED=1 for hybrid)");
            }
            if device == Device::Metal && talker == StageBackend::RlxHybrid {
                if crate::gpu_pipeline::talker_eager_decode_default(device) {
                    blockers.push("Metal talker: CPU eager decode + CP megakernel (parity; RLX_QWEN3_TTS_METAL_DECODE_NATIVE=1 for native decode)");
                } else {
                    blockers.push("Metal talker decode on CPU graphs (RLX_QWEN3_TTS_METAL_DECODE_NATIVE=1 for native Metal decode)");
                }
            }
            if device == Device::Mlx {
                blockers.push("MLX talker: bucketed inputs_embeds mis-bind");
            }
        }
        if cp != StageBackend::RlxFusedGraph && !crate::gpu_pipeline::gpu_session_enabled(device) {
            blockers.push("CP: CPU eager (use --device metal/cuda for GPU talker+speech)");
        } else if cp == StageBackend::CpuEager
            && device == Device::Metal
            && crate::gpu_pipeline::gpu_session_enabled(device)
        {
            if crate::cp_megakernel::cp_megakernel_enabled(device) {
                blockers.push("CP: CPU eager megakernel on Metal (~58 ms/frame; Metal CP currently slower — only enable RLX_QWEN3_TTS_CP_METAL=1 for kernel work)");
            } else {
                blockers.push("CP: CPU eager on Metal (faster than compiled; Metal CP currently slower — only enable RLX_QWEN3_TTS_CP_METAL=1 for kernel work)");
            }
        }

        let fully_fused_ar =
            talker == StageBackend::RlxFusedGraph && cp == StageBackend::RlxFusedGraph;

        Self {
            device,
            talker,
            code_predictor: cp,
            speech_pre_transformer: speech_pt,
            speech_conv: if crate::gpu_pipeline::speech_conv_use_gpu(device) {
                StageBackend::RlxFusedGraph
            } else {
                StageBackend::CpuEager
            },
            fully_fused_ar,
            blockers,
        }
    }

    pub fn log_plan(&self) {
        eprintln!(
            "[qwen3-tts e2e] plan: talker={:?} cp={:?} speech_pt={:?} speech_conv={:?} fully_fused_ar={}",
            self.talker,
            self.code_predictor,
            self.speech_pre_transformer,
            self.speech_conv,
            self.fully_fused_ar,
        );
        if !self.blockers.is_empty() {
            eprintln!(
                "[qwen3-tts e2e] fusion blockers: {}",
                self.blockers.join("; ")
            );
        }
    }
}

fn talker_stage(device: Device) -> StageBackend {
    if crate::talker::engine::talker_use_eager_for_device(device) {
        if device == Device::Metal && crate::gpu_pipeline::gpu_session_enabled(device) {
            return StageBackend::RlxHybrid;
        }
        return StageBackend::CpuEager;
    }
    if device == Device::Metal && talker_metal_native_compile(device) {
        if talker_decode_compile_device(device) == Device::Cpu {
            return StageBackend::RlxHybrid;
        }
        return StageBackend::RlxFusedGraph;
    }
    StageBackend::RlxFusedGraph
}

fn cp_stage(device: Device) -> StageBackend {
    if crate::code_predictor::engine::cp_use_compiled_for_device(device) {
        StageBackend::RlxFusedGraph
    } else {
        StageBackend::CpuEager
    }
}

fn speech_pt_stage(device: Device) -> StageBackend {
    if crate::speech_tokenizer::speech_pt_use_compiled(device) {
        StageBackend::RlxFusedGraph
    } else {
        StageBackend::CpuEager
    }
}

/// Reused buffers for the codec-AR inner loop (no per-frame alloc).
pub struct CodecFrameScratch {
    pub hidden: Vec<f32>,
    pub codec_emb: Vec<f32>,
    pub logits: Vec<f32>,
}

impl CodecFrameScratch {
    pub fn new(hidden: usize, vocab: usize) -> Self {
        Self {
            hidden: vec![0f32; hidden],
            codec_emb: vec![0f32; hidden],
            logits: vec![0f32; vocab],
        }
    }
}

/// Per-substage ms for one codec frame (when timing is enabled).
#[derive(Debug, Clone, Copy, Default)]
pub struct CodecFrameTimings {
    pub talk_logits_ms: f64,
    pub cp_ms: f64,
    pub talk_decode_ms: f64,
}

/// Result of one codec-AR frame step.
pub enum CodecFrameStep {
    /// Produced a full codec frame (16 groups).
    Frame(Vec<u32>),
    /// EOS before `min_frames`; continue without emitting.
    EosSkip,
    /// EOS after enough frames; stop AR.
    Done,
}

/// One codec frame: host talker lm_head + CP + talker decode graph.
pub fn codec_frame_fused_step(
    talker: &mut TalkerEngine,
    fused: &mut CodecFrameFusedEngine,
    cp: Option<&mut CodePredictorEngine>,
    scratch: &mut CodecFrameScratch,
    talker_cfg: &TalkerConfig,
    pad: &[f32],
    rep_penalty: f32,
    past_g0: &mut Vec<u32>,
    frames_emitted: usize,
    min_frames: usize,
    timings: &mut CodecFrameTimings,
) -> Result<CodecFrameStep> {
    let eos = talker_cfg.codec_eos_token_id;
    let (head, vocab, hdim) = talker.codec_head_flat();
    let t0 = Instant::now();
    linear_logits_flat_into(
        scratch.hidden.as_slice(),
        head,
        vocab,
        hdim,
        &mut scratch.logits,
    )?;
    apply_repetition_penalty(&mut scratch.logits, past_g0, rep_penalty);
    let g0 = sample_greedy_talker_codec(&scratch.logits, talker_cfg.vocab_size, eos);
    timings.talk_logits_ms = t0.elapsed().as_secs_f64() * 1000.0;
    if g0 == eos {
        return Ok(if frames_emitted >= min_frames {
            CodecFrameStep::Done
        } else {
            CodecFrameStep::EosSkip
        });
    }
    past_g0.push(g0);

    if crate::synth_opts::codec_frame_megagraph_enabled() {
        return codec_frame_megagraph_step(talker, fused, cp, scratch, g0, pad, timings);
    }

    let t1 = Instant::now();
    let groups = match cp {
        Some(cp) if !cp.is_eager() => {
            cp.predict_groups_fill_emb(scratch.hidden.as_slice(), g0, pad, &mut scratch.codec_emb)?
        }
        _ => fused.predict_codec_groups(
            scratch.hidden.as_slice(),
            g0,
            pad,
            &mut scratch.codec_emb,
        )?,
    };
    timings.cp_ms = t1.elapsed().as_secs_f64() * 1000.0;
    let t2 = Instant::now();
    talker.decode_hidden_into(
        ndarray::ArrayView1::from(scratch.codec_emb.as_slice()),
        &mut scratch.hidden,
    )?;
    timings.talk_decode_ms = t2.elapsed().as_secs_f64() * 1000.0;
    Ok(CodecFrameStep::Frame(groups))
}

fn codec_frame_megagraph_step(
    talker: &mut TalkerEngine,
    fused: &mut CodecFrameFusedEngine,
    cp: Option<&mut CodePredictorEngine>,
    scratch: &mut CodecFrameScratch,
    g0: u32,
    pad: &[f32],
    timings: &mut CodecFrameTimings,
) -> Result<CodecFrameStep> {
    let t1 = Instant::now();
    let groups = match cp {
        Some(cp) if !cp.is_eager() => {
            cp.predict_groups_fill_emb(scratch.hidden.as_slice(), g0, pad, &mut scratch.codec_emb)?
        }
        _ => fused.predict_codec_groups(
            scratch.hidden.as_slice(),
            g0,
            pad,
            &mut scratch.codec_emb,
        )?,
    };
    timings.cp_ms = t1.elapsed().as_secs_f64() * 1000.0;

    let step_embeds = fused.cp_step_embeds_from_groups(&groups)?;
    let g0_embed = fused.codec_embed_row(0, g0)?;
    fused.set_cp_prefill_g0_embed(&g0_embed);

    let talker_hidden = scratch.hidden.clone();
    let t2 = Instant::now();
    fused.run_full_megagraph(
        talker,
        talker_hidden.as_slice(),
        scratch.codec_emb.as_slice(),
        &step_embeds,
        &mut scratch.hidden,
    )?;
    timings.talk_decode_ms = t2.elapsed().as_secs_f64() * 1000.0;
    Ok(CodecFrameStep::Frame(groups))
}

/// Dispatch fused or legacy codec-frame step.
pub fn codec_frame_step_dispatch(
    talker: &mut TalkerEngine,
    cp: Option<&mut CodePredictorEngine>,
    fused: Option<&mut CodecFrameFusedEngine>,
    scratch: &mut CodecFrameScratch,
    talker_cfg: &TalkerConfig,
    pad: &[f32],
    rep_penalty: f32,
    past_g0: &mut Vec<u32>,
    frames_emitted: usize,
    min_frames: usize,
    timings: &mut CodecFrameTimings,
) -> Result<CodecFrameStep> {
    if let Some(engine) = fused {
        codec_frame_fused_step(
            talker,
            engine,
            cp,
            scratch,
            talker_cfg,
            pad,
            rep_penalty,
            past_g0,
            frames_emitted,
            min_frames,
            timings,
        )
    } else {
        let cp = cp.context("codec_frame_step requires CP engine")?;
        codec_frame_step(
            talker,
            cp,
            scratch,
            talker_cfg,
            pad,
            rep_penalty,
            past_g0,
            frames_emitted,
            min_frames,
            timings,
        )
    }
}

/// One fused codec frame: talker lm_head → CP AR + embed sum → talker decode into `scratch.hidden`.
pub fn codec_frame_step(
    talker: &mut TalkerEngine,
    cp: &mut CodePredictorEngine,
    scratch: &mut CodecFrameScratch,
    talker_cfg: &TalkerConfig,
    pad: &[f32],
    rep_penalty: f32,
    past_g0: &mut Vec<u32>,
    frames_emitted: usize,
    min_frames: usize,
    timings: &mut CodecFrameTimings,
) -> Result<CodecFrameStep> {
    let eos = talker_cfg.codec_eos_token_id;
    let (head, vocab, hdim) = talker.codec_head_flat();
    let t0 = Instant::now();
    linear_logits_flat_into(
        scratch.hidden.as_slice(),
        head,
        vocab,
        hdim,
        &mut scratch.logits,
    )?;
    apply_repetition_penalty(&mut scratch.logits, past_g0, rep_penalty);
    let g0 = sample_greedy_talker_codec(&scratch.logits, talker_cfg.vocab_size, eos);
    timings.talk_logits_ms = t0.elapsed().as_secs_f64() * 1000.0;
    if g0 == eos {
        return Ok(if frames_emitted >= min_frames {
            CodecFrameStep::Done
        } else {
            CodecFrameStep::EosSkip
        });
    }
    let t1 = Instant::now();
    let groups =
        cp.predict_groups_fill_emb(scratch.hidden.as_slice(), g0, pad, &mut scratch.codec_emb)?;
    timings.cp_ms = t1.elapsed().as_secs_f64() * 1000.0;
    past_g0.push(g0);
    let t2 = Instant::now();
    talker.decode_hidden_into(
        ndarray::ArrayView1::from(&scratch.codec_emb),
        &mut scratch.hidden,
    )?;
    timings.talk_decode_ms = t2.elapsed().as_secs_f64() * 1000.0;
    Ok(CodecFrameStep::Frame(groups))
}
