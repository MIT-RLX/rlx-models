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

//! Backend-tuned fused talker + code-predictor runtime (warm compile caches per device).

use crate::code_predictor::CodePredictorEngine;
use crate::codec_frame_fused::CodecFrameFusedEngine;
use crate::compile_opts::{ensure_metal_lowering_env, metal_compile_guard};
use crate::config::{CodePredictorConfig, TalkerConfig};
use crate::fused_e2e::{
    CodecFrameScratch, CodecFrameStep, CodecFrameTimings, E2EPipelinePlan,
    codec_frame_step_dispatch,
};
use crate::load::Qwen3TtsWeightStore;
use crate::progress::Progress;
use crate::talker::engine::TalkerEngine;
use anyhow::{Context, Result};
use ndarray::{Array2, ArrayView1, ArrayView2};

type FrameCallback<'a> = Option<&'a mut dyn FnMut(usize, &[u32])>;
use rlx_runtime::Device;
use std::path::Path;
use std::time::Instant;

/// Warmed talker + code predictor sharing per-device compile profiles.
pub struct Qwen3TtsMegakernel {
    talker: TalkerEngine,
    /// Legacy CP path only when codec-frame fusion is off.
    cp: Option<CodePredictorEngine>,
    fused: Option<CodecFrameFusedEngine>,
    device: Device,
    /// Prefill hidden from warmup (skipped when lazy buckets force a fresh prefill).
    prefill_hidden: Option<Array2<f32>>,
    /// Decode buckets dry-run through this past+frames horizon during warmup.
    pipeline_horizon: usize,
    decode_pipeline_ready: bool,
}

impl Qwen3TtsMegakernel {
    pub fn open(
        store: &Qwen3TtsWeightStore,
        talker_cfg: &TalkerConfig,
        cp_cfg: &CodePredictorConfig,
        device: Device,
    ) -> Result<Self> {
        Self::open_at(store.model_dir(), store, talker_cfg, cp_cfg, device)
    }

    pub fn open_at(
        _model_dir: &Path,
        store: &Qwen3TtsWeightStore,
        talker_cfg: &TalkerConfig,
        cp_cfg: &CodePredictorConfig,
        device: Device,
    ) -> Result<Self> {
        ensure_metal_lowering_env(device);
        E2EPipelinePlan::for_device(device).log_plan();
        let debug = std::env::var("RLX_QWEN3_TTS_OPEN_TIMING").ok().as_deref() == Some("1");
        let t = std::time::Instant::now();
        let talker = TalkerEngine::open(store, talker_cfg, device)?;
        if debug {
            eprintln!("[open] talker engine: {:.3}s", t.elapsed().as_secs_f64());
        }
        let fused_on = crate::synth_opts::codec_frame_fused_enabled(device);
        let t = std::time::Instant::now();
        let fused = if fused_on {
            Some(CodecFrameFusedEngine::open(
                store, talker_cfg, cp_cfg, device,
            )?)
        } else {
            None
        };
        if debug && fused_on {
            eprintln!(
                "[open] codec-frame fused engine: {:.3}s",
                t.elapsed().as_secs_f64()
            );
        }
        let t = std::time::Instant::now();
        let cp = if fused_on {
            None
        } else {
            Some(CodePredictorEngine::open(store, cp_cfg, device)?)
        };
        if debug && !fused_on {
            eprintln!("[open] CP engine: {:.3}s", t.elapsed().as_secs_f64());
        }
        Ok(Self {
            talker,
            cp,
            fused,
            device,
            prefill_hidden: None,
            pipeline_horizon: 0,
            decode_pipeline_ready: false,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    /// Compile prefill + decode buckets through `prefill_embeds` + `max_frames` + one CP step.
    pub fn warmup(
        &mut self,
        prefill_embeds: ArrayView2<f32>,
        max_frames: usize,
        progress: Option<&Progress>,
    ) -> Result<()> {
        if let Some(p) = progress {
            p.set(1, &format!("talker prefill ({:?})", self.device));
        }
        let hidden = metal_compile_guard(self.device, || {
            self.talker.warmup_embeds(prefill_embeds, max_frames)
        })?;
        let prefill_seq = prefill_embeds.nrows();
        let horizon = prefill_seq.saturating_add(max_frames);
        // Eager attention scratch defaults to 256 tokens; grow it for longer
        // prompts/AR horizons so we don't slice OOB during decode.
        self.talker.ensure_eager_horizon(horizon);
        let talker_eager = self.talker.is_eager();
        if !talker_eager {
            if crate::synth_opts::auto_precompile_horizon(max_frames) {
                self.talker
                    .precompile_decode_buckets_up_to(horizon, progress)?;
                if crate::synth_opts::megakernel_fast_path() {
                    metal_compile_guard(self.device, || {
                        self.talker.preinstall_gpu_kv_horizon(horizon)
                    })?;
                    if crate::synth_opts::talk_bucket_execution_warmup(horizon) {
                        metal_compile_guard(self.device, || {
                            self.talker.warmup_bucket_executions(horizon)
                        })?;
                    }
                }
                self.prefill_hidden = Some(hidden);
            } else {
                self.talker.precompile_decode_bucket_for_past(prefill_seq)?;
                if crate::synth_opts::megakernel_fast_path() {
                    metal_compile_guard(self.device, || {
                        self.talker.preinstall_gpu_kv_horizon(horizon)
                    })?;
                    if crate::synth_opts::talk_bucket_execution_warmup(horizon) {
                        metal_compile_guard(self.device, || {
                            self.talker.warmup_bucket_executions(horizon)
                        })?;
                    }
                }
            }
        } else {
            self.prefill_hidden = Some(hidden);
        }
        if let Some(cp) = self.cp.as_mut() {
            if let Some(p) = progress {
                p.set(2, &format!("code predictor ({:?})", cp.device()));
            }
            cp.warmup(max_frames)?;
        }
        if let Some(fused) = self.fused.as_mut() {
            if let Some(p) = progress {
                p.set(2, &format!("codec-frame fused ({:?})", self.device));
            }
            fused.warmup(&self.talker, horizon)?;
        }
        self.pipeline_horizon = horizon;
        self.decode_pipeline_ready = crate::synth_opts::megakernel_fast_path() && !talker_eager;
        Ok(())
    }

    /// Drop cached prefill hidden/KV after compile-only warmup (synthetic embeds).
    pub fn finish_compile_warmup(&mut self) {
        self.invalidate_warmup_hidden();
    }

    /// Clear warmup prefill hidden so the next utterance always runs a real prefill.
    pub fn invalidate_warmup_hidden(&mut self) {
        self.prefill_hidden = None;
        self.talker.reset_kv();
        self.decode_pipeline_ready = false;
        self.pipeline_horizon = 0;
    }

    /// Drop cached prefill hidden before a stepwise AR session so prefill
    /// replays into KV instead of reusing last_hidden only.
    pub(crate) fn clear_prefill_cache_for_stepwise(&mut self) {
        self.prefill_hidden = None;
    }

    /// Greedy codec autoregression (talker + CP). Returns frames and stage timings in seconds.
    /// Like [`Self::synthesize_codec_ar`] but invokes `on_frame` once for each
    /// codec frame as it's emitted (16 codec groups per call). Use this to drive
    /// progress UIs or to start downstream work (e.g. partial decoding,
    /// network streaming) before the full AR completes.
    pub fn synthesize_codec_ar_streaming<F>(
        &mut self,
        prefill_embeds: ArrayView2<f32>,
        talker_cfg: &TalkerConfig,
        max_steps: usize,
        min_frames: usize,
        rep_penalty: f32,
        tts_pad_embed: &[f32],
        progress: Option<&Progress>,
        mut on_frame: F,
    ) -> Result<(Vec<Vec<u32>>, CodecArTimings)>
    where
        F: FnMut(usize, &[u32]),
    {
        self.synthesize_codec_ar_inner(
            prefill_embeds,
            talker_cfg,
            max_steps,
            min_frames,
            rep_penalty,
            tts_pad_embed,
            progress,
            Some(&mut on_frame),
        )
    }

    pub fn synthesize_codec_ar(
        &mut self,
        prefill_embeds: ArrayView2<f32>,
        talker_cfg: &TalkerConfig,
        max_steps: usize,
        min_frames: usize,
        rep_penalty: f32,
        tts_pad_embed: &[f32],
        progress: Option<&Progress>,
    ) -> Result<(Vec<Vec<u32>>, CodecArTimings)> {
        self.synthesize_codec_ar_inner(
            prefill_embeds,
            talker_cfg,
            max_steps,
            min_frames,
            rep_penalty,
            tts_pad_embed,
            progress,
            None::<&mut dyn FnMut(usize, &[u32])>,
        )
    }

    /// Begin a stepwise codec-AR session. Use [`Self::codec_ar_step`] to drive
    /// it one frame at a time — this lets the caller interleave the AR loop
    /// with downstream work (partial decode, network send, audio device write)
    /// instead of waiting for the whole utterance.
    pub fn begin_codec_ar(
        &mut self,
        prefill_embeds: ArrayView2<f32>,
        talker_cfg: &TalkerConfig,
        max_steps: usize,
    ) -> Result<CodecArState> {
        let horizon = prefill_embeds.nrows().saturating_add(max_steps);
        let t_prefill = Instant::now();
        self.clear_prefill_cache_for_stepwise();
        self.talker_prefill_core(prefill_embeds)?;
        if self.talker.is_eager() {
            self.talker.warm_eager_decode_rope()?;
        }
        let prefill_secs = t_prefill.elapsed().as_secs_f64();
        self.talker_prepare_decode_pipeline(horizon)?;
        let hidden_dim = talker_cfg.hidden_size;
        let mut scratch = CodecFrameScratch::new(hidden_dim, talker_cfg.vocab_size);
        scratch
            .hidden
            .copy_from_slice(self.talker_hidden_row().as_slice().unwrap());
        Ok(CodecArState {
            scratch,
            past_g0: Vec::new(),
            codec_frames: Vec::new(),
            step: 0,
            max_steps,
            prefill_secs,
            talker_secs: 0.0,
            cp_secs: 0.0,
            done: false,
        })
    }

    /// Advance the AR session by one step.
    ///
    /// Returns:
    ///   - `Some(idx)` — emitted a new frame; `state.codec_frames[idx]` is it
    ///   - `None`      — no frame this step (EOS skip or terminal Done)
    ///
    /// Check `state.is_done()` to know when to stop calling.
    pub fn codec_ar_step(
        &mut self,
        state: &mut CodecArState,
        talker_cfg: &TalkerConfig,
        min_frames: usize,
        rep_penalty: f32,
        tts_pad_embed: &[f32],
    ) -> Result<Option<usize>> {
        if state.done || state.step >= state.max_steps {
            state.done = true;
            return Ok(None);
        }
        let mut frame_timings = CodecFrameTimings::default();
        let outcome = codec_frame_step_dispatch(
            &mut self.talker,
            self.cp.as_mut(),
            self.fused.as_mut(),
            &mut state.scratch,
            talker_cfg,
            tts_pad_embed,
            rep_penalty,
            &mut state.past_g0,
            state.codec_frames.len(),
            min_frames,
            &mut frame_timings,
        )?;
        state.step += 1;
        state.talker_secs += (frame_timings.talk_logits_ms + frame_timings.talk_decode_ms) / 1000.0;
        state.cp_secs += frame_timings.cp_ms / 1000.0;
        match outcome {
            CodecFrameStep::Done => {
                state.done = true;
                Ok(None)
            }
            CodecFrameStep::EosSkip => Ok(None),
            CodecFrameStep::Frame(groups) => {
                let idx = state.codec_frames.len();
                state.codec_frames.push(groups);
                Ok(Some(idx))
            }
        }
    }

    fn synthesize_codec_ar_inner(
        &mut self,
        prefill_embeds: ArrayView2<f32>,
        talker_cfg: &TalkerConfig,
        max_steps: usize,
        min_frames: usize,
        rep_penalty: f32,
        tts_pad_embed: &[f32],
        progress: Option<&Progress>,
        mut on_frame: FrameCallback<'_>,
    ) -> Result<(Vec<Vec<u32>>, CodecArTimings)> {
        let horizon = prefill_embeds.nrows().saturating_add(max_steps);
        let t_prefill = Instant::now();
        self.talker_prefill_core(prefill_embeds)?;
        if self.talker.is_eager() {
            self.talker.warm_eager_decode_rope()?;
        }
        let prefill_secs = t_prefill.elapsed().as_secs_f64();
        self.talker_prepare_decode_pipeline(horizon)?;
        let hidden_dim = talker_cfg.hidden_size;
        let mut codec_frames = Vec::new();
        let mut past_g0: Vec<u32> = Vec::new();
        let mut scratch = CodecFrameScratch::new(hidden_dim, talker_cfg.vocab_size);
        scratch
            .hidden
            .copy_from_slice(self.talker_hidden_row().as_slice().unwrap());
        let mut talker_secs = 0f64;
        let mut cp_secs = 0f64;
        let step_timing = crate::synth_opts::step_timing_enabled();

        for step in 0..max_steps {
            if let Some(p) = progress {
                p.set(
                    step,
                    &format!(
                        "frame {} (talker {:?} + CP {})",
                        codec_frames.len(),
                        self.device,
                        self.cp_backend_label()
                    ),
                );
            }
            let mut frame_timings = CodecFrameTimings::default();
            match codec_frame_step_dispatch(
                &mut self.talker,
                self.cp.as_mut(),
                self.fused.as_mut(),
                &mut scratch,
                talker_cfg,
                tts_pad_embed,
                rep_penalty,
                &mut past_g0,
                codec_frames.len(),
                min_frames,
                &mut frame_timings,
            )? {
                CodecFrameStep::Done => break,
                CodecFrameStep::EosSkip => continue,
                CodecFrameStep::Frame(groups) => {
                    let g0 = groups[0];
                    talker_secs +=
                        (frame_timings.talk_logits_ms + frame_timings.talk_decode_ms) / 1000.0;
                    cp_secs += frame_timings.cp_ms / 1000.0;
                    if std::env::var("RLX_QWEN3_TTS_SYNTH_DEBUG").ok().as_deref() == Some("1") {
                        eprintln!("step {step} g0={g0}");
                    }
                    let idx = codec_frames.len();
                    codec_frames.push(groups);
                    if let Some(cb) = on_frame.as_deref_mut() {
                        cb(idx, &codec_frames[idx]);
                    }
                    if step_timing {
                        eprintln!(
                            "[qwen3-tts step] frame={} g0={g0} past={} talker_logits={:.1}ms cp={:.1}ms talker_decode={:.1}ms ({})",
                            codec_frames.len(),
                            self.talker_past_len(),
                            frame_timings.talk_logits_ms,
                            frame_timings.cp_ms,
                            frame_timings.talk_decode_ms,
                            self.cp_backend_label(),
                        );
                    }
                }
            }
        }
        Ok((
            codec_frames,
            CodecArTimings {
                talker_secs,
                cp_secs,
                prefill_secs,
            },
        ))
    }

    pub fn sum_codec_groups_into(&self, groups: &[u32], out: &mut [f32]) -> Result<()> {
        let cp = self
            .cp
            .as_ref()
            .context("sum_codec_groups_into requires legacy CP engine")?;
        cp.sum_codec_groups_into(groups, out)
    }

    /// Sum codec groups and add TTS pad embed in one pass.
    pub fn sum_codec_groups_with_pad_into(
        &self,
        groups: &[u32],
        pad: &[f32],
        out: &mut [f32],
    ) -> Result<()> {
        self.sum_codec_groups_into(groups, out)?;
        for (j, v) in pad.iter().enumerate() {
            out[j] += *v;
        }
        Ok(())
    }

    pub fn talker_prefill(&mut self, embeds: ArrayView2<f32>) -> Result<()> {
        let horizon = embeds.nrows().saturating_add(64);
        self.talker_prefill_with_horizon(embeds, horizon)
    }

    /// Compile talker prefill graph for `embeds.nrows()` (cache miss only).
    pub fn ensure_talk_prefill_compiled(&mut self, embeds: ArrayView2<f32>) -> Result<()> {
        self.talker.ensure_prefill_compiled(embeds.nrows())
    }

    pub fn talker_prefill_with_horizon(
        &mut self,
        embeds: ArrayView2<f32>,
        horizon: usize,
    ) -> Result<()> {
        self.talker_prefill_core(embeds)?;
        self.talker_prepare_decode_pipeline(horizon)
    }

    /// Run talker prefill (or reuse warmup hidden when nrow matches).
    pub fn talker_prefill_core(&mut self, embeds: ArrayView2<f32>) -> Result<()> {
        // `warmup()` may stash `prefill_hidden` for bucket compile only. After
        // `invalidate_warmup_hidden()` the KV cache is empty — reusing that
        // hidden without a full prefill corrupts codec AR (garbled / truncated
        // speech on the 2nd+ utterance in a session).
        let _ = self.prefill_hidden.take();
        self.talker.reset_kv();
        let hidden = self.talker.prefill(embeds)?;
        let rows = hidden.nrows();
        self.talker
            .set_last_hidden(hidden.row(rows.saturating_sub(1)))?;
        Ok(())
    }

    /// GPU KV bind + incremental bucket dry-runs after prefill (not timed as prefill).
    pub fn talker_prepare_decode_pipeline(&mut self, horizon: usize) -> Result<()> {
        if !crate::synth_opts::megakernel_fast_path() || self.talker.is_eager() {
            return Ok(());
        }
        metal_compile_guard(self.device, || {
            if self.decode_pipeline_ready {
                self.talker.preinstall_gpu_kv_current()?;
                if horizon > self.pipeline_horizon
                    && crate::synth_opts::talk_bucket_execution_warmup(horizon)
                {
                    self.talker
                        .warmup_bucket_executions_from(self.pipeline_horizon, horizon)?;
                    self.pipeline_horizon = horizon;
                }
                Ok(())
            } else {
                self.talker.warmup_bucket_executions(horizon)
            }
        })
    }

    /// Last talker hidden row after prefill or decode.
    pub fn talker_hidden_row(&self) -> ArrayView1<'_, f32> {
        self.talker.last_hidden_view()
    }

    pub fn talker_past_len(&self) -> usize {
        self.talker.past_len()
    }

    /// KV decode; updates talker hidden row in place (no `Array2` alloc).
    pub fn talker_decode_into(&mut self, embed: &[f32], hidden_out: &mut [f32]) -> Result<()> {
        self.talker
            .decode_hidden_into(ArrayView1::from(embed), hidden_out)
    }

    pub fn predict_codec_groups_slice(
        &mut self,
        talker_hidden: &[f32],
        group0: u32,
    ) -> Result<Vec<u32>> {
        let cp = self
            .cp
            .as_mut()
            .context("predict_codec_groups_slice requires legacy CP engine")?;
        cp.predict_groups_slice(talker_hidden, group0)
    }

    pub fn codec_head(&self) -> ArrayView2<'_, f32> {
        self.talker.codec_head()
    }

    pub fn cp_backend_label(&self) -> String {
        if self.fused.is_some() {
            "CPU eager megakernel".into()
        } else if let Some(cp) = &self.cp {
            cp.cp_backend_label()
        } else {
            "n/a".into()
        }
    }

    /// Crate-internal accessor for the speculative-decoding path
    /// (see `megakernel_speculative.rs`). Not part of the public API.
    #[cfg(feature = "speculative-decode")]
    pub(crate) fn talker_engine_ref(&self) -> &TalkerEngine {
        &self.talker
    }

    #[cfg(feature = "speculative-decode")]
    pub(crate) fn talker_engine_mut(&mut self) -> &mut TalkerEngine {
        &mut self.talker
    }

    #[cfg(feature = "speculative-decode")]
    pub(crate) fn cp_engine_mut(&mut self) -> Option<&mut CodePredictorEngine> {
        self.cp.as_mut()
    }
}

/// Stage timings from [`Qwen3TtsMegakernel::synthesize_codec_ar`].
#[derive(Debug, Clone, Copy, Default)]
pub struct CodecArTimings {
    pub talker_secs: f64,
    pub cp_secs: f64,
    pub prefill_secs: f64,
}

/// Stepwise codec-AR session — held by the caller across calls to
/// [`Qwen3TtsMegakernel::codec_ar_step`]. The accumulator fields are exposed
/// directly; treat `codec_frames` as the source of truth for what's been
/// emitted so far.
pub struct CodecArState {
    pub(crate) scratch: CodecFrameScratch,
    pub(crate) past_g0: Vec<u32>,
    /// Frames emitted so far (in order). Index into this for partial decoding.
    pub codec_frames: Vec<Vec<u32>>,
    pub(crate) step: usize,
    pub(crate) max_steps: usize,
    pub prefill_secs: f64,
    pub talker_secs: f64,
    pub cp_secs: f64,
    done: bool,
}

impl CodecArState {
    /// True once the AR has hit its terminal state (EOS or max_steps).
    pub fn is_done(&self) -> bool {
        self.done || self.step >= self.max_steps
    }

    /// How many frames have been emitted so far.
    pub fn frames_emitted(&self) -> usize {
        self.codec_frames.len()
    }

    /// Finalize and return the produced frames + accumulated timings.
    pub fn finish(self) -> (Vec<Vec<u32>>, CodecArTimings) {
        (
            self.codec_frames,
            CodecArTimings {
                talker_secs: self.talker_secs,
                cp_secs: self.cp_secs,
                prefill_secs: self.prefill_secs,
            },
        )
    }
}
