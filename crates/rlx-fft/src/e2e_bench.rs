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

//! End-to-end validation bench — mel, welch, q8, denoise vs references.

use crate::butterfly::butterfly_forward_real_batch;
use crate::config::FftLearnConfig;
use crate::config::TransformDir;
use crate::distill_compile::{CompiledDistilledMel, compile_distilled_mel};
use crate::distill_fused::pick_fused_deploy_device;
use crate::distill_model::DistilledFftModel;
use crate::learned_compile::{
    CompiledLearnedMel, compile_learned_mel, default_hard_threshold, window_batch as compile_window,
};
use crate::learned_model::{FastLearnedFftModel, pipeline_max_err, ref_spectrum_batch, ref_welch};
use crate::mel::{hann_window, log_mel_from_spectrum_batch, log_mel_from_windowed_batch};
use crate::peak::{DEFAULT_PEAK_K, WelchPeakParams, WelchPeaksScratch, welch_peaks_rustfft};
use crate::pruned::DEFAULT_GATE_THRESHOLD;
use crate::q8::Q8Twiddles;
use crate::reference::fft_real_batch;
use crate::rlx_fft::{compile_rlx_fft, rlx_fft_forward};
use crate::train::random_batch;
use crate::twiddle::exact_twiddles;
use crate::welch::WelchParams;
use crate::welch_peaks_compile::{
    CompiledLearnedWelchPeaks, CompiledRlxWelchPeaks, compile_learned_welch_peaks,
    compile_rlx_welch_peaks,
};
use crate::welch_peaks_picker::AutoWelchPeaks;
use anyhow::{Context, Result, ensure};
use rand::prelude::*;
use rlx_runtime::{CompiledGraph, Device};
use serde::{Deserialize, Serialize};
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum E2ePipeline {
    Mel,
    Welch,
    WelchPeaks,
    WelchPeaksUltra,
    Q8Spectrum,
    Denoise,
}

impl E2ePipeline {
    pub fn all() -> &'static [Self] {
        &[
            Self::Mel,
            Self::Welch,
            Self::WelchPeaks,
            Self::WelchPeaksUltra,
            Self::Q8Spectrum,
            Self::Denoise,
        ]
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Mel => "mel",
            Self::Welch => "welch",
            Self::WelchPeaks => "welch_peaks",
            Self::WelchPeaksUltra => "welch_peaks_ultra",
            Self::Q8Spectrum => "q8",
            Self::Denoise => "denoise",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum E2eBackend {
    RustfftRef,
    RlxOpFft,
    ButterflyEager,
    LearnedModel,
    LearnedQ8,
    LearnedHard,
    LearnedCompiled,
    LearnedDistilled,
    LearnedDistilledTernary,
    /// IO-aware [`AutoWelchPeaks`] (adaptive fused / block RLX / rustfft).
    WelchPeaksAuto,
}

impl E2eBackend {
    pub fn label(self) -> &'static str {
        match self {
            Self::RustfftRef => "rustfft_ref",
            Self::RlxOpFft => "rlx_op_fft",
            Self::ButterflyEager => "butterfly_eager",
            Self::LearnedModel => "learned_model",
            Self::LearnedQ8 => "learned_q8",
            Self::LearnedHard => "learned_hard",
            Self::LearnedCompiled => "learned_compiled",
            Self::LearnedDistilled => "learned_distilled",
            Self::LearnedDistilledTernary => "learned_distilled_ternary",
            Self::WelchPeaksAuto => "welch_peaks_auto",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct E2eBenchRow {
    pub pipeline: String,
    pub backend: String,
    pub n_fft: usize,
    pub batch: usize,
    pub device: String,
    pub iters: usize,
    pub ms: f64,
    pub max_err: f32,
    pub mean_gate: Option<f32>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct E2eBatchTrainMeta {
    pub batch: usize,
    pub teacher: Option<crate::train_e2e::E2eTrainReport>,
    pub distill: Option<crate::train_distill::DistillTrainReport>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct E2eBenchMeta {
    pub n_fft: usize,
    pub seed: u64,
    pub devices: Vec<String>,
    pub batches: Vec<usize>,
    #[serde(default = "default_peak_k")]
    pub peak_k: usize,
    pub train_steps: Option<usize>,
    pub distill_steps: Option<usize>,
    /// Legacy single-batch summary (largest batch when per-batch training ran).
    pub teacher: Option<crate::train_e2e::E2eTrainReport>,
    pub distill: Option<crate::train_distill::DistillTrainReport>,
    #[serde(default)]
    pub per_batch: Vec<E2eBatchTrainMeta>,
}

fn default_peak_k() -> usize {
    DEFAULT_PEAK_K
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct E2eBenchReport {
    pub n_mels: usize,
    pub iters: usize,
    pub elapsed_ms: f64,
    pub rows: Vec<E2eBenchRow>,
    #[serde(default)]
    pub meta: E2eBenchMeta,
}

pub struct E2eBenchInputs<'a> {
    pub n_fft: usize,
    pub batch: usize,
    pub n_mels: usize,
    pub iters: usize,
    pub device: Device,
    pub seed: u64,
    pub model: Option<&'a FastLearnedFftModel>,
    pub distilled: Option<&'a DistilledFftModel>,
    pub distilled_ternary: Option<&'a crate::distill_ternary_model::DistilledTernaryFftModel>,
    pub with_learned_hard: bool,
    pub with_learned_compiled: bool,
    pub with_learned_distilled: bool,
    pub with_learned_distilled_ternary: bool,
    /// Include slow eager learned backends (default false — use compiled on GPU).
    pub with_eager_learned: bool,
    /// Top-K Welch peaks for `WelchPeaks` / `WelchPeaksUltra` pipelines (default 16).
    pub peak_k: usize,
}

pub fn run_e2e_bench(inputs: &E2eBenchInputs<'_>) -> Result<E2eBenchReport> {
    ensure!(inputs.iters >= 1);
    let peak_k = inputs.peak_k.max(1);
    let started = Instant::now();
    let cfg = FftLearnConfig::new(inputs.n_fft, inputs.batch)?;
    let n_mels = inputs.n_mels;
    let sr = 16_000.0f32;
    let welch_params = WelchParams::for_n_fft(inputs.n_fft);
    let peak_fast = WelchPeakParams::fast_for_n_fft(inputs.n_fft, peak_k);
    let peak_ultra = WelchPeakParams::ultra_fast_for_n_fft(inputs.n_fft, peak_k);
    let welch_frame = welch_params.frame_len();

    let mut rng = rand::rngs::StdRng::seed_from_u64(inputs.seed);
    let signal = random_batch(&mut rng, inputs.batch, inputs.n_fft);
    let welch_signal = random_batch(&mut rng, inputs.batch, welch_frame);
    let windowed = window_batch(&signal, inputs.batch, inputs.n_fft);

    let tw = exact_twiddles(&cfg);
    let _gates = crate::pruned::init_gates(inputs.n_fft);
    let _q8 = Q8Twiddles::from_f32(&tw);

    let mut rlx_exec_mel: Option<CompiledGraph> = None;
    let mut rlx_exec_welch: Option<CompiledGraph> = None;
    let mut compiled_learned: Option<CompiledLearnedMel> = None;
    let mut compiled_distilled: Option<crate::distill_compile::CompiledDistilledMel> = None;
    let mut compiled_distilled_ternary: Option<
        crate::distill_ternary_compile::CompiledDistilledTernaryMel,
    > = None;
    let mut compiled_rlx_peaks: Option<CompiledRlxWelchPeaks> = None;
    let mut compiled_learned_peaks: Option<CompiledLearnedWelchPeaks> = None;
    let mut auto_welch_peaks: Option<AutoWelchPeaks> = None;
    let mut peak_scratch = WelchPeaksScratch::new(inputs.batch, peak_fast.n_bins());
    let mut peak_fast_buf = Vec::with_capacity(inputs.batch * peak_fast.frame_len());
    let device_name = format!("{:?}", inputs.device).to_lowercase();

    let mut rows = Vec::new();

    for &pipeline in E2ePipeline::all() {
        if pipeline == E2ePipeline::Welch && inputs.batch >= 32 {
            continue;
        }
        let peak_params = match pipeline {
            E2ePipeline::WelchPeaksUltra => peak_ultra,
            E2ePipeline::WelchPeaks => peak_fast,
            _ => peak_fast,
        };
        let ref_out = reference_output(
            pipeline,
            &signal,
            &welch_signal,
            &windowed,
            inputs,
            welch_params,
            peak_params,
            n_mels,
            sr,
        )?;

        for backend in [
            E2eBackend::RustfftRef,
            E2eBackend::RlxOpFft,
            E2eBackend::ButterflyEager,
            E2eBackend::LearnedModel,
            E2eBackend::LearnedQ8,
            E2eBackend::LearnedHard,
            E2eBackend::LearnedCompiled,
            E2eBackend::LearnedDistilled,
            E2eBackend::LearnedDistilledTernary,
            E2eBackend::WelchPeaksAuto,
        ] {
            // Auto picker targets the 2-segment fast layout (not 1-seg ultra reference).
            if backend == E2eBackend::WelchPeaksAuto && pipeline != E2ePipeline::WelchPeaks {
                continue;
            }
            if matches!(
                backend,
                E2eBackend::LearnedModel
                    | E2eBackend::LearnedQ8
                    | E2eBackend::LearnedHard
                    | E2eBackend::LearnedCompiled
            ) && inputs.model.is_none()
            {
                continue;
            }
            if backend == E2eBackend::LearnedDistilled && inputs.distilled.is_none() {
                continue;
            }
            if backend == E2eBackend::LearnedDistilledTernary && inputs.distilled_ternary.is_none()
            {
                continue;
            }
            if backend == E2eBackend::LearnedQ8 {
                let m = inputs.model.expect("model");
                if !m.use_q8 {
                    continue;
                }
            }
            if backend == E2eBackend::LearnedHard && !inputs.with_learned_hard {
                continue;
            }
            if backend == E2eBackend::LearnedCompiled && !inputs.with_learned_compiled {
                continue;
            }
            if backend == E2eBackend::LearnedDistilled && !inputs.with_learned_distilled {
                continue;
            }
            if backend == E2eBackend::LearnedDistilledTernary
                && !inputs.with_learned_distilled_ternary
            {
                continue;
            }
            if matches!(
                backend,
                E2eBackend::LearnedModel | E2eBackend::LearnedQ8 | E2eBackend::LearnedHard
            ) && !inputs.with_eager_learned
            {
                continue;
            }
            if backend == E2eBackend::ButterflyEager {
                continue;
            }
            if inputs.device != Device::Cpu
                && matches!(
                    backend,
                    E2eBackend::ButterflyEager
                        | E2eBackend::LearnedModel
                        | E2eBackend::LearnedQ8
                        | E2eBackend::LearnedHard
                )
            {
                continue;
            }
            if backend == E2eBackend::RlxOpFft && inputs.device == Device::Cpu {
                // still valid on CPU
            }

            let (ms, max_err, mean_gate) = bench_backend(
                pipeline,
                backend,
                &signal,
                &welch_signal,
                &windowed,
                &ref_out,
                inputs,
                &tw,
                welch_params,
                peak_params,
                n_mels,
                sr,
                &mut rlx_exec_mel,
                &mut rlx_exec_welch,
                &mut compiled_learned,
                &mut compiled_distilled,
                &mut compiled_distilled_ternary,
                &mut compiled_rlx_peaks,
                &mut compiled_learned_peaks,
                &mut peak_scratch,
                &mut peak_fast_buf,
                &mut auto_welch_peaks,
            )?;

            rows.push(E2eBenchRow {
                pipeline: pipeline.label().to_string(),
                backend: backend.label().to_string(),
                n_fft: inputs.n_fft,
                batch: inputs.batch,
                device: device_name.clone(),
                iters: inputs.iters,
                ms,
                max_err,
                mean_gate,
            });
        }
    }

    Ok(E2eBenchReport {
        n_mels,
        iters: inputs.iters,
        elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
        rows,
        meta: E2eBenchMeta::default(),
    })
}

fn reference_output(
    pipeline: E2ePipeline,
    signal: &[f32],
    welch_signal: &[f32],
    windowed: &[f32],
    inputs: &E2eBenchInputs<'_>,
    welch_params: WelchParams,
    peak_params: WelchPeakParams,
    n_mels: usize,
    sr: f32,
) -> Result<Vec<f32>> {
    match pipeline {
        E2ePipeline::Mel => {
            log_mel_from_windowed_batch(windowed, inputs.batch, inputs.n_fft, n_mels, sr)
        }
        E2ePipeline::Welch => ref_welch(welch_signal, inputs.batch, welch_params),
        E2ePipeline::WelchPeaks | E2ePipeline::WelchPeaksUltra => welch_peaks_rustfft(
            welch_signal,
            inputs.batch,
            WelchPeakParams::reference_for_n_fft(inputs.n_fft, peak_params.k),
        ),
        E2ePipeline::Q8Spectrum => {
            let tw = Q8Twiddles::from_f32(&exact_twiddles(&FftLearnConfig::new(
                inputs.n_fft,
                inputs.batch,
            )?));
            tw.forward_real_batch(signal, inputs.batch, inputs.n_fft)
        }
        E2ePipeline::Denoise => ref_spectrum_batch(signal, inputs.batch, inputs.n_fft),
    }
}

#[allow(clippy::too_many_arguments)]
fn bench_backend(
    pipeline: E2ePipeline,
    backend: E2eBackend,
    signal: &[f32],
    welch_signal: &[f32],
    windowed: &[f32],
    ref_out: &[f32],
    inputs: &E2eBenchInputs<'_>,
    tw: &[f32],
    welch_params: WelchParams,
    peak_params: WelchPeakParams,
    n_mels: usize,
    sr: f32,
    rlx_exec_mel: &mut Option<CompiledGraph>,
    rlx_exec_welch: &mut Option<CompiledGraph>,
    compiled_learned: &mut Option<CompiledLearnedMel>,
    compiled_distilled: &mut Option<crate::distill_compile::CompiledDistilledMel>,
    compiled_distilled_ternary: &mut Option<
        crate::distill_ternary_compile::CompiledDistilledTernaryMel,
    >,
    compiled_rlx_peaks: &mut Option<CompiledRlxWelchPeaks>,
    compiled_learned_peaks: &mut Option<CompiledLearnedWelchPeaks>,
    peak_scratch: &mut WelchPeaksScratch,
    peak_fast_buf: &mut Vec<f32>,
    auto_welch_peaks: &mut Option<AutoWelchPeaks>,
) -> Result<(f64, f32, Option<f32>)> {
    let welch_frame = welch_params.frame_len();
    let mut run = || -> Result<Vec<f32>> {
        match (pipeline, backend) {
            (E2ePipeline::Mel, E2eBackend::RustfftRef) => {
                log_mel_from_windowed_batch(windowed, inputs.batch, inputs.n_fft, n_mels, sr)
            }
            (E2ePipeline::Mel, E2eBackend::RlxOpFft) => {
                rlx_mel(rlx_exec_mel, windowed, inputs, n_mels, sr)
            }
            (E2ePipeline::Mel, E2eBackend::ButterflyEager) => {
                mel_from_butterfly(windowed, tw, inputs.batch, inputs.n_fft, n_mels, sr)
            }
            (E2ePipeline::Mel, E2eBackend::LearnedModel) => inputs
                .model
                .expect("model")
                .log_mel_batch(signal, inputs.batch),
            (E2ePipeline::Mel, E2eBackend::LearnedQ8) => inputs
                .model
                .expect("model")
                .log_mel_batch(signal, inputs.batch),
            (E2ePipeline::Mel, E2eBackend::LearnedHard) => {
                hard_model(inputs.model.expect("model")).log_mel_batch(signal, inputs.batch)
            }
            (E2ePipeline::Mel, E2eBackend::LearnedCompiled) => {
                let win = compile_window(signal, inputs.batch, inputs.n_fft);
                compiled_learned_mel(compiled_learned, inputs, inputs.model.expect("model"), &win)
            }
            (E2ePipeline::Mel, E2eBackend::LearnedDistilled) => compiled_distilled_mel(
                compiled_distilled,
                inputs,
                inputs.distilled.expect("distilled"),
                signal,
            ),
            (E2ePipeline::Mel, E2eBackend::LearnedDistilledTernary) => {
                compiled_distilled_ternary_mel(
                    compiled_distilled_ternary,
                    inputs,
                    inputs.distilled_ternary.expect("distilled_ternary"),
                    signal,
                )
            }

            (E2ePipeline::Welch, E2eBackend::RustfftRef) => {
                ref_welch(welch_signal, inputs.batch, welch_params)
            }
            (E2ePipeline::Welch, E2eBackend::RlxOpFft) => {
                rlx_welch(rlx_exec_welch, welch_signal, inputs, welch_params)
            }
            (E2ePipeline::Welch, E2eBackend::ButterflyEager) => {
                crate::welch::welch_butterfly(welch_signal, tw, inputs.batch, welch_params)
            }
            (E2ePipeline::Welch, E2eBackend::LearnedModel | E2eBackend::LearnedQ8) => inputs
                .model
                .expect("model")
                .welch_psd_batch(welch_signal, inputs.batch, welch_params),
            (E2ePipeline::Welch, E2eBackend::LearnedHard) => hard_model(
                inputs.model.expect("model"),
            )
            .welch_psd_batch(welch_signal, inputs.batch, welch_params),
            (E2ePipeline::Welch, E2eBackend::LearnedCompiled) => inputs
                .model
                .expect("model")
                .welch_psd_batch(welch_signal, inputs.batch, welch_params),
            (E2ePipeline::Welch, E2eBackend::LearnedDistilled) => inputs
                .distilled
                .expect("distilled")
                .welch_psd_batch(welch_signal, inputs.batch, welch_params),
            (E2ePipeline::Welch, E2eBackend::LearnedDistilledTernary) => inputs
                .distilled_ternary
                .expect("distilled_ternary")
                .welch_psd_batch(welch_signal, inputs.batch, welch_params),

            (E2ePipeline::WelchPeaks | E2ePipeline::WelchPeaksUltra, backend)
                if matches!(
                    backend,
                    E2eBackend::RustfftRef
                        | E2eBackend::RlxOpFft
                        | E2eBackend::ButterflyEager
                        | E2eBackend::LearnedModel
                        | E2eBackend::LearnedQ8
                        | E2eBackend::LearnedHard
                        | E2eBackend::LearnedCompiled
                        | E2eBackend::LearnedDistilled
                        | E2eBackend::LearnedDistilledTernary
                        | E2eBackend::WelchPeaksAuto
                ) =>
            {
                welch_peaks_for_backend(
                    backend,
                    welch_signal,
                    inputs,
                    peak_params,
                    welch_frame,
                    tw,
                    compiled_rlx_peaks,
                    compiled_learned_peaks,
                    peak_scratch,
                    peak_fast_buf,
                    auto_welch_peaks,
                )
            }
            (E2ePipeline::WelchPeaks | E2ePipeline::WelchPeaksUltra, _) => {
                anyhow::bail!(
                    "welch peaks pipeline skipped for backend {}",
                    backend.label()
                )
            }

            (E2ePipeline::Q8Spectrum, E2eBackend::RustfftRef) => {
                fft_real_batch(signal, inputs.batch, inputs.n_fft)
            }
            (E2ePipeline::Q8Spectrum, E2eBackend::RlxOpFft) => {
                rlx_spectrum(rlx_exec_mel, signal, inputs)
            }
            (E2ePipeline::Q8Spectrum, E2eBackend::ButterflyEager) => {
                butterfly_forward_real_batch(signal, tw, inputs.batch, inputs.n_fft)
            }
            (E2ePipeline::Q8Spectrum, E2eBackend::LearnedModel) => {
                let m = inputs.model.expect("model");
                m.spectrum_batch_raw(signal, inputs.batch)
            }
            (E2ePipeline::Q8Spectrum, E2eBackend::LearnedQ8) => {
                let m = inputs.model.expect("model");
                ensure!(m.use_q8);
                m.spectrum_batch_raw(signal, inputs.batch)
            }
            (E2ePipeline::Q8Spectrum, E2eBackend::LearnedHard) => {
                hard_model(inputs.model.expect("model")).spectrum_batch_raw(signal, inputs.batch)
            }
            (E2ePipeline::Q8Spectrum, E2eBackend::LearnedCompiled) => {
                ensure_compiled_learned(compiled_learned, inputs, inputs.model.expect("model"))?
                    .spectrum_batch(signal)
            }
            (E2ePipeline::Q8Spectrum, E2eBackend::LearnedDistilled) => inputs
                .distilled
                .expect("distilled")
                .spectrum_batch_raw(signal, inputs.batch),
            (E2ePipeline::Q8Spectrum, E2eBackend::LearnedDistilledTernary) => inputs
                .distilled_ternary
                .expect("distilled_ternary")
                .spectrum_batch_raw(signal, inputs.batch),

            (E2ePipeline::Denoise, E2eBackend::RustfftRef) => {
                ref_spectrum_batch(signal, inputs.batch, inputs.n_fft)
            }
            (E2ePipeline::Denoise, E2eBackend::RlxOpFft) => {
                rlx_spectrum(rlx_exec_mel, signal, inputs)
            }
            (E2ePipeline::Denoise, E2eBackend::ButterflyEager) => {
                butterfly_forward_real_batch(signal, tw, inputs.batch, inputs.n_fft)
            }
            (E2ePipeline::Denoise, E2eBackend::LearnedModel | E2eBackend::LearnedQ8) => inputs
                .model
                .expect("model")
                .spectrum_batch(signal, inputs.batch),
            (E2ePipeline::Denoise, E2eBackend::LearnedHard) => {
                hard_model(inputs.model.expect("model")).spectrum_batch(signal, inputs.batch)
            }
            (E2ePipeline::Denoise, E2eBackend::LearnedCompiled) => {
                ensure_compiled_learned(compiled_learned, inputs, inputs.model.expect("model"))?
                    .spectrum_batch(signal)
            }
            (E2ePipeline::Denoise, E2eBackend::LearnedDistilled) => inputs
                .distilled
                .expect("distilled")
                .spectrum_batch_raw(signal, inputs.batch),
            (E2ePipeline::Denoise, E2eBackend::LearnedDistilledTernary) => inputs
                .distilled_ternary
                .expect("distilled_ternary")
                .spectrum_batch_raw(signal, inputs.batch),

            (_, E2eBackend::WelchPeaksAuto) => {
                anyhow::bail!("welch_peaks_auto only runs on welch peaks pipelines")
            }
        }
    };

    let pred = run()?;
    let mut max_err = if backend == E2eBackend::RustfftRef {
        0.0
    } else {
        pipeline_max_err(&pred, ref_out)
    };
    if pipeline == E2ePipeline::Mel && backend == E2eBackend::LearnedDistilledTernary {
        let model = inputs.distilled_ternary.expect("distilled_ternary");
        let ref_spec = fft_real_batch(windowed, inputs.batch, inputs.n_fft)?;
        let ref_mel = log_mel_from_spectrum_batch(
            &ref_spec,
            model.mel_filters(),
            inputs.batch,
            inputs.n_fft,
            n_mels,
        )?;
        max_err = pipeline_max_err(&pred, &ref_mel);
    }
    let mean_gate = match backend {
        E2eBackend::LearnedModel
        | E2eBackend::LearnedQ8
        | E2eBackend::LearnedHard
        | E2eBackend::LearnedCompiled => Some(inputs.model.expect("model").mean_gate()),
        E2eBackend::LearnedDistilledTernary => Some(
            inputs
                .distilled_ternary
                .expect("distilled_ternary")
                .compute_fraction(),
        ),
        _ => None,
    };

    let _ = run()?;
    let t0 = Instant::now();
    for _ in 0..inputs.iters {
        run()?;
    }
    let ms = t0.elapsed().as_secs_f64() * 1000.0 / inputs.iters as f64;
    Ok((ms, max_err, mean_gate))
}

fn truncate_peak_signal(
    peak_params: WelchPeakParams,
    welch_signal: &[f32],
    batch: usize,
    welch_frame: usize,
    buf: &mut Vec<f32>,
) -> Result<()> {
    peak_params
        .welch
        .truncate_batch_into(welch_signal, batch, welch_frame, buf)
}

#[allow(clippy::too_many_arguments)]
fn welch_peaks_for_backend(
    backend: E2eBackend,
    welch_signal: &[f32],
    inputs: &E2eBenchInputs<'_>,
    peak_params: WelchPeakParams,
    welch_frame: usize,
    tw: &[f32],
    compiled_rlx_peaks: &mut Option<CompiledRlxWelchPeaks>,
    compiled_learned_peaks: &mut Option<CompiledLearnedWelchPeaks>,
    peak_scratch: &mut WelchPeaksScratch,
    peak_fast_buf: &mut Vec<f32>,
    auto_welch_peaks: &mut Option<AutoWelchPeaks>,
) -> Result<Vec<f32>> {
    if compiled_rlx_peaks
        .as_ref()
        .is_some_and(|c| c.peak_params != peak_params)
    {
        *compiled_rlx_peaks = None;
    }
    if compiled_learned_peaks
        .as_ref()
        .is_some_and(|c| c.peak_params != peak_params)
    {
        *compiled_learned_peaks = None;
    }

    match backend {
        E2eBackend::RustfftRef => welch_peaks_rustfft(
            welch_signal,
            inputs.batch,
            WelchPeakParams::reference_for_n_fft(inputs.n_fft, peak_params.k),
        ),
        E2eBackend::RlxOpFft => {
            truncate_peak_signal(
                peak_params,
                welch_signal,
                inputs.batch,
                welch_frame,
                peak_fast_buf,
            )?;
            if compiled_rlx_peaks.is_none() {
                *compiled_rlx_peaks = Some(compile_rlx_welch_peaks(
                    inputs.batch,
                    peak_params,
                    inputs.device,
                )?);
            }
            compiled_rlx_peaks
                .as_mut()
                .expect("rlx peaks")
                .welch_peaks_batch(peak_fast_buf, peak_scratch)
        }
        E2eBackend::ButterflyEager => {
            truncate_peak_signal(
                peak_params,
                welch_signal,
                inputs.batch,
                welch_frame,
                peak_fast_buf,
            )?;
            let psd =
                crate::welch::welch_butterfly(peak_fast_buf, tw, inputs.batch, peak_params.welch)?;
            Ok(crate::peak::peaks_from_psd_batch(
                &psd,
                inputs.batch,
                peak_params.n_bins(),
                peak_params.k,
            ))
        }
        E2eBackend::LearnedModel | E2eBackend::LearnedQ8 => {
            truncate_peak_signal(
                peak_params,
                welch_signal,
                inputs.batch,
                welch_frame,
                peak_fast_buf,
            )?;
            inputs
                .model
                .expect("model")
                .welch_peaks_batch(peak_fast_buf, inputs.batch, peak_params)
        }
        E2eBackend::LearnedHard => {
            truncate_peak_signal(
                peak_params,
                welch_signal,
                inputs.batch,
                welch_frame,
                peak_fast_buf,
            )?;
            hard_model(inputs.model.expect("model")).welch_peaks_batch(
                peak_fast_buf,
                inputs.batch,
                peak_params,
            )
        }
        E2eBackend::LearnedCompiled => {
            truncate_peak_signal(
                peak_params,
                welch_signal,
                inputs.batch,
                welch_frame,
                peak_fast_buf,
            )?;
            let model = inputs.model.expect("model");
            if compiled_learned_peaks.is_none() {
                *compiled_learned_peaks = Some(compile_learned_welch_peaks(
                    &hard_model(model),
                    inputs.batch,
                    peak_params,
                    inputs.device,
                    default_hard_threshold(),
                )?);
            }
            compiled_learned_peaks
                .as_mut()
                .expect("learned peaks")
                .welch_peaks_batch(peak_fast_buf, peak_scratch)
        }
        E2eBackend::LearnedDistilled => {
            truncate_peak_signal(
                peak_params,
                welch_signal,
                inputs.batch,
                welch_frame,
                peak_fast_buf,
            )?;
            inputs.distilled.expect("distilled").welch_peaks_batch(
                peak_fast_buf,
                inputs.batch,
                peak_params,
            )
        }
        E2eBackend::LearnedDistilledTernary => {
            truncate_peak_signal(
                peak_params,
                welch_signal,
                inputs.batch,
                welch_frame,
                peak_fast_buf,
            )?;
            inputs
                .distilled_ternary
                .expect("distilled_ternary")
                .welch_peaks_batch(peak_fast_buf, inputs.batch, peak_params)
        }
        E2eBackend::WelchPeaksAuto => {
            if auto_welch_peaks
                .as_ref()
                .is_some_and(|p| p.peak_params() != peak_params)
            {
                *auto_welch_peaks = None;
            }
            if auto_welch_peaks.is_none() {
                *auto_welch_peaks = Some(AutoWelchPeaks::with_learned(
                    inputs.batch,
                    inputs.n_fft,
                    peak_params.k,
                    Some(&format!("{:?}", inputs.device).to_lowercase()),
                    inputs.model,
                )?);
            }
            truncate_peak_signal(
                peak_params,
                welch_signal,
                inputs.batch,
                welch_frame,
                peak_fast_buf,
            )?;
            auto_welch_peaks
                .as_mut()
                .expect("auto welch peaks")
                .welch_peaks_batch_fast(peak_fast_buf)
        }
    }
}

fn hard_model(model: &FastLearnedFftModel) -> FastLearnedFftModel {
    let mut m = model.clone();
    m.hard_gate_threshold = Some(DEFAULT_GATE_THRESHOLD);
    m
}

fn compiled_distilled_ternary_mel(
    cache: &mut Option<crate::distill_ternary_compile::CompiledDistilledTernaryMel>,
    inputs: &E2eBenchInputs<'_>,
    model: &crate::distill_ternary_model::DistilledTernaryFftModel,
    signal: &[f32],
) -> Result<Vec<f32>> {
    ensure_compiled_distilled_ternary(cache, inputs, model)?.log_mel_batch(signal)
}

fn ensure_compiled_distilled_ternary<'a>(
    cache: &'a mut Option<crate::distill_ternary_compile::CompiledDistilledTernaryMel>,
    inputs: &E2eBenchInputs<'_>,
    model: &crate::distill_ternary_model::DistilledTernaryFftModel,
) -> Result<&'a mut crate::distill_ternary_compile::CompiledDistilledTernaryMel> {
    if cache.is_none() {
        let cfg = FftLearnConfig::new(inputs.n_fft, inputs.batch)?;
        let deploy = pick_fused_deploy_device(inputs.batch, inputs.n_fft, inputs.device);
        *cache = Some(
            crate::distill_ternary_compile::compile_distilled_ternary_mel(model, &cfg, deploy)?,
        );
    }
    Ok(cache.as_mut().expect("distilled_ternary"))
}

fn compiled_distilled_mel(
    cache: &mut Option<CompiledDistilledMel>,
    inputs: &E2eBenchInputs<'_>,
    model: &DistilledFftModel,
    signal: &[f32],
) -> Result<Vec<f32>> {
    ensure_compiled_distilled(cache, inputs, model)?.log_mel_batch(signal)
}

fn ensure_compiled_distilled<'a>(
    cache: &'a mut Option<CompiledDistilledMel>,
    inputs: &E2eBenchInputs<'_>,
    model: &DistilledFftModel,
) -> Result<&'a mut CompiledDistilledMel> {
    if cache.is_none() {
        let cfg = FftLearnConfig::new(inputs.n_fft, inputs.batch)?;
        let deploy = pick_fused_deploy_device(inputs.batch, inputs.n_fft, inputs.device);
        *cache = Some(compile_distilled_mel(model, &cfg, deploy)?);
    }
    Ok(cache.as_mut().expect("distilled"))
}

fn compiled_learned_mel(
    cache: &mut Option<CompiledLearnedMel>,
    inputs: &E2eBenchInputs<'_>,
    model: &FastLearnedFftModel,
    windowed: &[f32],
) -> Result<Vec<f32>> {
    ensure_compiled_learned(cache, inputs, model)?.log_mel_batch(windowed)
}

fn ensure_compiled_learned<'a>(
    cache: &'a mut Option<CompiledLearnedMel>,
    inputs: &E2eBenchInputs<'_>,
    model: &FastLearnedFftModel,
) -> Result<&'a mut CompiledLearnedMel> {
    if cache.is_none() {
        let cfg = FftLearnConfig::new(inputs.n_fft, inputs.batch)?;
        *cache = Some(compile_learned_mel(
            model,
            &cfg,
            inputs.device,
            default_hard_threshold(),
        )?);
    }
    Ok(cache.as_mut().expect("compiled"))
}

fn rlx_spectrum(
    rlx_exec: &mut Option<CompiledGraph>,
    signal: &[f32],
    inputs: &E2eBenchInputs<'_>,
) -> Result<Vec<f32>> {
    if rlx_exec.is_none() {
        let cfg = FftLearnConfig::new(inputs.n_fft, inputs.batch)?;
        *rlx_exec = Some(compile_rlx_fft(&cfg, TransformDir::Forward, inputs.device)?);
    }
    let exec = rlx_exec.as_mut().unwrap();
    Ok(rlx_fft_forward(exec, signal, inputs.batch, inputs.n_fft))
}

fn rlx_mel(
    rlx_exec: &mut Option<CompiledGraph>,
    windowed: &[f32],
    inputs: &E2eBenchInputs<'_>,
    n_mels: usize,
    sr: f32,
) -> Result<Vec<f32>> {
    let spec = rlx_spectrum(rlx_exec, windowed, inputs)?;
    let filters = crate::mel::mel_filterbank(inputs.n_fft, n_mels, sr);
    crate::mel::log_mel_from_spectrum_batch(&spec, &filters, inputs.batch, inputs.n_fft, n_mels)
}

fn rlx_welch(
    rlx_exec: &mut Option<CompiledGraph>,
    welch_signal: &[f32],
    inputs: &E2eBenchInputs<'_>,
    params: WelchParams,
) -> Result<Vec<f32>> {
    let window = crate::welch::hann_window(params.n_fft);
    let segs = crate::welch::welch_windowed_segments(welch_signal, inputs.batch, params, &window)?;
    if rlx_exec.is_none() {
        let cfg = FftLearnConfig::new(params.n_fft, inputs.batch * params.n_segments)?;
        *rlx_exec = Some(compile_rlx_fft(&cfg, TransformDir::Forward, inputs.device)?);
    }
    let exec = rlx_exec.as_mut().unwrap();
    let spec = rlx_fft_forward(exec, &segs, inputs.batch * params.n_segments, params.n_fft);
    Ok(crate::welch::average_welch_psd(
        &spec,
        inputs.batch,
        params.n_segments,
        params.n_fft,
    ))
}

fn mel_from_butterfly(
    windowed: &[f32],
    tw: &[f32],
    batch: usize,
    n_fft: usize,
    n_mels: usize,
    sr: f32,
) -> Result<Vec<f32>> {
    let spec = butterfly_forward_real_batch(windowed, tw, batch, n_fft)?;
    let filters = crate::mel::mel_filterbank(n_fft, n_mels, sr);
    crate::mel::log_mel_from_spectrum_batch(&spec, &filters, batch, n_fft, n_mels)
}

fn window_batch(signal: &[f32], batch: usize, n_fft: usize) -> Vec<f32> {
    let w = hann_window(n_fft);
    let mut out = signal.to_vec();
    for b in 0..batch {
        for i in 0..n_fft {
            out[b * n_fft + i] *= w[i];
        }
    }
    out
}

pub fn print_e2e_table(report: &E2eBenchReport) {
    eprintln!(
        "\n=== E2E learned FFT validation (n_mels={}, iters={}) ===\n",
        report.n_mels, report.iters
    );
    let pipelines: Vec<String> = report
        .rows
        .iter()
        .map(|r| r.pipeline.clone())
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    for pipe in pipelines {
        eprintln!("--- pipeline: {pipe} ---");
        eprintln!(
            "{:<18} {:>8} {:>10} {:>12}",
            "backend", "ms", "max_err", "mean_gate"
        );
        let mut subset: Vec<_> = report.rows.iter().filter(|r| r.pipeline == pipe).collect();
        subset.sort_by(|a, b| a.ms.partial_cmp(&b.ms).unwrap_or(std::cmp::Ordering::Equal));
        for r in &subset {
            eprintln!(
                "{:<18} {:>8.4} {:>10.3e} {:>12}",
                r.backend,
                r.ms,
                r.max_err,
                r.mean_gate
                    .map(|g| format!("{g:.3}"))
                    .unwrap_or_else(|| "-".into())
            );
        }
        if let Some(best) = subset.first() {
            eprintln!("  → fastest: {} ({:.4} ms)\n", best.backend, best.ms);
        }
    }
    eprintln!("Total bench time: {:.1} ms\n", report.elapsed_ms);
    print_distilled_speed_gate(report);
    print_ternary_rustfft_speed_gate(report);
}

/// Success gate: fused ternary distilled mel within 5% of rustfft_ref.
pub fn print_ternary_rustfft_speed_gate(report: &E2eBenchReport) {
    let mut groups: std::collections::BTreeMap<(String, usize), Vec<&E2eBenchRow>> =
        std::collections::BTreeMap::new();
    for row in &report.rows {
        if row.pipeline == "mel" {
            groups
                .entry((row.device.clone(), row.batch))
                .or_default()
                .push(row);
        }
    }
    eprintln!("--- mel speed gate (learned_distilled_ternary vs rustfft_ref, +5% latency) ---");
    for ((dev, batch), rows) in groups {
        let rust = rows.iter().find(|r| r.backend == "rustfft_ref");
        let dist = rows
            .iter()
            .find(|r| r.backend == "learned_distilled_ternary");
        if let (Some(r), Some(d)) = (rust, dist) {
            let speed_ok = d.ms <= r.ms * 1.05;
            let acc_ok = d.max_err <= 0.55;
            let status = if speed_ok && acc_ok {
                "PASS"
            } else if speed_ok {
                "PASS_SPEED"
            } else if acc_ok {
                "PASS_ACC"
            } else {
                "FAIL"
            };
            eprintln!(
                "  {dev} batch={batch}: rust={:.4}ms err={:.3e} | ternary={:.4}ms err={:.3e} compute={:?} {}",
                r.ms, r.max_err, d.ms, d.max_err, d.mean_gate, status
            );
        }
    }
    eprintln!();
}

/// Success gate: distilled mel within 5% of rlx_op_fft on same device/batch.
pub fn print_distilled_speed_gate(report: &E2eBenchReport) {
    let mut groups: std::collections::BTreeMap<(String, usize), Vec<&E2eBenchRow>> =
        std::collections::BTreeMap::new();
    for row in &report.rows {
        if row.pipeline == "mel" {
            groups
                .entry((row.device.clone(), row.batch))
                .or_default()
                .push(row);
        }
    }
    eprintln!(
        "--- mel speed gate (learned_distilled vs rlx_op_fft, +5% latency, mel err ≤ 1.5× rlx) ---"
    );
    for ((dev, batch), rows) in groups {
        let rlx = rows.iter().find(|r| r.backend == "rlx_op_fft");
        let dist = rows.iter().find(|r| r.backend == "learned_distilled");
        if let (Some(r), Some(d)) = (rlx, dist) {
            let speed_ok = d.ms <= r.ms * 1.05;
            let acc_ok = d.max_err <= r.max_err * 1.5 + 1e-3;
            let status = if speed_ok && acc_ok {
                "PASS"
            } else if speed_ok {
                "PASS_SPEED"
            } else if acc_ok {
                "PASS_ACC"
            } else {
                "FAIL"
            };
            eprintln!(
                "  {dev} batch={batch}: rlx={:.4}ms err={:.3e} | distilled={:.4}ms err={:.3e} {}",
                r.ms, r.max_err, d.ms, d.max_err, status
            );
        }
    }
    eprintln!();
}

pub fn write_e2e_json(path: &std::path::Path, report: &E2eBenchReport) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, serde_json::to_vec_pretty(report)?)?;
    Ok(())
}

pub fn read_e2e_json(path: &std::path::Path) -> Result<E2eBenchReport> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("parse {}", path.display()))
}

/// Merge multiple device-specific e2e reports (e.g. Apple Silicon + CUDA rig) into one table/HTML.
pub fn merge_e2e_reports(reports: &[E2eBenchReport]) -> Result<E2eBenchReport> {
    ensure!(!reports.is_empty(), "merge_e2e_reports: empty input");
    let mut merged = reports[0].clone();
    merged.rows.clear();
    merged.elapsed_ms = 0.0;
    let mut devices = std::collections::BTreeSet::new();
    let mut batches = std::collections::BTreeSet::new();
    for report in reports {
        merged.rows.extend(report.rows.iter().cloned());
        devices.extend(report.meta.devices.iter().cloned());
        batches.extend(report.meta.batches.iter().copied());
        merged.elapsed_ms += report.elapsed_ms;
        for pb in &report.meta.per_batch {
            if !merged
                .meta
                .per_batch
                .iter()
                .any(|existing| existing.batch == pb.batch)
            {
                merged.meta.per_batch.push(pb.clone());
            }
        }
        if merged.meta.teacher.is_none() {
            merged.meta.teacher = report.meta.teacher.clone();
        }
        if merged.meta.distill.is_none() {
            merged.meta.distill = report.meta.distill.clone();
        }
    }
    merged.meta.devices = devices.into_iter().collect();
    merged.meta.batches = batches.into_iter().collect();
    merged.meta.per_batch.sort_by_key(|pb| pb.batch);
    Ok(merged)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e2e_bench_smoke() {
        let cfg = FftLearnConfig::new(64, 4).unwrap();
        let model = FastLearnedFftModel::new(&cfg, 16, 16_000.0).with_q8();
        let inputs = E2eBenchInputs {
            n_fft: 64,
            batch: 4,
            n_mels: 16,
            iters: 2,
            device: Device::Cpu,
            seed: 1,
            model: Some(&model),
            distilled: None,
            distilled_ternary: None,
            with_learned_hard: true,
            with_learned_compiled: true,
            with_learned_distilled: false,
            with_learned_distilled_ternary: false,
            with_eager_learned: false,
            peak_k: DEFAULT_PEAK_K,
        };
        let report = run_e2e_bench(&inputs).unwrap();
        assert!(report.rows.len() >= 4);
    }
}
