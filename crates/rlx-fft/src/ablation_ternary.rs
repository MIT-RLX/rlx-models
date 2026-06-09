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

//! Ternary architecture ablation — speed vs per-pipeline error (Pareto).

use crate::bench_sweep::available_devices;
use crate::butterfly::butterfly_forward_real_batch;
use crate::config::{FftLearnConfig, TransformDir};
use crate::device::{ensure_backend_ready, resolve_train_device};
use crate::distill_model::DistilledFftModel;
use crate::distill_ternary_compile::{CompiledDistilledTernaryMel, compile_distilled_ternary_mel};
use crate::distill_ternary_model::DistilledTernaryFftModel;
use crate::e2e_bench::E2ePipeline;
use crate::learned_model::{pipeline_max_err, ref_spectrum_batch, ref_welch};
use crate::mel::{log_mel_from_windowed_batch, ref_log_mel_batch};
use crate::peak::{WelchPeakParams, peaks_from_psd_batch, welch_peaks_rustfft};
use crate::q8::Q8Twiddles;
use crate::reference::{fft_real_batch, max_abs_error};
use crate::rlx_fft::{compile_rlx_fft, rlx_fft_forward};
use crate::ternary_arch::TernaryArchConfig;
use crate::train::random_batch;
use crate::train_distill::{DistillTrainConfig, distill_from_teacher};
use crate::train_distill_ternary::{DistillTernaryTrainConfig, distill_ternary_from_distilled};
use crate::train_e2e::{E2eTrainConfig, train_fast_learned_model};
use crate::twiddle::exact_twiddles;
use crate::welch::{WelchParams, welch_rustfft};
use anyhow::{Context, Result, ensure};
use rand::prelude::*;
use rlx_runtime::CompiledGraph;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TernaryExecMode {
    Eager,
    CompiledMel,
    CompiledSpectrum,
}

impl TernaryExecMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::Eager => "eager",
            Self::CompiledMel => "compiled_mel",
            Self::CompiledSpectrum => "compiled_spectrum",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TernaryArchVariantId {
    RefRustfft,
    RefRlxOpFft,
    RefButterflyCompiled,
    FusionOnly,
    SingleSparseBand,
    SingleSparseAffine,
    SingleSparseDense,
    DualPath,
}

impl TernaryArchVariantId {
    pub fn label(self) -> &'static str {
        match self {
            Self::RefRustfft => "ref_rustfft",
            Self::RefRlxOpFft => "ref_rlx_op_fft",
            Self::RefButterflyCompiled => "ref_butterfly_compiled",
            Self::FusionOnly => "fusion_only",
            Self::SingleSparseBand => "single_sparse_band",
            Self::SingleSparseAffine => "single_sparse_affine",
            Self::SingleSparseDense => "single_sparse_dense",
            Self::DualPath => "dual_path",
        }
    }

    pub fn needs_ternary_train(self) -> bool {
        !matches!(
            self,
            Self::RefRustfft | Self::RefRlxOpFft | Self::RefButterflyCompiled
        )
    }

    pub fn arch_config(self, prune_target: f32) -> Option<TernaryArchConfig> {
        match self {
            Self::FusionOnly => Some(TernaryArchConfig::fusion_only()),
            Self::SingleSparseBand => Some(TernaryArchConfig::single_sparse_band(prune_target)),
            Self::SingleSparseAffine => Some(TernaryArchConfig::single_sparse_affine(prune_target)),
            Self::SingleSparseDense => Some(TernaryArchConfig::single_sparse_dense(prune_target)),
            Self::DualPath => Some(TernaryArchConfig::dual_path_mel_sparse()),
            _ => None,
        }
    }

    pub fn exec_modes(self) -> &'static [TernaryExecMode] {
        match self {
            Self::RefRustfft | Self::RefRlxOpFft | Self::RefButterflyCompiled => {
                &[TernaryExecMode::Eager]
            }
            Self::FusionOnly | Self::SingleSparseDense | Self::DualPath => &[
                TernaryExecMode::CompiledMel,
                TernaryExecMode::CompiledSpectrum,
            ],
            Self::SingleSparseBand => &[
                TernaryExecMode::Eager,
                TernaryExecMode::CompiledMel,
                TernaryExecMode::CompiledSpectrum,
            ],
            Self::SingleSparseAffine => &[TernaryExecMode::Eager],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TernaryAblationRow {
    pub variant: String,
    pub pipeline: String,
    pub exec: String,
    pub n_fft: usize,
    pub batch: usize,
    pub device: String,
    pub iters: usize,
    pub ms: f64,
    pub max_err: f32,
    pub norm_err: f32,
    pub compute_fraction: Option<f32>,
    pub skip_gates: Option<usize>,
    pub prune_target: Option<f32>,
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TernaryAblationReport {
    pub quick: bool,
    pub iters: usize,
    pub teacher_steps: usize,
    pub distill_steps: usize,
    pub ternary_steps: usize,
    pub n_ffts: Vec<usize>,
    pub batches: Vec<usize>,
    pub prune_targets: Vec<f32>,
    pub elapsed_ms: f64,
    pub rows: Vec<TernaryAblationRow>,
}

#[derive(Debug, Clone)]
pub struct TernaryAblationOpts {
    pub n_ffts: Vec<usize>,
    pub batches: Vec<usize>,
    pub devices: Vec<String>,
    pub iters: usize,
    pub teacher_steps: usize,
    pub distill_steps: usize,
    pub ternary_steps: usize,
    pub prune_targets: Vec<f32>,
    pub seed: u64,
    pub quick: bool,
}

impl Default for TernaryAblationOpts {
    fn default() -> Self {
        Self {
            n_ffts: vec![128],
            batches: vec![8],
            devices: available_devices()
                .into_iter()
                .map(str::to_string)
                .collect(),
            iters: 20,
            teacher_steps: 400,
            distill_steps: 600,
            ternary_steps: 600,
            prune_targets: vec![1.0, 0.98, 0.96, 0.94],
            seed: 42,
            quick: false,
        }
    }
}

pub fn quick_ablation_opts() -> TernaryAblationOpts {
    TernaryAblationOpts {
        n_ffts: vec![128],
        batches: vec![8],
        iters: 12,
        teacher_steps: 200,
        distill_steps: 300,
        ternary_steps: 300,
        prune_targets: vec![1.0, 0.96],
        quick: true,
        ..TernaryAblationOpts::default()
    }
}

pub fn variant_matrix(quick: bool, prune_targets: &[f32]) -> Vec<(TernaryArchVariantId, f32)> {
    let mut out = vec![
        (TernaryArchVariantId::RefRustfft, 1.0),
        (TernaryArchVariantId::RefRlxOpFft, 1.0),
        (TernaryArchVariantId::RefButterflyCompiled, 1.0),
        (TernaryArchVariantId::FusionOnly, 1.0),
    ];
    for &t in prune_targets {
        out.push((TernaryArchVariantId::SingleSparseBand, t));
        if !quick {
            out.push((TernaryArchVariantId::SingleSparseAffine, t));
            if (t - 0.96).abs() < 0.01 || (t - 1.0).abs() < 0.01 {
                out.push((TernaryArchVariantId::SingleSparseDense, t));
            }
        }
    }
    if !quick {
        out.push((TernaryArchVariantId::DualPath, 0.94));
    } else {
        out.push((TernaryArchVariantId::DualPath, 0.96));
    }
    out
}

pub fn run_ternary_ablation(opts: &TernaryAblationOpts) -> Result<TernaryAblationReport> {
    ensure!(!opts.n_ffts.is_empty() && !opts.batches.is_empty());
    let started = Instant::now();
    let mut rows = Vec::new();
    let matrix = variant_matrix(opts.quick, &opts.prune_targets);

    for &n_fft in &opts.n_ffts {
        for &batch in &opts.batches {
            let n_mels = 40;
            let sr = 16_000.0f32;
            let welch_params = WelchParams::for_n_fft(n_fft);
            let welch_frame = welch_params.frame_len();

            let mut rng = StdRng::seed_from_u64(
                opts.seed
                    .wrapping_add(n_fft as u64)
                    .wrapping_add(batch as u64),
            );
            let signal = random_batch(&mut rng, batch, n_fft);
            let welch_signal = random_batch(&mut rng, batch, welch_frame);
            let windowed = crate::learned_compile::window_batch(&signal, batch, n_fft);

            eprintln!("[ablation-ternary] training teacher n={n_fft} batch={batch}");
            let teacher_cfg = E2eTrainConfig {
                n_fft,
                batch,
                n_mels,
                steps: opts.teacher_steps,
                seed: opts.seed,
                ..E2eTrainConfig::default()
            };
            let (teacher, _) = train_fast_learned_model(&teacher_cfg)?;
            let distill_cfg = DistillTrainConfig {
                n_fft,
                batch,
                n_mels,
                steps: opts.distill_steps,
                seed: opts.seed.wrapping_add(1),
                ..DistillTrainConfig::default()
            };
            let (distilled, _) = distill_from_teacher(&teacher, &distill_cfg)?;

            let ref_baselines = compute_ref_baselines(
                &signal,
                &welch_signal,
                &windowed,
                batch,
                n_fft,
                n_mels,
                sr,
                &welch_params,
            )?;

            for device_name in &opts.devices {
                let device = match resolve_train_device(Some(device_name)) {
                    Ok(d) => d,
                    Err(e) => {
                        eprintln!("[ablation-ternary] skip device {device_name}: {e:#}");
                        continue;
                    }
                };
                if let Err(e) = ensure_backend_ready(device) {
                    eprintln!("[ablation-ternary] skip device {device_name}: {e:#}");
                    continue;
                }

                for &(variant, prune_target) in &matrix {
                    eprintln!(
                        "[ablation-ternary] {} prune={prune_target:.2} n={n_fft} batch={batch} dev={device_name}",
                        variant.label()
                    );

                    let student = if variant.needs_ternary_train() {
                        let mut arch = variant.arch_config(prune_target).expect("arch");
                        arch.target_compute_fraction = prune_target;
                        let mut tcfg = DistillTernaryTrainConfig {
                            n_fft,
                            batch,
                            n_mels,
                            steps: opts.ternary_steps,
                            target_compute_fraction: prune_target,
                            seed: opts.seed.wrapping_add(3),
                            arch,
                            ..DistillTernaryTrainConfig::default()
                        };
                        if opts.quick {
                            tcfg.post_prune_ref_steps = 120;
                            tcfg.post_prune_mel_steps = 120;
                        }
                        let (m, _) = distill_ternary_from_distilled(&distilled, &teacher, &tcfg)?;
                        Some(m)
                    } else {
                        None
                    };

                    let compute_fraction = student.as_ref().map(|m| m.compute_fraction());
                    let skip_gates = student.as_ref().map(|m| m.gate_counts().0);

                    for &exec in variant.exec_modes() {
                        for pipeline in E2ePipeline::all() {
                            if matches!(
                                *pipeline,
                                E2ePipeline::WelchPeaks | E2ePipeline::WelchPeaksUltra
                            ) {
                                continue;
                            }
                            let bench = bench_ternary_row(
                                variant,
                                *pipeline,
                                exec,
                                student.as_ref(),
                                &distilled,
                                &signal,
                                &welch_signal,
                                &windowed,
                                batch,
                                n_fft,
                                n_mels,
                                sr,
                                &welch_params,
                                device,
                                device_name,
                                opts.iters,
                                prune_target,
                                compute_fraction,
                                skip_gates,
                            );
                            match bench {
                                Ok((ms, err)) => {
                                    let ref_err = ref_baselines[pipeline.label()];
                                    let norm = if ref_err > 1e-12 { err / ref_err } else { err };
                                    rows.push(TernaryAblationRow {
                                        variant: variant.label().to_string(),
                                        pipeline: pipeline.label().to_string(),
                                        exec: exec.label().to_string(),
                                        n_fft,
                                        batch,
                                        device: device_name.clone(),
                                        iters: opts.iters,
                                        ms,
                                        max_err: err,
                                        norm_err: norm,
                                        compute_fraction,
                                        skip_gates,
                                        prune_target: if variant.needs_ternary_train() {
                                            Some(prune_target)
                                        } else {
                                            None
                                        },
                                        status: "ok".into(),
                                        note: None,
                                    });
                                }
                                Err(e) => {
                                    rows.push(TernaryAblationRow {
                                        variant: variant.label().to_string(),
                                        pipeline: pipeline.label().to_string(),
                                        exec: exec.label().to_string(),
                                        n_fft,
                                        batch,
                                        device: device_name.clone(),
                                        iters: opts.iters,
                                        ms: f64::NAN,
                                        max_err: f32::NAN,
                                        norm_err: f32::NAN,
                                        compute_fraction,
                                        skip_gates,
                                        prune_target: Some(prune_target),
                                        status: "bench_fail".into(),
                                        note: Some(format!("{e:#}")),
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(TernaryAblationReport {
        quick: opts.quick,
        iters: opts.iters,
        teacher_steps: opts.teacher_steps,
        distill_steps: opts.distill_steps,
        ternary_steps: opts.ternary_steps,
        n_ffts: opts.n_ffts.clone(),
        batches: opts.batches.clone(),
        prune_targets: opts.prune_targets.clone(),
        elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
        rows,
    })
}

fn compute_ref_baselines(
    signal: &[f32],
    welch_signal: &[f32],
    windowed: &[f32],
    batch: usize,
    n_fft: usize,
    n_mels: usize,
    sr: f32,
    welch_params: &WelchParams,
) -> Result<HashMap<&'static str, f32>> {
    let mel = ref_log_mel_batch(windowed, batch, n_fft, n_mels, sr)?;
    let pred_mel = log_mel_from_windowed_batch(windowed, batch, n_fft, n_mels, sr)?;
    let denoise = ref_spectrum_batch(signal, batch, n_fft)?;
    let pred_d = fft_real_batch(signal, batch, n_fft)?;
    let tw = Q8Twiddles::from_f32(&exact_twiddles(&FftLearnConfig::new(n_fft, batch)?));
    let q8 = tw.forward_real_batch(signal, batch, n_fft)?;
    let pred_q8 = q8.clone();
    let welch = ref_welch(welch_signal, batch, *welch_params)?;
    let pred_w = welch_rustfft(welch_signal, batch, *welch_params)?;
    let mut m = HashMap::new();
    m.insert("mel", max_abs_error(&pred_mel, &mel).max(1e-9));
    m.insert("denoise", max_abs_error(&pred_d, &denoise).max(1e-9));
    m.insert("q8", max_abs_error(&pred_q8, &q8).max(1e-9));
    m.insert("welch", max_abs_error(&pred_w, &welch).max(1e-9));
    Ok(m)
}

#[allow(clippy::too_many_arguments)]
fn bench_ternary_row(
    variant: TernaryArchVariantId,
    pipeline: E2ePipeline,
    exec: TernaryExecMode,
    student: Option<&DistilledTernaryFftModel>,
    _distilled: &DistilledFftModel,
    signal: &[f32],
    welch_signal: &[f32],
    windowed: &[f32],
    batch: usize,
    n_fft: usize,
    n_mels: usize,
    sr: f32,
    welch_params: &WelchParams,
    device: rlx_runtime::Device,
    _device_name: &str,
    iters: usize,
    _prune_target: f32,
    _compute_fraction: Option<f32>,
    _skip_gates: Option<usize>,
) -> Result<(f64, f32)> {
    let cfg = FftLearnConfig::new(n_fft, batch)?;
    let tw = exact_twiddles(&cfg);
    let peak_k = 16usize;
    let peak_params = WelchPeakParams::fast_for_n_fft(n_fft, peak_k);
    let ref_out = match pipeline {
        E2ePipeline::Mel => ref_log_mel_batch(windowed, batch, n_fft, n_mels, sr)?,
        E2ePipeline::Welch => ref_welch(welch_signal, batch, *welch_params)?,
        E2ePipeline::WelchPeaks | E2ePipeline::WelchPeaksUltra => welch_peaks_rustfft(
            welch_signal,
            batch,
            WelchPeakParams::reference_for_n_fft(n_fft, peak_k),
        )?,
        E2ePipeline::Q8Spectrum => {
            Q8Twiddles::from_f32(&tw).forward_real_batch(signal, batch, n_fft)?
        }
        E2ePipeline::Denoise => ref_spectrum_batch(signal, batch, n_fft)?,
    };

    let mut compiled: Option<CompiledDistilledTernaryMel> = None;
    let mut rlx_ref: Option<CompiledGraph> = None;
    let run_once = |compiled: &mut Option<CompiledDistilledTernaryMel>,
                    rlx_ref: &mut Option<CompiledGraph>|
     -> Result<Vec<f32>> {
        match variant {
            TernaryArchVariantId::RefRustfft => match pipeline {
                E2ePipeline::Mel => log_mel_from_windowed_batch(windowed, batch, n_fft, n_mels, sr),
                E2ePipeline::Welch => ref_welch(welch_signal, batch, *welch_params),
                E2ePipeline::WelchPeaks | E2ePipeline::WelchPeaksUltra => {
                    welch_peaks_rustfft(welch_signal, batch, peak_params)
                }
                E2ePipeline::Q8Spectrum => fft_real_batch(signal, batch, n_fft),
                E2ePipeline::Denoise => ref_spectrum_batch(signal, batch, n_fft),
            },
            TernaryArchVariantId::RefRlxOpFft => {
                if rlx_ref.is_none() {
                    *rlx_ref = Some(compile_rlx_fft(&cfg, TransformDir::Forward, device)?);
                }
                let g = rlx_ref.as_mut().unwrap();
                match pipeline {
                    E2ePipeline::Mel => {
                        let spec = rlx_fft_forward(g, windowed, batch, n_fft);
                        crate::mel::log_mel_from_spectrum_batch(
                            &spec,
                            &crate::mel::mel_filterbank(n_fft, n_mels, sr),
                            batch,
                            n_fft,
                            n_mels,
                        )
                    }
                    E2ePipeline::Welch => ref_welch(welch_signal, batch, *welch_params),
                    E2ePipeline::WelchPeaks | E2ePipeline::WelchPeaksUltra => {
                        welch_peaks_rustfft(welch_signal, batch, peak_params)
                    }
                    E2ePipeline::Q8Spectrum | E2ePipeline::Denoise => {
                        Ok(rlx_fft_forward(g, signal, batch, n_fft))
                    }
                }
            }
            TernaryArchVariantId::RefButterflyCompiled => match pipeline {
                E2ePipeline::Mel => {
                    let spec = butterfly_forward_real_batch(signal, &tw, batch, n_fft)?;
                    crate::mel::log_mel_from_spectrum_batch(
                        &spec,
                        &crate::mel::mel_filterbank(n_fft, n_mels, sr),
                        batch,
                        n_fft,
                        n_mels,
                    )
                }
                E2ePipeline::Welch => {
                    crate::welch::welch_butterfly(welch_signal, &tw, batch, *welch_params)
                }
                E2ePipeline::WelchPeaks | E2ePipeline::WelchPeaksUltra => {
                    let psd =
                        crate::welch::welch_butterfly(welch_signal, &tw, batch, peak_params.welch)?;
                    Ok(peaks_from_psd_batch(
                        &psd,
                        batch,
                        peak_params.n_bins(),
                        peak_params.k,
                    ))
                }
                E2ePipeline::Q8Spectrum | E2ePipeline::Denoise => {
                    butterfly_forward_real_batch(signal, &tw, batch, n_fft)
                }
            },
            _ => {
                let m = student.context("ternary student")?;
                match (pipeline, exec) {
                    (E2ePipeline::Mel, TernaryExecMode::Eager) => m.log_mel_batch(signal, batch),
                    (E2ePipeline::Mel, TernaryExecMode::CompiledMel) => {
                        if compiled.is_none() {
                            *compiled = Some(compile_distilled_ternary_mel(m, &cfg, device)?);
                        }
                        compiled.as_mut().unwrap().log_mel_batch(signal)
                    }
                    (E2ePipeline::Denoise, TernaryExecMode::Eager) => {
                        m.spectrum_batch_raw(signal, batch)
                    }
                    (E2ePipeline::Denoise, TernaryExecMode::CompiledSpectrum) => {
                        if compiled.is_none() {
                            *compiled = Some(compile_distilled_ternary_mel(m, &cfg, device)?);
                        }
                        compiled.as_mut().unwrap().spectrum_batch(signal)
                    }
                    (E2ePipeline::Q8Spectrum, TernaryExecMode::Eager) => {
                        m.spectrum_batch_raw(signal, batch)
                    }
                    (E2ePipeline::Q8Spectrum, TernaryExecMode::CompiledSpectrum) => {
                        if compiled.is_none() {
                            *compiled = Some(compile_distilled_ternary_mel(m, &cfg, device)?);
                        }
                        compiled.as_mut().unwrap().spectrum_batch(signal)
                    }
                    (E2ePipeline::Welch, _) => {
                        m.welch_psd_batch(welch_signal, batch, *welch_params)
                    }
                    (E2ePipeline::WelchPeaks | E2ePipeline::WelchPeaksUltra, _) => {
                        m.welch_peaks_batch(welch_signal, batch, peak_params)
                    }
                    _ => anyhow::bail!("unsupported pipeline/exec {:?}/{:?}", pipeline, exec),
                }
            }
        }
    };

    for _ in 0..iters.max(1).saturating_sub(1) {
        let _ = run_once(&mut compiled, &mut rlx_ref)?;
    }
    let t0 = Instant::now();
    let pred = run_once(&mut compiled, &mut rlx_ref)?;
    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    let err = if matches!(variant, TernaryArchVariantId::RefRustfft) {
        0.0
    } else {
        pipeline_max_err(&pred, &ref_out)
    };
    Ok((ms, err))
}

pub fn ternary_ablation_row_ok(r: &TernaryAblationRow) -> bool {
    r.status == "ok" && r.ms.is_finite() && r.ms > 0.0
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TernaryParetoPoint {
    pub variant: String,
    pub exec: String,
    pub prune_target: Option<f32>,
    pub mean_ms: f64,
    pub max_norm_err: f32,
    pub mel_norm: f32,
    pub denoise_norm: f32,
    pub q8_norm: f32,
    pub welch_norm: f32,
    pub compute_fraction: Option<f32>,
}

fn prune_group_key(prune_target: Option<f32>) -> String {
    prune_target
        .map(|v| format!("{v:.6}"))
        .unwrap_or_else(|| "none".into())
}

pub fn ternary_aggregate_variants(
    report: &TernaryAblationReport,
    device: &str,
    n_fft: usize,
    batch: usize,
) -> Vec<TernaryParetoPoint> {
    let mut groups: HashMap<(String, String, String), Vec<&TernaryAblationRow>> = HashMap::new();
    for r in &report.rows {
        if r.device != device || r.n_fft != n_fft || r.batch != batch || !ternary_ablation_row_ok(r)
        {
            continue;
        }
        groups
            .entry((
                r.variant.clone(),
                r.exec.clone(),
                prune_group_key(r.prune_target),
            ))
            .or_default()
            .push(r);
    }
    let mut out = Vec::new();
    for ((variant, exec, _prune_key), rows) in groups {
        let prune_target = rows.first().and_then(|r| r.prune_target);
        let mut by_pipe: HashMap<&str, &TernaryAblationRow> = HashMap::new();
        for r in &rows {
            by_pipe.insert(r.pipeline.as_str(), r);
        }
        let mel = by_pipe.get("mel").map(|r| r.norm_err).unwrap_or(f32::NAN);
        let denoise = by_pipe
            .get("denoise")
            .map(|r| r.norm_err)
            .unwrap_or(f32::NAN);
        let q8 = by_pipe.get("q8").map(|r| r.norm_err).unwrap_or(f32::NAN);
        let welch = by_pipe.get("welch").map(|r| r.norm_err).unwrap_or(f32::NAN);
        let mean_ms = rows.iter().map(|r| r.ms).sum::<f64>() / rows.len().max(1) as f64;
        let max_norm = mel.max(denoise).max(q8).max(welch);
        out.push(TernaryParetoPoint {
            variant,
            exec,
            prune_target,
            mean_ms,
            max_norm_err: max_norm,
            mel_norm: mel,
            denoise_norm: denoise,
            q8_norm: q8,
            welch_norm: welch,
            compute_fraction: rows.first().and_then(|r| r.compute_fraction),
        });
    }
    out
}

pub fn ternary_pareto_frontier(points: &[TernaryParetoPoint]) -> Vec<&TernaryParetoPoint> {
    let mut front = Vec::new();
    'outer: for (i, a) in points.iter().enumerate() {
        for (j, b) in points.iter().enumerate() {
            if i == j {
                continue;
            }
            let dominates = b.mean_ms <= a.mean_ms
                && b.mel_norm <= a.mel_norm
                && b.denoise_norm <= a.denoise_norm
                && b.q8_norm <= a.q8_norm
                && b.welch_norm <= a.welch_norm
                && (b.mean_ms < a.mean_ms
                    || b.mel_norm < a.mel_norm
                    || b.denoise_norm < a.denoise_norm
                    || b.q8_norm < a.q8_norm
                    || b.welch_norm < a.welch_norm);
            if dominates {
                continue 'outer;
            }
        }
        front.push(a);
    }
    front.sort_by(|a, b| {
        a.mean_ms
            .partial_cmp(&b.mean_ms)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    front
}

pub fn ternary_recommendation(
    report: &TernaryAblationReport,
    device: &str,
    n_fft: usize,
    batch: usize,
) -> Option<String> {
    let points = ternary_aggregate_variants(report, device, n_fft, batch);
    let front = ternary_pareto_frontier(&points);
    front.first().map(|p| {
        format!(
            "{}@{} prune={:?} (mean_ms={:.4} max_norm={:.3e})",
            p.variant, p.exec, p.prune_target, p.mean_ms, p.max_norm_err
        )
    })
}

pub fn print_ternary_ablation_table(report: &TernaryAblationReport) {
    eprintln!("\n=== Ternary architecture ablation ===\n");
    let mut keys: Vec<(usize, usize, String)> = report
        .rows
        .iter()
        .map(|r| (r.n_fft, r.batch, r.device.clone()))
        .collect();
    keys.sort();
    keys.dedup();
    for (n, b, d) in keys {
        eprintln!("--- n_fft={n} batch={b} device={d} ---");
        if let Some(rec) = ternary_recommendation(report, &d, n, b) {
            eprintln!("  Pareto pick: {rec}");
        }
        let points = ternary_aggregate_variants(report, &d, n, b);
        let front = ternary_pareto_frontier(&points);
        eprintln!("  Frontier ({}):", front.len());
        for p in front.iter().take(8) {
            eprintln!(
                "    {} {} prune={:?} ms={:.4} mel={:.2e} den={:.2e} q8={:.2e} welch={:.2e} compute={:?}",
                p.variant,
                p.exec,
                p.prune_target,
                p.mean_ms,
                p.mel_norm,
                p.denoise_norm,
                p.q8_norm,
                p.welch_norm,
                p.compute_fraction
            );
        }
        eprintln!();
    }
    eprintln!("Total ablation time: {:.1} ms\n", report.elapsed_ms);
}

pub fn write_ternary_ablation_json(path: &Path, report: &TernaryAblationReport) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let bytes = serde_json::to_vec_pretty(report)?;
    std::fs::write(path, bytes).with_context(|| format!("write {}", path.display()))?;
    Ok(())
}

pub fn write_ternary_ablation_csv(path: &Path, report: &TernaryAblationReport) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut w = String::from(
        "variant,pipeline,exec,n_fft,batch,device,iters,ms,max_err,norm_err,compute_fraction,skip_gates,prune_target,status\n",
    );
    for r in &report.rows {
        w.push_str(&format!(
            "{},{},{},{},{},{},{},{:.6},{:.6e},{:.6e},{},{},{},{}\n",
            r.variant,
            r.pipeline,
            r.exec,
            r.n_fft,
            r.batch,
            r.device,
            r.iters,
            r.ms,
            r.max_err,
            r.norm_err,
            r.compute_fraction
                .map(|v| format!("{v:.4}"))
                .unwrap_or_else(|| "-".into()),
            r.skip_gates
                .map(|v| v.to_string())
                .unwrap_or_else(|| "-".into()),
            r.prune_target
                .map(|v| format!("{v:.3}"))
                .unwrap_or_else(|| "-".into()),
            r.status,
        ));
    }
    std::fs::write(path, w).with_context(|| format!("write {}", path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_point(
        variant: &str,
        ms: f64,
        mel: f32,
        den: f32,
        q8: f32,
        welch: f32,
    ) -> TernaryParetoPoint {
        TernaryParetoPoint {
            variant: variant.into(),
            exec: "eager".into(),
            prune_target: Some(0.96),
            mean_ms: ms,
            max_norm_err: mel.max(den).max(q8).max(welch),
            mel_norm: mel,
            denoise_norm: den,
            q8_norm: q8,
            welch_norm: welch,
            compute_fraction: Some(0.96),
        }
    }

    #[test]
    fn pareto_frontier_drops_dominated() {
        let points = vec![
            sample_point("slow_good", 10.0, 0.01, 0.01, 0.1, 0.01),
            sample_point("fast_ok", 5.0, 0.05, 0.05, 0.12, 0.05),
            sample_point("dominated", 12.0, 0.2, 0.2, 0.2, 0.2),
        ];
        let front = ternary_pareto_frontier(&points);
        let labels: Vec<_> = front.iter().map(|p| p.variant.as_str()).collect();
        assert!(!labels.contains(&"dominated"));
        assert!(labels.contains(&"slow_good"));
        assert!(labels.contains(&"fast_ok"));
    }

    #[test]
    fn variant_matrix_quick_smaller() {
        let q = variant_matrix(true, &[1.0, 0.96]);
        let f = variant_matrix(false, &[1.0, 0.98, 0.96, 0.94]);
        assert!(q.len() < f.len());
    }
}
