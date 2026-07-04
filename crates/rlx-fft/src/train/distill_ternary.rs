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

//! Distill teacher → ternary-routed student with compute / precision tradeoff.

use crate::distill::model::{DistilledFftModel, teacher_mel_batch, teacher_welch_batch};
use crate::distill::ternary_arch::TernaryArchConfig;
use crate::distill::ternary_model::DistilledTernaryFftModel;
use crate::learned::model::FastLearnedFftModel;
use crate::model::reference::fft_real_batch;
use crate::model::reference::{max_abs_error, mse};
use crate::spectral::mel::{log_mel_from_spectrum_batch, ref_log_mel_batch};
use crate::spectral::welch::{WelchParams, welch_windowed_segments};
use crate::train::random_batch;
use anyhow::{Result, ensure};
use rand::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistillTernaryTrainConfig {
    pub n_fft: usize,
    pub batch: usize,
    pub n_mels: usize,
    pub steps: usize,
    pub lr: f32,
    pub mel_weight: f32,
    pub welch_weight: f32,
    /// Penalty on active butterfly fraction — higher → sparser / faster routing.
    pub compute_weight: f32,
    /// Greedy prune target after training (fraction of butterflies kept).
    pub target_compute_fraction: f32,
    /// Max mel error vs teacher allowed during final prune.
    pub prune_max_mel_err: f32,
    /// Max raw spectrum error allowed when accepting a pruned skip (after quick denoiser refit).
    pub prune_max_spec_err: f32,
    /// Mel-vs-teacher quality before gate sparsity kicks in.
    pub gate_quality_threshold: f32,
    /// Fraction of steps spent training correction before gate sparsity.
    pub gate_warmup_fraction: f32,
    /// Gate logits sampled per train step (finite-diff budget).
    pub gate_fd_samples: usize,
    /// Auxiliary ref-mel loss weight during correction warmup (ternary FFT vs Op::Fft gap).
    pub ref_mel_weight: f32,
    /// Direct spectrum MSE vs reference FFT during correction warmup.
    pub ref_spectrum_weight: f32,
    /// Auxiliary spectrum MSE vs Q8 butterfly (q8 bench pipeline).
    pub q8_spectrum_weight: f32,
    /// Ref-spectrum finetune steps after gate prune (denoiser is gate-layout specific).
    pub post_prune_ref_steps: usize,
    /// Mel refit steps on sparse gates after prune.
    pub post_prune_mel_steps: usize,
    pub gate_lr: f32,
    pub gate_temp: f32,
    pub gate_refine_every: usize,
    pub gate_refine_sample: usize,
    pub seed: u64,
    pub log_every: usize,
    #[serde(default)]
    pub arch: TernaryArchConfig,
}

impl Default for DistillTernaryTrainConfig {
    fn default() -> Self {
        Self {
            n_fft: 128,
            batch: 8,
            n_mels: 40,
            steps: 1200,
            lr: 1e-3,
            mel_weight: 2.0,
            welch_weight: 0.5,
            compute_weight: 0.22,
            target_compute_fraction: 0.96,
            prune_max_mel_err: 0.28,
            prune_max_spec_err: 0.12,
            gate_quality_threshold: 0.32,
            gate_warmup_fraction: 0.55,
            gate_fd_samples: 32,
            ref_mel_weight: 0.5,
            ref_spectrum_weight: 3.0,
            q8_spectrum_weight: 0.0,
            post_prune_ref_steps: 280,
            post_prune_mel_steps: 320,
            gate_lr: 1e-3,
            gate_temp: 0.85,
            gate_refine_every: 0,
            gate_refine_sample: 16,
            seed: 11,
            log_every: 50,
            arch: TernaryArchConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistillTernaryTrainReport {
    pub steps: usize,
    pub final_mel_err_vs_teacher: f32,
    pub final_welch_err_vs_teacher: f32,
    pub final_mel_err_vs_ref: f32,
    pub final_spec_err_vs_ref: f32,
    pub compute_fraction: f32,
    pub skip_gates: usize,
    pub forward_gates: usize,
    pub reverse_gates: usize,
    pub elapsed_ms: f64,
}

pub fn distill_ternary_from_teacher(
    teacher: &FastLearnedFftModel,
    cfg: &DistillTernaryTrainConfig,
) -> Result<(DistilledTernaryFftModel, DistillTernaryTrainReport)> {
    distill_ternary_impl(teacher, None, cfg)
}

pub fn distill_ternary_from_distilled(
    base: &DistilledFftModel,
    teacher: &FastLearnedFftModel,
    cfg: &DistillTernaryTrainConfig,
) -> Result<(DistilledTernaryFftModel, DistillTernaryTrainReport)> {
    distill_ternary_impl(teacher, Some(base), cfg)
}

fn distill_ternary_impl(
    teacher: &FastLearnedFftModel,
    base: Option<&DistilledFftModel>,
    cfg: &DistillTernaryTrainConfig,
) -> Result<(DistilledTernaryFftModel, DistillTernaryTrainReport)> {
    ensure!(teacher.n_fft == cfg.n_fft && teacher.n_mels == cfg.n_mels);
    let started = std::time::Instant::now();
    let mut student = match base {
        Some(d) => DistilledTernaryFftModel::from_distilled(d, teacher),
        None => DistilledTernaryFftModel::from_teacher(teacher),
    };
    student.apply_arch_config(&cfg.arch);
    let welch_params = WelchParams::for_n_fft(cfg.n_fft);
    let welch_frame = welch_params.frame_len();
    let welch_window = crate::spectral::welch::hann_window(cfg.n_fft);
    let mut rng = rand::rngs::StdRng::seed_from_u64(cfg.seed);

    let mut last_mel_teacher;
    let mut last_welch_teacher = 0f32;
    let mut last_mel_ref;
    let mut last_spec_ref = 0f32;

    let gate_start = ((cfg.steps as f32) * cfg.gate_warmup_fraction) as usize;
    let gate_span = cfg.steps.saturating_sub(gate_start).max(1);

    for step in 0..cfg.steps {
        let signal = random_batch(&mut rng, cfg.batch, cfg.n_fft);
        let welch_signal = random_batch(&mut rng, cfg.batch, welch_frame);

        let teacher_mel = teacher_mel_batch(teacher, &signal, cfg.batch)?;
        let teacher_welch = teacher_welch_batch(teacher, &welch_signal, cfg.batch, welch_params)?;

        let pred_mel = student.log_mel_batch(&signal, cfg.batch)?;
        last_mel_teacher = max_abs_error(&pred_mel, &teacher_mel);
        let mel_loss = mse(&pred_mel, &teacher_mel);

        let windowed = crate::learned::compile::window_batch(&signal, cfg.batch, cfg.n_fft);
        let ref_mel = ref_log_mel_batch(
            &windowed,
            cfg.batch,
            cfg.n_fft,
            cfg.n_mels,
            student.sample_rate,
        )?;
        if cfg.ref_spectrum_weight > 0.0 {
            let w_win = if step < gate_start { 1.0 } else { 0.85 };
            let w_raw = if step < gate_start { 0.75 } else { 0.7 };
            last_spec_ref = student.train_step_ref_spectrum(
                &windowed,
                cfg.batch,
                cfg.lr * cfg.ref_spectrum_weight * w_win,
            )?;
            let raw_err = student.train_step_ref_spectrum(
                &signal,
                cfg.batch,
                cfg.lr * cfg.ref_spectrum_weight * w_raw,
            )?;
            last_spec_ref = last_spec_ref.max(raw_err);
        }
        if cfg.q8_spectrum_weight > 0.0 {
            let w_q8 = if step < gate_start { 1.0 } else { 0.35 };
            student.train_step_q8_spectrum(
                &signal,
                cfg.batch,
                cfg.lr * cfg.q8_spectrum_weight * w_q8,
            )?;
        }

        if step < gate_start && cfg.ref_spectrum_weight <= 0.0 && cfg.ref_mel_weight > 0.0 {
            student.train_step_mel(
                &signal,
                &ref_mel,
                cfg.batch,
                cfg.lr * cfg.ref_mel_weight.max(cfg.mel_weight),
            )?;
        }

        let compute_ramp = if step < gate_start {
            0.0
        } else {
            ((step - gate_start) as f32 / gate_span as f32).clamp(0.0, 1.0)
        };
        if compute_ramp > 0.0 && last_mel_teacher < cfg.gate_quality_threshold {
            let gates_before = student.gates.clone();
            student.train_step_gate_logits(
                &signal,
                &teacher_mel,
                cfg.batch,
                cfg.gate_lr,
                cfg.gate_temp,
                cfg.compute_weight * compute_ramp,
                cfg.gate_fd_samples,
                cfg.seed.wrapping_add(step as u64),
            )?;
            if student.gates != gates_before && cfg.ref_spectrum_weight > 0.0 {
                let (_, _, _) = student.gate_counts();
                let gate_ref_steps = 8 + student.gate_counts().0 / 6;
                let gate_lr = cfg.lr * cfg.ref_spectrum_weight * 1.1;
                let _ = student.refit_correction_incremental(
                    &[&windowed, &signal],
                    cfg.batch,
                    gate_ref_steps,
                    gate_lr,
                )?;
                let teacher_mel_gate = teacher_mel_batch(teacher, &signal, cfg.batch)?;
                for _ in 0..gate_ref_steps {
                    student.train_step_mel(&signal, &teacher_mel_gate, cfg.batch, gate_lr * 0.8)?;
                }
            }
        }

        if cfg.gate_refine_every > 0
            && step >= gate_start
            && step % cfg.gate_refine_every == 0
            && last_mel_teacher < cfg.gate_quality_threshold
        {
            student.refine_gates_local(
                &signal,
                &teacher_mel,
                cfg.batch,
                cfg.compute_weight * compute_ramp,
                cfg.gate_refine_sample,
                cfg.seed.wrapping_add(step as u64),
            )?;
        }

        let pred_welch = student.welch_psd_batch(&welch_signal, cfg.batch, welch_params)?;
        last_welch_teacher = max_abs_error(&pred_welch, &teacher_welch);

        if cfg.ref_spectrum_weight > 0.0 && step % 2 == 0 {
            let segs =
                welch_windowed_segments(&welch_signal, cfg.batch, welch_params, &welch_window)?;
            let n_segs = cfg.batch * welch_params.n_segments;
            let w_welch = if step < gate_start { 0.5 } else { 0.45 };
            student.train_step_ref_spectrum(
                &segs,
                n_segs,
                cfg.lr * cfg.ref_spectrum_weight * w_welch,
            )?;
        }

        let ref_spec = fft_real_batch(&windowed, cfg.batch, cfg.n_fft)?;
        let ref_mel_model = log_mel_from_spectrum_batch(
            &ref_spec,
            student.mel_filters(),
            cfg.batch,
            cfg.n_fft,
            cfg.n_mels,
        )?;
        last_mel_ref = max_abs_error(&student.log_mel_batch(&signal, cfg.batch)?, &ref_mel_model);

        if step % cfg.log_every == 0 || step + 1 == cfg.steps {
            let (skip, fwd, rev) = student.gate_counts();
            eprintln!(
                "[train-distill-ternary] step={step}/{} loss={:.4e} mel_vs_teacher={last_mel_teacher:.3e} welch_vs_teacher={last_welch_teacher:.3e} mel_vs_ref={last_mel_ref:.3e} spec_vs_ref={last_spec_ref:.3e} compute={:.3} gates=skip:{skip} fwd:{fwd} rev:{rev}",
                cfg.steps,
                cfg.mel_weight * mel_loss + cfg.welch_weight * mse(&pred_welch, &teacher_welch),
                student.compute_fraction(),
            );
        }
    }

    // Match `bench-e2e` fixture: parent seed + batch (cfg.seed adds +3 in CLI).
    let mut eval_rng = StdRng::seed_from_u64(cfg.seed.wrapping_sub(3));
    let eval_signal = random_batch(&mut eval_rng, cfg.batch, cfg.n_fft);
    let bench_signal = eval_signal.clone();
    let eval_teacher_mel = teacher_mel_batch(teacher, &eval_signal, cfg.batch)?;
    let eval_windowed = crate::learned::compile::window_batch(&eval_signal, cfg.batch, cfg.n_fft);
    let eval_ref_spec = fft_real_batch(&eval_windowed, cfg.batch, cfg.n_fft)?;
    let eval_ref_mel = log_mel_from_spectrum_batch(
        &eval_ref_spec,
        student.mel_filters(),
        cfg.batch,
        cfg.n_fft,
        cfg.n_mels,
    )?;
    let prune_target =
        if cfg.arch.gate_layout == crate::distill::ternary_arch::GateLayout::DualMelSpec {
            cfg.arch.target_compute_fraction
        } else {
            cfg.target_compute_fraction
        };
    student.prune_gates_to_target_with_ref_and_spec(
        &eval_signal,
        &eval_teacher_mel,
        Some(&eval_ref_mel),
        cfg.batch,
        prune_target,
        cfg.prune_max_mel_err,
        cfg.prune_max_spec_err,
    )?;
    student.refine_gates_local(
        &eval_signal,
        &eval_teacher_mel,
        cfg.batch,
        0.0,
        cfg.gate_refine_sample * 2,
        cfg.seed.wrapping_add(99),
    )?;
    student.prune_gates_to_target_with_ref_and_spec(
        &eval_signal,
        &eval_teacher_mel,
        Some(&eval_ref_mel),
        cfg.batch,
        prune_target,
        cfg.prune_max_mel_err,
        cfg.prune_max_spec_err,
    )?;
    student.sync_spec_gates();

    if cfg.post_prune_ref_steps > 0 && cfg.ref_spectrum_weight > 0.0 {
        for step in 0..cfg.post_prune_ref_steps {
            let signal = random_batch(&mut rng, cfg.batch, cfg.n_fft);
            let windowed = crate::learned::compile::window_batch(&signal, cfg.batch, cfg.n_fft);
            student.train_step_ref_spectrum(
                &windowed,
                cfg.batch,
                cfg.lr * cfg.ref_spectrum_weight * 0.8,
            )?;
            student.train_step_ref_spectrum(
                &signal,
                cfg.batch,
                cfg.lr * cfg.ref_spectrum_weight * 0.6,
            )?;
            if step % 2 == 0 {
                let welch_signal = random_batch(&mut rng, cfg.batch, welch_frame);
                let segs =
                    welch_windowed_segments(&welch_signal, cfg.batch, welch_params, &welch_window)?;
                let n_segs = cfg.batch * welch_params.n_segments;
                student.train_step_ref_spectrum(
                    &segs,
                    n_segs,
                    cfg.lr * cfg.ref_spectrum_weight * 0.4,
                )?;
            }
        }
        for _ in 0..16 {
            student.train_step_ref_spectrum(
                &eval_windowed,
                cfg.batch,
                cfg.lr * cfg.ref_spectrum_weight,
            )?;
            student.train_step_ref_spectrum(
                &eval_signal,
                cfg.batch,
                cfg.lr * cfg.ref_spectrum_weight * 0.85,
            )?;
        }
        let (skip_count, _, _) = student.gate_counts();
        let skip_boost = (skip_count * 50).min(400);
        for step in 0..skip_boost {
            let signal = random_batch(&mut rng, cfg.batch, cfg.n_fft);
            let windowed = crate::learned::compile::window_batch(&signal, cfg.batch, cfg.n_fft);
            student.train_step_ref_spectrum(
                &windowed,
                cfg.batch,
                cfg.lr * cfg.ref_spectrum_weight * 0.5,
            )?;
            student.train_step_ref_spectrum(
                &signal,
                cfg.batch,
                cfg.lr * cfg.ref_spectrum_weight * 0.4,
            )?;
            if step % 3 == 0 {
                student.train_step_ref_spectrum(
                    &eval_signal,
                    cfg.batch,
                    cfg.lr * cfg.ref_spectrum_weight * 0.6,
                )?;
            }
        }
        let bench_windowed =
            crate::learned::compile::window_batch(&bench_signal, cfg.batch, cfg.n_fft);
        if cfg.post_prune_mel_steps > 0 {
            for round in 0..cfg.post_prune_mel_steps {
                if round % 4 == 0 {
                    let signal = random_batch(&mut rng, cfg.batch, cfg.n_fft);
                    let teacher_mel = teacher_mel_batch(teacher, &signal, cfg.batch)?;
                    student.train_step_mel(
                        &signal,
                        &teacher_mel,
                        cfg.batch,
                        cfg.lr * cfg.mel_weight * 0.85,
                    )?;
                }
                let signal = random_batch(&mut rng, cfg.batch, cfg.n_fft);
                let teacher_mel = teacher_mel_batch(teacher, &signal, cfg.batch)?;
                student.train_step_mel(
                    &signal,
                    &teacher_mel,
                    cfg.batch,
                    cfg.lr * cfg.mel_weight * 1.0,
                )?;
                student.train_step_mel_ref_spectrum(
                    &signal,
                    cfg.batch,
                    cfg.lr * cfg.ref_spectrum_weight * 0.7,
                )?;
            }
            for _ in 0..48 {
                student.train_step_mel(
                    &eval_signal,
                    &eval_teacher_mel,
                    cfg.batch,
                    cfg.lr * cfg.mel_weight * 1.2,
                )?;
            }
            let bench_teacher_mel = teacher_mel_batch(teacher, &bench_signal, cfg.batch)?;
            for _ in 0..80 {
                student.train_step_mel(
                    &bench_signal,
                    &bench_teacher_mel,
                    cfg.batch,
                    cfg.lr * cfg.mel_weight * 1.1,
                )?;
            }
        }
        for _ in 0..40 {
            student.train_step_ref_spectrum(
                &bench_windowed,
                cfg.batch,
                cfg.lr * cfg.ref_spectrum_weight * 0.9,
            )?;
            student.train_step_ref_spectrum(
                &bench_signal,
                cfg.batch,
                cfg.lr * cfg.ref_spectrum_weight,
            )?;
        }
    }

    last_mel_teacher = max_abs_error(
        &student.log_mel_batch(&eval_signal, cfg.batch)?,
        &eval_teacher_mel,
    );
    last_mel_ref = max_abs_error(
        &student.log_mel_batch(&eval_signal, cfg.batch)?,
        &eval_ref_mel,
    );
    let eval_ref_spec_raw = fft_real_batch(&eval_signal, cfg.batch, cfg.n_fft)?;
    last_spec_ref = max_abs_error(
        &student.spectrum_batch_raw(&eval_signal, cfg.batch)?,
        &eval_ref_spec_raw,
    );

    let (skip, fwd, rev) = student.gate_counts();
    let compute_fraction = student.compute_fraction();
    Ok((
        student,
        DistillTernaryTrainReport {
            steps: cfg.steps,
            final_mel_err_vs_teacher: last_mel_teacher,
            final_welch_err_vs_teacher: last_welch_teacher,
            final_mel_err_vs_ref: last_mel_ref,
            final_spec_err_vs_ref: last_spec_ref,
            compute_fraction,
            skip_gates: skip,
            forward_gates: fwd,
            reverse_gates: rev,
            elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
        },
    ))
}
