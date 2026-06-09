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

//! Distill teacher learned model → fast Op::Fft + correction student.

use crate::distill_model::{DistilledFftModel, teacher_mel_batch, teacher_welch_batch};
use crate::learned_model::FastLearnedFftModel;
use crate::mel::ref_log_mel_batch;
use crate::reference::{max_abs_error, mse};
use crate::train::random_batch;
use crate::welch::{WelchParams, welch_windowed_segments};
use anyhow::{Result, ensure};
use rand::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistillTrainConfig {
    pub n_fft: usize,
    pub batch: usize,
    pub n_mels: usize,
    pub steps: usize,
    pub lr: f32,
    pub mel_weight: f32,
    pub welch_weight: f32,
    pub seed: u64,
    pub log_every: usize,
}

impl Default for DistillTrainConfig {
    fn default() -> Self {
        Self {
            n_fft: 128,
            batch: 8,
            n_mels: 40,
            steps: 600,
            lr: 1e-3,
            mel_weight: 2.0,
            welch_weight: 1.0,
            seed: 7,
            log_every: 50,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistillTrainReport {
    pub steps: usize,
    pub final_mel_err_vs_teacher: f32,
    pub final_welch_err_vs_teacher: f32,
    pub final_mel_err_vs_ref: f32,
    pub elapsed_ms: f64,
}

pub fn distill_from_teacher(
    teacher: &FastLearnedFftModel,
    cfg: &DistillTrainConfig,
) -> Result<(DistilledFftModel, DistillTrainReport)> {
    ensure!(teacher.n_fft == cfg.n_fft && teacher.n_mels == cfg.n_mels);
    let started = std::time::Instant::now();
    let mut student = DistilledFftModel::from_teacher(teacher);
    let welch_params = WelchParams::for_n_fft(cfg.n_fft);
    let welch_frame = welch_params.frame_len();
    let welch_window = crate::welch::hann_window(cfg.n_fft);
    let mut rng = rand::rngs::StdRng::seed_from_u64(cfg.seed);

    let mut last_mel_teacher = 0f32;
    let mut last_welch_teacher = 0f32;
    let mut last_mel_ref = 0f32;

    for step in 0..cfg.steps {
        let signal = random_batch(&mut rng, cfg.batch, cfg.n_fft);
        let welch_signal = random_batch(&mut rng, cfg.batch, welch_frame);

        let teacher_mel = teacher_mel_batch(teacher, &signal, cfg.batch)?;
        let teacher_welch = teacher_welch_batch(teacher, &welch_signal, cfg.batch, welch_params)?;

        let pred_mel = student.log_mel_batch(&signal, cfg.batch)?;
        last_mel_teacher = max_abs_error(&pred_mel, &teacher_mel);
        let mel_loss = mse(&pred_mel, &teacher_mel);

        let pred_welch = student.welch_psd_batch(&welch_signal, cfg.batch, welch_params)?;
        last_welch_teacher = max_abs_error(&pred_welch, &teacher_welch);

        let lr = cfg.lr;
        student.train_step_mel(&signal, &teacher_mel, cfg.batch, lr * cfg.mel_weight)?;

        if cfg.welch_weight > 0.0 && step % 2 == 0 {
            let segs =
                welch_windowed_segments(&welch_signal, cfg.batch, welch_params, &welch_window)?;
            let n_segs = cfg.batch * welch_params.n_segments;
            let student_spec = student.spectrum_batch(&segs, n_segs)?;
            let teacher_spec = teacher.spectrum_batch_raw(&segs, n_segs)?;
            student.train_step_welch_spectrum(
                &student_spec,
                &teacher_spec,
                welch_params.n_segments,
                lr * cfg.welch_weight * 0.5,
            )?;
        }

        let windowed = crate::learned_compile::window_batch(&signal, cfg.batch, cfg.n_fft);
        last_mel_ref = max_abs_error(
            &student.log_mel_batch(&signal, cfg.batch)?,
            &ref_log_mel_batch(
                &windowed,
                cfg.batch,
                cfg.n_fft,
                cfg.n_mels,
                student.sample_rate,
            )?,
        );

        if step % cfg.log_every == 0 || step + 1 == cfg.steps {
            let total =
                cfg.mel_weight * mel_loss + cfg.welch_weight * mse(&pred_welch, &teacher_welch);
            eprintln!(
                "[train-distill] step={step}/{} loss={total:.4e} mel_vs_teacher={last_mel_teacher:.3e} welch_vs_teacher={last_welch_teacher:.3e} mel_vs_ref={last_mel_ref:.3e}",
                cfg.steps
            );
        }
    }

    Ok((
        student,
        DistillTrainReport {
            steps: cfg.steps,
            final_mel_err_vs_teacher: last_mel_teacher,
            final_welch_err_vs_teacher: last_welch_teacher,
            final_mel_err_vs_ref: last_mel_ref,
            elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
        },
    ))
}

pub fn distill_e2e(
    teacher: &FastLearnedFftModel,
    cfg: &DistillTrainConfig,
) -> Result<(DistilledFftModel, DistillTrainReport)> {
    distill_from_teacher(teacher, cfg)
}
