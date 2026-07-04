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

//! Train Tier-D fast learned FFT model (pruned + mask + denoiser + optional Q8).

use crate::learned::model::FastLearnedFftModel;
use crate::model::config::FftLearnConfig;
use crate::model::pruned::{
    gate_sparsity_loss, gate_train_step, mean_gate, pruned_forward_real_batch,
};
use crate::model::reference::{fft_real_batch, max_abs_error, mse};
use crate::model::twiddle_stability::{lr_for_n_fft, project_twiddles_unit_circle};
use crate::spectral::mel::{log_mel_loss_grad_wrt_spectrum, ref_log_mel_batch};
use crate::spectral::peak::{
    WelchPeakParams, peak_loss_grad_wrt_spectrum, peak_match_loss, peak_max_err,
    peaks_from_psd_batch,
};
use crate::spectral::welch::hann_window as welch_hann;
use crate::spectral::welch::{
    WelchParams, welch_loss_grad_wrt_spectrum, welch_rustfft, welch_windowed_segments,
};
use crate::train::random_batch;
use anyhow::{Result, ensure};
use rand::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct E2eTrainConfig {
    pub n_fft: usize,
    pub batch: usize,
    pub n_mels: usize,
    pub steps: usize,
    pub lr: f64,
    /// L1 gate penalty multiplier (default 1e-2).
    pub sparsity_weight: f32,
    /// Gate learning rate (separate from twiddle lr).
    pub gate_lr: f32,
    /// Task weights — mel/welch prioritized over raw spectrum.
    pub mel_weight: f32,
    pub welch_weight: f32,
    /// Top-K Welch peak match (fast 2-segment path vs full Welch reference peaks).
    pub peak_weight: f32,
    pub peak_k: usize,
    pub spectrum_weight: f32,
    pub train_q8: bool,
    pub seed: u64,
    pub log_every: usize,
}

impl Default for E2eTrainConfig {
    fn default() -> Self {
        Self {
            n_fft: 256,
            batch: 8,
            n_mels: 40,
            steps: 2000,
            lr: 5e-4,
            sparsity_weight: 1e-2,
            gate_lr: crate::model::pruned::DEFAULT_GATE_LR,
            mel_weight: 2.0,
            welch_weight: 1.5,
            peak_weight: 2.0,
            peak_k: 16,
            spectrum_weight: 0.25,
            train_q8: true,
            seed: 42,
            log_every: 100,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct E2eTrainReport {
    pub steps: usize,
    pub final_spectrum_max_err: f32,
    pub final_mel_max_err: f32,
    pub final_welch_max_err: f32,
    pub final_peak_max_err: f32,
    pub mean_gate: f32,
    pub active_gates: usize,
    pub active_gates_hard: usize,
    pub q8_enabled: bool,
    pub elapsed_ms: f64,
}

pub fn train_fast_learned_model(
    cfg: &E2eTrainConfig,
) -> Result<(FastLearnedFftModel, E2eTrainReport)> {
    ensure!(cfg.steps >= 1);
    let started = std::time::Instant::now();
    let model_cfg = FftLearnConfig::new(cfg.n_fft, cfg.batch)?;
    let mut model = FastLearnedFftModel::new(&model_cfg, cfg.n_mels, 16_000.0);
    if cfg.train_q8 {
        model = model.with_q8();
    }

    let welch_params = WelchParams::for_n_fft(cfg.n_fft);
    let peak_params = WelchPeakParams::fast_for_n_fft(cfg.n_fft, cfg.peak_k);
    let welch_frame = welch_params.frame_len();
    let welch_window = welch_hann(cfg.n_fft);
    let gate_warmup = cfg.steps / 4;
    let mut rng = rand::rngs::StdRng::seed_from_u64(cfg.seed);

    let mut last_spec_err = 0f32;
    let mut last_mel_err = 0f32;
    let mut last_welch_err = 0f32;
    let mut last_peak_err = 0f32;

    for step in 0..cfg.steps {
        let signal = random_batch(&mut rng, cfg.batch, cfg.n_fft);
        let welch_signal = random_batch(&mut rng, cfg.batch, welch_frame);

        let ref_spec = fft_real_batch(&signal, cfg.batch, cfg.n_fft)?;
        let ref_mel = ref_log_mel_batch(
            &window_signal(&signal, cfg.batch, cfg.n_fft),
            cfg.batch,
            cfg.n_fft,
            cfg.n_mels,
            16_000.0,
        )?;
        let ref_welch = welch_rustfft(&welch_signal, cfg.batch, welch_params)?;

        let pred_mel = model.log_mel_batch(&signal, cfg.batch)?;
        last_mel_err = max_abs_error(&pred_mel, &ref_mel);
        let mel_loss = mse(&pred_mel, &ref_mel);

        let pred_welch = model.welch_psd_batch(&welch_signal, cfg.batch, welch_params)?;
        last_welch_err = max_abs_error(&pred_welch, &ref_welch);
        let welch_loss = mse(&pred_welch, &ref_welch);

        let ref_peaks =
            peaks_from_psd_batch(&ref_welch, cfg.batch, welch_params.n_bins(), peak_params.k);
        let fast_welch_signal =
            peak_params
                .welch
                .truncate_batch(&welch_signal, cfg.batch, welch_frame)?;
        let pred_peaks = model.welch_peaks_batch(&fast_welch_signal, cfg.batch, peak_params)?;
        last_peak_err = peak_max_err(&pred_peaks, &ref_peaks);
        let peak_loss = peak_match_loss(&pred_peaks, &ref_peaks, cfg.batch, peak_params.k);

        let tw = model.twiddles_for_forward();
        let mut pred_spec =
            pruned_forward_real_batch(&signal, &tw, &model.gates, cfg.batch, cfg.n_fft)?;
        apply_mask_inplace(&mut pred_spec, &model.freq_mask, cfg.batch, cfg.n_fft);
        let pred_spec_denoised = model
            .denoiser
            .apply_batch(&pred_spec, cfg.batch, cfg.n_fft)?;
        last_spec_err = max_abs_error(&pred_spec_denoised, &ref_spec);
        let spec_loss = mse(&pred_spec_denoised, &ref_spec);

        let gate_penalty = gate_sparsity_loss(&model.gates);
        let total_loss = cfg.spectrum_weight * spec_loss
            + cfg.mel_weight * mel_loss
            + cfg.welch_weight * welch_loss
            + cfg.peak_weight * peak_loss
            + cfg.sparsity_weight * gate_penalty;

        let lr = lr_for_n_fft(cfg.lr, cfg.n_fft);

        if step % 2 == 0 {
            crate::model::butterfly::butterfly_train_step(
                &signal,
                &mut model.twiddles,
                cfg.batch,
                cfg.n_fft,
                lr * cfg.spectrum_weight.max(0.1),
            )?;
            model.sync_q8();
        }

        let _ =
            model
                .denoiser
                .train_step_affine(&pred_spec, &ref_spec, cfg.batch, cfg.n_fft, lr * 0.5);

        // Gate update via task gradients (after warmup, every 4 steps); skip if task error spikes.
        if step >= gate_warmup && step % 4 == 0 && last_mel_err < 0.35 && last_spec_err < 0.35 {
            let mut grad_task = log_mel_loss_grad_wrt_spectrum(
                &pred_mel,
                &ref_mel,
                &pred_spec_denoised,
                model.mel_filters(),
                cfg.batch,
                cfg.n_fft,
                cfg.n_mels,
            );
            let norm = (cfg.batch * cfg.n_fft * 2) as f32;
            for (i, g) in grad_task.iter_mut().enumerate() {
                *g *= cfg.mel_weight;
                let bin = i % (cfg.n_fft * 2);
                *g += cfg.spectrum_weight * 2.0 * (pred_spec_denoised[i] - ref_spec[i]) / norm
                    * model.denoiser.scale[bin];
            }

            gate_train_step(
                &signal,
                &model.twiddles_for_forward(),
                &mut model.gates,
                &grad_task,
                &model.freq_mask,
                &model.denoiser.scale,
                cfg.batch,
                cfg.n_fft,
                cfg.gate_lr,
                cfg.sparsity_weight,
            )?;

            if cfg.welch_weight > 0.0 && step % 8 == 0 && last_welch_err < 2.0 {
                let welch_segs =
                    welch_windowed_segments(&welch_signal, cfg.batch, welch_params, &welch_window)?;
                let tw_w = model.twiddles_for_forward();
                let mut welch_spec = pruned_forward_real_batch(
                    &welch_segs,
                    &tw_w,
                    &model.gates,
                    cfg.batch * welch_params.n_segments,
                    cfg.n_fft,
                )?;
                apply_mask_inplace(
                    &mut welch_spec,
                    &model.freq_mask,
                    cfg.batch * welch_params.n_segments,
                    cfg.n_fft,
                );
                let welch_spec = model.denoiser.apply_batch(
                    &welch_spec,
                    cfg.batch * welch_params.n_segments,
                    cfg.n_fft,
                )?;
                let mut grad_welch = welch_loss_grad_wrt_spectrum(
                    &pred_welch,
                    &ref_welch,
                    &welch_spec,
                    cfg.batch,
                    welch_params.n_segments,
                    cfg.n_fft,
                );
                for g in &mut grad_welch {
                    *g *= cfg.welch_weight;
                }
                gate_train_step(
                    &welch_segs,
                    &tw_w,
                    &mut model.gates,
                    &grad_welch,
                    &model.freq_mask,
                    &model.denoiser.scale,
                    cfg.batch * welch_params.n_segments,
                    cfg.n_fft,
                    cfg.gate_lr * 0.5,
                    cfg.sparsity_weight * 0.5,
                )?;
            }

            if cfg.peak_weight > 0.0 && step % 8 == 0 && last_peak_err < 5.0 {
                let fast_segs = welch_windowed_segments(
                    &fast_welch_signal,
                    cfg.batch,
                    peak_params.welch,
                    &welch_window,
                )?;
                let tw_w = model.twiddles_for_forward();
                let mut fast_spec = pruned_forward_real_batch(
                    &fast_segs,
                    &tw_w,
                    &model.gates,
                    cfg.batch * peak_params.welch.n_segments,
                    cfg.n_fft,
                )?;
                apply_mask_inplace(
                    &mut fast_spec,
                    &model.freq_mask,
                    cfg.batch * peak_params.welch.n_segments,
                    cfg.n_fft,
                );
                let fast_spec = model.denoiser.apply_batch(
                    &fast_spec,
                    cfg.batch * peak_params.welch.n_segments,
                    cfg.n_fft,
                )?;
                let pred_fast_psd = crate::spectral::welch::average_welch_psd(
                    &fast_spec,
                    cfg.batch,
                    peak_params.welch.n_segments,
                    cfg.n_fft,
                );
                let grad_peak = peak_loss_grad_wrt_spectrum(
                    &pred_fast_psd,
                    &ref_welch,
                    &ref_peaks,
                    cfg.batch,
                    peak_params.n_bins(),
                    peak_params.k,
                    peak_params.band_half_width,
                );
                let mut grad_peak_spec = vec![0f32; fast_spec.len()];
                let n_bins = peak_params.n_bins();
                for seg in 0..(cfg.batch * peak_params.welch.n_segments) {
                    let b = seg / peak_params.welch.n_segments;
                    for k in 0..n_bins {
                        let g = grad_peak[b * n_bins + k] / peak_params.welch.n_segments as f32;
                        for c in 0..2 {
                            grad_peak_spec[seg * cfg.n_fft * 2 + k * 2 + c] = g;
                        }
                    }
                }
                for g in &mut grad_peak_spec {
                    *g *= cfg.peak_weight;
                }
                gate_train_step(
                    &fast_segs,
                    &tw_w,
                    &mut model.gates,
                    &grad_peak_spec,
                    &model.freq_mask,
                    &model.denoiser.scale,
                    cfg.batch * peak_params.welch.n_segments,
                    cfg.n_fft,
                    cfg.gate_lr * 0.75,
                    cfg.sparsity_weight * 0.25,
                )?;
            }
        }

        for m in model.freq_mask.iter_mut() {
            *m += lr * 0.0005 * (1.0 - *m);
        }

        project_twiddles_unit_circle(&mut model.twiddles);

        if step % cfg.log_every == 0 || step + 1 == cfg.steps {
            eprintln!(
                "[train-e2e] step={step}/{} loss={total_loss:.4e} spec={last_spec_err:.3e} mel={last_mel_err:.3e} welch={last_welch_err:.3e} peak={last_peak_err:.3e} mean_gate={:.3} active={}",
                cfg.steps,
                mean_gate(&model.gates),
                model.active_gates(crate::model::pruned::DEFAULT_GATE_THRESHOLD)
            );
        }
    }

    let hard = crate::model::pruned::hard_gates(
        &model.gates,
        crate::model::pruned::DEFAULT_GATE_THRESHOLD,
    );
    let report = E2eTrainReport {
        steps: cfg.steps,
        final_spectrum_max_err: last_spec_err,
        final_mel_max_err: last_mel_err,
        final_welch_max_err: last_welch_err,
        final_peak_max_err: last_peak_err,
        mean_gate: model.mean_gate(),
        active_gates: model.active_gates(crate::model::pruned::DEFAULT_GATE_THRESHOLD),
        active_gates_hard: hard.iter().filter(|&&g| g >= 0.5).count(),
        q8_enabled: model.use_q8,
        elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
    };
    Ok((model, report))
}

fn window_signal(signal: &[f32], batch: usize, n_fft: usize) -> Vec<f32> {
    let w = crate::spectral::mel::hann_window(n_fft);
    let mut out = signal.to_vec();
    for b in 0..batch {
        for i in 0..n_fft {
            out[b * n_fft + i] *= w[i];
        }
    }
    out
}

fn apply_mask_inplace(spec: &mut [f32], mask: &[f32], batch: usize, n_fft: usize) {
    for b in 0..batch {
        for i in 0..n_fft * 2 {
            spec[b * n_fft * 2 + i] *= mask[i];
        }
    }
}
