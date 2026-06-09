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

//! Distilled ternary-routed FFT — sparse exact butterfly + correction.

use crate::config::FftLearnConfig;
use crate::distill_model::DistilledFftModel;
use crate::learned_model::FastLearnedFftModel;
use crate::mel::{hann_window, log_mel_from_spectrum_batch, mel_filterbank, ref_log_mel_batch};
use crate::q8::Q8Twiddles;
use crate::reference::{fft_real_batch, max_abs_error};
use crate::ternary_arch::{
    GateLayout, SpectrumCorrection, TernaryArchConfig, all_forward_gates, strip_reverse_gates,
    sync_spec_gates_for_layout,
};
use crate::ternary_gates::{
    GateMode, compute_fraction, gate_mode_counts, hard_gates_from_logits, init_ternary_gates,
    init_ternary_logits, logits_from_gates, ternary_forward_real_batch,
    ternary_forward_real_batch_soft, ternary_logits_from_teacher,
};
use crate::twiddle::exact_twiddles;
use crate::welch::{WelchParams, average_welch_psd, welch_windowed_segments};
use anyhow::{Result, ensure};
/// Student with ternary-routed exact butterfly + distilled correction.
///
/// Routes each butterfly as skip (0), forward (+1), or reverse (−1) to trade
/// compute, latency, and precision against the teacher / reference.
#[derive(Debug, Clone)]
pub struct DistilledTernaryFftModel {
    pub n_fft: usize,
    pub n_mels: usize,
    pub sample_rate: f32,
    pub twiddles: Vec<f32>,
    pub gates: Vec<i8>,
    /// Spectrum-path gates (`DualMelSpec`: all-forward; else mirrors `gates`).
    pub spec_gates: Vec<i8>,
    pub gate_layout: GateLayout,
    pub gate_logits: Vec<[f32; 3]>,
    pub freq_mask: Vec<f32>,
    pub denoiser: SpectrumCorrection,
    pub mel_denoiser: SpectrumCorrection,
    mel_filters: Vec<f32>,
}

impl DistilledTernaryFftModel {
    pub fn new(n_fft: usize, n_mels: usize, sample_rate: f32) -> Self {
        let cfg = FftLearnConfig::new(n_fft, 1).expect("n_fft");
        Self {
            n_fft,
            n_mels,
            sample_rate,
            twiddles: exact_twiddles(&cfg),
            gates: init_ternary_gates(n_fft),
            spec_gates: init_ternary_gates(n_fft),
            gate_layout: GateLayout::SingleSparse,
            gate_logits: init_ternary_logits(n_fft),
            freq_mask: vec![1.0; n_fft * 2],
            denoiser: SpectrumCorrection::from_kind(
                crate::ternary_arch::CorrectorKind::BandWide,
                n_fft,
                false,
            ),
            mel_denoiser: SpectrumCorrection::from_kind(
                crate::ternary_arch::CorrectorKind::BandWide,
                n_fft,
                true,
            ),
            mel_filters: mel_filterbank(n_fft, n_mels, sample_rate),
        }
    }

    pub fn from_teacher(teacher: &FastLearnedFftModel) -> Self {
        // Start all-forward; sparsity is learned after correction converges.
        let gates = init_ternary_gates(teacher.n_fft);
        let spec_gates = gates.clone();
        let cfg = FftLearnConfig::new(teacher.n_fft, 1).expect("n_fft");
        Self {
            n_fft: teacher.n_fft,
            n_mels: teacher.n_mels,
            sample_rate: teacher.sample_rate,
            twiddles: exact_twiddles(&cfg),
            gates,
            spec_gates,
            gate_layout: GateLayout::SingleSparse,
            gate_logits: ternary_logits_from_teacher(teacher),
            freq_mask: teacher.freq_mask.clone(),
            denoiser: SpectrumCorrection::from_kind(
                crate::ternary_arch::CorrectorKind::BandWide,
                teacher.n_fft,
                false,
            ),
            mel_denoiser: SpectrumCorrection::from_kind(
                crate::ternary_arch::CorrectorKind::BandWide,
                teacher.n_fft,
                true,
            ),
            mel_filters: teacher.mel_filters().to_vec(),
        }
    }

    pub fn from_distilled(base: &DistilledFftModel, teacher: &FastLearnedFftModel) -> Self {
        let mut m = Self::from_teacher(teacher);
        let _ = base;
        m.mel_filters = base.mel_filters().to_vec();
        m.freq_mask = vec![1.0; teacher.n_fft * 2];
        m
    }

    pub fn apply_arch_config(&mut self, arch: &TernaryArchConfig) {
        self.gate_layout = arch.gate_layout;
        self.denoiser = SpectrumCorrection::from_kind(arch.corrector, self.n_fft, false);
        self.mel_denoiser = SpectrumCorrection::from_kind(arch.corrector, self.n_fft, true);
        if arch.gate_layout == GateLayout::AllForward {
            self.gates = all_forward_gates(self.n_fft);
        }
        self.spec_gates = all_forward_gates(self.n_fft);
        sync_spec_gates_for_layout(self.gate_layout, &self.gates, &mut self.spec_gates);
        if !arch.allow_reverse {
            strip_reverse_gates(&mut self.gates);
            strip_reverse_gates(&mut self.spec_gates);
        }
        self.gate_logits = logits_from_gates(&self.gates);
    }

    pub fn mel_gates(&self) -> &[i8] {
        &self.gates
    }

    pub fn spec_gates_slice(&self) -> &[i8] {
        if self.gate_layout == GateLayout::DualMelSpec {
            &self.spec_gates
        } else {
            &self.gates
        }
    }

    pub fn sync_spec_gates(&mut self) {
        sync_spec_gates_for_layout(self.gate_layout, &self.gates, &mut self.spec_gates);
    }

    pub fn mel_filters(&self) -> &[f32] {
        &self.mel_filters
    }

    pub fn compute_fraction(&self) -> f32 {
        compute_fraction(&self.gates)
    }

    pub fn gate_counts(&self) -> (usize, usize, usize) {
        gate_mode_counts(&self.gates)
    }

    pub fn sync_gates_from_logits(&mut self) {
        self.gates = hard_gates_from_logits(&self.gate_logits);
    }

    /// Reset affine correction for the current gate layout (denoiser + freq mask).
    pub fn reset_correction_for_gates(&mut self) {
        self.denoiser = SpectrumCorrection::from_kind(
            crate::ternary_arch::CorrectorKind::BandWide,
            self.n_fft,
            false,
        );
        self.mel_denoiser = SpectrumCorrection::from_kind(
            crate::ternary_arch::CorrectorKind::BandWide,
            self.n_fft,
            true,
        );
        self.freq_mask.fill(1.0);
    }

    /// Max-abs spectrum error vs reference FFT on raw (unwindowed) batch.
    pub fn spec_err_vs_ref(&self, signal: &[f32], batch: usize) -> Result<f32> {
        let pred = self.spectrum_batch_raw(signal, batch)?;
        let ref_spec = fft_real_batch(signal, batch, self.n_fft)?;
        Ok(max_abs_error(&pred, &ref_spec))
    }

    /// Micro refit after a gate change; returns post-fit spectrum error vs reference.
    pub fn refit_correction_quick(
        &mut self,
        signal: &[f32],
        batch: usize,
        steps: usize,
        lr: f32,
    ) -> Result<f32> {
        self.reset_correction_for_gates();
        self.refit_correction_incremental(&[signal], batch, steps, lr)
    }

    /// Continue band correction for the current gate layout (raw + windowed mel paths).
    pub fn refit_correction_incremental(
        &mut self,
        signals: &[&[f32]],
        batch: usize,
        steps: usize,
        lr: f32,
    ) -> Result<f32> {
        ensure!(!signals.is_empty());
        let window = hann_window(self.n_fft);
        for _ in 0..steps {
            for signal in signals {
                self.train_step_ref_spectrum(signal, batch, lr)?;
                let mut windowed = signal.to_vec();
                for b in 0..batch {
                    for i in 0..self.n_fft {
                        windowed[b * self.n_fft + i] *= window[i];
                    }
                }
                self.train_step_mel_ref_spectrum(&windowed, batch, lr * 0.85)?;
            }
        }
        signals
            .iter()
            .map(|signal| self.spec_err_vs_ref(signal, batch))
            .try_fold(0.0f32, |acc, err| err.map(|e| acc.max(e)))
    }

    fn forward_masked_with_gates(
        &self,
        signal: &[f32],
        gates: &[i8],
        batch: usize,
    ) -> Result<Vec<f32>> {
        ensure!(signal.len() == batch * self.n_fft);
        let mut spec =
            ternary_forward_real_batch(signal, &self.twiddles, gates, batch, self.n_fft)?;
        for b in 0..batch {
            for i in 0..self.n_fft * 2 {
                let idx = b * self.n_fft * 2 + i;
                spec[idx] *= self.freq_mask[i];
            }
        }
        Ok(spec)
    }

    fn forward_masked_mel(&self, signal: &[f32], batch: usize) -> Result<Vec<f32>> {
        self.forward_masked_with_gates(signal, self.mel_gates(), batch)
    }

    fn forward_masked_spec(&self, signal: &[f32], batch: usize) -> Result<Vec<f32>> {
        self.forward_masked_with_gates(signal, self.spec_gates_slice(), batch)
    }

    /// Windowed spectrum for mel-style paths (sparse gates).
    pub fn spectrum_batch(&self, windowed: &[f32], batch: usize) -> Result<Vec<f32>> {
        ensure!(windowed.len() == batch * self.n_fft);
        let spec = self.forward_masked_mel(windowed, batch)?;
        self.mel_denoiser.apply_batch(&spec, batch, self.n_fft)
    }

    /// Raw spectrum for denoise/q8 benches (spec gates + correction).
    pub fn spectrum_batch_raw(&self, signal: &[f32], batch: usize) -> Result<Vec<f32>> {
        let spec = self.forward_masked_spec(signal, batch)?;
        self.denoiser.apply_batch(&spec, batch, self.n_fft)
    }

    /// Windowed spectrum with spec-path gates (compiled spectrum parity).
    pub fn spectrum_batch_accurate(&self, windowed: &[f32], batch: usize) -> Result<Vec<f32>> {
        ensure!(windowed.len() == batch * self.n_fft);
        let spec = self.forward_masked_spec(windowed, batch)?;
        self.denoiser.apply_batch(&spec, batch, self.n_fft)
    }

    /// Ternary butterfly output after freq mask, before denoiser (mel gates).
    #[allow(dead_code)]
    pub(crate) fn butterfly_spectrum_masked(
        &self,
        signal: &[f32],
        batch: usize,
    ) -> Result<Vec<f32>> {
        self.forward_masked_mel(signal, batch)
    }

    /// Direct MSE correction: affine denoiser bridges ternary butterfly → reference FFT.
    pub fn train_step_ref_spectrum(
        &mut self,
        signal: &[f32],
        batch: usize,
        lr: f32,
    ) -> Result<f32> {
        self.train_step_spectrum_target(signal, batch, lr, fft_real_batch)
    }

    /// Align denoiser output to Q8-quantized butterfly spectrum (q8 bench pipeline).
    pub fn train_step_q8_spectrum(&mut self, signal: &[f32], batch: usize, lr: f32) -> Result<f32> {
        let q8 = Q8Twiddles::from_f32(&self.twiddles);
        self.train_step_spectrum_target(signal, batch, lr, move |sig, b, n| {
            q8.forward_real_batch(sig, b, n)
        })
    }

    /// Train sparse-gate mel denoiser toward reference FFT (mel path correction).
    pub fn train_step_mel_ref_spectrum(
        &mut self,
        signal: &[f32],
        batch: usize,
        lr: f32,
    ) -> Result<f32> {
        ensure!(signal.len() == batch * self.n_fft);
        let sparse = self.forward_masked_mel(signal, batch)?;
        let ref_spec = fft_real_batch(signal, batch, self.n_fft)?;
        let pred = self.mel_denoiser.apply_batch(&sparse, batch, self.n_fft)?;
        let err = max_abs_error(&pred, &ref_spec);
        self.mel_denoiser
            .train_step_mse(&sparse, &ref_spec, batch, self.n_fft, lr)?;
        Ok(err)
    }

    fn train_step_spectrum_target(
        &mut self,
        signal: &[f32],
        batch: usize,
        lr: f32,
        target: impl FnOnce(&[f32], usize, usize) -> Result<Vec<f32>>,
    ) -> Result<f32> {
        ensure!(signal.len() == batch * self.n_fft);
        let masked = self.forward_masked_spec(signal, batch)?;
        let ref_spec = target(signal, batch, self.n_fft)?;
        let pred = self.denoiser.apply_batch(&masked, batch, self.n_fft)?;
        let err = max_abs_error(&pred, &ref_spec);
        self.denoiser
            .train_step_mse(&masked, &ref_spec, batch, self.n_fft, lr)?;
        Ok(err)
    }

    pub fn log_mel_batch(&self, signal: &[f32], batch: usize) -> Result<Vec<f32>> {
        let window = hann_window(self.n_fft);
        let mut windowed = signal.to_vec();
        for b in 0..batch {
            for i in 0..self.n_fft {
                windowed[b * self.n_fft + i] *= window[i];
            }
        }
        let spec = self.spectrum_batch(&windowed, batch)?;
        log_mel_from_spectrum_batch(&spec, &self.mel_filters, batch, self.n_fft, self.n_mels)
    }

    pub fn welch_psd_batch(
        &self,
        signal: &[f32],
        batch: usize,
        params: WelchParams,
    ) -> Result<Vec<f32>> {
        ensure!(params.n_fft == self.n_fft);
        let window = crate::welch::hann_window(self.n_fft);
        let segs = welch_windowed_segments(signal, batch, params, &window)?;
        let n_segs = batch * params.n_segments;
        let mut spec = ternary_forward_real_batch(
            &segs,
            &self.twiddles,
            self.spec_gates_slice(),
            n_segs,
            self.n_fft,
        )?;
        for seg in 0..n_segs {
            for i in 0..self.n_fft * 2 {
                let idx = seg * self.n_fft * 2 + i;
                spec[idx] *= self.freq_mask[i];
            }
        }
        let spec = self.denoiser.apply_batch(&spec, n_segs, self.n_fft)?;
        Ok(average_welch_psd(
            &spec,
            batch,
            params.n_segments,
            self.n_fft,
        ))
    }

    pub fn welch_peaks_batch(
        &self,
        signal: &[f32],
        batch: usize,
        params: crate::peak::WelchPeakParams,
    ) -> Result<Vec<f32>> {
        ensure!(params.welch.n_fft == self.n_fft);
        let window = crate::welch::hann_window(self.n_fft);
        let segs = welch_windowed_segments(signal, batch, params.welch, &window)?;
        let n_segs = batch * params.welch.n_segments;
        let mut spec = ternary_forward_real_batch(
            &segs,
            &self.twiddles,
            self.spec_gates_slice(),
            n_segs,
            self.n_fft,
        )?;
        for seg in 0..n_segs {
            for i in 0..self.n_fft * 2 {
                let idx = seg * self.n_fft * 2 + i;
                spec[idx] *= self.freq_mask[i];
            }
        }
        let spec = self.denoiser.apply_batch(&spec, n_segs, self.n_fft)?;
        Ok(crate::peak::welch_peaks_from_segment_spectrum(
            &spec, batch, params,
        ))
    }

    pub fn train_step_mel(
        &mut self,
        signal: &[f32],
        target_mel: &[f32],
        batch: usize,
        lr: f32,
    ) -> Result<f32> {
        let window = hann_window(self.n_fft);
        let mut windowed = signal.to_vec();
        for b in 0..batch {
            for i in 0..self.n_fft {
                windowed[b * self.n_fft + i] *= window[i];
            }
        }
        let spec = self.spectrum_batch(&windowed, batch)?;
        let pred =
            log_mel_from_spectrum_batch(&spec, &self.mel_filters, batch, self.n_fft, self.n_mels)?;
        let err = max_abs_error(&pred, target_mel);
        let grad = crate::mel::log_mel_loss_grad_wrt_spectrum(
            &pred,
            target_mel,
            &spec,
            &self.mel_filters,
            batch,
            self.n_fft,
            self.n_mels,
        );
        let masked = self.forward_masked_mel(&windowed, batch)?;
        self.mel_denoiser
            .train_step_spectrum_grad(&masked, &grad, batch, self.n_fft, lr);
        let n = (batch * self.n_fft * 2) as f32;
        for i in 0..self.n_fft * 2 {
            let mut gm = 0f32;
            for b in 0..batch {
                gm += grad[b * self.n_fft * 2 + i] * masked[b * self.n_fft * 2 + i];
            }
            self.freq_mask[i] -= lr * 0.05 * gm / n.max(1.0);
            self.freq_mask[i] = self.freq_mask[i].clamp(0.0, 4.0);
        }
        Ok(err)
    }

    pub fn train_step_gate_logits(
        &mut self,
        signal: &[f32],
        target_mel: &[f32],
        batch: usize,
        lr: f32,
        temp: f32,
        compute_weight: f32,
        fd_samples: usize,
        seed: u64,
    ) -> Result<f32> {
        let window = hann_window(self.n_fft);
        let mut windowed = signal.to_vec();
        for b in 0..batch {
            for i in 0..self.n_fft {
                windowed[b * self.n_fft + i] *= window[i];
            }
        }
        let spec = ternary_forward_real_batch_soft(
            &windowed,
            &self.twiddles,
            &self.gate_logits,
            batch,
            self.n_fft,
            temp,
        )?;
        let mut corrected = spec.clone();
        for b in 0..batch {
            for i in 0..self.n_fft * 2 {
                let idx = b * self.n_fft * 2 + i;
                corrected[idx] *= self.freq_mask[i];
            }
        }
        corrected = self
            .mel_denoiser
            .apply_batch(&corrected, batch, self.n_fft)?;
        let pred = log_mel_from_spectrum_batch(
            &corrected,
            &self.mel_filters,
            batch,
            self.n_fft,
            self.n_mels,
        )?;
        let err = max_abs_error(&pred, target_mel);
        let grad = crate::mel::log_mel_loss_grad_wrt_spectrum(
            &pred,
            target_mel,
            &corrected,
            &self.mel_filters,
            batch,
            self.n_fft,
            self.n_mels,
        );
        let n = (batch * self.n_fft * 2) as f32;
        let mut spec_grad = vec![0f32; batch * self.n_fft * 2];
        if let SpectrumCorrection::Band(ref band) = self.mel_denoiser {
            for b in 0..batch {
                for i in 0..self.n_fft * 2 {
                    let idx = b * self.n_fft * 2 + i;
                    spec_grad[idx] = grad[idx]
                        * band.weights[i * band.band_width + band.radius]
                        * self.freq_mask[i]
                        / n.max(1.0);
                }
            }
        }
        self.gate_logits = finite_diff_gate_logits(
            &windowed,
            &self.twiddles,
            &self.gate_logits,
            &spec_grad,
            batch,
            self.n_fft,
            temp,
            lr,
            compute_weight,
            fd_samples,
            seed,
        );
        if compute_weight > 0.0 {
            self.sync_gates_from_logits();
        }
        Ok(err)
    }

    /// Greedy local search: try flipping a sample of gates to improve mel + compute tradeoff.
    pub fn refine_gates_local(
        &mut self,
        signal: &[f32],
        target_mel: &[f32],
        batch: usize,
        compute_weight: f32,
        sample: usize,
        seed: u64,
    ) -> Result<()> {
        use rand::prelude::*;
        let mut rng = StdRng::seed_from_u64(seed);
        let mut baseline_mel = max_abs_error(&self.log_mel_batch(signal, batch)?, target_mel);
        let mut baseline = baseline_mel + compute_weight * self.compute_fraction();
        let n = self.gates.len();
        let tries = sample.min(n);
        for _ in 0..tries {
            let gi = rng.gen_range(0..n);
            let old = self.gates[gi];
            let candidates = [GateMode::Skip, GateMode::Forward, GateMode::Reverse]
                .map(|m| m.to_i8())
                .into_iter()
                .filter(|&c| c != old);
            let mut best = old;
            let mut best_loss = baseline;
            for cand in candidates {
                self.gates[gi] = cand;
                let mel = max_abs_error(&self.log_mel_batch(signal, batch)?, target_mel);
                let loss = mel + compute_weight * self.compute_fraction();
                if loss < best_loss && mel <= baseline_mel + 0.04 {
                    best_loss = loss;
                    best = cand;
                }
            }
            self.gates[gi] = best;
            if best != old {
                baseline = best_loss;
                baseline_mel = max_abs_error(&self.log_mel_batch(signal, batch)?, target_mel);
            }
        }
        self.gate_logits = logits_from_gates(&self.gates);
        Ok(())
    }

    /// Local gate search with optional spectrum guard after incremental denoiser refit.
    pub fn refine_gates_local_with_spec(
        &mut self,
        signal: &[f32],
        target_mel: &[f32],
        batch: usize,
        compute_weight: f32,
        sample: usize,
        seed: u64,
        max_spec_err: f32,
        extra_signals: &[&[f32]],
    ) -> Result<()> {
        use rand::prelude::*;
        let mut rng = StdRng::seed_from_u64(seed);
        let mut signals: Vec<&[f32]> = vec![signal];
        signals.extend_from_slice(extra_signals);
        let mut baseline_mel = max_abs_error(&self.log_mel_batch(signal, batch)?, target_mel);
        let mut baseline = baseline_mel + compute_weight * self.compute_fraction();
        let n = self.gates.len();
        let tries = sample.min(n);
        for _ in 0..tries {
            let gi = rng.gen_range(0..n);
            let old = self.gates[gi];
            let saved_denoiser = self.denoiser.clone();
            let saved_mask = self.freq_mask.clone();
            let candidates = [GateMode::Skip, GateMode::Forward, GateMode::Reverse]
                .map(|m| m.to_i8())
                .into_iter()
                .filter(|&c| c != old);
            let mut best = old;
            let mut best_loss = baseline;
            for cand in candidates {
                self.gates[gi] = cand;
                if max_spec_err > 0.0 {
                    let spec_err = self.refit_correction_incremental(&signals, batch, 16, 8e-3)?;
                    if spec_err > max_spec_err {
                        self.gates[gi] = old;
                        self.denoiser = saved_denoiser.clone();
                        self.freq_mask = saved_mask.clone();
                        continue;
                    }
                }
                let mel = max_abs_error(&self.log_mel_batch(signal, batch)?, target_mel);
                let loss = mel + compute_weight * self.compute_fraction();
                if loss < best_loss && mel <= baseline_mel + 0.04 {
                    best_loss = loss;
                    best = cand;
                } else {
                    self.gates[gi] = old;
                    self.denoiser = saved_denoiser.clone();
                    self.freq_mask = saved_mask.clone();
                }
            }
            self.gates[gi] = best;
            if best != old {
                let _ = self.refit_correction_incremental(&signals, batch, 12, 8e-3)?;
                baseline = best_loss;
                baseline_mel = max_abs_error(&self.log_mel_batch(signal, batch)?, target_mel);
            } else {
                self.denoiser = saved_denoiser;
                self.freq_mask = saved_mask;
            }
        }
        self.gate_logits = logits_from_gates(&self.gates);
        Ok(())
    }

    #[allow(dead_code)]
    fn mel_objective(
        &self,
        signal: &[f32],
        target_mel: &[f32],
        batch: usize,
        compute_weight: f32,
    ) -> Result<f32> {
        let pred = self.log_mel_batch(signal, batch)?;
        let mel = max_abs_error(&pred, target_mel);
        Ok(mel + compute_weight * self.compute_fraction())
    }

    /// Greedy gate pruning until `compute_fraction <= target` (later stages first).
    pub fn prune_gates_to_target(
        &mut self,
        signal: &[f32],
        target_mel: &[f32],
        batch: usize,
        target: f32,
        max_mel_err: f32,
    ) -> Result<()> {
        self.prune_gates_to_target_with_ref(signal, target_mel, None, batch, target, max_mel_err)?;
        Ok(())
    }

    /// Like [`prune_gates_to_target`] but also guards mel-vs-reference when `ref_mel` is set.
    pub fn prune_gates_to_target_with_ref(
        &mut self,
        signal: &[f32],
        target_mel: &[f32],
        ref_mel: Option<&[f32]>,
        batch: usize,
        target: f32,
        max_mel_err: f32,
    ) -> Result<()> {
        self.prune_gates_to_target_with_ref_impl(
            signal,
            target_mel,
            ref_mel,
            batch,
            target,
            max_mel_err,
            0.12,
        )
    }

    /// Prune with mel + optional ref-mel guards and a raw-spectrum guard after quick denoiser refit.
    pub fn prune_gates_to_target_with_ref_and_spec(
        &mut self,
        signal: &[f32],
        target_mel: &[f32],
        ref_mel: Option<&[f32]>,
        batch: usize,
        target: f32,
        max_mel_err: f32,
        max_spec_err: f32,
    ) -> Result<()> {
        self.prune_gates_to_target_with_ref_impl(
            signal,
            target_mel,
            ref_mel,
            batch,
            target,
            max_mel_err,
            max_spec_err,
        )
    }

    fn mel_max_err(&self, signals: &[&[f32]], targets: &[&[f32]], batch: usize) -> Result<f32> {
        ensure!(signals.len() == targets.len());
        signals
            .iter()
            .zip(targets.iter())
            .map(|(sig, target)| Ok(max_abs_error(&self.log_mel_batch(sig, batch)?, target)))
            .try_fold(0.0f32, |acc, e| e.map(|v| acc.max(v)))
    }

    fn prune_gates_to_target_with_ref_impl(
        &mut self,
        signal: &[f32],
        target_mel: &[f32],
        ref_mel: Option<&[f32]>,
        batch: usize,
        target: f32,
        max_mel_err: f32,
        max_spec_err: f32,
    ) -> Result<()> {
        use rand::prelude::*;
        let half = self.n_fft / 2;
        let stages = self.n_fft.trailing_zeros() as usize;
        let mut order: Vec<usize> = (0..self.gates.len()).collect();
        order.sort_by_key(|&gi| {
            let stage = gi / half;
            std::cmp::Reverse(stage * 10_000 + gi)
        });

        let mut bench_rng = StdRng::seed_from_u64(42);
        let bench_signal = crate::train::random_batch(&mut bench_rng, batch, self.n_fft);
        let prune_signals: [&[f32]; 2] = [signal, &bench_signal];
        let bench_ref_mel = ref_mel_for_ternary(
            &bench_signal,
            batch,
            self.n_fft,
            self.n_mels,
            self.sample_rate,
        )?;
        let mel_targets: [&[f32]; 2] = [target_mel, &bench_ref_mel];
        let baseline = self.mel_max_err(&prune_signals, &mel_targets, batch)?;
        let baseline_ref = if let Some(r) = ref_mel {
            self.mel_max_err(&prune_signals, &[r, r], batch)?
        } else {
            0.0
        };
        let baseline_spec = if max_spec_err > 0.0 {
            self.refit_correction_incremental(&prune_signals, batch, 24, 6e-3)?
        } else {
            0.0
        };
        let mut best_err = baseline;
        let mut best_ref = baseline_ref;
        let mut best_spec = baseline_spec;
        let mut pruned: Vec<usize> = Vec::new();
        let mel_slack = (baseline * 0.35 + 0.03).clamp(0.02, 0.15);
        let ref_slack = if ref_mel.is_some() {
            (baseline_ref * 0.35 + 0.03).clamp(0.02, 0.15)
        } else {
            0.0
        };
        let hard_max = (baseline + mel_slack).min(max_mel_err.max(baseline + 0.12));
        let hard_ref =
            ref_mel.map(|_| (baseline_ref + ref_slack).min(max_mel_err.max(baseline_ref + 0.15)));
        let spec_slack = if max_spec_err > 0.0 {
            (baseline_spec * 0.25 + 0.01).clamp(0.005, 0.04)
        } else {
            0.0
        };
        let hard_spec = if max_spec_err > 0.0 {
            (baseline_spec + spec_slack).min(max_spec_err.max(baseline_spec + 0.05))
        } else {
            0.0
        };

        for &gi in &order {
            if self.compute_fraction() <= target {
                break;
            }
            if self.gates[gi] == GateMode::Forward.to_i8()
                || self.gates[gi] == GateMode::Reverse.to_i8()
            {
                let old = self.gates[gi];
                let saved_denoiser = self.denoiser.clone();
                let saved_mel = self.mel_denoiser.clone();
                let saved_mask = self.freq_mask.clone();
                self.gates[gi] = GateMode::Skip.to_i8();
                let refit_steps = (64 + pruned.len() * 2).min(200);
                let refit_lr = 1.2e-2;
                let spec_err = if max_spec_err > 0.0 {
                    self.refit_correction_incremental(&prune_signals, batch, refit_steps, refit_lr)?
                } else {
                    0.0
                };
                let err = self.mel_max_err(&prune_signals, &mel_targets, batch)?;
                let err_ref = if let Some(ref_mel) = ref_mel {
                    self.mel_max_err(&prune_signals, &[ref_mel, ref_mel], batch)?
                } else {
                    0.0
                };
                let ref_ok =
                    hard_ref.is_none_or(|hr| err_ref <= hr && err_ref <= best_ref + ref_slack);
                // Greedy prune prioritizes mel; spectrum is recovered in post-prune refit.
                let spec_ok = max_spec_err <= 0.0
                    || spec_err <= hard_spec.max(2.0)
                        && spec_err <= best_spec + spec_slack.max(0.5);
                if err <= hard_max && err <= best_err + mel_slack && ref_ok && spec_ok {
                    best_err = err;
                    best_ref = err_ref;
                    best_spec = spec_err;
                    pruned.push(gi);
                    let mel_ref_steps = (24 + pruned.len() / 4).min(96);
                    for _ in 0..mel_ref_steps {
                        for (i, sig) in prune_signals.iter().enumerate() {
                            self.train_step_mel(sig, mel_targets[i], batch, 8e-3)?;
                            if let Some(r) = ref_mel {
                                self.train_step_mel(sig, r, batch, 6e-3)?;
                            }
                            self.train_step_mel_ref_spectrum(sig, batch, 5e-3)?;
                        }
                    }
                    if max_spec_err > 0.0 && pruned.len().is_multiple_of(12) {
                        best_spec =
                            self.refit_correction_incremental(&prune_signals, batch, 36, 8e-3)?;
                    }
                } else {
                    self.gates[gi] = old;
                    self.denoiser = saved_denoiser;
                    self.mel_denoiser = saved_mel;
                    self.freq_mask = saved_mask;
                }
            }
        }

        let accepted_skips = pruned.clone();
        let _ = self.refit_correction_incremental(&prune_signals, batch, 64, 8e-3)?;
        for _ in 0..32 {
            for (i, sig) in prune_signals.iter().enumerate() {
                self.train_step_mel(sig, mel_targets[i], batch, 8e-3)?;
                if let Some(r) = ref_mel {
                    self.train_step_mel(sig, r, batch, 6e-3)?;
                }
            }
        }
        let mut active_mel: Vec<usize> = accepted_skips
            .iter()
            .copied()
            .filter(|&gi| self.gates[gi] == GateMode::Skip.to_i8())
            .collect();
        let mut final_err = self.mel_max_err(&prune_signals, &mel_targets, batch)?;
        let mut final_ref = if let Some(r) = ref_mel {
            self.mel_max_err(&prune_signals, &[r, r], batch)?
        } else {
            0.0
        };
        while (final_err > hard_max || hard_ref.is_some_and(|hr| final_ref > hr))
            && !active_mel.is_empty()
        {
            let gi = active_mel.pop().expect("active_mel");
            self.gates[gi] = GateMode::Forward.to_i8();
            let _ = self.refit_correction_incremental(&prune_signals, batch, 32, 8e-3)?;
            final_err = self.mel_max_err(&prune_signals, &mel_targets, batch)?;
            final_ref = if let Some(r) = ref_mel {
                self.mel_max_err(&prune_signals, &[r, r], batch)?
            } else {
                0.0
            };
        }

        if max_spec_err > 0.0 {
            let mut active: Vec<usize> = accepted_skips
                .iter()
                .copied()
                .filter(|&gi| self.gates[gi] == GateMode::Skip.to_i8())
                .collect();
            let mut spec = self.refit_correction_incremental(&prune_signals, batch, 320, 1.2e-2)?;
            while spec > max_spec_err && !active.is_empty() {
                let gi = active.pop().expect("active");
                self.gates[gi] = GateMode::Forward.to_i8();
                spec = self.refit_correction_incremental(&prune_signals, batch, 120, 1e-2)?;
            }
        }

        self.sync_spec_gates();
        self.gate_logits = logits_from_gates(&self.gates);
        let _ = stages;
        Ok(())
    }
}

fn finite_diff_gate_logits(
    windowed: &[f32],
    twiddles: &[f32],
    logits: &[[f32; 3]],
    spec_grad: &[f32],
    batch: usize,
    n_fft: usize,
    temp: f32,
    lr: f32,
    compute_weight: f32,
    fd_samples: usize,
    seed: u64,
) -> Vec<[f32; 3]> {
    use rand::prelude::*;
    let eps = 5e-3;
    let mut out = logits.to_vec();
    let n = logits.len();
    if n == 0 {
        return out;
    }
    let mut rng = StdRng::seed_from_u64(seed);
    let samples = fd_samples.min(n).max(1);
    let mut picked = std::collections::HashSet::new();
    while picked.len() < samples {
        picked.insert(rng.gen_range(0..n));
    }
    for gi in picked {
        for k in 0..3 {
            let mut plus = logits.to_vec();
            plus[gi][k] += eps;
            let mut minus = logits.to_vec();
            minus[gi][k] -= eps;
            let fp = soft_mel_proxy(windowed, twiddles, &plus, spec_grad, batch, n_fft, temp);
            let fm = soft_mel_proxy(windowed, twiddles, &minus, spec_grad, batch, n_fft, temp);
            let grad = (fp - fm) / (2.0 * eps);
            let compute_prior = match k {
                0 => -compute_weight,
                1 => compute_weight,
                2 => compute_weight * 0.5,
                _ => 0.0,
            };
            let delta = (lr * (grad + compute_prior)).clamp(-0.25, 0.25);
            out[gi][k] -= delta;
        }
    }
    out
}

fn soft_mel_proxy(
    windowed: &[f32],
    twiddles: &[f32],
    logits: &[[f32; 3]],
    spec_grad: &[f32],
    batch: usize,
    n_fft: usize,
    temp: f32,
) -> f32 {
    ternary_forward_real_batch_soft(windowed, twiddles, logits, batch, n_fft, temp)
        .map(|spec| {
            spec.iter()
                .zip(spec_grad.iter())
                .map(|(s, g)| s * g)
                .sum::<f32>()
        })
        .unwrap_or(0.0)
}

pub fn ref_mel_for_ternary(
    signal: &[f32],
    batch: usize,
    n_fft: usize,
    n_mels: usize,
    sr: f32,
) -> Result<Vec<f32>> {
    let window = hann_window(n_fft);
    let mut windowed = signal.to_vec();
    for b in 0..batch {
        for i in 0..n_fft {
            windowed[b * n_fft + i] *= window[i];
        }
    }
    ref_log_mel_batch(&windowed, batch, n_fft, n_mels, sr)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mel::hann_window;

    #[test]
    fn e2e_denoise_alignment_short_train() {
        use crate::q8::Q8Twiddles;
        use crate::train::random_batch;
        use crate::train_distill::{DistillTrainConfig, distill_from_teacher};
        use crate::train_distill_ternary::{
            DistillTernaryTrainConfig, distill_ternary_from_distilled,
        };
        use crate::train_e2e::{E2eTrainConfig, train_fast_learned_model};
        use rand::prelude::*;

        let batch = 8;
        let n_fft = 128;
        let (teacher, _) = train_fast_learned_model(&E2eTrainConfig {
            n_fft,
            batch,
            steps: 100,
            seed: 42,
            ..E2eTrainConfig::default()
        })
        .unwrap();
        let (base, _) = distill_from_teacher(
            &teacher,
            &DistillTrainConfig {
                n_fft,
                batch,
                steps: 100,
                seed: 43,
                ..DistillTrainConfig::default()
            },
        )
        .unwrap();
        let (student, _) = distill_ternary_from_distilled(
            &base,
            &teacher,
            &DistillTernaryTrainConfig {
                n_fft,
                batch,
                steps: 100,
                seed: 44,
                target_compute_fraction: 1.0,
                post_prune_ref_steps: 0,
                post_prune_mel_steps: 0,
                ..DistillTernaryTrainConfig::default()
            },
        )
        .unwrap();

        let mut rng = StdRng::seed_from_u64(42);
        let signal = random_batch(&mut rng, batch, n_fft);
        let ref_spec = fft_real_batch(&signal, batch, n_fft).unwrap();
        let pred = student.spectrum_batch_raw(&signal, batch).unwrap();
        let denoise_err = max_abs_error(&pred, &ref_spec);
        let cfg = FftLearnConfig::new(n_fft, batch).unwrap();
        let q8 = Q8Twiddles::from_f32(&exact_twiddles(&cfg));
        let q8_err = max_abs_error(
            &pred,
            &q8.forward_real_batch(&signal, batch, n_fft).unwrap(),
        );
        eprintln!("e2e_short denoise_err={denoise_err} q8_err={q8_err}");
        assert!(denoise_err < 0.05, "denoise_err={denoise_err}");
        assert!(q8_err < 0.2, "q8_err={q8_err}");
    }

    #[test]
    fn denoise_and_q8_align_after_distilled_warm_start() {
        use crate::config::FftLearnConfig;

        use crate::q8::Q8Twiddles;
        use crate::train::random_batch;
        use crate::train_distill::{DistillTrainConfig, distill_from_teacher};
        use crate::train_e2e::{E2eTrainConfig, train_fast_learned_model};
        use rand::prelude::*;

        let teacher_cfg = E2eTrainConfig {
            n_fft: 128,
            batch: 8,
            steps: 80,
            seed: 42,
            ..E2eTrainConfig::default()
        };
        let (teacher, _) = train_fast_learned_model(&teacher_cfg).unwrap();
        let distill_cfg = DistillTrainConfig {
            n_fft: 128,
            batch: 8,
            steps: 80,
            seed: 43,
            ..DistillTrainConfig::default()
        };
        let (base, _) = distill_from_teacher(&teacher, &distill_cfg).unwrap();
        let mut student = DistilledTernaryFftModel::from_distilled(&base, &teacher);

        let mut rng = StdRng::seed_from_u64(42);
        let signal = random_batch(&mut rng, 8, 128);
        let w = hann_window(128);
        let mut windowed = signal.clone();
        for b in 0..8 {
            for i in 0..128 {
                windowed[b * 128 + i] *= w[i];
            }
        }
        for _ in 0..150 {
            student.train_step_ref_spectrum(&windowed, 8, 8e-3).unwrap();
            student.train_step_ref_spectrum(&signal, 8, 6e-3).unwrap();
            student.train_step_q8_spectrum(&signal, 8, 5e-3).unwrap();
        }

        let pred = student.spectrum_batch_raw(&signal, 8).unwrap();
        let denoise_err = max_abs_error(&pred, &fft_real_batch(&signal, 8, 128).unwrap());
        let cfg = FftLearnConfig::new(128, 8).unwrap();
        let q8 = Q8Twiddles::from_f32(&exact_twiddles(&cfg));
        let q8_err = max_abs_error(&pred, &q8.forward_real_batch(&signal, 8, 128).unwrap());
        assert!(denoise_err < 0.2, "denoise_err={denoise_err}");
        assert!(q8_err < 0.2, "q8_err={q8_err}");
    }

    #[test]
    fn raw_spectrum_aligns_after_ref_train_e2e_seed() {
        use crate::train::random_batch;
        use rand::prelude::*;
        let mut rng = StdRng::seed_from_u64(42);
        let batch = 8;
        let n_fft = 128;
        let signal = random_batch(&mut rng, batch, n_fft);
        let mut model = DistilledTernaryFftModel::new(n_fft, 40, 16_000.0);
        let w = hann_window(n_fft);
        let mut windowed = signal.clone();
        for b in 0..batch {
            for i in 0..n_fft {
                windowed[b * n_fft + i] *= w[i];
            }
        }
        for _ in 0..200 {
            model
                .train_step_ref_spectrum(&windowed, batch, 8e-3)
                .unwrap();
            model.train_step_ref_spectrum(&signal, batch, 6e-3).unwrap();
        }
        let pred = model.spectrum_batch_raw(&signal, batch).unwrap();
        let ref_spec = fft_real_batch(&signal, batch, n_fft).unwrap();
        let err = max_abs_error(&pred, &ref_spec);
        assert!(err < 0.05, "raw spec err={err}");
    }

    #[test]
    fn ref_spectrum_training_aligns_mel() {
        let mut model = DistilledTernaryFftModel::new(128, 40, 16_000.0);
        let batch = 8;
        let signal: Vec<f32> = (0..batch * 128).map(|i| (i as f32 * 0.01).sin()).collect();
        let w = hann_window(128);
        let mut windowed = signal.clone();
        for b in 0..batch {
            for i in 0..128 {
                windowed[b * 128 + i] *= w[i];
            }
        }
        for _ in 0..120 {
            model
                .train_step_ref_spectrum(&windowed, batch, 8e-3)
                .unwrap();
        }
        let spec_err = model
            .train_step_ref_spectrum(&windowed, batch, 0.0)
            .unwrap();
        let pred_spec = model.spectrum_batch(&windowed, batch).unwrap();
        let ref_spec = fft_real_batch(&windowed, batch, 128).unwrap();
        let spec_path_err = max_abs_error(&pred_spec, &ref_spec);
        let pred_mel_direct =
            log_mel_from_spectrum_batch(&pred_spec, model.mel_filters(), batch, 128, 40).unwrap();
        let pred_mel = model.log_mel_batch(&signal, batch).unwrap();
        let log_mel_path_err = max_abs_error(&pred_mel, &pred_mel_direct);
        let ref_mel = ref_log_mel_batch(&windowed, batch, 128, 40, 16_000.0).unwrap();
        let ref_mel_same_filters =
            log_mel_from_spectrum_batch(&ref_spec, model.mel_filters(), batch, 128, 40).unwrap();
        let mel_err = max_abs_error(&pred_mel, &ref_mel);
        let filter_err = max_abs_error(&ref_mel_same_filters, &ref_mel);
        let mel_self_err = max_abs_error(&pred_mel_direct, &ref_mel_same_filters);
        assert!(spec_err < 1e-3, "spec_err={spec_err}");
        assert!(
            spec_path_err < 1e-3,
            "spec_path_err={spec_path_err} spec_err={spec_err}"
        );
        assert!(
            log_mel_path_err < 1e-4,
            "log_mel_path_err={log_mel_path_err}"
        );
        assert!(
            mel_self_err < 0.01,
            "mel_self_err={mel_self_err} filter_err={filter_err} mel_err={mel_err} spec_path_err={spec_path_err}"
        );
    }

    #[test]
    #[ignore = "slow integration; run with --ignored"]
    fn prune_band_corrector_accepts_skips() {
        use crate::train::random_batch;
        use crate::train_distill::{DistillTrainConfig, distill_from_teacher};
        use crate::train_e2e::{E2eTrainConfig, train_fast_learned_model};
        use rand::prelude::*;
        let teacher_cfg = E2eTrainConfig {
            n_fft: 128,
            batch: 8,
            steps: 80,
            seed: 42,
            ..E2eTrainConfig::default()
        };
        let (teacher, _) = train_fast_learned_model(&teacher_cfg).unwrap();
        let distill_cfg = DistillTrainConfig {
            n_fft: 128,
            batch: 8,
            steps: 80,
            seed: 43,
            ..DistillTrainConfig::default()
        };
        let (base, _) = distill_from_teacher(&teacher, &distill_cfg).unwrap();
        let mut student = DistilledTernaryFftModel::from_distilled(&base, &teacher);
        let mut rng = StdRng::seed_from_u64(42);
        let signal = random_batch(&mut rng, 8, 128);
        let w = hann_window(128);
        let mut windowed = signal.clone();
        for b in 0..8 {
            for i in 0..128 {
                windowed[b * 128 + i] *= w[i];
            }
        }
        for _ in 0..300 {
            student.train_step_ref_spectrum(&windowed, 8, 8e-3).unwrap();
            student.train_step_ref_spectrum(&signal, 8, 6e-3).unwrap();
            student
                .train_step_mel_ref_spectrum(&windowed, 8, 7e-3)
                .unwrap();
        }
        let teacher_mel = crate::distill_model::teacher_mel_batch(&teacher, &signal, 8).unwrap();
        let ref_mel = ref_mel_for_ternary(&signal, 8, 128, 40, 16_000.0).unwrap();
        student
            .prune_gates_to_target_with_ref_and_spec(
                &signal,
                &teacher_mel,
                Some(&ref_mel),
                8,
                0.72,
                0.28,
                0.12,
            )
            .unwrap();
        let (skip, _, _) = student.gate_counts();
        let spec_after = student.spec_err_vs_ref(&signal, 8).unwrap();
        assert!(skip > 8, "skip={skip}");
        assert!(
            student.compute_fraction() <= 0.85,
            "compute={}",
            student.compute_fraction()
        );
        assert!(spec_after < 0.12, "spec_after={spec_after}");
    }

    #[test]
    fn ref_spectrum_training_reduces_error() {
        let mut model = DistilledTernaryFftModel::new(64, 16, 16_000.0);
        if let SpectrumCorrection::Band(ref mut band) = model.denoiser {
            for w in &mut band.weights {
                *w = 0.7;
            }
            for b in &mut band.bias {
                *b = 0.05;
            }
        }
        let batch = 4;
        let signal: Vec<f32> = (0..batch * 64).map(|i| (i as f32 * 0.03).sin()).collect();
        let w = hann_window(64);
        let mut windowed = signal.clone();
        for b in 0..batch {
            for i in 0..64 {
                windowed[b * 64 + i] *= w[i];
            }
        }
        let before = model
            .train_step_ref_spectrum(&windowed, batch, 0.0)
            .unwrap();
        assert!(before > 1e-4, "perturbed denoiser should be off: {before}");
        for _ in 0..80 {
            model
                .train_step_ref_spectrum(&windowed, batch, 8e-3)
                .unwrap();
        }
        let after = model
            .train_step_ref_spectrum(&windowed, batch, 0.0)
            .unwrap();
        assert!(
            after < before * 0.2,
            "spec err should drop: before={before} after={after}"
        );
    }
}
