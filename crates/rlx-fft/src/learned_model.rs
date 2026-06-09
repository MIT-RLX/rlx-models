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

//! Fast learned FFT model — pruned butterfly + optional Q8 + denoiser + freq mask (Tier D).

use crate::config::FftLearnConfig;
use crate::denoise::SpectrumDenoiser;
use crate::mel::{hann_window, log_mel_from_spectrum_batch, mel_filterbank, ref_log_mel_batch};
use crate::peak::{WelchPeakParams, welch_peaks_from_segment_spectrum};
use crate::pruned::{init_gates, pruned_forward_real_batch};
use crate::q8::Q8Twiddles;
use crate::reference::{fft_real_batch, max_abs_error};
use crate::twiddle::exact_twiddles;
use crate::welch::{WelchParams, average_welch_psd, welch_rustfft, welch_windowed_segments};
use anyhow::{Result, ensure};

/// End-to-end learned spectral model for fast inference paths.
#[derive(Debug, Clone)]
pub struct FastLearnedFftModel {
    pub n_fft: usize,
    pub n_mels: usize,
    pub sample_rate: f32,
    pub twiddles: Vec<f32>,
    pub gates: Vec<f32>,
    /// Per complex-bin mask (learnable), init 1.
    pub freq_mask: Vec<f32>,
    pub denoiser: SpectrumDenoiser,
    pub use_q8: bool,
    q8: Option<Q8Twiddles>,
    mel_filters: Vec<f32>,
    /// When set, inference uses hard gates (0/1) at this threshold.
    pub hard_gate_threshold: Option<f32>,
}

impl FastLearnedFftModel {
    pub fn new(cfg: &FftLearnConfig, n_mels: usize, sample_rate: f32) -> Self {
        let n_fft = cfg.n_fft;
        Self {
            n_fft,
            n_mels,
            sample_rate,
            twiddles: exact_twiddles(cfg),
            gates: init_gates(n_fft),
            freq_mask: vec![1.0; n_fft * 2],
            denoiser: SpectrumDenoiser::identity(n_fft),
            use_q8: false,
            q8: None,
            mel_filters: mel_filterbank(n_fft, n_mels, sample_rate),
            hard_gate_threshold: None,
        }
    }

    pub fn with_hard_gates(mut self, threshold: f32) -> Self {
        self.hard_gate_threshold = Some(threshold);
        self
    }

    pub fn mel_filters(&self) -> &[f32] {
        &self.mel_filters
    }

    fn gates_for_inference(&self) -> Vec<f32> {
        match self.hard_gate_threshold {
            Some(t) => crate::pruned::hard_gates(&self.gates, t),
            None => self.gates.clone(),
        }
    }

    fn forward_spectrum(
        &self,
        signal: &[f32],
        batch: usize,
        apply_denoiser: bool,
    ) -> Result<Vec<f32>> {
        ensure!(signal.len() == batch * self.n_fft);
        let tw = self.effective_twiddles();
        let gates = self.gates_for_inference();
        let mut spec = pruned_forward_real_batch(signal, &tw, &gates, batch, self.n_fft)?;
        for b in 0..batch {
            for i in 0..self.n_fft * 2 {
                let idx = b * self.n_fft * 2 + i;
                spec[idx] *= self.freq_mask[i];
            }
        }
        if apply_denoiser {
            self.denoiser.apply_batch(&spec, batch, self.n_fft)
        } else {
            Ok(spec)
        }
    }

    pub fn with_q8(mut self) -> Self {
        self.use_q8 = true;
        self.q8 = Some(Q8Twiddles::from_f32(&self.twiddles));
        self
    }

    pub fn sync_q8(&mut self) {
        if self.use_q8 {
            self.q8 = Some(Q8Twiddles::from_f32(&self.twiddles));
        }
    }

    pub fn twiddles_for_forward(&self) -> Vec<f32> {
        self.effective_twiddles()
    }

    fn effective_twiddles(&self) -> Vec<f32> {
        if self.use_q8 {
            self.q8.as_ref().expect("q8").dequant()
        } else {
            self.twiddles.clone()
        }
    }

    pub fn spectrum_batch_raw(&self, signal: &[f32], batch: usize) -> Result<Vec<f32>> {
        self.forward_spectrum(signal, batch, false)
    }

    pub fn spectrum_batch(&self, signal: &[f32], batch: usize) -> Result<Vec<f32>> {
        self.forward_spectrum(signal, batch, true)
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
        let tw = self.effective_twiddles();
        let gates = self.gates_for_inference();
        let mut spec =
            pruned_forward_real_batch(&segs, &tw, &gates, batch * params.n_segments, self.n_fft)?;
        for seg in 0..(batch * params.n_segments) {
            for i in 0..self.n_fft * 2 {
                let idx = seg * self.n_fft * 2 + i;
                spec[idx] *= self.freq_mask[i];
            }
        }
        let spec = self
            .denoiser
            .apply_batch(&spec, batch * params.n_segments, self.n_fft)?;
        Ok(average_welch_psd(
            &spec,
            batch,
            params.n_segments,
            self.n_fft,
        ))
    }

    /// Fast Welch path (few segments) → top-K `(bin, power)` spikes only.
    pub fn welch_peaks_batch(
        &self,
        signal: &[f32],
        batch: usize,
        params: WelchPeakParams,
    ) -> Result<Vec<f32>> {
        ensure!(params.welch.n_fft == self.n_fft);
        let window = crate::welch::hann_window(self.n_fft);
        let segs = welch_windowed_segments(signal, batch, params.welch, &window)?;
        let tw = self.effective_twiddles();
        let gates = self.gates_for_inference();
        let mut spec = pruned_forward_real_batch(
            &segs,
            &tw,
            &gates,
            batch * params.welch.n_segments,
            self.n_fft,
        )?;
        for seg in 0..(batch * params.welch.n_segments) {
            for i in 0..self.n_fft * 2 {
                let idx = seg * self.n_fft * 2 + i;
                spec[idx] *= self.freq_mask[i];
            }
        }
        let spec = self
            .denoiser
            .apply_batch(&spec, batch * params.welch.n_segments, self.n_fft)?;
        Ok(welch_peaks_from_segment_spectrum(&spec, batch, params))
    }

    pub fn mean_gate(&self) -> f32 {
        crate::pruned::mean_gate(&self.gates)
    }

    pub fn active_gates(&self, threshold: f32) -> usize {
        crate::pruned::active_gate_count(&self.gates, threshold)
    }
}

/// Reference pipelines for validation.
pub fn ref_spectrum_batch(signal: &[f32], batch: usize, n_fft: usize) -> Result<Vec<f32>> {
    fft_real_batch(signal, batch, n_fft)
}

pub fn ref_log_mel(
    signal: &[f32],
    batch: usize,
    n_fft: usize,
    n_mels: usize,
    sr: f32,
) -> Result<Vec<f32>> {
    ref_log_mel_batch(signal, batch, n_fft, n_mels, sr)
}

pub fn ref_welch(signal: &[f32], batch: usize, params: WelchParams) -> Result<Vec<f32>> {
    welch_rustfft(signal, batch, params)
}

pub fn pipeline_max_err(pred: &[f32], target: &[f32]) -> f32 {
    max_abs_error(pred, target)
}
