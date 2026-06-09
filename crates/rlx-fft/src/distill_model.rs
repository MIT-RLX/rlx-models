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

//! Distilled fast deploy model — Op::Fft + learned correction (+ optional mel adapter).

use crate::config::{FftLearnConfig, TransformDir};
use crate::denoise::SpectrumDenoiser;
use crate::learned_model::FastLearnedFftModel;
use crate::mel::{hann_window, log_mel_from_spectrum_batch, mel_filterbank, ref_log_mel_batch};
use crate::reference::{block_to_interleaved, max_abs_error};
use crate::rlx_fft::{compile_rlx_fft, rlx_fft_forward};
use crate::welch::{WelchParams, average_welch_psd, welch_windowed_segments};
use anyhow::{Result, ensure};
use rlx_runtime::{CompiledGraph, Device};
use std::cell::RefCell;

thread_local! {
    static RLX_FFT_CACHE: RefCell<Option<(usize, usize, CompiledGraph)>> = const { RefCell::new(None) };
}

fn rlx_fft_interleaved(windowed: &[f32], batch: usize, n_fft: usize) -> Result<Vec<f32>> {
    RLX_FFT_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        let needs_compile = cache
            .as_ref()
            .is_none_or(|(b, n, _)| *b != batch || *n != n_fft);
        if needs_compile {
            let cfg = FftLearnConfig::new(n_fft, batch)?;
            *cache = Some((
                batch,
                n_fft,
                compile_rlx_fft(&cfg, TransformDir::Forward, Device::Cpu)?,
            ));
        }
        let exec = &mut cache.as_mut().unwrap().2;
        Ok(rlx_fft_forward(exec, windowed, batch, n_fft))
    })
}

/// Student model: native FFT + per-bin correction (distilled from [`FastLearnedFftModel`]).
#[derive(Debug, Clone)]
pub struct DistilledFftModel {
    pub n_fft: usize,
    pub n_mels: usize,
    pub sample_rate: f32,
    pub freq_mask: Vec<f32>,
    pub denoiser: SpectrumDenoiser,
    mel_filters: Vec<f32>,
}

impl DistilledFftModel {
    pub fn new(n_fft: usize, n_mels: usize, sample_rate: f32) -> Self {
        Self {
            n_fft,
            n_mels,
            sample_rate,
            freq_mask: vec![1.0; n_fft * 2],
            denoiser: SpectrumDenoiser::identity(n_fft),
            mel_filters: mel_filterbank(n_fft, n_mels, sample_rate),
        }
    }

    pub fn from_teacher(teacher: &FastLearnedFftModel) -> Self {
        Self {
            n_fft: teacher.n_fft,
            n_mels: teacher.n_mels,
            sample_rate: teacher.sample_rate,
            freq_mask: teacher.freq_mask.clone(),
            denoiser: teacher.denoiser.clone(),
            mel_filters: teacher.mel_filters().to_vec(),
        }
    }

    pub fn mel_filters(&self) -> &[f32] {
        &self.mel_filters
    }

    pub fn hann(&self) -> Vec<f32> {
        hann_window(self.n_fft)
    }

    pub fn spectrum_batch(&self, windowed: &[f32], batch: usize) -> Result<Vec<f32>> {
        ensure!(windowed.len() == batch * self.n_fft);
        self.correct_fft_spectrum(windowed, batch)
    }

    /// Raw (unwindowed) signal → corrected spectrum (denoise / q8 pipelines).
    pub fn spectrum_batch_raw(&self, signal: &[f32], batch: usize) -> Result<Vec<f32>> {
        ensure!(signal.len() == batch * self.n_fft);
        self.correct_fft_spectrum(signal, batch)
    }

    fn correct_fft_spectrum(&self, signal: &[f32], batch: usize) -> Result<Vec<f32>> {
        let mut spec = rlx_fft_interleaved(signal, batch, self.n_fft)?;
        for b in 0..batch {
            for i in 0..self.n_fft * 2 {
                let idx = b * self.n_fft * 2 + i;
                spec[idx] *= self.freq_mask[i];
            }
        }
        self.denoiser.apply_batch(&spec, batch, self.n_fft)
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
        let block = {
            let cfg = FftLearnConfig::new(self.n_fft, batch * params.n_segments)?;
            let mut exec = compile_rlx_fft(&cfg, TransformDir::Forward, Device::Cpu)?;
            exec.run(&[("signal", &segs)]).remove(0)
        };
        let mut spec = block_to_interleaved(&block, batch * params.n_segments, self.n_fft);
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

    pub fn welch_peaks_batch(
        &self,
        signal: &[f32],
        batch: usize,
        params: crate::peak::WelchPeakParams,
    ) -> Result<Vec<f32>> {
        let psd = self.welch_psd_batch(signal, batch, params.welch)?;
        Ok(crate::peak::peaks_from_psd_batch(
            &psd,
            batch,
            params.n_bins(),
            params.k,
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
        let n = (batch * self.n_fft * 2) as f32;
        for i in 0..self.n_fft * 2 {
            let mut gs = 0f32;
            for b in 0..batch {
                gs += grad[b * self.n_fft * 2 + i];
            }
            let gs = gs / n.max(1.0);
            self.denoiser.scale[i] -= lr * gs * spec[i];
            self.denoiser.bias[i] -= lr * gs;
            self.freq_mask[i] -= lr * 0.05 * gs * spec[i];
            self.freq_mask[i] = self.freq_mask[i].clamp(0.0, 2.0);
        }
        Ok(err)
    }

    pub fn train_step_welch_spectrum(
        &mut self,
        welch_spec: &[f32],
        target_spec: &[f32],
        n_segs: usize,
        lr: f32,
    ) -> Result<()> {
        let batch_total = welch_spec.len() / (self.n_fft * 2);
        ensure!(batch_total >= n_segs);
        let _ = self.denoiser.train_step_affine(
            welch_spec,
            target_spec,
            batch_total,
            self.n_fft,
            lr,
        )?;
        Ok(())
    }
}

pub fn teacher_mel_batch(
    teacher: &FastLearnedFftModel,
    signal: &[f32],
    batch: usize,
) -> Result<Vec<f32>> {
    teacher.log_mel_batch(signal, batch)
}

pub fn teacher_welch_batch(
    teacher: &FastLearnedFftModel,
    signal: &[f32],
    batch: usize,
    params: WelchParams,
) -> Result<Vec<f32>> {
    teacher.welch_psd_batch(signal, batch, params)
}

pub fn ref_mel_for_distill(
    signal: &[f32],
    batch: usize,
    n_fft: usize,
    n_mels: usize,
    sr: f32,
) -> Result<Vec<f32>> {
    ref_log_mel_batch(signal, batch, n_fft, n_mels, sr)
}
