//! Compiled Welch peaks — RLX Op::Fft or learned spectrum + streaming top-K.

use crate::config::FftLearnConfig;
use crate::learned_compile::{CompiledLearnedMel, compile_learned_mel, default_hard_threshold};
use crate::learned_model::FastLearnedFftModel;
use crate::peak::{WelchPeakParams, WelchPeaksScratch, peaks_from_segment_spectrum_streaming};
use crate::rlx_fft::rlx_fft_forward;
use crate::welch::{compile_welch_rlx_fft, hann_window, welch_windowed_segments};
use anyhow::{Result, ensure};
use rlx_runtime::{CompiledGraph, Device};

/// Compiled `Op::Fft` Welch segments → streaming top-K peaks.
pub struct CompiledRlxWelchPeaks {
    exec: CompiledGraph,
    pub peak_params: WelchPeakParams,
    batch: usize,
    window: Vec<f32>,
    pub run_device: Device,
}

pub fn compile_rlx_welch_peaks(
    batch: usize,
    peak_params: WelchPeakParams,
    device: Device,
) -> Result<CompiledRlxWelchPeaks> {
    let exec = compile_welch_rlx_fft(batch, peak_params.welch, device)?;
    Ok(CompiledRlxWelchPeaks {
        exec,
        peak_params,
        batch,
        window: hann_window(peak_params.welch.n_fft),
        run_device: device,
    })
}

impl CompiledRlxWelchPeaks {
    pub fn welch_peaks_batch(
        &mut self,
        signal: &[f32],
        scratch: &mut WelchPeaksScratch,
    ) -> Result<Vec<f32>> {
        let frame = self.peak_params.frame_len();
        ensure!(signal.len() == self.batch * frame);
        let segs =
            welch_windowed_segments(signal, self.batch, self.peak_params.welch, &self.window)?;
        let n_seg = self.peak_params.welch.n_segments;
        let spec = rlx_fft_forward(
            &mut self.exec,
            &segs,
            self.batch * n_seg,
            self.peak_params.welch.n_fft,
        );
        let psd = scratch.ensure(self.batch, self.peak_params.n_bins());
        Ok(peaks_from_segment_spectrum_streaming(
            &spec,
            self.batch,
            self.peak_params,
            psd,
        ))
    }
}

/// Compiled learned spectrum (all segments) → streaming top-K peaks.
pub struct CompiledLearnedWelchPeaks {
    spectrum: CompiledLearnedMel,
    pub peak_params: WelchPeakParams,
    welch_batch: usize,
    window: Vec<f32>,
}

pub fn compile_learned_welch_peaks(
    model: &FastLearnedFftModel,
    welch_batch: usize,
    peak_params: WelchPeakParams,
    device: Device,
    hard_gate_threshold: f32,
) -> Result<CompiledLearnedWelchPeaks> {
    ensure!(peak_params.welch.n_fft == model.n_fft);
    let seg_batch = welch_batch * peak_params.welch.n_segments;
    let cfg = FftLearnConfig::new(model.n_fft, seg_batch)?;
    let spectrum = compile_learned_mel(model, &cfg, device, hard_gate_threshold)?;
    Ok(CompiledLearnedWelchPeaks {
        spectrum,
        peak_params,
        welch_batch,
        window: hann_window(peak_params.welch.n_fft),
    })
}

impl CompiledLearnedWelchPeaks {
    pub fn run_device(&self) -> Device {
        self.spectrum.run_device
    }

    pub fn welch_peaks_batch(
        &mut self,
        signal: &[f32],
        scratch: &mut WelchPeaksScratch,
    ) -> Result<Vec<f32>> {
        let frame = self.peak_params.frame_len();
        ensure!(signal.len() == self.welch_batch * frame);
        let segs = welch_windowed_segments(
            signal,
            self.welch_batch,
            self.peak_params.welch,
            &self.window,
        )?;
        let spec = self.spectrum.spectrum_batch(&segs)?;
        let psd = scratch.ensure(self.welch_batch, self.peak_params.n_bins());
        Ok(peaks_from_segment_spectrum_streaming(
            &spec,
            self.welch_batch,
            self.peak_params,
            psd,
        ))
    }
}

pub fn default_welch_peaks_hard_threshold() -> f32 {
    default_hard_threshold()
}
