//! Compiled Welch peaks — RLX Op::Fft or learned spectrum + streaming top-K.

use crate::config::FftLearnConfig;
use crate::learned_compile::{CompiledLearnedMel, compile_learned_mel, default_hard_threshold};
use crate::learned_model::FastLearnedFftModel;
use crate::peak::{
    WelchPeakParams, WelchPeaksScratch, peaks_from_block_segment_spectrum_streaming,
    peaks_from_segment_spectrum_streaming,
};
use crate::rlx_fft::{rlx_fft_forward, rlx_fft_forward_block};
use crate::welch::{compile_welch_rlx_fft, hann_window, welch_windowed_segments};
use anyhow::{Result, ensure};
use rlx_ir::infer::GraphExt;
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{CompiledGraph, Device, Session};

/// Fused FFT → `Op::WelchPeaks` graph (peaks-only host readback).
pub fn build_welch_peaks_fused_graph(batch: usize, peak_params: WelchPeakParams) -> Graph {
    let n = peak_params.welch.n_fft;
    let seg_batch = batch * peak_params.welch.n_segments;
    let mut g = Graph::new("welch_peaks_fused");
    let segs = g.input("segs", Shape::new(&[seg_batch, n], DType::F32));
    let zeros = g.sub(segs, segs);
    let block_in = g.concat_(vec![segs, zeros], 1);
    let spec = g.fft(block_in, false);
    let peaks = g.welch_peaks(spec, peak_params.k, peak_params.welch.n_segments);
    g.set_outputs(vec![peaks]);
    g
}

pub fn compile_welch_peaks_fused(
    batch: usize,
    peak_params: WelchPeakParams,
    device: Device,
) -> Result<CompiledGraph> {
    Ok(Session::new(device).compile(build_welch_peaks_fused_graph(batch, peak_params)))
}

/// Compiled fused Welch peaks (FFT + top-K in one graph).
pub struct CompiledRlxWelchPeaksFused {
    exec: CompiledGraph,
    pub peak_params: WelchPeakParams,
    batch: usize,
    window: Vec<f32>,
    pub run_device: Device,
}

impl CompiledRlxWelchPeaksFused {
    pub fn compile(batch: usize, peak_params: WelchPeakParams, device: Device) -> Result<Self> {
        Ok(Self {
            exec: compile_welch_peaks_fused(batch, peak_params, device)?,
            peak_params,
            batch,
            window: hann_window(peak_params.welch.n_fft),
            run_device: device,
        })
    }

    pub fn welch_peaks_batch(&mut self, signal: &[f32]) -> Result<Vec<f32>> {
        let frame = self.peak_params.frame_len();
        ensure!(signal.len() == self.batch * frame);
        let segs =
            welch_windowed_segments(signal, self.batch, self.peak_params.welch, &self.window)?;
        Ok(self.exec.run(&[("segs", &segs)]).remove(0))
    }
}

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
    /// Legacy path: FFT readback + interleaved layout + host top-K.
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

    /// Phase 1 — FFT block layout on host, skip interleaved convert.
    pub fn welch_peaks_batch_block(
        &mut self,
        signal: &[f32],
        scratch: &mut WelchPeaksScratch,
    ) -> Result<Vec<f32>> {
        let frame = self.peak_params.frame_len();
        ensure!(signal.len() == self.batch * frame);
        let segs =
            welch_windowed_segments(signal, self.batch, self.peak_params.welch, &self.window)?;
        let n_seg = self.peak_params.welch.n_segments;
        let spec = rlx_fft_forward_block(
            &mut self.exec,
            &segs,
            self.batch * n_seg,
            self.peak_params.welch.n_fft,
        );
        let psd = scratch.ensure(self.batch, self.peak_params.n_bins());
        Ok(peaks_from_block_segment_spectrum_streaming(
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
