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
use crate::welch_peaks_cost::{fused_welch_peaks_auto_viable, welch_peaks_io_fusion_gate};
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

/// Which RLX compile path `compile_rlx_welch_peaks_adaptive` selects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RlxWelchPeaksExecKind {
    /// `Op::Fft` + `Op::WelchPeaks` in one graph (peaks-only readback).
    FusedOp,
    /// GPU `Op::Fft` block layout + host streaming top-K.
    BlockFftHostPeaks,
}

impl RlxWelchPeaksExecKind {
    pub fn label(self) -> &'static str {
        match self {
            Self::FusedOp => "fused_op",
            Self::BlockFftHostPeaks => "block_fft_host",
        }
    }
}

/// Pick fused vs block RLX path from IO gate + device viability (no compile).
pub fn rlx_welch_peaks_exec_kind(
    device: Device,
    batch: usize,
    peak_params: WelchPeakParams,
) -> RlxWelchPeaksExecKind {
    let n_fft = peak_params.welch.n_fft;
    let k = peak_params.k;
    if fused_welch_peaks_auto_viable(device) && welch_peaks_io_fusion_gate(device, batch, n_fft, k)
    {
        RlxWelchPeaksExecKind::FusedOp
    } else {
        RlxWelchPeaksExecKind::BlockFftHostPeaks
    }
}

/// Adaptive RLX Welch peaks — fused when IO gate passes, else block FFT + host top-K.
pub struct CompiledRlxWelchPeaksExec {
    pub kind: RlxWelchPeaksExecKind,
    pub peak_params: WelchPeakParams,
    pub run_device: Device,
    fused: Option<CompiledRlxWelchPeaksFused>,
    block: Option<CompiledRlxWelchPeaks>,
}

impl CompiledRlxWelchPeaksExec {
    pub fn compile_adaptive(
        batch: usize,
        peak_params: WelchPeakParams,
        device: Device,
    ) -> Result<Self> {
        match rlx_welch_peaks_exec_kind(device, batch, peak_params) {
            RlxWelchPeaksExecKind::FusedOp => {
                let fused = CompiledRlxWelchPeaksFused::compile(batch, peak_params, device)?;
                let run_device = fused.run_device;
                Ok(Self {
                    kind: RlxWelchPeaksExecKind::FusedOp,
                    peak_params,
                    run_device,
                    fused: Some(fused),
                    block: None,
                })
            }
            RlxWelchPeaksExecKind::BlockFftHostPeaks => {
                let block = compile_rlx_welch_peaks(batch, peak_params, device)?;
                let run_device = block.run_device;
                Ok(Self {
                    kind: RlxWelchPeaksExecKind::BlockFftHostPeaks,
                    peak_params,
                    run_device,
                    fused: None,
                    block: Some(block),
                })
            }
        }
    }

    pub fn welch_peaks_batch(
        &mut self,
        signal: &[f32],
        scratch: &mut WelchPeaksScratch,
    ) -> Result<Vec<f32>> {
        match self.kind {
            RlxWelchPeaksExecKind::FusedOp => self
                .fused
                .as_mut()
                .expect("fused exec")
                .welch_peaks_batch(signal),
            RlxWelchPeaksExecKind::BlockFftHostPeaks => self
                .block
                .as_mut()
                .expect("block exec")
                .welch_peaks_batch_block(signal, scratch),
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_runtime::Device;

    #[test]
    fn exec_kind_metal_large_batch_fused() {
        let params = WelchPeakParams::fast_for_n_fft(256, 16);
        assert_eq!(
            rlx_welch_peaks_exec_kind(Device::Metal, 8192, params),
            RlxWelchPeaksExecKind::FusedOp
        );
    }

    #[test]
    fn exec_kind_metal_small_batch_block() {
        let params = WelchPeakParams::fast_for_n_fft(256, 16);
        assert_eq!(
            rlx_welch_peaks_exec_kind(Device::Metal, 256, params),
            RlxWelchPeaksExecKind::BlockFftHostPeaks
        );
    }

    #[test]
    fn exec_kind_wgpu_small_batch_block() {
        let params = WelchPeakParams::fast_for_n_fft(256, 16);
        assert_eq!(
            rlx_welch_peaks_exec_kind(Device::Gpu, 256, params),
            RlxWelchPeaksExecKind::BlockFftHostPeaks
        );
    }

    #[test]
    fn exec_kind_wgpu_large_batch_fused() {
        let params = WelchPeakParams::fast_for_n_fft(256, 16);
        assert_eq!(
            rlx_welch_peaks_exec_kind(Device::Gpu, 8192, params),
            RlxWelchPeaksExecKind::FusedOp
        );
    }

    #[cfg(feature = "metal")]
    #[test]
    fn compile_fusion_pipeline_drops_dual_spectrum_output() {
        use crate::welch_peaks_cost::welch_peaks_fusion_target;
        use rlx_compile::{FusionOptions, run_fusion_pipeline, supported_for_target, supports_op};
        use rlx_ir::{Op, OpKind};
        use rlx_runtime::graph_io::profile_graph_io;

        let batch = 1024;
        let params = WelchPeakParams::fast_for_n_fft(256, 16);
        let mut dual = build_welch_peaks_fused_graph(batch, params);
        let peaks_id = dual.outputs[0];
        let spec_id = dual.node(peaks_id).inputs[0];
        dual.set_outputs(vec![spec_id, peaks_id]);
        let before = profile_graph_io(&dual);

        let target = welch_peaks_fusion_target(Device::Metal);
        let mut supported: Vec<OpKind> = supported_for_target(target).to_vec();
        if !supports_op(&supported, OpKind::Fft) {
            supported.push(OpKind::Fft);
        }
        if !supports_op(&supported, OpKind::WelchPeaks) {
            supported.push(OpKind::WelchPeaks);
        }
        let out = run_fusion_pipeline(dual, target, &supported, FusionOptions::default());
        let after = profile_graph_io(&out);
        assert_eq!(out.outputs.len(), 1);
        assert!(matches!(out.node(out.outputs[0]).op, Op::WelchPeaks { .. }));
        assert!(after.host_output_bytes < before.host_output_bytes);
    }
}
