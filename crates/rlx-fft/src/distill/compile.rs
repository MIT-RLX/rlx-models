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

//! Compiled distilled deploy — fused Hann → FFT → correction → `Op::LogMel`.

use crate::distill::fused::{
    append_hann_window, append_log_mel_head, append_spectrum_correction, load_hann_param,
    pick_fused_deploy_device,
};
use crate::distill::model::DistilledFftModel;
use crate::exec::compile::{compile_graph_with_cpu_fallback, try_compile_graph};
use crate::model::config::FftLearnConfig;
use crate::model::reference::block_to_interleaved_correct;
use anyhow::{Result, ensure};
use rlx_ir::infer::GraphExt;
use rlx_ir::{DType, Graph, NodeId, Shape};
use rlx_runtime::{CompiledGraph, Device};

pub struct CompiledDistilledMel {
    pub compiled: CompiledGraph,
    fft_only: CompiledGraph,
    pub run_device: Device,
    pub n_fft: usize,
    pub n_mels: usize,
    pub batch: usize,
    correction_gain: Vec<f32>,
    correction_bias: Vec<f32>,
    mel_filters: Vec<f32>,
    use_cpu_fast_path: bool,
}

/// RLX `Op::Fft` block `[batch, n*2]` (re ∥ im planes) → interleaved `[batch, n*2]`.
pub fn append_block_to_interleaved(
    g: &mut Graph,
    block: NodeId,
    batch: usize,
    n_fft: usize,
) -> NodeId {
    let re_plane = g.narrow_(block, 1, 0, n_fft);
    let im_plane = g.narrow_(block, 1, n_fft, n_fft);
    let re = g.reshape_(re_plane, vec![batch as i64, n_fft as i64, 1]);
    let im = g.reshape_(im_plane, vec![batch as i64, n_fft as i64, 1]);
    let interleaved = g.concat_(vec![re, im], 2);
    g.reshape_(interleaved, vec![batch as i64, (n_fft * 2) as i64])
}

pub fn build_distilled_spectrum_graph(cfg: &FftLearnConfig) -> Graph {
    crate::exec::rlx_fft::build_rlx_fft_forward_graph(cfg)
}

/// Fused mel graph: raw signal → Hann → FFT → correction → log-mel.
pub fn build_distilled_mel_graph(
    cfg: &FftLearnConfig,
    n_mels: usize,
) -> Result<(Graph, Vec<String>)> {
    let n = cfg.n_fft;
    let batch = cfg.batch;
    let f = DType::F32;
    let mut g = Graph::new("distilled_mel_fused");
    let mut names = Vec::new();

    let signal = g.input("signal", Shape::new(&[batch, n], f));
    let (windowed, hann_name) = append_hann_window(&mut g, signal, batch, n);
    names.push(hann_name.into());

    let zeros = g.sub(windowed, windowed);
    let block_in = g.concat_(vec![windowed, zeros], 1);
    let fft_block = g.fft(block_in, false);
    let flat = g.reshape_(fft_block, vec![batch as i64, (n * 2) as i64]);
    let corrected = append_spectrum_correction(&mut g, flat, cfg, &mut names);
    let mel = append_log_mel_head(&mut g, corrected, cfg, n_mels, &mut names);
    g.set_outputs(vec![mel]);
    Ok((g, names))
}

fn correction_params(model: &DistilledFftModel) -> (Vec<f32>, Vec<f32>) {
    let flat = model.n_fft * 2;
    let mut gain = vec![0f32; flat];
    let bias = model.denoiser.bias.clone();
    for i in 0..flat {
        gain[i] = model.freq_mask[i] * model.denoiser.scale[i];
    }
    (gain, bias)
}

/// Interleaved correction vectors → RLX FFT block layout (re plane, im plane).
pub(crate) fn block_affine_params(
    gain: &[f32],
    bias: &[f32],
    n_fft: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut block_gain = vec![0f32; n_fft * 2];
    let mut block_bias = vec![0f32; n_fft * 2];
    for i in 0..n_fft {
        block_gain[i] = gain[i * 2];
        block_gain[n_fft + i] = gain[i * 2 + 1];
        block_bias[i] = bias[i * 2];
        block_bias[n_fft + i] = bias[i * 2 + 1];
    }
    (block_gain, block_bias)
}

pub fn compile_distilled_mel(
    model: &DistilledFftModel,
    cfg: &FftLearnConfig,
    device: Device,
) -> Result<CompiledDistilledMel> {
    let deploy = pick_fused_deploy_device(cfg.batch, cfg.n_fft, device);
    let (graph, _names) = build_distilled_mel_graph(cfg, model.n_mels)?;
    let (deploy, mut compiled) = compile_graph_with_cpu_fallback(deploy, graph, "distilled_mel")?;
    let fft_only = try_compile_graph(deploy, build_distilled_spectrum_graph(cfg))?;
    let (gain, bias) = correction_params(model);
    let (block_gain, block_bias) = block_affine_params(&gain, &bias, cfg.n_fft);
    load_hann_param(&mut compiled, cfg.n_fft);
    compiled.set_param("corr.gain", &block_gain);
    compiled.set_param("corr.bias", &block_bias);
    compiled.set_param("mel.filters", model.mel_filters());
    Ok(CompiledDistilledMel {
        compiled,
        fft_only,
        run_device: deploy,
        n_fft: cfg.n_fft,
        n_mels: model.n_mels,
        batch: cfg.batch,
        correction_gain: gain,
        correction_bias: bias,
        mel_filters: model.mel_filters().to_vec(),
        use_cpu_fast_path: deploy == Device::Cpu,
    })
}

impl CompiledDistilledMel {
    /// Corrected interleaved spectrum (FFT + CPU layout fixup for non-mel pipelines).
    pub fn spectrum_batch(&mut self, windowed: &[f32]) -> Result<Vec<f32>> {
        ensure!(windowed.len() == self.batch * self.n_fft);
        let block = self.fft_only.run(&[("signal", windowed)]).remove(0);
        Ok(block_to_interleaved_correct(
            &block,
            self.batch,
            self.n_fft,
            &self.correction_gain,
            &self.correction_bias,
        ))
    }

    fn log_mel_batch_cpu_fast(&self, signal: &[f32]) -> Result<Vec<f32>> {
        use crate::model::reference::fft_real_batch;
        use crate::spectral::mel::{hann_window, log_mel_from_spectrum_batch};
        let w = hann_window(self.n_fft);
        let mut windowed = signal.to_vec();
        for b in 0..self.batch {
            for i in 0..self.n_fft {
                windowed[b * self.n_fft + i] *= w[i];
            }
        }
        let mut spec = fft_real_batch(&windowed, self.batch, self.n_fft)?;
        for b in 0..self.batch {
            for i in 0..self.n_fft * 2 {
                let idx = b * self.n_fft * 2 + i;
                spec[idx] = spec[idx] * self.correction_gain[i] + self.correction_bias[i];
            }
        }
        log_mel_from_spectrum_batch(
            &spec,
            &self.mel_filters,
            self.batch,
            self.n_fft,
            self.n_mels,
        )
    }

    /// Raw signal → fused Hann + FFT + correction + log-mel.
    pub fn log_mel_batch(&mut self, signal: &[f32]) -> Result<Vec<f32>> {
        ensure!(signal.len() == self.batch * self.n_fft);
        if self.use_cpu_fast_path {
            return self.log_mel_batch_cpu_fast(signal);
        }
        Ok(self.compiled.run(&[("signal", signal)]).remove(0))
    }

    /// Pre-windowed signal (skips Hann in graph — for legacy callers).
    pub fn log_mel_batch_windowed(&mut self, windowed: &[f32]) -> Result<Vec<f32>> {
        ensure!(windowed.len() == self.batch * self.n_fft);
        Ok(self.compiled.run(&[("signal", windowed)]).remove(0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distill::model::DistilledFftModel;
    use crate::model::reference::max_abs_error;

    #[test]
    fn distilled_mel_graph_builds() {
        let cfg = FftLearnConfig::new(64, 4).unwrap();
        build_distilled_mel_graph(&cfg, 16).unwrap();
    }

    #[test]
    fn compiled_distilled_matches_eager() {
        let cfg = FftLearnConfig::new(64, 4).unwrap();
        let model = DistilledFftModel::new(64, 16, 16_000.0);
        let signal: Vec<f32> = (0..256).map(|i| (i as f32 * 0.02).sin()).collect();
        let eager = model.log_mel_batch(&signal, 4).unwrap();
        let mut compiled = compile_distilled_mel(&model, &cfg, Device::Cpu).unwrap();
        let comp = compiled.log_mel_batch(&signal).unwrap();
        let err = max_abs_error(&eager, &comp);
        assert!(err < 1e-3, "eager vs compiled mel err={err}");
    }

    #[test]
    #[cfg(feature = "metal")]
    fn compile_distilled_metal() {
        let cfg = FftLearnConfig::new(64, 4).unwrap();
        let model = DistilledFftModel::new(64, 16, 16_000.0);
        let mut c = compile_distilled_mel(&model, &cfg, Device::Metal).unwrap();
        let signal = vec![0.1f32; 4 * 64];
        c.log_mel_batch(&signal).unwrap();
    }
}
