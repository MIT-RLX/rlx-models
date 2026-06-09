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

//! Compiled learned spectrum + mel deploy (Tier D) — gated butterfly + mask + denoiser.

use crate::butterfly::build_butterfly_forward_graph;
use crate::compile::try_compile_graph;
use crate::config::FftLearnConfig;
use crate::distill_compile::append_block_to_interleaved;
use crate::learned_model::FastLearnedFftModel;
use crate::mel::{hann_window, log_mel_from_spectrum_batch};
use crate::pruned::{
    DEFAULT_GATE_THRESHOLD, build_gated_butterfly_forward_graph, gate_param_name, hard_gates,
};
use crate::weights::WeightStore;
use anyhow::{Result, ensure};
use rlx_ir::infer::GraphExt;
use rlx_ir::{DType, Graph, NodeId, Shape};
use rlx_runtime::{CompiledGraph, Device, Session};

/// How the compiled spectrum graph was built.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LearnedSpectrumKind {
    /// Full learned butterfly (optionally gated).
    Butterfly,
    /// Native `Op::Fft` + learned mask/denoiser (GPU-safe deploy fallback).
    NativeFftPost,
}

/// Compiled butterfly FFT + freq mask + denoiser; log-mel applied after run.
pub struct CompiledLearnedMel {
    pub compiled: CompiledGraph,
    pub run_device: Device,
    pub kind: LearnedSpectrumKind,
    pub n_fft: usize,
    pub n_mels: usize,
    pub batch: usize,
    mel_filters: Vec<f32>,
}

pub fn build_learned_spectrum_graph(
    cfg: &FftLearnConfig,
    gates: &[f32],
    hard_threshold: f32,
) -> Result<(Graph, Vec<String>)> {
    let hard = hard_gates(gates, hard_threshold);
    let all_active = hard.iter().all(|&g| g >= 0.5);
    let (mut g, spectrum_out, mut param_names): (Graph, NodeId, Vec<String>) = if all_active {
        let built = build_butterfly_forward_graph(cfg)?;
        let names = built.params.iter().map(|p| p.name.clone()).collect();
        (built.graph, built.spectrum_out, names)
    } else {
        let built = build_gated_butterfly_forward_graph(cfg)?;
        let names = built
            .twiddle_params
            .iter()
            .chain(built.gate_params.iter())
            .map(|p| p.name.clone())
            .collect();
        (built.graph, built.spectrum_out, names)
    };
    append_mask_denoiser(&mut g, spectrum_out, cfg, &mut param_names);
    Ok((g, param_names))
}

/// GPU-safe graph: `Op::Fft` + learned mask/denoiser (no per-bin narrow butterfly ops).
pub fn build_native_fft_post_graph(cfg: &FftLearnConfig) -> Result<(Graph, Vec<String>)> {
    let n = cfg.n_fft;
    let batch = cfg.batch;
    let f = DType::F32;
    let mut g = Graph::new("learned_fft_post");
    let signal = g.input("signal", Shape::new(&[batch, n], f));
    let zeros = g.sub(signal, signal);
    let block = g.concat_(vec![signal, zeros], 1);
    let fft_out = g.fft(block, false);
    let flat = g.reshape_(fft_out, vec![batch as i64, (n * 2) as i64]);
    let interleaved = append_block_to_interleaved(&mut g, flat, batch, n);
    let mut param_names = Vec::new();
    append_mask_denoiser(&mut g, interleaved, cfg, &mut param_names);
    Ok((g, param_names))
}

fn append_mask_denoiser(
    g: &mut Graph,
    spectrum: NodeId,
    cfg: &FftLearnConfig,
    param_names: &mut Vec<String>,
) {
    let flat_len = cfg.n_fft * 2;
    let batch = cfg.batch;
    let f = DType::F32;
    let flat = g.reshape_(spectrum, vec![batch as i64, flat_len as i64]);
    let mask = g.param("freq_mask", Shape::new(&[flat_len], f));
    let scale = g.param("denoise.scale", Shape::new(&[flat_len], f));
    let bias = g.param("denoise.bias", Shape::new(&[flat_len], f));
    param_names.push("freq_mask".into());
    param_names.push("denoise.scale".into());
    param_names.push("denoise.bias".into());
    let masked = g.mul(flat, mask);
    let scaled = g.mul(masked, scale);
    let state = g.add(scaled, bias);
    g.set_outputs(vec![state]);
}

fn load_butterfly_params(
    compiled: &mut CompiledGraph,
    model: &FastLearnedFftModel,
    cfg: &FftLearnConfig,
    hard_gate_threshold: f32,
) {
    let store = WeightStore::from_twiddles(&model.twiddles, cfg.n_fft);
    store.apply_butterfly(compiled, cfg.batch, cfg.n_fft);
    let gates = hard_gates(&model.gates, hard_gate_threshold);
    if !gates.iter().all(|&g| g >= 0.5) {
        compiled.set_param("const.one", &[1.0]);
        let half = cfg.n_fft / 2;
        for s in 0..cfg.num_stages() {
            for b in 0..half {
                let gi = s * half + b;
                let name = gate_param_name(s, b);
                compiled.set_param(&name, &[gates[gi]]);
            }
        }
    }
}

fn load_mask_denoiser(compiled: &mut CompiledGraph, model: &FastLearnedFftModel) {
    compiled.set_param("freq_mask", &model.freq_mask);
    compiled.set_param("denoise.scale", &model.denoiser.scale);
    compiled.set_param("denoise.bias", &model.denoiser.bias);
}

pub fn compile_learned_mel(
    model: &FastLearnedFftModel,
    cfg: &FftLearnConfig,
    device: Device,
    hard_gate_threshold: f32,
) -> Result<CompiledLearnedMel> {
    let butterfly_graph = build_learned_spectrum_graph(cfg, &model.gates, hard_gate_threshold)?;

    // Try learned butterfly (Metal uses RLX_DISABLE_MPSGRAPH via try_compile_graph).
    if let Ok(mut compiled) = try_compile_graph(device, butterfly_graph.0.clone()) {
        load_butterfly_params(&mut compiled, model, cfg, hard_gate_threshold);
        load_mask_denoiser(&mut compiled, model);
        return Ok(CompiledLearnedMel {
            compiled,
            run_device: device,
            kind: LearnedSpectrumKind::Butterfly,
            n_fft: cfg.n_fft,
            n_mels: model.n_mels,
            batch: cfg.batch,
            mel_filters: model.mel_filters().to_vec(),
        });
    }

    if device != Device::Cpu {
        let native = build_native_fft_post_graph(cfg)?;
        let mut compiled = try_compile_graph(device, native.0)?;
        eprintln!(
            "[learned_compile] butterfly compile failed on {device:?}; using Op::Fft + mask/denoiser fallback"
        );
        load_mask_denoiser(&mut compiled, model);
        return Ok(CompiledLearnedMel {
            compiled,
            run_device: device,
            kind: LearnedSpectrumKind::NativeFftPost,
            n_fft: cfg.n_fft,
            n_mels: model.n_mels,
            batch: cfg.batch,
            mel_filters: model.mel_filters().to_vec(),
        });
    }

    eprintln!("[learned_compile] butterfly compile failed on CPU — retrying eager build");
    let mut compiled = Session::new(Device::Cpu).compile(butterfly_graph.0);
    load_butterfly_params(&mut compiled, model, cfg, hard_gate_threshold);
    load_mask_denoiser(&mut compiled, model);
    Ok(CompiledLearnedMel {
        compiled,
        run_device: Device::Cpu,
        kind: LearnedSpectrumKind::Butterfly,
        n_fft: cfg.n_fft,
        n_mels: model.n_mels,
        batch: cfg.batch,
        mel_filters: model.mel_filters().to_vec(),
    })
}

impl CompiledLearnedMel {
    pub fn mel_filters(&self) -> &[f32] {
        &self.mel_filters
    }

    /// Windowed `[batch, n_fft]` in → log-mel `[batch, n_mels]`.
    pub fn log_mel_batch(&mut self, windowed: &[f32]) -> Result<Vec<f32>> {
        ensure!(windowed.len() == self.batch * self.n_fft);
        let spec = self.compiled.run(&[("signal", windowed)]).remove(0);
        log_mel_from_spectrum_batch(
            &spec,
            &self.mel_filters,
            self.batch,
            self.n_fft,
            self.n_mels,
        )
    }

    pub fn spectrum_batch(&mut self, signal: &[f32]) -> Result<Vec<f32>> {
        ensure!(signal.len() == self.batch * self.n_fft);
        Ok(self.compiled.run(&[("signal", signal)]).remove(0))
    }
}

pub fn window_batch(signal: &[f32], batch: usize, n_fft: usize) -> Vec<f32> {
    let w = hann_window(n_fft);
    let mut out = signal.to_vec();
    for b in 0..batch {
        for i in 0..n_fft {
            out[b * n_fft + i] *= w[i];
        }
    }
    out
}

pub fn default_hard_threshold() -> f32 {
    DEFAULT_GATE_THRESHOLD
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pruned::init_gates;

    #[test]
    fn native_fft_post_graph_builds() {
        let cfg = FftLearnConfig::new(64, 4).unwrap();
        build_native_fft_post_graph(&cfg).unwrap();
    }

    #[test]
    fn compile_learned_cpu_butterfly() {
        let cfg = FftLearnConfig::new(64, 4).unwrap();
        let model = FastLearnedFftModel::new(&cfg, 16, 16_000.0);
        let c = compile_learned_mel(&model, &cfg, Device::Cpu, 0.5).unwrap();
        assert_eq!(c.kind, LearnedSpectrumKind::Butterfly);
        let signal = vec![0.1f32; 4 * 64];
        let win = window_batch(&signal, 4, 64);
        let mut c = c;
        c.log_mel_batch(&win).unwrap();
    }

    #[test]
    #[cfg(feature = "metal")]
    fn compile_learned_metal_butterfly_or_fallback() {
        let cfg = FftLearnConfig::new(64, 4).unwrap();
        let model = FastLearnedFftModel::new(&cfg, 16, 16_000.0);
        let c = compile_learned_mel(&model, &cfg, Device::Metal, 0.5).unwrap();
        assert_eq!(c.run_device, Device::Metal);
        assert!(
            matches!(
                c.kind,
                LearnedSpectrumKind::Butterfly | LearnedSpectrumKind::NativeFftPost
            ),
            "expected butterfly or native fallback, got {:?}",
            c.kind
        );
        let signal = vec![0.1f32; 4 * 64];
        let mut c = c;
        c.spectrum_batch(&signal).unwrap();
    }

    #[test]
    fn compile_gated_pruned_cpu() {
        let cfg = FftLearnConfig::new(64, 4).unwrap();
        let mut model = FastLearnedFftModel::new(&cfg, 16, 16_000.0);
        let n_gates = init_gates(64).len();
        for (i, g) in model.gates.iter_mut().enumerate() {
            *g = if i % 3 == 0 { 0.2 } else { 1.0 };
        }
        let c = compile_learned_mel(&model, &cfg, Device::Cpu, 0.5).unwrap();
        assert_eq!(c.kind, LearnedSpectrumKind::Butterfly);
        assert!(model.active_gates(0.5) < n_gates);
    }
}
