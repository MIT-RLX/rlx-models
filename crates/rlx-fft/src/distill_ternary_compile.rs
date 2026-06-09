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

//! Compiled ternary-routed distilled deploy — pruned Hann → sparse butterfly → mel.

use crate::butterfly::split_complex_planes;
use crate::compile::{compile_graph_with_cpu_fallback, try_compile_graph};
use crate::config::FftLearnConfig;
use crate::distill_fused::{
    append_banded_spectrum_correction, append_hann_window, append_log_mel_head,
    pick_fused_deploy_device,
};
use crate::distill_ternary_model::DistilledTernaryFftModel;
use crate::pruned::{active_ternary_butterfly_count, append_ternary_butterfly_from_real_signal};
use anyhow::{Result, ensure};
use rlx_ir::infer::GraphExt;
use rlx_ir::{DType, Graph, NodeId, Shape};
use rlx_runtime::{CompiledGraph, Device};

pub struct CompiledDistilledTernaryMel {
    pub mel_compiled: CompiledGraph,
    spectrum_compiled: CompiledGraph,
    pub run_device: Device,
    pub n_fft: usize,
    pub n_mels: usize,
    pub batch: usize,
    pub compute_fraction: f32,
    pub active_butterflies: usize,
}

fn bake_band(
    c: &crate::ternary_arch::SpectrumCorrection,
    freq_mask: &[f32],
) -> Result<(Vec<f32>, Vec<f32>)> {
    use crate::ternary_arch::SpectrumCorrection;
    match c {
        SpectrumCorrection::Band(b) => Ok((b.dense_rhs_with_freq_mask(freq_mask), b.bias.clone())),
        SpectrumCorrection::Identity => {
            let n = freq_mask.len();
            let mut dense = vec![0f32; n * n];
            for i in 0..n {
                dense[i * n + i] = freq_mask[i];
            }
            Ok((dense, vec![0.0; n]))
        }
        SpectrumCorrection::Affine(_) => {
            anyhow::bail!("affine correction uses eager path or affine compile graph")
        }
    }
}

fn spectrum_from_butterfly_state(
    g: &mut Graph,
    state: NodeId,
    batch: usize,
    n_fft: usize,
) -> NodeId {
    let flat_len = n_fft * 2;
    g.reshape_(state, vec![batch as i64, flat_len as i64])
}

/// Butterfly `[batch, n, 2]` interleaved → RLX `Op::LogMel` block `[batch, n*2]`.
fn append_interleaved_to_block(
    g: &mut Graph,
    interleaved: NodeId,
    batch: usize,
    n_fft: usize,
) -> NodeId {
    let (re, im) = split_complex_planes(g, interleaved, batch, n_fft);
    g.concat_(vec![re, im], 1)
}

/// Fused: raw signal → Hann → pruned ternary butterfly → correction → log-mel.
pub fn build_distilled_ternary_mel_graph(
    cfg: &FftLearnConfig,
    n_mels: usize,
    gates: &[i8],
) -> Result<(Graph, Vec<String>)> {
    let n = cfg.n_fft;
    let batch = cfg.batch;
    let f = DType::F32;
    let mut g = Graph::new("distilled_ternary_mel_pruned");
    let mut names = Vec::new();

    let signal = g.input("signal", Shape::new(&[batch, n], f));
    let (windowed, hann_name) = append_hann_window(&mut g, signal, batch, n);
    names.push(hann_name.into());

    let (twiddle_params, butterfly_out) =
        append_ternary_butterfly_from_real_signal(&mut g, cfg, windowed, gates)?;
    for p in twiddle_params {
        names.push(p.name);
    }

    let block = append_interleaved_to_block(&mut g, butterfly_out, batch, n);
    let corrected = append_banded_spectrum_correction(&mut g, block, cfg, "corr", &mut names);
    let mel = append_log_mel_head(&mut g, corrected, cfg, n_mels, &mut names);
    g.set_outputs(vec![mel]);
    Ok((g, names))
}

/// Spectrum-only graph (Welch / denoise): pruned ternary butterfly → correction.
pub fn build_distilled_ternary_spectrum_graph(
    cfg: &FftLearnConfig,
    gates: &[i8],
) -> Result<(Graph, Vec<String>)> {
    let n = cfg.n_fft;
    let batch = cfg.batch;
    let f = DType::F32;
    let mut g = Graph::new("distilled_ternary_spectrum_pruned");
    let mut names = Vec::new();
    let signal = g.input("signal", Shape::new(&[batch, n], f));
    let (twiddle_params, butterfly_out) =
        append_ternary_butterfly_from_real_signal(&mut g, cfg, signal, gates)?;
    for p in twiddle_params {
        names.push(p.name);
    }
    let flat = spectrum_from_butterfly_state(&mut g, butterfly_out, batch, n);
    let corrected = append_banded_spectrum_correction(&mut g, flat, cfg, "corr", &mut names);
    g.set_outputs(vec![corrected]);
    Ok((g, names))
}

pub fn compile_distilled_ternary_mel(
    model: &DistilledTernaryFftModel,
    cfg: &FftLearnConfig,
    device: Device,
) -> Result<CompiledDistilledTernaryMel> {
    let deploy = pick_fused_deploy_device(cfg.batch, cfg.n_fft, device);
    let mel_gates = model.mel_gates().to_vec();
    let spec_gates = model.spec_gates_slice().to_vec();
    let active = active_ternary_butterfly_count(&mel_gates);
    let (mel_graph, _) = build_distilled_ternary_mel_graph(cfg, model.n_mels, &mel_gates)?;
    let (deploy, mut mel_compiled) =
        compile_graph_with_cpu_fallback(deploy, mel_graph, "distilled_ternary_mel")?;

    let (spec_graph, _) = build_distilled_ternary_spectrum_graph(cfg, &spec_gates)?;
    let mut spectrum_compiled = try_compile_graph(deploy, spec_graph)?;

    let store = crate::weights::WeightStore::from_twiddles(&model.twiddles, cfg.n_fft);
    store.apply_butterfly_for_gates(&mut mel_compiled, cfg.n_fft, &mel_gates);
    store.apply_butterfly_for_gates(&mut spectrum_compiled, cfg.n_fft, &spec_gates);
    let (mel_w, mel_b) = bake_band(&model.mel_denoiser, &model.freq_mask)?;
    let (spec_w, spec_b) = bake_band(&model.denoiser, &model.freq_mask)?;
    mel_compiled.set_param("hann", &crate::mel::hann_window(cfg.n_fft));
    mel_compiled.set_param("corr.band_rhs", &mel_w);
    mel_compiled.set_param("corr.bias", &mel_b);
    mel_compiled.set_param("mel.filters", model.mel_filters());
    spectrum_compiled.set_param("corr.band_rhs", &spec_w);
    spectrum_compiled.set_param("corr.bias", &spec_b);

    Ok(CompiledDistilledTernaryMel {
        mel_compiled,
        spectrum_compiled,
        run_device: deploy,
        n_fft: cfg.n_fft,
        n_mels: model.n_mels,
        batch: cfg.batch,
        compute_fraction: crate::ternary_gates::compute_fraction(&mel_gates),
        active_butterflies: active,
    })
}

impl CompiledDistilledTernaryMel {
    pub fn spectrum_batch(&mut self, windowed: &[f32]) -> Result<Vec<f32>> {
        ensure!(windowed.len() == self.batch * self.n_fft);
        Ok(self
            .spectrum_compiled
            .run(&[("signal", windowed)])
            .remove(0))
    }

    /// Raw signal → fused pruned ternary FFT + correction + log-mel.
    pub fn log_mel_batch(&mut self, signal: &[f32]) -> Result<Vec<f32>> {
        ensure!(signal.len() == self.batch * self.n_fft);
        Ok(self.mel_compiled.run(&[("signal", signal)]).remove(0))
    }

    pub fn log_mel_batch_windowed(&mut self, windowed: &[f32]) -> Result<Vec<f32>> {
        self.log_mel_batch(windowed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distill_ternary_model::DistilledTernaryFftModel;
    use crate::reference::{fft_real_batch, max_abs_error};
    use crate::ternary_gates::{GateMode, init_ternary_gates};

    #[test]
    fn ternary_mel_graph_builds() {
        let cfg = FftLearnConfig::new(64, 4).unwrap();
        let gates = init_ternary_gates(64);
        build_distilled_ternary_mel_graph(&cfg, 16, &gates).unwrap();
    }

    #[test]
    fn pruned_graph_omits_skip_twiddles() {
        let cfg = FftLearnConfig::new(64, 4).unwrap();
        let mut gates = init_ternary_gates(64);
        gates.fill(GateMode::Skip.to_i8());
        gates[0] = GateMode::Forward.to_i8();
        let (g, names) = build_distilled_ternary_mel_graph(&cfg, 16, &gates).unwrap();
        let twiddle_names: Vec<_> = names
            .iter()
            .filter(|n| n.contains(".re") || n.contains(".im"))
            .collect();
        assert_eq!(
            twiddle_names.len(),
            2,
            "one forward butterfly → 2 twiddle params"
        );
        let _ = g;
    }

    #[test]
    fn compiled_ternary_all_forward_spectrum_matches_eager() {
        let cfg = FftLearnConfig::new(64, 4).unwrap();
        let model = DistilledTernaryFftModel::new(64, 16, 16_000.0);
        let signal: Vec<f32> = (0..256).map(|i| (i as f32 * 0.02).sin()).collect();
        let w = crate::mel::hann_window(64);
        let mut windowed = signal.clone();
        for b in 0..4 {
            for i in 0..64 {
                windowed[b * 64 + i] *= w[i];
            }
        }
        let eager_spec = model.spectrum_batch(&windowed, 4).unwrap();
        let mut compiled = compile_distilled_ternary_mel(&model, &cfg, Device::Cpu).unwrap();
        let comp_spec = compiled.spectrum_batch(&windowed).unwrap();
        let err = max_abs_error(&eager_spec, &comp_spec);
        assert!(err < 1e-3, "spectrum eager vs pruned compiled err={err}");
    }

    fn pruned_gate_fixture(n_fft: usize) -> DistilledTernaryFftModel {
        let mut model = DistilledTernaryFftModel::new(n_fft, 40, 16_000.0);
        model.gates.fill(GateMode::Skip.to_i8());
        for (i, g) in model.gates.iter_mut().enumerate() {
            if i % 3 == 0 {
                *g = GateMode::Forward.to_i8();
            } else if i % 5 == 0 {
                *g = GateMode::Reverse.to_i8();
            }
        }
        model
    }

    #[test]
    fn compiled_ternary_pruned_spectrum_matches_eager_n128() {
        let cfg = FftLearnConfig::new(128, 8).unwrap();
        let model = pruned_gate_fixture(128);
        let signal: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.01).sin()).collect();
        let w = crate::mel::hann_window(128);
        let mut windowed = signal.clone();
        for b in 0..8 {
            for i in 0..128 {
                windowed[b * 128 + i] *= w[i];
            }
        }
        let eager_spec = model.spectrum_batch_accurate(&windowed, 8).unwrap();
        let mut compiled = compile_distilled_ternary_mel(&model, &cfg, Device::Cpu).unwrap();
        let comp_spec = compiled.spectrum_batch(&windowed).unwrap();
        let err = max_abs_error(&eager_spec, &comp_spec);
        assert!(
            err < 0.05,
            "pruned spectrum n=128 eager vs compiled err={err}"
        );
    }

    #[test]
    fn compiled_spectrum_one_skip_matches_eager() {
        let cfg = FftLearnConfig::new(128, 8).unwrap();
        let mut model = DistilledTernaryFftModel::new(128, 40, 16_000.0);
        model.gates[0] = GateMode::Skip.to_i8();
        model.reset_correction_for_gates();
        let batch = 8;
        let signal: Vec<f32> = (0..batch * 128).map(|i| (i as f32 * 0.01).sin()).collect();
        for _ in 0..120 {
            model.train_step_ref_spectrum(&signal, batch, 8e-3).unwrap();
        }
        let eager = model.spectrum_batch_raw(&signal, batch).unwrap();
        let mut compiled = compile_distilled_ternary_mel(&model, &cfg, Device::Cpu).unwrap();
        let comp = compiled.spectrum_batch(&signal).unwrap();
        let ref_spec = fft_real_batch(&signal, batch, 128).unwrap();
        let eager_comp = max_abs_error(&eager, &comp);
        let eager_ref = max_abs_error(&eager, &ref_spec);
        let comp_ref = max_abs_error(&comp, &ref_spec);
        assert!(eager_comp < 0.02, "eager vs compiled err={eager_comp}");
        assert!(
            (eager_ref - comp_ref).abs() < 1e-5,
            "eager_ref={eager_ref} comp_ref={comp_ref}"
        );
    }

    #[test]
    fn compiled_ternary_spectrum_matches_eager_after_ref_spectrum_train() {
        let cfg = FftLearnConfig::new(128, 8).unwrap();
        let mut model = DistilledTernaryFftModel::new(128, 40, 16_000.0);
        let batch = 8;
        let signal: Vec<f32> = (0..batch * 128).map(|i| (i as f32 * 0.01).sin()).collect();
        for _ in 0..120 {
            model.train_step_ref_spectrum(&signal, batch, 8e-3).unwrap();
            model.train_step_q8_spectrum(&signal, batch, 5e-3).unwrap();
        }
        let eager = model.spectrum_batch_raw(&signal, batch).unwrap();
        let mut compiled = compile_distilled_ternary_mel(&model, &cfg, Device::Cpu).unwrap();
        let comp = compiled.spectrum_batch(&signal).unwrap();
        let err = max_abs_error(&eager, &comp);
        assert!(err < 0.02, "spectrum eager vs compiled err={err}");
    }

    #[test]
    fn compiled_ternary_mel_matches_eager_after_ref_spectrum_train() {
        let cfg = FftLearnConfig::new(128, 8).unwrap();
        let mut model = DistilledTernaryFftModel::new(128, 40, 16_000.0);
        let batch = 8;
        let signal: Vec<f32> = (0..batch * 128).map(|i| (i as f32 * 0.01).sin()).collect();
        let w = crate::mel::hann_window(128);
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
        let eager_mel = model.log_mel_batch(&signal, batch).unwrap();
        let mut compiled = compile_distilled_ternary_mel(&model, &cfg, Device::Cpu).unwrap();
        let comp_mel = compiled.log_mel_batch(&signal).unwrap();
        let err = max_abs_error(&eager_mel, &comp_mel);
        assert!(err < 0.02, "trained mel eager vs compiled err={err}");
    }

    #[test]
    fn compiled_ternary_mel_matches_eager_pruned() {
        let cfg = FftLearnConfig::new(128, 8).unwrap();
        let model = pruned_gate_fixture(128);
        let signal: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.01).sin()).collect();
        let eager_mel = model.log_mel_batch(&signal, 8).unwrap();
        let mut compiled = compile_distilled_ternary_mel(&model, &cfg, Device::Cpu).unwrap();
        let comp_mel = compiled.log_mel_batch(&signal).unwrap();
        let err = max_abs_error(&eager_mel, &comp_mel);
        assert!(err < 0.05, "pruned mel eager vs compiled err={err}");
    }

    #[test]
    fn compiled_ternary_reverse_gates_match_eager() {
        let cfg = FftLearnConfig::new(64, 4).unwrap();
        let mut model = DistilledTernaryFftModel::new(64, 16, 16_000.0);
        model.gates.fill(GateMode::Skip.to_i8());
        for (i, g) in model.gates.iter_mut().enumerate() {
            if i % 4 == 1 {
                *g = GateMode::Reverse.to_i8();
            } else if i % 4 == 2 {
                *g = GateMode::Forward.to_i8();
            }
        }
        let signal: Vec<f32> = (0..256).map(|i| (i as f32 * 0.025).sin()).collect();
        let w = crate::mel::hann_window(64);
        let mut windowed = signal.clone();
        for b in 0..4 {
            for i in 0..64 {
                windowed[b * 64 + i] *= w[i];
            }
        }
        let eager_spec = model.spectrum_batch_accurate(&windowed, 4).unwrap();
        let mut compiled = compile_distilled_ternary_mel(&model, &cfg, Device::Cpu).unwrap();
        let comp_spec = compiled.spectrum_batch(&windowed).unwrap();
        let err = max_abs_error(&eager_spec, &comp_spec);
        assert!(err < 0.05, "reverse/forward mix spectrum err={err}");
    }

    #[test]
    fn compiled_ternary_accurate_spectrum_matches_eager() {
        let cfg = FftLearnConfig::new(64, 4).unwrap();
        let mut model = DistilledTernaryFftModel::new(64, 16, 16_000.0);
        model.gates.fill(GateMode::Skip.to_i8());
        for (i, g) in model.gates.iter_mut().enumerate() {
            if i % 3 == 0 {
                *g = GateMode::Forward.to_i8();
            }
        }
        let signal: Vec<f32> = (0..256).map(|i| (i as f32 * 0.03).sin()).collect();
        let w = crate::mel::hann_window(64);
        let mut windowed = signal.clone();
        for b in 0..4 {
            for i in 0..64 {
                windowed[b * 64 + i] *= w[i];
            }
        }
        let eager_spec = model.spectrum_batch_accurate(&windowed, 4).unwrap();
        let mut compiled = compile_distilled_ternary_mel(&model, &cfg, Device::Cpu).unwrap();
        let comp_spec = compiled.spectrum_batch(&windowed).unwrap();
        let err = max_abs_error(&eager_spec, &comp_spec);
        assert!(err < 0.05, "accurate spectrum eager vs compiled err={err}");
    }
}
