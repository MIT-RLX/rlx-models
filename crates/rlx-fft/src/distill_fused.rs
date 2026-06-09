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

//! Shared fused deploy helpers — Hann window, correction, log-mel, device routing.

use crate::config::FftLearnConfig;
use crate::mel::hann_window;
use rlx_ir::infer::GraphExt;
use rlx_ir::{DType, Graph, NodeId, Shape};
use rlx_runtime::Device;

/// Pick compile/run device for fused distilled graphs.
///
/// Small speech batches on GPU pay more in launch/sync than they gain from FFT offload;
/// CPU fused graphs are often faster vs `rustfft` there.
pub fn pick_fused_deploy_device(batch: usize, n_fft: usize, requested: Device) -> Device {
    match requested {
        Device::Metal
        | Device::Cuda
        | Device::Mlx
        | Device::Gpu
        | Device::Rocm
        | Device::Vulkan => {
            if batch <= 64 && n_fft <= 512 {
                Device::Cpu
            } else {
                requested
            }
        }
        other => other,
    }
}

/// `signal [batch, n]` → Hann-windowed `[batch, n]` (param `hann`).
pub fn append_hann_window(
    g: &mut Graph,
    signal: NodeId,
    batch: usize,
    n_fft: usize,
) -> (NodeId, &'static str) {
    let f = DType::F32;
    let hann = g.param("hann", Shape::new(&[n_fft], f));
    let hann_bc = g.reshape_(hann, vec![1, n_fft as i64]);
    let windowed = g.mul(signal, hann_bc);
    let _ = batch;
    (windowed, "hann")
}

/// Interleaved spectrum `[batch, n*2]` → per-bin affine correction.
pub fn append_spectrum_correction(
    g: &mut Graph,
    spectrum: NodeId,
    cfg: &FftLearnConfig,
    param_names: &mut Vec<String>,
) -> NodeId {
    let flat_len = cfg.n_fft * 2;
    let batch = cfg.batch;
    let f = DType::F32;
    let flat = g.reshape_(spectrum, vec![batch as i64, flat_len as i64]);
    let gain = g.param("corr.gain", Shape::new(&[flat_len], f));
    let bias = g.param("corr.bias", Shape::new(&[flat_len], f));
    param_names.push("corr.gain".into());
    param_names.push("corr.bias".into());
    let scaled = g.mul(flat, gain);
    g.add(scaled, bias)
}

/// Fused banded correction: one `matmul` + bias (freq mask folded into weights at bake).
pub fn append_banded_spectrum_correction(
    g: &mut Graph,
    spectrum: NodeId,
    cfg: &FftLearnConfig,
    prefix: &str,
    param_names: &mut Vec<String>,
) -> NodeId {
    let flat_len = cfg.n_fft * 2;
    let batch = cfg.batch;
    let f = DType::F32;
    let flat = g.reshape_(spectrum, vec![batch as i64, flat_len as i64]);
    let w_name = format!("{prefix}.band_rhs");
    let b_name = format!("{prefix}.bias");
    let w = g.param(&w_name, Shape::new(&[flat_len, flat_len], f));
    let bias = g.param(&b_name, Shape::new(&[flat_len], f));
    param_names.push(w_name);
    param_names.push(b_name);
    let out_shape = Shape::new(&[batch, flat_len], f);
    let mixed = g.matmul(flat, w, out_shape);
    let bias_bc = g.reshape_(bias, vec![1, flat_len as i64]);
    g.add(mixed, bias_bc)
}

/// Corrected spectrum → `Op::LogMel` (param `mel.filters`).
pub fn append_log_mel_head(
    g: &mut Graph,
    corrected: NodeId,
    cfg: &FftLearnConfig,
    n_mels: usize,
    param_names: &mut Vec<String>,
) -> NodeId {
    let n_bins = cfg.n_fft / 2 + 1;
    let f = DType::F32;
    let filters = g.param("mel.filters", Shape::new(&[n_mels, n_bins], f));
    param_names.push("mel.filters".into());
    g.log_mel(corrected, filters)
}

pub fn load_hann_param(compiled: &mut rlx_runtime::CompiledGraph, n_fft: usize) {
    let w = hann_window(n_fft);
    compiled.set_param("hann", &w);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fused_hann_graph_builds() {
        let cfg = FftLearnConfig::new(64, 4).unwrap();
        let mut g = Graph::new("test_hann");
        let f = DType::F32;
        let signal = g.input("signal", Shape::new(&[cfg.batch, cfg.n_fft], f));
        let (windowed, _) = append_hann_window(&mut g, signal, cfg.batch, cfg.n_fft);
        g.set_outputs(vec![windowed]);
    }
}
