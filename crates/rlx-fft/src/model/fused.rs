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

//! Fused FFT → spectral mask → IFFT (Tier A).

use crate::model::butterfly::{butterfly_forward_real_batch, butterfly_inverse_complex_batch};
use crate::model::config::FftLearnConfig;
use crate::model::reference::roundtrip_scale;
use crate::model::twiddle::exact_twiddles;
use anyhow::{Result, ensure};
use rlx_ir::infer::GraphExt;
use rlx_ir::{DType, Graph, Shape};

/// Eager fused roundtrip: real signal → FFT → complex multiply → IFFT → scaled real.
pub fn fused_spectral_eager(
    signal: &[f32],
    twiddles: &[f32],
    mask: &[f32],
    batch: usize,
    n_fft: usize,
) -> Result<Vec<f32>> {
    ensure!(mask.len() == n_fft * 2);
    let spec = butterfly_forward_real_batch(signal, twiddles, batch, n_fft)?;
    let mut masked = spec.clone();
    for b in 0..batch {
        for i in 0..n_fft * 2 {
            masked[b * n_fft * 2 + i] *= mask[i];
        }
    }
    butterfly_inverse_complex_batch(&masked, twiddles, batch, n_fft)
}

pub fn unit_mask(n_fft: usize) -> Vec<f32> {
    vec![1f32; n_fft * 2]
}

pub fn build_fused_spectral_graph(cfg: &FftLearnConfig) -> Result<(Graph, Vec<String>)> {
    let n = cfg.n_fft;
    let batch = cfg.batch;
    let f = DType::F32;
    let mut g = Graph::new("fused_spectral");
    let signal = g.input("signal", Shape::new(&[batch, n], f));
    let zeros = g.sub(signal, signal);
    let block = g.concat_(vec![signal, zeros], 1);
    let spec = g.fft(block, false);
    let mut param_names = Vec::new();
    let mut masked_parts = Vec::new();
    for i in 0..(n * 2) {
        let name = format!("mask.{i}");
        let w = g.param(&name, Shape::new(&[1], f));
        param_names.push(name);
        let col = g.narrow_(spec, 1, i, 1);
        masked_parts.push(g.mul(col, w));
    }
    let masked = g.concat_(masked_parts, 1);
    let out = g.fft(masked, true);
    g.set_outputs(vec![out]);
    Ok((g, param_names))
}

pub fn fused_roundtrip_error(
    signal: &[f32],
    twiddles: &[f32],
    mask: &[f32],
    batch: usize,
    n_fft: usize,
) -> Result<f32> {
    let recovered = fused_spectral_eager(signal, twiddles, mask, batch, n_fft)?;
    let scale = roundtrip_scale(n_fft);
    let mut max_err = 0f32;
    for b in 0..batch {
        for i in 0..n_fft {
            let base = b * n_fft * 2 + i * 2;
            let expected = signal[b * n_fft + i] * scale;
            max_err = max_err.max((recovered[base] - expected).abs());
        }
    }
    Ok(max_err)
}

pub fn default_twiddles(cfg: &FftLearnConfig) -> Vec<f32> {
    exact_twiddles(cfg)
}
