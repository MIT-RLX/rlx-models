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

//! Architecture knobs for ternary distilled FFT (ablation + deploy).

use crate::distill::band_correct::BandedCorrector;
use crate::distill::ternary_gates::{GateMode, init_ternary_gates};
use crate::model::denoise::SpectrumDenoiser;
use anyhow::Result;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GateLayout {
    /// One gate mask for mel + spectrum.
    SingleSparse,
    /// Mel uses pruned `gates`; spectrum uses `spec_gates` (typically all-forward).
    DualMelSpec,
    /// No skips — fusion-only baseline.
    AllForward,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CorrectorKind {
    Identity,
    Affine,
    BandNarrow,
    BandWide,
    Dense,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TernaryArchConfig {
    pub gate_layout: GateLayout,
    pub corrector: CorrectorKind,
    /// Greedy prune target for mel / shared gates.
    pub target_compute_fraction: f32,
    /// Spectrum-path prune target (`DualMelSpec` only; usually 1.0).
    pub target_spec_compute_fraction: f32,
    pub allow_reverse: bool,
}

impl Default for TernaryArchConfig {
    fn default() -> Self {
        Self {
            gate_layout: GateLayout::SingleSparse,
            corrector: CorrectorKind::BandWide,
            target_compute_fraction: 0.96,
            target_spec_compute_fraction: 1.0,
            allow_reverse: true,
        }
    }
}

impl TernaryArchConfig {
    pub fn fusion_only() -> Self {
        Self {
            gate_layout: GateLayout::AllForward,
            corrector: CorrectorKind::Identity,
            target_compute_fraction: 1.0,
            target_spec_compute_fraction: 1.0,
            allow_reverse: false,
        }
    }

    pub fn dual_path_mel_sparse() -> Self {
        Self {
            gate_layout: GateLayout::DualMelSpec,
            corrector: CorrectorKind::BandWide,
            target_compute_fraction: 0.94,
            target_spec_compute_fraction: 1.0,
            allow_reverse: false,
        }
    }

    pub fn single_sparse_band(target: f32) -> Self {
        Self {
            gate_layout: GateLayout::SingleSparse,
            corrector: CorrectorKind::BandWide,
            target_compute_fraction: target,
            target_spec_compute_fraction: target,
            allow_reverse: true,
        }
    }

    pub fn single_sparse_affine(target: f32) -> Self {
        Self {
            gate_layout: GateLayout::SingleSparse,
            corrector: CorrectorKind::Affine,
            target_compute_fraction: target,
            target_spec_compute_fraction: target,
            allow_reverse: false,
        }
    }

    pub fn single_sparse_dense(target: f32) -> Self {
        Self {
            gate_layout: GateLayout::SingleSparse,
            corrector: CorrectorKind::Dense,
            target_compute_fraction: target,
            target_spec_compute_fraction: target,
            allow_reverse: false,
        }
    }
}

/// Spectrum / mel correction — banded, affine, or passthrough.
#[derive(Debug, Clone)]
pub enum SpectrumCorrection {
    Identity,
    Affine(SpectrumDenoiser),
    Band(BandedCorrector),
}

impl SpectrumCorrection {
    pub fn from_kind(kind: CorrectorKind, n_fft: usize, mel_path: bool) -> Self {
        match kind {
            CorrectorKind::Identity => Self::Identity,
            CorrectorKind::Affine => Self::Affine(SpectrumDenoiser::identity(n_fft)),
            CorrectorKind::BandNarrow => Self::Band(BandedCorrector::identity(n_fft)),
            CorrectorKind::BandWide => {
                if mel_path {
                    Self::Band(BandedCorrector::identity(n_fft))
                } else {
                    Self::Band(BandedCorrector::identity_spectrum(n_fft))
                }
            }
            CorrectorKind::Dense => Self::Band(BandedCorrector::identity_dense(n_fft)),
        }
    }

    pub fn apply_batch(&self, spectrum: &[f32], batch: usize, n_fft: usize) -> Result<Vec<f32>> {
        match self {
            Self::Identity => Ok(spectrum.to_vec()),
            Self::Affine(d) => d.apply_batch(spectrum, batch, n_fft),
            Self::Band(b) => b.apply_batch(spectrum, batch, n_fft),
        }
    }

    pub fn train_step_mse(
        &mut self,
        input: &[f32],
        target: &[f32],
        batch: usize,
        n_fft: usize,
        lr: f32,
    ) -> Result<f32> {
        match self {
            Self::Identity => Ok(0.0),
            Self::Affine(d) => d.train_step_affine(input, target, batch, n_fft, lr),
            Self::Band(b) => b.train_step_mse(input, target, batch, n_fft, lr),
        }
    }

    pub fn train_step_spectrum_grad(
        &mut self,
        input: &[f32],
        spec_grad: &[f32],
        batch: usize,
        n_fft: usize,
        lr: f32,
    ) {
        if let Self::Band(b) = self {
            b.train_step_spectrum_grad(input, spec_grad, batch, n_fft, lr);
        }
    }

    pub fn dense_rhs_with_freq_mask(&self, freq_mask: &[f32]) -> Option<Vec<f32>> {
        match self {
            Self::Band(b) => Some(b.dense_rhs_with_freq_mask(freq_mask)),
            _ => None,
        }
    }

    pub fn bias(&self) -> Option<Vec<f32>> {
        match self {
            Self::Affine(d) => Some(d.bias.clone()),
            Self::Band(b) => Some(b.bias.clone()),
            Self::Identity => None,
        }
    }

    pub fn affine_gain_bias(&self) -> Option<(Vec<f32>, Vec<f32>)> {
        match self {
            Self::Affine(d) => Some((d.scale.clone(), d.bias.clone())),
            _ => None,
        }
    }
}

pub fn all_forward_gates(n_fft: usize) -> Vec<i8> {
    init_ternary_gates(n_fft)
}

pub fn strip_reverse_gates(gates: &mut [i8]) {
    for g in gates.iter_mut() {
        if *g == GateMode::Reverse.to_i8() {
            *g = GateMode::Forward.to_i8();
        }
    }
}

pub fn sync_spec_gates_for_layout(
    gate_layout: GateLayout,
    mel_gates: &[i8],
    spec_gates: &mut [i8],
) {
    assert_eq!(mel_gates.len(), spec_gates.len());
    match gate_layout {
        GateLayout::DualMelSpec => {
            spec_gates.fill(GateMode::Forward.to_i8());
        }
        GateLayout::AllForward | GateLayout::SingleSparse => {
            spec_gates.copy_from_slice(mel_gates);
        }
    }
}
