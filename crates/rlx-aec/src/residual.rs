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

//! Per-bin spectral residual echo suppressor (RLX-compatible affine mask).

use anyhow::{Context, Result, ensure};
use safetensors::SafeTensors;

#[derive(Debug, Clone)]
pub struct ResidualWeights {
    pub n_fft: usize,
    pub scale: Vec<f32>,
    pub bias: Vec<f32>,
}

impl ResidualWeights {
    pub fn identity(n_fft: usize) -> Self {
        Self {
            n_fft,
            scale: vec![1.0; n_fft * 2],
            bias: vec![0.0; n_fft * 2],
        }
    }

    pub fn from_safetensors_bytes(bytes: &[u8]) -> Result<Self> {
        let tensors = SafeTensors::deserialize(bytes).context("parse residual safetensors")?;
        let scale = tensor_f32(&tensors, "scale")?;
        let bias = tensor_f32(&tensors, "bias")?;
        ensure!(scale.len() == bias.len(), "scale/bias length mismatch");
        let n_fft = scale.len() / 2;
        ensure!(n_fft >= 64 && n_fft.is_power_of_two(), "invalid n_fft");
        Ok(Self { n_fft, scale, bias })
    }

    pub fn apply_spectrum(&self, spectrum: &mut [f32]) {
        let n = self.n_fft * 2;
        if spectrum.len() < n {
            return;
        }
        for i in 0..n {
            spectrum[i] = spectrum[i] * self.scale[i] + self.bias[i];
        }
    }
}

fn tensor_f32(tensors: &SafeTensors, name: &str) -> Result<Vec<f32>> {
    let t = tensors
        .tensor(name)
        .with_context(|| format!("missing tensor {name}"))?;
    ensure!(t.dtype() == safetensors::Dtype::F32, "{name}: expected F32");
    let bytes = t.data();
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

#[cfg(feature = "residual")]
mod embedded {
    use super::ResidualWeights;
    use anyhow::Result;

    const SAFETENSORS: &[u8] = include_bytes!("../weights/residual_aec.safetensors");

    pub fn load_embedded() -> Result<ResidualWeights> {
        ResidualWeights::from_safetensors_bytes(SAFETENSORS)
    }
}

#[cfg(feature = "residual")]
pub fn embedded_residual_weights() -> Result<ResidualWeights> {
    embedded::load_embedded()
}

#[cfg(not(feature = "residual"))]
pub fn embedded_residual_weights() -> Result<ResidualWeights> {
    anyhow::bail!("residual feature disabled");
}
