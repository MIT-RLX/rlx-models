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

//! Conditioning vector `vec` (host side): sinusoidal timestep / guidance
//! embeddings through their MLPs, plus the pooled-CLIP projection.
//!
//! These are three 2-layer MLPs on a single row — a few dozen MFLOP — so they
//! run on the host in f32 rather than as graph nodes; the per-block
//! modulation linears that consume `vec` (a quarter of the model) stay in the
//! graph.

use anyhow::{Result, ensure};
use rayon::prelude::*;

use crate::config::{Flux1Config, TIME_EMBED_DIM};
use crate::weights::DitWeights;

/// BFL `timestep_embedding(t, dim)`: `t · 1000`, frequencies
/// `exp(-ln(10000) · i / half)`, output `[cos | sin]`.
pub fn timestep_embedding(t: f32, dim: usize) -> Vec<f32> {
    let half = dim / 2;
    let t = t * 1000.0;
    let mut out = vec![0.0f32; dim];
    for i in 0..half {
        let freq = (-(10_000f32.ln()) * i as f32 / half as f32).exp();
        let arg = t * freq;
        out[i] = arg.cos();
        out[half + i] = arg.sin();
    }
    out
}

/// A dense layer held on the host: `y = W x + b`, `W` row-major `[out, in]`.
#[derive(Debug, Clone)]
pub struct HostLinear {
    pub w: Vec<f32>,
    pub b: Vec<f32>,
    pub out_dim: usize,
    pub in_dim: usize,
}

impl HostLinear {
    pub fn load(src: &mut dyn DitWeights, prefix: &str) -> Result<Self> {
        let (w, shape) = src.load_merged(&format!("{prefix}.weight"))?;
        let (b, _) = src.load_merged(&format!("{prefix}.bias"))?;
        ensure!(
            shape.len() == 2 && b.len() == shape[0],
            "{prefix}: bad linear shape {shape:?}"
        );
        Ok(Self {
            w,
            b,
            out_dim: shape[0],
            in_dim: shape[1],
        })
    }

    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        debug_assert_eq!(x.len(), self.in_dim);
        self.w
            .par_chunks(self.in_dim)
            .zip(self.b.par_iter())
            .map(|(row, &b)| {
                row.iter()
                    .zip(x)
                    .map(|(&w, &v)| w as f64 * v as f64)
                    .sum::<f64>() as f32
                    + b
            })
            .collect()
    }
}

/// `out_layer(silu(in_layer(x)))`.
#[derive(Debug, Clone)]
pub struct MlpEmbedder {
    pub in_layer: HostLinear,
    pub out_layer: HostLinear,
}

impl MlpEmbedder {
    pub fn load(src: &mut dyn DitWeights, prefix: &str) -> Result<Self> {
        Ok(Self {
            in_layer: HostLinear::load(src, &format!("{prefix}.in_layer"))?,
            out_layer: HostLinear::load(src, &format!("{prefix}.out_layer"))?,
        })
    }

    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        let h: Vec<f32> = self
            .in_layer
            .forward(x)
            .into_iter()
            .map(|v| v / (1.0 + (-v).exp()))
            .collect();
        self.out_layer.forward(&h)
    }
}

/// The three conditioning MLPs.
#[derive(Debug, Clone)]
pub struct VecEmbedder {
    pub time_in: MlpEmbedder,
    pub guidance_in: Option<MlpEmbedder>,
    pub vector_in: MlpEmbedder,
}

impl VecEmbedder {
    pub fn load(src: &mut dyn DitWeights, cfg: &Flux1Config) -> Result<Self> {
        Ok(Self {
            time_in: MlpEmbedder::load(src, "time_in")?,
            guidance_in: if cfg.guidance_embed {
                Some(MlpEmbedder::load(src, "guidance_in")?)
            } else {
                None
            },
            vector_in: MlpEmbedder::load(src, "vector_in")?,
        })
    }

    /// `vec = time_in(emb(t)) [+ guidance_in(emb(g))] + vector_in(pooled)`.
    /// `t` is the flow time in `[0, 1]`; `guidance` is ignored by `[schnell]`.
    pub fn forward(&self, t: f32, guidance: f32, pooled: &[f32]) -> Vec<f32> {
        let mut v = self.time_in.forward(&timestep_embedding(t, TIME_EMBED_DIM));
        if let Some(g) = &self.guidance_in {
            let gv = g.forward(&timestep_embedding(guidance, TIME_EMBED_DIM));
            v.iter_mut().zip(gv).for_each(|(a, b)| *a += b);
        }
        let y = self.vector_in.forward(pooled);
        v.iter_mut().zip(y).for_each(|(a, b)| *a += b);
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timestep_embedding_is_cos_then_sin() {
        let e = timestep_embedding(0.5, 8);
        assert!((e[0] - 500f32.cos()).abs() < 1e-6);
        assert!((e[4] - 500f32.sin()).abs() < 1e-6);
        let f1 = (-(10_000f32.ln()) / 4.0).exp();
        assert!((e[1] - (500.0 * f1).cos()).abs() < 1e-5);
    }
}
