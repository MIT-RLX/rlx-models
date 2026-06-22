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

//! Plain multi-layer perceptron (`GroundingDinoMLPPredictionHead`): ReLU
//! between layers, no activation on the last. Used by the box heads and the
//! decoder reference-point head.

use crate::nn;
use crate::weights::{get, get_with_shape};
use anyhow::Result;
use rlx_core::weight_map::WeightMap;

struct Layer {
    w: Vec<f32>, // [out, in]
    b: Vec<f32>,
    in_dim: usize,
    out_dim: usize,
}

pub struct Mlp {
    layers: Vec<Layer>,
}

impl Mlp {
    /// Load `{prefix}.layers.{0..n-1}.{weight,bias}`.
    pub fn from_weights(wm: &WeightMap, prefix: &str, n: usize) -> Result<Self> {
        let layers = (0..n)
            .map(|i| {
                let (w, shape) = get_with_shape(wm, &format!("{prefix}.layers.{i}.weight"))?;
                let b = get(wm, &format!("{prefix}.layers.{i}.bias"))?;
                Ok(Layer {
                    w,
                    b,
                    out_dim: shape[0],
                    in_dim: shape[1],
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { layers })
    }

    #[cfg(test)]
    pub fn from_parts(layers: Vec<(Vec<f32>, Vec<f32>, usize, usize)>) -> Self {
        Self {
            layers: layers
                .into_iter()
                .map(|(w, b, in_dim, out_dim)| Layer {
                    w,
                    b,
                    in_dim,
                    out_dim,
                })
                .collect(),
        }
    }

    /// Forward `x [rows, in_dim]` → `[rows, last_out_dim]`.
    pub fn forward(&self, x: &[f32], rows: usize, _in_dim: usize) -> Vec<f32> {
        let mut cur = x.to_vec();
        let last = self.layers.len() - 1;
        for (i, layer) in self.layers.iter().enumerate() {
            let mut y = nn::linear(&cur, rows, layer.in_dim, &layer.w, layer.out_dim, &layer.b);
            if i != last {
                nn::relu(&mut y);
            }
            cur = y;
        }
        cur
    }

    pub fn out_dim(&self) -> usize {
        self.layers.last().map(|l| l.out_dim).unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_layer_relu_mlp() {
        // layer0: 2→2 identity, relu; layer1: 2→1 sum.
        let l0 = (vec![1.0, 0.0, 0.0, 1.0], vec![0.0, 0.0], 2, 2);
        let l1 = (vec![1.0, 1.0], vec![0.0], 2, 1);
        let mlp = Mlp::from_parts(vec![l0, l1]);
        // x = [[1,-3]] → relu([1,-3]) = [1,0] → sum = 1.
        let out = mlp.forward(&[1.0, -3.0], 1, 2);
        assert_eq!(out, vec![1.0]);
        assert_eq!(mlp.out_dim(), 1);
    }
}
