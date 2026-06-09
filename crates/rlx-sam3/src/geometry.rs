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

//! Native SAM3 geometry prompt scaffolding.

use super::config::SAM3_DET_DIM;

#[derive(Debug, Clone, Default)]
pub struct Sam3GeometryWeights {
    pub loaded: bool,
}

#[derive(Debug, Clone)]
pub struct Sam3GeometryFeatures {
    pub features: Vec<f32>,
    pub tokens: usize,
    pub dim: usize,
}

pub fn encode_geometry_native(
    _weights: &Sam3GeometryWeights,
    boxes: Option<&[f32]>,
    points: Option<(&[f32], &[f32])>,
) -> Sam3GeometryFeatures {
    let box_tokens = boxes.map(|b| b.len() / 4).unwrap_or(0);
    let point_tokens = points.map(|(p, _)| p.len() / 2).unwrap_or(0);
    let tokens = (box_tokens + point_tokens).max(1);
    let mut features = vec![0.0; tokens * SAM3_DET_DIM];
    if let Some(b) = boxes {
        for (i, chunk) in b.chunks_exact(4).enumerate() {
            for (j, v) in chunk.iter().enumerate() {
                features[i * SAM3_DET_DIM + j] = *v;
            }
        }
    }
    if let Some((coords, labels)) = points {
        let base = box_tokens;
        for (i, xy) in coords.chunks_exact(2).enumerate() {
            let row = base + i;
            if row >= tokens {
                break;
            }
            features[row * SAM3_DET_DIM] = xy[0];
            features[row * SAM3_DET_DIM + 1] = xy[1];
            features[row * SAM3_DET_DIM + 2] = labels.get(i).copied().unwrap_or(0.0);
        }
    }
    Sam3GeometryFeatures {
        features,
        tokens,
        dim: SAM3_DET_DIM,
    }
}
