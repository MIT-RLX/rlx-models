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

//! Native SAM3 detector scaffolding.

use super::config::Sam3DetectorConfig;
use super::geometry::Sam3GeometryFeatures;
use super::neck::Sam3FeatureLevel;
use super::text_encoder::Sam3TextEncoded;
use anyhow::{Result, ensure};

#[derive(Debug, Clone, Default)]
pub struct Sam3DetectorWeights {
    pub loaded: bool,
}

#[derive(Debug, Clone)]
pub struct Sam3DetectorOutput {
    pub query_features: Vec<f32>,
    pub boxes: Vec<f32>,
    pub scores: Vec<f32>,
    pub num_queries: usize,
    pub dim: usize,
}

pub fn detector_forward_native(
    _weights: &Sam3DetectorWeights,
    cfg: &Sam3DetectorConfig,
    levels: &[Sam3FeatureLevel],
    text: &Sam3TextEncoded,
    geometry: &Sam3GeometryFeatures,
) -> Result<Sam3DetectorOutput> {
    ensure!(
        !levels.is_empty(),
        "SAM3 detector needs at least one feature level"
    );
    let level = &levels[0];
    ensure!(
        level.channels == cfg.d_model,
        "SAM3 detector feature dim mismatch"
    );
    let mut pooled = vec![0.0; cfg.d_model];
    let rows = level.h * level.w;
    for r in 0..rows {
        for c in 0..cfg.d_model {
            pooled[c] += level.features[r * cfg.d_model + c] / rows as f32;
        }
    }
    if !text.text_memory_resized.is_empty() {
        for c in 0..cfg.d_model {
            pooled[c] += text.text_memory_resized[c] * 0.01;
        }
    }
    for c in 0..cfg.d_model {
        pooled[c] += geometry.features[c] * 0.001;
    }
    let mut query_features = vec![0.0; cfg.num_queries * cfg.d_model];
    for q in 0..cfg.num_queries {
        query_features[q * cfg.d_model..(q + 1) * cfg.d_model].copy_from_slice(&pooled);
    }
    Ok(Sam3DetectorOutput {
        query_features,
        boxes: vec![0.0, 0.0, 1.0, 1.0],
        scores: vec![pooled.iter().copied().sum::<f32>() / cfg.d_model as f32],
        num_queries: cfg.num_queries,
        dim: cfg.d_model,
    })
}
