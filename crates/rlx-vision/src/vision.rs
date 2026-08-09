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

//! NomicVision graph builder — delegates to [`crate::flow::NomicVisionFlow`].

use anyhow::Result;
use rlx_core::config::NomicVisionConfig;
use rlx_core::weight_map::WeightMap;
use rlx_ir::Graph;
use std::collections::HashMap;

/// Build a NomicVision encoder IR graph via native `ModelFlow`.
pub fn build_vision_graph_sized(
    cfg: &NomicVisionConfig,
    weights: &mut WeightMap,
    batch: usize,
) -> Result<(Graph, HashMap<String, Vec<f32>>, VisionPreprocessWeights)> {
    let built = crate::flow::build_nomic_vision_built(cfg, weights, batch)?;
    let (graph, params) = rlx_core::flow_util::graph_from_built(built.model)?;
    Ok((graph, params, built.preprocess))
}

/// Preprocessing weights extracted from safetensors for the caller to
/// assemble the "hidden" input before graph execution.
pub struct VisionPreprocessWeights {
    /// Patch projection weight [patch_dim, H] (pre-transposed for sgemm)
    pub proj_w: Vec<f32>,
    /// Number of columns in proj_w (= hidden_size)
    pub proj_w_cols: usize,
    /// Patch projection bias \[H\]
    pub proj_b: Vec<f32>,
    /// CLS token \[H\] (or [1, 1, H] flattened)
    pub cls_token: Vec<f32>,
    /// Position embeddings [1+np, H] (or [1, 1+np, H] flattened)
    pub pos_embed: Vec<f32>,
}
