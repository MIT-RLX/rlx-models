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

//! DINOv2 graph builder — delegates to [`super::flow::DinoV2Flow`].
//!
//! Input: `"hidden"` `[batch, seq, hidden_size]` — caller assembles
//!   `[CLS, register_tokens…, projected_patches] + pos_embed`.
//!   (See `crate::dinov2::preprocess::assemble_hidden`.)

use super::config::DinoV2Config;
use super::preprocess::DinoV2PreprocessWeights;
use anyhow::Result;
use rlx_core::weight_map::WeightMap;
use rlx_ir::Graph;

/// Build the DINOv2 IR graph via native `ModelFlow`.
pub fn build_dinov2_graph_sized(
    cfg: &DinoV2Config,
    weights: &mut WeightMap,
    batch: usize,
) -> Result<(
    Graph,
    std::collections::HashMap<String, Vec<f32>>,
    DinoV2PreprocessWeights,
)> {
    let built = super::flow::build_dinov2_built(cfg, weights, batch)?;
    let (graph, params) = rlx_core::flow_util::graph_from_built(built.model)?;
    Ok((graph, params, built.preprocess))
}
