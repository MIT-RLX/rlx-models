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

//! Temporal sliding windows with padding for batched HOCT inference.

use crate::config::GraphConfig;
use crate::features::{NodeFeatures, nodes_to_array, regionprops_2d};
use crate::graph::CandidateGraph;
use ndarray::{Array2, Array3, Axis};

#[derive(Debug, Clone)]
pub struct WindowBatch {
    pub node_features: Array3<f32>,
    pub node_pos: Array3<f32>,
    pub edge_pos: Array3<f32>,
    pub edge_indices: Array3<i64>,
    pub node_mask: Array2<bool>,
    pub edge_mask: Array2<bool>,
    pub frame_t: i32,
}

#[derive(Debug, Clone, Default)]
pub struct TemporalDataset {
    pub window_size: usize,
    pub stride: usize,
    pub graph_cfg: GraphConfig,
}

impl TemporalDataset {
    pub fn windows_from_labels(
        &self,
        labels: &Array3<u32>,
        images: Option<&Array3<f32>>,
    ) -> Vec<WindowBatch> {
        let t_max = labels.len_of(Axis(0));
        let mut out = Vec::new();
        let mut t = 0i32;
        while t < t_max as i32 {
            let end = (t + self.window_size as i32).min(t_max as i32);
            let mut nodes: Vec<NodeFeatures> = Vec::new();
            for fi in t..end {
                let lab2 = labels.slice(ndarray::s![fi..fi + 1, .., ..]).to_owned();
                let img2 = images.map(|img| img.slice(ndarray::s![fi..fi + 1, .., ..]).to_owned());
                let mut frame_nodes =
                    regionprops_2d(&lab2, fi, img2.as_ref().map(|a| a.view()), 1.0);
                nodes.append(&mut frame_nodes);
            }
            if !nodes.is_empty() {
                let graph = CandidateGraph::build(nodes, &self.graph_cfg);
                let feats = nodes_to_array(&graph.nodes);
                let pos = crate::features::nodes_to_pos(&graph.nodes);
                let batch = WindowBatch {
                    node_features: feats,
                    node_pos: pos,
                    edge_pos: graph.edge_pos(1),
                    edge_indices: graph.edge_indices(1),
                    node_mask: graph.node_mask(1),
                    edge_mask: graph.edge_mask(1),
                    frame_t: t,
                };
                out.push(batch);
            }
            t += self.stride as i32;
        }
        out
    }

    /// Pad variable-size graphs to `(B, N_max, ·)` / `(B, E_max, ·)`.
    pub fn pad_batches(batches: &[WindowBatch]) -> Option<(WindowBatch, usize, usize)> {
        if batches.is_empty() {
            return None;
        }
        let b = batches.len();
        let n_max = batches
            .iter()
            .map(|w| w.node_features.len_of(Axis(1)))
            .max()?;
        let e_max = batches
            .iter()
            .map(|w| w.edge_indices.len_of(Axis(1)))
            .max()?;
        let d = batches[0].node_features.len_of(Axis(2));

        let mut node_features = Array3::<f32>::zeros((b, n_max, d));
        let mut node_pos = Array3::<f32>::zeros((b, n_max, 3));
        let mut edge_pos = Array3::<f32>::zeros((b, e_max, 3));
        let mut edge_indices = Array3::<i64>::zeros((b, e_max, 2));
        let mut node_mask = Array2::<bool>::from_elem((b, n_max), false);
        let mut edge_mask = Array2::<bool>::from_elem((b, e_max), false);

        for (bi, w) in batches.iter().enumerate() {
            let n = w.node_features.len_of(Axis(1));
            let e = w.edge_indices.len_of(Axis(1));
            for i in 0..n {
                for k in 0..d {
                    node_features[[bi, i, k]] = w.node_features[[0, i, k]];
                }
                for k in 0..3 {
                    node_pos[[bi, i, k]] = w.node_pos[[0, i, k]];
                }
                node_mask[[bi, i]] = w.node_mask[[0, i]];
            }
            for ei in 0..e {
                for k in 0..2 {
                    edge_indices[[bi, ei, k]] = w.edge_indices[[0, ei, k]];
                }
                for k in 0..3 {
                    edge_pos[[bi, ei, k]] = w.edge_pos[[0, ei, k]];
                }
                edge_mask[[bi, ei]] = w.edge_mask[[0, ei]];
            }
        }

        Some((
            WindowBatch {
                node_features,
                node_pos,
                edge_pos,
                edge_indices,
                node_mask,
                edge_mask,
                frame_t: batches[0].frame_t,
            },
            n_max,
            e_max,
        ))
    }
}
