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

//! Candidate graph construction (kNN edges in space-time).
//!
//! Edges only go forward in time (`Δt > 0`), capped by
//! [`GraphConfig::max_delta_t`](crate::config::GraphConfig::max_delta_t) and
//! [`GraphConfig::n_neighbors`](crate::config::GraphConfig::n_neighbors).

use crate::config::GraphConfig;
use crate::features::NodeFeatures;
use ndarray::{Array2, Array3};

#[derive(Debug, Clone)]
pub struct GraphEdge {
    pub src: usize,
    pub dst: usize,
    pub delta_t: i32,
    pub dist: f32,
}

#[derive(Debug, Clone)]
pub struct CandidateGraph {
    pub nodes: Vec<NodeFeatures>,
    pub edges: Vec<GraphEdge>,
}

impl CandidateGraph {
    pub fn build(nodes: Vec<NodeFeatures>, cfg: &GraphConfig) -> Self {
        let mut edges = Vec::new();
        let n = nodes.len();
        for i in 0..n {
            let mut dists: Vec<(usize, f32, i32)> = Vec::new();
            for j in 0..n {
                if i == j {
                    continue;
                }
                let dt = (nodes[j].t - nodes[i].t).round() as i32;
                if dt <= 0 || dt > cfg.max_delta_t {
                    continue;
                }
                let dy = nodes[j].y - nodes[i].y;
                let dx = nodes[j].x - nodes[i].x;
                let dz = nodes[j].z - nodes[i].z;
                let dist = (dy * dy + dx * dx + dz * dz).sqrt() * cfg.spatial_scale;
                if dist <= cfg.distance_threshold {
                    dists.push((j, dist, dt));
                }
            }
            dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            for (j, dist, dt) in dists.into_iter().take(cfg.n_neighbors) {
                edges.push(GraphEdge {
                    src: i,
                    dst: j,
                    delta_t: dt,
                    dist,
                });
            }
        }
        Self { nodes, edges }
    }

    pub fn edge_indices(&self, batch: usize) -> Array3<i64> {
        let e = self.edges.len();
        let mut idx = Array3::<i64>::zeros((batch, e, 2));
        for (ei, edge) in self.edges.iter().enumerate() {
            idx[[0, ei, 0]] = edge.src as i64;
            idx[[0, ei, 1]] = edge.dst as i64;
        }
        idx
    }

    pub fn edge_pos(&self, batch: usize) -> Array3<f32> {
        let e = self.edges.len();
        let mut pos = Array3::<f32>::zeros((batch, e, 3));
        for (ei, edge) in self.edges.iter().enumerate() {
            // Match HOCT: RoPE uses spatial (z, y, x); time is in features.
            let mid_z = 0.5 * (self.nodes[edge.src].z + self.nodes[edge.dst].z);
            let mid_y = 0.5 * (self.nodes[edge.src].y + self.nodes[edge.dst].y);
            let mid_x = 0.5 * (self.nodes[edge.src].x + self.nodes[edge.dst].x);
            pos[[0, ei, 0]] = mid_z;
            pos[[0, ei, 1]] = mid_y;
            pos[[0, ei, 2]] = mid_x;
        }
        pos
    }

    pub fn node_mask(&self, batch: usize) -> Array2<bool> {
        let n = self.nodes.len();
        Array2::<bool>::from_elem((batch, n), true)
    }

    pub fn edge_mask(&self, batch: usize) -> Array2<bool> {
        let e = self.edges.len();
        Array2::<bool>::from_elem((batch, e), !self.edges.is_empty())
    }
}
