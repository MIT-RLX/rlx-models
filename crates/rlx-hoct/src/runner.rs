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

//! End-to-end HOCT tracking: features → graph → model → softmax → ILP.
//!
//! Prefer [`HoctRunner::builder`] for CLI and library use. For device-backed
//! score heads alone, see [`crate::device::HoctDeviceRunner`].

use crate::config::{GraphConfig, HoctConfig, IlpWeights};
use crate::dataset::{TemporalDataset, WindowBatch};
use crate::features::{NodeFeatures, nodes_to_array, nodes_to_pos, regionprops_2d};
use crate::graph::CandidateGraph;
use crate::ilp::{IlpSolution, solve_tracklets};
use crate::io::OutputFormat;
use crate::model::HoctModel;
use crate::softmax::logits_to_window_rows;
use anyhow::{Context, Result};
use ndarray::{Array3, Axis};
use std::path::{Path, PathBuf};

/// Full tracking pipeline over a label volume.
#[derive(Debug, Clone)]
pub struct HoctRunner {
    /// Eager HOCT model.
    pub model: HoctModel,
    /// Candidate-graph hyper-parameters.
    pub graph_cfg: GraphConfig,
    ilp_weights: IlpWeights,
    dataset: TemporalDataset,
}

/// Builder for [`HoctRunner`].
#[derive(Debug, Clone, Default)]
pub struct HoctRunnerBuilder {
    weights: Option<PathBuf>,
    graph_cfg: GraphConfig,
    ilp_weights: IlpWeights,
    window_size: usize,
    stride: usize,
}

impl HoctRunnerBuilder {
    /// Path to `general_v0.safetensors` (required).
    pub fn weights<P: Into<PathBuf>>(mut self, p: P) -> Self {
        self.weights = Some(p.into());
        self
    }

    /// Override candidate-graph settings.
    pub fn graph_cfg(mut self, cfg: GraphConfig) -> Self {
        self.graph_cfg = cfg;
        self
    }

    /// Override ILP objective weights.
    pub fn ilp_weights(mut self, w: IlpWeights) -> Self {
        self.ilp_weights = w;
        self
    }

    /// Temporal window length in frames (default 5).
    pub fn window_size(mut self, n: usize) -> Self {
        self.window_size = n;
        self
    }

    /// Window stride in frames (default 1).
    pub fn stride(mut self, n: usize) -> Self {
        self.stride = n;
        self
    }

    /// Load weights and return a ready runner.
    pub fn build(self) -> Result<HoctRunner> {
        let weights = self
            .weights
            .ok_or_else(|| anyhow::anyhow!("--weights / -m is required"))?;
        let model = HoctModel::from_weights(&weights)?;
        let window_size = if self.window_size == 0 {
            5
        } else {
            self.window_size
        };
        let stride = if self.stride == 0 { 1 } else { self.stride };
        Ok(HoctRunner {
            model,
            graph_cfg: self.graph_cfg,
            ilp_weights: self.ilp_weights,
            dataset: TemporalDataset {
                window_size,
                stride,
                graph_cfg: self.graph_cfg,
            },
        })
    }
}

impl HoctRunner {
    /// Start a builder (`weights` is required before [`HoctRunnerBuilder::build`]).
    pub fn builder() -> HoctRunnerBuilder {
        HoctRunnerBuilder::default()
    }

    /// Model hyper-parameters.
    pub fn config(&self) -> &HoctConfig {
        &self.model.cfg
    }

    /// Score one window batch with the eager model.
    pub fn forward_batch(&self, batch: &WindowBatch) -> crate::model::HoctOutput {
        self.model.forward(
            &batch.node_features.view(),
            &batch.node_pos.view(),
            &batch.edge_pos.view(),
            &batch.edge_indices,
            &batch.node_mask,
            &batch.edge_mask,
        )
    }

    /// Padded multi-window forward (eager kernels; same math as unpadded).
    pub fn forward_padded(&self, batches: &[WindowBatch]) -> Option<crate::model::HoctOutput> {
        let (padded, _, _) = TemporalDataset::pad_batches(batches)?;
        Some(self.forward_batch(&padded))
    }

    /// Build nodes + candidate graph for a label volume.
    pub fn build_graph(
        &self,
        labels: &Array3<u32>,
        images: Option<&Array3<f32>>,
    ) -> CandidateGraph {
        let t_max = labels.len_of(Axis(0));
        let mut all_nodes = Vec::new();
        for fi in 0..t_max {
            let lab2 = labels.slice(ndarray::s![fi..fi + 1, .., ..]).to_owned();
            let img2 = images.map(|img| img.slice(ndarray::s![fi..fi + 1, .., ..]).to_owned());
            let mut frame_nodes =
                regionprops_2d(&lab2, fi as i32, img2.as_ref().map(|a| a.view()), 1.0);
            all_nodes.append(&mut frame_nodes);
        }
        CandidateGraph::build(all_nodes, &self.graph_cfg)
    }

    /// Track a `(T,Y,X)` label volume; optional intensity `(T,Y,X)`.
    ///
    /// Returns the ILP solution and the node list used to build the graph
    /// (needed for CTC / GEFF export).
    pub fn track_labels(
        &self,
        labels: &Array3<u32>,
        images: Option<&Array3<f32>>,
    ) -> Result<(IlpSolution, Vec<NodeFeatures>)> {
        let graph = self.build_graph(labels, images);
        let node_times: Vec<f32> = graph.nodes.iter().map(|n| n.t).collect();
        let edge_ids: Vec<usize> = (0..graph.edges.len()).collect();

        // Score full candidate graph (stable global edge ids). Windowed
        // scoring is equivalent when the window covers all frames.
        let feats = nodes_to_array(&graph.nodes);
        let pos = nodes_to_pos(&graph.nodes);
        let batch = WindowBatch {
            node_features: feats,
            node_pos: pos,
            edge_pos: graph.edge_pos(1),
            edge_indices: graph.edge_indices(1),
            node_mask: graph.node_mask(1),
            edge_mask: graph.edge_mask(1),
            frame_t: 0,
        };
        let out = self.forward_batch(&batch);
        let rows = logits_to_window_rows(
            &batch.edge_indices,
            &out.edge_logits,
            &edge_ids,
            &node_times,
        );
        let edge_windows = vec![rows];
        let orphan_windows = vec![
            (0..graph.nodes.len())
                .map(|i| (i, out.orphan_logits[[0, i, 0]]))
                .collect(),
        ];

        let (edges, orphans) = crate::softmax::parental_softmax_aggregate(
            &edge_windows,
            &orphan_windows,
            self.ilp_weights.delta_t_weight,
        );
        let sol = solve_tracklets(&edges, &orphans, &node_times, &self.ilp_weights)
            .context("tracklet ILP")?;
        Ok((sol, graph.nodes))
    }

    /// Load labels from a raw file, track, and optionally write CTC/GEFF.
    pub fn track_path(
        &self,
        labels_path: impl AsRef<Path>,
        out_dir: Option<&Path>,
        format: OutputFormat,
    ) -> Result<IlpSolution> {
        let labels = crate::io::load_labels_raw(labels_path)?;
        let (sol, nodes) = self.track_labels(&labels, None)?;
        if let Some(dir) = out_dir {
            crate::io::write_solution(dir, &labels, &nodes, &sol, format)?;
        }
        Ok(sol)
    }

    /// Expose dataset for windowed / compiled paths.
    pub fn dataset(&self) -> &TemporalDataset {
        &self.dataset
    }
}
