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

//! Compiled / padded HOCT path and edge-head [`ModelFlow`](rlx_flow::ModelFlow).
//!
//! Gated 3D-RoPE attention is not yet a first-class flow stage.
//! [`HoctCompiled`] pads variable graphs to a fixed `(N_max, E_max)` contract
//! and runs the same eager kernels as [`crate::model::HoctModel`] (bit-identical
//! on the live mask). [`HoctFlow::build_head_flow`] builds the LayerNorm→Linear
//! score head for backend compile coverage.

use crate::config::HoctConfig;
use crate::dataset::{TemporalDataset, WindowBatch};
use crate::model::{HoctModel, HoctOutput};
use crate::weights::{HoctWeights, load_hoct_weights};
use anyhow::{Context, Result};
use rlx_core::flow_util::WeightMapSource;
use rlx_core::weight_map::WeightMap;
use rlx_flow::{BuiltModel, CompileProfile};
use rlx_ir::{DType, Shape};
use rlx_runtime::Device;
use std::collections::HashMap;
use std::path::Path;

/// Flow factory: eager model, padded compiled runner, and score-head graph.
#[derive(Debug, Clone)]
pub struct HoctFlow {
    pub cfg: HoctConfig,
    /// Pad width for nodes when building compile-shaped batches.
    pub max_nodes: usize,
    /// Pad width for edges (also sizes the compiled score head).
    pub max_edges: usize,
    /// Batch size for the compiled score head.
    pub batch: usize,
}

impl HoctFlow {
    /// Defaults: `max_nodes=256`, `max_edges=1024`, `batch=1`.
    pub fn new(cfg: HoctConfig) -> Self {
        Self {
            cfg,
            max_nodes: 256,
            max_edges: 1024,
            batch: 1,
        }
    }

    pub fn with_pad(mut self, max_nodes: usize, max_edges: usize) -> Self {
        self.max_nodes = max_nodes;
        self.max_edges = max_edges;
        self
    }

    pub fn with_batch(mut self, batch: usize) -> Self {
        self.batch = batch;
        self
    }

    pub fn build_eager(&self, weights: HoctWeights) -> HoctModel {
        HoctModel::new(self.cfg.clone(), weights)
    }

    pub fn build_from_path(&self, path: impl AsRef<Path>) -> Result<HoctModel> {
        HoctModel::from_weights(path)
    }

    pub fn build_compiled(&self, weights: HoctWeights) -> HoctCompiled {
        HoctCompiled {
            model: HoctModel::new(self.cfg.clone(), weights),
            max_nodes: self.max_nodes,
            max_edges: self.max_edges,
            device: Device::Cpu,
        }
    }

    /// Build a ModelFlow for the edge score head (`head_norm` → `head` + bias).
    pub fn build_head_flow(&self, weights: &mut WeightMap) -> Result<BuiltModel> {
        use rlx_flow::ModelFlow;
        use rlx_ir::HirGraphExt;
        use rlx_ir::hir::HirMut;

        let b = self.batch;
        let e = self.max_edges;
        let c = self.cfg.hidden_dim;
        let f = DType::F32;
        let eps = self.cfg.head_ln_eps;
        let flow = ModelFlow::new("hoct_edge_head")
            .with_profile(CompileProfile::encoder())
            .input("edge_h", Shape::new(&[b, e, c], f))
            .layer_norm("head_norm.weight", "head_norm.bias", eps)
            .plugin_named("hoct.head_linear", move |emit, hidden| {
                let h = hidden.ok_or_else(|| anyhow::anyhow!("head linear needs hidden"))?;
                let w = emit.load_param("head.weight", true)?;
                let bias = emit.load_param("head.bias", false)?;
                let mut gb = HirMut::new(emit.hir());
                // Flatten [B,E,C] → [B*E, C] for mm, then reshape back.
                let flat = gb.reshape_(h.hir_id(), vec![(b * e) as i64, c as i64]);
                let mm = gb.mm(flat, w);
                let with_b = gb.add(mm, bias);
                let out = gb.reshape_(with_b, vec![b as i64, e as i64, 1]);
                Ok(Some(emit.wrap(out, Shape::new(&[b, e, 1], DType::F32))))
            })
            .output("edge_logits");
        flow.build_with(&mut WeightMapSource(weights), None)
            .context("build HOCT edge-head ModelFlow")
    }

    /// Construct a WeightMap containing only the score-head tensors.
    pub fn head_weight_map(weights: &HoctWeights) -> WeightMap {
        let mut tensors = HashMap::new();
        tensors.insert(
            "head_norm.weight".into(),
            (
                weights.head_norm_weight.clone(),
                vec![weights.head_norm_weight.len()],
            ),
        );
        tensors.insert(
            "head_norm.bias".into(),
            (
                weights.head_norm_bias.clone(),
                vec![weights.head_norm_bias.len()],
            ),
        );
        tensors.insert(
            "head.weight".into(),
            (
                weights.head_weight.clone(),
                vec![1, weights.head_weight.len()],
            ),
        );
        tensors.insert(
            "head.bias".into(),
            (weights.head_bias.clone(), vec![weights.head_bias.len()]),
        );
        WeightMap::from_tensors(tensors)
    }
}

/// Padded eager runner with encoder-shaped `(N_max, E_max)` contract.
///
/// Use [`HoctCompiled::forward_padded_fixed`] to pad a single batch up to the
/// compile limits before calling the eager forward.
#[derive(Debug, Clone)]
pub struct HoctCompiled {
    pub model: HoctModel,
    pub max_nodes: usize,
    pub max_edges: usize,
    pub device: Device,
}

impl HoctCompiled {
    pub fn from_weights(path: impl AsRef<Path>) -> Result<Self> {
        let weights = load_hoct_weights(path)?;
        Ok(HoctFlow::new(HoctConfig::default()).build_compiled(weights))
    }

    pub fn with_device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }

    pub fn with_pad(mut self, max_nodes: usize, max_edges: usize) -> Self {
        self.max_nodes = max_nodes;
        self.max_edges = max_edges;
        self
    }

    pub fn forward_windows(&self, windows: &[WindowBatch]) -> Option<HoctOutput> {
        let (padded, _, _) = TemporalDataset::pad_batches(windows)?;
        Some(self.forward_batch(&padded))
    }

    pub fn forward_batch(&self, batch: &WindowBatch) -> HoctOutput {
        let _ = self.device;
        self.model.forward(
            &batch.node_features.view(),
            &batch.node_pos.view(),
            &batch.edge_pos.view(),
            &batch.edge_indices,
            &batch.node_mask,
            &batch.edge_mask,
        )
    }

    /// Pad a single batch up to `(max_nodes, max_edges)` then forward.
    pub fn forward_padded_fixed(&self, batch: &WindowBatch) -> HoctOutput {
        let padded = pad_to_fixed(batch, self.max_nodes, self.max_edges);
        self.forward_batch(&padded)
    }
}

fn pad_to_fixed(batch: &WindowBatch, n_max: usize, e_max: usize) -> WindowBatch {
    let b = batch.node_features.len_of(ndarray::Axis(0));
    let n = batch.node_features.len_of(ndarray::Axis(1));
    let e = batch.edge_indices.len_of(ndarray::Axis(1));
    let d = batch.node_features.len_of(ndarray::Axis(2));
    assert!(n <= n_max && e <= e_max, "batch exceeds pad limits");

    let mut node_features = ndarray::Array3::<f32>::zeros((b, n_max, d));
    let mut node_pos = ndarray::Array3::<f32>::zeros((b, n_max, 3));
    let mut edge_pos = ndarray::Array3::<f32>::zeros((b, e_max, 3));
    let mut edge_indices = ndarray::Array3::<i64>::zeros((b, e_max, 2));
    let mut node_mask = ndarray::Array2::<bool>::from_elem((b, n_max), false);
    let mut edge_mask = ndarray::Array2::<bool>::from_elem((b, e_max), false);

    for bi in 0..b {
        for ni in 0..n {
            for k in 0..d {
                node_features[[bi, ni, k]] = batch.node_features[[bi, ni, k]];
            }
            for k in 0..3 {
                node_pos[[bi, ni, k]] = batch.node_pos[[bi, ni, k]];
            }
            node_mask[[bi, ni]] = batch.node_mask[[bi, ni]];
        }
        for ei in 0..e {
            for k in 0..3 {
                edge_pos[[bi, ei, k]] = batch.edge_pos[[bi, ei, k]];
            }
            edge_indices[[bi, ei, 0]] = batch.edge_indices[[bi, ei, 0]];
            edge_indices[[bi, ei, 1]] = batch.edge_indices[[bi, ei, 1]];
            edge_mask[[bi, ei]] = batch.edge_mask[[bi, ei]];
        }
    }

    WindowBatch {
        node_features,
        node_pos,
        edge_pos,
        edge_indices,
        node_mask,
        edge_mask,
        frame_t: batch.frame_t,
    }
}
