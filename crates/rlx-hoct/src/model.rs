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

//! Eager HOCT forward pass matching TorchScript `general_v0.pt`.
//!
//! This is the numerical reference for parity tests. The compiled score head
//! ([`crate::device::HoctDeviceRunner`]) reuses the same body then runs
//! LayerNorm→Linear on a backend device.

use crate::attn::{
    dist_attn_bias, edge_cross_block, edge_self_block, layer_norm, linear3d, node_block,
};
use crate::config::HoctConfig;
use crate::geometry::{line_to_line_distances, make_attn_mask};
use crate::weights::HoctWeights;
use anyhow::Result;
use ndarray::{Array2, Array3, ArrayView3, Axis};

/// Outputs of [`HoctModel::forward`].
///
/// Shapes: `edge_logits` `[B,E,1]`, `node_hidden` / `edge_hidden` `[B,*,288]`,
/// `orphan_logits` `[B,N,1]` (always zeros for `general_v0`).
#[derive(Debug, Clone)]
pub struct HoctOutput {
    /// Edge classification logits (pre–parental-softmax).
    pub edge_logits: Array3<f32>,
    /// Contextualized node embeddings after node blocks.
    pub node_hidden: Array3<f32>,
    /// Edge embeddings after edge blocks (input to the score head).
    pub edge_hidden: Array3<f32>,
    /// Orphan logits; shipped checkpoint is all zeros.
    pub orphan_logits: Array3<f32>,
}

/// Eager HOCT edge model (`general_v0` dimensions).
#[derive(Debug, Clone)]
pub struct HoctModel {
    pub cfg: HoctConfig,
    pub weights: HoctWeights,
}

impl HoctModel {
    /// Wrap config + already-loaded weights.
    pub fn new(cfg: HoctConfig, weights: HoctWeights) -> Self {
        Self { cfg, weights }
    }

    /// Load safetensors weights and use [`HoctConfig::default`].
    pub fn from_weights(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let weights = crate::weights::load_hoct_weights(path)?;
        Ok(Self::new(HoctConfig::default(), weights))
    }

    /// Cross block is `edge_blocks.0` in the checkpoint; self blocks 1–3 follow.
    const EDGE_ORDER: [usize; 4] = [0, 1, 2, 3];

    /// Run the full eager forward.
    ///
    /// # Arguments
    ///
    /// - `node_features` — `[B,N,19]` standardized regionprops
    /// - `node_pos` / `edge_pos` — `[B,*,3]` spatial positions `(z,y,x)` for RoPE
    /// - `edge_indices` — `[B,E,2]` `(src, dst)` node indices
    /// - `node_mask` / `edge_mask` — live tokens (`true` = valid)
    pub fn forward(
        &self,
        node_features: &ArrayView3<f32>,
        node_pos: &ArrayView3<f32>,
        edge_pos: &ArrayView3<f32>,
        edge_indices: &Array3<i64>,
        node_mask: &Array2<bool>,
        edge_mask: &Array2<bool>,
    ) -> HoctOutput {
        let cfg = &self.cfg;
        let w = &self.weights;
        let b = node_features.len_of(Axis(0));
        let n = node_features.len_of(Axis(1));
        let _e = edge_indices.len_of(Axis(1));

        let node_pos_owned = node_pos.to_owned();
        let edge_pos_owned = edge_pos.to_owned();

        let node_attn_mask = make_attn_mask(&node_pos_owned, node_mask, cfg.tau_sq());
        let edge_attn_mask = make_attn_mask(&edge_pos_owned, edge_mask, cfg.tau_sq());

        let x0 = linear3d(
            node_features,
            &w.input_proj_weight,
            cfg.hidden_dim,
            cfg.feature_dim,
            Some(&w.input_proj_bias),
        );

        let mut x = x0.clone();
        for block in &w.node_blocks {
            x = node_block(cfg, &x.view(), node_pos, &node_attn_mask.view(), block);
        }

        let h_e = self.edge_gatherer(edge_indices, &x.view());

        let dist = line_to_line_distances(&node_pos_owned, edge_indices);

        let raw = node_features.to_owned();
        let f_e = self.edge_feature_proj(edge_indices, &raw.view());

        let mut edge_h = h_e;
        for (step, &bi) in Self::EDGE_ORDER.iter().enumerate() {
            let block = &w.edge_blocks[bi];
            let bias = dist_attn_bias(
                &dist.view(),
                &block.dist_scaling,
                &block.dist_scaling_head_direction,
                &edge_attn_mask.view(),
                cfg.num_heads,
            );
            edge_h = if step == 0 {
                edge_cross_block(
                    cfg,
                    &edge_h.view(),
                    &f_e.view(),
                    edge_pos,
                    &bias.view(),
                    block,
                )
            } else {
                edge_self_block(cfg, &edge_h.view(), edge_pos, &bias.view(), block)
            };
        }

        let head_in = layer_norm(
            &edge_h.view(),
            &w.head_norm_weight,
            &w.head_norm_bias,
            cfg.head_ln_eps,
        );
        let edge_logits = linear3d(
            &head_in.view(),
            &w.head_weight,
            1,
            cfg.hidden_dim,
            Some(&w.head_bias),
        );

        let orphan_logits = Array3::<f32>::zeros((b, n, 1));

        HoctOutput {
            edge_logits,
            node_hidden: x,
            edge_hidden: edge_h,
            orphan_logits,
        }
    }

    pub fn edge_gatherer_out(
        &self,
        edge_indices: &Array3<i64>,
        x: &ArrayView3<f32>,
    ) -> Array3<f32> {
        self.edge_gatherer(edge_indices, x)
    }

    fn edge_gatherer(&self, edge_indices: &Array3<i64>, x: &ArrayView3<f32>) -> Array3<f32> {
        let b = x.len_of(Axis(0));
        let e = edge_indices.len_of(Axis(1));
        let c = x.len_of(Axis(2));
        let mut pairs = Array3::<f32>::zeros((b, e, 2 * c));
        for bi in 0..b {
            for ei in 0..e {
                let i = edge_indices[[bi, ei, 0]] as usize;
                let j = edge_indices[[bi, ei, 1]] as usize;
                for d in 0..c {
                    pairs[[bi, ei, d]] = x[[bi, i, d]];
                    pairs[[bi, ei, c + d]] = x[[bi, j, d]];
                }
            }
        }
        let w = &self.weights.edge_gatherer;
        let h1 = linear3d(
            &pairs.view(),
            &w.fc1_weight,
            self.cfg.hidden_dim,
            2 * self.cfg.hidden_dim,
            Some(&w.fc1_bias),
        );
        let mut h1a = h1;
        h1a.mapv_inplace(crate::attn::gelu_tanh);
        let h2 = linear3d(
            &h1a.view(),
            &w.fc2_weight,
            self.cfg.hidden_dim,
            self.cfg.hidden_dim,
            Some(&w.fc2_bias),
        );
        h2.mapv(crate::attn::gelu_tanh)
    }

    fn edge_feature_proj(&self, edge_indices: &Array3<i64>, raw: &ArrayView3<f32>) -> Array3<f32> {
        let cfg = &self.cfg;
        let b = raw.len_of(Axis(0));
        let e = edge_indices.len_of(Axis(1));
        let d = cfg.feature_dim;
        let mut diff = Array3::<f32>::zeros((b, e, d));
        for bi in 0..b {
            for ei in 0..e {
                let i = edge_indices[[bi, ei, 0]] as usize;
                let j = edge_indices[[bi, ei, 1]] as usize;
                for k in 0..d {
                    diff[[bi, ei, k]] = raw[[bi, j, k]] - raw[[bi, i, k]];
                }
            }
        }
        linear3d(
            &diff.view(),
            &self.weights.edge_input_proj_weight,
            cfg.hidden_dim,
            d,
            Some(&self.weights.edge_input_proj_bias),
        )
    }
}
