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

//! Model dimensions and tracking hyper-parameters for HOCT `general_v0`.

/// Per-feature mean for 19-d node descriptors (from upstream `hoct._api._MEAN`).
pub const FEATURE_MEAN: [f32; 19] = [
    463.26, 2.938, 356.49, 344.91, 11.521, 0.276, 0.966, 0.574, 0.162, 167.81, -0.027, 0.05,
    -0.027, 87.012, -1.401, 0.05, -1.401, 83.695, 0.009,
];

/// Per-feature std for standardization (from upstream `hoct._api._STD`).
pub const FEATURE_STD: [f32; 19] = [
    555.78, 7.6, 195.88, 226.1, 8.199, 0.216, 0.281, 0.193, 0.069, 678.45, 3.167, 2.875, 3.167,
    512.92, 182.74, 2.875, 182.74, 306.08, 0.078,
];

/// Model dimensions matching TorchScript `general_v0.pt`.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct HoctConfig {
    /// Node feature width (19 regionprops).
    pub feature_dim: usize,
    /// Transformer channel width (`C=288`).
    pub hidden_dim: usize,
    /// Attention heads (`H=4`).
    pub num_heads: usize,
    /// Per-head dim (`Hd=72`).
    pub head_dim: usize,
    /// Node self-attention blocks (`L_n=4`).
    pub num_node_blocks: usize,
    /// Edge blocks (`L_e=4`; first is cross-attn).
    pub num_edge_blocks: usize,
    /// Spatial distance scale for attention mask (mask uses `τ²`).
    pub tau: f32,
    /// RMSNorm epsilon in blocks.
    pub rms_eps: f32,
    /// LayerNorm epsilon on the score head.
    pub head_ln_eps: f32,
}

impl Default for HoctConfig {
    fn default() -> Self {
        Self {
            feature_dim: 19,
            hidden_dim: 288,
            num_heads: 4,
            head_dim: 72,
            num_node_blocks: 4,
            num_edge_blocks: 4,
            tau: 300.0,
            rms_eps: 1e-6,
            head_ln_eps: 1e-5,
        }
    }
}

impl HoctConfig {
    /// `τ²` used in the spatial attention mask.
    pub fn tau_sq(&self) -> f32 {
        self.tau * self.tau
    }

    /// `H * Hd` (Q/K/V channel width before gating split).
    pub fn qkv_dim(&self) -> usize {
        self.num_heads * self.head_dim
    }

    /// Gate channel width (same as [`Self::qkv_dim`]).
    pub fn gate_dim(&self) -> usize {
        self.qkv_dim()
    }

    /// `q_proj` output width (`2 * H * Hd` for Q‖gate).
    pub fn q_proj_out(&self) -> usize {
        2 * self.qkv_dim()
    }

    /// MLP intermediate width (`2C`).
    pub fn mlp_hidden(&self) -> usize {
        2 * self.hidden_dim
    }
}

/// Candidate-graph construction defaults (kNN in space-time).
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct GraphConfig {
    /// Max spatial distance for an edge (after optional scale).
    pub distance_threshold: f32,
    /// Max neighbors retained per source node.
    pub n_neighbors: usize,
    /// Max temporal gap `|Δt|` (frames).
    pub max_delta_t: i32,
    /// Multiplier on Euclidean distance before the threshold check.
    pub spatial_scale: f32,
}

impl Default for GraphConfig {
    fn default() -> Self {
        Self {
            distance_threshold: 300.0,
            n_neighbors: 5,
            max_delta_t: 3,
            spatial_scale: 1.0,
        }
    }
}

/// ILP objective weights (paper Appendix B / upstream `ILPSolverConfig.default`).
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct IlpWeights {
    /// Appearance cost, scaled by `(1 - orphan_prob)` off the first frame.
    pub appearance: f32,
    /// Disappearance cost off the last frame.
    pub disappearance: f32,
    /// Division (`δ`) cost.
    pub division: f32,
    /// Node selection cost (`y`); negative encourages keeping detections.
    pub node: f32,
    /// Added to every edge cost before the `Δt` dampening.
    pub edge_bias: f32,
    /// Temporal dampening: `exp(-λ (|Δt|-1))` on edge weights and orphan aggregation.
    pub delta_t_weight: f32,
}

impl Default for IlpWeights {
    fn default() -> Self {
        Self {
            appearance: 0.5,
            disappearance: 0.25,
            division: 0.25,
            node: -10.0,
            edge_bias: 0.5,
            delta_t_weight: 0.5,
        }
    }
}
