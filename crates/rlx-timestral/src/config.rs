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

//! `config.json` of a Timestral checkpoint (upstream `timestral.config.TimestralConfig`).

use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};

/// One transformer layer: the attention axes it runs, in order, before its MLP.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LayerKind {
    Time,
    Variate,
    TimeVariate,
    VariateTime,
}

impl LayerKind {
    /// Attention axes in execution order; `true` = time axis.
    pub fn axes(self) -> &'static [bool] {
        match self {
            LayerKind::Time => &[true],
            LayerKind::Variate => &[false],
            LayerKind::TimeVariate => &[true, false],
            LayerKind::VariateTime => &[false, true],
        }
    }
}

/// Architecture and decode settings of a Timestral checkpoint.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TimestralConfig {
    pub patch_size: usize,
    pub d_model: usize,
    pub num_heads: usize,
    pub mlp_hidden_dim: usize,
    pub block_schedule: Vec<LayerKind>,
    pub quantile_knots: Vec<f32>,
    /// Trained decode block size, in patches.
    pub block_size: usize,
    /// Trained context length, in time steps.
    pub context_length: usize,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default = "default_rope_partial")]
    pub rope_partial_factor: f32,
    #[serde(default = "default_norm_eps")]
    pub norm_eps: f32,
    #[serde(default)]
    pub attn_bias: bool,
    #[serde(default)]
    pub mlp_bias: bool,
    #[serde(default = "default_gate")]
    pub time_attn_gate: String,
    #[serde(default = "default_gate")]
    pub variate_attn_gate: String,
    #[serde(default = "default_minimum_scale")]
    pub minimum_scale: f32,
    #[serde(default = "default_true")]
    pub use_asinh: bool,
    #[serde(default)]
    pub stabilize_with_global: bool,
    #[serde(default = "default_true")]
    pub covariate_enabled: bool,
    #[serde(default = "default_true")]
    pub split_context_horizon_attn: bool,
    #[serde(default = "default_true")]
    pub attend_missing_patches: bool,
    #[serde(default = "default_width_factor")]
    pub quantile_width_factor: f32,
}

fn default_rope_theta() -> f32 {
    10000.0
}
fn default_rope_partial() -> f32 {
    0.5
}
fn default_norm_eps() -> f32 {
    1e-6
}
fn default_gate() -> String {
    "sdpa_sigmoid".into()
}
fn default_minimum_scale() -> f32 {
    1e-6
}
fn default_true() -> bool {
    true
}
fn default_width_factor() -> f32 {
    2.0
}

impl TimestralConfig {
    pub fn from_file(path: &Path) -> Result<Self> {
        let text =
            std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
        let cfg: Self =
            serde_json::from_str(&text).with_context(|| format!("parse {}", path.display()))?;
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.d_model.is_multiple_of(self.num_heads),
            "d_model {} not divisible by num_heads {}",
            self.d_model,
            self.num_heads
        );
        ensure!(
            self.rope_dim() > 0 && self.rope_dim().is_multiple_of(2),
            "rope_partial_factor {} gives an invalid rope_dim {}",
            self.rope_partial_factor,
            self.rope_dim()
        );
        for gate in [&self.time_attn_gate, &self.variate_attn_gate] {
            if gate != "none" && gate != "sdpa_sigmoid" {
                bail!("unknown attention gate {gate:?}");
            }
        }
        ensure!(
            self.quantile_knots.contains(&0.5),
            "quantile_knots must contain 0.5"
        );
        ensure!(
            self.quantile_knots.windows(2).all(|w| w[0] < w[1]),
            "quantile_knots must be increasing"
        );
        ensure!(
            !self.stabilize_with_global,
            "the advancing refine frame needs stabilize_with_global=false"
        );
        ensure!(
            !self.attn_bias && !self.mlp_bias,
            "biased attention/MLP checkpoints are not supported"
        );
        Ok(())
    }

    pub fn head_dim(&self) -> usize {
        self.d_model / self.num_heads
    }

    /// Head dimensions rotated by RoPE; the rest pass through.
    pub fn rope_dim(&self) -> usize {
        (self.head_dim() as f32 * self.rope_partial_factor) as usize
    }

    pub fn num_quantiles(&self) -> usize {
        self.quantile_knots.len()
    }

    pub fn median_index(&self) -> usize {
        self.quantile_knots.iter().position(|&q| q == 0.5).unwrap()
    }

    pub fn num_layers(&self) -> usize {
        self.block_schedule.len()
    }

    pub fn gated(&self, is_time: bool) -> bool {
        let gate = if is_time {
            &self.time_attn_gate
        } else {
            &self.variate_attn_gate
        };
        gate == "sdpa_sigmoid"
    }

    /// Whether the layer schedule routes a known covariate's future to the targets.
    ///
    /// Some time op must be followed by at least one more time op and one variate op.
    pub fn supports_known_covariates(&self) -> bool {
        if !self.covariate_enabled {
            return false;
        }
        let ops: Vec<bool> = self
            .block_schedule
            .iter()
            .flat_map(|k| k.axes().iter().copied())
            .collect();
        ops.iter().enumerate().any(|(i, &is_time)| {
            let rest = &ops[i + 1..];
            is_time && rest.iter().any(|&t| t) && !rest.iter().all(|&t| t)
        })
    }

    /// A tiny random-weight configuration for tests that need no download.
    pub fn synth_tiny() -> Self {
        Self {
            patch_size: 8,
            d_model: 32,
            num_heads: 2,
            mlp_hidden_dim: 48,
            block_schedule: vec![LayerKind::Time, LayerKind::TimeVariate, LayerKind::Time],
            quantile_knots: vec![0.1, 0.25, 0.5, 0.75, 0.9],
            block_size: 4,
            context_length: 512,
            rope_theta: 10000.0,
            rope_partial_factor: 0.5,
            norm_eps: 1e-6,
            attn_bias: false,
            mlp_bias: false,
            time_attn_gate: "sdpa_sigmoid".into(),
            variate_attn_gate: "sdpa_sigmoid".into(),
            minimum_scale: 1e-6,
            use_asinh: true,
            stabilize_with_global: false,
            covariate_enabled: true,
            split_context_horizon_attn: true,
            attend_missing_patches: true,
            quantile_width_factor: 3.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn released_schedule_supports_known_covariates() {
        assert!(TimestralConfig::synth_tiny().supports_known_covariates());
        let mut cfg = TimestralConfig::synth_tiny();
        cfg.block_schedule = vec![LayerKind::Time, LayerKind::Variate];
        assert!(!cfg.supports_known_covariates());
    }
}
