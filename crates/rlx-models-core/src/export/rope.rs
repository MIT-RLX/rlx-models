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

//! RoPE tables as a package holds them, from a checkpoint's config: plain
//! `1/θ^(2i/d)`, or scaled as its `rope_scaling` says (Llama 3's piecewise
//! rescale, linear, YaRN, LongRoPE). A table is data the exporter computes,
//! so a scaled model needs no graph of its own.

use serde_json::Value;

use crate::standard_decoder::{RopeScaling, parse_rope_scaling};

/// A model's rotary frequencies.
#[derive(Debug, Clone)]
pub struct Rope {
    inv_freq: Vec<f64>,
    /// Multiplies cos and sin (YaRN's and LongRoPE's attention factor).
    mscale: f64,
    scaled: bool,
}

impl Rope {
    /// `1/θ^(2i/d)` for each of the `head_dim / 2` pairs.
    pub fn plain(head_dim: usize, theta: f64) -> Self {
        Self {
            inv_freq: (0..head_dim / 2)
                .map(|i| 1.0 / theta.powf((2 * i) as f64 / head_dim as f64))
                .collect(),
            mscale: 1.0,
            scaled: false,
        }
    }

    /// The frequencies `config` (a `config.json`, or its `text_config`)
    /// gives: scaled when it has a `rope_scaling` (or transformers 5's
    /// `rope_parameters`) of a known type.
    pub fn from_config(config: &Value, head_dim: usize, theta: f64) -> Self {
        let mut config = config.get("text_config").unwrap_or(config).clone();
        crate::config::legacy_rope_keys(&mut config);
        let scaling = parse_rope_scaling(config.get("rope_scaling"), &config);
        if matches!(scaling, RopeScaling::None) {
            return Self::plain(head_dim, theta);
        }
        Self {
            inv_freq: (0..head_dim / 2)
                .map(|i| scaling.inv_freq(i, head_dim, theta))
                .collect(),
            mscale: scaling.mscale(),
            scaled: true,
        }
    }

    /// Whether the frequencies are scaled.
    pub fn is_scaled(&self) -> bool {
        self.scaled
    }

    /// The inverse frequencies.
    pub fn inv_freq(&self) -> &[f64] {
        &self.inv_freq
    }

    /// The table's width: one column per rotated pair.
    pub fn half(&self) -> usize {
        self.inv_freq.len()
    }

    /// Row `pos`: cos and sin of `pos · inv_freq`, times the attention
    /// factor.
    pub fn row(&self, pos: usize) -> (Vec<f32>, Vec<f32>) {
        self.inv_freq
            .iter()
            .map(|f| {
                let (s, c) = (pos as f64 * f).sin_cos();
                ((c * self.mscale) as f32, (s * self.mscale) as f32)
            })
            .unzip()
    }

    /// Rows `0..rows`, each [`Self::half`] wide.
    pub fn table(&self, rows: usize) -> (Vec<f32>, Vec<f32>) {
        let (mut cos, mut sin) = (Vec::new(), Vec::new());
        for pos in 0..rows {
            let (c, s) = self.row(pos);
            cos.extend(c);
            sin.extend(s);
        }
        (cos, sin)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn llama3_scaling_divides_only_the_long_wavelengths() {
        // Llama 3.2's: factor 32, original context 8192.
        let cfg = json!({"rope_scaling": {"rope_type": "llama3", "factor": 32.0,
            "low_freq_factor": 1.0, "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192}});
        let (plain, scaled) = (
            Rope::plain(64, 500000.0),
            Rope::from_config(&cfg, 64, 500000.0),
        );
        assert!(scaled.is_scaled() && !plain.is_scaled());
        // Short wavelengths (high frequencies) are kept, the longest divided
        // by the factor.
        assert_eq!(scaled.inv_freq()[0], plain.inv_freq()[0]);
        let last = plain.inv_freq().len() - 1;
        let ratio = plain.inv_freq()[last] / scaled.inv_freq()[last];
        assert!((ratio - 32.0).abs() < 1e-9, "{ratio}");
        // No scaling: the plain table, row for row.
        let none = Rope::from_config(&json!({"rope_scaling": null}), 64, 1e6);
        assert_eq!(none.row(7), Rope::plain(64, 1e6).row(7));
    }

    #[test]
    fn yarn_scales_the_table_by_its_attention_factor() {
        let cfg = json!({"rope_scaling": {"rope_type": "yarn", "factor": 4.0,
            "original_max_position_embeddings": 4096}});
        let r = Rope::from_config(&cfg, 128, 1e6);
        let m = 0.1 * 4f64.ln() + 1.0;
        let (c, _) = r.row(0);
        assert!((f64::from(c[0]) - m).abs() < 1e-6, "{} vs {m}", c[0]);
    }
}
