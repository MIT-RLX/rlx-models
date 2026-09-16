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

//! Embedded TEN-VAD weights (`weights/ten_vad.safetensors`, 305 KB).
//!
//! Produced from the upstream `src/onnx_model/ten-vad.onnx` plus `src/coeff.h`
//! by `scripts/export_ten_vad_onnx_weights.py`, which also does the two
//! layout conversions the runtime would otherwise repeat every load: the ONNX
//! `i,o,f,c` LSTM gate order becomes rlx's `i,f,g,o`, and the separable kernels
//! move their length axis from W to H.

use anyhow::{Context, Result};
use rlx_core::embedded_safetensors::EmbeddedSafetensors;
use std::path::Path;
use std::sync::OnceLock;

use crate::{CONTEXT_FRAMES, FEATURE_LEN, HIDDEN, MEL_BANDS};

const SAFETENSORS: &[u8] = include_bytes!("../weights/ten_vad.safetensors");

/// Conv channels after the first pointwise projection.
pub const CONV_CH: usize = 16;
/// Flattened conv-stack output feeding the first LSTM.
pub const LSTM1_INPUT: usize = 80;
/// Width of the penultimate dense layer.
pub const DENSE_HIDDEN: usize = 32;

/// Every tensor of the network, row-major, in rlx layout.
#[derive(Clone)]
pub struct TenVadWeights {
    /// `[1, 1, 3, 3]` — genuine 2-D conv over (context, feature).
    pub conv0_depthwise: Vec<f32>,
    /// `[16, 1, 1, 1]` pointwise projection, plus its `[16]` bias.
    pub conv0_pointwise: Vec<f32>,
    pub conv0_bias: Vec<f32>,
    /// `[16, 1, 3, 1]` depthwise, `[16, 16, 1, 1]` pointwise, `[16]` bias.
    pub sep1_depthwise: Vec<f32>,
    pub sep1_pointwise: Vec<f32>,
    pub sep1_bias: Vec<f32>,
    pub sep2_depthwise: Vec<f32>,
    pub sep2_pointwise: Vec<f32>,
    pub sep2_bias: Vec<f32>,
    /// `[4*64, 80]`, `[4*64, 64]`, `[4*64]` — gate order `i, f, g, o`.
    pub lstm1_weight_ih: Vec<f32>,
    pub lstm1_weight_hh: Vec<f32>,
    pub lstm1_bias: Vec<f32>,
    pub lstm2_weight_ih: Vec<f32>,
    pub lstm2_weight_hh: Vec<f32>,
    pub lstm2_bias: Vec<f32>,
    /// `[128, 32]` then `[32, 1]`, applied as `x @ W + b`.
    pub dense1_weight: Vec<f32>,
    pub dense1_bias: Vec<f32>,
    pub dense2_weight: Vec<f32>,
    pub dense2_bias: Vec<f32>,
    /// Per-feature standardization, `[41]` each.
    pub feature_mean: Vec<f32>,
    pub feature_std: Vec<f32>,
    /// Hann-768 STFT analysis window.
    pub window: Vec<f32>,
}

static PARSED: OnceLock<TenVadWeights> = OnceLock::new();

impl TenVadWeights {
    /// Borrow the frontend's tables in the shape the `no_std` core wants.
    ///
    /// Lets a runtime `--weights` override drive the shared DSP just as the
    /// embedded blob does on an MCU.
    pub fn core(&self) -> rlx_ten_vad_core::CoreWeights<'_> {
        rlx_ten_vad_core::CoreWeights {
            window: &self.window,
            feature_mean: &self.feature_mean,
            feature_std: &self.feature_std,
        }
    }

    /// The weights compiled into this binary — no external files.
    pub fn embedded() -> &'static Self {
        PARSED.get_or_init(|| parse(SAFETENSORS).expect("embedded ten-vad safetensors"))
    }

    /// Load the same layout from disk (for re-exported or patched weights).
    pub fn load(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
        parse(&bytes)
    }
}

fn parse(bytes: &[u8]) -> Result<TenVadWeights> {
    let st = EmbeddedSafetensors::parse(bytes)?;
    let w = TenVadWeights {
        conv0_depthwise: st.tensor_f32("conv0.depthwise.weight")?,
        conv0_pointwise: st.tensor_f32("conv0.pointwise.weight")?,
        conv0_bias: st.tensor_f32("conv0.bias")?,
        sep1_depthwise: st.tensor_f32("sep1.depthwise.weight")?,
        sep1_pointwise: st.tensor_f32("sep1.pointwise.weight")?,
        sep1_bias: st.tensor_f32("sep1.bias")?,
        sep2_depthwise: st.tensor_f32("sep2.depthwise.weight")?,
        sep2_pointwise: st.tensor_f32("sep2.pointwise.weight")?,
        sep2_bias: st.tensor_f32("sep2.bias")?,
        lstm1_weight_ih: st.tensor_f32("lstm1.weight_ih")?,
        lstm1_weight_hh: st.tensor_f32("lstm1.weight_hh")?,
        lstm1_bias: st.tensor_f32("lstm1.bias")?,
        lstm2_weight_ih: st.tensor_f32("lstm2.weight_ih")?,
        lstm2_weight_hh: st.tensor_f32("lstm2.weight_hh")?,
        lstm2_bias: st.tensor_f32("lstm2.bias")?,
        dense1_weight: st.tensor_f32("dense1.weight")?,
        dense1_bias: st.tensor_f32("dense1.bias")?,
        dense2_weight: st.tensor_f32("dense2.weight")?,
        dense2_bias: st.tensor_f32("dense2.bias")?,
        feature_mean: st.tensor_f32("feature.mean")?,
        feature_std: st.tensor_f32("feature.std")?,
        window: st.tensor_f32("stft.window")?,
    };
    w.check()?;
    Ok(w)
}

impl TenVadWeights {
    fn check(&self) -> Result<()> {
        let g = 4 * HIDDEN;
        let expect: [(&str, usize, usize); 22] = [
            ("conv0.depthwise.weight", self.conv0_depthwise.len(), 3 * 3),
            (
                "conv0.pointwise.weight",
                self.conv0_pointwise.len(),
                CONV_CH,
            ),
            ("conv0.bias", self.conv0_bias.len(), CONV_CH),
            (
                "sep1.depthwise.weight",
                self.sep1_depthwise.len(),
                CONV_CH * 3,
            ),
            (
                "sep1.pointwise.weight",
                self.sep1_pointwise.len(),
                CONV_CH * CONV_CH,
            ),
            ("sep1.bias", self.sep1_bias.len(), CONV_CH),
            (
                "sep2.depthwise.weight",
                self.sep2_depthwise.len(),
                CONV_CH * 3,
            ),
            (
                "sep2.pointwise.weight",
                self.sep2_pointwise.len(),
                CONV_CH * CONV_CH,
            ),
            ("sep2.bias", self.sep2_bias.len(), CONV_CH),
            (
                "lstm1.weight_ih",
                self.lstm1_weight_ih.len(),
                g * LSTM1_INPUT,
            ),
            ("lstm1.weight_hh", self.lstm1_weight_hh.len(), g * HIDDEN),
            ("lstm1.bias", self.lstm1_bias.len(), g),
            ("lstm2.weight_ih", self.lstm2_weight_ih.len(), g * HIDDEN),
            ("lstm2.weight_hh", self.lstm2_weight_hh.len(), g * HIDDEN),
            ("lstm2.bias", self.lstm2_bias.len(), g),
            (
                "dense1.weight",
                self.dense1_weight.len(),
                2 * HIDDEN * DENSE_HIDDEN,
            ),
            ("dense1.bias", self.dense1_bias.len(), DENSE_HIDDEN),
            ("dense2.weight", self.dense2_weight.len(), DENSE_HIDDEN),
            ("dense2.bias", self.dense2_bias.len(), 1),
            ("feature.mean", self.feature_mean.len(), FEATURE_LEN),
            ("feature.std", self.feature_std.len(), FEATURE_LEN),
            ("stft.window", self.window.len(), crate::WINDOW_SIZE),
        ];
        for (name, got, want) in expect {
            anyhow::ensure!(got == want, "{name}: {got} values, expected {want}");
        }
        // The conv stack flattens `[5 positions, 16 channels]` into the LSTM.
        anyhow::ensure!(
            LSTM1_INPUT == 5 * CONV_CH && FEATURE_LEN == MEL_BANDS + 1 && CONTEXT_FRAMES == 3,
            "architecture constants disagree with the exported weights"
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_weights_parse() {
        let w = TenVadWeights::embedded();
        assert_eq!(w.lstm1_weight_ih.len(), 256 * 80);
        assert_eq!(w.window.len(), 768);
        // Hann-768 is periodic: zero at 0, exactly 1 at the midpoint.
        assert_eq!(w.window[0], 0.0);
        assert!((w.window[384] - 1.0).abs() < 1e-6);
    }
}
