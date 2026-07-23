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

//! Bridge a [`rlx_nemo::NemoModel`] checkpoint to the [`WeightSource`]
//! trait, plus canonical NeMo Conformer-CTC state-dict key names.

use anyhow::{Result, anyhow};
use rlx_flow::WeightSource;
use rlx_nemo::NemoModel;

/// Adapter exposing a `.nemo` checkpoint as a [`WeightSource`].
pub struct NemoWeights<'a> {
    model: &'a NemoModel,
}

impl<'a> NemoWeights<'a> {
    /// Borrow an opened [`NemoModel`] as a weight source.
    pub fn new(model: &'a NemoModel) -> Self {
        Self { model }
    }
}

impl WeightSource for NemoWeights<'_> {
    fn take(&mut self, key: &str, transpose: bool) -> Result<(Vec<f32>, Vec<usize>)> {
        let t = self
            .model
            .tensor(key)
            .map_err(|e| anyhow!("nemo weight {key:?}: {e}"))?;
        if transpose && t.shape.len() == 2 {
            let (r, c) = (t.shape[0], t.shape[1]);
            let mut out = vec![0.0f32; r * c];
            for i in 0..r {
                for j in 0..c {
                    out[j * r + i] = t.data[i * c + j];
                }
            }
            Ok((out, vec![c, r]))
        } else {
            Ok((t.data, t.shape))
        }
    }

    fn has(&self, key: &str) -> bool {
        self.model.shape_of(key).is_some()
    }
}

/// Canonical NeMo ConformerEncoder + ConvASRDecoder parameter names.
///
/// Use with [`NemoWeights`] / `dump-keys` when reconciling a new checkpoint.
pub mod keys {
    /// `encoder.layers.{i}.{suffix}`.
    pub fn enc_layer(i: usize, suffix: &str) -> String {
        format!("encoder.layers.{i}.{suffix}")
    }

    /// `encoder.pre_encode.conv.{idx}.{wb}` (Conv2d stack).
    pub fn pre_encode_conv(idx: usize, wb: &str) -> String {
        format!("encoder.pre_encode.conv.{idx}.{wb}")
    }
    pub const PRE_ENCODE_OUT_W: &str = "encoder.pre_encode.out.weight";
    pub const PRE_ENCODE_OUT_B: &str = "encoder.pre_encode.out.bias";

    pub const NORM_FF1_W: &str = "norm_feed_forward1.weight";
    pub const NORM_FF1_B: &str = "norm_feed_forward1.bias";
    pub const FF1_L1_W: &str = "feed_forward1.linear1.weight";
    pub const FF1_L1_B: &str = "feed_forward1.linear1.bias";
    pub const FF1_L2_W: &str = "feed_forward1.linear2.weight";
    pub const FF1_L2_B: &str = "feed_forward1.linear2.bias";

    pub const NORM_ATT_W: &str = "norm_self_att.weight";
    pub const NORM_ATT_B: &str = "norm_self_att.bias";
    pub const ATT_Q_W: &str = "self_attn.linear_q.weight";
    pub const ATT_Q_B: &str = "self_attn.linear_q.bias";
    pub const ATT_K_W: &str = "self_attn.linear_k.weight";
    pub const ATT_K_B: &str = "self_attn.linear_k.bias";
    pub const ATT_V_W: &str = "self_attn.linear_v.weight";
    pub const ATT_V_B: &str = "self_attn.linear_v.bias";
    pub const ATT_O_W: &str = "self_attn.linear_out.weight";
    pub const ATT_O_B: &str = "self_attn.linear_out.bias";
    pub const ATT_POS_W: &str = "self_attn.linear_pos.weight";
    pub const ATT_POS_U: &str = "self_attn.pos_bias_u";
    pub const ATT_POS_V: &str = "self_attn.pos_bias_v";

    pub const NORM_CONV_W: &str = "norm_conv.weight";
    pub const NORM_CONV_B: &str = "norm_conv.bias";
    pub const CONV_PW1_W: &str = "conv.pointwise_conv1.weight";
    pub const CONV_PW1_B: &str = "conv.pointwise_conv1.bias";
    pub const CONV_DW_W: &str = "conv.depthwise_conv.weight";
    pub const CONV_DW_B: &str = "conv.depthwise_conv.bias";
    pub const CONV_BN_W: &str = "conv.batch_norm.weight";
    pub const CONV_BN_B: &str = "conv.batch_norm.bias";
    pub const CONV_BN_MEAN: &str = "conv.batch_norm.running_mean";
    pub const CONV_BN_VAR: &str = "conv.batch_norm.running_var";
    pub const CONV_PW2_W: &str = "conv.pointwise_conv2.weight";
    pub const CONV_PW2_B: &str = "conv.pointwise_conv2.bias";

    pub const NORM_FF2_W: &str = "norm_feed_forward2.weight";
    pub const NORM_FF2_B: &str = "norm_feed_forward2.bias";
    pub const FF2_L1_W: &str = "feed_forward2.linear1.weight";
    pub const FF2_L1_B: &str = "feed_forward2.linear1.bias";
    pub const FF2_L2_W: &str = "feed_forward2.linear2.weight";
    pub const FF2_L2_B: &str = "feed_forward2.linear2.bias";

    pub const NORM_OUT_W: &str = "norm_out.weight";
    pub const NORM_OUT_B: &str = "norm_out.bias";

    /// ConvASRDecoder CTC head (Conv1d kernel 1 ≡ linear).
    pub const CTC_W: &str = "decoder.decoder_layers.0.weight";
    pub const CTC_B: &str = "decoder.decoder_layers.0.bias";
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_flow::WeightSource;

    struct Mem {
        data: Vec<f32>,
        shape: Vec<usize>,
    }
    impl WeightSource for Mem {
        fn take(&mut self, _k: &str, transpose: bool) -> Result<(Vec<f32>, Vec<usize>)> {
            if transpose && self.shape.len() == 2 {
                let (r, c) = (self.shape[0], self.shape[1]);
                let mut out = vec![0.0; r * c];
                for i in 0..r {
                    for j in 0..c {
                        out[j * r + i] = self.data[i * c + j];
                    }
                }
                return Ok((out, vec![c, r]));
            }
            Ok((self.data.clone(), self.shape.clone()))
        }
    }

    #[test]
    fn transpose_2d() {
        let mut m = Mem {
            data: vec![1., 2., 3., 4., 5., 6.],
            shape: vec![2, 3],
        };
        let (d, s) = m.take("w", true).unwrap();
        assert_eq!(s, vec![3, 2]);
        assert_eq!(d, vec![1., 4., 2., 5., 3., 6.]);
    }

    #[test]
    fn key_formatting() {
        assert_eq!(
            keys::enc_layer(3, keys::ATT_Q_W),
            "encoder.layers.3.self_attn.linear_q.weight"
        );
    }
}
