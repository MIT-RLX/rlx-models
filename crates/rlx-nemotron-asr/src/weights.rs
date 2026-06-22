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
//! trait used by the graph builders, plus the canonical NeMo
//! FastConformer / RNN-T state-dict key names.

use anyhow::{Result, anyhow};
use rlx_flow::WeightSource;
use rlx_nemo::NemoModel;

/// Adapter exposing a `.nemo` checkpoint as a [`WeightSource`].
///
/// `take(key, transpose)` returns the tensor as contiguous f32; when
/// `transpose` is set and the tensor is 2-D it is transposed in place so
/// linear layers stored `[out, in]` can be consumed as `[in, out]` by the
/// graph's `mm`.
pub struct NemoWeights<'a> {
    model: &'a NemoModel,
}

impl<'a> NemoWeights<'a> {
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

/// Canonical NeMo FastConformer + RNN-T parameter names.
///
/// These follow upstream NeMo's `ConformerEncoder` / `RNNTDecoder` /
/// `RNNTJoint` module hierarchy. Confirm exact spellings for a specific
/// checkpoint with `rlx-nemotron-asr dump-keys --nemo <file>`.
pub mod keys {
    /// `encoder.layers.{i}.{suffix}`.
    pub fn enc_layer(i: usize, suffix: &str) -> String {
        format!("encoder.layers.{i}.{suffix}")
    }

    // ── subsampling (dw_striding pre_encode) ──
    /// `encoder.pre_encode.conv.{idx}.{wb}` (Conv2d stack).
    pub fn pre_encode_conv(idx: usize, wb: &str) -> String {
        format!("encoder.pre_encode.conv.{idx}.{wb}")
    }
    /// Final projection of the subsampling stack to `d_model`.
    pub const PRE_ENCODE_OUT_W: &str = "encoder.pre_encode.out.weight";
    pub const PRE_ENCODE_OUT_B: &str = "encoder.pre_encode.out.bias";

    // ── conformer block sub-modules (relative to enc_layer prefix) ──
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

    // ── RNN-T prediction network (stacked LSTM, per-layer keys) ──
    pub const PRED_EMBED: &str = "decoder.prediction.embed.weight";
    pub fn pred_lstm(part: &str, layer: usize) -> String {
        // part ∈ {weight_ih, weight_hh, bias_ih, bias_hh}
        format!("decoder.prediction.dec_rnn.lstm.{part}_l{layer}")
    }

    // ── language conditioning (Nemotron prompt_kernel MLP) ──
    pub const PROMPT_K0_W: &str = "prompt_kernel.0.weight";
    pub const PROMPT_K0_B: &str = "prompt_kernel.0.bias";
    pub const PROMPT_K2_W: &str = "prompt_kernel.2.weight";
    pub const PROMPT_K2_B: &str = "prompt_kernel.2.bias";

    // ── RNN-T joint ──
    pub const JOINT_ENC_W: &str = "joint.enc.weight";
    pub const JOINT_ENC_B: &str = "joint.enc.bias";
    pub const JOINT_PRED_W: &str = "joint.pred.weight";
    pub const JOINT_PRED_B: &str = "joint.pred.bias";
    /// Final classifier inside `joint.joint_net` (the last `Linear`).
    pub const JOINT_NET_W: &str = "joint.joint_net.2.weight";
    pub const JOINT_NET_B: &str = "joint.joint_net.2.bias";
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_flow::WeightSource;

    // A tiny in-memory WeightSource to exercise the transpose path without
    // needing a real .nemo on disk.
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
        // [[1,2,3],[4,5,6]] -> transpose [[1,4],[2,5],[3,6]].
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
