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

//! The Token-and-Duration Transducer joint network.
//!
//! Identical to a standard RNN-T joint (project encoder + prediction features,
//! sum, ReLU, classify) except the final classifier emits
//! `n_classes + num_durations` outputs: the first `n_classes` are the token
//! logits (the last of those is the blank), and the trailing `num_durations` are
//! the duration-head logits that index the duration table during decoding.

use anyhow::{Result, ensure};
use rlx_audio_blocks::decoders::tdt::TdtJointStep;
use rlx_flow::WeightSource;

/// Parakeet NeMo joint parameter keys (state-dict names).
mod keys {
    pub const ENC_W: &str = "joint.enc.weight";
    pub const ENC_B: &str = "joint.enc.bias";
    pub const PRED_W: &str = "joint.pred.weight";
    pub const PRED_B: &str = "joint.pred.bias";
    /// Final classifier inside `joint.joint_net` (the last `Linear`).
    pub const NET_W: &str = "joint.joint_net.2.weight";
    pub const NET_B: &str = "joint.joint_net.2.bias";
}

/// TDT joint network. `n_classes` counts the blank (which is the last token
/// class, `n_classes - 1`); `num_durations` is the width of the duration head.
pub struct TdtJoint {
    enc_w: Vec<f32>, // [joint_hidden, d_model]
    enc_b: Vec<f32>,
    d_model: usize,
    pred_w: Vec<f32>, // [joint_hidden, pred_in]
    pred_b: Vec<f32>,
    pred_in: usize,
    net_w: Vec<f32>, // [n_classes + num_durations, joint_hidden]
    net_b: Vec<f32>,
    joint_hidden: usize,
    /// Token classes including the blank at index `n_classes - 1`.
    pub n_classes: usize,
    /// Duration-head width; the trailing outputs of the classifier.
    pub num_durations: usize,
}

impl TdtJoint {
    /// Construct directly from raw row-major weights (used by tests and by any
    /// caller that has already materialised the joint tensors).
    #[allow(clippy::too_many_arguments)]
    pub fn from_raw(
        enc_w: Vec<f32>,
        enc_b: Vec<f32>,
        d_model: usize,
        pred_w: Vec<f32>,
        pred_b: Vec<f32>,
        pred_in: usize,
        net_w: Vec<f32>,
        net_b: Vec<f32>,
        joint_hidden: usize,
        n_classes: usize,
        num_durations: usize,
    ) -> Result<Self> {
        let n_out = n_classes + num_durations;
        ensure!(enc_w.len() == joint_hidden * d_model, "joint enc_w shape");
        ensure!(enc_b.len() == joint_hidden, "joint enc_b shape");
        ensure!(pred_w.len() == joint_hidden * pred_in, "joint pred_w shape");
        ensure!(pred_b.len() == joint_hidden, "joint pred_b shape");
        ensure!(net_w.len() == n_out * joint_hidden, "joint net_w shape");
        ensure!(net_b.len() == n_out, "joint net_b shape");
        ensure!(n_classes >= 2, "joint needs ≥1 token + blank");
        ensure!(num_durations >= 1, "TDT needs ≥1 duration");
        Ok(Self {
            enc_w,
            enc_b,
            d_model,
            pred_w,
            pred_b,
            pred_in,
            net_w,
            net_b,
            joint_hidden,
            n_classes,
            num_durations,
        })
    }

    /// Load the joint from a [`WeightSource`]. `num_durations` comes from the
    /// model config (the length of the `durations` table); `n_classes` is then
    /// inferred from the classifier output width.
    pub fn from_weights(w: &mut dyn WeightSource, num_durations: usize) -> Result<Self> {
        let (enc_w, enc_sh) = w.take(keys::ENC_W, false)?;
        let (enc_b, _) = w.take(keys::ENC_B, false)?;
        let (pred_w, pred_sh) = w.take(keys::PRED_W, false)?;
        let (pred_b, _) = w.take(keys::PRED_B, false)?;
        let (net_w, net_sh) = w.take(keys::NET_W, false)?;
        let (net_b, _) = w.take(keys::NET_B, false)?;

        let joint_hidden = enc_sh[0];
        let d_model = enc_sh[1];
        let pred_in = pred_sh[1];
        let n_out = net_sh[0];
        ensure!(pred_sh[0] == joint_hidden, "joint pred/enc hidden mismatch");
        ensure!(net_sh[1] == joint_hidden, "joint_net input width mismatch");
        ensure!(
            n_out > num_durations,
            "classifier width {n_out} must exceed num_durations {num_durations}"
        );
        let n_classes = n_out - num_durations;
        Self::from_raw(
            enc_w,
            enc_b,
            d_model,
            pred_w,
            pred_b,
            pred_in,
            net_w,
            net_b,
            joint_hidden,
            n_classes,
            num_durations,
        )
    }

    /// The blank token id (the last token class).
    pub fn blank_id(&self) -> usize {
        self.n_classes - 1
    }

    pub fn d_model(&self) -> usize {
        self.d_model
    }

    /// Project one encoder frame (`[d_model]`) into the joint hidden space.
    pub fn enc_proj(&self, enc_frame: &[f32]) -> Vec<f32> {
        matvec(
            &self.enc_w,
            &self.enc_b,
            &enc_frame[..self.d_model.min(enc_frame.len())],
            self.joint_hidden,
            self.d_model,
        )
    }

    fn pred_proj(&self, pred_out: &[f32]) -> Vec<f32> {
        matvec(
            &self.pred_w,
            &self.pred_b,
            pred_out,
            self.joint_hidden,
            self.pred_in,
        )
    }

    /// Full classifier output (`[n_classes + num_durations]`) for a projected
    /// encoder frame and a prediction-net hidden vector.
    pub fn logits(&self, enc_proj: &[f32], pred_out: &[f32]) -> Vec<f32> {
        let pp = self.pred_proj(pred_out);
        let mut h = vec![0.0f32; self.joint_hidden];
        for j in 0..self.joint_hidden {
            h[j] = (enc_proj[j] + pp[j]).max(0.0); // sum + ReLU
        }
        matvec(
            &self.net_w,
            &self.net_b,
            &h,
            self.n_classes + self.num_durations,
            self.joint_hidden,
        )
    }

    /// Argmax the token head and the duration head independently — the single
    /// step consumed by the TDT greedy loop.
    pub fn argmax_step(&self, enc_proj: &[f32], pred_out: &[f32]) -> TdtJointStep {
        let logits = self.logits(enc_proj, pred_out);
        let (token_logits, dur_logits) = logits.split_at(self.n_classes);
        let (label, label_score) = argmax(token_logits);
        let (duration_index, _) = argmax(dur_logits);
        TdtJointStep {
            label: label as i32,
            label_score,
            duration_index: duration_index as i32,
        }
    }
}

fn matvec(w: &[f32], b: &[f32], x: &[f32], out: usize, inp: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; out];
    for o in 0..out {
        let row = &w[o * inp..(o + 1) * inp];
        let mut acc = b.get(o).copied().unwrap_or(0.0);
        for k in 0..inp.min(x.len()) {
            acc += row[k] * x[k];
        }
        y[o] = acc;
    }
    y
}

fn argmax(v: &[f32]) -> (usize, f32) {
    let mut best = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > bv {
            bv = x;
            best = i;
        }
    }
    (best, bv)
}
