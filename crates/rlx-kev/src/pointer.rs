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

//! The readout: score each option's `</opt>` hidden state against the
//! question's `<decide>` hidden state.
//!
//! ```text
//! z_j = ( K(h_opt_j) · Q(h_decide) ) / sqrt(dp)          dp = 256
//! p   = softmax(z / temperature)
//! ```
//!
//! Two deliberate choices, both inherited from `kev/model.py::PointerHead`:
//!
//! * **The head stays f32** even when the backbone runs bf16. It is two
//!   `[256, d]` matrices — the cost is nil and the calibration is not robust
//!   to being computed in half precision.
//! * **Temperature divides logits at inference only.** Training always sees
//!   `T = 1`, so a fitted value stays meaningful, and dividing by a positive
//!   scalar cannot change the argmax — only the probabilities.

use anyhow::{Result, bail};

/// The pointer dimension the released checkpoints use.
pub const DEFAULT_HEAD_DIM: usize = 256;

/// `q`/`k` projections plus the calibration temperature.
#[derive(Debug, Clone)]
pub struct PointerHead {
    /// `[dp, d]`, row-major (PyTorch `nn.Linear.weight` layout).
    pub q_weight: Vec<f32>,
    pub q_bias: Vec<f32>,
    /// `[dp, d]`, row-major.
    pub k_weight: Vec<f32>,
    pub k_bias: Vec<f32>,
    /// Backbone hidden size.
    pub hidden: usize,
    /// Pointer dimension `dp`.
    pub head_dim: usize,
    /// Divides the logits at inference. `1.0` = raw.
    pub temperature: f32,
}

impl PointerHead {
    /// Validate shapes up front so a mis-transposed checkpoint fails loudly
    /// instead of producing plausible-looking garbage.
    pub fn new(
        q_weight: Vec<f32>,
        q_bias: Vec<f32>,
        k_weight: Vec<f32>,
        k_bias: Vec<f32>,
        hidden: usize,
        head_dim: usize,
        temperature: f32,
    ) -> Result<Self> {
        let want = head_dim * hidden;
        if q_weight.len() != want || k_weight.len() != want {
            bail!(
                "pointer head weights must be [{head_dim}, {hidden}] = {want} values; \
                 got q={} k={}",
                q_weight.len(),
                k_weight.len()
            );
        }
        if q_bias.len() != head_dim || k_bias.len() != head_dim {
            bail!(
                "pointer head biases must be [{head_dim}]; got q={} k={}",
                q_bias.len(),
                k_bias.len()
            );
        }
        if !(temperature.is_finite() && temperature > 0.0) {
            bail!("temperature must be finite and positive, got {temperature}");
        }
        Ok(Self {
            q_weight,
            q_bias,
            k_weight,
            k_bias,
            hidden,
            head_dim,
            temperature,
        })
    }

    /// `1 / sqrt(dp)`.
    pub fn scale(&self) -> f32 {
        1.0 / (self.head_dim as f32).sqrt()
    }

    /// `W x + b` for a `[dp, d]` weight.
    fn project(&self, w: &[f32], b: &[f32], x: &[f32]) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.head_dim);
        for p in 0..self.head_dim {
            let row = &w[p * self.hidden..(p + 1) * self.hidden];
            let mut acc = b[p];
            for (wi, xi) in row.iter().zip(x) {
                acc += wi * xi;
            }
            out.push(acc);
        }
        out
    }

    /// Raw logits for one question. `h_opts` is `[K, hidden]`, row-major.
    ///
    /// Temperature is **not** applied here — see [`Self::logits_calibrated`].
    pub fn logits(&self, h_decide: &[f32], h_opts: &[f32]) -> Result<Vec<f32>> {
        if h_decide.len() != self.hidden {
            bail!(
                "decide hidden is {} wide, head expects {}",
                h_decide.len(),
                self.hidden
            );
        }
        if !h_opts.len().is_multiple_of(self.hidden) {
            bail!(
                "option hidden block of {} is not a multiple of {}",
                h_opts.len(),
                self.hidden
            );
        }
        let k = h_opts.len() / self.hidden;
        let qv = self.project(&self.q_weight, &self.q_bias, h_decide);
        let scale = self.scale();
        let mut z = Vec::with_capacity(k);
        for j in 0..k {
            let kv = self.project(
                &self.k_weight,
                &self.k_bias,
                &h_opts[j * self.hidden..(j + 1) * self.hidden],
            );
            let dot: f32 = kv.iter().zip(&qv).map(|(a, b)| a * b).sum();
            z.push(dot * scale);
        }
        Ok(z)
    }

    /// [`Self::logits`] divided by the checkpoint's temperature.
    pub fn logits_calibrated(&self, h_decide: &[f32], h_opts: &[f32]) -> Result<Vec<f32>> {
        let mut z = self.logits(h_decide, h_opts)?;
        if self.temperature != 1.0 {
            for v in &mut z {
                *v /= self.temperature;
            }
        }
        Ok(z)
    }

    /// Calibrated probabilities for one question.
    pub fn probs(&self, h_decide: &[f32], h_opts: &[f32]) -> Result<Vec<f32>> {
        Ok(softmax(&self.logits_calibrated(h_decide, h_opts)?))
    }
}

/// Numerically stable softmax.
pub fn softmax(z: &[f32]) -> Vec<f32> {
    if z.is_empty() {
        return Vec::new();
    }
    let m = z.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut out: Vec<f32> = z.iter().map(|v| (v - m).exp()).collect();
    let sum: f32 = out.iter().sum();
    if sum > 0.0 {
        for v in &mut out {
            *v /= sum;
        }
    }
    out
}
