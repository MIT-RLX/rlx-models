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

//! CTC linear head + greedy decode for NeMo `ConvASRDecoder`.

use anyhow::{Result, ensure};
use rlx_flow::WeightSource;

use crate::weights::keys;

/// Host-side CTC classifier: `logits = enc @ Wᵀ + b` with W stored as
/// Conv1d `[C, D, 1]` in the checkpoint.
pub struct CtcHead {
    /// `[d_model, num_classes]` (transposed for row-vector × matrix).
    weight: Vec<f32>,
    bias: Vec<f32>,
    /// Encoder feature width.
    pub d_model: usize,
    /// Logit classes including blank.
    pub num_classes: usize,
}

impl CtcHead {
    /// Load `decoder.decoder_layers.0.{weight,bias}` from a weight source.
    pub fn from_weights(w: &mut dyn WeightSource) -> Result<Self> {
        let (wd, sh) = w.take(keys::CTC_W, false)?;
        // [num_classes, d_model, 1] or [num_classes, d_model]
        ensure!(
            sh.len() == 2 || sh.len() == 3,
            "CTC weight shape {sh:?}; expected [C,D] or [C,D,1]"
        );
        let num_classes = sh[0];
        let d_model = sh[1];
        let (bias, _) = w.take(keys::CTC_B, false)?;
        ensure!(bias.len() == num_classes, "CTC bias len {}", bias.len());

        // Transpose [C, D] -> [D, C] for enc_row @ W.
        let mut weight = vec![0.0f32; d_model * num_classes];
        for c in 0..num_classes {
            for d in 0..d_model {
                weight[d * num_classes + c] = wd[c * d_model + d];
            }
        }
        Ok(Self {
            weight,
            bias,
            d_model,
            num_classes,
        })
    }

    /// `enc` is `[t, d_model]` row-major → logits `[t, num_classes]`.
    pub fn logits(&self, enc: &[f32], t: usize) -> Result<Vec<f32>> {
        ensure!(
            enc.len() == t * self.d_model,
            "encoder len {} != t*d {}",
            enc.len(),
            t * self.d_model
        );
        let mut out = vec![0.0f32; t * self.num_classes];
        for ti in 0..t {
            let x = &enc[ti * self.d_model..(ti + 1) * self.d_model];
            let row = &mut out[ti * self.num_classes..(ti + 1) * self.num_classes];
            row.copy_from_slice(&self.bias);
            for d in 0..self.d_model {
                let xv = x[d];
                let wrow = &self.weight[d * self.num_classes..(d + 1) * self.num_classes];
                for c in 0..self.num_classes {
                    row[c] += xv * wrow[c];
                }
            }
        }
        Ok(out)
    }
}

/// Greedy CTC: argmax per frame → collapse repeats → drop blank.
pub fn greedy_decode(logits: &[f32], t: usize, num_classes: usize, blank_id: usize) -> Vec<u32> {
    let mut ids = Vec::new();
    let mut prev = blank_id;
    for ti in 0..t {
        let row = &logits[ti * num_classes..(ti + 1) * num_classes];
        let mut best = 0usize;
        let mut best_v = f32::NEG_INFINITY;
        for (c, &v) in row.iter().enumerate() {
            if v > best_v {
                best_v = v;
                best = c;
            }
        }
        if best != blank_id && best != prev {
            ids.push(best as u32);
        }
        prev = best;
    }
    ids
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_collapses_and_drops_blank() {
        // 3 frames, 4 classes, blank=3: [0,0,1] → [0,1]; blanks ignored.
        let logits = vec![
            2.0, 0.0, 0.0, 1.0, // 0
            3.0, 0.0, 0.0, 0.0, // 0 (repeat)
            0.0, 4.0, 0.0, 0.0, // 1
        ];
        let ids = greedy_decode(&logits, 3, 4, 3);
        assert_eq!(ids, vec![0, 1]);
    }
}
