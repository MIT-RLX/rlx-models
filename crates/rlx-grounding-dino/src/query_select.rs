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

//! Language-guided query selection (two-stage), matching HF
//! `generate_encoder_output_proposals` + top-k selection.

use crate::config::GroundingDinoConfig;
use crate::deform_attn::LevelShape;
use crate::mlp::Mlp;
use crate::nn;
use crate::weights::get;
use anyhow::Result;
use rlx_core::weight_map::WeightMap;

const INF: f32 = f32::INFINITY;

/// Query selection weights.
pub struct QuerySelect {
    enc_output_w: Vec<f32>, // Linear 256→256
    enc_output_b: Vec<f32>,
    enc_output_norm_w: Vec<f32>,
    enc_output_norm_b: Vec<f32>,
    bbox_embed: Mlp, // 256→256→256→4
    d: usize,
    num_queries: usize,
    eps: f32,
}

/// Selected decoder initialization.
pub struct Selection {
    /// `[num_queries, d]` content queries (`target`).
    pub target: Vec<f32>,
    /// `[num_queries, 4]` reference boxes (cxcywh, sigmoid space).
    pub reference_points: Vec<f32>,
}

impl QuerySelect {
    pub fn from_weights(wm: &WeightMap, cfg: &GroundingDinoConfig) -> Result<Self> {
        Ok(Self {
            enc_output_w: get(wm, "model.enc_output.weight")?,
            enc_output_b: get(wm, "model.enc_output.bias")?,
            enc_output_norm_w: get(wm, "model.enc_output_norm.weight")?,
            enc_output_norm_b: get(wm, "model.enc_output_norm.bias")?,
            bbox_embed: Mlp::from_weights(wm, "model.encoder_output_bbox_embed", 3)?,
            d: cfg.d_model,
            num_queries: cfg.num_queries,
            eps: 1e-5,
        })
    }

    /// `vision` is `[seq, d]` (enhancer output), `text` is `[Lt, d]`,
    /// `text_mask` is `[Lt]` (1 = valid token).
    pub fn forward(
        &self,
        vision: &[f32],
        text: &[f32],
        text_mask: &[u8],
        shapes: &[LevelShape],
    ) -> Selection {
        let d = self.d;
        let seq = vision.len() / d;
        let lt = text.len() / d;

        // Proposals (inverse-sigmoid) + validity per token.
        let (proposals, valid) = generate_proposals(shapes);

        // object_query = enc_output_norm(enc_output(vision)), invalid rows zeroed.
        let mut oq = vision.to_vec();
        for q in 0..seq {
            if !valid[q] {
                for c in 0..d {
                    oq[q * d + c] = 0.0;
                }
            }
        }
        oq = nn::linear(&oq, seq, d, &self.enc_output_w, d, &self.enc_output_b);
        oq = nn::layer_norm(
            &oq,
            &self.enc_output_norm_w,
            &self.enc_output_norm_b,
            d,
            self.eps,
        );

        // coord_logits = bbox_embed(oq) + proposals.
        let mut coord = self.bbox_embed.forward(&oq, seq, d);
        for i in 0..seq * 4 {
            coord[i] += proposals[i];
        }

        // class score per token = max over valid text tokens of dot product.
        let mut score = vec![-INF; seq];
        for q in 0..seq {
            if !valid[q] {
                continue; // invalid proposals never selected
            }
            let mut best = -INF;
            for j in 0..lt {
                if text_mask[j] == 0 {
                    continue;
                }
                let mut dot = 0f32;
                for c in 0..d {
                    dot += oq[q * d + c] * text[j * d + c];
                }
                if dot > best {
                    best = dot;
                }
            }
            score[q] = best;
        }

        // top-k indices by score (descending), stable on ties by index.
        let k = self.num_queries.min(seq);
        let mut order: Vec<usize> = (0..seq).collect();
        order.sort_by(|&a, &b| {
            score[b]
                .partial_cmp(&score[a])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });
        let topk = &order[..k];

        let mut target = vec![0f32; k * d];
        let mut reference_points = vec![0f32; k * 4];
        for (qi, &idx) in topk.iter().enumerate() {
            target[qi * d..(qi + 1) * d].copy_from_slice(&oq[idx * d..(idx + 1) * d]);
            for c in 0..4 {
                reference_points[qi * 4 + c] = nn::sigmoid(coord[idx * 4 + c]);
            }
        }
        Selection {
            target,
            reference_points,
        }
    }
}

/// Build `[seq, 4]` inverse-sigmoid proposals and a `[seq]` validity mask,
/// matching `generate_encoder_output_proposals`.
fn generate_proposals(shapes: &[LevelShape]) -> (Vec<f32>, Vec<bool>) {
    let seq: usize = shapes.iter().map(|s| s.h * s.w).sum();
    let mut proposals = vec![0f32; seq * 4];
    let mut valid = vec![true; seq];
    let mut q = 0;
    for (level, s) in shapes.iter().enumerate() {
        let wh = 0.05f32 * 2f32.powi(level as i32);
        for y in 0..s.h {
            let cy = (y as f32 + 0.5) / s.h as f32;
            for x in 0..s.w {
                let cx = (x as f32 + 0.5) / s.w as f32;
                let p = [cx, cy, wh, wh];
                let v = p.iter().all(|&z| z > 0.01 && z < 0.99);
                valid[q] = v;
                for c in 0..4 {
                    // inverse sigmoid; invalid → +inf (won't be selected).
                    proposals[q * 4 + c] = if v { (p[c] / (1.0 - p[c])).ln() } else { INF };
                }
                q += 1;
            }
        }
    }
    (proposals, valid)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn proposals_mark_edges_invalid() {
        // A wide level → leftmost column centers cx ≈ small < 0.01 → invalid.
        let shapes = [LevelShape { h: 1, w: 200 }];
        let (prop, valid) = generate_proposals(&shapes);
        assert_eq!(prop.len(), 200 * 4);
        assert!(!valid[0]); // cx = 0.5/200 = 0.0025 < 0.01
        assert!(valid[100]); // middle column valid
    }
}
