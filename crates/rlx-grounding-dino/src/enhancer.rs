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

//! Feature enhancer (`model.encoder`, 6 layers). Each layer runs, in order:
//! bidirectional vision↔text fusion → text self-attention enhancer → vision
//! multi-scale deformable self-attention. CPU-native reference.

use crate::config::GroundingDinoConfig;
use crate::deform_attn::{LevelShape, MsDeformAttn, RefPoints, level_start_index};
use crate::nn::{self, AttnBias, softmax_rows};
use crate::weights::get;
use anyhow::Result;
use rlx_core::weight_map::WeightMap;

const CLAMP: f32 = 50000.0;
const NEG_INF: f32 = -1e30;

struct FusionLayer {
    vision_proj_w: Vec<f32>,
    vision_proj_b: Vec<f32>,
    text_proj_w: Vec<f32>,
    text_proj_b: Vec<f32>,
    values_vision_proj_w: Vec<f32>,
    values_vision_proj_b: Vec<f32>,
    values_text_proj_w: Vec<f32>,
    values_text_proj_b: Vec<f32>,
    out_vision_proj_w: Vec<f32>,
    out_vision_proj_b: Vec<f32>,
    out_text_proj_w: Vec<f32>,
    out_text_proj_b: Vec<f32>,
    ln_vision_w: Vec<f32>,
    ln_vision_b: Vec<f32>,
    ln_text_w: Vec<f32>,
    ln_text_b: Vec<f32>,
    vision_param: Vec<f32>,
    text_param: Vec<f32>,
}

struct TextEnhancer {
    q_w: Vec<f32>,
    q_b: Vec<f32>,
    k_w: Vec<f32>,
    k_b: Vec<f32>,
    v_w: Vec<f32>,
    v_b: Vec<f32>,
    out_w: Vec<f32>,
    out_b: Vec<f32>,
    fc1_w: Vec<f32>,
    fc1_b: Vec<f32>,
    fc2_w: Vec<f32>,
    fc2_b: Vec<f32>,
    ln_before_w: Vec<f32>,
    ln_before_b: Vec<f32>,
    ln_after_w: Vec<f32>,
    ln_after_b: Vec<f32>,
}

struct DeformLayer {
    attn: MsDeformAttn,
    sa_ln_w: Vec<f32>,
    sa_ln_b: Vec<f32>,
    fc1_w: Vec<f32>,
    fc1_b: Vec<f32>,
    fc2_w: Vec<f32>,
    fc2_b: Vec<f32>,
    final_ln_w: Vec<f32>,
    final_ln_b: Vec<f32>,
}

struct EncoderLayer {
    fusion: FusionLayer,
    text_enhancer: TextEnhancer,
    deform: DeformLayer,
}

/// Feature enhancer encoder.
pub struct Encoder {
    layers: Vec<EncoderLayer>,
    d: usize,
    n_heads: usize,
    n_levels: usize,
    n_points: usize,
    /// Fusion BiMultiHeadAttention inner dim (`encoder_ffn_dim / 2`).
    bi_dim: usize,
    eps: f32,
}

/// Outputs of the enhancer.
pub struct EncoderOutput {
    /// `[seq, d]` enhanced vision features.
    pub vision: Vec<f32>,
    /// `[Lt, d]` enhanced text features.
    pub text: Vec<f32>,
}

impl Encoder {
    pub fn from_weights(wm: &WeightMap, cfg: &GroundingDinoConfig) -> Result<Self> {
        let d = cfg.d_model;
        let mut layers = Vec::with_capacity(cfg.encoder_layers);
        for i in 0..cfg.encoder_layers {
            let fp = format!("model.encoder.layers.{i}.fusion_layer.");
            let tp = format!("model.encoder.layers.{i}.text_enhancer_layer.");
            let dp = format!("model.encoder.layers.{i}.deformable_layer.");
            let fusion = FusionLayer {
                vision_proj_w: get(wm, &format!("{fp}attn.vision_proj.weight"))?,
                vision_proj_b: get(wm, &format!("{fp}attn.vision_proj.bias"))?,
                text_proj_w: get(wm, &format!("{fp}attn.text_proj.weight"))?,
                text_proj_b: get(wm, &format!("{fp}attn.text_proj.bias"))?,
                values_vision_proj_w: get(wm, &format!("{fp}attn.values_vision_proj.weight"))?,
                values_vision_proj_b: get(wm, &format!("{fp}attn.values_vision_proj.bias"))?,
                values_text_proj_w: get(wm, &format!("{fp}attn.values_text_proj.weight"))?,
                values_text_proj_b: get(wm, &format!("{fp}attn.values_text_proj.bias"))?,
                out_vision_proj_w: get(wm, &format!("{fp}attn.out_vision_proj.weight"))?,
                out_vision_proj_b: get(wm, &format!("{fp}attn.out_vision_proj.bias"))?,
                out_text_proj_w: get(wm, &format!("{fp}attn.out_text_proj.weight"))?,
                out_text_proj_b: get(wm, &format!("{fp}attn.out_text_proj.bias"))?,
                ln_vision_w: get(wm, &format!("{fp}layer_norm_vision.weight"))?,
                ln_vision_b: get(wm, &format!("{fp}layer_norm_vision.bias"))?,
                ln_text_w: get(wm, &format!("{fp}layer_norm_text.weight"))?,
                ln_text_b: get(wm, &format!("{fp}layer_norm_text.bias"))?,
                vision_param: get(wm, &format!("{fp}vision_param"))?,
                text_param: get(wm, &format!("{fp}text_param"))?,
            };
            let text_enhancer = TextEnhancer {
                q_w: get(wm, &format!("{tp}self_attn.query.weight"))?,
                q_b: get(wm, &format!("{tp}self_attn.query.bias"))?,
                k_w: get(wm, &format!("{tp}self_attn.key.weight"))?,
                k_b: get(wm, &format!("{tp}self_attn.key.bias"))?,
                v_w: get(wm, &format!("{tp}self_attn.value.weight"))?,
                v_b: get(wm, &format!("{tp}self_attn.value.bias"))?,
                out_w: get(wm, &format!("{tp}self_attn.out_proj.weight"))?,
                out_b: get(wm, &format!("{tp}self_attn.out_proj.bias"))?,
                fc1_w: get(wm, &format!("{tp}fc1.weight"))?,
                fc1_b: get(wm, &format!("{tp}fc1.bias"))?,
                fc2_w: get(wm, &format!("{tp}fc2.weight"))?,
                fc2_b: get(wm, &format!("{tp}fc2.bias"))?,
                ln_before_w: get(wm, &format!("{tp}layer_norm_before.weight"))?,
                ln_before_b: get(wm, &format!("{tp}layer_norm_before.bias"))?,
                ln_after_w: get(wm, &format!("{tp}layer_norm_after.weight"))?,
                ln_after_b: get(wm, &format!("{tp}layer_norm_after.bias"))?,
            };
            let deform = DeformLayer {
                attn: MsDeformAttn::from_weights(
                    wm,
                    &format!("{dp}self_attn."),
                    d,
                    cfg.encoder_attention_heads,
                    cfg.num_feature_levels,
                    cfg.encoder_n_points,
                )?,
                sa_ln_w: get(wm, &format!("{dp}self_attn_layer_norm.weight"))?,
                sa_ln_b: get(wm, &format!("{dp}self_attn_layer_norm.bias"))?,
                fc1_w: get(wm, &format!("{dp}fc1.weight"))?,
                fc1_b: get(wm, &format!("{dp}fc1.bias"))?,
                fc2_w: get(wm, &format!("{dp}fc2.weight"))?,
                fc2_b: get(wm, &format!("{dp}fc2.bias"))?,
                final_ln_w: get(wm, &format!("{dp}final_layer_norm.weight"))?,
                final_ln_b: get(wm, &format!("{dp}final_layer_norm.bias"))?,
            };
            layers.push(EncoderLayer {
                fusion,
                text_enhancer,
                deform,
            });
        }
        Ok(Self {
            layers,
            d,
            n_heads: cfg.encoder_attention_heads,
            n_levels: cfg.num_feature_levels,
            n_points: cfg.encoder_n_points,
            bi_dim: cfg.encoder_ffn_dim / 2,
            eps: 1e-5,
        })
    }

    /// Reference points for deformable self-attention: every image token uses
    /// its own normalized `(x, y)` center, replicated across all levels.
    /// Returns `[seq, n_levels, 2]`.
    pub fn reference_points(shapes: &[LevelShape]) -> Vec<f32> {
        let n_levels = shapes.len();
        let seq: usize = shapes.iter().map(|s| s.h * s.w).sum();
        let mut out = vec![0f32; seq * n_levels * 2];
        let mut q = 0;
        for s in shapes {
            for y in 0..s.h {
                let ry = (y as f32 + 0.5) / s.h as f32;
                for x in 0..s.w {
                    let rx = (x as f32 + 0.5) / s.w as f32;
                    for l in 0..n_levels {
                        out[(q * n_levels + l) * 2] = rx;
                        out[(q * n_levels + l) * 2 + 1] = ry;
                    }
                    q += 1;
                }
            }
        }
        out
    }

    /// Run all enhancer layers. `vision`/`vision_pos` are `[seq, d]`, `text`/
    /// `text_pos` are `[Lt, d]`, `text_self_mask` is `[Lt, Lt]` (1 = attend).
    pub fn forward(
        &self,
        vision: &[f32],
        vision_pos: &[f32],
        text: &[f32],
        text_pos: &[f32],
        text_self_mask: &[u8],
        shapes: &[LevelShape],
    ) -> EncoderOutput {
        let d = self.d;
        let lt = text.len() / d;
        let starts = level_start_index(shapes);
        let ref_points = Self::reference_points(shapes);

        // Additive text self-attention bias [Lt, Lt].
        let mut text_bias = vec![0f32; lt * lt];
        for i in 0..lt * lt {
            if text_self_mask[i] == 0 {
                text_bias[i] = NEG_INF;
            }
        }

        let mut vision = vision.to_vec();
        let mut text = text.to_vec();

        for layer in &self.layers {
            // 1. Fusion (bidirectional cross-attention; residual + layerscale inside).
            let (v2, t2) = self.fusion(&layer.fusion, &vision, &text);
            vision = v2;
            text = t2;

            // 2. Text enhancer (text self-attention + FFN).
            text = self.text_enhancer(&layer.text_enhancer, &text, text_pos, &text_bias);

            // 3. Deformable vision self-attention + FFN.
            vision = self.deformable(
                &layer.deform,
                &vision,
                vision_pos,
                &ref_points,
                shapes,
                &starts,
            );
        }

        EncoderOutput { vision, text }
    }

    /// BiMultiHeadAttention. Returns the updated `(vision, text)` — the
    /// layerscale delta added to the LAYER-NORMED features (matching HF).
    fn fusion(&self, f: &FusionLayer, vision: &[f32], text: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let d = self.d;
        let lv = vision.len() / d;
        let lt = text.len() / d;
        // HF GroundingDinoBiMultiHeadAttention uses `encoder_attention_heads // 2`.
        let nh = self.n_heads / 2;
        let bi_dim = self.bi_dim;
        let hd = bi_dim / nh;
        let scale = 1.0 / (hd as f32).sqrt();

        let vn = nn::layer_norm(vision, &f.ln_vision_w, &f.ln_vision_b, d, self.eps);
        let tn = nn::layer_norm(text, &f.ln_text_w, &f.ln_text_b, d, self.eps);

        let mut qv = nn::linear(&vn, lv, d, &f.vision_proj_w, bi_dim, &f.vision_proj_b);
        for v in qv.iter_mut() {
            *v *= scale;
        }
        let kt = nn::linear(&tn, lt, d, &f.text_proj_w, bi_dim, &f.text_proj_b);
        let vv = nn::linear(
            &vn,
            lv,
            d,
            &f.values_vision_proj_w,
            bi_dim,
            &f.values_vision_proj_b,
        );
        let vt = nn::linear(
            &tn,
            lt,
            d,
            &f.values_text_proj_w,
            bi_dim,
            &f.values_text_proj_b,
        );

        let mut vis_ctx = vec![0f32; lv * bi_dim];
        let mut txt_ctx = vec![0f32; lt * bi_dim];

        for h in 0..nh {
            // A[i,j] = <qv_i, kt_j>, clamped.
            let mut a = vec![0f32; lv * lt];
            for i in 0..lv {
                for j in 0..lt {
                    let mut s = 0f32;
                    for c in 0..hd {
                        s += qv[i * bi_dim + h * hd + c] * kt[j * bi_dim + h * hd + c];
                    }
                    a[i * lt + j] = s.clamp(-CLAMP, CLAMP);
                }
            }
            // vision attends to text: softmax over j → @ vt.
            let mut av = a.clone();
            softmax_rows(&mut av, lv, lt);
            for i in 0..lv {
                let ctx = &mut vis_ctx[i * bi_dim + h * hd..i * bi_dim + h * hd + hd];
                for j in 0..lt {
                    let p = av[i * lt + j];
                    for c in 0..hd {
                        ctx[c] += p * vt[j * bi_dim + h * hd + c];
                    }
                }
            }
            // text attends to vision: softmax of A^T over i → @ vv.
            let mut at = vec![0f32; lt * lv];
            for j in 0..lt {
                for i in 0..lv {
                    at[j * lv + i] = a[i * lt + j];
                }
            }
            softmax_rows(&mut at, lt, lv);
            for j in 0..lt {
                let ctx = &mut txt_ctx[j * bi_dim + h * hd..j * bi_dim + h * hd + hd];
                for i in 0..lv {
                    let p = at[j * lv + i];
                    for c in 0..hd {
                        ctx[c] += p * vv[i * bi_dim + h * hd + c];
                    }
                }
            }
        }
        let dv = nn::linear(
            &vis_ctx,
            lv,
            bi_dim,
            &f.out_vision_proj_w,
            d,
            &f.out_vision_proj_b,
        );
        let dt = nn::linear(
            &txt_ctx,
            lt,
            bi_dim,
            &f.out_text_proj_w,
            d,
            &f.out_text_proj_b,
        );
        // Residual on the layer-normed features (vn/tn), per HF.
        let mut vo = vn;
        for i in 0..lv * d {
            vo[i] += f.vision_param[i % d] * dv[i];
        }
        let mut to = tn;
        for i in 0..lt * d {
            to[i] += f.text_param[i % d] * dt[i];
        }
        (vo, to)
    }

    fn text_enhancer(
        &self,
        t: &TextEnhancer,
        text: &[f32],
        text_pos: &[f32],
        bias: &[f32],
    ) -> Vec<f32> {
        let d = self.d;
        let lt = text.len() / d;
        // q = k = text + pos; v = text.
        let mut qk = vec![0f32; lt * d];
        for i in 0..lt * d {
            qk[i] = text[i] + text_pos[i];
        }
        // HF GroundingDinoTextEnhancerLayer uses `encoder_attention_heads // 2`.
        let attn = nn::mha(
            &qk,
            &qk,
            text,
            lt,
            lt,
            d,
            self.n_heads / 2,
            &t.q_w,
            &t.q_b,
            &t.k_w,
            &t.k_b,
            &t.v_w,
            &t.v_b,
            &t.out_w,
            &t.out_b,
            AttnBias::Shared(bias),
        );
        let mut hidden = vec![0f32; lt * d];
        for i in 0..lt * d {
            hidden[i] = text[i] + attn[i];
        }
        hidden = nn::layer_norm(&hidden, &t.ln_before_w, &t.ln_before_b, d, self.eps);
        // FFN (relu).
        let inter = t.fc1_b.len();
        let mut f = nn::linear(&hidden, lt, d, &t.fc1_w, inter, &t.fc1_b);
        nn::relu(&mut f);
        let f2 = nn::linear(&f, lt, inter, &t.fc2_w, d, &t.fc2_b);
        for i in 0..lt * d {
            hidden[i] += f2[i];
        }
        nn::layer_norm(&hidden, &t.ln_after_w, &t.ln_after_b, d, self.eps)
    }

    fn deformable(
        &self,
        dl: &DeformLayer,
        vision: &[f32],
        vision_pos: &[f32],
        ref_points: &[f32],
        shapes: &[LevelShape],
        starts: &[usize],
    ) -> Vec<f32> {
        let d = self.d;
        let seq = vision.len() / d;
        // query = vision + pos (for offsets/weights); value = vision.
        let mut query = vec![0f32; seq * d];
        for i in 0..seq * d {
            query[i] = vision[i] + vision_pos[i];
        }
        let attn = dl.attn.forward(
            &query,
            vision,
            &RefPoints::Two(ref_points),
            shapes,
            starts,
            None,
        );
        let mut hidden = vec![0f32; seq * d];
        for i in 0..seq * d {
            hidden[i] = vision[i] + attn[i];
        }
        hidden = nn::layer_norm(&hidden, &dl.sa_ln_w, &dl.sa_ln_b, d, self.eps);
        let inter = dl.fc1_b.len();
        let mut f = nn::linear(&hidden, seq, d, &dl.fc1_w, inter, &dl.fc1_b);
        nn::relu(&mut f);
        let f2 = nn::linear(&f, seq, inter, &dl.fc2_w, d, &dl.fc2_b);
        for i in 0..seq * d {
            hidden[i] += f2[i];
        }
        nn::layer_norm(&hidden, &dl.final_ln_w, &dl.final_ln_b, d, self.eps)
    }

    pub fn n_levels(&self) -> usize {
        self.n_levels
    }
    pub fn n_points(&self) -> usize {
        self.n_points
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_points_are_normalized_centers() {
        let shapes = [LevelShape { h: 2, w: 2 }, LevelShape { h: 1, w: 1 }];
        let rp = Encoder::reference_points(&shapes);
        // seq = 4 + 1 = 5, n_levels = 2.
        assert_eq!(rp.len(), 5 * 2 * 2);
        // first token of level0: center (0.25, 0.25), same for both levels.
        assert!((rp[0] - 0.25).abs() < 1e-6);
        assert!((rp[1] - 0.25).abs() < 1e-6);
        assert!((rp[2] - 0.25).abs() < 1e-6);
        // last token (level1 single cell) center (0.5,0.5).
        let last = (4 * 2) * 2;
        assert!((rp[last] - 0.5).abs() < 1e-6);
    }
}
