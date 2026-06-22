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

//! Cross-modality decoder (`model.decoder`, 6 layers): query self-attention →
//! text cross-attention → image multi-scale deformable cross-attention → FFN,
//! with iterative box refinement. CPU-native reference.

use crate::config::GroundingDinoConfig;
use crate::decoder_ir::{DecoderLayerIr, DecoderLayerWeights};
use crate::deform_attn::{LevelShape, MsDeformAttn, RefPoints, level_start_index};
use crate::mlp::Mlp;
use crate::nn::{self, AttnBias};
use crate::weights::get;
use anyhow::Result;
use rlx_core::weight_map::WeightMap;
use rlx_runtime::Device;
use std::f32::consts::PI;

const SINE_TEMP: f32 = 10000.0;
const IS_EPS: f32 = 1e-5;
const NEG_INF: f32 = -1e30;

struct DecoderLayer {
    // query self-attention
    sa_q_w: Vec<f32>,
    sa_q_b: Vec<f32>,
    sa_k_w: Vec<f32>,
    sa_k_b: Vec<f32>,
    sa_v_w: Vec<f32>,
    sa_v_b: Vec<f32>,
    sa_o_w: Vec<f32>,
    sa_o_b: Vec<f32>,
    sa_ln_w: Vec<f32>,
    sa_ln_b: Vec<f32>,
    // text cross-attention
    ta_q_w: Vec<f32>,
    ta_q_b: Vec<f32>,
    ta_k_w: Vec<f32>,
    ta_k_b: Vec<f32>,
    ta_v_w: Vec<f32>,
    ta_v_b: Vec<f32>,
    ta_o_w: Vec<f32>,
    ta_o_b: Vec<f32>,
    ta_ln_w: Vec<f32>,
    ta_ln_b: Vec<f32>,
    // image deformable cross-attention
    deform: MsDeformAttn,
    da_ln_w: Vec<f32>,
    da_ln_b: Vec<f32>,
    // FFN
    fc1_w: Vec<f32>,
    fc1_b: Vec<f32>,
    fc2_w: Vec<f32>,
    fc2_b: Vec<f32>,
    final_ln_w: Vec<f32>,
    final_ln_b: Vec<f32>,
}

/// Cross-modality decoder.
pub struct Decoder {
    layers: Vec<DecoderLayer>,
    reference_points_head: Mlp, // 512→256→256
    bbox_embed: Mlp,            // shared 256→256→256→4
    layer_norm_w: Vec<f32>,
    layer_norm_b: Vec<f32>,
    d: usize,
    n_heads: usize,
    eps: f32,
}

/// Final decoder output.
pub struct DecoderOutput {
    /// `[nq, d]` last-layer hidden states (post `decoder.layer_norm`).
    pub hidden: Vec<f32>,
    /// `[nq, 4]` predicted boxes (cxcywh, normalized, sigmoid space).
    pub boxes: Vec<f32>,
}

impl Decoder {
    pub fn from_weights(wm: &WeightMap, cfg: &GroundingDinoConfig) -> Result<Self> {
        let d = cfg.d_model;
        let mut layers = Vec::with_capacity(cfg.decoder_layers);
        for i in 0..cfg.decoder_layers {
            let p = format!("model.decoder.layers.{i}.");
            layers.push(DecoderLayer {
                sa_q_w: get(wm, &format!("{p}self_attn.query.weight"))?,
                sa_q_b: get(wm, &format!("{p}self_attn.query.bias"))?,
                sa_k_w: get(wm, &format!("{p}self_attn.key.weight"))?,
                sa_k_b: get(wm, &format!("{p}self_attn.key.bias"))?,
                sa_v_w: get(wm, &format!("{p}self_attn.value.weight"))?,
                sa_v_b: get(wm, &format!("{p}self_attn.value.bias"))?,
                sa_o_w: get(wm, &format!("{p}self_attn.out_proj.weight"))?,
                sa_o_b: get(wm, &format!("{p}self_attn.out_proj.bias"))?,
                sa_ln_w: get(wm, &format!("{p}self_attn_layer_norm.weight"))?,
                sa_ln_b: get(wm, &format!("{p}self_attn_layer_norm.bias"))?,
                ta_q_w: get(wm, &format!("{p}encoder_attn_text.query.weight"))?,
                ta_q_b: get(wm, &format!("{p}encoder_attn_text.query.bias"))?,
                ta_k_w: get(wm, &format!("{p}encoder_attn_text.key.weight"))?,
                ta_k_b: get(wm, &format!("{p}encoder_attn_text.key.bias"))?,
                ta_v_w: get(wm, &format!("{p}encoder_attn_text.value.weight"))?,
                ta_v_b: get(wm, &format!("{p}encoder_attn_text.value.bias"))?,
                ta_o_w: get(wm, &format!("{p}encoder_attn_text.out_proj.weight"))?,
                ta_o_b: get(wm, &format!("{p}encoder_attn_text.out_proj.bias"))?,
                ta_ln_w: get(wm, &format!("{p}encoder_attn_text_layer_norm.weight"))?,
                ta_ln_b: get(wm, &format!("{p}encoder_attn_text_layer_norm.bias"))?,
                deform: MsDeformAttn::from_weights(
                    wm,
                    &format!("{p}encoder_attn."),
                    d,
                    cfg.decoder_attention_heads,
                    cfg.num_feature_levels,
                    cfg.decoder_n_points,
                )?,
                da_ln_w: get(wm, &format!("{p}encoder_attn_layer_norm.weight"))?,
                da_ln_b: get(wm, &format!("{p}encoder_attn_layer_norm.bias"))?,
                fc1_w: get(wm, &format!("{p}fc1.weight"))?,
                fc1_b: get(wm, &format!("{p}fc1.bias"))?,
                fc2_w: get(wm, &format!("{p}fc2.weight"))?,
                fc2_b: get(wm, &format!("{p}fc2.bias"))?,
                final_ln_w: get(wm, &format!("{p}final_layer_norm.weight"))?,
                final_ln_b: get(wm, &format!("{p}final_layer_norm.bias"))?,
            });
        }
        Ok(Self {
            layers,
            reference_points_head: Mlp::from_weights(wm, "model.decoder.reference_points_head", 2)?,
            bbox_embed: Mlp::from_weights(wm, "model.decoder.bbox_embed.0", 3)?,
            layer_norm_w: get(wm, "model.decoder.layer_norm.weight")?,
            layer_norm_b: get(wm, "model.decoder.layer_norm.bias")?,
            d,
            n_heads: cfg.decoder_attention_heads,
            eps: 1e-5,
        })
    }

    /// Run the decoder. `target`/`reference_points` come from query selection
    /// (`[nq, d]` / `[nq, 4]`). `memory` is the enhanced vision `[seq, d]`,
    /// `text` is `[Lt, d]`, `text_mask` is `[Lt]` (1 = valid).
    pub fn forward(
        &self,
        target: &[f32],
        reference_points: &[f32],
        memory: &[f32],
        text: &[f32],
        text_mask: &[u8],
        shapes: &[LevelShape],
    ) -> DecoderOutput {
        let d = self.d;
        let nq = target.len() / d;
        let lt = text.len() / d;
        let n_levels = shapes.len();
        let starts = level_start_index(shapes);

        // Text cross-attention key padding bias [nq, Lt] (shared across queries).
        let mut text_bias = vec![0f32; nq * lt];
        for q in 0..nq {
            for j in 0..lt {
                if text_mask[j] == 0 {
                    text_bias[q * lt + j] = NEG_INF;
                }
            }
        }

        let mut hidden = target.to_vec();
        let mut reference = reference_points.to_vec(); // [nq,4]
        // reference used by the head for the LAST layer (before its refinement).
        let mut ref_before_last = reference.clone();
        let mut last_normed = vec![0f32; nq * d];

        for (li, layer) in self.layers.iter().enumerate() {
            // query position embedding from the current reference points.
            let npf = d / 2;
            let sine = sine_pos_embed_boxes(&reference, nq, npf); // [nq, 4*npf]
            let query_pos = self.reference_points_head.forward(&sine, nq, 4 * npf);

            // reference points expanded across levels (valid_ratios = 1).
            let mut ref_input = vec![0f32; nq * n_levels * 4];
            for q in 0..nq {
                for l in 0..n_levels {
                    for c in 0..4 {
                        ref_input[(q * n_levels + l) * 4 + c] = reference[q * 4 + c];
                    }
                }
            }

            // 1. query self-attention (q=k=hidden+query_pos, v=hidden).
            let mut qk = vec![0f32; nq * d];
            for i in 0..nq * d {
                qk[i] = hidden[i] + query_pos[i];
            }
            let sa = nn::mha(
                &qk,
                &qk,
                &hidden,
                nq,
                nq,
                d,
                self.n_heads,
                &layer.sa_q_w,
                &layer.sa_q_b,
                &layer.sa_k_w,
                &layer.sa_k_b,
                &layer.sa_v_w,
                &layer.sa_v_b,
                &layer.sa_o_w,
                &layer.sa_o_b,
                AttnBias::None,
            );
            for i in 0..nq * d {
                hidden[i] += sa[i];
            }
            hidden = nn::layer_norm(&hidden, &layer.sa_ln_w, &layer.sa_ln_b, d, self.eps);

            // 2. text cross-attention (q=hidden+query_pos, k=v=text).
            let mut q2 = vec![0f32; nq * d];
            for i in 0..nq * d {
                q2[i] = hidden[i] + query_pos[i];
            }
            let ta = nn::mha(
                &q2,
                text,
                text,
                nq,
                lt,
                d,
                self.n_heads,
                &layer.ta_q_w,
                &layer.ta_q_b,
                &layer.ta_k_w,
                &layer.ta_k_b,
                &layer.ta_v_w,
                &layer.ta_v_b,
                &layer.ta_o_w,
                &layer.ta_o_b,
                AttnBias::Shared(&text_bias),
            );
            for i in 0..nq * d {
                hidden[i] += ta[i];
            }
            hidden = nn::layer_norm(&hidden, &layer.ta_ln_w, &layer.ta_ln_b, d, self.eps);

            // 3. image deformable cross-attention (query = hidden + query_pos).
            let mut q3 = vec![0f32; nq * d];
            for i in 0..nq * d {
                q3[i] = hidden[i] + query_pos[i];
            }
            let da = layer.deform.forward(
                &q3,
                memory,
                &RefPoints::Four(&ref_input),
                shapes,
                &starts,
                None,
            );
            for i in 0..nq * d {
                hidden[i] += da[i];
            }
            hidden = nn::layer_norm(&hidden, &layer.da_ln_w, &layer.da_ln_b, d, self.eps);

            // 4. FFN.
            let inter = layer.fc1_b.len();
            let mut f = nn::linear(&hidden, nq, d, &layer.fc1_w, inter, &layer.fc1_b);
            nn::relu(&mut f);
            let f2 = nn::linear(&f, nq, inter, &layer.fc2_w, d, &layer.fc2_b);
            for i in 0..nq * d {
                hidden[i] += f2[i];
            }
            hidden = nn::layer_norm(&hidden, &layer.final_ln_w, &layer.final_ln_b, d, self.eps);

            // box refinement (uses un-normed hidden, shared bbox_embed).
            let is_last = li == self.layers.len() - 1;
            if is_last {
                ref_before_last = reference.clone();
                last_normed =
                    nn::layer_norm(&hidden, &self.layer_norm_w, &self.layer_norm_b, d, self.eps);
            }
            let delta = self.bbox_embed.forward(&hidden, nq, d);
            let mut new_ref = vec![0f32; nq * 4];
            for i in 0..nq * 4 {
                new_ref[i] = nn::sigmoid(delta[i] + inverse_sigmoid(reference[i]));
            }
            reference = new_ref;
        }

        // Final boxes: head uses the NORMED last hidden + reference-before-last.
        let delta = self.bbox_embed.forward(&last_normed, nq, d);
        let mut boxes = vec![0f32; nq * 4];
        for i in 0..nq * 4 {
            boxes[i] = nn::sigmoid(delta[i] + inverse_sigmoid(ref_before_last[i]));
        }
        DecoderOutput {
            hidden: last_normed,
            boxes,
        }
    }

    /// Same as [`Self::forward`] but runs each layer's attention + FFN stack as a
    /// compiled HIR graph on `device` (the per-layer scalar glue — sine embed,
    /// reference-point head, box refinement — stays on the host). Bit-for-bit
    /// equivalent to the native path on `Device::Cpu`.
    pub fn forward_on_device(
        &self,
        target: &[f32],
        reference_points: &[f32],
        memory: &[f32],
        text: &[f32],
        text_mask: &[u8],
        shapes: &[LevelShape],
        device: Device,
    ) -> Result<DecoderOutput> {
        let d = self.d;
        let nq = target.len() / d;
        let lt = text.len() / d;
        let n_levels = shapes.len();

        let mut text_bias = vec![0f32; nq * lt];
        for q in 0..nq {
            for j in 0..lt {
                if text_mask[j] == 0 {
                    text_bias[q * lt + j] = NEG_INF;
                }
            }
        }

        let mut hidden = target.to_vec();
        let mut reference = reference_points.to_vec();
        let mut ref_before_last = reference.clone();
        let mut last_normed = vec![0f32; nq * d];

        for (li, layer) in self.layers.iter().enumerate() {
            let npf = d / 2;
            let sine = sine_pos_embed_boxes(&reference, nq, npf);
            let query_pos = self.reference_points_head.forward(&sine, nq, 4 * npf);

            let mut ref_input = vec![0f32; nq * n_levels * 4];
            for q in 0..nq {
                for l in 0..n_levels {
                    for c in 0..4 {
                        ref_input[(q * n_levels + l) * 4 + c] = reference[q * 4 + c];
                    }
                }
            }

            let dlw = layer_weights(layer);
            let ir = DecoderLayerIr::new(dlw, d, self.n_heads, layer.deform.n_points(), device);
            hidden = ir.forward(
                &hidden, &query_pos, &ref_input, memory, text, &text_bias, shapes,
            )?;

            let is_last = li == self.layers.len() - 1;
            if is_last {
                ref_before_last = reference.clone();
                last_normed =
                    nn::layer_norm(&hidden, &self.layer_norm_w, &self.layer_norm_b, d, self.eps);
            }
            let delta = self.bbox_embed.forward(&hidden, nq, d);
            let mut new_ref = vec![0f32; nq * 4];
            for i in 0..nq * 4 {
                new_ref[i] = nn::sigmoid(delta[i] + inverse_sigmoid(reference[i]));
            }
            reference = new_ref;
        }

        let delta = self.bbox_embed.forward(&last_normed, nq, d);
        let mut boxes = vec![0f32; nq * 4];
        for i in 0..nq * 4 {
            boxes[i] = nn::sigmoid(delta[i] + inverse_sigmoid(ref_before_last[i]));
        }
        Ok(DecoderOutput {
            hidden: last_normed,
            boxes,
        })
    }
}

/// Map a native decoder layer's weights into the IR layer-weight struct.
fn layer_weights(l: &DecoderLayer) -> DecoderLayerWeights {
    let (vw, vb, sw, sb, aw, ab, ow, ob) = l.deform.clone_proj();
    DecoderLayerWeights {
        sa_q_w: l.sa_q_w.clone(),
        sa_q_b: l.sa_q_b.clone(),
        sa_k_w: l.sa_k_w.clone(),
        sa_k_b: l.sa_k_b.clone(),
        sa_v_w: l.sa_v_w.clone(),
        sa_v_b: l.sa_v_b.clone(),
        sa_o_w: l.sa_o_w.clone(),
        sa_o_b: l.sa_o_b.clone(),
        sa_ln_w: l.sa_ln_w.clone(),
        sa_ln_b: l.sa_ln_b.clone(),
        ta_q_w: l.ta_q_w.clone(),
        ta_q_b: l.ta_q_b.clone(),
        ta_k_w: l.ta_k_w.clone(),
        ta_k_b: l.ta_k_b.clone(),
        ta_v_w: l.ta_v_w.clone(),
        ta_v_b: l.ta_v_b.clone(),
        ta_o_w: l.ta_o_w.clone(),
        ta_o_b: l.ta_o_b.clone(),
        ta_ln_w: l.ta_ln_w.clone(),
        ta_ln_b: l.ta_ln_b.clone(),
        da_value_w: vw,
        da_value_b: vb,
        da_samp_w: sw,
        da_samp_b: sb,
        da_attw_w: aw,
        da_attw_b: ab,
        da_out_w: ow,
        da_out_b: ob,
        da_ln_w: l.da_ln_w.clone(),
        da_ln_b: l.da_ln_b.clone(),
        fc1_w: l.fc1_w.clone(),
        fc1_b: l.fc1_b.clone(),
        fc2_w: l.fc2_w.clone(),
        fc2_b: l.fc2_b.clone(),
        final_ln_w: l.final_ln_w.clone(),
        final_ln_b: l.final_ln_b.clone(),
    }
}

fn inverse_sigmoid(x: f32) -> f32 {
    let x = x.clamp(0.0, 1.0);
    let x1 = x.max(IS_EPS);
    let x2 = (1.0 - x).max(IS_EPS);
    (x1 / x2).ln()
}

/// Sine position embedding of box references `[nq, 4]` → `[nq, 4*128]`, matching
/// `get_sine_pos_embed(..., exchange_xy=True)`.
fn sine_pos_embed_boxes(reference: &[f32], nq: usize, npf: usize) -> Vec<f32> {
    let scale = 2.0 * PI;
    let dim_t: Vec<f32> = (0..npf)
        .map(|i| SINE_TEMP.powf((2 * (i / 2)) as f32 / npf as f32))
        .collect();
    let coord_embed = |v: f32, out: &mut [f32]| {
        for i in 0..npf {
            let s = v * scale / dim_t[i];
            out[i] = if i % 2 == 0 { s.sin() } else { s.cos() };
        }
    };
    let mut out = vec![0f32; nq * 4 * npf];
    let mut tmp = vec![0f32; npf];
    for q in 0..nq {
        let cx = reference[q * 4];
        let cy = reference[q * 4 + 1];
        let w = reference[q * 4 + 2];
        let h = reference[q * 4 + 3];
        // exchange_xy: order becomes (cy, cx, w, h).
        let base = q * 4 * npf;
        coord_embed(cy, &mut tmp);
        out[base..base + npf].copy_from_slice(&tmp);
        coord_embed(cx, &mut tmp);
        out[base + npf..base + 2 * npf].copy_from_slice(&tmp);
        coord_embed(w, &mut tmp);
        out[base + 2 * npf..base + 3 * npf].copy_from_slice(&tmp);
        coord_embed(h, &mut tmp);
        out[base + 3 * npf..base + 4 * npf].copy_from_slice(&tmp);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inverse_sigmoid_roundtrip() {
        for &p in &[0.1f32, 0.3, 0.5, 0.7, 0.9] {
            let r = nn::sigmoid(inverse_sigmoid(p));
            assert!((r - p).abs() < 1e-4, "{p} -> {r}");
        }
    }

    #[test]
    fn sine_pos_embed_shape_and_finite() {
        let rp = vec![0.5f32, 0.5, 0.1, 0.2, 0.25, 0.75, 0.3, 0.4];
        let e = sine_pos_embed_boxes(&rp, 2, 128);
        assert_eq!(e.len(), 2 * 4 * 128);
        assert!(e.iter().all(|v| v.is_finite() && v.abs() <= 1.0001));
    }
}
