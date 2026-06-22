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

//! BERT text backbone as an on-device IR graph (native GPU on all backends).
//!
//! Token embeddings (cheap gathers) are computed on the host; the 12 transformer
//! layers and the 768→256 text projection run as a compiled HIR graph. Identical
//! math to [`crate::text_encoder`] — verified by the native-vs-IR parity test.

use crate::config::TextConfig;
use crate::ir::{self, Params};
use crate::nn;
use crate::text_encoder::TextFeatures;
use crate::tokenizer::TextTokens;
use crate::weights::get;
use anyhow::Result;
use rlx_ir::{DType, HirGraphExt, HirModule, HirMut, Shape};
use rlx_runtime::Device;

const NEG_INF: f32 = -1e30;

struct Layer {
    q_w: Vec<f32>,
    q_b: Vec<f32>,
    k_w: Vec<f32>,
    k_b: Vec<f32>,
    v_w: Vec<f32>,
    v_b: Vec<f32>,
    o_w: Vec<f32>,
    o_b: Vec<f32>,
    attn_ln_w: Vec<f32>,
    attn_ln_b: Vec<f32>,
    inter_w: Vec<f32>,
    inter_b: Vec<f32>,
    out_w: Vec<f32>,
    out_b: Vec<f32>,
    out_ln_w: Vec<f32>,
    out_ln_b: Vec<f32>,
}

/// IR text encoder.
pub struct TextEncoderIr {
    cfg: TextConfig,
    word_emb: Vec<f32>,
    pos_emb: Vec<f32>,
    type_emb: Vec<f32>,
    emb_ln_w: Vec<f32>,
    emb_ln_b: Vec<f32>,
    layers: Vec<Layer>,
    proj_w: Vec<f32>,
    proj_b: Vec<f32>,
    d_model: usize,
    device: Device,
}

impl TextEncoderIr {
    pub fn from_weights(
        wm: &rlx_core::weight_map::WeightMap,
        cfg: TextConfig,
        d_model: usize,
        device: Device,
    ) -> Result<Self> {
        let p = "model.text_backbone.";
        let layers = (0..cfg.num_hidden_layers)
            .map(|i| {
                let lp = format!("{p}encoder.layer.{i}.");
                Ok(Layer {
                    q_w: get(wm, &format!("{lp}attention.self.query.weight"))?,
                    q_b: get(wm, &format!("{lp}attention.self.query.bias"))?,
                    k_w: get(wm, &format!("{lp}attention.self.key.weight"))?,
                    k_b: get(wm, &format!("{lp}attention.self.key.bias"))?,
                    v_w: get(wm, &format!("{lp}attention.self.value.weight"))?,
                    v_b: get(wm, &format!("{lp}attention.self.value.bias"))?,
                    o_w: get(wm, &format!("{lp}attention.output.dense.weight"))?,
                    o_b: get(wm, &format!("{lp}attention.output.dense.bias"))?,
                    attn_ln_w: get(wm, &format!("{lp}attention.output.LayerNorm.weight"))?,
                    attn_ln_b: get(wm, &format!("{lp}attention.output.LayerNorm.bias"))?,
                    inter_w: get(wm, &format!("{lp}intermediate.dense.weight"))?,
                    inter_b: get(wm, &format!("{lp}intermediate.dense.bias"))?,
                    out_w: get(wm, &format!("{lp}output.dense.weight"))?,
                    out_b: get(wm, &format!("{lp}output.dense.bias"))?,
                    out_ln_w: get(wm, &format!("{lp}output.LayerNorm.weight"))?,
                    out_ln_b: get(wm, &format!("{lp}output.LayerNorm.bias"))?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            word_emb: get(wm, &format!("{p}embeddings.word_embeddings.weight"))?,
            pos_emb: get(wm, &format!("{p}embeddings.position_embeddings.weight"))?,
            type_emb: get(wm, &format!("{p}embeddings.token_type_embeddings.weight"))?,
            emb_ln_w: get(wm, &format!("{p}embeddings.LayerNorm.weight"))?,
            emb_ln_b: get(wm, &format!("{p}embeddings.LayerNorm.bias"))?,
            layers,
            proj_w: get(wm, "model.text_projection.weight")?,
            proj_b: get(wm, "model.text_projection.bias")?,
            cfg,
            d_model,
            device,
        })
    }

    /// Build parts directly (tests).
    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    fn from_parts(
        cfg: TextConfig,
        d_model: usize,
        device: Device,
        word_emb: Vec<f32>,
        pos_emb: Vec<f32>,
        type_emb: Vec<f32>,
        emb_ln_w: Vec<f32>,
        emb_ln_b: Vec<f32>,
        layers_raw: Vec<crate::text_encoder::test_support::LayerRaw>,
        proj_w: Vec<f32>,
        proj_b: Vec<f32>,
    ) -> Self {
        let layers = layers_raw
            .into_iter()
            .map(|r| Layer {
                q_w: r.q_w,
                q_b: r.q_b,
                k_w: r.k_w,
                k_b: r.k_b,
                v_w: r.v_w,
                v_b: r.v_b,
                o_w: r.attn_out_w,
                o_b: r.attn_out_b,
                attn_ln_w: r.attn_ln_w,
                attn_ln_b: r.attn_ln_b,
                inter_w: r.inter_w,
                inter_b: r.inter_b,
                out_w: r.out_w,
                out_b: r.out_b,
                out_ln_w: r.out_ln_w,
                out_ln_b: r.out_ln_b,
            })
            .collect();
        Self {
            cfg,
            word_emb,
            pos_emb,
            type_emb,
            emb_ln_w,
            emb_ln_b,
            layers,
            proj_w,
            proj_b,
            d_model,
            device,
        }
    }

    /// Run the text backbone on the chosen device. Returns the same
    /// [`TextFeatures`] as the native path.
    pub fn forward(&self, tokens: &TextTokens) -> Result<TextFeatures> {
        let h = self.cfg.hidden_size;
        let l = tokens.seq_len;
        let eps = self.cfg.layer_norm_eps as f32;
        let nh = self.cfg.num_attention_heads;

        // Host-side embeddings + embedding LayerNorm.
        let mut x = vec![0f32; l * h];
        for i in 0..l {
            let wid = tokens.input_ids[i] as usize;
            let pid = tokens.position_ids[i] as usize;
            let tid = tokens.token_type_ids[i] as usize;
            let row = &mut x[i * h..(i + 1) * h];
            let we = &self.word_emb[wid * h..(wid + 1) * h];
            let pe = &self.pos_emb[pid * h..(pid + 1) * h];
            let te = &self.type_emb[tid * h..(tid + 1) * h];
            for dd in 0..h {
                row[dd] = we[dd] + pe[dd] + te[dd];
            }
        }
        let hidden0 = nn::layer_norm(&x, &self.emb_ln_w, &self.emb_ln_b, h, eps);

        // Additive [1, l, l] attention bias from the phrase mask.
        let mut bias = vec![0f32; l * l];
        for i in 0..l * l {
            if tokens.self_attn_mask[i] == 0 {
                bias[i] = NEG_INF;
            }
        }

        // Build the transformer graph.
        let mut hir = HirModule::new("gdino_text");
        let mut params = Params::new();
        let mut g = HirMut::new(&mut hir);
        let hidden_in = g.input("hidden", Shape::new(&[l, h], DType::F32));
        let bias_in = g.input("bias", Shape::new(&[1, l, l], DType::F32));

        let mut cur = hidden_in;
        for (li, layer) in self.layers.iter().enumerate() {
            let attn = ir::mha(
                &mut g,
                &mut params,
                &format!("l{li}.attn"),
                cur,
                cur,
                cur,
                l,
                l,
                h,
                nh,
                &layer.q_w,
                &layer.q_b,
                &layer.k_w,
                &layer.k_b,
                &layer.v_w,
                &layer.v_b,
                &layer.o_w,
                &layer.o_b,
                bias_in,
            );
            let res = g.add(attn, cur);
            let normed = ir::layer_norm(
                &mut g,
                &mut params,
                &format!("l{li}.aln"),
                res,
                &layer.attn_ln_w,
                &layer.attn_ln_b,
                eps,
            );
            // FFN
            let inter_dim = layer.inter_b.len();
            let f1 = ir::linear(
                &mut g,
                &mut params,
                &format!("l{li}.fc1"),
                normed,
                h,
                inter_dim,
                &layer.inter_w,
                &layer.inter_b,
                1.0,
            );
            let act = g.gelu(f1);
            let f2 = ir::linear(
                &mut g,
                &mut params,
                &format!("l{li}.fc2"),
                act,
                inter_dim,
                h,
                &layer.out_w,
                &layer.out_b,
                1.0,
            );
            let res2 = g.add(f2, normed);
            cur = ir::layer_norm(
                &mut g,
                &mut params,
                &format!("l{li}.oln"),
                res2,
                &layer.out_ln_w,
                &layer.out_ln_b,
                eps,
            );
        }
        // Text projection h → d_model.
        let proj = ir::linear(
            &mut g,
            &mut params,
            "proj",
            cur,
            h,
            self.d_model,
            &self.proj_w,
            &self.proj_b,
            1.0,
        );
        g.set_outputs(vec![cur, proj]);

        let outs = ir::compile_and_run(
            hir,
            params,
            self.device,
            &[("hidden", &hidden0), ("bias", &bias)],
        )?;
        let hidden = outs[0].clone();
        let features = outs[1].clone();
        Ok(TextFeatures {
            features,
            hidden,
            seq_len: l,
            d_model: self.d_model,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::text_encoder::test_support::{LayerRaw, synth_text_encoder};
    use crate::tokenizer::text_tokens_from_ids;

    #[test]
    fn ir_matches_native_text_encoder() {
        let (cfg, d_model, parts) = synth_text_encoder();
        let native = parts.native();

        let layers_raw: Vec<LayerRaw> = parts.layers.clone();
        let ir = TextEncoderIr::from_parts(
            cfg.clone(),
            d_model,
            Device::Cpu,
            parts.word_emb.clone(),
            parts.pos_emb.clone(),
            parts.type_emb.clone(),
            parts.emb_ln_w.clone(),
            parts.emb_ln_b.clone(),
            layers_raw,
            parts.proj_w.clone(),
            parts.proj_b.clone(),
        );

        let toks = text_tokens_from_ids(vec![101, 500, 501, 1012, 502, 102]);
        let n = native.forward(&toks);
        let i = ir.forward(&toks).unwrap();
        assert_eq!(n.features.len(), i.features.len());
        let mut max_err = 0f32;
        for (a, b) in n.features.iter().zip(i.features.iter()) {
            max_err = max_err.max((a - b).abs());
        }
        assert!(
            max_err < 1e-3,
            "native vs IR text encoder max_err={max_err}"
        );
    }
}
