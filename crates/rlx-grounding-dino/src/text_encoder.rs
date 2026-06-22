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

//! BERT text backbone (CPU-native), with the Grounding DINO phrase
//! self-attention mask, plus the learned 768→256 text projection.

use crate::config::TextConfig;
use crate::nn::{self, AttnBias};
use crate::tokenizer::TextTokens;
use crate::weights::get;
use anyhow::Result;
use rlx_core::weight_map::WeightMap;

const NEG_INF: f32 = -1e30;

struct BertLayer {
    q_w: Vec<f32>,
    q_b: Vec<f32>,
    k_w: Vec<f32>,
    k_b: Vec<f32>,
    v_w: Vec<f32>,
    v_b: Vec<f32>,
    attn_out_w: Vec<f32>,
    attn_out_b: Vec<f32>,
    attn_ln_w: Vec<f32>,
    attn_ln_b: Vec<f32>,
    inter_w: Vec<f32>,
    inter_b: Vec<f32>,
    out_w: Vec<f32>,
    out_b: Vec<f32>,
    out_ln_w: Vec<f32>,
    out_ln_b: Vec<f32>,
}

/// Extracted BERT text backbone + text projection.
pub struct TextEncoder {
    cfg: TextConfig,
    word_emb: Vec<f32>,
    pos_emb: Vec<f32>,
    type_emb: Vec<f32>,
    emb_ln_w: Vec<f32>,
    emb_ln_b: Vec<f32>,
    layers: Vec<BertLayer>,
    proj_w: Vec<f32>, // [256, 768]
    proj_b: Vec<f32>, // [256]
    d_model: usize,
}

/// Output of the text backbone.
#[derive(Debug, Clone)]
pub struct TextFeatures {
    /// `[L, d_model]` projected text features.
    pub features: Vec<f32>,
    /// `[L, hidden]` raw BERT hidden states (pre-projection).
    pub hidden: Vec<f32>,
    pub seq_len: usize,
    pub d_model: usize,
}

impl TextEncoder {
    /// Extract from a full Grounding DINO [`WeightMap`].
    pub fn from_weights(wm: &WeightMap, cfg: TextConfig, d_model: usize) -> Result<Self> {
        let p = "model.text_backbone.";
        let layers = (0..cfg.num_hidden_layers)
            .map(|i| {
                let lp = format!("{p}encoder.layer.{i}.");
                Ok(BertLayer {
                    q_w: get(wm, &format!("{lp}attention.self.query.weight"))?,
                    q_b: get(wm, &format!("{lp}attention.self.query.bias"))?,
                    k_w: get(wm, &format!("{lp}attention.self.key.weight"))?,
                    k_b: get(wm, &format!("{lp}attention.self.key.bias"))?,
                    v_w: get(wm, &format!("{lp}attention.self.value.weight"))?,
                    v_b: get(wm, &format!("{lp}attention.self.value.bias"))?,
                    attn_out_w: get(wm, &format!("{lp}attention.output.dense.weight"))?,
                    attn_out_b: get(wm, &format!("{lp}attention.output.dense.bias"))?,
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
        })
    }

    /// Construct directly from in-memory parts (used by tests).
    #[allow(clippy::too_many_arguments)]
    #[cfg(test)]
    fn from_parts(
        cfg: TextConfig,
        d_model: usize,
        word_emb: Vec<f32>,
        pos_emb: Vec<f32>,
        type_emb: Vec<f32>,
        emb_ln_w: Vec<f32>,
        emb_ln_b: Vec<f32>,
        layers: Vec<BertLayer>,
        proj_w: Vec<f32>,
        proj_b: Vec<f32>,
    ) -> Self {
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
        }
    }

    /// Run the text backbone on a tokenized prompt.
    pub fn forward(&self, tokens: &TextTokens) -> TextFeatures {
        let h = self.cfg.hidden_size;
        let l = tokens.seq_len;
        let eps = self.cfg.layer_norm_eps as f32;

        // Embeddings.
        let mut x = vec![0f32; l * h];
        for i in 0..l {
            let wid = tokens.input_ids[i] as usize;
            let pid = tokens.position_ids[i] as usize;
            let tid = tokens.token_type_ids[i] as usize;
            let row = &mut x[i * h..(i + 1) * h];
            let we = &self.word_emb[wid * h..(wid + 1) * h];
            let pe = &self.pos_emb[pid * h..(pid + 1) * h];
            let te = &self.type_emb[tid * h..(tid + 1) * h];
            for d in 0..h {
                row[d] = we[d] + pe[d] + te[d];
            }
        }
        x = nn::layer_norm(&x, &self.emb_ln_w, &self.emb_ln_b, h, eps);

        // Additive 2-D self-attention bias from the phrase mask.
        let mut bias = vec![0f32; l * l];
        for i in 0..l * l {
            if tokens.self_attn_mask[i] == 0 {
                bias[i] = NEG_INF;
            }
        }

        for layer in &self.layers {
            // Self-attention + residual + LN.
            let attn = nn::mha(
                &x,
                &x,
                &x,
                l,
                l,
                h,
                self.cfg.num_attention_heads,
                &layer.q_w,
                &layer.q_b,
                &layer.k_w,
                &layer.k_b,
                &layer.v_w,
                &layer.v_b,
                &layer.attn_out_w,
                &layer.attn_out_b,
                AttnBias::Shared(&bias),
            );
            let mut res = vec![0f32; l * h];
            for i in 0..l * h {
                res[i] = attn[i] + x[i];
            }
            x = nn::layer_norm(&res, &layer.attn_ln_w, &layer.attn_ln_b, h, eps);

            // FFN + residual + LN.
            let inter_dim = layer.inter_b.len();
            let mut inter = nn::linear(&x, l, h, &layer.inter_w, inter_dim, &layer.inter_b);
            nn::gelu_erf(&mut inter);
            let ffn = nn::linear(&inter, l, inter_dim, &layer.out_w, h, &layer.out_b);
            let mut res2 = vec![0f32; l * h];
            for i in 0..l * h {
                res2[i] = ffn[i] + x[i];
            }
            x = nn::layer_norm(&res2, &layer.out_ln_w, &layer.out_ln_b, h, eps);
        }

        // Text projection 768 → d_model.
        let features = nn::linear(&x, l, h, &self.proj_w, self.d_model, &self.proj_b);
        TextFeatures {
            features,
            hidden: x,
            seq_len: l,
            d_model: self.d_model,
        }
    }
}

/// Shared synthetic-weights fixtures so the native and IR paths test against
/// identical parameters.
#[cfg(test)]
pub mod test_support {
    use super::*;

    #[derive(Clone)]
    pub struct LayerRaw {
        pub q_w: Vec<f32>,
        pub q_b: Vec<f32>,
        pub k_w: Vec<f32>,
        pub k_b: Vec<f32>,
        pub v_w: Vec<f32>,
        pub v_b: Vec<f32>,
        pub attn_out_w: Vec<f32>,
        pub attn_out_b: Vec<f32>,
        pub attn_ln_w: Vec<f32>,
        pub attn_ln_b: Vec<f32>,
        pub inter_w: Vec<f32>,
        pub inter_b: Vec<f32>,
        pub out_w: Vec<f32>,
        pub out_b: Vec<f32>,
        pub out_ln_w: Vec<f32>,
        pub out_ln_b: Vec<f32>,
    }

    pub struct Parts {
        pub cfg: TextConfig,
        pub d_model: usize,
        pub word_emb: Vec<f32>,
        pub pos_emb: Vec<f32>,
        pub type_emb: Vec<f32>,
        pub emb_ln_w: Vec<f32>,
        pub emb_ln_b: Vec<f32>,
        pub layers: Vec<LayerRaw>,
        pub proj_w: Vec<f32>,
        pub proj_b: Vec<f32>,
    }

    impl Parts {
        pub fn native(&self) -> TextEncoder {
            let layers = self
                .layers
                .iter()
                .map(|r| BertLayer {
                    q_w: r.q_w.clone(),
                    q_b: r.q_b.clone(),
                    k_w: r.k_w.clone(),
                    k_b: r.k_b.clone(),
                    v_w: r.v_w.clone(),
                    v_b: r.v_b.clone(),
                    attn_out_w: r.attn_out_w.clone(),
                    attn_out_b: r.attn_out_b.clone(),
                    attn_ln_w: r.attn_ln_w.clone(),
                    attn_ln_b: r.attn_ln_b.clone(),
                    inter_w: r.inter_w.clone(),
                    inter_b: r.inter_b.clone(),
                    out_w: r.out_w.clone(),
                    out_b: r.out_b.clone(),
                    out_ln_w: r.out_ln_w.clone(),
                    out_ln_b: r.out_ln_b.clone(),
                })
                .collect();
            TextEncoder::from_parts(
                self.cfg.clone(),
                self.d_model,
                self.word_emb.clone(),
                self.pos_emb.clone(),
                self.type_emb.clone(),
                self.emb_ln_w.clone(),
                self.emb_ln_b.clone(),
                layers,
                self.proj_w.clone(),
                self.proj_b.clone(),
            )
        }
    }

    fn det(n: usize, seed: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (((i * 13 + seed * 7) % 17) as f32 - 8.0) * 0.02)
            .collect()
    }

    pub fn synth_text_encoder() -> (TextConfig, usize, Parts) {
        let cfg = TextConfig {
            vocab_size: 1100,
            hidden_size: 8,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            intermediate_size: 16,
            max_position_embeddings: 32,
            type_vocab_size: 2,
            layer_norm_eps: 1e-12,
            hidden_act: "gelu".into(),
        };
        let h = cfg.hidden_size;
        let inter = cfg.intermediate_size;
        let d_model = 4;
        let layers = (0..cfg.num_hidden_layers)
            .map(|i| LayerRaw {
                q_w: det(h * h, 10 + i),
                q_b: vec![0.0; h],
                k_w: det(h * h, 20 + i),
                k_b: vec![0.0; h],
                v_w: det(h * h, 30 + i),
                v_b: vec![0.0; h],
                attn_out_w: det(h * h, 40 + i),
                attn_out_b: vec![0.0; h],
                attn_ln_w: vec![1.0; h],
                attn_ln_b: vec![0.0; h],
                inter_w: det(inter * h, 50 + i),
                inter_b: vec![0.0; inter],
                out_w: det(h * inter, 60 + i),
                out_b: vec![0.0; h],
                out_ln_w: vec![1.0; h],
                out_ln_b: vec![0.0; h],
            })
            .collect();
        let parts = Parts {
            word_emb: det(cfg.vocab_size * h, 1),
            pos_emb: det(cfg.max_position_embeddings * h, 2),
            type_emb: det(cfg.type_vocab_size * h, 3),
            emb_ln_w: vec![1.0; h],
            emb_ln_b: vec![0.0; h],
            layers,
            proj_w: det(d_model * h, 70),
            proj_b: vec![0.0; d_model],
            cfg: cfg.clone(),
            d_model,
        };
        (cfg, d_model, parts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::text_tokens_from_ids;

    fn tiny_cfg() -> TextConfig {
        TextConfig {
            vocab_size: 1100, // must fit the real special-token ids used below
            hidden_size: 8,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            intermediate_size: 16,
            max_position_embeddings: 16,
            type_vocab_size: 2,
            layer_norm_eps: 1e-12,
            hidden_act: "gelu".into(),
        }
    }

    fn z(n: usize) -> Vec<f32> {
        vec![0f32; n]
    }
    fn ones(n: usize) -> Vec<f32> {
        vec![1f32; n]
    }

    #[test]
    fn forward_shapes_and_finiteness() {
        let cfg = tiny_cfg();
        let h = cfg.hidden_size;
        let d_model = 4;
        let inter = cfg.intermediate_size;
        let layers: Vec<BertLayer> = (0..cfg.num_hidden_layers)
            .map(|i| {
                // Small deterministic, non-degenerate weights.
                let f = |scale: f32| -> Vec<f32> {
                    (0..h * h)
                        .map(|j| ((i * 7 + j) as f32 % 5.0 - 2.0) * 0.1 * scale)
                        .collect()
                };
                BertLayer {
                    q_w: f(1.0),
                    q_b: z(h),
                    k_w: f(1.1),
                    k_b: z(h),
                    v_w: f(0.9),
                    v_b: z(h),
                    attn_out_w: f(1.0),
                    attn_out_b: z(h),
                    attn_ln_w: ones(h),
                    attn_ln_b: z(h),
                    inter_w: (0..inter * h)
                        .map(|j| (j as f32 % 3.0 - 1.0) * 0.05)
                        .collect(),
                    inter_b: z(inter),
                    out_w: (0..h * inter)
                        .map(|j| (j as f32 % 3.0 - 1.0) * 0.05)
                        .collect(),
                    out_b: z(h),
                    out_ln_w: ones(h),
                    out_ln_b: z(h),
                }
            })
            .collect();
        let word_emb: Vec<f32> = (0..cfg.vocab_size * h)
            .map(|j| (j as f32 % 7.0) * 0.1)
            .collect();
        let enc = TextEncoder::from_parts(
            cfg.clone(),
            d_model,
            word_emb,
            (0..cfg.max_position_embeddings * h)
                .map(|j| (j as f32 % 4.0) * 0.05)
                .collect(),
            z(cfg.type_vocab_size * h),
            ones(h),
            z(h),
            layers,
            (0..d_model * h)
                .map(|j| (j as f32 % 5.0 - 2.0) * 0.1)
                .collect(),
            z(d_model),
        );
        // [CLS] a b . [SEP]
        let toks = text_tokens_from_ids(vec![101, 5, 6, 1012, 102]);
        let out = enc.forward(&toks);
        assert_eq!(out.seq_len, 5);
        assert_eq!(out.features.len(), 5 * d_model);
        assert_eq!(out.hidden.len(), 5 * h);
        assert!(out.features.iter().all(|v| v.is_finite()));
    }
}
