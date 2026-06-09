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

use super::config::Wav2Vec2BertConfig;
use anyhow::{Result, bail};
use rlx_core::flow_util::WeightMapSource;
use rlx_core::weight_map::WeightMap;
use rlx_flow::WeightSource;
use rlx_ir::hir::{FusionPolicy, HirModule, HirNodeId};
use rlx_ir::op::{Activation, BinaryOp, MaskKind};
use rlx_ir::{DType, Graph, Op, Shape};
use std::collections::HashMap;

const ATTN_MASK_NEG: f32 = -1e4;
const FFN_RESIDUAL_SCALE: f32 = 0.5;

/// Stop building at an intermediate point inside encoder layer `L`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum W2vLayerStop {
    AfterFfn1,
    AfterAttn,
    AfterConv,
    AfterFfn2,
    Final,
}

pub(crate) struct W2vBuilder<'a> {
    hir: &'a mut HirModule,
    params: &'a mut HashMap<String, Vec<f32>>,
    weights: &'a mut dyn WeightSource,
    batch: usize,
    seq: usize,
    f: DType,
}

fn build_wav2vec2_bert_hir_inner(
    cfg: &Wav2Vec2BertConfig,
    weights: &mut WeightMap,
    batch: usize,
    seq: usize,
    probe: Option<(usize, W2vLayerStop)>,
) -> Result<(HirModule, HashMap<String, Vec<f32>>)> {
    validate_cfg(cfg)?;
    let feat_dim = cfg.feature_projection_input_dim;
    let f = DType::F32;
    let mut hir = HirModule::new("wav2vec2_bert").with_fusion_policy(FusionPolicy::Direct);
    let mut params = HashMap::new();
    let input_features = hir.input("input_features", Shape::new(&[batch, seq, feat_dim], f));
    let attention_mask = hir.input("attention_mask", Shape::new(&[batch, seq], f));
    let mut src = WeightMapSource(weights);
    let mut b = W2vBuilder::from_emit_parts(&mut hir, &mut params, &mut src, batch, seq);
    let hidden = b.emit_encoder(input_features, attention_mask, probe, cfg)?;
    hir.outputs = vec![hidden];
    Ok((hir, params))
}

/// Build a Wav2Vec2-BERT encoder IR graph for concrete `batch` × `seq`.
pub fn build_wav2vec2_bert_graph_sized(
    cfg: &Wav2Vec2BertConfig,
    weights: &mut WeightMap,
    batch: usize,
    seq: usize,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    rlx_core::flow_util::graph_from_built(crate::flow::build_wav2vec2_bert_built(
        cfg, weights, batch, seq,
    )?)
}

/// Build a graph that stops after sub-step `stop` of encoder layer `stop_layer`.
pub fn build_wav2vec2_bert_graph_probe(
    cfg: &Wav2Vec2BertConfig,
    weights: &mut WeightMap,
    batch: usize,
    seq: usize,
    stop_layer: usize,
    stop: W2vLayerStop,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    build_wav2vec2_bert_graph_inner(cfg, weights, batch, seq, Some((stop_layer, stop)))
}

/// Build a Wav2Vec2-BERT encoder HIR module (no MIR/graph lowering).
pub fn build_wav2vec2_bert_hir_sized(
    cfg: &Wav2Vec2BertConfig,
    weights: &mut WeightMap,
    batch: usize,
    seq: usize,
) -> Result<(HirModule, HashMap<String, Vec<f32>>)> {
    build_wav2vec2_bert_hir_inner(cfg, weights, batch, seq, None)
}

fn build_wav2vec2_bert_graph_inner(
    cfg: &Wav2Vec2BertConfig,
    weights: &mut WeightMap,
    batch: usize,
    seq: usize,
    probe: Option<(usize, W2vLayerStop)>,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    let (hir, params) = build_wav2vec2_bert_hir_inner(cfg, weights, batch, seq, probe)?;
    let mir = hir.lower_to_mir()?;
    Ok((mir.into_graph(), params))
}

impl<'a> W2vBuilder<'a> {
    pub(crate) fn from_emit_parts(
        hir: &'a mut HirModule,
        params: &'a mut HashMap<String, Vec<f32>>,
        weights: &'a mut dyn WeightSource,
        batch: usize,
        seq: usize,
    ) -> Self {
        Self {
            hir,
            params,
            weights,
            batch,
            seq,
            f: DType::F32,
        }
    }

    pub(crate) fn emit_encoder(
        &mut self,
        input_features: HirNodeId,
        attention_mask: HirNodeId,
        probe: Option<(usize, W2vLayerStop)>,
        cfg: &Wav2Vec2BertConfig,
    ) -> Result<HirNodeId> {
        let h = cfg.hidden_size;
        let feat_dim = cfg.feature_projection_input_dim;
        let batch = self.batch;
        let seq = self.seq;
        let b3 = self.shape3(batch, seq, feat_dim);
        let b3h = self.shape3(batch, seq, h);

        let fp_ln_w =
            self.load_param("feature_projection.layer_norm.weight", &[feat_dim], false)?;
        let fp_ln_b = self.load_param("feature_projection.layer_norm.bias", &[feat_dim], false)?;
        let fp_norm = self.layer_norm(
            input_features,
            fp_ln_w,
            fp_ln_b,
            cfg.layer_norm_eps as f32,
            b3,
        );

        let rows = batch * seq;
        let fp_flat = self.reshape(fp_norm, vec![rows as i64, feat_dim as i64]);
        let fp_proj_w =
            self.load_param("feature_projection.projection.weight", &[h, feat_dim], true)?;
        let fp_proj_b = self.load_param("feature_projection.projection.bias", &[h], false)?;
        let fp_out = self.hir.linear_fused(
            fp_flat,
            fp_proj_w,
            fp_proj_b,
            None,
            Shape::new(&[rows, h], self.f),
        );
        let mut hidden = self.reshape(fp_out, vec![batch as i64, seq as i64, h as i64]);

        hidden = self.apply_padding_mask(hidden, attention_mask, b3h.clone());
        let attn_mask_bias = self.build_extended_attention_mask(attention_mask);

        for layer_idx in 0..cfg.num_hidden_layers {
            let lp = format!("encoder.layers.{layer_idx}");
            hidden = self.encoder_layer(
                cfg,
                layer_idx,
                &lp,
                hidden,
                attn_mask_bias,
                attention_mask,
                probe,
            )?;
        }
        Ok(hidden)
    }

    fn encoder_layer(
        &mut self,
        cfg: &Wav2Vec2BertConfig,
        layer_idx: usize,
        lp: &str,
        x: HirNodeId,
        attn_mask_bias: HirNodeId,
        pad_mask: HirNodeId,
        probe: Option<(usize, W2vLayerStop)>,
    ) -> Result<HirNodeId> {
        let batch = self.batch;
        let seq = self.seq;
        let h = cfg.hidden_size;
        let nh = cfg.num_attention_heads;
        let dh = cfg.head_dim();
        let k_conv = cfg.conv_depthwise_kernel_size;
        let eps = cfg.layer_norm_eps as f32;

        // 1. FFN1 (0.5 residual)
        let residual = x;
        let ffn1_ln_w = self.load_param(&format!("{lp}.ffn1_layer_norm.weight"), &[h], false)?;
        let ffn1_ln_b = self.load_param(&format!("{lp}.ffn1_layer_norm.bias"), &[h], false)?;
        let ffn1_norm = self.layer_norm(x, ffn1_ln_w, ffn1_ln_b, eps, self.shape3(batch, seq, h));
        let ffn1_out =
            self.feed_forward(&format!("{lp}.ffn1"), ffn1_norm, cfg.intermediate_size)?;
        let after_ffn1 = self.scaled_residual(
            &format!("{lp}.ffn1"),
            ffn1_out,
            residual,
            self.shape3(batch, seq, h),
        );
        if probe == Some((layer_idx, W2vLayerStop::AfterFfn1)) {
            return Ok(after_ffn1);
        }

        // 2. Self-attention (relative-key via Op::Attention + bias)
        let residual = after_ffn1;
        let sa_ln_w = self.load_param(&format!("{lp}.self_attn_layer_norm.weight"), &[h], false)?;
        let sa_ln_b = self.load_param(&format!("{lp}.self_attn_layer_norm.bias"), &[h], false)?;
        let sa_norm = self.layer_norm(
            after_ffn1,
            sa_ln_w,
            sa_ln_b,
            eps,
            self.shape3(batch, seq, h),
        );
        let sa_out = self.relative_key_attention(
            cfg,
            &format!("{lp}.self_attn"),
            sa_norm,
            attn_mask_bias,
            nh,
            dh,
        )?;
        let after_attn = self.add(sa_out, residual, self.shape3(batch, seq, h));
        if probe == Some((layer_idx, W2vLayerStop::AfterAttn)) {
            return Ok(after_attn);
        }

        // 3. Conformer conv
        let residual = after_attn;
        let conv_out = self.conv_module(
            &format!("{lp}.conv_module"),
            after_attn,
            pad_mask,
            h,
            k_conv,
            eps,
        )?;
        let after_conv = self.add(conv_out, residual, self.shape3(batch, seq, h));
        if probe == Some((layer_idx, W2vLayerStop::AfterConv)) {
            return Ok(after_conv);
        }

        // 4. FFN2 (0.5 residual) + final LN
        let residual = after_conv;
        let ffn2_ln_w = self.load_param(&format!("{lp}.ffn2_layer_norm.weight"), &[h], false)?;
        let ffn2_ln_b = self.load_param(&format!("{lp}.ffn2_layer_norm.bias"), &[h], false)?;
        let ffn2_norm = self.layer_norm(
            after_conv,
            ffn2_ln_w,
            ffn2_ln_b,
            eps,
            self.shape3(batch, seq, h),
        );
        let ffn2_out =
            self.feed_forward(&format!("{lp}.ffn2"), ffn2_norm, cfg.intermediate_size)?;
        let after_ffn2 = self.scaled_residual(
            &format!("{lp}.ffn2"),
            ffn2_out,
            residual,
            self.shape3(batch, seq, h),
        );
        if probe == Some((layer_idx, W2vLayerStop::AfterFfn2)) {
            return Ok(after_ffn2);
        }

        let final_ln_w = self.load_param(&format!("{lp}.final_layer_norm.weight"), &[h], false)?;
        let final_ln_b = self.load_param(&format!("{lp}.final_layer_norm.bias"), &[h], false)?;
        Ok(self.layer_norm(
            after_ffn2,
            final_ln_w,
            final_ln_b,
            eps,
            self.shape3(batch, seq, h),
        ))
    }

    fn feed_forward(&mut self, prefix: &str, x: HirNodeId, int_dim: usize) -> Result<HirNodeId> {
        let batch = self.batch;
        let seq = self.seq;
        let h = self.hir.node(x).shape.dim(2).unwrap_static();
        let rows = batch * seq;
        let flat = self.reshape(x, vec![rows as i64, h as i64]);
        let w1 = self.load_param(
            &format!("{prefix}.intermediate_dense.weight"),
            &[int_dim, h],
            true,
        )?;
        let b1 = self.load_param(
            &format!("{prefix}.intermediate_dense.bias"),
            &[int_dim],
            false,
        )?;
        let h1 = self.hir.linear_fused(
            flat,
            w1,
            b1,
            Some(Activation::Silu),
            Shape::new(&[rows, int_dim], self.f),
        );
        let w2 = self.load_param(
            &format!("{prefix}.output_dense.weight"),
            &[h, int_dim],
            true,
        )?;
        let b2 = self.load_param(&format!("{prefix}.output_dense.bias"), &[h], false)?;
        let out = self
            .hir
            .linear_fused(h1, w2, b2, None, Shape::new(&[rows, h], self.f));
        Ok(self.reshape(out, vec![batch as i64, seq as i64, h as i64]))
    }

    fn relative_key_attention(
        &mut self,
        cfg: &Wav2Vec2BertConfig,
        prefix: &str,
        x: HirNodeId,
        attn_mask_bias: HirNodeId,
        nh: usize,
        dh: usize,
    ) -> Result<HirNodeId> {
        let batch = self.batch;
        let seq = self.seq;
        let h = nh * dh;
        let rows = batch * seq;
        let b3h = self.shape3(batch, seq, h);
        let flat = self.reshape(x, vec![rows as i64, h as i64]);

        let q_w = self.load_param(&format!("{prefix}.linear_q.weight"), &[h, h], true)?;
        let q_b = self.load_param(&format!("{prefix}.linear_q.bias"), &[h], false)?;
        let k_w = self.load_param(&format!("{prefix}.linear_k.weight"), &[h, h], true)?;
        let k_b = self.load_param(&format!("{prefix}.linear_k.bias"), &[h], false)?;
        let v_w = self.load_param(&format!("{prefix}.linear_v.weight"), &[h, h], true)?;
        let v_b = self.load_param(&format!("{prefix}.linear_v.bias"), &[h], false)?;

        let q = self
            .hir
            .linear_fused(flat, q_w, q_b, None, Shape::new(&[rows, h], self.f));
        let q = self.reshape(q, vec![batch as i64, seq as i64, h as i64]);
        let k = self
            .hir
            .linear_fused(flat, k_w, k_b, None, Shape::new(&[rows, h], self.f));
        let k = self.reshape(k, vec![batch as i64, seq as i64, h as i64]);
        let v = self
            .hir
            .linear_fused(flat, v_w, v_b, None, Shape::new(&[rows, h], self.f));
        let v = self.reshape(v, vec![batch as i64, seq as i64, h as i64]);

        // Relative-key score bias: einsum("bhld,lrd->bhlr", Q, P) / sqrt(d)
        let q_heads = self.reshape_bhsd(q, batch, seq, nh, dh);
        let (dist_w, _) = self
            .weights
            .take(&format!("{prefix}.distance_embedding.weight"), false)?;
        let pos_emb = build_relative_position_table(
            &dist_w,
            seq,
            cfg.left_max_position_embeddings,
            cfg.right_max_position_embeddings,
            dh,
        );
        let pos_name = format!("{prefix}.rel_pos");
        let pos_id = self.register_param(
            &pos_name,
            pos_emb,
            Shape::new(&[1, 1, seq, seq, dh], self.f),
        );
        let scale = 1.0f32 / (dh as f32).sqrt();
        let q_exp = self.reshape(
            q_heads,
            vec![batch as i64, nh as i64, seq as i64, 1, dh as i64],
        );
        let pos_broadcast = self.reshape(pos_id, vec![1, 1, seq as i64, seq as i64, dh as i64]);
        let rel_prod = self.mul(
            q_exp,
            pos_broadcast,
            Shape::new(&[batch, nh, seq, seq, dh], self.f),
        );
        let rel_scores = self.sum(
            rel_prod,
            vec![4],
            Shape::new(&[batch, nh, seq, seq], self.f),
        );
        let rel_scaled = self.mul_scalar(&format!("{prefix}.rel_scale"), rel_scores, scale);

        let total_bias = self.add(
            rel_scaled,
            attn_mask_bias,
            Shape::new(&[batch, nh, seq, seq], self.f),
        );

        let attn = self.hir.attention(
            q,
            k,
            v,
            Some(total_bias),
            nh,
            dh,
            MaskKind::Bias,
            b3h.clone(),
        );

        let out_w = self.load_param(&format!("{prefix}.linear_out.weight"), &[h, h], true)?;
        let out_b = self.load_param(&format!("{prefix}.linear_out.bias"), &[h], false)?;
        let out_flat = self.reshape(attn, vec![rows as i64, h as i64]);
        let out =
            self.hir
                .linear_fused(out_flat, out_w, out_b, None, Shape::new(&[rows, h], self.f));
        Ok(self.reshape(out, vec![batch as i64, seq as i64, h as i64]))
    }

    fn conv_module(
        &mut self,
        prefix: &str,
        x: HirNodeId,
        pad_mask: HirNodeId,
        h: usize,
        k: usize,
        eps: f32,
    ) -> Result<HirNodeId> {
        let batch = self.batch;
        let seq = self.seq;
        let b3h = self.shape3(batch, seq, h);

        let ln_w = self.load_param(&format!("{prefix}.layer_norm.weight"), &[h], false)?;
        let ln_b = self.load_param(&format!("{prefix}.layer_norm.bias"), &[h], false)?;
        let mut hidden = self.layer_norm(x, ln_w, ln_b, eps, b3h.clone());
        hidden = self.apply_padding_mask(hidden, pad_mask, b3h.clone());

        let rows = batch * seq;
        let flat = self.reshape(hidden, vec![rows as i64, h as i64]);
        let (pw1_w_raw, pw1_shape) = self
            .weights
            .take(&format!("{prefix}.pointwise_conv1.weight"), false)?;
        if pw1_shape != [2 * h, h, 1] {
            bail!(
                "conv_module.pointwise_conv1: expected [{}, {}, 1], got {:?}",
                2 * h,
                h,
                pw1_shape
            );
        }
        let pw1_w = self.register_param(
            &format!("{prefix}.pointwise_conv1.weight"),
            transpose_conv1d_pw1(&pw1_w_raw, 2 * h, h),
            Shape::new(&[h, 2 * h], self.f),
        );
        let pw1_b = self.zeros(&format!("{prefix}.pointwise_conv1.bias"), 2 * h)?;
        let pw1 =
            self.hir
                .linear_fused(flat, pw1_w, pw1_b, None, Shape::new(&[rows, 2 * h], self.f));
        let gate = self.narrow(pw1, 1, h, h, Shape::new(&[rows, h], self.f));
        let val = self.narrow(pw1, 1, 0, h, Shape::new(&[rows, h], self.f));
        let gate_sig = self.sigmoid(gate, Shape::new(&[rows, h], self.f));
        let glu = self.mul(val, gate_sig, Shape::new(&[rows, h], self.f));
        let glu_3d = self.reshape(glu, vec![batch as i64, seq as i64, h as i64]);

        let (dw_w_raw, _) = self
            .weights
            .take(&format!("{prefix}.depthwise_conv.weight"), false)?;
        let dw_out = self.depthwise_conv1d_causal(
            &format!("{prefix}.depthwise_conv.weight"),
            &dw_w_raw,
            glu_3d,
            h,
            k,
        )?;

        let dw_ln_w = self.load_param(
            &format!("{prefix}.depthwise_layer_norm.weight"),
            &[h],
            false,
        )?;
        let dw_ln_b =
            self.load_param(&format!("{prefix}.depthwise_layer_norm.bias"), &[h], false)?;
        let dw_norm = self.layer_norm(dw_out, dw_ln_w, dw_ln_b, eps, b3h.clone());
        let dw_act = self.silu(dw_norm, b3h);

        let (pw2_w_raw, pw2_shape) = self
            .weights
            .take(&format!("{prefix}.pointwise_conv2.weight"), false)?;
        if pw2_shape != [h, h, 1] {
            bail!("conv_module.pointwise_conv2: expected [{h}, {h}, 1], got {pw2_shape:?}");
        }
        let pw2_w = self.register_param(
            &format!("{prefix}.pointwise_conv2.weight"),
            transpose_conv1d_pw1(&pw2_w_raw, h, h),
            Shape::new(&[h, h], self.f),
        );
        let flat2 = self.reshape(dw_act, vec![rows as i64, h as i64]);
        let pw2_b = self.zeros(&format!("{prefix}.pointwise_conv2.bias"), h)?;
        let pw2 = self
            .hir
            .linear_fused(flat2, pw2_w, pw2_b, None, Shape::new(&[rows, h], self.f));
        Ok(self.reshape(pw2, vec![batch as i64, seq as i64, h as i64]))
    }

    fn build_extended_attention_mask(&mut self, attention_mask: HirNodeId) -> HirNodeId {
        let batch = self.batch;
        let seq = self.seq;
        let ones = self.register_param(
            "attn_mask.ones",
            vec![1.0f32; batch * seq],
            Shape::new(&[batch, seq], self.f),
        );
        let inv = self.sub(ones, attention_mask, Shape::new(&[batch, seq], self.f));
        let neg = self.mul_scalar("attn_mask.neg_scale", inv, ATTN_MASK_NEG);
        self.reshape(neg, vec![batch as i64, 1, 1, seq as i64])
    }

    fn apply_padding_mask(
        &mut self,
        hidden: HirNodeId,
        attention_mask: HirNodeId,
        out_shape: Shape,
    ) -> HirNodeId {
        let batch = self.batch;
        let seq = self.seq;
        let mask_3d = self.reshape(attention_mask, vec![batch as i64, seq as i64, 1]);
        self.mul(hidden, mask_3d, out_shape)
    }

    fn scaled_residual(
        &mut self,
        prefix: &str,
        x: HirNodeId,
        residual: HirNodeId,
        shape: Shape,
    ) -> HirNodeId {
        let scaled = self.mul_scalar(&format!("{prefix}.res_scale"), x, FFN_RESIDUAL_SCALE);
        self.add(scaled, residual, shape)
    }

    fn depthwise_conv1d_causal(
        &mut self,
        name: &str,
        weight: &[f32],
        input: HirNodeId,
        channels: usize,
        k: usize,
    ) -> Result<HirNodeId> {
        let batch = self.batch;
        let seq = self.seq;
        let b3h = self.shape3(batch, seq, channels);
        let pad_name = format!("{name}.causal_pad");
        let pad_data = vec![0f32; batch * (k - 1) * channels];
        let pad = self.register_param(
            &pad_name,
            pad_data,
            Shape::new(&[batch, k - 1, channels], self.f),
        );
        let w_data = pack_depthwise_conv_weight(weight, k, channels);
        let w = self.register_param(name, w_data, Shape::new(&[channels, 1, 1, k], self.f));
        Ok(self.hir.depthwise_conv1d_causal(input, w, pad, k, b3h))
    }

    // ── HIR MIR helpers ───────────────────────────────────────────

    fn layer_norm(
        &mut self,
        x: HirNodeId,
        gamma: HirNodeId,
        beta: HirNodeId,
        eps: f32,
        shape: Shape,
    ) -> HirNodeId {
        self.hir
            .mir(Op::LayerNorm { axis: -1, eps }, vec![x, gamma, beta], shape)
    }

    fn reshape(&mut self, x: HirNodeId, new_shape: Vec<i64>) -> HirNodeId {
        let shape = self.infer_reshape(&self.hir.node(x).shape, &new_shape);
        self.hir.mir(Op::Reshape { new_shape }, vec![x], shape)
    }

    fn transpose(&mut self, x: HirNodeId, perm: Vec<usize>) -> HirNodeId {
        let shape =
            rlx_ir::shape::transpose_shape(&self.hir.node(x).shape, &perm).expect("transpose");
        self.hir.mir(Op::Transpose { perm }, vec![x], shape)
    }

    fn narrow(
        &mut self,
        x: HirNodeId,
        axis: usize,
        start: usize,
        len: usize,
        shape: Shape,
    ) -> HirNodeId {
        self.hir
            .mir(Op::Narrow { axis, start, len }, vec![x], shape)
    }

    fn add(&mut self, a: HirNodeId, b: HirNodeId, shape: Shape) -> HirNodeId {
        self.hir.mir(Op::Binary(BinaryOp::Add), vec![a, b], shape)
    }

    fn sub(&mut self, a: HirNodeId, b: HirNodeId, shape: Shape) -> HirNodeId {
        self.hir.mir(Op::Binary(BinaryOp::Sub), vec![a, b], shape)
    }

    fn mul(&mut self, a: HirNodeId, b: HirNodeId, shape: Shape) -> HirNodeId {
        self.hir.mir(Op::Binary(BinaryOp::Mul), vec![a, b], shape)
    }

    fn sum(&mut self, x: HirNodeId, axes: Vec<usize>, shape: Shape) -> HirNodeId {
        self.hir.mir(
            Op::Reduce {
                op: rlx_ir::op::ReduceOp::Sum,
                axes,
                keep_dim: false,
            },
            vec![x],
            shape,
        )
    }

    fn sigmoid(&mut self, x: HirNodeId, shape: Shape) -> HirNodeId {
        self.hir
            .mir(Op::Activation(Activation::Sigmoid), vec![x], shape)
    }

    fn silu(&mut self, x: HirNodeId, shape: Shape) -> HirNodeId {
        self.hir
            .mir(Op::Activation(Activation::Silu), vec![x], shape)
    }

    fn mul_scalar(&mut self, name: &str, x: HirNodeId, v: f32) -> HirNodeId {
        let scale = self.register_param(name, vec![v], Shape::new(&[1], self.f));
        let shape = self.hir.node(x).shape.clone();
        self.mul(x, scale, shape)
    }

    fn reshape_bhsd(
        &mut self,
        x: HirNodeId,
        batch: usize,
        seq: usize,
        nh: usize,
        dh: usize,
    ) -> HirNodeId {
        let x4 = self.reshape(x, vec![batch as i64, seq as i64, nh as i64, dh as i64]);
        self.transpose(x4, vec![0, 2, 1, 3])
    }

    fn infer_reshape(&self, input: &Shape, new_shape: &[i64]) -> Shape {
        let static_dims: Vec<usize> = new_shape.iter().map(|&d| d as usize).collect();
        Shape::new(&static_dims, input.dtype())
    }

    fn shape3(&self, batch: usize, seq: usize, h: usize) -> Shape {
        Shape::new(&[batch, seq, h], self.f)
    }

    fn zeros(&mut self, name: &str, n: usize) -> Result<HirNodeId> {
        Ok(self.register_param(name, vec![0f32; n], Shape::new(&[n], self.f)))
    }

    fn load_param(
        &mut self,
        key: &str,
        _expected_shape: &[usize],
        transpose: bool,
    ) -> Result<HirNodeId> {
        let (data, shape) = self.weights.take(key, transpose)?;
        Ok(self.register_param(key, data, Shape::new(&shape, self.f)))
    }

    fn register_param(&mut self, name: &str, data: Vec<f32>, shape: Shape) -> HirNodeId {
        let id = self.hir.param(name, shape);
        self.params.insert(name.to_string(), data);
        id
    }
}

fn validate_cfg(cfg: &Wav2Vec2BertConfig) -> Result<()> {
    if cfg.position_embeddings_type != "relative_key" {
        bail!(
            "wav2vec2_bert: only position_embeddings_type=relative_key is wired (got {})",
            cfg.position_embeddings_type
        );
    }
    if cfg.add_adapter {
        bail!("wav2vec2_bert: add_adapter=true not wired yet");
    }
    if cfg.use_intermediate_ffn_before_adapter {
        bail!("wav2vec2_bert: use_intermediate_ffn_before_adapter=true not wired yet");
    }
    if cfg.hidden_act != "swish" {
        bail!(
            "wav2vec2_bert: only hidden_act=swish is wired (got {})",
            cfg.hidden_act
        );
    }
    Ok(())
}

fn build_relative_position_table(
    dist_emb: &[f32],
    seq: usize,
    left: usize,
    right: usize,
    dh: usize,
) -> Vec<f32> {
    let num_pos = left + right + 1;
    debug_assert_eq!(dist_emb.len(), num_pos * dh);
    let mut out = vec![0f32; seq * seq * dh];
    for l in 0..seq {
        for r in 0..seq {
            let dist = (r as i64 - l as i64).clamp(-(left as i64), right as i64);
            let idx = (dist + left as i64) as usize;
            let dst = (l * seq + r) * dh;
            let src = idx * dh;
            out[dst..dst + dh].copy_from_slice(&dist_emb[src..src + dh]);
        }
    }
    out
}

fn pack_depthwise_conv_weight(weight: &[f32], k: usize, channels: usize) -> Vec<f32> {
    let mut out = vec![0f32; channels * k];
    for c in 0..channels {
        for ki in 0..k {
            out[c * k + ki] = weight[c * k + ki];
        }
    }
    out
}

fn transpose_conv1d_pw1(raw: &[f32], out_c: usize, in_c: usize) -> Vec<f32> {
    let mut out = vec![0f32; in_c * out_c];
    for oc in 0..out_c {
        for ic in 0..in_c {
            out[ic * out_c + oc] = raw[oc * in_c + ic];
        }
    }
    out
}
