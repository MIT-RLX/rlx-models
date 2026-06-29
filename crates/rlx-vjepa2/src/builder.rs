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

//! HIR-first graph builders for the V-JEPA2 encoder, predictor, and pooler.
//!
//! Builders emit [`HirModule`] with fused linear/attention blocks and lower to
//! legacy [`Graph`] (MIR) for `Session::compile` / `Session::compile_hir`.

use super::config::Vjepa2Config;
use super::predictor::Vjepa2PredictorLayout;
use super::preprocess::Vjepa2PatchEmbedWeights;
use super::rope::build_vjepa2_rope_tables;
use super::weights::{
    Vjepa2BlockWeights, Vjepa2EncoderWeights, Vjepa2PoolerCrossWeights,
    Vjepa2PoolerSelfBlockWeights, Vjepa2PoolerWeights, Vjepa2PredictorWeights,
};
use anyhow::Result;
use rlx_ir::hir::{FusionPolicy, HirModule, HirNodeId};
use rlx_ir::op::{Activation, BinaryOp, MaskKind};
use rlx_ir::{DType, Graph, Op, Shape};
use std::collections::HashMap;

/// Host-side patch-embed weights used with the compiled graph.
pub struct Vjepa2GraphPreprocess {
    pub patch: Vjepa2PatchEmbedWeights,
}

/// F32 params returned by graph builders (including gather indices stored as f32).
pub struct Vjepa2GraphParams {
    pub f32: HashMap<String, Vec<f32>>,
}

impl Vjepa2GraphParams {
    pub fn from_f32(map: HashMap<String, Vec<f32>>) -> Self {
        Self { f32: map }
    }

    pub fn load(&self, compiled: &mut rlx_runtime::CompiledGraph) {
        for (name, data) in &self.f32 {
            compiled.set_param(name, data);
        }
    }
}

#[allow(dead_code)]
fn lower_hir(hir: HirModule) -> Result<Graph> {
    Ok(hir
        .lower_to_mir()
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .into_graph())
}

struct VjepaBuilder {
    hir: HirModule,
    params: HashMap<String, Vec<f32>>,
    f: DType,
}

impl VjepaBuilder {
    fn new(name: &str) -> Self {
        Self {
            hir: HirModule::new(name).with_fusion_policy(FusionPolicy::Direct),
            params: HashMap::new(),
            f: DType::F32,
        }
    }

    #[allow(dead_code)]
    fn finish(self) -> Result<Graph> {
        lower_hir(self.hir)
    }

    fn shape3(&self, batch: usize, seq: usize, h: usize) -> Shape {
        Shape::new(&[batch, seq, h], self.f)
    }

    fn node_shape(&self, id: HirNodeId) -> Shape {
        self.hir.node(id).shape.clone()
    }

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
        let in_shape = self.hir.node(x).shape.clone();
        let static_dims: Vec<usize> = new_shape.iter().map(|&d| d as usize).collect();
        let out = Shape::new(&static_dims, in_shape.dtype());
        self.hir.mir(Op::Reshape { new_shape }, vec![x], out)
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

    fn concat(&mut self, inputs: Vec<HirNodeId>, axis: usize, shape: Shape) -> HirNodeId {
        self.hir.mir(Op::Concat { axis }, inputs, shape)
    }

    fn gather(&mut self, table: HirNodeId, indices: HirNodeId, axis: usize) -> HirNodeId {
        let out = rlx_ir::shape::gather_shape(
            &self.hir.node(table).shape,
            &self.hir.node(indices).shape,
            axis,
        )
        .expect("gather shape");
        self.hir.mir(Op::Gather { axis }, vec![table, indices], out)
    }

    fn add(&mut self, a: HirNodeId, b: HirNodeId, shape: Shape) -> HirNodeId {
        self.hir.mir(Op::Binary(BinaryOp::Add), vec![a, b], shape)
    }

    fn mm(&mut self, lhs: HirNodeId, rhs: HirNodeId) -> HirNodeId {
        let out = rlx_ir::shape::matmul_shape(&self.hir.node(lhs).shape, &self.hir.node(rhs).shape)
            .expect("matmul shape");
        self.hir.mir(Op::MatMul, vec![lhs, rhs], out)
    }

    fn rope_n(
        &mut self,
        x: HirNodeId,
        cos: HirNodeId,
        sin: HirNodeId,
        head_dim: usize,
        n_rot: usize,
    ) -> HirNodeId {
        let shape = self.hir.node(x).shape.clone();
        self.hir.mir(
            Op::Rope {
                head_dim,
                n_rot,
                style: rlx_ir::op::RopeStyle::NeoX,
            },
            vec![x, cos, sin],
            shape,
        )
    }

    #[allow(dead_code)]
    fn gelu_approx(&mut self, x: HirNodeId, shape: Shape) -> HirNodeId {
        self.hir
            .mir(Op::Activation(Activation::GeluApprox), vec![x], shape)
    }

    fn attention_custom(
        &mut self,
        q: HirNodeId,
        k: HirNodeId,
        v: HirNodeId,
        mask: HirNodeId,
        nh: usize,
        dh: usize,
    ) -> HirNodeId {
        let out = rlx_ir::shape::attention_shape(&self.hir.node(q).shape);
        self.hir
            .attention(q, k, v, Some(mask), nh, dh, MaskKind::Custom, out)
    }

    fn attention_none(
        &mut self,
        q: HirNodeId,
        k: HirNodeId,
        v: HirNodeId,
        nh: usize,
        dh: usize,
    ) -> HirNodeId {
        let out = rlx_ir::shape::attention_shape(&self.hir.node(q).shape);
        self.hir
            .attention(q, k, v, None, nh, dh, MaskKind::None, out)
    }

    fn bind_vec(&mut self, name: &str, data: &[f32]) -> HirNodeId {
        let id = self.hir.param(name, Shape::new(&[data.len()], self.f));
        self.params.insert(name.to_string(), data.to_vec());
        id
    }

    fn bind_mat(&mut self, name: &str, w_t: &[f32], in_dim: usize, out_dim: usize) -> HirNodeId {
        let id = self.hir.param(name, Shape::new(&[in_dim, out_dim], self.f));
        self.params.insert(name.to_string(), w_t.to_vec());
        id
    }

    fn bind_indices(&mut self, name: &str, data: &[i64], shape: &[usize]) -> HirNodeId {
        let f32_data: Vec<f32> = data.iter().map(|&v| v as f32).collect();
        let id = self.hir.param(name, Shape::new(shape, self.f));
        self.params.insert(name.to_string(), f32_data);
        id
    }

    fn linear_named(
        &mut self,
        name: &str,
        input: HirNodeId,
        in_dim: usize,
        w_t: &[f32],
        b: &[f32],
    ) -> HirNodeId {
        let out_dim = b.len();
        let w = self.bind_mat(&format!("{name}.weight"), w_t, in_dim, out_dim);
        let bias = self.bind_vec(&format!("{name}.bias"), b);
        let out_shape =
            rlx_ir::shape::matmul_shape(&self.hir.node(input).shape, &self.hir.node(w).shape)
                .expect("linear matmul shape");
        self.hir.linear_fused(input, w, bias, None, out_shape)
    }

    fn mlp_block(
        &mut self,
        lp: &str,
        x: HirNodeId,
        embed: usize,
        fc1_w_t: &[f32],
        fc1_b: &[f32],
        fc2_w_t: &[f32],
        fc2_b: &[f32],
        residual: HirNodeId,
        out_shape: Shape,
    ) -> HirNodeId {
        let hidden = fc1_b.len();
        let fc1_w = self.bind_mat(&format!("{lp}.mlp.fc1.weight"), fc1_w_t, embed, hidden);
        let fc1_bias = self.bind_vec(&format!("{lp}.mlp.fc1.bias"), fc1_b);
        let fc1_shape =
            rlx_ir::shape::matmul_shape(&self.hir.node(x).shape, &self.hir.node(fc1_w).shape)
                .expect("fc1 shape");
        let up = self
            .hir
            .linear_fused(x, fc1_w, fc1_bias, Some(Activation::GeluApprox), fc1_shape);

        let fc2_w = self.bind_mat(&format!("{lp}.mlp.fc2.weight"), fc2_w_t, hidden, embed);
        let fc2_bias = self.bind_vec(&format!("{lp}.mlp.fc2.bias"), fc2_b);
        let fc2_shape =
            rlx_ir::shape::matmul_shape(&self.hir.node(up).shape, &self.hir.node(fc2_w).shape)
                .expect("fc2 shape");
        let ffn = self.hir.linear_fused(up, fc2_w, fc2_bias, None, fc2_shape);
        self.add(residual, ffn, out_shape)
    }
}

/// Build the V-JEPA2 encoder HIR module from extracted weights.
pub fn build_vjepa2_encoder_hir_sized(
    cfg: &Vjepa2Config,
    enc: &Vjepa2EncoderWeights,
    batch: usize,
) -> Result<(HirModule, HashMap<String, Vec<f32>>, Vjepa2GraphPreprocess)> {
    let mut b = VjepaBuilder::new("vjepa2_encoder");

    let h = cfg.hidden_size;
    let nh = cfg.num_attention_heads;
    let dh = cfg.head_dim();
    let eps = cfg.layer_norm_eps as f32;
    let seq = cfg.num_patches();
    let (d_dim, hd_dim, w_dim) = cfg.rope_segment_dims();
    let grid_h = cfg.grid_spatial();
    let grid_w = cfg.grid_spatial();
    let n_rot = d_dim + hd_dim + w_dim;

    let preprocess = Vjepa2GraphPreprocess {
        patch: enc.patch.clone(),
    };

    let (cos_data, sin_data) =
        build_vjepa2_rope_tables(seq, dh, d_dim, hd_dim, w_dim, grid_h, grid_w);
    let half = dh / 2;
    let cos_id = b.bind_mat("rope_cos", &cos_data, seq, half);
    let sin_id = b.bind_mat("rope_sin", &sin_data, seq, half);

    let mask_data = vec![1.0f32; batch * seq];
    let mask_id = b.hir.param("attn_mask", Shape::new(&[batch, seq], b.f));
    b.params.insert("attn_mask".into(), mask_data);

    let hidden_input = b.hir.input("hidden", b.shape3(batch, seq, h));
    let mut x = hidden_input;
    let enc_shape = b.shape3(batch, seq, h);

    for (layer_idx, block) in enc.blocks.iter().enumerate() {
        let lp = format!("blocks.{layer_idx}");
        x = append_rope_block(
            &mut b,
            x,
            block,
            &lp,
            h,
            nh,
            dh,
            n_rot,
            cos_id,
            sin_id,
            Some(mask_id),
            eps,
            true,
            enc_shape.clone(),
        );
    }

    let fn_g = b.bind_vec("norm.weight", &enc.norm_w);
    let fn_b = b.bind_vec("norm.bias", &enc.norm_b);
    let encoded = b.layer_norm(x, fn_g, fn_b, eps, enc_shape);
    b.hir.outputs = vec![encoded];

    Ok((b.hir, b.params, preprocess))
}

/// Build the V-JEPA2 encoder IR graph from extracted weights (via [`super::flow::Vjepa2EncoderFlow`]).
pub fn build_vjepa2_encoder_graph_sized(
    cfg: &Vjepa2Config,
    enc: &Vjepa2EncoderWeights,
    batch: usize,
) -> Result<(Graph, HashMap<String, Vec<f32>>, Vjepa2GraphPreprocess)> {
    let built = super::flow::Vjepa2EncoderFlow::new(cfg, enc, batch).build()?;
    let (graph, params) = rlx_core::flow_util::graph_from_built(built.model)?;
    Ok((graph, params, built.preprocess))
}

/// Build the JEPA predictor HIR module for fixed context/target masks.
pub fn build_vjepa2_predictor_hir_sized(
    cfg: &Vjepa2Config,
    pred: &Vjepa2PredictorWeights,
    layout: &Vjepa2PredictorLayout,
    mask_rows: &[f32],
    batch: usize,
) -> Result<(HirModule, Vjepa2GraphParams)> {
    let mut b = VjepaBuilder::new("vjepa2_predictor");

    let enc = cfg.hidden_size;
    let pred_h = cfg.pred_hidden_size;
    let nh = cfg.pred_num_attention_heads;
    let dh = cfg.pred_head_dim();
    let eps = cfg.layer_norm_eps as f32;
    let enc_seq = cfg.num_patches();
    let (d_dim, hd_dim, w_dim) = cfg.pred_rope_segment_dims();
    let n_rot = d_dim + hd_dim + w_dim;
    let n_ctxt = layout.n_ctxt;
    let n_tgt = layout.n_tgt;
    let n_combined = layout.n_combined;
    let half = dh / 2;

    let encoder = b.hir.input("encoder", b.shape3(batch, enc_seq, enc));

    let ctxt_idx_id = b.bind_indices("ctxt_idx", &layout.ctxt_idx, &[batch, n_ctxt]);
    let ctxt = b.gather(encoder, ctxt_idx_id, 1);
    let ctxt = b.reshape(ctxt, vec![batch as i64, n_ctxt as i64, enc as i64]);

    let embed_w = b.bind_mat("embed.weight", &pred.embed_w_t, enc, pred_h);
    let embed_b = b.bind_vec("embed.bias", &pred.embed_b);
    let mm_embed = b.mm(ctxt, embed_w);
    let ctxt_up = b.add(mm_embed, embed_b, b.shape3(batch, n_ctxt, pred_h));
    let ctxt_embed = b.reshape(ctxt_up, vec![batch as i64, n_ctxt as i64, pred_h as i64]);

    let mask_id = b
        .hir
        .param("mask_rows", Shape::new(&[batch, n_tgt, pred_h], b.f));
    b.params.insert("mask_rows".into(), mask_rows.to_vec());
    let mut x = b.concat(
        vec![ctxt_embed, mask_id],
        1,
        b.shape3(batch, n_combined, pred_h),
    );
    x = b.reshape(x, vec![batch as i64, n_combined as i64, pred_h as i64]);

    let sort_idx_id = b.bind_indices("sort_idx", &layout.sort_idx, &[batch, n_combined]);
    x = b.gather(x, sort_idx_id, 1);
    x = b.reshape(x, vec![batch as i64, n_combined as i64, pred_h as i64]);

    let cos_id = b.bind_mat("rope_cos", &layout.rope_cos, n_combined, half);
    let sin_id = b.bind_mat("rope_sin", &layout.rope_sin, n_combined, half);
    let pred_shape = b.shape3(batch, n_combined, pred_h);

    for (layer_idx, block) in pred.blocks.iter().enumerate() {
        let lp = format!("blocks.{layer_idx}");
        x = append_rope_block(
            &mut b,
            x,
            block,
            &lp,
            pred_h,
            nh,
            dh,
            n_rot,
            cos_id,
            sin_id,
            None,
            eps,
            false,
            pred_shape.clone(),
        );
    }

    let fn_g = b.bind_vec("norm.weight", &pred.norm_w);
    let fn_b = b.bind_vec("norm.bias", &pred.norm_b);
    x = b.layer_norm(x, fn_g, fn_b, eps, pred_shape.clone());

    let unsort_idx_id = b.bind_indices("unsort_idx", &layout.unsort_idx, &[batch, n_combined]);
    x = b.gather(x, unsort_idx_id, 1);
    x = b.reshape(x, vec![batch as i64, n_combined as i64, pred_h as i64]);
    x = b.narrow(x, 1, n_ctxt, n_tgt, b.shape3(batch, n_tgt, pred_h));
    x = b.reshape(x, vec![batch as i64, n_tgt as i64, pred_h as i64]);

    let proj_w = b.bind_mat("proj.weight", &pred.proj_w_t, pred_h, enc);
    let proj_b = b.bind_vec("proj.bias", &pred.proj_b);
    let mm_proj = b.mm(x, proj_w);
    let out = b.add(mm_proj, proj_b, b.shape3(batch, n_tgt, enc));
    b.hir.outputs = vec![out];

    Ok((b.hir, Vjepa2GraphParams { f32: b.params }))
}

/// Build the JEPA predictor IR graph for fixed context/target masks (via [`super::flow::Vjepa2PredictorFlow`]).
pub fn build_vjepa2_predictor_graph_sized(
    cfg: &Vjepa2Config,
    pred: &Vjepa2PredictorWeights,
    layout: &Vjepa2PredictorLayout,
    mask_rows: &[f32],
    batch: usize,
) -> Result<(Graph, Vjepa2GraphParams)> {
    let built =
        super::flow::Vjepa2PredictorFlow::new(cfg, pred, layout, mask_rows, batch).build()?;
    let (graph, params) = rlx_core::flow_util::graph_from_built(built)?;
    Ok((graph, Vjepa2GraphParams { f32: params }))
}

/// Build the attentive pooler HIR module (+ optional classifier head).
pub fn build_vjepa2_pooler_hir_sized(
    cfg: &Vjepa2Config,
    pooler: &Vjepa2PoolerWeights,
    batch: usize,
) -> Result<(HirModule, Vjepa2GraphParams)> {
    let mut b = VjepaBuilder::new("vjepa2_pooler");

    let e = cfg.hidden_size;
    let nh = cfg.num_attention_heads;
    let dh = cfg.head_dim();
    let hidden = cfg.pooler_intermediate_size();
    let eps = cfg.layer_norm_eps as f32;
    let seq = cfg.num_patches();

    let encoder = b.hir.input("encoder", b.shape3(batch, seq, e));
    let mut ctx = encoder;
    let ctx_shape = b.shape3(batch, seq, e);

    for (layer_idx, block) in pooler.self_blocks.iter().enumerate() {
        let lp = format!("self.{layer_idx}");
        ctx = append_pooler_self_block(
            &mut b,
            ctx,
            block,
            &lp,
            e,
            nh,
            dh,
            hidden,
            eps,
            ctx_shape.clone(),
        );
    }

    let mut query_data = Vec::with_capacity(batch * e);
    for _ in 0..batch {
        query_data.extend_from_slice(&pooler.query_tokens);
    }
    let query_id = b.bind_vec("query_tokens", &query_data);
    let mut queries = b.reshape(query_id, vec![batch as i64, 1, e as i64]);
    let query_shape = b.shape3(batch, 1, e);

    queries = append_pooler_cross_block(
        &mut b,
        queries,
        ctx,
        &pooler.cross,
        "cross",
        e,
        nh,
        dh,
        hidden,
        eps,
        query_shape.clone(),
    );

    queries = b.narrow(queries, 1, 0, 1, query_shape.clone());
    let embedding = b.reshape(queries, vec![batch as i64, e as i64]);

    let mut outputs = vec![embedding];
    if let (Some(w_t), Some(bias)) = (&pooler.classifier_w_t, &pooler.classifier_b) {
        let nc = bias.len();
        let cls_w = b.bind_mat("classifier.weight", w_t, e, nc);
        let cls_b = b.bind_vec("classifier.bias", bias);
        let mm = b.mm(embedding, cls_w);
        let logits = b.add(mm, cls_b, Shape::new(&[batch, nc], b.f));
        outputs.push(logits);
    }
    b.hir.outputs = outputs;

    Ok((b.hir, Vjepa2GraphParams { f32: b.params }))
}

/// Build the attentive pooler IR graph (+ optional classifier head) (via [`super::flow::Vjepa2PoolerFlow`]).
pub fn build_vjepa2_pooler_graph_sized(
    cfg: &Vjepa2Config,
    pooler: &Vjepa2PoolerWeights,
    batch: usize,
) -> Result<(Graph, Vjepa2GraphParams)> {
    let built = super::flow::Vjepa2PoolerFlow::new(cfg, pooler, batch).build()?;
    let (graph, params) = rlx_core::flow_util::graph_from_built(built)?;
    Ok((graph, Vjepa2GraphParams { f32: params }))
}

/// Compile encoder HIR on the given device (HIR → MIR → LIR).
pub fn compile_vjepa2_encoder(
    cfg: &Vjepa2Config,
    enc: &Vjepa2EncoderWeights,
    batch: usize,
    device: rlx_runtime::Device,
) -> Result<(
    rlx_runtime::CompiledGraph,
    HashMap<String, Vec<f32>>,
    Vjepa2GraphPreprocess,
)> {
    use rlx_runtime::Session;

    let (hir, params, preprocess) = build_vjepa2_encoder_hir_sized(cfg, enc, batch)?;
    let opts = rlx_core::flow_bridge::compile_options_for_profile(
        &rlx_flow::CompileProfile::encoder(),
        device,
    );
    let mut compiled = Session::new(device).compile_hir_with(hir, &opts)?;
    for (name, data) in &params {
        compiled.set_param(name, data);
    }
    Ok((compiled, params, preprocess))
}

#[allow(clippy::too_many_arguments)]
fn append_rope_block(
    b: &mut VjepaBuilder,
    x: HirNodeId,
    block: &Vjepa2BlockWeights,
    lp: &str,
    embed: usize,
    nh: usize,
    dh: usize,
    n_rot: usize,
    cos_id: HirNodeId,
    sin_id: HirNodeId,
    mask_id: Option<HirNodeId>,
    eps: f32,
    use_mask: bool,
    block_shape: Shape,
) -> HirNodeId {
    let n1_g = b.bind_vec(&format!("{lp}.norm1.weight"), &block.norm1_w);
    let n1_b = b.bind_vec(&format!("{lp}.norm1.bias"), &block.norm1_b);
    let normed1 = b.layer_norm(x, n1_g, n1_b, eps, block_shape.clone());

    let q = b.linear_named(
        &format!("{lp}.attn.q"),
        normed1,
        embed,
        &block.q_w_t,
        &block.q_b,
    );
    let k = b.linear_named(
        &format!("{lp}.attn.k"),
        normed1,
        embed,
        &block.k_w_t,
        &block.k_b,
    );
    let v = b.linear_named(
        &format!("{lp}.attn.v"),
        normed1,
        embed,
        &block.v_w_t,
        &block.v_b,
    );

    let q_rot = b.rope_n(q, cos_id, sin_id, dh, n_rot);
    let k_rot = b.rope_n(k, cos_id, sin_id, dh, n_rot);
    let attn = if use_mask {
        let mask = mask_id.expect("rope block with use_mask requires attn mask");
        b.attention_custom(q_rot, k_rot, v, mask, nh, dh)
    } else {
        b.attention_none(q_rot, k_rot, v, nh, dh)
    };

    let p_w = b.bind_mat(
        &format!("{lp}.attn.proj.weight"),
        &block.proj_w_t,
        embed,
        embed,
    );
    let p_b = b.bind_vec(&format!("{lp}.attn.proj.bias"), &block.proj_b);
    let mm_proj = b.mm(attn, p_w);
    let proj = b.add(mm_proj, p_b, block_shape.clone());
    let x = b.add(x, proj, block_shape.clone());

    let n2_g = b.bind_vec(&format!("{lp}.norm2.weight"), &block.norm2_w);
    let n2_b = b.bind_vec(&format!("{lp}.norm2.bias"), &block.norm2_b);
    let normed2 = b.layer_norm(x, n2_g, n2_b, eps, block_shape.clone());

    b.mlp_block(
        lp,
        normed2,
        embed,
        &block.mlp_fc1_w_t,
        &block.mlp_fc1_b,
        &block.mlp_fc2_w_t,
        &block.mlp_fc2_b,
        x,
        block_shape,
    )
}

#[allow(clippy::too_many_arguments)]
fn append_pooler_self_block(
    b: &mut VjepaBuilder,
    x: HirNodeId,
    block: &Vjepa2PoolerSelfBlockWeights,
    lp: &str,
    embed: usize,
    nh: usize,
    dh: usize,
    _hidden: usize,
    eps: f32,
    block_shape: Shape,
) -> HirNodeId {
    let n1_g = b.bind_vec(&format!("{lp}.norm1.weight"), &block.norm1_w);
    let n1_b = b.bind_vec(&format!("{lp}.norm1.bias"), &block.norm1_b);
    let normed1 = b.layer_norm(x, n1_g, n1_b, eps, block_shape.clone());

    let q = b.linear_named(&format!("{lp}.q"), normed1, embed, &block.q_w_t, &block.q_b);
    let k = b.linear_named(&format!("{lp}.k"), normed1, embed, &block.k_w_t, &block.k_b);
    let v = b.linear_named(&format!("{lp}.v"), normed1, embed, &block.v_w_t, &block.v_b);
    let attn = b.attention_none(q, k, v, nh, dh);

    let out_w = b.bind_mat(&format!("{lp}.out.weight"), &block.out_w_t, embed, embed);
    let out_b = b.bind_vec(&format!("{lp}.out.bias"), &block.out_b);
    let mm_out = b.mm(attn, out_w);
    let proj = b.add(mm_out, out_b, block_shape.clone());
    let x = b.add(x, proj, block_shape.clone());

    let n2_g = b.bind_vec(&format!("{lp}.norm2.weight"), &block.norm2_w);
    let n2_b = b.bind_vec(&format!("{lp}.norm2.bias"), &block.norm2_b);
    let normed2 = b.layer_norm(x, n2_g, n2_b, eps, block_shape.clone());

    b.mlp_block(
        lp,
        normed2,
        embed,
        &block.mlp_fc1_w_t,
        &block.mlp_fc1_b,
        &block.mlp_fc2_w_t,
        &block.mlp_fc2_b,
        x,
        block_shape,
    )
}

#[allow(clippy::too_many_arguments)]
fn append_pooler_cross_block(
    b: &mut VjepaBuilder,
    queries: HirNodeId,
    context: HirNodeId,
    block: &Vjepa2PoolerCrossWeights,
    lp: &str,
    embed: usize,
    nh: usize,
    dh: usize,
    _hidden: usize,
    eps: f32,
    query_shape: Shape,
) -> HirNodeId {
    let ctx_shape = b.node_shape(context);
    let residual = queries;

    let n1_g = b.bind_vec(&format!("{lp}.norm1.weight"), &block.norm1_w);
    let n1_b = b.bind_vec(&format!("{lp}.norm1.bias"), &block.norm1_b);
    let ctx_norm = b.layer_norm(context, n1_g, n1_b, eps, ctx_shape);

    let q = b.linear_named(&format!("{lp}.q"), queries, embed, &block.q_w_t, &block.q_b);
    let k = b.linear_named(
        &format!("{lp}.k"),
        ctx_norm,
        embed,
        &block.k_w_t,
        &block.k_b,
    );
    let v = b.linear_named(
        &format!("{lp}.v"),
        ctx_norm,
        embed,
        &block.v_w_t,
        &block.v_b,
    );
    let attn = b.attention_none(q, k, v, nh, dh);
    let queries = b.add(residual, attn, query_shape.clone());

    let n2_g = b.bind_vec(&format!("{lp}.norm2.weight"), &block.norm2_w);
    let n2_b = b.bind_vec(&format!("{lp}.norm2.bias"), &block.norm2_b);
    let normed2 = b.layer_norm(queries, n2_g, n2_b, eps, query_shape.clone());

    b.mlp_block(
        lp,
        normed2,
        embed,
        &block.mlp_fc1_w_t,
        &block.mlp_fc1_b,
        &block.mlp_fc2_w_t,
        &block.mlp_fc2_b,
        queries,
        query_shape,
    )
}
