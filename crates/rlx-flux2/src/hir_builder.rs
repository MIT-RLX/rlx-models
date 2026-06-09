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

//! Full FLUX.2 transformer HIR builder (dual-stream + single-stream blocks).

use super::config::Flux2Config;
use super::layers::time_guidance_embed;
use super::packed::{Flux2GgufLinearPacked, Flux2PackedParams, Nvfp4LinearPacked};
use super::rope::flux2_pos_embed;
use super::typed_linear::{TypedLinear, TypedLinearStore};
use super::weights::{
    Flux2DualAttnWeights, Flux2FeedForwardWeights, Flux2ModulationWeights, Flux2NormOutWeights,
    Flux2ParallelAttnWeights, Flux2Weights, LinearWeights, RmsNormWeight,
};
use crate::builder::Flux2GraphParams;
use anyhow::Result;
use rlx_ir::hir::{FusionPolicy, HirModule, HirNodeId};
use rlx_ir::op::{Activation, BinaryOp, MaskKind};
use rlx_ir::{DType, Dim, Graph, Op, Shape};

/// Non-f32 parameter blobs (`set_param_typed` at compile time).
pub type Flux2TypedParams = Vec<(String, Vec<u8>, DType)>;

pub struct Flux2ForwardGraph {
    pub hir: HirModule,
    pub params: Flux2GraphParams,
    pub typed_params: Flux2TypedParams,
}

/// Build the full denoiser forward graph in HIR.
///
/// Inputs:
///   - `hidden` `[batch, img_seq, in_channels]`
///   - `encoder` `[batch, txt_seq, joint_attention_dim]`
///   - `temb` `[batch, inner_dim]` — host-side timestep+guidance embedding
///     (see [`super::layers::time_guidance_embed`])
pub fn build_flux2_forward_hir(
    cfg: &Flux2Config,
    weights: &Flux2Weights,
    batch: usize,
    img_seq: usize,
    txt_seq: usize,
    img_ids: &[f32],
    txt_ids: &[f32],
    packed: Option<&Flux2PackedParams>,
    typed_linears: Option<&TypedLinearStore>,
) -> Result<Flux2ForwardGraph> {
    let mut hir = HirModule::new("flux2_forward").with_fusion_policy(FusionPolicy::Direct);
    let mut params = Flux2GraphParams::new();
    let mut typed_params = Flux2TypedParams::new();
    let mut b = Flux2HirBuilder::new(
        &mut hir,
        &mut params,
        &mut typed_params,
        cfg,
        weights,
        batch,
        img_seq,
        txt_seq,
        packed,
        typed_linears,
    );
    b.build_forward(img_ids, txt_ids)?;
    Ok(Flux2ForwardGraph {
        hir,
        params,
        typed_params,
    })
}

pub fn build_flux2_forward_graph(
    cfg: &Flux2Config,
    weights: &Flux2Weights,
    batch: usize,
    img_seq: usize,
    txt_seq: usize,
    img_ids: &[f32],
    txt_ids: &[f32],
) -> Result<(Graph, Flux2GraphParams)> {
    let built = crate::flow::Flux2Flow::new(cfg, weights)
        .batch(batch)
        .img_seq(img_seq)
        .txt_seq(txt_seq)
        .position_ids(img_ids.to_vec(), txt_ids.to_vec())
        .build_forward(img_ids, txt_ids)?;
    let (graph, params) = rlx_core::flow_util::graph_from_built(built.model)?;
    Ok((graph, params))
}

pub fn compile_flux2_forward(
    cfg: &Flux2Config,
    weights: &Flux2Weights,
    batch: usize,
    img_seq: usize,
    txt_seq: usize,
    img_ids: &[f32],
    txt_ids: &[f32],
    device: rlx_runtime::Device,
    packed: Option<&Flux2PackedParams>,
    typed_linears: Option<&TypedLinearStore>,
    aot: Option<&rlx_runtime::AotCache>,
) -> Result<(rlx_runtime::CompiledGraph, Flux2GraphParams)> {
    use crate::compile_util::{compile_hir_cached, flux2_denoiser_aot_key};

    super::device::assert_flux2_device_available(device)?;
    let g = build_flux2_forward_hir(
        cfg,
        weights,
        batch,
        img_seq,
        txt_seq,
        img_ids,
        txt_ids,
        packed,
        typed_linears,
    )?;
    let key = flux2_denoiser_aot_key(
        device,
        batch,
        img_seq,
        txt_seq,
        img_ids,
        txt_ids,
        packed.is_some(),
    );
    let mut compiled = compile_hir_cached(
        device,
        aot,
        &key,
        g.hir,
        &super::compile_util::flux2_compile_profile(),
    )?;
    for (name, data) in &g.params {
        compiled.set_param(name, data);
    }
    for (name, data, dtype) in &g.typed_params {
        compiled.set_param_typed(name, data, *dtype);
    }
    Ok((compiled, g.params))
}

/// Dual-stream section only: embed → mod → dual blocks → img hidden output.
pub fn build_flux2_dual_section_hir(
    cfg: &Flux2Config,
    weights: &Flux2Weights,
    batch: usize,
    img_seq: usize,
    txt_seq: usize,
    img_ids: &[f32],
    txt_ids: &[f32],
) -> Result<Flux2ForwardGraph> {
    let mut hir = HirModule::new("flux2_dual").with_fusion_policy(FusionPolicy::Direct);
    let mut params = Flux2GraphParams::new();
    let mut typed_params = Flux2TypedParams::new();
    let mut b = Flux2HirBuilder::new(
        &mut hir,
        &mut params,
        &mut typed_params,
        cfg,
        weights,
        batch,
        img_seq,
        txt_seq,
        None,
        None,
    );
    let (hidden, _encoder, _cos, _sin, _temb) = b.build_dual_section(img_ids, txt_ids)?;
    hir.outputs = vec![hidden];
    Ok(Flux2ForwardGraph {
        hir,
        params,
        typed_params,
    })
}

pub(crate) struct Flux2HirBuilder<'a> {
    hir: &'a mut HirModule,
    params: &'a mut Flux2GraphParams,
    typed_params: &'a mut Flux2TypedParams,
    weights: &'a Flux2Weights,
    packed: Option<&'a Flux2PackedParams>,
    typed_linears: Option<&'a TypedLinearStore>,
    cfg: &'a Flux2Config,
    batch: usize,
    img_seq: usize,
    txt_seq: usize,
    dim: usize,
    heads: usize,
    head_dim: usize,
    eps: f32,
    rope_dim: usize,
    mlp_hidden: usize,
    f: DType,
}

/// MSA + MLP modulation triples from [`Flux2HirBuilder::modulation_params`].
pub(crate) type Flux2DoubleMod = (
    (HirNodeId, HirNodeId, HirNodeId),
    (HirNodeId, HirNodeId, HirNodeId),
);

impl<'a> Flux2HirBuilder<'a> {
    fn new(
        hir: &'a mut HirModule,
        params: &'a mut Flux2GraphParams,
        typed_params: &'a mut Flux2TypedParams,
        cfg: &'a Flux2Config,
        weights: &'a Flux2Weights,
        batch: usize,
        img_seq: usize,
        txt_seq: usize,
        packed: Option<&'a Flux2PackedParams>,
        typed_linears: Option<&'a TypedLinearStore>,
    ) -> Self {
        let dim = cfg.inner_dim();
        Self {
            hir,
            params,
            typed_params,
            weights,
            packed,
            typed_linears,
            cfg,
            batch,
            img_seq,
            txt_seq,
            dim,
            heads: cfg.num_attention_heads,
            head_dim: cfg.attention_head_dim,
            eps: cfg.eps as f32,
            rope_dim: cfg.axes_dims_rope.iter().sum(),
            mlp_hidden: cfg.ff_inner_dim(),
            f: DType::F32,
        }
    }

    pub(crate) fn from_emit_parts(
        hir: &'a mut HirModule,
        params: &'a mut Flux2GraphParams,
        typed_params: &'a mut Flux2TypedParams,
        cfg: &'a Flux2Config,
        weights: &'a Flux2Weights,
        batch: usize,
        img_seq: usize,
        txt_seq: usize,
    ) -> Self {
        Self::new(
            hir,
            params,
            typed_params,
            cfg,
            weights,
            batch,
            img_seq,
            txt_seq,
            None,
            None,
        )
    }

    fn build_dual_section(
        &mut self,
        img_ids: &[f32],
        txt_ids: &[f32],
    ) -> Result<(HirNodeId, HirNodeId, HirNodeId, HirNodeId, HirNodeId)> {
        let hidden_in = self.hir.input(
            "hidden",
            Shape::new(&[self.batch, self.img_seq, self.cfg.in_channels], self.f),
        );
        let enc_in = self.hir.input(
            "encoder",
            Shape::new(
                &[self.batch, self.txt_seq, self.cfg.joint_attention_dim],
                self.f,
            ),
        );
        let temb_in = self.hir.input("temb", self.b1());

        let mod_img = self.modulation_params(&self.weights.double_mod_img, "mod_img", temb_in)?;
        let mod_txt = self.modulation_params(&self.weights.double_mod_txt, "mod_txt", temb_in)?;

        let mut hidden = self.linear(
            hidden_in,
            &self.weights.x_embedder,
            "x_embedder",
            self.b3i(),
        )?;
        let mut encoder = self.linear(
            enc_in,
            &self.weights.context_embedder,
            "context_embedder",
            self.b3t(),
        )?;

        let (cos_id, sin_id) = self.rope_params(img_ids, txt_ids)?;

        for (li, block) in self.weights.transformer_blocks.iter().enumerate() {
            (hidden, encoder) = self.emit_dual_stream_block(
                li, block, hidden, encoder, &mod_img, &mod_txt, cos_id, sin_id,
            )?;
        }

        Ok((hidden, encoder, cos_id, sin_id, temb_in))
    }

    fn build_forward(&mut self, img_ids: &[f32], txt_ids: &[f32]) -> Result<()> {
        let (hidden, encoder, cos_id, sin_id, temb_in) =
            self.build_dual_section(img_ids, txt_ids)?;
        let out = self.emit_single_stream_tail(hidden, encoder, cos_id, sin_id, temb_in)?;
        self.hir.outputs = vec![out];
        Ok(())
    }

    /// Concat img/txt streams → single-stream blocks → ada-norm → proj_out.
    pub(crate) fn emit_single_stream_tail(
        &mut self,
        hidden: HirNodeId,
        encoder: HirNodeId,
        cos_id: HirNodeId,
        sin_id: HirNodeId,
        temb_in: HirNodeId,
    ) -> Result<HirNodeId> {
        let single_mod =
            self.single_modulation_params(&self.weights.single_mod, "mod_single", temb_in)?;

        let stream = self.concat(
            vec![encoder, hidden],
            1,
            self.b3(self.txt_seq + self.img_seq),
        );
        let mut stream = stream;
        for (li, block) in self.weights.single_transformer_blocks.iter().enumerate() {
            let lp = format!("sblk{li}");
            let n = self.layer_norm_no_affine(
                stream,
                self.b3(self.txt_seq + self.img_seq),
                &format!("{lp}.n"),
            )?;
            let n = self.modulate(n, single_mod.0, single_mod.1, self.txt_seq + self.img_seq);
            let attn =
                self.parallel_attention(&block.attn, &format!("{lp}.attn"), n, cos_id, sin_id)?;
            let attn_g = self.gate(attn, single_mod.2, self.txt_seq + self.img_seq);
            stream = self.add(stream, attn_g, self.b3(self.txt_seq + self.img_seq));
        }

        let hidden_out = self.narrow(stream, 1, self.txt_seq, self.img_seq, self.b3i());
        let normed = self.ada_norm_out(hidden_out, temb_in, &self.weights.norm_out)?;
        self.linear(normed, &self.weights.proj_out, "proj_out", self.b3o())
    }

    /// One FLUX dual-stream transformer block (img + txt).
    pub(crate) fn emit_dual_stream_block(
        &mut self,
        li: usize,
        block: &super::weights::Flux2DoubleBlockWeights,
        hidden: HirNodeId,
        encoder: HirNodeId,
        mod_img: &Flux2DoubleMod,
        mod_txt: &Flux2DoubleMod,
        cos_id: HirNodeId,
        sin_id: HirNodeId,
    ) -> Result<(HirNodeId, HirNodeId)> {
        let lp = format!("blk{li}");
        let (img_msa, img_mlp) = mod_img;
        let (txt_msa, txt_mlp) = mod_txt;

        let n1 = self.layer_norm_no_affine(hidden, self.b3i(), &format!("{lp}.n1"))?;
        let n1 = self.modulate(n1, img_msa.0, img_msa.1, self.img_seq);
        let nc = self.layer_norm_no_affine(encoder, self.b3t(), &format!("{lp}.nc"))?;
        let nc = self.modulate(nc, txt_msa.0, txt_msa.1, self.txt_seq);

        let (enc_a, img_a) =
            self.dual_attention(&block.attn, &format!("{lp}.attn"), n1, nc, cos_id, sin_id)?;
        let img_g = self.gate(img_a, img_msa.2, self.img_seq);
        let hidden = self.add(hidden, img_g, self.b3i());
        let txt_g = self.gate(enc_a, txt_msa.2, self.txt_seq);
        let encoder = self.add(encoder, txt_g, self.b3t());

        let n2 = self.layer_norm_no_affine(hidden, self.b3i(), &format!("{lp}.n2"))?;
        let n2 = self.modulate_scale_shift(n2, img_mlp.1, img_mlp.0, self.img_seq);
        let ff = self.feed_forward(&block.ff, &format!("{lp}.ff"), n2, self.img_seq)?;
        let ff_g = self.gate(ff, img_mlp.2, self.img_seq);
        let hidden = self.add(hidden, ff_g, self.b3i());

        let nc2 = self.layer_norm_no_affine(encoder, self.b3t(), &format!("{lp}.nc2"))?;
        let nc2 = self.modulate_scale_shift(nc2, txt_mlp.1, txt_mlp.0, self.txt_seq);
        let ffc = self.feed_forward(&block.ff_context, &format!("{lp}.ffc"), nc2, self.txt_seq)?;
        let ffc_g = self.gate(ffc, txt_mlp.2, self.txt_seq);
        let encoder = self.add(encoder, ffc_g, self.b3t());
        Ok((hidden, encoder))
    }

    fn b1(&self) -> Shape {
        Shape::new(&[self.batch, self.dim], self.f)
    }
    fn b3i(&self) -> Shape {
        self.b3(self.img_seq)
    }
    fn b3t(&self) -> Shape {
        self.b3(self.txt_seq)
    }
    fn b3o(&self) -> Shape {
        Shape::new(&[self.batch, self.img_seq, self.cfg.proj_out_dim()], self.f)
    }
    fn b3(&self, seq: usize) -> Shape {
        Shape::new(&[self.batch, seq, self.dim], self.f)
    }

    fn register_param(&mut self, name: &str, data: Vec<f32>, shape: Shape) -> HirNodeId {
        let id = self.hir.param(name, shape);
        self.params.insert(name.to_string(), data);
        id
    }

    pub(crate) fn linear(
        &mut self,
        x: HirNodeId,
        lw: &LinearWeights,
        name: &str,
        out_shape: Shape,
    ) -> Result<HirNodeId> {
        if let Some(p) = self.packed.and_then(|m| m.get_nvfp4(name)) {
            return self.linear_nvfp4(x, p, name, out_shape);
        }
        if let Some(p) = self.packed.and_then(|m| m.get_gguf(name)) {
            return self.linear_gguf(x, p, name, out_shape);
        }
        if let Some(tl) = self.typed_linears.and_then(|t| t.get(name)) {
            return self.linear_typed(x, tl, name, out_shape);
        }
        let w = self.register_param(
            &format!("{name}.weight"),
            lw.w_t.clone(),
            Shape::new(&[lw.in_dim, lw.out_dim], self.f),
        );
        let bias = if lw.bias.iter().all(|&v| v == 0.0) {
            None
        } else {
            let b = self.register_param(
                &format!("{name}.bias"),
                lw.bias.clone(),
                Shape::new(&[lw.out_dim], self.f),
            );
            Some(b)
        };
        Ok(self.hir.linear(x, w, bias, None, out_shape))
    }

    fn linear_typed(
        &mut self,
        x: HirNodeId,
        tl: &TypedLinear,
        name: &str,
        out_shape: Shape,
    ) -> Result<HirNodeId> {
        let w = self.register_typed_param_shaped(
            &format!("{name}.weight"),
            tl.weight_bytes.clone(),
            tl.dtype,
            Shape::new(&[tl.in_dim, tl.out_dim], tl.dtype),
        );
        let bias = if tl.bias.iter().all(|&v| v == 0.0) {
            None
        } else {
            let b = self.register_param(
                &format!("{name}.bias"),
                tl.bias.clone(),
                Shape::new(&[tl.out_dim], self.f),
            );
            Some(b)
        };
        Ok(self.hir.linear(x, w, bias, None, out_shape))
    }

    fn linear_nvfp4(
        &mut self,
        x: HirNodeId,
        p: &Nvfp4LinearPacked,
        name: &str,
        out_shape: Shape,
    ) -> Result<HirNodeId> {
        use rlx_ir::QuantScheme;

        let w_name = format!("{name}.weight");
        let s_name = format!("{name}.scale");
        let gs_name = format!("{name}.global_scale");
        let w = self.register_typed_param(&w_name, p.w_q.clone(), DType::U8);
        let scale = self.register_typed_param(&s_name, p.scale.clone(), DType::U8);
        let gs = self.register_param(&gs_name, vec![p.global_scale], Shape::scalar(self.f));
        let mut y = self.hir.dequant_matmul(
            x,
            w,
            Some(scale),
            Some(gs),
            QuantScheme::Nvfp4Block,
            out_shape.clone(),
        );
        if p.bias.iter().any(|&v| v != 0.0) {
            let b = self.register_param(
                &format!("{name}.bias"),
                p.bias.clone(),
                Shape::new(&[p.out_dim], self.f),
            );
            y = self
                .hir
                .mir(Op::Binary(BinaryOp::Add), vec![y, b], out_shape);
        }
        Ok(y)
    }

    fn linear_gguf(
        &mut self,
        x: HirNodeId,
        p: &Flux2GgufLinearPacked,
        name: &str,
        out_shape: Shape,
    ) -> Result<HirNodeId> {
        let w_name = format!("{name}.weight");
        let w = self.register_typed_param(&w_name, p.w_q.clone(), DType::U8);
        let mut y = self
            .hir
            .dequant_matmul(x, w, None, None, p.scheme, out_shape.clone());
        if p.bias.iter().any(|&v| v != 0.0) {
            let b = self.register_param(
                &format!("{name}.bias"),
                p.bias.clone(),
                Shape::new(&[p.out_dim], self.f),
            );
            y = self
                .hir
                .mir(Op::Binary(BinaryOp::Add), vec![y, b], out_shape);
        }
        Ok(y)
    }

    fn register_typed_param(&mut self, name: &str, data: Vec<u8>, dtype: DType) -> HirNodeId {
        let shape = Shape::new(&[data.len()], dtype);
        let id = self.hir.param(name, shape);
        self.typed_params.push((name.to_string(), data, dtype));
        id
    }

    fn register_typed_param_shaped(
        &mut self,
        name: &str,
        data: Vec<u8>,
        dtype: DType,
        shape: Shape,
    ) -> HirNodeId {
        let id = self.hir.param(name, shape);
        self.typed_params.push((name.to_string(), data, dtype));
        id
    }

    fn layer_norm_no_affine(&mut self, x: HirNodeId, shape: Shape, tag: &str) -> Result<HirNodeId> {
        let d = self.dim;
        let g = self.register_param(
            &format!("{tag}.ln1"),
            vec![1.0f32; d],
            Shape::new(&[d], self.f),
        );
        let b = self.register_param(
            &format!("{tag}.ln0"),
            vec![0.0f32; d],
            Shape::new(&[d], self.f),
        );
        Ok(self.hir.mir(
            Op::LayerNorm {
                axis: -1,
                eps: self.eps,
            },
            vec![x, g, b],
            shape,
        ))
    }

    pub(crate) fn modulation_params(
        &mut self,
        m: &Flux2ModulationWeights,
        tag: &str,
        temb: HirNodeId,
    ) -> Result<Flux2DoubleMod> {
        let h = self
            .hir
            .mir(Op::Activation(Activation::Silu), vec![temb], self.b1());
        let mod_shape = Shape::new(&[self.batch, 6 * self.dim], self.f);
        let mod_out = self.linear(h, &m.linear, tag, mod_shape)?;
        let last = self.hir.node(mod_out).shape.rank() - 1;
        let d = self.dim;
        let b1 = self.b1();
        let s0 = self.narrow(mod_out, last, 0, d, b1.clone());
        let s1 = self.narrow(mod_out, last, d, d, b1.clone());
        let s2 = self.narrow(mod_out, last, 2 * d, d, b1.clone());
        let s3 = self.narrow(mod_out, last, 3 * d, d, b1.clone());
        let s4 = self.narrow(mod_out, last, 4 * d, d, b1.clone());
        let s5 = self.narrow(mod_out, last, 5 * d, d, b1);
        Ok(((s0, s1, s2), (s3, s4, s5)))
    }

    fn single_modulation_params(
        &mut self,
        m: &Flux2ModulationWeights,
        tag: &str,
        temb: HirNodeId,
    ) -> Result<(HirNodeId, HirNodeId, HirNodeId)> {
        let h = self
            .hir
            .mir(Op::Activation(Activation::Silu), vec![temb], self.b1());
        let mod_shape = Shape::new(&[self.batch, 3 * self.dim], self.f);
        let mod_out = self.linear(h, &m.linear, tag, mod_shape)?;
        let last = self.hir.node(mod_out).shape.rank() - 1;
        let d = self.dim;
        let b1 = self.b1();
        let s0 = self.narrow(mod_out, last, 0, d, b1.clone());
        let s1 = self.narrow(mod_out, last, d, d, b1.clone());
        let s2 = self.narrow(mod_out, last, 2 * d, d, b1);
        Ok((s0, s1, s2))
    }

    fn broadcast_bd(&mut self, v: HirNodeId, seq: usize) -> HirNodeId {
        let b1d = self.reshape(v, vec![self.batch as i64, 1, self.dim as i64]);
        self.mir_expand(b1d, vec![self.batch as i64, seq as i64, self.dim as i64])
    }

    fn modulate(
        &mut self,
        x: HirNodeId,
        shift: HirNodeId,
        scale: HirNodeId,
        seq: usize,
    ) -> HirNodeId {
        let shape = self.b3(seq);
        let shift_b = self.broadcast_bd(shift, seq);
        let scale_b = self.broadcast_bd(scale, seq);
        let ones = self.ones3(seq);
        let scaled_base = self.add(ones, scale_b, shape.clone());
        let scaled = self.mul(x, scaled_base, shape.clone());
        self.add(scaled, shift_b, shape)
    }

    fn modulate_scale_shift(
        &mut self,
        x: HirNodeId,
        scale: HirNodeId,
        shift: HirNodeId,
        seq: usize,
    ) -> HirNodeId {
        let shape = self.b3(seq);
        let shift_b = self.broadcast_bd(shift, seq);
        let scale_b = self.broadcast_bd(scale, seq);
        let ones = self.ones3(seq);
        let scaled_base = self.add(ones, scale_b, shape.clone());
        let scaled = self.mul(x, scaled_base, shape.clone());
        self.add(scaled, shift_b, shape)
    }

    fn gate(&mut self, x: HirNodeId, gate: HirNodeId, seq: usize) -> HirNodeId {
        let g = self.broadcast_bd(gate, seq);
        self.mul(x, g, self.b3(seq))
    }

    fn feed_forward(
        &mut self,
        ff: &Flux2FeedForwardWeights,
        tag: &str,
        x: HirNodeId,
        seq: usize,
    ) -> Result<HirNodeId> {
        let rows = self.batch * seq;
        let inner = ff.linear_in.out_dim / 2;
        let flat = self.reshape(x, vec![rows as i64, self.dim as i64]);
        let h = self.linear(
            flat,
            &ff.linear_in,
            &format!("{tag}.in"),
            Shape::new(&[rows, ff.linear_in.out_dim], self.f),
        )?;
        let h3 = self.reshape(
            h,
            vec![self.batch as i64, seq as i64, ff.linear_in.out_dim as i64],
        );
        let act = self.hir.mir(
            Op::FusedSwiGLU {
                cast_to: None,
                gate_first: true,
            },
            vec![h3],
            self.b3(seq).with_dim(2, Dim::Static(inner)),
        );
        let act_flat = self.reshape(act, vec![rows as i64, inner as i64]);
        self.linear(
            act_flat,
            &ff.linear_out,
            &format!("{tag}.out"),
            Shape::new(&[rows, self.dim], self.f),
        )
        .map(|o| self.reshape(o, vec![self.batch as i64, seq as i64, self.dim as i64]))
    }

    fn rms_gamma(&mut self, rms: &RmsNormWeight, name: &str) -> HirNodeId {
        let mut g = vec![0.0f32; self.dim];
        for h in 0..self.heads {
            g[h * self.head_dim..(h + 1) * self.head_dim].copy_from_slice(&rms.scale);
        }
        self.register_param(name, g, Shape::new(&[self.dim], self.f))
    }

    fn rms_norm(&mut self, x: HirNodeId, gamma: HirNodeId, shape: Shape) -> HirNodeId {
        let beta = self.register_param(
            &format!("rmsb_{}", self.params.len()),
            vec![0.0f32; self.dim],
            Shape::new(&[self.dim], self.f),
        );
        self.hir.mir(
            Op::RmsNorm {
                axis: -1,
                eps: 1e-6,
            },
            vec![x, gamma, beta],
            shape,
        )
    }

    fn linear_rms(
        &mut self,
        x: HirNodeId,
        lw: &LinearWeights,
        rms: &RmsNormWeight,
        name: &str,
        shape: Shape,
    ) -> Result<HirNodeId> {
        let h = self.linear(x, lw, name, shape.clone())?;
        let g = self.rms_gamma(rms, &format!("{name}.rms"));
        Ok(self.rms_norm(h, g, shape))
    }

    fn dual_attention(
        &mut self,
        attn: &Flux2DualAttnWeights,
        tag: &str,
        hidden: HirNodeId,
        encoder: HirNodeId,
        cos: HirNodeId,
        sin: HirNodeId,
    ) -> Result<(HirNodeId, HirNodeId)> {
        let total = self.txt_seq + self.img_seq;
        let b3i = self.b3i();
        let b3t = self.b3t();
        let q_i = self.linear_rms(
            hidden,
            &attn.to_q,
            &attn.norm_q,
            &format!("{tag}.q"),
            b3i.clone(),
        )?;
        let k_i = self.linear_rms(
            hidden,
            &attn.to_k,
            &attn.norm_k,
            &format!("{tag}.k"),
            b3i.clone(),
        )?;
        let v_i = self.linear(hidden, &attn.to_v, &format!("{tag}.v"), b3i)?;
        let q_t = self.linear_rms(
            encoder,
            &attn.add_q,
            &attn.norm_added_q,
            &format!("{tag}.aq"),
            b3t.clone(),
        )?;
        let k_t = self.linear_rms(
            encoder,
            &attn.add_k,
            &attn.norm_added_k,
            &format!("{tag}.ak"),
            b3t.clone(),
        )?;
        let v_t = self.linear(encoder, &attn.add_v, &format!("{tag}.av"), b3t)?;

        let q = self.concat(vec![q_t, q_i], 1, self.b3(total));
        let k = self.concat(vec![k_t, k_i], 1, self.b3(total));
        let v = self.concat(vec![v_t, v_i], 1, self.b3(total));

        let q = self.rope(q, cos, sin, self.b3(total));
        let k = self.rope(k, cos, sin, self.b3(total));

        let out = self.hir.attention(
            q,
            k,
            v,
            None,
            self.heads,
            self.head_dim,
            MaskKind::None,
            self.b3(total),
        );

        let txt_out = self.narrow(out, 1, 0, self.txt_seq, self.b3t());
        let img_out = self.narrow(out, 1, self.txt_seq, self.img_seq, self.b3i());
        let enc_proj = self.linear(txt_out, &attn.to_add_out, &format!("{tag}.ao"), self.b3t())?;
        let img_proj = self.linear(img_out, &attn.to_out, &format!("{tag}.o"), self.b3i())?;
        Ok((enc_proj, img_proj))
    }

    fn parallel_attention(
        &mut self,
        attn: &Flux2ParallelAttnWeights,
        tag: &str,
        x: HirNodeId,
        cos: HirNodeId,
        sin: HirNodeId,
    ) -> Result<HirNodeId> {
        let seq = self.txt_seq + self.img_seq;
        let rows = self.batch * seq;
        let flat = self.reshape(x, vec![rows as i64, self.dim as i64]);
        let fused = self.linear(
            flat,
            &attn.to_qkv_mlp,
            &format!("{tag}.fused"),
            Shape::new(&[rows, attn.to_qkv_mlp.out_dim], self.f),
        )?;
        let fused3 = self.reshape(
            fused,
            vec![
                self.batch as i64,
                seq as i64,
                attn.to_qkv_mlp.out_dim as i64,
            ],
        );
        let last = 2;
        let b3s = self.b3(seq);
        let q = self.narrow(fused3, last, 0, self.dim, b3s.clone());
        let k = self.narrow(fused3, last, self.dim, self.dim, b3s.clone());
        let v = self.narrow(fused3, last, 2 * self.dim, self.dim, b3s.clone());
        let mlp = self.narrow(
            fused3,
            last,
            3 * self.dim,
            2 * self.mlp_hidden,
            Shape::new(&[self.batch, seq, 2 * self.mlp_hidden], self.f),
        );

        let nq = self.rms_gamma(&attn.norm_q, &format!("{tag}.nq"));
        let nk = self.rms_gamma(&attn.norm_k, &format!("{tag}.nk"));
        let q = self.rms_norm(q, nq, b3s.clone());
        let k = self.rms_norm(k, nk, b3s.clone());
        let q = self.rope(q, cos, sin, self.b3(seq));
        let k = self.rope(k, cos, sin, self.b3(seq));
        let attn_out = self.hir.attention(
            q,
            k,
            v,
            None,
            self.heads,
            self.head_dim,
            MaskKind::None,
            self.b3(seq),
        );

        let mlp_act = self.hir.mir(
            Op::FusedSwiGLU {
                cast_to: None,
                gate_first: true,
            },
            vec![mlp],
            self.b3(seq).with_dim(2, Dim::Static(self.mlp_hidden)),
        );
        let cat = self.concat(
            vec![attn_out, mlp_act],
            2,
            Shape::new(&[self.batch, seq, self.dim + self.mlp_hidden], self.f),
        );
        let cat_flat = self.reshape(cat, vec![rows as i64, (self.dim + self.mlp_hidden) as i64]);
        let out = self.linear(
            cat_flat,
            &attn.to_out,
            &format!("{tag}.out"),
            Shape::new(&[rows, self.dim], self.f),
        )?;
        Ok(self.reshape(out, vec![self.batch as i64, seq as i64, self.dim as i64]))
    }

    fn ada_norm_out(
        &mut self,
        x: HirNodeId,
        temb: HirNodeId,
        norm: &Flux2NormOutWeights,
    ) -> Result<HirNodeId> {
        let h = self
            .hir
            .mir(Op::Activation(Activation::Silu), vec![temb], self.b1());
        let emb = self.linear(
            h,
            &norm.linear,
            "norm_out",
            Shape::new(&[self.batch, 2 * self.dim], self.f),
        )?;
        let last = self.hir.node(emb).shape.rank() - 1;
        let b1 = self.b1();
        let scale = self.narrow(emb, last, 0, self.dim, b1.clone());
        let shift = self.narrow(emb, last, self.dim, self.dim, b1);
        let n = self.layer_norm_no_affine(x, self.b3i(), "norm_out_ln")?;
        let b3i = self.b3i();
        let scale_b = self.broadcast_bd(scale, self.img_seq);
        let shift_b = self.broadcast_bd(shift, self.img_seq);
        let ones = self.ones3(self.img_seq);
        let scaled_base = self.add(ones, scale_b, b3i.clone());
        let scaled = self.mul(n, scaled_base, b3i.clone());
        Ok(self.add(scaled, shift_b, b3i))
    }

    pub(crate) fn rope_params(
        &mut self,
        img_ids: &[f32],
        txt_ids: &[f32],
    ) -> Result<(HirNodeId, HirNodeId)> {
        let n_axes = 4usize;
        let total = self.txt_seq + self.img_seq;
        let mut ids = vec![0.0f32; total * n_axes];
        for t in 0..self.txt_seq {
            for a in 0..n_axes {
                ids[t * n_axes + a] = txt_ids[t * n_axes + a];
            }
        }
        for t in 0..self.img_seq {
            for a in 0..n_axes {
                ids[(self.txt_seq + t) * n_axes + a] = img_ids[t * n_axes + a];
            }
        }
        let (cos, sin) = flux2_pos_embed(self.cfg, &ids, total, n_axes);
        let cos_id =
            self.register_param("rope_cos", cos, Shape::new(&[total, self.rope_dim], self.f));
        let sin_id =
            self.register_param("rope_sin", sin, Shape::new(&[total, self.rope_dim], self.f));
        Ok((cos_id, sin_id))
    }

    fn rope(&mut self, x: HirNodeId, cos: HirNodeId, sin: HirNodeId, shape: Shape) -> HirNodeId {
        self.hir.mir(
            Op::Rope {
                head_dim: self.head_dim,
                n_rot: self.rope_dim.min(self.head_dim),
            },
            vec![x, cos, sin],
            shape,
        )
    }

    #[allow(dead_code)]
    fn ones1(&mut self) -> HirNodeId {
        self.register_param(
            &format!("ones1_{}", self.params.len()),
            vec![1.0f32; self.dim],
            Shape::new(&[self.dim], self.f),
        )
    }

    fn ones3(&mut self, seq: usize) -> HirNodeId {
        let id = self.register_param(
            &format!("ones3_{}", self.params.len()),
            vec![1.0f32; self.dim],
            Shape::new(&[1, 1, self.dim], self.f),
        );
        self.mir_expand(id, vec![self.batch as i64, seq as i64, self.dim as i64])
    }

    fn reshape(&mut self, x: HirNodeId, new_shape: Vec<i64>) -> HirNodeId {
        let shape = self.infer_reshape(&self.hir.node(x).shape, &new_shape);
        self.hir.mir(Op::Reshape { new_shape }, vec![x], shape)
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

    fn add(&mut self, a: HirNodeId, b: HirNodeId, shape: Shape) -> HirNodeId {
        self.hir.mir(Op::Binary(BinaryOp::Add), vec![a, b], shape)
    }

    fn mul(&mut self, a: HirNodeId, b: HirNodeId, shape: Shape) -> HirNodeId {
        self.hir.mir(Op::Binary(BinaryOp::Mul), vec![a, b], shape)
    }

    fn mir_expand(&mut self, x: HirNodeId, target: Vec<i64>) -> HirNodeId {
        let shape = self.infer_reshape(&self.hir.node(x).shape, &target);
        self.hir.mir(
            Op::Expand {
                target_shape: target,
            },
            vec![x],
            shape,
        )
    }

    fn infer_reshape(&self, input: &Shape, new_shape: &[i64]) -> Shape {
        let static_dims: Vec<usize> = new_shape.iter().map(|&d| d as usize).collect();
        Shape::new(&static_dims, input.dtype())
    }
}

/// Host-side temb for compiled forward (timestep × 1000, optional guidance × 1000).
pub fn host_temb(
    weights: &Flux2Weights,
    cfg: &Flux2Config,
    timestep: &[f32],
    guidance: Option<&[f32]>,
) -> Result<Vec<f32>> {
    let t_scaled: Vec<f32> = timestep.iter().map(|t| t * 1000.0).collect();
    let g_scaled = guidance.map(|g| g.iter().map(|x| x * 1000.0).collect::<Vec<_>>());
    time_guidance_embed(
        &t_scaled,
        g_scaled.as_deref(),
        &weights.time_guidance,
        cfg.inner_dim(),
    )
}

/// Dual-time temb for flow-map forwards: mean(embed(t), embed(t′)).
pub fn host_temb_dual(
    weights: &Flux2Weights,
    cfg: &Flux2Config,
    timestep: &[f32],
    timestep_target: &[f32],
    guidance: Option<&[f32]>,
) -> Result<Vec<f32>> {
    let t_scaled: Vec<f32> = timestep.iter().map(|t| t * 1000.0).collect();
    let t2_scaled: Vec<f32> = timestep_target.iter().map(|t| t * 1000.0).collect();
    let g_scaled = guidance.map(|g| g.iter().map(|x| x * 1000.0).collect::<Vec<_>>());
    let tg_tgt = weights
        .time_guidance_target
        .as_ref()
        .unwrap_or(&weights.time_guidance);
    crate::layers::time_guidance_embed_dual(
        &t_scaled,
        &t2_scaled,
        g_scaled.as_deref(),
        &weights.time_guidance,
        tg_tgt,
        cfg.inner_dim(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Flux2Config, Flux2ForwardInput, extract_flux2_weights, flux2_transformer_forward,
        prepare_weight_map, synthetic_weights,
    };

    #[test]
    fn nvfp4_x_embedder_lowers() {
        use crate::synthetic_flux2_packed_tiny;

        let cfg = Flux2Config::tiny();
        let wm = synthetic_weights(&cfg);
        let w = extract_flux2_weights(prepare_weight_map(wm), &cfg).unwrap();
        let packed = synthetic_flux2_packed_tiny(&cfg);
        let g = build_flux2_forward_hir(
            &cfg,
            &w,
            1,
            4,
            3,
            &[0.0; 16],
            &[0.0; 12],
            Some(&packed),
            None,
        )
        .unwrap();
        assert!(!g.typed_params.is_empty());
        g.hir.lower_to_mir().expect("lower nvfp4");
    }

    #[test]
    fn forward_hir_lowers() {
        let cfg = Flux2Config::tiny();
        let wm = synthetic_weights(&cfg);
        let w = extract_flux2_weights(prepare_weight_map(wm), &cfg).unwrap();
        let g =
            build_flux2_forward_hir(&cfg, &w, 1, 4, 3, &[0.0; 16], &[0.0; 12], None, None).unwrap();
        assert_eq!(g.hir.outputs.len(), 1);
        g.hir.lower_to_mir().expect("lower");
    }

    #[test]
    fn compiled_forward_matches_native() {
        let cfg = Flux2Config::tiny();
        let wm = synthetic_weights(&cfg);
        let w = extract_flux2_weights(prepare_weight_map(wm), &cfg).unwrap();
        let b = 1usize;
        let img_seq = 4usize;
        let txt_seq = 3usize;
        let hidden = (0..b * img_seq * cfg.in_channels)
            .map(|i| (i as f32 * 0.01).sin())
            .collect::<Vec<_>>();
        let encoder = (0..b * txt_seq * cfg.joint_attention_dim)
            .map(|i| (i as f32 * 0.02).cos())
            .collect::<Vec<_>>();
        let timestep = vec![0.5f32];
        let guidance = vec![3.5f32];
        let img_ids = vec![0.0f32; img_seq * 4];
        let txt_ids = vec![0.0f32; txt_seq * 4];

        let native = flux2_transformer_forward(
            &w,
            &cfg,
            Flux2ForwardInput {
                hidden_states: &hidden,
                encoder_hidden_states: &encoder,
                timestep: &timestep,
                timestep_target: None,
                guidance: Some(&guidance),
                img_ids: &img_ids,
                txt_ids: &txt_ids,
                batch: b,
                img_seq,
                txt_seq,
            },
        )
        .unwrap();

        let temb = host_temb(&w, &cfg, &timestep, Some(&guidance)).unwrap();
        let (mut compiled, _) = compile_flux2_forward(
            &cfg,
            &w,
            b,
            img_seq,
            txt_seq,
            &img_ids,
            &txt_ids,
            rlx_runtime::Device::Cpu,
            None,
            None,
            None,
        )
        .unwrap();
        let out = compiled
            .run(&[
                ("hidden", hidden.as_slice()),
                ("encoder", encoder.as_slice()),
                ("temb", temb.as_slice()),
            ])
            .remove(0);

        assert_eq!(out.len(), native.len());
        let max_diff = native
            .iter()
            .zip(&out)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 2e-2, "HIR vs native max_abs_diff={max_diff}");
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn compiled_forward_matches_native_on_cuda() {
        use rlx_runtime::Device;

        if !rlx_runtime::is_available(Device::Cuda) {
            eprintln!("skip: CUDA not available");
            return;
        }
        let cfg = Flux2Config::tiny();
        let wm = synthetic_weights(&cfg);
        let w = extract_flux2_weights(prepare_weight_map(wm), &cfg).unwrap();
        let b = 1usize;
        let img_seq = 4usize;
        let txt_seq = 3usize;
        let hidden = (0..b * img_seq * cfg.in_channels)
            .map(|i| (i as f32 * 0.01).sin())
            .collect::<Vec<_>>();
        let encoder = (0..b * txt_seq * cfg.joint_attention_dim)
            .map(|i| (i as f32 * 0.02).cos())
            .collect::<Vec<_>>();
        let timestep = vec![0.5f32];
        let guidance = vec![3.5f32];
        let img_ids = vec![0.0f32; img_seq * 4];
        let txt_ids = vec![0.0f32; txt_seq * 4];

        let native = flux2_transformer_forward(
            &w,
            &cfg,
            Flux2ForwardInput {
                hidden_states: &hidden,
                encoder_hidden_states: &encoder,
                timestep: &timestep,
                timestep_target: None,
                guidance: Some(&guidance),
                img_ids: &img_ids,
                txt_ids: &txt_ids,
                batch: b,
                img_seq,
                txt_seq,
            },
        )
        .unwrap();

        let temb = host_temb(&w, &cfg, &timestep, Some(&guidance)).unwrap();
        let (mut compiled, _) = compile_flux2_forward(
            &cfg,
            &w,
            b,
            img_seq,
            txt_seq,
            &img_ids,
            &txt_ids,
            Device::Cuda,
            None,
            None,
            None,
        )
        .unwrap();
        let out = compiled
            .run(&[
                ("hidden", hidden.as_slice()),
                ("encoder", encoder.as_slice()),
                ("temb", temb.as_slice()),
            ])
            .remove(0);

        assert_eq!(out.len(), native.len());
        let max_diff = native
            .iter()
            .zip(&out)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 2e-2,
            "CUDA HIR vs native max_abs_diff={max_diff}"
        );
    }
}
