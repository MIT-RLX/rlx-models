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

//! HIR-native SAM3 detector encoder (6 transformer layers).

use super::detector_encoder::{Sam3EncoderLayerWeights, Sam3EncoderWeights};
use super::packed_gguf::packed_linear;
use anyhow::{Result, ensure};
use rlx_flow::CompileProfile;
use rlx_flow::{GgufPackedLinear, GgufPackedParams};
use rlx_ir::hir::{HirGraphExt, HirModule, HirMut, HirNodeId};
use rlx_ir::op::{MaskKind, Op};
use rlx_ir::shape;
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{CompiledGraph, Device};
use std::collections::HashMap;

/// HIR build product (F32 norms + optional U8 packed linears).
pub struct Sam3EncoderHirParts {
    pub hir: HirModule,
    pub params: HashMap<String, Vec<f32>>,
    pub typed_params: Vec<(String, Vec<u8>, DType)>,
}

/// Compiled encoder graph + uploaded parameters, ready for many `run`s.
pub struct Sam3CompiledEncoder {
    pub compiled: CompiledGraph,
    pub batch: usize,
    pub hw: usize,
    pub seq: usize,
    pub d: usize,
}

impl Sam3CompiledEncoder {
    pub fn new(
        weights: &Sam3EncoderWeights,
        batch: usize,
        hw: usize,
        seq: usize,
        device: Device,
    ) -> Result<Self> {
        Self::new_with_profile(weights, batch, hw, seq, device, &CompileProfile::sam3())
    }

    pub fn new_with_profile(
        weights: &Sam3EncoderWeights,
        batch: usize,
        hw: usize,
        seq: usize,
        device: Device,
        profile: &CompileProfile,
    ) -> Result<Self> {
        Self::new_with_profile_and_gguf(weights, batch, hw, seq, device, profile, None)
    }

    pub fn new_with_profile_and_gguf(
        weights: &Sam3EncoderWeights,
        batch: usize,
        hw: usize,
        seq: usize,
        device: Device,
        profile: &CompileProfile,
        gguf_packed: Option<&GgufPackedParams>,
    ) -> Result<Self> {
        let parts = build_encoder_hir(weights, batch, hw, seq, gguf_packed)?;
        let mut compiled =
            rlx_core::flow_bridge::compile_hir_with_profile(device, parts.hir, profile)?;
        rlx_core::flow_util::attach_built_params(&mut compiled, parts.params, &parts.typed_params);
        Ok(Self {
            compiled,
            batch,
            hw,
            seq,
            d: D_MODEL,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &mut self,
        src_bchw: &[f32],
        src_pos_bchw: &[f32],
        prompt_seq_first: &[f32],
        prompt_kpm: &[u8],
        src_h: usize,
        src_w: usize,
    ) -> Result<Vec<f32>> {
        let hw = src_h * src_w;
        ensure!(
            hw == self.hw,
            "compiled encoder expects hw={}, got {hw}",
            self.hw
        );
        let mut src_bhwc = vec![0f32; self.batch * hw * self.d];
        let mut pos_bhwc = vec![0f32; self.batch * hw * self.d];
        for b in 0..self.batch {
            for s in 0..hw {
                for c in 0..self.d {
                    src_bhwc[(b * hw + s) * self.d + c] = src_bchw[((b * self.d + c) * hw) + s];
                    pos_bhwc[(b * hw + s) * self.d + c] = src_pos_bchw[((b * self.d + c) * hw) + s];
                }
            }
        }
        let mut prompt_bf = vec![0f32; self.batch * self.seq * self.d];
        for b in 0..self.batch {
            for l in 0..self.seq {
                let s = (l * self.batch + b) * self.d;
                let dst = (b * self.seq + l) * self.d;
                prompt_bf[dst..dst + self.d].copy_from_slice(&prompt_seq_first[s..s + self.d]);
            }
        }
        let prompt_kpm_inv: Vec<f32> = prompt_kpm
            .iter()
            .map(|&v| if v == 0 { 1.0 } else { 0.0 })
            .collect();
        let outputs = self.compiled.run(&[
            ("src", src_bhwc.as_slice()),
            ("src_pos", pos_bhwc.as_slice()),
            ("prompt", prompt_bf.as_slice()),
            ("prompt_kpm_inv", prompt_kpm_inv.as_slice()),
        ]);
        outputs
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("encoder graph produced no outputs"))
    }
}

const D_MODEL: usize = 256;
const DIM_FF: usize = 2048;
const N_HEADS: usize = 8;
const HEAD_DIM: usize = D_MODEL / N_HEADS;

fn enc_layer_key(base: &str, li: usize, suffix: &str) -> String {
    format!("{base}.layers.{li}.{suffix}")
}

fn gguf_weight_param(
    g: &mut HirMut<'_>,
    typed: &mut Vec<(String, Vec<u8>, DType)>,
    cache: &mut HashMap<String, HirNodeId>,
    ir_name: &str,
    p: &GgufPackedLinear,
) -> HirNodeId {
    if let Some(&id) = cache.get(ir_name) {
        return id;
    }
    let id = g.param(ir_name, Shape::new(&[p.w_q.len()], DType::U8));
    typed.push((ir_name.to_string(), p.w_q.clone(), DType::U8));
    cache.insert(ir_name.to_string(), id);
    id
}

fn linear_gguf_matmul(
    g: &mut HirMut<'_>,
    typed: &mut Vec<(String, Vec<u8>, DType)>,
    cache: &mut HashMap<String, HirNodeId>,
    ir_stem: &str,
    p: &GgufPackedLinear,
    input: HirNodeId,
    in_dim: usize,
    out_dim: usize,
) -> Result<HirNodeId> {
    ensure!(
        p.in_dim == in_dim && p.out_dim == out_dim,
        "packed linear {ir_stem}: shape {}x{} vs {in_dim}x{out_dim}",
        p.in_dim,
        p.out_dim
    );
    let w_name = format!("{ir_stem}.w");
    let w_id = gguf_weight_param(g, typed, cache, &w_name, p);
    let cur = g.shape(input);
    let mut dims: Vec<usize> = cur.dims().iter().map(|d| d.unwrap_static()).collect();
    *dims.last_mut().unwrap() = out_dim;
    let out_shape = Shape::new(&dims, DType::F32);
    Ok(g.add_node(
        Op::DequantMatMul { scheme: p.scheme },
        vec![input, w_id],
        out_shape,
    ))
}

fn add_f32_bias(
    g: &mut HirMut<'_>,
    params: &mut HashMap<String, Vec<f32>>,
    name: &str,
    input: HirNodeId,
    bias: &[f32],
) -> HirNodeId {
    if bias.iter().all(|&v| v == 0.0) {
        return input;
    }
    let out_dim = bias.len();
    let b_id = add_param(g, name, bias.to_vec(), Shape::new(&[out_dim], DType::F32));
    params.insert(name.to_string(), bias.to_vec());
    g.add(input, b_id)
}

fn linear_gguf_bias(
    g: &mut HirMut<'_>,
    params: &mut HashMap<String, Vec<f32>>,
    typed: &mut Vec<(String, Vec<u8>, DType)>,
    cache: &mut HashMap<String, HirNodeId>,
    ir_stem: &str,
    p: &GgufPackedLinear,
    input: HirNodeId,
    bias: &[f32],
    in_dim: usize,
    out_dim: usize,
) -> Result<HirNodeId> {
    let y = linear_gguf_matmul(g, typed, cache, ir_stem, p, input, in_dim, out_dim)?;
    Ok(add_f32_bias(g, params, &format!("{ir_stem}.b"), y, bias))
}

fn in_proj_qkv(
    g: &mut HirMut<'_>,
    params: &mut HashMap<String, Vec<f32>>,
    typed: &mut Vec<(String, Vec<u8>, DType)>,
    cache: &mut HashMap<String, HirNodeId>,
    gguf_packed: Option<&GgufPackedParams>,
    gguf_key: &str,
    ir_stem: &str,
    layer_w_t: &[f32],
    layer_b: &[f32],
    input_q: HirNodeId,
    input_k: HirNodeId,
    input_v: HirNodeId,
    d: usize,
) -> Result<(HirNodeId, HirNodeId, HirNodeId)> {
    if let Some(p) = gguf_packed.and_then(|m| packed_linear(m, gguf_key)) {
        let qkv_q = linear_gguf_bias(
            g,
            params,
            typed,
            cache,
            ir_stem,
            p,
            input_q,
            layer_b,
            d,
            3 * d,
        )?;
        let qkv_k = linear_gguf_bias(
            g,
            params,
            typed,
            cache,
            ir_stem,
            p,
            input_k,
            layer_b,
            d,
            3 * d,
        )?;
        let qkv_v = linear_gguf_bias(
            g,
            params,
            typed,
            cache,
            ir_stem,
            p,
            input_v,
            layer_b,
            d,
            3 * d,
        )?;
        let axis = g.shape(qkv_q).rank().saturating_sub(1);
        let q = g.narrow_(qkv_q, axis, 0, d);
        let k = g.narrow_(qkv_k, axis, d, d);
        let v = g.narrow_(qkv_v, axis, 2 * d, d);
        return Ok((q, k, v));
    }
    let (wq, wk, wv) = split_qkv(layer_w_t, d);
    let bq = layer_b[0..d].to_vec();
    let bk = layer_b[d..2 * d].to_vec();
    let bv = layer_b[2 * d..3 * d].to_vec();
    let batch_q = g.shape(input_q).dims()[0].unwrap_static();
    let seq_q = g.shape(input_q).dims()[1].unwrap_static();
    let batch_k = g.shape(input_k).dims()[0].unwrap_static();
    let seq_k = g.shape(input_k).dims()[1].unwrap_static();
    let batch_v = g.shape(input_v).dims()[0].unwrap_static();
    let seq_v = g.shape(input_v).dims()[1].unwrap_static();
    let q = qkv_linear(
        g,
        params,
        &format!("{ir_stem}.q"),
        input_q,
        wq,
        bq,
        batch_q,
        seq_q,
        d,
    );
    let k = qkv_linear(
        g,
        params,
        &format!("{ir_stem}.k"),
        input_k,
        wk,
        bk,
        batch_k,
        seq_k,
        d,
    );
    let v = qkv_linear(
        g,
        params,
        &format!("{ir_stem}.v"),
        input_v,
        wv,
        bv,
        batch_v,
        seq_v,
        d,
    );
    Ok((q, k, v))
}

fn linear_fused_or_gguf(
    g: &mut HirMut<'_>,
    params: &mut HashMap<String, Vec<f32>>,
    typed: &mut Vec<(String, Vec<u8>, DType)>,
    cache: &mut HashMap<String, HirNodeId>,
    gguf_packed: Option<&GgufPackedParams>,
    gguf_key: &str,
    ir_stem: &str,
    input: HirNodeId,
    w_t: Vec<f32>,
    bias: Vec<f32>,
    in_dim: usize,
    out_dim: usize,
) -> Result<HirNodeId> {
    if let Some(p) = gguf_packed.and_then(|m| packed_linear(m, gguf_key)) {
        return linear_gguf_bias(
            g, params, typed, cache, ir_stem, p, input, &bias, in_dim, out_dim,
        );
    }
    Ok(linear_with_bias(
        g, params, ir_stem, input, w_t, bias, in_dim, out_dim,
    ))
}

fn split_qkv(w_t: &[f32], e: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut wq = vec![0f32; e * e];
    let mut wk = vec![0f32; e * e];
    let mut wv = vec![0f32; e * e];
    for i in 0..e {
        for j in 0..e {
            wq[i * e + j] = w_t[i * 3 * e + j];
            wk[i * e + j] = w_t[i * 3 * e + e + j];
            wv[i * e + j] = w_t[i * 3 * e + 2 * e + j];
        }
    }
    (wq, wk, wv)
}

fn add_param(g: &mut HirMut<'_>, name: &str, _data: Vec<f32>, shape: Shape) -> HirNodeId {
    g.param(name, shape)
}

/// Lower encoder HIR to legacy [`Graph`] (via [`super::flow::Sam3DetectorEncoderFlow`]).
pub fn build_sam3_detector_encoder_graph(
    weights: &Sam3EncoderWeights,
    batch: usize,
    hw: usize,
    seq: usize,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    let parts = build_encoder_hir(weights, batch, hw, seq, None)?;
    rlx_core::flow_util::graph_from_hir(parts.hir, parts.params)
}

/// Build native HIR encoder module + parameter blobs.
pub fn build_encoder_hir(
    weights: &Sam3EncoderWeights,
    batch: usize,
    hw: usize,
    seq: usize,
    gguf_packed: Option<&GgufPackedParams>,
) -> Result<Sam3EncoderHirParts> {
    let mut hir = HirModule::new("sam3_detector_encoder");
    let mut g = HirMut::new(&mut hir);
    let mut params: HashMap<String, Vec<f32>> = HashMap::new();
    let mut typed_params = Vec::new();
    let mut gguf_w_cache: HashMap<String, HirNodeId> = HashMap::new();
    let f = DType::F32;
    let d = D_MODEL;
    let enc_base = &weights.prefix;

    let src = g.input("src", Shape::new(&[batch, hw, d], f));
    let src_pos = g.input("src_pos", Shape::new(&[batch, hw, d], f));
    let prompt = g.input("prompt", Shape::new(&[batch, seq, d], f));
    let prompt_kpm_inv = g.input("prompt_kpm_inv", Shape::new(&[batch, seq], f));

    let mut tgt = src;
    for (li, layer) in weights.layers.iter().enumerate() {
        tgt = emit_sam3_detector_encoder_layer(
            &mut g,
            &mut params,
            &mut typed_params,
            &mut gguf_w_cache,
            gguf_packed,
            enc_base,
            li,
            layer,
            batch,
            hw,
            seq,
            tgt,
            src_pos,
            prompt,
            prompt_kpm_inv,
        )?;
    }
    g.set_outputs(vec![tgt]);
    Ok(Sam3EncoderHirParts {
        hir,
        params,
        typed_params,
    })
}

/// One SAM3 fusion encoder layer (self-attn + cross-attn + FFN).
#[allow(clippy::too_many_arguments)]
pub fn emit_sam3_detector_encoder_layer(
    g: &mut HirMut<'_>,
    params: &mut HashMap<String, Vec<f32>>,
    typed_params: &mut Vec<(String, Vec<u8>, DType)>,
    gguf_w_cache: &mut HashMap<String, HirNodeId>,
    gguf_packed: Option<&GgufPackedParams>,
    enc_base: &str,
    li: usize,
    layer: &Sam3EncoderLayerWeights,
    _batch: usize,
    _hw: usize,
    _seq: usize,
    tgt: HirNodeId,
    src_pos: HirNodeId,
    prompt: HirNodeId,
    prompt_kpm_inv: HirNodeId,
) -> Result<HirNodeId> {
    let f = DType::F32;
    let d = D_MODEL;
    let nh = N_HEADS;
    let dh = HEAD_DIM;
    let dim_ff = DIM_FF;

    let n1_w = add_param(
        g,
        &format!("l{li}.norm1.w"),
        layer.norm1_w.clone(),
        Shape::new(&[d], f),
    );
    params.insert(format!("l{li}.norm1.w"), layer.norm1_w.clone());
    let n1_b = add_param(
        g,
        &format!("l{li}.norm1.b"),
        layer.norm1_b.clone(),
        Shape::new(&[d], f),
    );
    params.insert(format!("l{li}.norm1.b"), layer.norm1_b.clone());
    let n1 = g.ln(tgt, n1_w, n1_b, 1e-5);

    let qk_in = g.add(n1, src_pos);

    let (q_node, k_node, v_node) = in_proj_qkv(
        g,
        params,
        typed_params,
        gguf_w_cache,
        gguf_packed,
        &enc_layer_key(enc_base, li, "self_attn.in_proj_weight"),
        &format!("l{li}.sa.in_proj"),
        &layer.self_attn_in_w_t,
        &layer.self_attn_in_b,
        qk_in,
        qk_in,
        n1,
        d,
    )?;

    let sa_attn = g.attention_kind(
        q_node,
        k_node,
        v_node,
        nh,
        dh,
        MaskKind::None,
        shape::attention_shape(g.shape(q_node)),
    );
    let sa_out = linear_fused_or_gguf(
        g,
        params,
        typed_params,
        gguf_w_cache,
        gguf_packed,
        &enc_layer_key(enc_base, li, "self_attn.out_proj.weight"),
        &format!("l{li}.sa.proj"),
        sa_attn,
        layer.self_attn_out_w_t.clone(),
        layer.self_attn_out_b.clone(),
        d,
        d,
    )?;
    let mut tgt = g.add(tgt, sa_out);

    let n2_w = add_param(
        g,
        &format!("l{li}.norm2.w"),
        layer.norm2_w.clone(),
        Shape::new(&[d], f),
    );
    params.insert(format!("l{li}.norm2.w"), layer.norm2_w.clone());
    let n2_b = add_param(
        g,
        &format!("l{li}.norm2.b"),
        layer.norm2_b.clone(),
        Shape::new(&[d], f),
    );
    params.insert(format!("l{li}.norm2.b"), layer.norm2_b.clone());
    let n2 = g.ln(tgt, n2_w, n2_b, 1e-5);

    let (qc, kc, vc) = in_proj_qkv(
        g,
        params,
        typed_params,
        gguf_w_cache,
        gguf_packed,
        &enc_layer_key(enc_base, li, "cross_attn_image.in_proj_weight"),
        &format!("l{li}.ca.in_proj"),
        &layer.cross_attn_in_w_t,
        &layer.cross_attn_in_b,
        n2,
        prompt,
        prompt,
        d,
    )?;

    let ca_attn = g.attention(
        qc,
        kc,
        vc,
        prompt_kpm_inv,
        nh,
        dh,
        shape::attention_shape(g.shape(qc)),
    );
    let ca_out = linear_fused_or_gguf(
        g,
        params,
        typed_params,
        gguf_w_cache,
        gguf_packed,
        &enc_layer_key(enc_base, li, "cross_attn_image.out_proj.weight"),
        &format!("l{li}.ca.proj"),
        ca_attn,
        layer.cross_attn_out_w_t.clone(),
        layer.cross_attn_out_b.clone(),
        d,
        d,
    )?;
    tgt = g.add(tgt, ca_out);

    let n3_w = add_param(
        g,
        &format!("l{li}.norm3.w"),
        layer.norm3_w.clone(),
        Shape::new(&[d], f),
    );
    params.insert(format!("l{li}.norm3.w"), layer.norm3_w.clone());
    let n3_b = add_param(
        g,
        &format!("l{li}.norm3.b"),
        layer.norm3_b.clone(),
        Shape::new(&[d], f),
    );
    params.insert(format!("l{li}.norm3.b"), layer.norm3_b.clone());
    let n3 = g.ln(tgt, n3_w, n3_b, 1e-5);

    let ff1 = linear_fused_or_gguf(
        g,
        params,
        typed_params,
        gguf_w_cache,
        gguf_packed,
        &enc_layer_key(enc_base, li, "linear1.weight"),
        &format!("l{li}.ffn.fc1"),
        n3,
        layer.linear1_w_t.clone(),
        layer.linear1_b.clone(),
        d,
        dim_ff,
    )?;
    let relud = g.relu(ff1);
    let ff2 = linear_fused_or_gguf(
        g,
        params,
        typed_params,
        gguf_w_cache,
        gguf_packed,
        &enc_layer_key(enc_base, li, "linear2.weight"),
        &format!("l{li}.ffn.fc2"),
        relud,
        layer.linear2_w_t.clone(),
        layer.linear2_b.clone(),
        dim_ff,
        d,
    )?;
    Ok(g.add(tgt, ff2))
}

fn qkv_linear(
    g: &mut HirMut<'_>,
    params: &mut HashMap<String, Vec<f32>>,
    name: &str,
    input: HirNodeId,
    w: Vec<f32>,
    b: Vec<f32>,
    batch: usize,
    seq: usize,
    d: usize,
) -> HirNodeId {
    let f = DType::F32;
    let w_name = format!("{name}.w");
    let b_name = format!("{name}.b");
    let w_id = g.param(&w_name, Shape::new(&[d, d], f));
    params.insert(w_name, w);
    let b_id = g.param(&b_name, Shape::new(&[d], f));
    params.insert(b_name, b);
    let out_shape = Shape::new(&[batch, seq, d], f);
    g.add_node(
        Op::FusedMatMulBiasAct { activation: None },
        vec![input, w_id, b_id],
        out_shape,
    )
}

fn linear_with_bias(
    g: &mut HirMut<'_>,
    params: &mut HashMap<String, Vec<f32>>,
    name: &str,
    input: HirNodeId,
    w: Vec<f32>,
    b: Vec<f32>,
    in_dim: usize,
    out_dim: usize,
) -> HirNodeId {
    let f = DType::F32;
    let w_name = format!("{name}.w");
    let b_name = format!("{name}.b");
    let w_id = g.param(&w_name, Shape::new(&[in_dim, out_dim], f));
    params.insert(w_name, w);
    let b_id = g.param(&b_name, Shape::new(&[out_dim], f));
    params.insert(b_name, b);
    let cur_shape = g.shape(input);
    let mut out_dims: Vec<usize> = cur_shape.dims().iter().map(|d| d.unwrap_static()).collect();
    *out_dims.last_mut().unwrap() = out_dim;
    g.add_node(
        Op::FusedMatMulBiasAct { activation: None },
        vec![input, w_id, b_id],
        Shape::new(&out_dims, f),
    )
}

#[allow(clippy::too_many_arguments)]
pub fn forward_encoder_ir_on(
    weights: &Sam3EncoderWeights,
    src_bchw: &[f32],
    src_pos_bchw: &[f32],
    prompt_seq_first: &[f32],
    prompt_kpm: &[u8],
    batch: usize,
    src_h: usize,
    src_w: usize,
    prompt_len: usize,
    device: Device,
) -> Result<Vec<f32>> {
    forward_encoder_ir_on_with_profile(
        weights,
        src_bchw,
        src_pos_bchw,
        prompt_seq_first,
        prompt_kpm,
        batch,
        src_h,
        src_w,
        prompt_len,
        device,
        &CompileProfile::sam3(),
        None,
    )
}

/// Same as [`forward_encoder_ir_on`] with an explicit tier-1 profile.
#[allow(clippy::too_many_arguments)]
pub fn forward_encoder_ir_on_with_profile(
    weights: &Sam3EncoderWeights,
    src_bchw: &[f32],
    src_pos_bchw: &[f32],
    prompt_seq_first: &[f32],
    prompt_kpm: &[u8],
    batch: usize,
    src_h: usize,
    src_w: usize,
    prompt_len: usize,
    device: Device,
    profile: &CompileProfile,
    gguf_packed: Option<&GgufPackedParams>,
) -> Result<Vec<f32>> {
    ensure!(weights.loaded, "SAM3 detector encoder not loaded");
    let hw = src_h * src_w;
    ensure!(
        src_bchw.len() == batch * D_MODEL * hw,
        "encoder src shape mismatch"
    );
    ensure!(
        prompt_seq_first.len() == prompt_len * batch * D_MODEL,
        "encoder prompt shape mismatch"
    );

    let mut src_bhwc = vec![0f32; batch * hw * D_MODEL];
    let mut pos_bhwc = vec![0f32; batch * hw * D_MODEL];
    for b in 0..batch {
        for s in 0..hw {
            for c in 0..D_MODEL {
                src_bhwc[(b * hw + s) * D_MODEL + c] = src_bchw[((b * D_MODEL + c) * hw) + s];
                pos_bhwc[(b * hw + s) * D_MODEL + c] = src_pos_bchw[((b * D_MODEL + c) * hw) + s];
            }
        }
    }

    let mut prompt_bf = vec![0f32; batch * prompt_len * D_MODEL];
    for b in 0..batch {
        for l in 0..prompt_len {
            let s = (l * batch + b) * D_MODEL;
            let dst = (b * prompt_len + l) * D_MODEL;
            prompt_bf[dst..dst + D_MODEL].copy_from_slice(&prompt_seq_first[s..s + D_MODEL]);
        }
    }
    let prompt_kpm_inv: Vec<f32> = prompt_kpm
        .iter()
        .map(|&v| if v == 0 { 1.0 } else { 0.0 })
        .collect();

    let parts = build_encoder_hir(weights, batch, hw, prompt_len, gguf_packed)?;
    let mut compiled = rlx_core::flow_bridge::compile_hir_with_profile(device, parts.hir, profile)?;
    rlx_core::flow_util::attach_built_params(&mut compiled, parts.params, &parts.typed_params);
    let outputs = compiled.run(&[
        ("src", src_bhwc.as_slice()),
        ("src_pos", pos_bhwc.as_slice()),
        ("prompt", prompt_bf.as_slice()),
        ("prompt_kpm_inv", prompt_kpm_inv.as_slice()),
    ]);
    let out = outputs
        .into_iter()
        .next()
        .ok_or_else(|| anyhow::anyhow!("encoder graph produced no outputs"))?;
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
pub fn forward_encoder_ir(
    weights: &Sam3EncoderWeights,
    src_bchw: &[f32],
    src_pos_bchw: &[f32],
    prompt_seq_first: &[f32],
    prompt_kpm: &[u8],
    batch: usize,
    src_h: usize,
    src_w: usize,
    prompt_len: usize,
) -> Result<Vec<f32>> {
    forward_encoder_ir_on_with_profile(
        weights,
        src_bchw,
        src_pos_bchw,
        prompt_seq_first,
        prompt_kpm,
        batch,
        src_h,
        src_w,
        prompt_len,
        Device::Cpu,
        &CompileProfile::sam3(),
        None,
    )
}
