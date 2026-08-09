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

//! Feature-enhancer fusion (BiMultiHeadAttention) as an on-device HIR graph.
//!
//! The bidirectional vision↔text attention shares one score matrix
//! `A = Qv·Ktᵀ` and softmaxes it both ways (over text for the vision update,
//! over vision for the text update). Built from primitive ops (batched `mm`,
//! `sm`, `transpose`, `ln`) so it runs natively on every backend. The
//! `±50000` score clamp HF applies is omitted — it never triggers for normal
//! activations. Pairs with the validated `ir::mha` (text self-attn) and the
//! fused deformable op to put the whole enhancer layer on-device.

use crate::config::GroundingDinoConfig;
use crate::deform_attn::{DeformWeights, LevelShape};
use crate::deform_attn_ir::DeformParams;
use crate::enhancer::{Encoder, EncoderOutput};
use crate::ir::{self, Params};
use crate::weights::get;
use anyhow::Result;
use rlx_core::weight_map::WeightMap;
use rlx_ir::{DType, HirGraphExt, HirModule, HirMut, Shape};
use rlx_runtime::Device;

const NEG_INF: f32 = -1e30;

/// Fusion-layer weights (PyTorch `[out, in]`).
#[derive(Clone)]
pub struct FusionWeights {
    pub vision_proj_w: Vec<f32>,
    pub vision_proj_b: Vec<f32>,
    pub text_proj_w: Vec<f32>,
    pub text_proj_b: Vec<f32>,
    pub values_vision_proj_w: Vec<f32>,
    pub values_vision_proj_b: Vec<f32>,
    pub values_text_proj_w: Vec<f32>,
    pub values_text_proj_b: Vec<f32>,
    pub out_vision_proj_w: Vec<f32>,
    pub out_vision_proj_b: Vec<f32>,
    pub out_text_proj_w: Vec<f32>,
    pub out_text_proj_b: Vec<f32>,
    pub ln_vision_w: Vec<f32>,
    pub ln_vision_b: Vec<f32>,
    pub ln_text_w: Vec<f32>,
    pub ln_text_b: Vec<f32>,
    pub vision_param: Vec<f32>,
    pub text_param: Vec<f32>,
}

/// On-device fusion layer.
pub struct FusionIr {
    w: FusionWeights,
    d: usize,
    bi_dim: usize,
    n_heads: usize,
    eps: f32,
    device: Device,
}

impl FusionIr {
    pub fn new(w: FusionWeights, d: usize, bi_dim: usize, n_heads: usize, device: Device) -> Self {
        Self {
            w,
            d,
            bi_dim,
            n_heads,
            eps: 1e-5,
            device,
        }
    }

    /// `vision` is `[lv, d]`, `text` is `[lt, d]`. Returns updated
    /// `(vision, text)` (layerscale + residual applied), matching the native
    /// `Encoder::fusion` + residual. Standalone runner (one graph); the fused
    /// enhancer uses `build_fusion` to compose into a shared graph instead.
    pub fn forward(&self, vision: &[f32], text: &[f32]) -> Result<(Vec<f32>, Vec<f32>)> {
        let d = self.d;
        let lv = vision.len() / d;
        let lt = text.len() / d;

        let mut hir = HirModule::new("gdino_fusion");
        let mut params = Params::new();
        let mut g = HirMut::new(&mut hir);

        let vis_n = g.input("vision", Shape::new(&[lv, d], DType::F32));
        let txt_n = g.input("text", Shape::new(&[lt, d], DType::F32));
        let (vis_out, txt_out) = build_fusion(
            &mut g,
            &mut params,
            &self.w,
            "",
            vis_n,
            txt_n,
            d,
            self.bi_dim,
            self.n_heads,
            lv,
            lt,
            self.eps,
        );
        g.set_outputs(vec![vis_out, txt_out]);

        let outs = ir::compile_and_run(
            hir,
            params,
            self.device,
            &[("vision", vision), ("text", text)],
        )?;
        let mut it = outs.into_iter();
        Ok((it.next().unwrap_or_default(), it.next().unwrap_or_default()))
    }
}

/// Build the bidirectional fusion math into a shared graph and return the
/// updated `(vision, text)` nodes. `prefix` namespaces this layer's params.
#[allow(clippy::too_many_arguments)]
fn build_fusion(
    g: &mut HirMut<'_>,
    params: &mut Params,
    w: &FusionWeights,
    prefix: &str,
    vis_n: rlx_ir::HirNodeId,
    txt_n: rlx_ir::HirNodeId,
    d: usize,
    bi: usize,
    nh: usize,
    lv: usize,
    lt: usize,
    eps: f32,
) -> (rlx_ir::HirNodeId, rlx_ir::HirNodeId) {
    let hd = bi / nh;
    let scale = 1.0 / (hd as f32).sqrt();
    let n = |s: &str| format!("{prefix}{s}");

    let vn = ir::layer_norm(
        g,
        params,
        &n("lnv"),
        vis_n,
        &w.ln_vision_w,
        &w.ln_vision_b,
        eps,
    );
    let tn = ir::layer_norm(g, params, &n("lnt"), txt_n, &w.ln_text_w, &w.ln_text_b, eps);

    // Projections to bi_dim (scale folded into the query/vision proj).
    let qv = ir::linear(
        g,
        params,
        &n("vproj"),
        vn,
        d,
        bi,
        &w.vision_proj_w,
        &w.vision_proj_b,
        scale,
    );
    let kt = ir::linear(
        g,
        params,
        &n("tproj"),
        tn,
        d,
        bi,
        &w.text_proj_w,
        &w.text_proj_b,
        1.0,
    );
    let vv = ir::linear(
        g,
        params,
        &n("vvproj"),
        vn,
        d,
        bi,
        &w.values_vision_proj_w,
        &w.values_vision_proj_b,
        1.0,
    );
    let vt = ir::linear(
        g,
        params,
        &n("vtproj"),
        tn,
        d,
        bi,
        &w.values_text_proj_w,
        &w.values_text_proj_b,
        1.0,
    );

    // [l, bi] → [heads, l, hd] (and [heads, hd, lt] for K).
    let qv = g.reshape_(qv, vec![lv as i64, nh as i64, hd as i64]);
    let qv = g.transpose_(qv, vec![1, 0, 2]);
    let kt2 = g.reshape_(kt, vec![lt as i64, nh as i64, hd as i64]);
    let kt2 = g.transpose_(kt2, vec![1, 2, 0]);
    let vv2 = g.reshape_(vv, vec![lv as i64, nh as i64, hd as i64]);
    let vv2 = g.transpose_(vv2, vec![1, 0, 2]);
    let vt2 = g.reshape_(vt, vec![lt as i64, nh as i64, hd as i64]);
    let vt2 = g.transpose_(vt2, vec![1, 0, 2]);

    let a = g.mm(qv, kt2); // [heads, lv, lt]

    // Vision update: softmax over lt → @ Vt.
    let pv = g.sm(a, -1);
    let ctxv = g.mm(pv, vt2); // [heads, lv, hd]
    let ctxv = g.transpose_(ctxv, vec![1, 0, 2]);
    let ctxv = g.reshape_(ctxv, vec![lv as i64, bi as i64]);
    let dv = ir::linear(
        g,
        params,
        &n("outv"),
        ctxv,
        bi,
        d,
        &w.out_vision_proj_w,
        &w.out_vision_proj_b,
        1.0,
    );

    // Text update: softmax of Aᵀ over lv → @ Vv.
    let at = g.transpose_(a, vec![0, 2, 1]); // [heads, lt, lv]
    let pt = g.sm(at, -1);
    let ctxt = g.mm(pt, vv2); // [heads, lt, hd]
    let ctxt = g.transpose_(ctxt, vec![1, 0, 2]);
    let ctxt = g.reshape_(ctxt, vec![lt as i64, bi as i64]);
    let dt = ir::linear(
        g,
        params,
        &n("outt"),
        ctxt,
        bi,
        d,
        &w.out_text_proj_w,
        &w.out_text_proj_b,
        1.0,
    );

    // Layerscale + residual. HF adds the delta to the LAYER-NORMED features
    // (`vn`/`tn`), not the original inputs.
    let vparam = ir::vec_param(g, params, &n("vparam"), &w.vision_param, 1.0);
    let tparam = ir::vec_param(g, params, &n("tparam"), &w.text_param, 1.0);
    let sv = g.mul(dv, vparam);
    let vis_out = g.add(vn, sv);
    let st = g.mul(dt, tparam);
    let txt_out = g.add(tn, st);
    (vis_out, txt_out)
}

/// Text-enhancer weights (PyTorch `[out, in]`).
#[derive(Clone)]
struct TeWeights {
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

/// Deformable-layer residual/FFN wrapper weights (the deform op itself is the
/// fused custom op in [`MsDeformAttnIr`]).
#[derive(Clone)]
struct DwWeights {
    sa_ln_w: Vec<f32>,
    sa_ln_b: Vec<f32>,
    fc1_w: Vec<f32>,
    fc1_b: Vec<f32>,
    fc2_w: Vec<f32>,
    fc2_b: Vec<f32>,
    final_ln_w: Vec<f32>,
    final_ln_b: Vec<f32>,
}

/// Text self-attention enhancer as one HIR graph: `mha(text+pos, text+pos,
/// text) + residual → LN → ReLU-FFN + residual → LN`. Mirrors
/// `Encoder::text_enhancer`.
/// Standalone single-graph runner — kept for the parity tests; the production
/// path composes [`build_text_enhancer`] into the fused enhancer graph.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn text_enhancer_ir(
    te: &TeWeights,
    text: &[f32],
    text_pos: &[f32],
    bias: &[f32],
    d: usize,
    nh: usize,
    eps: f32,
    device: Device,
) -> Result<Vec<f32>> {
    let lt = text.len() / d;
    let mut qk = vec![0f32; lt * d];
    for i in 0..lt * d {
        qk[i] = text[i] + text_pos[i];
    }

    let mut hir = HirModule::new("gdino_text_enhancer");
    let mut params = Params::new();
    let mut g = HirMut::new(&mut hir);
    let text_n = g.input("text", Shape::new(&[lt, d], DType::F32));
    let qk_n = g.input("qk", Shape::new(&[lt, d], DType::F32));
    let bias_n = g.input("bias", Shape::new(&[1, lt, lt], DType::F32));
    let out = build_text_enhancer(
        &mut g,
        &mut params,
        te,
        "",
        text_n,
        qk_n,
        bias_n,
        lt,
        d,
        nh,
        eps,
    );
    g.set_outputs(vec![out]);

    let outs = ir::compile_and_run(
        hir,
        params,
        device,
        &[("text", text), ("qk", &qk), ("bias", bias)],
    )?;
    Ok(outs.into_iter().next().unwrap_or_default())
}

/// Build the text self-attention enhancer (mha + residual → LN → FFN → LN) into
/// a shared graph. `text_n` is the value branch; `qk_n` is the query/key branch
/// (text + pos). `prefix` namespaces this layer's params.
#[allow(clippy::too_many_arguments)]
fn build_text_enhancer(
    g: &mut HirMut<'_>,
    params: &mut Params,
    te: &TeWeights,
    prefix: &str,
    text_n: rlx_ir::HirNodeId,
    qk_n: rlx_ir::HirNodeId,
    bias_n: rlx_ir::HirNodeId,
    lt: usize,
    d: usize,
    nh: usize,
    eps: f32,
) -> rlx_ir::HirNodeId {
    let inter = te.fc1_b.len();
    let n = |s: &str| format!("{prefix}{s}");
    let attn = ir::mha(
        g,
        params,
        &n("te"),
        qk_n,
        qk_n,
        text_n,
        lt,
        lt,
        d,
        nh,
        &te.q_w,
        &te.q_b,
        &te.k_w,
        &te.k_b,
        &te.v_w,
        &te.v_b,
        &te.out_w,
        &te.out_b,
        bias_n,
    );
    let res = g.add(attn, text_n);
    let normed = ir::layer_norm(
        g,
        params,
        &n("lnb"),
        res,
        &te.ln_before_w,
        &te.ln_before_b,
        eps,
    );
    let f1 = ir::linear(
        g,
        params,
        &n("fc1"),
        normed,
        d,
        inter,
        &te.fc1_w,
        &te.fc1_b,
        1.0,
    );
    let act = g.relu(f1);
    let f2 = ir::linear(
        g,
        params,
        &n("fc2"),
        act,
        inter,
        d,
        &te.fc2_w,
        &te.fc2_b,
        1.0,
    );
    let res2 = g.add(f2, normed);
    ir::layer_norm(
        g,
        params,
        &n("lna"),
        res2,
        &te.ln_after_w,
        &te.ln_after_b,
        eps,
    )
}

/// Deformable-layer residual + FFN as one HIR graph: `vision + deform → LN →
/// ReLU-FFN + residual → LN`. Mirrors the tail of `Encoder::deformable`.
/// Standalone single-graph runner — kept for the parity tests; the production
/// path composes [`build_deform_post`] into the fused enhancer graph.
#[cfg(test)]
fn deform_post_ir(
    dw: &DwWeights,
    vision: &[f32],
    deform_out: &[f32],
    d: usize,
    eps: f32,
    device: Device,
) -> Result<Vec<f32>> {
    let seq = vision.len() / d;

    let mut hir = HirModule::new("gdino_deform_post");
    let mut params = Params::new();
    let mut g = HirMut::new(&mut hir);
    let vision_n = g.input("vision", Shape::new(&[seq, d], DType::F32));
    let deform_n = g.input("deform", Shape::new(&[seq, d], DType::F32));
    let out = build_deform_post(&mut g, &mut params, dw, "", vision_n, deform_n, d, eps);
    g.set_outputs(vec![out]);

    let outs = ir::compile_and_run(
        hir,
        params,
        device,
        &[("vision", vision), ("deform", deform_out)],
    )?;
    Ok(outs.into_iter().next().unwrap_or_default())
}

/// Build `vision + deform → LN → ReLU-FFN + residual → LN` into a shared graph.
#[allow(clippy::too_many_arguments)]
fn build_deform_post(
    g: &mut HirMut<'_>,
    params: &mut Params,
    dw: &DwWeights,
    prefix: &str,
    vision_n: rlx_ir::HirNodeId,
    deform_n: rlx_ir::HirNodeId,
    d: usize,
    eps: f32,
) -> rlx_ir::HirNodeId {
    let inter = dw.fc1_b.len();
    let n = |s: &str| format!("{prefix}{s}");
    let res = g.add(vision_n, deform_n);
    let normed = ir::layer_norm(g, params, &n("saln"), res, &dw.sa_ln_w, &dw.sa_ln_b, eps);
    let f1 = ir::linear(
        g,
        params,
        &n("fc1"),
        normed,
        d,
        inter,
        &dw.fc1_w,
        &dw.fc1_b,
        1.0,
    );
    let act = g.relu(f1);
    let f2 = ir::linear(
        g,
        params,
        &n("fc2"),
        act,
        inter,
        d,
        &dw.fc2_w,
        &dw.fc2_b,
        1.0,
    );
    let res2 = g.add(f2, normed);
    ir::layer_norm(
        g,
        params,
        &n("finln"),
        res2,
        &dw.final_ln_w,
        &dw.final_ln_b,
        eps,
    )
}

/// Build the fused multi-scale deformable-attention custom op into the enhancer
/// graph via the shared [`crate::deform_attn_ir::build_deform_node`]. Encoder
/// deform self-attention: queries are the vision tokens, so the output row count
/// equals the flattened multi-scale sequence length.
#[allow(clippy::too_many_arguments)]
fn build_deform(
    g: &mut HirMut<'_>,
    params: &mut Params,
    w: &DeformParams,
    prefix: &str,
    query_n: rlx_ir::HirNodeId,
    value_n: rlx_ir::HirNodeId,
    ref_n: rlx_ir::HirNodeId,
    d: usize,
    nh: usize,
    np: usize,
    ref_dim: usize,
    shapes: &[LevelShape],
) -> rlx_ir::HirNodeId {
    let seq: usize = shapes.iter().map(|s| s.h * s.w).sum();
    let dw = DeformWeights {
        value_proj_w: &w.value_proj_w,
        value_proj_b: &w.value_proj_b,
        sampling_offsets_w: &w.sampling_offsets_w,
        sampling_offsets_b: &w.sampling_offsets_b,
        attention_weights_w: &w.attention_weights_w,
        attention_weights_b: &w.attention_weights_b,
        output_proj_w: &w.output_proj_w,
        output_proj_b: &w.output_proj_b,
    };
    crate::deform_attn_ir::build_deform_node(
        g, params, prefix, query_n, value_n, ref_n, &dw, d, nh, np, ref_dim, seq, shapes,
    )
}

struct EncLayerIr {
    fusion: FusionIr,
    te: TeWeights,
    deform_params: DeformParams,
    np: usize,
    dw: DwWeights,
}

/// Full feature enhancer (all layers) on-device. Each layer: bidirectional
/// fusion → text self-attention enhancer → vision deformable self-attention,
/// numerically matching [`Encoder::forward`] on `Device::Cpu`.
pub struct EncoderIr {
    layers: Vec<EncLayerIr>,
    d: usize,
    n_heads: usize,
    eps: f32,
    device: Device,
}

impl EncoderIr {
    pub fn from_weights(wm: &WeightMap, cfg: &GroundingDinoConfig, device: Device) -> Result<Self> {
        let d = cfg.d_model;
        let nh = cfg.encoder_attention_heads;
        let bi_dim = cfg.encoder_ffn_dim / 2;
        let np = cfg.encoder_n_points;
        let mut layers = Vec::with_capacity(cfg.encoder_layers);
        for i in 0..cfg.encoder_layers {
            let fp = format!("model.encoder.layers.{i}.fusion_layer.");
            let tp = format!("model.encoder.layers.{i}.text_enhancer_layer.");
            let dp = format!("model.encoder.layers.{i}.deformable_layer.");
            let fw = FusionWeights {
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
            let te = TeWeights {
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
            let dparams = DeformParams {
                value_proj_w: get(wm, &format!("{dp}self_attn.value_proj.weight"))?,
                value_proj_b: get(wm, &format!("{dp}self_attn.value_proj.bias"))?,
                sampling_offsets_w: get(wm, &format!("{dp}self_attn.sampling_offsets.weight"))?,
                sampling_offsets_b: get(wm, &format!("{dp}self_attn.sampling_offsets.bias"))?,
                attention_weights_w: get(wm, &format!("{dp}self_attn.attention_weights.weight"))?,
                attention_weights_b: get(wm, &format!("{dp}self_attn.attention_weights.bias"))?,
                output_proj_w: get(wm, &format!("{dp}self_attn.output_proj.weight"))?,
                output_proj_b: get(wm, &format!("{dp}self_attn.output_proj.bias"))?,
            };
            let dw = DwWeights {
                sa_ln_w: get(wm, &format!("{dp}self_attn_layer_norm.weight"))?,
                sa_ln_b: get(wm, &format!("{dp}self_attn_layer_norm.bias"))?,
                fc1_w: get(wm, &format!("{dp}fc1.weight"))?,
                fc1_b: get(wm, &format!("{dp}fc1.bias"))?,
                fc2_w: get(wm, &format!("{dp}fc2.weight"))?,
                fc2_b: get(wm, &format!("{dp}fc2.bias"))?,
                final_ln_w: get(wm, &format!("{dp}final_layer_norm.weight"))?,
                final_ln_b: get(wm, &format!("{dp}final_layer_norm.bias"))?,
            };
            layers.push(EncLayerIr {
                // HF GroundingDinoBiMultiHeadAttention uses `encoder_attention_heads // 2`.
                fusion: FusionIr::new(fw, d, bi_dim, nh / 2, device),
                te,
                deform_params: dparams,
                np,
                dw,
            });
        }
        // The fused enhancer graph inlines the deform custom op (see
        // `build_deform`); make sure its IR shape rule + kernels are registered.
        crate::deform_op::ensure_registered();
        Ok(Self {
            layers,
            d,
            n_heads: nh,
            eps: 1e-5,
            device,
        })
    }

    /// Run all enhancer layers. Same signature/semantics as
    /// [`Encoder::forward`].
    pub fn forward(
        &self,
        vision: &[f32],
        vision_pos: &[f32],
        text: &[f32],
        text_pos: &[f32],
        text_self_mask: &[u8],
        shapes: &[LevelShape],
    ) -> Result<EncoderOutput> {
        let d = self.d;
        let nh = self.n_heads;
        let eps = self.eps;
        let lv = vision.len() / d;
        let lt = text.len() / d;
        let nl = shapes.len();
        let ref_points = Encoder::reference_points(shapes);

        let mut text_bias = vec![0f32; lt * lt];
        for i in 0..lt * lt {
            if text_self_mask[i] == 0 {
                text_bias[i] = NEG_INF;
            }
        }

        // Build the ENTIRE enhancer (all layers) as one HIR graph so the vision
        // and text tensors stay on-device across layers — no per-sub-op host
        // round-trip. Previously each layer ran 4 separate `compile_and_run`
        // calls (fusion / text-attn / deform / deform-post), each re-uploading
        // weights and eval-ing to a host `Vec`; that per-graph overhead made the
        // GPU backends slower than CPU. One graph amortizes it.
        let mut hir = HirModule::new("gdino_enhancer");
        let mut params = Params::new();
        let mut g = HirMut::new(&mut hir);

        let vis_in = g.input("vision", Shape::new(&[lv, d], DType::F32));
        let vpos_in = g.input("vision_pos", Shape::new(&[lv, d], DType::F32));
        let txt_in = g.input("text", Shape::new(&[lt, d], DType::F32));
        let tpos_in = g.input("text_pos", Shape::new(&[lt, d], DType::F32));
        let bias_in = g.input("text_bias", Shape::new(&[1, lt, lt], DType::F32));
        let ref_in = g.input("ref", Shape::new(&[lv, nl, 2], DType::F32));

        let mut vis = vis_in;
        let mut txt = txt_in;
        for (i, layer) in self.layers.iter().enumerate() {
            // Distinct sub-prefix per component so same-named params (e.g. both
            // the text-enhancer FFN and the deform-post FFN call theirs `fc1`)
            // don't collide in the shared graph's param map.
            // 1. Bidirectional fusion (residual + layerscale applied inside).
            let (v2, t2) = build_fusion(
                &mut g,
                &mut params,
                &layer.fusion.w,
                &format!("L{i}.fu."),
                vis,
                txt,
                d,
                layer.fusion.bi_dim,
                layer.fusion.n_heads,
                lv,
                lt,
                eps,
            );
            // 2. Text self-attention enhancer (qk = text + pos; HF uses nh/2).
            let qk = g.add(t2, tpos_in);
            let t3 = build_text_enhancer(
                &mut g,
                &mut params,
                &layer.te,
                &format!("L{i}.te."),
                t2,
                qk,
                bias_in,
                lt,
                d,
                nh / 2,
                eps,
            );
            // 3. Vision multi-scale deformable self-attention (query = vision + pos).
            let query = g.add(v2, vpos_in);
            let deform_out = build_deform(
                &mut g,
                &mut params,
                &layer.deform_params,
                &format!("L{i}.df."),
                query,
                v2,
                ref_in,
                d,
                nh,
                layer.np,
                2,
                shapes,
            );
            let v3 = build_deform_post(
                &mut g,
                &mut params,
                &layer.dw,
                &format!("L{i}.dp."),
                v2,
                deform_out,
                d,
                eps,
            );
            vis = v3;
            txt = t3;
        }
        g.set_outputs(vec![vis, txt]);

        let outs = ir::compile_and_run(
            hir,
            params,
            self.device,
            &[
                ("vision", vision),
                ("vision_pos", vision_pos),
                ("text", text),
                ("text_pos", text_pos),
                ("text_bias", &text_bias),
                ("ref", &ref_points),
            ],
        )?;
        let mut it = outs.into_iter();
        Ok(EncoderOutput {
            vision: it.next().unwrap_or_default(),
            text: it.next().unwrap_or_default(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn;

    fn det(n: usize, seed: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (((i * 13 + seed * 7) % 17) as f32 - 8.0) * 0.02)
            .collect()
    }

    /// Host reference for the fusion layer (mirrors `Encoder::fusion` + residual,
    /// without the score clamp which never triggers here).
    fn native(
        w: &FusionWeights,
        vision: &[f32],
        text: &[f32],
        d: usize,
        bi: usize,
        nh: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let hd = bi / nh;
        let lv = vision.len() / d;
        let lt = text.len() / d;
        let scale = 1.0 / (hd as f32).sqrt();
        let eps = 1e-5;
        let vn = nn::layer_norm(vision, &w.ln_vision_w, &w.ln_vision_b, d, eps);
        let tn = nn::layer_norm(text, &w.ln_text_w, &w.ln_text_b, d, eps);
        let mut qv = nn::linear(&vn, lv, d, &w.vision_proj_w, bi, &w.vision_proj_b);
        for v in qv.iter_mut() {
            *v *= scale;
        }
        let kt = nn::linear(&tn, lt, d, &w.text_proj_w, bi, &w.text_proj_b);
        let vv = nn::linear(
            &vn,
            lv,
            d,
            &w.values_vision_proj_w,
            bi,
            &w.values_vision_proj_b,
        );
        let vt = nn::linear(&tn, lt, d, &w.values_text_proj_w, bi, &w.values_text_proj_b);
        let mut visc = vec![0f32; lv * bi];
        let mut txtc = vec![0f32; lt * bi];
        for h in 0..nh {
            let mut a = vec![0f32; lv * lt];
            for i in 0..lv {
                for j in 0..lt {
                    let mut s = 0f32;
                    for c in 0..hd {
                        s += qv[i * bi + h * hd + c] * kt[j * bi + h * hd + c];
                    }
                    a[i * lt + j] = s;
                }
            }
            let mut av = a.clone();
            nn::softmax_rows(&mut av, lv, lt);
            for i in 0..lv {
                let ctx = &mut visc[i * bi + h * hd..i * bi + h * hd + hd];
                for j in 0..lt {
                    let p = av[i * lt + j];
                    for c in 0..hd {
                        ctx[c] += p * vt[j * bi + h * hd + c];
                    }
                }
            }
            let mut at = vec![0f32; lt * lv];
            for j in 0..lt {
                for i in 0..lv {
                    at[j * lv + i] = a[i * lt + j];
                }
            }
            nn::softmax_rows(&mut at, lt, lv);
            for j in 0..lt {
                let ctx = &mut txtc[j * bi + h * hd..j * bi + h * hd + hd];
                for i in 0..lv {
                    let p = at[j * lv + i];
                    for c in 0..hd {
                        ctx[c] += p * vv[i * bi + h * hd + c];
                    }
                }
            }
        }
        let dv = nn::linear(&visc, lv, bi, &w.out_vision_proj_w, d, &w.out_vision_proj_b);
        let dt = nn::linear(&txtc, lt, bi, &w.out_text_proj_w, d, &w.out_text_proj_b);
        // Residual on the layer-normed features (vn/tn), per HF.
        let mut vo = vn;
        for i in 0..lv * d {
            vo[i] += w.vision_param[i % d] * dv[i];
        }
        let mut to = tn;
        for i in 0..lt * d {
            to[i] += w.text_param[i % d] * dt[i];
        }
        (vo, to)
    }

    #[test]
    fn fusion_ir_matches_native() {
        let (d, bi, nh) = (8usize, 8usize, 2usize);
        let lv = 6usize;
        let lt = 4usize;
        let proj = |s| det(bi * d, s);
        let outp = |s| det(d * bi, s);
        let w = FusionWeights {
            vision_proj_w: proj(1),
            vision_proj_b: vec![0.0; bi],
            text_proj_w: proj(2),
            text_proj_b: vec![0.0; bi],
            values_vision_proj_w: proj(3),
            values_vision_proj_b: vec![0.0; bi],
            values_text_proj_w: proj(4),
            values_text_proj_b: vec![0.0; bi],
            out_vision_proj_w: outp(5),
            out_vision_proj_b: vec![0.0; d],
            out_text_proj_w: outp(6),
            out_text_proj_b: vec![0.0; d],
            ln_vision_w: vec![1.0; d],
            ln_vision_b: vec![0.0; d],
            ln_text_w: vec![1.0; d],
            ln_text_b: vec![0.0; d],
            vision_param: det(d, 7),
            text_param: det(d, 8),
        };
        let vision = det(lv * d, 20);
        let text = det(lt * d, 21);

        let (nv, nt) = native(&w, &vision, &text, d, bi, nh);
        let ir = FusionIr::new(w, d, bi, nh, Device::Cpu);
        let (iv, it) = ir.forward(&vision, &text).unwrap();

        let ev = nv
            .iter()
            .zip(&iv)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        let et = nt
            .iter()
            .zip(&it)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(ev < 1e-4, "vision max_err={ev}");
        assert!(et < 1e-4, "text max_err={et}");
    }

    #[test]
    fn text_enhancer_ir_matches_native() {
        let (d, nh, lt) = (8usize, 2usize, 5usize);
        let inter = 16usize;
        let eps = 1e-5;
        let te = TeWeights {
            q_w: det(d * d, 1),
            q_b: det(d, 2),
            k_w: det(d * d, 3),
            k_b: det(d, 4),
            v_w: det(d * d, 5),
            v_b: det(d, 6),
            out_w: det(d * d, 7),
            out_b: det(d, 8),
            fc1_w: det(inter * d, 9),
            fc1_b: det(inter, 10),
            fc2_w: det(d * inter, 11),
            fc2_b: det(d, 12),
            ln_before_w: vec![1.0; d],
            ln_before_b: det(d, 13),
            ln_after_w: vec![1.0; d],
            ln_after_b: det(d, 14),
        };
        let text = det(lt * d, 20);
        let text_pos = det(lt * d, 21);
        let bias = vec![0f32; lt * lt];

        // Native reference (mirrors Encoder::text_enhancer).
        let mut qk = vec![0f32; lt * d];
        for i in 0..lt * d {
            qk[i] = text[i] + text_pos[i];
        }
        let attn = nn::mha(
            &qk,
            &qk,
            &text,
            lt,
            lt,
            d,
            nh,
            &te.q_w,
            &te.q_b,
            &te.k_w,
            &te.k_b,
            &te.v_w,
            &te.v_b,
            &te.out_w,
            &te.out_b,
            nn::AttnBias::Shared(&bias),
        );
        let mut hidden = vec![0f32; lt * d];
        for i in 0..lt * d {
            hidden[i] = text[i] + attn[i];
        }
        hidden = nn::layer_norm(&hidden, &te.ln_before_w, &te.ln_before_b, d, eps);
        let mut f = nn::linear(&hidden, lt, d, &te.fc1_w, inter, &te.fc1_b);
        nn::relu(&mut f);
        let f2 = nn::linear(&f, lt, inter, &te.fc2_w, d, &te.fc2_b);
        for i in 0..lt * d {
            hidden[i] += f2[i];
        }
        let native = nn::layer_norm(&hidden, &te.ln_after_w, &te.ln_after_b, d, eps);

        let got = text_enhancer_ir(&te, &text, &text_pos, &bias, d, nh, eps, Device::Cpu).unwrap();
        let e = native
            .iter()
            .zip(&got)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(e < 1e-4, "text_enhancer max_err={e}");
    }

    #[test]
    fn deform_post_ir_matches_native() {
        let (d, seq) = (8usize, 7usize);
        let inter = 16usize;
        let eps = 1e-5;
        let dw = DwWeights {
            sa_ln_w: vec![1.0; d],
            sa_ln_b: det(d, 1),
            fc1_w: det(inter * d, 2),
            fc1_b: det(inter, 3),
            fc2_w: det(d * inter, 4),
            fc2_b: det(d, 5),
            final_ln_w: vec![1.0; d],
            final_ln_b: det(d, 6),
        };
        let vision = det(seq * d, 20);
        let deform = det(seq * d, 21);

        // Native reference (mirrors the tail of Encoder::deformable).
        let mut hidden = vec![0f32; seq * d];
        for i in 0..seq * d {
            hidden[i] = vision[i] + deform[i];
        }
        hidden = nn::layer_norm(&hidden, &dw.sa_ln_w, &dw.sa_ln_b, d, eps);
        let mut f = nn::linear(&hidden, seq, d, &dw.fc1_w, inter, &dw.fc1_b);
        nn::relu(&mut f);
        let f2 = nn::linear(&f, seq, inter, &dw.fc2_w, d, &dw.fc2_b);
        for i in 0..seq * d {
            hidden[i] += f2[i];
        }
        let native = nn::layer_norm(&hidden, &dw.final_ln_w, &dw.final_ln_b, d, eps);

        let got = deform_post_ir(&dw, &vision, &deform, d, eps, Device::Cpu).unwrap();
        let e = native
            .iter()
            .zip(&got)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(e < 1e-4, "deform_post max_err={e}");
    }
}
