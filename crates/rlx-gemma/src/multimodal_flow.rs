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

//! Gemma prefill helpers for multimodal (`prefill_hidden` + optional vision bias).

use rlx_flow::blocks::{
    CustomStage, GeGluStage, GemmaKvTapStage, GemmaLayerStyle, GemmaRmsNormStage,
    SelfAttnPrefillSpec, gemma_prefill_layer_composed,
};
use rlx_flow::{FlowStage, LayerStack};
use rlx_ir::HirGraphExt;
use rlx_ir::hir::HirMut;
use rlx_ir::op::MaskKind;
use rlx_ir::shape;

use crate::flow::GemmaLayerCtx;

fn repeat_kv(
    g: &mut HirMut,
    x: rlx_ir::HirNodeId,
    num_kv_heads: usize,
    head_dim: usize,
    group: usize,
) -> rlx_ir::HirNodeId {
    if group == 1 {
        return x;
    }
    let last_ax = g.shape(x).rank() - 1;
    let mut pieces = Vec::with_capacity(num_kv_heads * group);
    for h in 0..num_kv_heads {
        let slice = g.narrow_(x, last_ax, h * head_dim, head_dim);
        for _ in 0..group {
            pieces.push(slice);
        }
    }
    g.concat_(pieces, last_ax)
}

fn biased_self_attn_stage(spec: SelfAttnPrefillSpec) -> FlowStage {
    FlowStage::Custom(CustomStage::named(
        "biased_self_attn_prefill",
        move |emit, input| {
            let input = input.ok_or_else(|| anyhow::anyhow!("biased self-attn requires input"))?;
            let (cos, sin) = if let Some(slot) = spec.rope_table.as_deref() {
                let cos_key = format!("{slot}_cos");
                let sin_key = format!("{slot}_sin");
                (emit.named(&cos_key)?, emit.named(&sin_key)?)
            } else {
                let cos = emit
                    .state
                    .rope_cos
                    .ok_or_else(|| anyhow::anyhow!("biased self-attn requires RopeTables"))?;
                let sin = emit
                    .state
                    .rope_sin
                    .ok_or_else(|| anyhow::anyhow!("biased self-attn requires RopeTables"))?;
                (cos, sin)
            };

            let q_w = emit.load_param(&spec.q_key, true)?;
            let k_w = emit.load_param(&spec.k_key, true)?;
            let v_w = if spec.k_eq_v {
                None
            } else {
                Some(emit.load_param(&spec.v_key, true)?)
            };
            let bias = emit.flow_input("attn_bias")?.hir_id();

            let mut gb = HirMut::new(emit.hir());
            let q = gb.mm(input.hir_id(), q_w);
            let k = gb.mm(input.hir_id(), k_w);
            let v = match v_w {
                Some(w) => gb.mm(input.hir_id(), w),
                None => k,
            };
            let q_rope = gb.rope_n(q, cos, sin, spec.head_dim, spec.n_rot);
            let k_rope = gb.rope_n(k, cos, sin, spec.head_dim, spec.n_rot);

            let group = spec.num_heads / spec.num_kv_heads;
            let k_rep = repeat_kv(&mut gb, k_rope, spec.num_kv_heads, spec.head_dim, group);
            let v_rep = repeat_kv(&mut gb, v, spec.num_kv_heads, spec.head_dim, group);

            let attn_shape = shape::attention_shape(gb.shape(q_rope));
            let attn = gb.inner().mir(
                rlx_ir::ops::attention::attention_kind_op(
                    spec.num_heads,
                    spec.head_dim,
                    MaskKind::Bias,
                    spec.score_scale,
                    spec.attn_logit_softcap,
                ),
                vec![q_rope, k_rep, v_rep, bias],
                attn_shape,
            );
            Ok(Some(emit.wrap(attn, input.shape().clone())))
        },
    ))
}

fn gemma_prefill_layer_with_bias(
    layer_idx: usize,
    style: GemmaLayerStyle,
    attn: SelfAttnPrefillSpec,
    eps: f32,
    kv_sink: Option<std::sync::Arc<std::sync::Mutex<Vec<rlx_ir::HirNodeId>>>>,
) -> FlowStage {
    let prefix = format!("model.layers.{layer_idx}");
    let mut stack = LayerStack::named(format!("layer{layer_idx}"))
        .residual_save()
        .stage(FlowStage::GemmaRmsNorm(GemmaRmsNormStage::hf_layer(
            format!("{prefix}.input_layernorm"),
            eps,
        )));

    if let Some(sink) = kv_sink {
        let mut tap = GemmaKvTapStage::layer(layer_idx, attn.head_dim, eps, sink);
        if attn.n_rot != attn.head_dim {
            tap = tap.with_n_rot(attn.n_rot);
        }
        if let Some(name) = attn.rope_table.as_deref() {
            tap = tap.with_rope_table(name);
        }
        if attn.k_eq_v {
            tap = tap.with_k_eq_v();
        }
        stack = stack.stage(FlowStage::GemmaKvTap(tap));
    }

    stack = stack
        .stage(biased_self_attn_stage(attn.clone()))
        .linear(format!("{prefix}.self_attn.o_proj.weight"), true)
        .residual_add()
        .residual_save();

    stack = if matches!(
        style,
        GemmaLayerStyle::Gemma2 | GemmaLayerStyle::Gemma3 | GemmaLayerStyle::Gemma4
    ) {
        stack.stage(FlowStage::GemmaRmsNorm(GemmaRmsNormStage::hf_layer(
            format!("{prefix}.pre_feedforward_layernorm"),
            eps,
        )))
    } else {
        stack.stage(FlowStage::GemmaRmsNorm(GemmaRmsNormStage::hf_layer(
            format!("{prefix}.post_attention_layernorm"),
            eps,
        )))
    };

    stack = stack.stage(FlowStage::GeGlu(GeGluStage::hf_mlp(&prefix)));

    if matches!(
        style,
        GemmaLayerStyle::Gemma2 | GemmaLayerStyle::Gemma3 | GemmaLayerStyle::Gemma4
    ) {
        stack = stack.stage(FlowStage::GemmaRmsNorm(GemmaRmsNormStage::hf_layer(
            format!("{prefix}.post_feedforward_layernorm"),
            eps,
        )));
    }

    stack.residual_add().build()
}

/// Layer override: sliding layers use [`MaskKind::Bias`] + `attn_bias` input;
/// full-attention layers stay causal/sliding as usual.
pub fn multimodal_layer_override(ctx: GemmaLayerCtx<'_>, use_vision_bias: bool) -> FlowStage {
    if !use_vision_bias {
        return ctx.default_stage();
    }
    match ctx {
        GemmaLayerCtx::Prefill {
            index,
            style,
            attn,
            kv_sink,
            export_kv,
            eps,
            ..
        } => {
            let sink = if export_kv {
                Some(kv_sink.inner())
            } else {
                None
            };
            let mut attn = attn;
            if !matches!(attn.mask, MaskKind::Causal) {
                attn.mask = MaskKind::Bias;
                return gemma_prefill_layer_with_bias(index, style, attn, eps, sink);
            }
            gemma_prefill_layer_composed(index, style, attn, eps, sink)
        }
        GemmaLayerCtx::Decode { .. } => ctx.default_stage(),
    }
}
