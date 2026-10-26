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

use rlx_flow::FlowStage;
use rlx_flow::blocks::gemma_prefill_layer_composed;
use rlx_ir::op::MaskKind;

use crate::flow::GemmaLayerCtx;

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
            // The shared layer, with the `attn_bias` input as the mask on
            // sliding layers — same norms, KV tap and residuals as text-only.
            let attn = if matches!(attn.mask, MaskKind::Causal) {
                attn
            } else {
                attn.with_bias_input("attn_bias")
            };
            gemma_prefill_layer_composed(index, style, attn, eps, sink)
        }
        GemmaLayerCtx::Decode { .. } => ctx.default_stage(),
    }
}
