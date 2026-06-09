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

//! MiniMax M2 decode-step LM topology.
//!
//! Builds a single-step decode graph that consumes one token's hidden
//! state plus per-layer Lightning Attention state (`[b, h, n, n]`) and
//! emits next-token logits + updated state per layer.
//!
//! Per layer:
//!
//! ```text
//!   h_in [b, 1, hidden]
//!     ↓ RMSNorm(input_norm)
//!     ↓ q/k/v = mm(W*);  reshape [b, 1, h, n]
//!     ↓ gate     = mm(Wg); reshape [b, 1, h]
//!     ↓ beta     = sigmoid(mm(Wb)); reshape [b, 1, h]
//!     ↓ state_in[layer] bound by runner
//!     ↓ LightningAttentionStepStage → packed [b, h, n + n*n]
//!     ↓ split via Narrow → y[b, h, n] + state_out[b, h, n, n]
//!     ↓ state_out → registered under `minimax.state_out_{layer}`
//!     ↓ y reshape [b, 1, hidden]; o = mm(Wo); residual
//!     ↓ RMSNorm → SwiGLU(gate, up, down) → residual
//!   h_out [b, 1, hidden]
//! ```

use anyhow::Result;
use rlx_flow::FlowValue;
use rlx_flow::escape::Emit;
use rlx_ir::hir::HirMut;
use rlx_ir::{DType, HirGraphExt, Shape};
use rlx_ssm::LightningAttentionStepStage;
use std::sync::{Arc, Mutex};

use super::config::MiniMaxConfig;

fn mm_with_loaded(
    emit: &mut Emit<'_>,
    x: rlx_ir::HirNodeId,
    w_key: &str,
) -> Result<rlx_ir::HirNodeId> {
    let w = emit.load_param(w_key, true)?;
    let mut gb = HirMut::new(emit.hir());
    Ok(gb.mm(x, w))
}

pub fn minimax_decode_layer_plugin(
    cfg: MiniMaxConfig,
    layer_idx: usize,
) -> impl Fn(&mut Emit<'_>, Option<FlowValue>) -> Result<Option<FlowValue>> + Send + Sync + 'static
{
    minimax_decode_layer_plugin_with_sink(cfg, layer_idx, None)
}

/// Variant of [`minimax_decode_layer_plugin`] that also pushes the
/// per-layer state-out HirNodeId into a [`SideOutputs`]-shaped sink so
/// the runner can declare them as extra graph outputs after build.
pub fn minimax_decode_layer_plugin_with_sink(
    cfg: MiniMaxConfig,
    layer_idx: usize,
    state_out_sink: Option<Arc<Mutex<Vec<rlx_ir::HirNodeId>>>>,
) -> impl Fn(&mut Emit<'_>, Option<FlowValue>) -> Result<Option<FlowValue>> + Send + Sync + 'static
{
    let lp = format!("blk.{layer_idx}");
    let b = 1usize;
    let h = cfg.num_attention_heads;
    let n = cfg.head_dim;
    let hidden = cfg.hidden_size;
    let eps = cfg.rms_norm_eps as f32;

    move |emit, input| {
        let x = input.ok_or_else(|| anyhow::anyhow!("minimax layer requires input"))?;
        let in_shape = x.shape().clone();

        // --- Input RMSNorm ---
        let attn_norm_w = emit.load_param(&format!("{lp}.attn_norm.weight"), false)?;
        let beta0 = emit.synth_zeros(&format!("{lp}.attn_norm.zero_beta"), hidden);
        let normed = {
            let mut gb = HirMut::new(emit.hir());
            gb.rms_norm(x.hir_id(), attn_norm_w, beta0, eps)
        };

        // --- Q/K/V/gate/beta projections ---
        let q_mm = mm_with_loaded(emit, normed, &format!("{lp}.attn_q.weight"))?;
        let k_mm = mm_with_loaded(emit, normed, &format!("{lp}.attn_k.weight"))?;
        let v_mm = mm_with_loaded(emit, normed, &format!("{lp}.attn_v.weight"))?;
        let gate_mm = mm_with_loaded(emit, normed, &format!("{lp}.attn_gate.weight"))?;
        let beta_mm = mm_with_loaded(emit, normed, &format!("{lp}.attn_beta.weight"))?;

        let g_shape = Shape::new(&[b, 1, h], DType::F32);
        let q4;
        let k4;
        let v4;
        let g3;
        let bt3;
        {
            let mut gb = HirMut::new(emit.hir());
            q4 = gb.reshape_(q_mm, vec![b as i64, 1, h as i64, n as i64]);
            k4 = gb.reshape_(k_mm, vec![b as i64, 1, h as i64, n as i64]);
            v4 = gb.reshape_(v_mm, vec![b as i64, 1, h as i64, n as i64]);
            g3 = gb.reshape_(gate_mm, vec![b as i64, 1, h as i64]);
            let bt_pre = gb.reshape_(beta_mm, vec![b as i64, 1, h as i64]);
            bt3 = gb.activation(rlx_ir::op::Activation::Sigmoid, bt_pre, g_shape.clone());
        }
        emit.state.named.insert("lightning.q".into(), q4);
        emit.state.named.insert("lightning.k".into(), k4);
        emit.state.named.insert("lightning.v".into(), v4);
        emit.state.named.insert("lightning.gate".into(), g3);
        emit.state.named.insert("lightning.beta".into(), bt3);

        let state_in_key = format!("minimax.state_in_{layer_idx}");
        let state_id = emit.named(&state_in_key).map_err(|_| {
            anyhow::anyhow!(
                "minimax layer {layer_idx}: missing `{state_in_key}` — runner must bind \
                 per-layer state inputs before building the graph"
            )
        })?;
        emit.state
            .named
            .insert("lightning.state_in".into(), state_id);

        let step = LightningAttentionStepStage::new(&lp, b, h, n);
        let packed = (step.plugin())(emit, None)?
            .ok_or_else(|| anyhow::anyhow!("LightningAttentionStepStage produced no output"))?;

        let y_hidden;
        let state_out_3d;
        {
            let mut gb = HirMut::new(emit.hir());
            let y_slice = gb.narrow_(packed.hir_id(), 2, 0, n);
            let state_out = gb.narrow_(packed.hir_id(), 2, n, n * n);
            state_out_3d = gb.reshape_(state_out, vec![b as i64, h as i64, n as i64, n as i64]);
            y_hidden = gb.reshape_(y_slice, vec![b as i64, 1, hidden as i64]);
        }
        emit.state
            .named
            .insert(format!("minimax.state_out_{layer_idx}"), state_out_3d);
        if let Some(sink) = &state_out_sink {
            sink.lock().expect("sink").push(state_out_3d);
        }

        let o_mm = mm_with_loaded(emit, y_hidden, &format!("{lp}.attn_output.weight"))?;
        let after_attn = {
            let mut gb = HirMut::new(emit.hir());
            gb.add(x.hir_id(), o_mm)
        };

        let ffn_norm_w = emit.load_param(&format!("{lp}.ffn_norm.weight"), false)?;
        let ffn_beta = emit.synth_zeros(&format!("{lp}.ffn_norm.zero_beta"), hidden);
        let ffn_normed = {
            let mut gb = HirMut::new(emit.hir());
            gb.rms_norm(after_attn, ffn_norm_w, ffn_beta, eps)
        };
        let gate_w = emit.load_param(&format!("{lp}.ffn_gate.weight"), true)?;
        let up_w = emit.load_param(&format!("{lp}.ffn_up.weight"), true)?;
        let down_w = emit.load_param(&format!("{lp}.ffn_down.weight"), true)?;

        let h_out = {
            let mut gb = HirMut::new(emit.hir());
            let gate_p = gb.mm(ffn_normed, gate_w);
            let up_p = gb.mm(ffn_normed, up_w);
            let gate_act = gb.silu(gate_p);
            let prod = gb.mul(gate_act, up_p);
            let down_p = gb.mm(prod, down_w);
            gb.add(after_attn, down_p)
        };

        Ok(Some(FlowValue::new(h_out, in_shape)))
    }
}
