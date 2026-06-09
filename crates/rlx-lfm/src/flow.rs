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

//! LFM2.5 decode-step LM topology.
//!
//! Per layer (single-step decode):
//!
//! ```text
//!   h_in [b, 1, hidden]
//!     ↓ RMSNorm
//!     ↓ x_proj = mm(Wx);  reshape [b, 1, c]
//!     ↓ b_proj = mm(Wb);  reshape [b, 1, n]
//!     ↓ c_proj = mm(Wc);  reshape [b, 1, n]
//!     ↓ gate_p = silu(mm(Wgate)); reshape [b, 1, c]
//!     ↓ a load_param → [c, n]
//!     ↓ state_in[layer] bound by runner
//!     ↓ LfmSsmStepStage → packed [b, c + c*n]
//!     ↓ split → y[b, c]  +  state_out[b, c, n]
//!     ↓ y reshape [b, 1, hidden]; o = mm(Wo); residual
//!     ↓ RMSNorm → SwiGLU → residual
//!   h_out [b, 1, hidden]
//! ```

use anyhow::Result;
use rlx_flow::FlowValue;
use rlx_flow::escape::Emit;
use rlx_ir::hir::HirMut;
use rlx_ir::{DType, HirGraphExt, Shape};
use rlx_ssm::LfmSsmStepStage;
use std::sync::{Arc, Mutex};

use super::config::LfmConfig;

fn mm_with_loaded(
    emit: &mut Emit<'_>,
    x: rlx_ir::HirNodeId,
    w_key: &str,
) -> Result<rlx_ir::HirNodeId> {
    let w = emit.load_param(w_key, true)?;
    let mut gb = HirMut::new(emit.hir());
    Ok(gb.mm(x, w))
}

pub fn lfm_decode_layer_plugin(
    cfg: LfmConfig,
    layer_idx: usize,
) -> impl Fn(&mut Emit<'_>, Option<FlowValue>) -> Result<Option<FlowValue>> + Send + Sync + 'static
{
    lfm_decode_layer_plugin_with_sink(cfg, layer_idx, None)
}

pub fn lfm_decode_layer_plugin_with_sink(
    cfg: LfmConfig,
    layer_idx: usize,
    state_out_sink: Option<Arc<Mutex<Vec<rlx_ir::HirNodeId>>>>,
) -> impl Fn(&mut Emit<'_>, Option<FlowValue>) -> Result<Option<FlowValue>> + Send + Sync + 'static
{
    let lp = format!("blk.{layer_idx}");
    let b = 1usize;
    let c = cfg.ssm_channels;
    let n = cfg.ssm_state_size;
    let hidden = cfg.hidden_size;
    let eps = cfg.rms_norm_eps as f32;

    move |emit, input| {
        let x = input.ok_or_else(|| anyhow::anyhow!("lfm layer requires input"))?;
        let in_shape = x.shape().clone();

        let attn_norm_w = emit.load_param(&format!("{lp}.attn_norm.weight"), false)?;
        let beta0 = emit.synth_zeros(&format!("{lp}.attn_norm.zero_beta"), hidden);
        let normed = {
            let mut gb = HirMut::new(emit.hir());
            gb.rms_norm(x.hir_id(), attn_norm_w, beta0, eps)
        };

        let x_proj = mm_with_loaded(emit, normed, &format!("{lp}.ssm_x.weight"))?;
        let b_proj = mm_with_loaded(emit, normed, &format!("{lp}.ssm_b.weight"))?;
        let cp_proj = mm_with_loaded(emit, normed, &format!("{lp}.ssm_c.weight"))?;
        let gate_p = mm_with_loaded(emit, normed, &format!("{lp}.ssm_gate.weight"))?;
        let a_param = emit.load_param(&format!("{lp}.ssm_a.weight"), false)?;

        let x_shape = Shape::new(&[b, 1, c], DType::F32);
        let g_shape = x_shape.clone();
        let bn_shape = Shape::new(&[b, 1, n], DType::F32);

        let x3;
        let b3;
        let c3;
        let gate3;
        {
            let mut gb = HirMut::new(emit.hir());
            x3 = gb.reshape_(x_proj, vec![b as i64, 1, c as i64]);
            b3 = gb.reshape_(b_proj, vec![b as i64, 1, n as i64]);
            c3 = gb.reshape_(cp_proj, vec![b as i64, 1, n as i64]);
            let gate_pre = gb.reshape_(gate_p, vec![b as i64, 1, c as i64]);
            gate3 = gb.activation(rlx_ir::op::Activation::Silu, gate_pre, g_shape.clone());
        }
        let _ = (x_shape, bn_shape);

        emit.state.named.insert("lfm.x".into(), x3);
        emit.state.named.insert("lfm.a".into(), a_param);
        emit.state.named.insert("lfm.b".into(), b3);
        emit.state.named.insert("lfm.c_proj".into(), c3);
        emit.state.named.insert("lfm.gate".into(), gate3);

        let state_in_key = format!("lfm.state_in_{layer_idx}");
        let state_id = emit.named(&state_in_key).map_err(|_| {
            anyhow::anyhow!(
                "lfm layer {layer_idx}: missing `{state_in_key}` — runner must bind \
                 per-layer state inputs before building the graph"
            )
        })?;
        emit.state.named.insert("lfm.state_in".into(), state_id);

        let step = LfmSsmStepStage::new(&lp, b, c, n);
        let packed = (step.plugin())(emit, None)?
            .ok_or_else(|| anyhow::anyhow!("LfmSsmStepStage produced no output"))?;

        let y_hidden;
        let state_out_2d;
        {
            let mut gb = HirMut::new(emit.hir());
            let y_slice = gb.narrow_(packed.hir_id(), 1, 0, c);
            let state_out = gb.narrow_(packed.hir_id(), 1, c, c * n);
            state_out_2d = gb.reshape_(state_out, vec![b as i64, c as i64, n as i64]);
            y_hidden = gb.reshape_(y_slice, vec![b as i64, 1, hidden as i64]);
        }
        emit.state
            .named
            .insert(format!("lfm.state_out_{layer_idx}"), state_out_2d);
        if let Some(sink) = &state_out_sink {
            sink.lock().expect("sink").push(state_out_2d);
        }

        let o_mm = mm_with_loaded(emit, y_hidden, &format!("{lp}.ssm_o.weight"))?;
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
