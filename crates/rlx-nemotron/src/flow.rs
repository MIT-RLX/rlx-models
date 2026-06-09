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

//! Nemotron-H per-layer decode-step blocks.
//!
//! Two layer flavors:
//!   * **Mamba2 layer** — projects hidden → x/dt_raw/b/c via linears,
//!     loads `a_log` and optional `d_skip`, runs `Mamba2StepStage`,
//!     splits packed `[y | state_out]`, projects `y` back, adds
//!     residual, then an RMSNorm + SwiGLU FFN.
//!   * **Attention layer** — standard pre-norm GQA attention with
//!     RoPE + SwiGLU FFN. (Stateless per token; KV cache is the
//!     concern of the runner, mirroring rlx-llama32's decode path.)
//!
//! Per-layer state buffers (Mamba2 only — attention layers have no
//! SSM state) are registered by the runner under
//! `nemotron.state_in_{layer}` and `nemotron.state_out_{layer}`.

use anyhow::Result;
use rlx_flow::{FlowValue, escape::Emit};
use rlx_ir::hir::HirMut;
use rlx_ir::{DType, HirGraphExt, Shape};
use rlx_ssm::Mamba2StepStage;
use std::sync::{Arc, Mutex};

use super::config::NemotronHybridConfig;

fn mm_with_loaded(
    emit: &mut Emit<'_>,
    x: rlx_ir::HirNodeId,
    w_key: &str,
) -> Result<rlx_ir::HirNodeId> {
    let w = emit.load_param(w_key, true)?;
    let mut gb = HirMut::new(emit.hir());
    Ok(gb.mm(x, w))
}

pub fn mamba2_decode_layer_plugin_with_sink(
    cfg: NemotronHybridConfig,
    layer_idx: usize,
    state_out_sink: Option<Arc<Mutex<Vec<rlx_ir::HirNodeId>>>>,
) -> impl Fn(&mut Emit<'_>, Option<FlowValue>) -> Result<Option<FlowValue>> + Send + Sync + 'static
{
    let lp = format!("blk.{layer_idx}");
    let b = 1usize;
    let h = cfg.mamba2_num_heads;
    let n = cfg.mamba2_state_size;
    let hidden = cfg.hidden_size;
    let eps = cfg.rms_norm_eps as f32;

    move |emit, input| {
        let x = input.ok_or_else(|| anyhow::anyhow!("nemotron mamba2 layer requires input"))?;
        let in_shape = x.shape().clone();

        let attn_norm_w = emit.load_param(&format!("{lp}.attn_norm.weight"), false)?;
        let beta0 = emit.synth_zeros(&format!("{lp}.attn_norm.zero_beta"), hidden);
        let normed = {
            let mut gb = HirMut::new(emit.hir());
            gb.rms_norm(x.hir_id(), attn_norm_w, beta0, eps)
        };

        // Projections: x_proj [hidden → h], dt_raw [hidden → h], b/c [hidden → n].
        let x_proj = mm_with_loaded(emit, normed, &format!("{lp}.ssm_x.weight"))?;
        let dt_proj = mm_with_loaded(emit, normed, &format!("{lp}.ssm_dt.weight"))?;
        let b_proj = mm_with_loaded(emit, normed, &format!("{lp}.ssm_b.weight"))?;
        let cp_proj = mm_with_loaded(emit, normed, &format!("{lp}.ssm_c.weight"))?;
        let a_log = emit.load_param(&format!("{lp}.ssm_a_log.weight"), false)?;
        // d_skip optional — synth a zero-length zero if absent. The
        // CPU kernel treats length==0 as "no D-skip".
        let d_key = format!("{lp}.ssm_d.weight");
        let d_id = emit.load_param(&d_key, false).unwrap_or_else(|_| {
            // Zero-length zeros: matches the kernel's "no D-skip" branch.
            emit.synth_zeros(&format!("{lp}.ssm_d.zero"), 0)
        });

        let x3;
        let dt3;
        let b3;
        let c3;
        {
            let mut gb = HirMut::new(emit.hir());
            x3 = gb.reshape_(x_proj, vec![b as i64, 1, h as i64]);
            dt3 = gb.reshape_(dt_proj, vec![b as i64, 1, h as i64]);
            b3 = gb.reshape_(b_proj, vec![b as i64, 1, n as i64]);
            c3 = gb.reshape_(cp_proj, vec![b as i64, 1, n as i64]);
        }

        emit.state.named.insert("mamba2.x".into(), x3);
        emit.state.named.insert("mamba2.dt_raw".into(), dt3);
        emit.state.named.insert("mamba2.a_log".into(), a_log);
        emit.state.named.insert("mamba2.b".into(), b3);
        emit.state.named.insert("mamba2.c_proj".into(), c3);
        emit.state.named.insert("mamba2.d_skip".into(), d_id);

        let state_in_key = format!("nemotron.state_in_{layer_idx}");
        let state_id = emit.named(&state_in_key).map_err(|_| {
            anyhow::anyhow!(
                "nemotron mamba2 layer {layer_idx}: missing `{state_in_key}` — \
                 runner must bind per-layer state inputs"
            )
        })?;
        emit.state.named.insert("mamba2.state_in".into(), state_id);

        let step = Mamba2StepStage::new(&lp, b, h, n);
        let packed = (step.plugin())(emit, None)?
            .ok_or_else(|| anyhow::anyhow!("Mamba2StepStage produced no output"))?;

        let y_hidden;
        let state_out_2d;
        {
            let mut gb = HirMut::new(emit.hir());
            // Packed shape: [b, h + h*n]. Split on axis 1.
            let y_slice = gb.narrow_(packed.hir_id(), 1, 0, h);
            let state_out = gb.narrow_(packed.hir_id(), 1, h, h * n);
            state_out_2d = gb.reshape_(state_out, vec![b as i64, h as i64, n as i64]);
            // y is [b, h]. Project back to hidden via ssm_o.
            y_hidden = gb.reshape_(y_slice, vec![b as i64, 1, h as i64]);
        }
        emit.state
            .named
            .insert(format!("nemotron.state_out_{layer_idx}"), state_out_2d);
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

/// Lightweight attention layer for Nemotron-H — pre-norm GQA, **no
/// KV cache** (stateless per-token). Suitable for short-context quick check
/// testing; for long-context production add a KV-cache extension
/// (analogous to rlx-llama32's decode path with `past_k_*`/`past_v_*`
/// inputs). Used in catalog quick-check tests; production deployments
/// should swap for an rlx-llama32-style cached attention block.
pub fn stateless_attention_layer_plugin(
    cfg: NemotronHybridConfig,
    layer_idx: usize,
) -> impl Fn(&mut Emit<'_>, Option<FlowValue>) -> Result<Option<FlowValue>> + Send + Sync + 'static
{
    let lp = format!("blk.{layer_idx}");
    let hidden = cfg.hidden_size;
    let nh = cfg.num_attention_heads;
    let dh = cfg.head_dim;
    let eps = cfg.rms_norm_eps as f32;
    let _kv_h = cfg.num_key_value_heads;

    move |emit, input| {
        let x = input.ok_or_else(|| anyhow::anyhow!("nemotron attn layer requires input"))?;
        let in_shape = x.shape().clone();

        let attn_norm_w = emit.load_param(&format!("{lp}.attn_norm.weight"), false)?;
        let beta0 = emit.synth_zeros(&format!("{lp}.attn_norm.zero_beta"), hidden);
        let normed = {
            let mut gb = HirMut::new(emit.hir());
            gb.rms_norm(x.hir_id(), attn_norm_w, beta0, eps)
        };

        let q_mm = mm_with_loaded(emit, normed, &format!("{lp}.attn_q.weight"))?;
        let k_mm = mm_with_loaded(emit, normed, &format!("{lp}.attn_k.weight"))?;
        let v_mm = mm_with_loaded(emit, normed, &format!("{lp}.attn_v.weight"))?;

        // Single-token decode with no past KV — pass rank-3 q/k/v
        // [1, 1, nh*dh] to `attention_` and let the IR shape inference
        // produce the matching output shape.
        let attn_out = {
            let mut gb = HirMut::new(emit.hir());
            let mask = gb.add_node(
                rlx_ir::op::Op::Constant { data: vec![0u8; 4] },
                vec![],
                Shape::new(&[1], DType::F32),
            );
            gb.attention_(q_mm, k_mm, v_mm, mask, nh, dh)
        };
        let o_mm = mm_with_loaded(emit, attn_out, &format!("{lp}.attn_output.weight"))?;
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
