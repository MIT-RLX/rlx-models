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

//! mllama cross-attention decoder layer as a [`FlowStage`].
//!
//! Returned from `Llama32Flow::layer()` for indices in
//! `text_config.cross_attention_layers`. Unlike a self-attention layer, Q comes
//! from the text hidden state while K/V come from the (constant) vision
//! `cross_states`, there is **no RoPE**, and the attention / MLP sub-block
//! outputs are scaled by `tanh(cross_attn_attn_gate)` / `tanh(cross_attn_mlp_gate)`
//! before the residual add (`MllamaCrossAttentionDecoderLayer`):
//! ```text
//!   h = x + tanh(attn_gate) · o_proj( xattn( q_norm(q_proj(rms(x))),
//!                                            k_norm(k_proj(cross)), v_proj(cross) ) )
//!   h = h + tanh(mlp_gate)  · down( silu(gate(rms(h))) · up(rms(h)) )
//! ```
//!
//! The `cross_states` `[1, kv_seq, hidden]` (vision features projected into the
//! text embedding space) are read from a shared graph input named
//! [`CROSS_STATES_INPUT`], declared on the flow by the caller (e.g. via
//! `Llama32Flow::patch_flow`). This keeps the compiled text graph independent of
//! the image, so the 11B weights are consumed once and only the fed value
//! changes per image.

use anyhow::{Result, anyhow};
use rlx_flow::blocks::CustomStage;
use rlx_flow::{Emit, FlowStage};
use rlx_ir::hir::HirMut;
use rlx_ir::op::MaskKind;
use rlx_ir::{DType, HirGraphExt, HirNodeId, Shape};

/// Name of the shared vision-features graph input the cross-attention layers read.
pub const CROSS_STATES_INPUT: &str = "cross_states";

/// Static dimensions a cross-attention layer needs, matching the text config.
#[derive(Debug, Clone, Copy)]
pub struct CrossAttnDims {
    pub hidden: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub eps: f32,
    /// Text (query) sequence length the graph is compiled for.
    pub text_seq: usize,
    /// Vision key/value sequence length (`num_tiles * num_patches`).
    pub kv_seq: usize,
}

/// `x @ Wᵀ` under `{prefix}.weight` (HF `[out,in]`, no bias).
fn linear(emit: &mut Emit<'_>, prefix: &str, x: HirNodeId) -> Result<HirNodeId> {
    let w = emit.load_param(&format!("{prefix}.weight"), true)?;
    let mut gb = HirMut::new(emit.hir());
    Ok(gb.mm(x, w))
}

/// GQA key/value expansion: repeat each of `num_kv_heads` head-slices `group`
/// times along the feature axis (repeat-interleave), matching HF `repeat_kv`.
fn repeat_kv(
    gb: &mut HirMut<'_>,
    x: HirNodeId,
    num_kv_heads: usize,
    head_dim: usize,
    group: usize,
) -> HirNodeId {
    if group == 1 {
        return x;
    }
    let last_ax = gb.shape(x).rank() - 1;
    let mut pieces: Vec<HirNodeId> = Vec::with_capacity(num_kv_heads * group);
    for h in 0..num_kv_heads {
        let slice = gb.narrow_(x, last_ax, h * head_dim, head_dim);
        for _ in 0..group {
            pieces.push(slice);
        }
    }
    gb.concat_(pieces, last_ax)
}

/// Build the cross-attention decoder layer for `model.layers.{weight_index}`.
/// Reads the shared vision features from the [`CROSS_STATES_INPUT`] graph input.
pub fn cross_attn_stage(weight_index: usize, d: CrossAttnDims) -> FlowStage {
    let f = DType::F32;
    FlowStage::Custom(CustomStage::named(
        format!("mllama.cross_attn.{weight_index}"),
        move |emit, prev| {
            let x_fv = prev.ok_or_else(|| anyhow!("cross-attn layer requires a hidden input"))?;
            let x_shape = x_fv.shape.clone();
            let x = x_fv.hir_id();
            let prefix = format!("model.layers.{weight_index}");
            let group = d.num_heads / d.num_kv_heads;
            let kv_dim = d.num_kv_heads * d.head_dim;

            // Vision K/V source, fed per-image as a shared graph input.
            let cross = emit.flow_input(CROSS_STATES_INPUT)?.hir_id();
            // Weight-only RMSNorm needs a zero bias; two widths are reused.
            let zb_h = emit.synth_param(
                &format!("mllama.cross{weight_index}.zb_h"),
                vec![0.0; d.hidden],
                Shape::new(&[d.hidden], f),
            );
            let zb_hd = emit.synth_param(
                &format!("mllama.cross{weight_index}.zb_hd"),
                vec![0.0; d.head_dim],
                Shape::new(&[d.head_dim], f),
            );

            // --- Cross-attention sub-block ---
            let normed = {
                let g = emit.load_param(&format!("{prefix}.input_layernorm.weight"), false)?;
                let mut gb = HirMut::new(emit.hir());
                gb.rms_norm(x, g, zb_h, d.eps)
            };
            let q = linear(emit, &format!("{prefix}.cross_attn.q_proj"), normed)?;
            let q = {
                let gn = emit.load_param(&format!("{prefix}.cross_attn.q_norm.weight"), false)?;
                let mut gb = HirMut::new(emit.hir());
                let qr = gb.reshape_(
                    q,
                    vec![1, (d.text_seq * d.num_heads) as i64, d.head_dim as i64],
                );
                let qn = gb.rms_norm(qr, gn, zb_hd, d.eps);
                gb.reshape_(qn, vec![1, d.text_seq as i64, d.hidden as i64])
            };
            let k = linear(emit, &format!("{prefix}.cross_attn.k_proj"), cross)?;
            let k = {
                let gn = emit.load_param(&format!("{prefix}.cross_attn.k_norm.weight"), false)?;
                let mut gb = HirMut::new(emit.hir());
                let kr = gb.reshape_(
                    k,
                    vec![1, (d.kv_seq * d.num_kv_heads) as i64, d.head_dim as i64],
                );
                let kn = gb.rms_norm(kr, gn, zb_hd, d.eps);
                gb.reshape_(kn, vec![1, d.kv_seq as i64, kv_dim as i64])
            };
            let v = linear(emit, &format!("{prefix}.cross_attn.v_proj"), cross)?;
            let attn = {
                let mut gb = HirMut::new(emit.hir());
                let k_rep = repeat_kv(&mut gb, k, d.num_kv_heads, d.head_dim, group);
                let v_rep = repeat_kv(&mut gb, v, d.num_kv_heads, d.head_dim, group);
                let shape = Shape::new(&[1, d.text_seq, d.hidden], f);
                gb.attention_kind(
                    q,
                    k_rep,
                    v_rep,
                    d.num_heads,
                    d.head_dim,
                    MaskKind::None,
                    shape,
                )
            };
            let attn = linear(emit, &format!("{prefix}.cross_attn.o_proj"), attn)?;
            let x = {
                let gate = emit.load_param(&format!("{prefix}.cross_attn_attn_gate"), false)?;
                let mut gb = HirMut::new(emit.hir());
                let g = gb.tanh(gate);
                let gated = gb.mul(attn, g);
                gb.add(x, gated)
            };

            // --- MLP sub-block (SwiGLU), tanh-gated ---
            let normed = {
                let g =
                    emit.load_param(&format!("{prefix}.post_attention_layernorm.weight"), false)?;
                let mut gb = HirMut::new(emit.hir());
                gb.rms_norm(x, g, zb_h, d.eps)
            };
            let gate_p = linear(emit, &format!("{prefix}.mlp.gate_proj"), normed)?;
            let up_p = linear(emit, &format!("{prefix}.mlp.up_proj"), normed)?;
            let swiglu = {
                let mut gb = HirMut::new(emit.hir());
                let a = gb.silu(gate_p);
                gb.mul(a, up_p)
            };
            let down = linear(emit, &format!("{prefix}.mlp.down_proj"), swiglu)?;
            let out = {
                let gate = emit.load_param(&format!("{prefix}.cross_attn_mlp_gate"), false)?;
                let mut gb = HirMut::new(emit.hir());
                let g = gb.tanh(gate);
                let gated = gb.mul(down, g);
                gb.add(x, gated)
            };

            Ok(Some(emit.wrap(out, x_shape)))
        },
    ))
}
