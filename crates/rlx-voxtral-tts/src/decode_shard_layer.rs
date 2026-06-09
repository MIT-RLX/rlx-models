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

//! Decode layer for wgpu LM shards — global checkpoint keys, local `past_k_*` inputs.

use rlx_flow::blocks::{CustomStage, LlamaDecodeLayerSpec};
use rlx_flow::{FlowStage, SideOutputs};
use rlx_ir::HirGraphExt;
use rlx_ir::hir::HirMut;
use rlx_ir::op::MaskKind;
use rlx_ir::shape;
use std::sync::{Arc, Mutex};

pub fn tts_decode_shard_layer(
    global_layer: usize,
    local_kv: usize,
    spec: LlamaDecodeLayerSpec,
    kv_out: Arc<Mutex<Vec<rlx_ir::HirNodeId>>>,
) -> FlowStage {
    FlowStage::Custom(CustomStage::named(
        format!("layer{global_layer}"),
        move |emit, input| {
            let input =
                input.ok_or_else(|| anyhow::anyhow!("decode shard layer requires input"))?;
            let decode =
                emit.state.decode.clone().ok_or_else(|| {
                    anyhow::anyhow!("decode shard layer requires BindDecodeInputs")
                })?;
            let zero_beta = emit
                .state
                .zero_beta
                .ok_or_else(|| anyhow::anyhow!("decode shard layer requires ZeroBeta"))?;

            let lp = format!("model.layers.{global_layer}");
            let in_ln_g = emit.load_param(&format!("{lp}.input_layernorm.weight"), false)?;
            let q_w = emit.load_param(&format!("{lp}.self_attn.q_proj.weight"), true)?;
            let k_w = emit.load_param(&format!("{lp}.self_attn.k_proj.weight"), true)?;
            let v_w = emit.load_param(&format!("{lp}.self_attn.v_proj.weight"), true)?;
            let o_w = emit.load_param(&format!("{lp}.self_attn.o_proj.weight"), true)?;
            let post_ln_g =
                emit.load_param(&format!("{lp}.post_attention_layernorm.weight"), false)?;
            let gate_w = emit.load_param(&format!("{lp}.mlp.gate_proj.weight"), true)?;
            let up_w = emit.load_param(&format!("{lp}.mlp.up_proj.weight"), true)?;
            let down_w = emit.load_param(&format!("{lp}.mlp.down_proj.weight"), true)?;

            let past_k = decode.past_k[local_kv];
            let past_v = decode.past_v[local_kv];

            let input_id = input.hir_id();
            let mut gb = HirMut::new(emit.hir());
            let normed_in = gb.rms_norm(input_id, in_ln_g, zero_beta, spec.eps);
            let q = gb.mm(normed_in, q_w);
            let k = gb.mm(normed_in, k_w);
            let v = gb.mm(normed_in, v_w);

            let q_rope = gb.rope(q, decode.cos, decode.sin, spec.head_dim);
            let k_rope = gb.rope(k, decode.cos, decode.sin, spec.head_dim);

            let new_k = gb.concat_(vec![past_k, k_rope], 1);
            let new_v = gb.concat_(vec![past_v, v], 1);
            kv_out.lock().expect("kv out").push(new_k);
            kv_out.lock().expect("kv out").push(new_v);

            let k_rep = repeat_kv(
                &mut gb,
                new_k,
                spec.num_kv_heads,
                spec.head_dim,
                spec.kv_group_size,
            );
            let v_rep = repeat_kv(
                &mut gb,
                new_v,
                spec.num_kv_heads,
                spec.head_dim,
                spec.kv_group_size,
            );

            let attn_shape = shape::attention_shape(gb.shape(q_rope));
            let attn = if spec.use_custom_mask {
                let mask = decode
                    .mask
                    .ok_or_else(|| anyhow::anyhow!("custom mask requested but not bound"))?;
                gb.attention(
                    q_rope,
                    k_rep,
                    v_rep,
                    mask,
                    spec.num_heads,
                    spec.head_dim,
                    attn_shape,
                )
            } else {
                gb.attention_kind(
                    q_rope,
                    k_rep,
                    v_rep,
                    spec.num_heads,
                    spec.head_dim,
                    MaskKind::Causal,
                    attn_shape,
                )
            };

            let attn_out = gb.mm(attn, o_w);
            let post_attn = gb.add(input_id, attn_out);
            let normed_post = gb.rms_norm(post_attn, post_ln_g, zero_beta, spec.eps);
            let gate = gb.mm(normed_post, gate_w);
            let up = gb.mm(normed_post, up_w);
            let gate_act = gb.silu(gate);
            let swiglu = gb.mul(gate_act, up);
            let ffn_out = gb.mm(swiglu, down_w);
            let h_id = gb.add(post_attn, ffn_out);

            Ok(Some(emit.wrap(h_id, spec.hidden_shape.clone())))
        },
    ))
}

pub fn tts_decode_shard_layer_from_sink(
    global_layer: usize,
    local_kv: usize,
    spec: LlamaDecodeLayerSpec,
    sink: &SideOutputs,
) -> FlowStage {
    tts_decode_shard_layer(global_layer, local_kv, spec, sink.inner())
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shard_layer_names_global_index() {
        let spec = LlamaDecodeLayerSpec {
            num_heads: 8,
            head_dim: 128,
            num_kv_heads: 8,
            kv_group_size: 1,
            eps: 1e-5,
            use_custom_mask: true,
            hidden_shape: rlx_ir::Shape::new(&[1, 1, 3072], rlx_ir::DType::F32),
        };
        let sink = SideOutputs::new();
        let stage = tts_decode_shard_layer_from_sink(12, 0, spec, &sink);
        assert!(matches!(stage, FlowStage::Custom(_)));
    }
}
