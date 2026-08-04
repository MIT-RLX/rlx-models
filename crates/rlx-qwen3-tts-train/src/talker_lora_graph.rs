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

//! LoRA adapter graph on the Qwen3 talker (Q/K/V/O + gate/up/down).

use anyhow::Result;
use rlx_autodiff::grad_with_loss;
use rlx_compile::legalize_broadcast;
use rlx_ir::infer::GraphExt;
use rlx_ir::op::MaskKind;
use rlx_ir::{DType, Graph, NodeId, Op, Shape};
use rlx_qwen3::Qwen3Config;

#[derive(Debug, Clone)]
pub struct LoraParamSlot {
    pub name: String,
    pub param: NodeId,
    pub grad: Option<NodeId>,
}

#[derive(Debug)]
pub struct LoraTrainGraph {
    pub forward: Graph,
    pub backward: Graph,
    pub params: Vec<LoraParamSlot>,
    pub d_output: NodeId,
    pub loss: NodeId,
    pub rope_cos: Vec<f32>,
    pub rope_sin: Vec<f32>,
}

const ATTN_PROJS: [&str; 4] = ["q_proj", "k_proj", "v_proj", "o_proj"];
const FFN_PROJS: [&str; 3] = ["gate_proj", "up_proj", "down_proj"];

fn i64s(v: usize) -> i64 {
    v as i64
}

fn lora_delta(g: &mut Graph, hn: NodeId, lora_a: NodeId, lora_b: NodeId) -> NodeId {
    let lora_t = g.transpose_(lora_a, vec![1, 0]);
    let mid = g.mm(hn, lora_t);
    g.mm(mid, lora_b)
}

fn proj_with_lora(g: &mut Graph, hn: NodeId, w: NodeId, lora_a: NodeId, lora_b: NodeId) -> NodeId {
    let base = g.mm(hn, w);
    let delta = lora_delta(g, hn, lora_a, lora_b);
    g.add(base, delta)
}

fn push_lora_param(params: &mut Vec<LoraParamSlot>, li: usize, proj: &str, a: NodeId, b: NodeId) {
    params.push(LoraParamSlot {
        name: format!("lora.{li}.{proj}_a"),
        param: a,
        grad: None,
    });
    params.push(LoraParamSlot {
        name: format!("lora.{li}.{proj}_b"),
        param: b,
        grad: None,
    });
}

fn ffn_lora_shapes(proj: &str, rank: usize, h: usize, ffn: usize) -> (Vec<usize>, Vec<usize>) {
    match proj {
        "gate_proj" | "up_proj" => (vec![rank, h], vec![rank, ffn]),
        "down_proj" => (vec![rank, ffn], vec![rank, h]),
        _ => (vec![rank, h], vec![rank, h]),
    }
}

fn repeat_kv(
    g: &mut Graph,
    x: NodeId,
    num_kv_heads: usize,
    head_dim: usize,
    group: usize,
) -> NodeId {
    if group == 1 {
        return x;
    }
    let last_ax = g.shape(x).rank() - 1;
    let mut pieces: Vec<NodeId> = Vec::with_capacity(num_kv_heads * group);
    for h in 0..num_kv_heads {
        let slice = g.narrow_(x, last_ax, h * head_dim, head_dim);
        for _ in 0..group {
            pieces.push(slice);
        }
    }
    g.concat_(pieces, last_ax)
}

fn build_rope_tables(theta: f64, head_dim: usize, max_pos: usize) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;
    let mut inv = Vec::with_capacity(half);
    for i in 0..half {
        inv.push(1.0 / theta.powf(2.0 * i as f64 / head_dim as f64));
    }
    let mut cos = vec![0f32; max_pos * half];
    let mut sin = vec![0f32; max_pos * half];
    for p in 0..max_pos {
        for i in 0..half {
            let angle = p as f64 * inv[i];
            cos[p * half + i] = angle.cos() as f32;
            sin[p * half + i] = angle.sin() as f32;
        }
    }
    (cos, sin)
}

pub fn build_talker_lora_graph(
    cfg: &Qwen3Config,
    seq: usize,
    rank: usize,
    n_layers: usize,
) -> Result<LoraTrainGraph> {
    let f = DType::F32;
    let h = cfg.hidden_size;
    let ffn = cfg.intermediate_size;
    let start_layer = cfg.num_hidden_layers.saturating_sub(n_layers);
    let layers = n_layers.min(cfg.num_hidden_layers);
    let dh = cfg.head_dim;
    let q_dim = cfg.q_proj_dim();
    let kv_dim = cfg.kv_proj_dim();
    let eps = cfg.rms_norm_eps as f32;

    let mut g = Graph::new("qwen3_tts_talker_lora");
    let mut params = Vec::new();
    let inputs = g.input("inputs_embeds", Shape::new(&[seq, h], f));
    let target = g.input("target_embeds", Shape::new(&[seq, h], f));
    let beta = g.param("__zero", Shape::new(&[h], f));
    let (rope_cos, rope_sin) = build_rope_tables(cfg.rope_theta, dh, cfg.max_position_embeddings);
    let half = dh / 2;
    let rope_cos_id = g.param(
        "rope.cos",
        Shape::new(&[cfg.max_position_embeddings, half], f),
    );
    let rope_sin_id = g.param(
        "rope.sin",
        Shape::new(&[cfg.max_position_embeddings, half], f),
    );

    let mut x = inputs;
    for li in start_layer..start_layer + layers {
        let prefix = format!("model.layers.{li}");
        let attn_norm = g.param(
            format!("{prefix}.input_layernorm.weight"),
            Shape::new(&[h], f),
        );
        let wq = g.param(
            format!("{prefix}.self_attn.q_proj.weight"),
            Shape::new(&[h, q_dim], f),
        );
        let wk = g.param(
            format!("{prefix}.self_attn.k_proj.weight"),
            Shape::new(&[h, kv_dim], f),
        );
        let wv = g.param(
            format!("{prefix}.self_attn.v_proj.weight"),
            Shape::new(&[h, kv_dim], f),
        );
        let wo = g.param(
            format!("{prefix}.self_attn.o_proj.weight"),
            Shape::new(&[q_dim, h], f),
        );

        let mut attn_lora = Vec::new();
        for proj in ATTN_PROJS {
            let (a_shape, b_shape): ([usize; 2], [usize; 2]) = match proj {
                "q_proj" => ([rank, h], [rank, q_dim]),
                "k_proj" | "v_proj" => ([rank, h], [rank, kv_dim]),
                "o_proj" => ([rank, q_dim], [rank, h]),
                _ => ([rank, h], [rank, h]),
            };
            let a = g.param(format!("lora.{li}.{proj}_a"), Shape::new(&a_shape, f));
            let b = g.param(format!("lora.{li}.{proj}_b"), Shape::new(&b_shape, f));
            push_lora_param(&mut params, li, proj, a, b);
            attn_lora.push((proj, a, b));
        }

        let hn = g.rms_norm(x, attn_norm, beta, eps);
        let q = proj_with_lora(&mut g, hn, wq, attn_lora[0].1, attn_lora[0].2);
        let k = proj_with_lora(&mut g, hn, wk, attn_lora[1].1, attn_lora[1].2);
        let v = proj_with_lora(&mut g, hn, wv, attn_lora[2].1, attn_lora[2].2);

        let q3 = g.reshape_(q, vec![1, i64s(seq), i64s(q_dim)]);
        let k3 = g.reshape_(k, vec![1, i64s(seq), i64s(kv_dim)]);
        let v3 = g.reshape_(v, vec![1, i64s(seq), i64s(kv_dim)]);
        let q3 = g.rope(q3, rope_cos_id, rope_sin_id, dh);
        let k3 = g.rope(k3, rope_cos_id, rope_sin_id, dh);
        let kv_group = cfg.num_attention_heads / cfg.num_key_value_heads;
        let k_rep = repeat_kv(&mut g, k3, cfg.num_key_value_heads, dh, kv_group);
        let v_rep = repeat_kv(&mut g, v3, cfg.num_key_value_heads, dh, kv_group);
        let attn_s = rlx_ir::shape::attention_shape(g.shape(q3));
        let attn = g.add_node(
            Op::Attention {
                num_heads: cfg.num_attention_heads,
                head_dim: dh,
                v_head_dim: None,
                mask_kind: MaskKind::Causal,
                score_scale: None,
                attn_logit_softcap: None,
            },
            vec![q3, k_rep, v_rep],
            attn_s,
        );
        let attn2 = g.reshape_(attn, vec![i64s(seq), i64s(q_dim)]);
        let attn_out = proj_with_lora(&mut g, attn2, wo, attn_lora[3].1, attn_lora[3].2);
        x = g.add(x, attn_out);

        let ffn_norm = g.param(
            format!("{prefix}.post_attention_layernorm.weight"),
            Shape::new(&[h], f),
        );
        let gate_w = g.param(
            format!("{prefix}.mlp.gate_proj.weight"),
            Shape::new(&[h, ffn], f),
        );
        let up_w = g.param(
            format!("{prefix}.mlp.up_proj.weight"),
            Shape::new(&[h, ffn], f),
        );
        let down_w = g.param(
            format!("{prefix}.mlp.down_proj.weight"),
            Shape::new(&[ffn, h], f),
        );

        let mut ffn_lora = Vec::new();
        for proj in FFN_PROJS {
            let (a_shape, b_shape) = ffn_lora_shapes(proj, rank, h, ffn);
            let a = g.param(
                format!("lora.{li}.{proj}_a"),
                Shape::new(a_shape.as_slice(), f),
            );
            let b = g.param(
                format!("lora.{li}.{proj}_b"),
                Shape::new(b_shape.as_slice(), f),
            );
            push_lora_param(&mut params, li, proj, a, b);
            ffn_lora.push((proj, a, b));
        }

        let hn2 = g.rms_norm(x, ffn_norm, beta, eps);
        let gate = proj_with_lora(&mut g, hn2, gate_w, ffn_lora[0].1, ffn_lora[0].2);
        let up = proj_with_lora(&mut g, hn2, up_w, ffn_lora[1].1, ffn_lora[1].2);
        let gate_act = g.silu(gate);
        let prod = g.mul(gate_act, up);
        let ff = proj_with_lora(&mut g, prod, down_w, ffn_lora[2].1, ffn_lora[2].2);
        x = g.add(x, ff);
    }

    let diff = g.sub(x, target);
    let sq = g.mul(diff, diff);
    let flat = g.reshape_(sq, vec![i64s(seq * h)]);
    let loss = g.mean(flat, vec![0], false);
    g.set_outputs(vec![loss]);
    let loss_node = loss;

    let (g, remap) = legalize_broadcast::run_with_remap(g);
    let mut params: Vec<LoraParamSlot> = params
        .into_iter()
        .map(|mut p| {
            p.param = remap[&p.param];
            p
        })
        .collect();
    let wrt: Vec<NodeId> = params.iter().map(|p| p.param).collect();
    let bwd = grad_with_loss(&g, &wrt);
    let d_output = bwd
        .nodes()
        .iter()
        .find(|n| matches!(&n.op, Op::Input { name } if name == "d_output"))
        .map(|n| n.id)
        .expect("d_output");
    let grad_ids: Vec<NodeId> = bwd.outputs[1..=params.len()].to_vec();
    for (slot, grad) in params.iter_mut().zip(grad_ids) {
        slot.grad = Some(grad);
    }

    Ok(LoraTrainGraph {
        forward: g,
        backward: bwd,
        params,
        d_output,
        loss: remap[&loss_node],
        rope_cos,
        rope_sin,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_qwen3::Qwen3Config;

    fn talker_qwen3() -> Qwen3Config {
        Qwen3Config {
            vocab_size: 3072,
            hidden_size: 1024,
            intermediate_size: 3072,
            num_hidden_layers: 28,
            num_attention_heads: 16,
            num_key_value_heads: 8,
            head_dim: 128,
            max_position_embeddings: 32768,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            hidden_act: "silu".into(),
            tie_word_embeddings: false,
            attention_bias: false,
            qk_norm: true,
            sliding_window: None,
            max_window_layers: 28,
            use_sliding_window: false,
            num_experts: 0,
            num_experts_used: 0,
            expert_ffn_size: 0,
            shared_expert_ffn_size: 0,
            expert_weights_scale: 1.0,
        }
    }

    #[test]
    fn talker_lora_graph_builds_one_layer() {
        build_talker_lora_graph(&talker_qwen3(), 64, 8, 1).expect("graph");
    }
}
