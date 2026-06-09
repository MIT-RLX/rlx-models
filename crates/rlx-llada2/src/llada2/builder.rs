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

// RLX — LLaDA2 MoE forward graph (block diffusion, bidirectional attention).

use crate::config::LLaDA2MoeConfig;
use crate::weights::{LLaDA2Weights, LayerFfn};
use anyhow::{Result, anyhow};
use rlx_ir::infer::GraphExt;
use rlx_ir::{DType, Graph, NodeId, Op, Shape};
use std::collections::HashMap;

fn param(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    name: &str,
    data: &[f32],
    shape: &[usize],
) -> NodeId {
    let id = g.param(name, Shape::new(shape, DType::F32));
    params.insert(name.to_string(), data.to_vec());
    id
}

fn synth_zero(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    name: &str,
    len: usize,
) -> NodeId {
    param(g, params, name, &vec![0f32; len], &[len])
}

fn rms_norm_layer(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    x: NodeId,
    gamma_name: &str,
    gamma: &[f32],
    eps: f32,
) -> NodeId {
    let n = gamma.len();
    let w = param(g, params, gamma_name, gamma, &[n]);
    let b = synth_zero(g, params, &format!("{gamma_name}.beta"), n);
    g.rms_norm(x, w, b, eps)
}

fn split_qkv(
    g: &mut Graph,
    qkv: NodeId,
    batch: usize,
    seq: usize,
    n_head: usize,
    n_kv: usize,
    head_dim: usize,
) -> (NodeId, NodeId, NodeId) {
    let q_dim = n_head * head_dim;
    let kv_dim = n_kv * head_dim;
    let last = g.shape(qkv).rank() - 1;
    let q = g.narrow_(qkv, last, 0, q_dim);
    let k = g.narrow_(qkv, last, q_dim, kv_dim);
    let v = g.narrow_(qkv, last, q_dim + kv_dim, kv_dim);
    let q3 = g.reshape_(q, vec![batch as i64, seq as i64, q_dim as i64]);
    let k3 = g.reshape_(k, vec![batch as i64, seq as i64, kv_dim as i64]);
    let v3 = g.reshape_(v, vec![batch as i64, seq as i64, kv_dim as i64]);
    (q3, k3, v3)
}

fn head_rms_norm(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    x: NodeId,
    gamma: &[f32],
    name: &str,
    batch: usize,
    seq: usize,
    heads: usize,
    head_dim: usize,
    eps: f32,
) -> NodeId {
    let flat = (batch * seq * heads) as i64;
    let dh = head_dim as i64;
    let r = g.reshape_(x, vec![flat, dh]);
    let w = param(g, params, name, gamma, &[head_dim]);
    let b = synth_zero(g, params, &format!("{name}.beta"), head_dim);
    let n = g.rms_norm(r, w, b, eps);
    g.reshape_(n, vec![batch as i64, seq as i64, (heads * head_dim) as i64])
}

fn to_bhsd(
    g: &mut Graph,
    x: NodeId,
    batch: usize,
    seq: usize,
    heads: usize,
    head_dim: usize,
) -> NodeId {
    let x4 = g.reshape_(
        x,
        vec![batch as i64, seq as i64, heads as i64, head_dim as i64],
    );
    g.transpose_(x4, vec![0, 2, 1, 3])
}

fn expand_to(g: &mut Graph, x: NodeId, target: &[i64]) -> NodeId {
    let out = Shape::new(
        &target.iter().map(|&d| d as usize).collect::<Vec<_>>(),
        g.shape(x).dtype(),
    );
    g.add_node(
        Op::Expand {
            target_shape: target.to_vec(),
        },
        vec![x],
        out,
    )
}

fn repeat_kv_bhsd(g: &mut Graph, x: NodeId, num_kv_heads: usize, group: usize) -> NodeId {
    if group == 1 {
        return x;
    }
    // Expand along a synthetic GQA axis — MLX `concat` with repeated
    // narrow slices of the same NodeId can inflate the sequence axis.
    let sh = g.shape(x);
    let b = sh.dim(0).unwrap_static() as i64;
    let s = sh.dim(2).unwrap_static() as i64;
    let d = sh.dim(3).unwrap_static() as i64;
    let x5 = g.reshape_(x, vec![b, num_kv_heads as i64, 1, s, d]);
    let x6 = expand_to(g, x5, &[b, num_kv_heads as i64, group as i64, s, d]);
    g.reshape_(x6, vec![b, (num_kv_heads * group) as i64, s, d])
}

fn gather_rope(
    g: &mut Graph,
    table: NodeId,
    position_ids: NodeId,
    _batch: usize,
    seq: usize,
    tab_half: usize,
) -> NodeId {
    let gathered = g.gather_(table, position_ids, 0);
    // `[seq, half]` — matches MLX `Op::Rope` and broadcasts on CPU/Metal/CUDA.
    g.reshape_(gathered, vec![seq as i64, tab_half as i64])
}

fn build_dense_ffn(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    h_in: NodeId,
    il: usize,
    gate: &[f32],
    up: &[f32],
    down: &[f32],
    n_embd: usize,
    n_ff: usize,
    batch: usize,
    seq: usize,
) -> NodeId {
    let rows = batch * seq;
    let h2 = g.reshape_(h_in, vec![rows as i64, n_embd as i64]);
    let gate_w = param(
        g,
        params,
        &layer_key(il, "mlp.gate_proj.weight"),
        gate,
        &[n_embd, n_ff],
    );
    let up_w = param(
        g,
        params,
        &layer_key(il, "mlp.up_proj.weight"),
        up,
        &[n_embd, n_ff],
    );
    let down_w = param(
        g,
        params,
        &layer_key(il, "mlp.down_proj.weight"),
        down,
        &[n_ff, n_embd],
    );
    let g_proj = g.mm(h2, gate_w);
    let u_proj = g.mm(h2, up_w);
    let act = g.silu(g_proj);
    let swiglu = g.mul(act, u_proj);
    let out = g.mm(swiglu, down_w);
    let out3 = g.reshape_(out, vec![batch as i64, seq as i64, n_embd as i64]);
    g.add(h_in, out3)
}

fn build_moe_ffn(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    cfg: &LLaDA2MoeConfig,
    il: usize,
    h_in: NodeId,
    moe: &crate::weights::MoeLayerWeights,
    batch: usize,
    seq: usize,
) -> Result<NodeId> {
    let n_embd = cfg.hidden_size;
    let n_ff = cfg.expert_ffn_dim();
    let n_expert = cfg.num_experts;
    let top_k = cfg.num_experts_per_tok;
    let rows = batch * seq;

    let h2 = g.reshape_(h_in, vec![rows as i64, n_embd as i64]);
    let router_w = param(
        g,
        params,
        &layer_key(il, "mlp.gate.weight"),
        &moe.router,
        &[n_embd, n_expert],
    );
    let bias = param(
        g,
        params,
        &layer_key(il, "mlp.gate.expert_bias"),
        &moe.expert_bias,
        &[n_expert],
    );
    let (top_idx, top_probs) =
        crate::gate::emit_group_limited_gate(g, h2, router_w, bias, cfg, rows);

    let gate_w = param(
        g,
        params,
        &layer_key(il, "mlp.gate_exps.weight"),
        &moe.gate_exps,
        &[n_expert, n_embd, n_ff],
    );
    let up_w = param(
        g,
        params,
        &layer_key(il, "mlp.up_exps.weight"),
        &moe.up_exps,
        &[n_expert, n_embd, n_ff],
    );
    let down_w = param(
        g,
        params,
        &layer_key(il, "mlp.down_exps.weight"),
        &moe.down_exps,
        &[n_expert, n_ff, n_embd],
    );

    let mut acc: Option<NodeId> = None;
    for ki in 0..top_k {
        let expert_col = g.narrow_(top_idx, 1, ki, 1);
        // MLX `gather_mm` rhs_indices must be flat `[M]` (see rlx_mlx_shim.h).
        let expert_idx = g.reshape_(expert_col, vec![rows as i64]);
        let prob_col = g.narrow_(top_probs, 1, ki, 1);
        let prob = g.reshape_(prob_col, vec![rows as i64, 1]);
        let gate = g.add_node(
            Op::GroupedMatMul,
            vec![h2, gate_w, expert_idx],
            Shape::new(&[rows, n_ff], DType::F32),
        );
        let up = g.add_node(
            Op::GroupedMatMul,
            vec![h2, up_w, expert_idx],
            Shape::new(&[rows, n_ff], DType::F32),
        );
        let act = g.silu(gate);
        let swiglu = g.mul(act, up);
        let down = g.add_node(
            Op::GroupedMatMul,
            vec![swiglu, down_w, expert_idx],
            Shape::new(&[rows, n_embd], DType::F32),
        );
        let weighted = g.mul(down, prob);
        acc = Some(match acc {
            None => weighted,
            Some(a) => g.add(a, weighted),
        });
    }
    let mut moe_flat = acc.expect("top_k >= 1");

    if let (Some(sg), Some(su), Some(sd)) = (
        moe.shared_gate.as_ref(),
        moe.shared_up.as_ref(),
        moe.shared_down.as_ref(),
    ) {
        let s_gate_w = param(
            g,
            params,
            &layer_key(il, "mlp.shared_experts.gate_proj.weight"),
            sg,
            &[n_embd, n_ff],
        );
        let s_up_w = param(
            g,
            params,
            &layer_key(il, "mlp.shared_experts.up_proj.weight"),
            su,
            &[n_embd, n_ff],
        );
        let s_down_w = param(
            g,
            params,
            &layer_key(il, "mlp.shared_experts.down_proj.weight"),
            sd,
            &[n_ff, n_embd],
        );
        let s_gate = g.mm(h2, s_gate_w);
        let s_up = g.mm(h2, s_up_w);
        let s_act = g.silu(s_gate);
        let s_swiglu = g.mul(s_act, s_up);
        let s_down = g.mm(s_swiglu, s_down_w);
        moe_flat = g.add(moe_flat, s_down);
    }

    let out3 = g.reshape_(moe_flat, vec![batch as i64, seq as i64, n_embd as i64]);
    Ok(g.add(h_in, out3))
}

fn layer_key(il: usize, tail: &str) -> String {
    format!("model.layers.{il}.{tail}")
}

fn build_attention(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    cfg: &LLaDA2MoeConfig,
    il: usize,
    h: NodeId,
    attn_mask: NodeId,
    cos: NodeId,
    sin: NodeId,
    layer: &crate::weights::LayerWeights,
    batch: usize,
    seq: usize,
) -> Result<NodeId> {
    let n_head = cfg.num_attention_heads;
    let n_kv = cfg.num_kv_heads();
    let head_dim = cfg.head_dim();
    let n_rot = cfg.rope_dim();
    let group = cfg.kv_group_size();
    let eps = cfg.rms_norm_eps as f32;

    let qkv_w = param(
        g,
        params,
        &layer_key(il, "self_attn.query_key_value.weight"),
        &layer.qkv,
        &[cfg.hidden_size, (n_head + 2 * n_kv) * head_dim],
    );
    let h2 = g.reshape_(h, vec![(batch * seq) as i64, cfg.hidden_size as i64]);
    let qkv = g.mm(h2, qkv_w);
    let qkv3 = g.reshape_(
        qkv,
        vec![
            batch as i64,
            seq as i64,
            ((n_head + 2 * n_kv) * head_dim) as i64,
        ],
    );
    let (q, k, v) = split_qkv(g, qkv3, batch, seq, n_head, n_kv, head_dim);

    let mut q_n = q;
    let mut k_n = k;
    if let Some(q_gamma) = &layer.q_norm {
        q_n = head_rms_norm(
            g,
            params,
            q,
            q_gamma,
            &layer_key(il, "self_attn.query_layernorm.weight"),
            batch,
            seq,
            n_head,
            head_dim,
            eps,
        );
    }
    if let Some(k_gamma) = &layer.k_norm {
        k_n = head_rms_norm(
            g,
            params,
            k,
            k_gamma,
            &layer_key(il, "self_attn.key_layernorm.weight"),
            batch,
            seq,
            n_kv,
            head_dim,
            eps,
        );
    }

    let q_rot = g.rope_n(q_n, cos, sin, head_dim, n_rot);
    let k_rot = g.rope_n(k_n, cos, sin, head_dim, n_rot);

    let q_bhsd = to_bhsd(g, q_rot, batch, seq, n_head, head_dim);
    let k_bhsd = to_bhsd(g, k_rot, batch, seq, n_kv, head_dim);
    let v_bhsd = to_bhsd(g, v, batch, seq, n_kv, head_dim);

    let k_full = repeat_kv_bhsd(g, k_bhsd, n_kv, group);
    let v_full = repeat_kv_bhsd(g, v_bhsd, n_kv, group);

    let attn_shape = g.shape(q_bhsd).clone();
    let attn_out = g.attention_bias(
        q_bhsd, k_full, v_full, attn_mask, n_head, head_dim, attn_shape,
    );

    let o_w = param(
        g,
        params,
        &layer_key(il, "self_attn.dense.weight"),
        &layer.o_proj,
        &[n_head * head_dim, cfg.hidden_size],
    );
    let attn_bshd = g.transpose_(attn_out, vec![0, 2, 1, 3]);
    let attn2 = g.reshape_(
        attn_bshd,
        vec![(batch * seq) as i64, (n_head * head_dim) as i64],
    );
    let proj = g.mm(attn2, o_w);
    let proj3 = g.reshape_(proj, vec![batch as i64, seq as i64, cfg.hidden_size as i64]);
    // Pre-LN: outer block adds `residual`; attention returns proj only (TIDE decoder layer).
    Ok(proj3)
}

/// Full-sequence forward with custom block-diffusion mask.
///
/// Inputs: `input_ids` `[B,S]`, `position_ids` `[B,S]`, `attn_mask` `[B,1,S,S]`.
/// Output: `logits` `[B,S,V]`.
pub fn build_llada2_forward_graph(
    cfg: &LLaDA2MoeConfig,
    weights: &LLaDA2Weights,
    batch: usize,
    seq: usize,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    if batch == 0 || seq == 0 {
        return Err(anyhow!("batch and seq must be positive"));
    }
    let mut g = Graph::new("llada2_forward");
    let mut params = HashMap::new();
    let inv = crate::rope::inv_freq(cfg);
    let (cos_data, sin_data) =
        crate::rope::build_rope_tables(cfg, &inv, cfg.max_position_embeddings);
    let tab_half = cfg.head_dim() / 2;
    crate::weights::register_params(cfg, weights, &mut params);
    params.insert("rope.cos".into(), cos_data.clone());
    params.insert("rope.sin".into(), sin_data.clone());

    let input_ids = g.input("input_ids", Shape::new(&[batch, seq], DType::F32));
    let position_ids = g.input("position_ids", Shape::new(&[batch, seq], DType::F32));
    let attn_mask = g.input("attn_mask", Shape::new(&[batch, 1, seq, seq], DType::F32));

    let embed_w = param(
        &mut g,
        &mut params,
        "model.embed_tokens.weight",
        &weights.embed,
        &[cfg.vocab_size, cfg.hidden_size],
    );
    // Flatten token ids for embed gather — MLX `take` on 2D indices can
    // broadcast to the wrong rank; CPU/Metal/WGPU accept [B,S] directly.
    let ids_flat = g.reshape_(input_ids, vec![(batch * seq) as i64]);
    let h_flat = g.gather_(embed_w, ids_flat, 0);
    let mut h = g.reshape_(
        h_flat,
        vec![batch as i64, seq as i64, cfg.hidden_size as i64],
    );

    let cos_tab = param(
        &mut g,
        &mut params,
        "rope.cos",
        &cos_data,
        &[cfg.max_position_embeddings, tab_half],
    );
    let sin_tab = param(
        &mut g,
        &mut params,
        "rope.sin",
        &sin_data,
        &[cfg.max_position_embeddings, tab_half],
    );
    let cos = gather_rope(&mut g, cos_tab, position_ids, batch, seq, tab_half);
    let sin = gather_rope(&mut g, sin_tab, position_ids, batch, seq, tab_half);

    let eps = cfg.rms_norm_eps as f32;
    for (il, layer) in weights.layers.iter().enumerate() {
        let residual = h;
        h = rms_norm_layer(
            &mut g,
            &mut params,
            h,
            &layer_key(il, "input_layernorm.weight"),
            &layer.input_norm,
            eps,
        );
        h = build_attention(
            &mut g,
            &mut params,
            cfg,
            il,
            h,
            attn_mask,
            cos,
            sin,
            layer,
            batch,
            seq,
        )?;
        h = g.add(residual, h);

        let residual2 = h;
        h = rms_norm_layer(
            &mut g,
            &mut params,
            h,
            &layer_key(il, "post_attention_layernorm.weight"),
            &layer.post_attn_norm,
            eps,
        );
        h = match &layer.ffn {
            LayerFfn::Dense(d) => build_dense_ffn(
                &mut g,
                &mut params,
                h,
                il,
                &d.gate,
                &d.up,
                &d.down,
                cfg.hidden_size,
                cfg.intermediate_size(),
                batch,
                seq,
            ),
            LayerFfn::Moe(m) => build_moe_ffn(&mut g, &mut params, cfg, il, h, m, batch, seq)?,
        };
        h = g.add(residual2, h);
    }

    h = rms_norm_layer(
        &mut g,
        &mut params,
        h,
        "model.norm.weight",
        &weights.final_norm,
        eps,
    );
    let rows = batch * seq;
    let h2 = g.reshape_(h, vec![rows as i64, cfg.hidden_size as i64]);
    let lm_w = param(
        &mut g,
        &mut params,
        "lm_head.weight",
        &weights.lm_head,
        &[cfg.hidden_size, cfg.vocab_size],
    );
    let logits2 = g.mm(h2, lm_w);
    let logits = g.reshape_(
        logits2,
        vec![batch as i64, seq as i64, cfg.vocab_size as i64],
    );
    g.set_outputs(vec![logits]);
    Ok((g, params))
}
