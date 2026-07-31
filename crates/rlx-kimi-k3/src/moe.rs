// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! Kimi-K3 **LatentMoE** FFN (`KimiSparseMoeBlock`).
//!
//! A sigmoid `noaux_tc` grouped-topk router (reusing rlx-llada2's
//! `group_limited_gate` op — bit-identical to KimiMoEGate) selects `top_k` of
//! `num_experts`. The routed experts run in a **low-rank latent space**: a shared
//! `routed_expert_down_proj` maps `hidden → L`, each expert is a situ-SwiGLU over
//! `L` (`w1/w3: L→I_moe`, `w2: I_moe→L`), the top-k-weighted expert outputs are
//! summed in `L`, RMSNorm'd (`routed_expert_norm`), then a shared
//! `routed_expert_up_proj` lifts `L → hidden`. Two always-on shared experts run in
//! `hidden` space and are added after the up-projection.

use crate::common::{linear, reg, situ};
use anyhow::Result;
use rlx_ir::hir::{HirMut, HirNodeId};
use rlx_ir::op::{Activation, Op};
use rlx_ir::{DType, HirGraphExt, Shape};
use rlx_llada2::llada2::gate_op::{
    OP_NAME, ensure_group_limited_gate_registered, gate_attrs_bytes,
};
use std::collections::HashMap;

type Params = HashMap<String, Vec<f32>>;

#[derive(Debug, Clone, Copy)]
pub struct MoeDims {
    pub hidden: usize,
    pub latent: usize,    // routed_expert_hidden_size (L)
    pub moe_inter: usize, // per-expert FFN width (I_moe)
    pub num_experts: usize,
    pub top_k: usize,
    pub num_shared: usize,
    pub routed_scaling: f32,
    pub eps: f32,
    pub situ_beta: f32,
    pub situ_linear_beta: Option<f32>,
    pub batch: usize,
    pub seq: usize,
}

/// Dense LatentMoE weights. Expert weights are already in the `[E, K, N]`
/// GroupedMatMul layout (the loader stacks `w1‖w3`/`w2` and transposes).
#[derive(Debug, Clone, Default)]
pub struct MoeWeights {
    pub router: Vec<f32>,          // [hidden, num_experts]
    pub e_score_bias: Vec<f32>,    // [num_experts]
    pub down_latent: Vec<f32>,     // [hidden, L]
    pub up_latent: Vec<f32>,       // [L, hidden]
    pub routed_norm: Vec<f32>,     // [L]
    pub experts_gate_up: Vec<f32>, // [E, L, 2*I_moe]
    pub experts_down: Vec<f32>,    // [E, I_moe, L]
    pub shared_gate: Vec<f32>,     // [hidden, num_shared*I_moe]
    pub shared_up: Vec<f32>,       // [hidden, num_shared*I_moe]
    pub shared_down: Vec<f32>,     // [num_shared*I_moe, hidden]
}

/// Dense (non-MoE) FFN weights — `mlp.{gate,up,down}_proj` (layer 0).
#[derive(Debug, Clone, Default)]
pub struct DenseMlpWeights {
    pub gate: Vec<f32>, // [hidden, inter]
    pub up: Vec<f32>,   // [hidden, inter]
    pub down: Vec<f32>, // [inter, hidden]
}

/// The dense situ-SwiGLU MLP used by the first `first_k_dense_replace` layers.
#[allow(clippy::too_many_arguments)]
pub fn build_dense_mlp(
    g: &mut HirMut,
    params: &mut Params,
    prefix: &str,
    h_in: HirNodeId,
    w: &DenseMlpWeights,
    hidden: usize,
    inter: usize,
    batch: usize,
    seq: usize,
    situ_beta: f32,
    situ_linear_beta: Option<f32>,
) -> Result<HirNodeId> {
    let rows = batch * seq;
    let h2d = g.reshape_(h_in, vec![rows as i64, hidden as i64]);
    let gate = linear(g, params, prefix, "gate_proj", h2d, &w.gate, hidden, inter);
    let up = linear(g, params, prefix, "up_proj", h2d, &w.up, hidden, inter);
    let gate_up = g.concat_(vec![gate, up], 1);
    let hx = situ(g, gate_up, rows, inter, situ_beta, situ_linear_beta);
    let down = linear(g, params, prefix, "down_proj", hx, &w.down, inter, hidden);
    Ok(g.reshape_(down, vec![batch as i64, seq as i64, hidden as i64]))
}

/// Build the LatentMoE FFN on `h_in` `[batch, seq, hidden]`; returns
/// `[batch, seq, hidden]` (no residual add — the caller owns the residual).
pub fn build_latent_moe(
    g: &mut HirMut,
    params: &mut Params,
    prefix: &str,
    h_in: HirNodeId,
    w: &MoeWeights,
    d: MoeDims,
) -> Result<HirNodeId> {
    ensure_group_limited_gate_registered();
    let f = DType::F32;
    let (b, s, hidden, l, mi, e, k) = (
        d.batch,
        d.seq,
        d.hidden,
        d.latent,
        d.moe_inter,
        d.num_experts,
        d.top_k,
    );
    let rows = b * s;
    let shared_inter = d.num_shared * mi;

    let h2d = g.reshape_(h_in, vec![rows as i64, hidden as i64]);

    // ── Router: sigmoid(mm) + e_score_correction_bias → group_limited_gate ──
    let router_w = reg(
        g,
        params,
        &format!("{prefix}.gate.weight"),
        w.router.clone(),
        &[hidden, e],
    );
    let logits = g.mm(h2d, router_w);
    let sig = g.add_node(
        Op::Activation(Activation::Sigmoid),
        vec![logits],
        Shape::new(&[rows, e], f),
    );
    let ebias = reg(
        g,
        params,
        &format!("{prefix}.gate.e_score_correction_bias"),
        w.e_score_bias.clone(),
        &[1, e],
    );
    let route = g.add(sig, ebias);
    let attrs = gate_attrs_bytes(1, 1, k, d.routed_scaling, e); // n_group=1, topk_group=1
    let packed = g.add_node(
        Op::Custom {
            name: OP_NAME.to_string(),
            num_inputs: 2,
            attrs,
        },
        vec![sig, route],
        Shape::new(&[rows, k * 2], f),
    );
    let top_idx = g.narrow_(packed, 1, 0, k);
    let top_probs = g.narrow_(packed, 1, k, k);

    // ── Latent down-projection: experts operate on [rows, L] ──
    let h_lat = linear(
        g,
        params,
        prefix,
        "routed_expert_down_proj",
        h2d,
        &w.down_latent,
        hidden,
        l,
    );
    let gate_up_w = reg(
        g,
        params,
        &format!("{prefix}.experts.gate_up"),
        w.experts_gate_up.clone(),
        &[e, l, 2 * mi],
    );
    let down_w = reg(
        g,
        params,
        &format!("{prefix}.experts.down"),
        w.experts_down.clone(),
        &[e, mi, l],
    );

    let mut acc: Option<HirNodeId> = None;
    for ki in 0..k {
        let idx_col = g.narrow_(top_idx, 1, ki, 1);
        let eidx = g.reshape_(idx_col, vec![rows as i64]);
        let prob_col = g.narrow_(top_probs, 1, ki, 1);
        let prob = g.reshape_(prob_col, vec![rows as i64, 1]);
        let gate_up = g.add_node(
            Op::GroupedMatMul,
            vec![h_lat, gate_up_w, eidx],
            Shape::new(&[rows, 2 * mi], f),
        );
        let hx = situ(g, gate_up, rows, mi, d.situ_beta, d.situ_linear_beta);
        let down = g.add_node(
            Op::GroupedMatMul,
            vec![hx, down_w, eidx],
            Shape::new(&[rows, l], f),
        );
        let weighted = g.mul(down, prob);
        acc = Some(match acc {
            Some(a) => g.add(a, weighted),
            None => weighted,
        });
    }
    let routed = acc.expect("top_k >= 1");

    // ── latent RMSNorm + up-projection: [rows, L] → [rows, hidden] ──
    let norm_w = reg(
        g,
        params,
        &format!("{prefix}.routed_expert_norm"),
        w.routed_norm.clone(),
        &[l],
    );
    let zero_beta = reg(
        g,
        params,
        &format!("{prefix}.routed_expert_norm.zero_beta"),
        vec![0f32; l],
        &[l],
    );
    let routed = g.rms_norm(routed, norm_w, zero_beta, d.eps);
    let routed_h = linear(
        g,
        params,
        prefix,
        "routed_expert_up_proj",
        routed,
        &w.up_latent,
        l,
        hidden,
    );

    // ── shared experts (situ SwiGLU in hidden space) ──
    let sg = linear(
        g,
        params,
        prefix,
        "shared_experts.gate_proj",
        h2d,
        &w.shared_gate,
        hidden,
        shared_inter,
    );
    let su = linear(
        g,
        params,
        prefix,
        "shared_experts.up_proj",
        h2d,
        &w.shared_up,
        hidden,
        shared_inter,
    );
    let s_gate_up = g.concat_(vec![sg, su], 1);
    let sh = situ(
        g,
        s_gate_up,
        rows,
        shared_inter,
        d.situ_beta,
        d.situ_linear_beta,
    );
    let shared = linear(
        g,
        params,
        prefix,
        "shared_experts.down_proj",
        sh,
        &w.shared_down,
        shared_inter,
        hidden,
    );

    let out = g.add(routed_h, shared);
    Ok(g.reshape_(out, vec![b as i64, s as i64, hidden as i64]))
}
