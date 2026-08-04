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
use rlx_ir::quant::QuantScheme;
use rlx_ir::{DType, HirGraphExt, Shape};
use rlx_llada2::llada2::gate_op::{
    OP_NAME, ensure_group_limited_gate_registered, gate_attrs_bytes,
};
use std::collections::HashMap;

type Params = HashMap<String, Vec<f32>>;

/// Batched (rows×k, two grouped matmuls) MoE expert path — ON by default.
/// Set `RLX_KIMI_BATCHED_MOE=0` to fall back to the exact per-k unroll (bit-for-
/// bit vs the reference; the batched path differs by ~1e-6 in reduction order).
fn batched_moe() -> bool {
    std::env::var("RLX_KIMI_BATCHED_MOE")
        .map(|v| v != "0")
        .unwrap_or(true)
}

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

    let routed = if batched_moe() {
        // BATCHED (default; disable with `RLX_KIMI_BATCHED_MOE=0`): expand
        // rows → rows*k (each row paired with its k experts) and run ONE grouped
        // matmul each for gate_up / down instead of k, then weighted-sum over the
        // k axis. Collapses the per-k unroll (k× the GroupedMatMul + situ +
        // GroupedMatMul + elementwise chain — O(k) graph nodes, k=16 in the real
        // model) into a handful of ops. CAVEAT: the final `sum` over k reduces in
        // a different ORDER than the sequential `acc` adds, so results can differ
        // by ~1e-6 → a near-tie token could flip; `=0` restores the exact unroll.
        let rk = rows * k;
        let h_lat3 = g.reshape_(h_lat, vec![rows as i64, 1, l as i64]);
        let zeros_k = reg(
            g,
            params,
            &format!("{prefix}.moe_batch_zero"),
            vec![0f32; k],
            &[1, k, 1],
        );
        let h_exp = g.add(h_lat3, zeros_k); // [rows,k,L] via broadcast (each row ×k)
        let h_exp = g.reshape_(h_exp, vec![rk as i64, l as i64]);
        let eidx = g.reshape_(top_idx, vec![rk as i64]); // [rows,k] → row r*k+ki = expert
        let gate_up = g.add_node(
            Op::GroupedMatMul,
            vec![h_exp, gate_up_w, eidx],
            Shape::new(&[rk, 2 * mi], f),
        );
        let hx = situ(g, gate_up, rk, mi, d.situ_beta, d.situ_linear_beta);
        let down = g.add_node(
            Op::GroupedMatMul,
            vec![hx, down_w, eidx],
            Shape::new(&[rk, l], f),
        );
        let probs = g.reshape_(top_probs, vec![rk as i64, 1]);
        let weighted = g.mul(down, probs);
        let weighted3 = g.reshape_(weighted, vec![rows as i64, k as i64, l as i64]);
        g.sum(weighted3, vec![1], false) // Σ over k → [rows, L]
    } else {
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
        acc.expect("top_k >= 1")
    };

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

/// The MoE **tail** for the disaggregated (expert-parallel) path: given the gathered
/// routed-expert latent partial `routed [rows, L]` (summed across workers by
/// `rlx_distributed::dispatch_experts`) and the original hidden input `h2d [rows,
/// hidden]`, apply `routed_expert_norm` (RMSNorm) → `routed_expert_up_proj` (L→hidden)
/// and add the always-on **shared experts** (situ SwiGLU in hidden) → `[b, s, hidden]`.
///
/// Runs on the ORCHESTRATOR (not the workers): the RMSNorm is nonlinear so it must
/// apply AFTER the cross-worker sum. Replicates the tail of [`build_latent_moe`]; the
/// worker head is [`crate::dist_experts::KimiExpertProvider`].
pub fn build_moe_tail(
    g: &mut HirMut,
    params: &mut Params,
    prefix: &str,
    routed: HirNodeId,
    h2d: HirNodeId,
    w: &MoeWeights,
    d: MoeDims,
) -> Result<HirNodeId> {
    let (b, s, hidden, l) = (d.batch, d.seq, d.hidden, d.latent);
    let rows = b * s;
    let shared_inter = d.num_shared * d.moe_inter;
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

/// **Paged MoE — router half.** Runs the (resident) router + latent down-proj and
/// exposes `(h_lat [rows,L], top_idx [rows,k], top_probs [rows,k])` so the host
/// can read which experts fired, page ONLY those from disk, and finish in
/// [`build_moe_experts_paged`]. Same math as the router part of
/// [`build_latent_moe`].
#[allow(clippy::too_many_arguments)]
pub fn build_moe_route(
    g: &mut HirMut,
    params: &mut Params,
    prefix: &str,
    h_in: HirNodeId,
    router: &[f32],
    e_score_bias: &[f32],
    down_latent: &[f32],
    d: MoeDims,
) -> Result<(HirNodeId, HirNodeId, HirNodeId)> {
    ensure_group_limited_gate_registered();
    let f = DType::F32;
    let (b, s, hidden, l, e, k) = (d.batch, d.seq, d.hidden, d.latent, d.num_experts, d.top_k);
    let rows = b * s;
    let h2d = g.reshape_(h_in, vec![rows as i64, hidden as i64]);
    let router_w = reg(
        g,
        params,
        &format!("{prefix}.gate.weight"),
        router.to_vec(),
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
        e_score_bias.to_vec(),
        &[1, e],
    );
    let route = g.add(sig, ebias);
    let attrs = gate_attrs_bytes(1, 1, k, d.routed_scaling, e);
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
    let h_lat = linear(
        g,
        params,
        prefix,
        "routed_expert_down_proj",
        h2d,
        down_latent,
        hidden,
        l,
    );
    Ok((h_lat, top_idx, top_probs))
}

/// **Packed (opt #2)** counterpart of [`build_moe_experts_paged`]: the routed
/// gate/up/down expert matmuls run as fused `DequantGroupedMatMulMlx{MlxMxfp4}`
/// over the RAW MXFP4 bytes (no CPU dequant, no f32 materialize, 4× less
/// bandwidth). The `*_codes` (U8) / `*_scales` (BF16) / `*_biases` (BF16) params
/// are declared here by NAME and their bytes attached after compile via
/// `CompiledGraph::set_param_typed` (see [`crate::runner::run_moe_paged`]). Latent
/// projections + shared experts stay f32 (dense), identical to the f32 path.
#[allow(clippy::too_many_arguments)]
/// Packed routed-expert **PRE-NORM latent partial** `Σ_k prob·expert_k(h_lat)` →
/// `[rows, L]`, computed via MXFP4 `DequantGroupedMatMulMlx` (the GPU does dequant +
/// matmul — NO CPU dequant, and only the ~17.5MB packed codes/scales move, not the
/// ~132MB f32 expansion). The `n_uniq` compact experts' packed codes/scales are declared
/// as params here (names `moe.{gate,up,down}_{codes,scales,biases}`, data fed post-compile
/// via `set_param_typed`). This is the shared pre-norm core:
/// [`build_moe_experts_paged_packed`] appends the tail (routed_norm + up_proj + shared);
/// the distributed worker provider ([`crate::dist_experts`]) calls it DIRECTLY on its
/// OWNED shard (idx/prob zeroed for non-owned slots) to emit its latent partial.
pub fn build_packed_routed_latent(
    g: &mut HirMut,
    h_lat: HirNodeId,
    remapped_idx: HirNodeId,
    top_probs: HirNodeId,
    n_uniq: usize,
    rows: usize,
    d: MoeDims,
) -> HirNodeId {
    let f = DType::F32;
    let (l, mi, k) = (d.latent, d.moe_inter, d.top_k);
    let gs = 32usize; // Kimi MXFP4 group size
    let scheme = QuantScheme::MlxMxfp4 {
        group_size: gs as u32,
    };
    // packed expert params (data set post-compile via set_param_typed). gate/up:
    // [n_uniq, mi, L] (K=L); down: [n_uniq, L, mi] (K=mi).
    let u8p = |g: &mut HirMut, name: &str, n: usize| g.param(name, Shape::new(&[n], DType::U8));
    let bf = |g: &mut HirMut, name: &str, sh: &[usize]| g.param(name, Shape::new(sh, DType::BF16));
    let gc = u8p(g, "moe.gate_codes", n_uniq * mi * (l / 2));
    let gsc = bf(g, "moe.gate_scales", &[n_uniq, mi, l / gs]);
    let gb = bf(g, "moe.gate_biases", &[n_uniq, mi, l / gs]);
    let uc = u8p(g, "moe.up_codes", n_uniq * mi * (l / 2));
    let usc = bf(g, "moe.up_scales", &[n_uniq, mi, l / gs]);
    let ub = bf(g, "moe.up_biases", &[n_uniq, mi, l / gs]);
    let dc = u8p(g, "moe.down_codes", n_uniq * l * (mi / 2));
    let dsc = bf(g, "moe.down_scales", &[n_uniq, l, mi / gs]);
    let db = bf(g, "moe.down_biases", &[n_uniq, l, mi / gs]);

    let mut acc: Option<HirNodeId> = None;
    for ki in 0..k {
        let idx_col = g.narrow_(remapped_idx, 1, ki, 1);
        let eidx = g.reshape_(idx_col, vec![rows as i64]);
        let prob_col = g.narrow_(top_probs, 1, ki, 1);
        let prob = g.reshape_(prob_col, vec![rows as i64, 1]);
        let gmm = |g: &mut HirMut, x, c, sc, bi, out_n| {
            g.add_node(
                Op::DequantGroupedMatMulMlx { scheme },
                vec![x, c, sc, bi, eidx],
                Shape::new(&[rows, out_n], f),
            )
        };
        let gate = gmm(g, h_lat, gc, gsc, gb, mi); // [rows, mi]
        let up = gmm(g, h_lat, uc, usc, ub, mi); // [rows, mi]
        let gate_up = g.concat_(vec![gate, up], 1); // [rows, 2mi] for situ
        let hx = situ(g, gate_up, rows, mi, d.situ_beta, d.situ_linear_beta);
        let down = gmm(g, hx, dc, dsc, db, l); // [rows, L]
        let weighted = g.mul(down, prob);
        acc = Some(match acc {
            Some(a) => g.add(a, weighted),
            None => weighted,
        });
    }
    acc.expect("top_k >= 1")
}

pub fn build_moe_experts_paged_packed(
    g: &mut HirMut,
    params: &mut Params,
    prefix: &str,
    h_in: HirNodeId,
    h_lat: HirNodeId,
    remapped_idx: HirNodeId,
    top_probs: HirNodeId,
    n_uniq: usize,
    w: &MoeWeights,
    d: MoeDims,
) -> Result<HirNodeId> {
    let (b, s, hidden, l, mi) = (d.batch, d.seq, d.hidden, d.latent, d.moe_inter);
    let rows = b * s;
    let shared_inter = d.num_shared * mi;
    let h2d = g.reshape_(h_in, vec![rows as i64, hidden as i64]);

    // pre-norm routed latent partial (shared packed core, all experts owned here).
    let routed = build_packed_routed_latent(g, h_lat, remapped_idx, top_probs, n_uniq, rows, d);
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
    // shared experts (f32 dense) — identical to the f32 path
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

/// **Paged MoE — expert half.** Runs the routed FFN over a COMPACT expert set
/// (`n_uniq` experts the host paged from disk), with host-remapped `remapped_idx`
/// `[rows,k]` (compact indices) + `top_probs` `[rows,k]` fed as inputs, then the
/// latent norm/up-proj + the 2 shared experts. Returns `[b,s,hidden]`. Identical
/// math to [`build_latent_moe`]'s expert half, just over the selected experts.
#[allow(clippy::too_many_arguments)]
pub fn build_moe_experts_paged(
    g: &mut HirMut,
    params: &mut Params,
    prefix: &str,
    h_in: HirNodeId,
    h_lat: HirNodeId,
    remapped_idx: HirNodeId,
    top_probs: HirNodeId,
    compact_gate_up: &[f32],
    compact_down: &[f32],
    n_uniq: usize,
    w: &MoeWeights,
    d: MoeDims,
) -> Result<HirNodeId> {
    let f = DType::F32;
    let (b, s, hidden, l, mi, k) = (d.batch, d.seq, d.hidden, d.latent, d.moe_inter, d.top_k);
    let rows = b * s;
    let shared_inter = d.num_shared * mi;
    let h2d = g.reshape_(h_in, vec![rows as i64, hidden as i64]);

    let gate_up_w = reg(
        g,
        params,
        &format!("{prefix}.experts.gate_up"),
        compact_gate_up.to_vec(),
        &[n_uniq, l, 2 * mi],
    );
    let down_w = reg(
        g,
        params,
        &format!("{prefix}.experts.down"),
        compact_down.to_vec(),
        &[n_uniq, mi, l],
    );
    let routed = if batched_moe() {
        // BATCHED (default; disable with `RLX_KIMI_BATCHED_MOE=0`): expand
        // rows → rows*k and run ONE grouped matmul each for gate_up / down instead
        // of k, then weighted-sum over the k axis — collapses the per-k unroll
        // (O(k) graph nodes) into a handful of ops. CAVEAT: the final `sum` over k
        // reduces in a different ORDER than the sequential `acc` adds → results
        // can differ by ~1e-6, so a near-tie token could flip; `=0` restores it.
        let rk = rows * k;
        let h_lat3 = g.reshape_(h_lat, vec![rows as i64, 1, l as i64]);
        let zeros_k = reg(
            g,
            params,
            &format!("{prefix}.moe_batch_zero"),
            vec![0f32; k],
            &[1, k, 1],
        );
        let h_exp = g.add(h_lat3, zeros_k); // [rows,k,L] via broadcast (each row ×k)
        let h_exp = g.reshape_(h_exp, vec![rk as i64, l as i64]);
        let eidx = g.reshape_(remapped_idx, vec![rk as i64]); // row r*k+ki = expert
        let gate_up = g.add_node(
            Op::GroupedMatMul,
            vec![h_exp, gate_up_w, eidx],
            Shape::new(&[rk, 2 * mi], f),
        );
        let hx = situ(g, gate_up, rk, mi, d.situ_beta, d.situ_linear_beta);
        let down = g.add_node(
            Op::GroupedMatMul,
            vec![hx, down_w, eidx],
            Shape::new(&[rk, l], f),
        );
        let probs = g.reshape_(top_probs, vec![rk as i64, 1]);
        let weighted = g.mul(down, probs);
        let weighted3 = g.reshape_(weighted, vec![rows as i64, k as i64, l as i64]);
        g.sum(weighted3, vec![1], false) // Σ over k → [rows, L]
    } else {
        let mut acc: Option<HirNodeId> = None;
        for ki in 0..k {
            let idx_col = g.narrow_(remapped_idx, 1, ki, 1);
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
        acc.expect("top_k >= 1")
    };
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
