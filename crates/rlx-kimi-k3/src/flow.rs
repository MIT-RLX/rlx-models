// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! KimiLinear text-decoder flow: the hybrid KDA/MLA layer stack stitched with
//! **Attention Residuals** (a block-residual, refreshed every
//! `attn_res_block_size` layers), a final RMSNorm, and the untied `lm_head`.
//!
//! Attention Residuals (`_forward_attn_residual` / `_apply_attn_res`): within a
//! block the running `prefix_sum` is a normal additive residual stream, but every
//! sublayer's norm input is first re-mixed with all stored block snapshots via a
//! learned-query softmax attention over `{snapshots…, prefix_sum}`. At each block
//! boundary the incoming stream is snapshotted and the direct residual is reset.

use crate::common::{act, linear, reg, scalar_const};
use crate::kda::{KdaDims, KdaWeights, build_kda_layer};
use crate::mla::{MlaDims, MlaWeights, build_mla_layer};
use crate::moe::{DenseMlpWeights, MoeDims, MoeWeights, build_dense_mlp, build_latent_moe};
use anyhow::Result;
use rlx_ir::hir::{HirMut, HirNodeId};
use rlx_ir::op::Activation;
use rlx_ir::{DType, HirGraphExt, Shape};
use std::collections::HashMap;

type Params = HashMap<String, Vec<f32>>;

pub enum AttnWeights {
    Kda(Box<KdaWeights>),
    Mla(Box<MlaWeights>),
}

pub enum FfnWeights {
    Dense(Box<DenseMlpWeights>),
    Moe(Box<MoeWeights>),
}

pub struct LayerWeights {
    pub input_ln: Vec<f32>,     // [hidden]
    pub post_ln: Vec<f32>,      // [hidden]
    pub sa_res_norm: Vec<f32>,  // [hidden]
    pub sa_res_proj: Vec<f32>,  // [hidden] (nn.Linear(H,1).weight squeezed)
    pub mlp_res_norm: Vec<f32>, // [hidden]
    pub mlp_res_proj: Vec<f32>, // [hidden]
    pub attn: AttnWeights,
    pub ffn: FfnWeights,
}

pub struct FlowWeights {
    pub layers: Vec<LayerWeights>,
    pub final_norm: Vec<f32>,   // [hidden]
    pub out_res_norm: Vec<f32>, // [hidden]
    pub out_res_proj: Vec<f32>, // [hidden]
    pub lm_head: Vec<f32>,      // [hidden, vocab]
}

/// Shared shape config for the flow (one Dims per attention/FFN kind).
pub struct FlowConfig {
    pub hidden: usize,
    pub vocab: usize,
    pub attn_res_block_size: usize,
    pub eps: f32,
    pub kda: KdaDims,
    pub mla: MlaDims,
    pub moe: MoeDims,
    pub dense_inter: usize,
    pub situ_beta: f32,
    pub situ_linear_beta: Option<f32>,
    pub batch: usize,
    pub seq: usize,
}

fn rms_norm(
    g: &mut HirMut,
    params: &mut Params,
    name: &str,
    x: HirNodeId,
    w: &[f32],
    hidden: usize,
    eps: f32,
) -> HirNodeId {
    let gamma = reg(g, params, name, w.to_vec(), &[hidden]);
    let zb = reg(
        g,
        params,
        &format!("{name}.zero_beta"),
        vec![0f32; hidden],
        &[hidden],
    );
    g.rms_norm(x, gamma, zb, eps)
}

/// `_apply_attn_res`: softmax-attention over `{snapshots…, prefix_sum}` with a
/// learned query (`norm.weight ⊙ proj.weight`), RMSNorm'd keys, raw values.
/// All tensors are `[batch, seq, hidden]`; returns `[batch, seq, hidden]`.
#[allow(clippy::too_many_arguments)]
fn apply_attn_res(
    g: &mut HirMut,
    params: &mut Params,
    name: &str,
    prefix_sum: HirNodeId,
    snapshots: &[HirNodeId],
    proj_w: &[f32],
    norm_w: &[f32],
    rows: usize,
    hidden: usize,
    eps: f32,
) -> HirNodeId {
    let f = DType::F32;
    // v = stack([snapshots…, prefix_sum]) → [rows, m, hidden]
    let mut elems: Vec<HirNodeId> = snapshots
        .iter()
        .map(|&sn| g.reshape_(sn, vec![rows as i64, 1, hidden as i64]))
        .collect();
    let ps3 = g.reshape_(prefix_sum, vec![rows as i64, 1, hidden as i64]);
    elems.push(ps3);
    let m = elems.len();
    let v = g.concat_(elems, 1); // [rows, m, hidden]

    // kf = v / sqrt(mean(v^2, -1) + eps)  (RMSNorm keys, no weight)
    let sq = g.mul(v, v);
    let msq = g.mean(sq, vec![2], true); // [rows, m, 1]
    let eps_c = scalar_const(g, eps);
    let denom_pre = g.add(msq, eps_c);
    let denom = act(g, Activation::Sqrt, denom_pre, Shape::new(&[rows, m, 1], f));
    let kf = g.div(v, denom); // [rows, m, hidden]

    // query = norm_w ⊙ proj_w  (host-precomputed) → [1, 1, hidden]
    let q_data: Vec<f32> = norm_w.iter().zip(proj_w).map(|(a, b)| a * b).collect();
    let q = reg(g, params, &format!("{name}.query"), q_data, &[1, 1, hidden]);
    let kq = g.mul(kf, q); // [rows, m, hidden]
    let scores = g.sum(kq, vec![2], false); // [rows, m]
    let probs = g.sm(scores, 1); // softmax over m
    let probs3 = g.reshape_(probs, vec![rows as i64, m as i64, 1]);
    let weighted = g.mul(probs3, v); // [rows, m, hidden]
    g.sum(weighted, vec![1], false) // [rows, hidden]  (weighted avg of raw v)
}

/// Build the KimiLinear text decoder on `h_in` `[batch, seq, hidden]` (embeddings);
/// returns logits `[batch, seq, vocab]`.
pub fn build_kimi_text_flow(
    g: &mut HirMut,
    params: &mut Params,
    h_in: HirNodeId,
    w: &FlowWeights,
    cfg: &FlowConfig,
) -> Result<HirNodeId> {
    let (logits, _) =
        build_kimi_text_stage(g, params, h_in, Vec::new(), &w.layers, 0, true, w, cfg)?;
    Ok(logits)
}

/// Build ONE pipeline stage of the KimiLinear decoder: transformer `layers`
/// (global indices starting at `layer_offset`), consuming the boundary
/// `hidden_in` `[b,s,hidden]` and the Attention-Residual `snapshots_in`
/// accumulated by earlier stages (each `[b,s,hidden]`). Returns
/// `(out, snapshots_out)`:
/// - non-last stage → `out = hidden_out [b,s,hidden]`, `snapshots_out` = the
///   snapshot list to relay to the next stage;
/// - last stage → `out = logits [b,s,vocab]` (out-residual + final norm + untied
///   lm_head from `w`), `snapshots_out` empty.
///
/// Cut stages on AttnRes block boundaries (`layer_offset % attn_res_block_size ==
/// 0`) so a boundary never lands mid-block. This is the in-graph analogue of the
/// distributed `build_kimi_k3_stage` seam — the snapshots are the extra boundary
/// tensors a Kimi pipeline must carry beyond the hidden state.
#[allow(clippy::too_many_arguments)]
pub fn build_kimi_text_stage(
    g: &mut HirMut,
    params: &mut Params,
    hidden_in: HirNodeId,
    snapshots_in: Vec<HirNodeId>,
    layers: &[LayerWeights],
    layer_offset: usize,
    last: bool,
    w: &FlowWeights,
    cfg: &FlowConfig,
) -> Result<(HirNodeId, Vec<HirNodeId>)> {
    let (b, s, hidden) = (cfg.batch, cfg.seq, cfg.hidden);
    let rows = b * s;
    let bsh = |g: &mut HirMut, x: HirNodeId| g.reshape_(x, vec![b as i64, s as i64, hidden as i64]);

    let mut snapshots: Vec<HirNodeId> = snapshots_in;
    let mut hidden_t = hidden_in; // [b, s, hidden]

    for (li, lw) in layers.iter().enumerate() {
        let i = layer_offset + li;
        let incoming = hidden_t;
        // prefix_sum for this layer; `None` after a block boundary resets the residual.
        let mut stream = Some(incoming);

        // (1) self-attention residual mix (skip when there are no snapshots)
        let mut hs = incoming;
        if !snapshots.is_empty() {
            let mixed = apply_attn_res(
                g,
                params,
                &format!("l{i}.sa_res"),
                incoming,
                &snapshots,
                &lw.sa_res_proj,
                &lw.sa_res_norm,
                rows,
                hidden,
                cfg.eps,
            );
            hs = bsh(g, mixed);
        }

        // (2) block boundary: snapshot the incoming stream, reset the residual
        if i % cfg.attn_res_block_size == 0 {
            snapshots.push(incoming);
            stream = None;
        }

        // (3) input norm → attention (raw output)
        let hn = rms_norm(
            g,
            params,
            &format!("l{i}.input_ln"),
            hs,
            &lw.input_ln,
            hidden,
            cfg.eps,
        );
        let attn = match &lw.attn {
            AttnWeights::Kda(kw) => {
                build_kda_layer(g, params, &format!("l{i}.self_attn"), hn, kw, cfg.kda)?
            }
            AttnWeights::Mla(mw) => {
                build_mla_layer(g, params, &format!("l{i}.self_attn"), hn, mw, cfg.mla)?
            }
        };

        // (4) accumulate attention into the stream
        stream = Some(match stream {
            Some(p) => g.add(p, attn),
            None => attn,
        });

        // (5) mlp residual mix → post norm → FFN
        let ps = stream.expect("stream set after attn");
        let mixed = apply_attn_res(
            g,
            params,
            &format!("l{i}.mlp_res"),
            ps,
            &snapshots,
            &lw.mlp_res_proj,
            &lw.mlp_res_norm,
            rows,
            hidden,
            cfg.eps,
        );
        let mixed = bsh(g, mixed);
        let mn = rms_norm(
            g,
            params,
            &format!("l{i}.post_ln"),
            mixed,
            &lw.post_ln,
            hidden,
            cfg.eps,
        );
        let ffn = match &lw.ffn {
            FfnWeights::Dense(dw) => build_dense_mlp(
                g,
                params,
                &format!("l{i}.mlp"),
                mn,
                dw,
                hidden,
                cfg.dense_inter,
                b,
                s,
                cfg.situ_beta,
                cfg.situ_linear_beta,
            )?,
            FfnWeights::Moe(mw) => build_latent_moe(
                g,
                params,
                &format!("l{i}.block_sparse_moe"),
                mn,
                mw,
                cfg.moe,
            )?,
        };

        // (6) accumulate FFN into the stream → carry to the next layer
        let ps = stream.expect("stream set");
        stream = Some(g.add(ps, ffn));
        hidden_t = stream.expect("stream set");
    }

    // Non-last stage: the boundary is the hidden state + the accumulated
    // snapshots (every layer's AttnRes attends over all of them, so they must
    // cross to the next stage).
    if !last {
        return Ok((hidden_t, snapshots));
    }

    // output attention residual, final norm, untied lm_head
    let final_mixed = apply_attn_res(
        g,
        params,
        "out_res",
        hidden_t,
        &snapshots,
        &w.out_res_proj,
        &w.out_res_norm,
        rows,
        hidden,
        cfg.eps,
    );
    let final_mixed = bsh(g, final_mixed);
    let normed = rms_norm(
        g,
        params,
        "final_norm",
        final_mixed,
        &w.final_norm,
        hidden,
        cfg.eps,
    );
    let n2d = g.reshape_(normed, vec![rows as i64, hidden as i64]);
    let logits = linear(
        g, params, "lm_head", "weight", n2d, &w.lm_head, hidden, cfg.vocab,
    );
    let logits3 = g.reshape_(logits, vec![b as i64, s as i64, cfg.vocab as i64]);
    Ok((logits3, Vec::new()))
}
