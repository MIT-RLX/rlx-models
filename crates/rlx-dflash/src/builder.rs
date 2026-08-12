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

//! DFlash drafter graph.
//!
//! Transcribed from llama.cpp `src/models/dflash.cpp`. Two stages:
//!
//! **Encoder** — `concat(target taps) -> fc -> enc.output_norm`. `fc` is
//! `[n_taps*hidden, hidden]`; the norm is applied AFTER `fc` (dflash.cpp:100
//! calls it "encoder hidden_norm (after fc)", and 208/211 order it that way).
//! Getting that order backwards is silent garbage, so it is worth stating.
//!
//! **Decoder** — `n` pre-norm blocks: `attn_norm -> qkv -> per-head QK-norm ->
//! RoPE -> SWA attention -> o_proj -> residual -> ffn_norm -> SwiGLU ->
//! residual`, then `output_norm`. Simpler than the Muse Glimmer target block:
//! two norms per layer, no attention gate, no post-norms, every layer local.
//!
//! The drafter has NO embedding and NO LM head — it reuses the target's. That is
//! what makes it Eagle-style rather than a standalone small model.

use std::collections::HashMap;

use anyhow::{Result, anyhow};
use rlx_core::weight_loader::WeightLoader;
use rlx_ir::infer::GraphExt;
use rlx_ir::op::MaskKind;
use rlx_ir::{DType, Graph, NodeId, Op, Shape};

use crate::config::DflashConfig;

type Packed = HashMap<String, (rlx_ir::quant::QuantScheme, Vec<usize>)>;
type Params = HashMap<String, Vec<f32>>;

/// Load a norm gain (F32, 1-D).
fn load_norm(
    g: &mut Graph,
    params: &mut Params,
    weights: &mut dyn WeightLoader,
    key: &str,
) -> Result<NodeId> {
    if let Some(id) = g.param_id(key) {
        return Ok(id);
    }
    let (data, shape) = weights
        .take(key)
        .map_err(|e| anyhow!("dflash: missing norm {key}: {e}"))?;
    let id = g.param(key, Shape::new(&shape, DType::F32));
    params.insert(key.to_string(), data);
    Ok(id)
}

/// Load a projection, keeping K-quant weights packed when the loader offers it.
fn load_proj(
    g: &mut Graph,
    params: &mut Params,
    packed: &mut Packed,
    weights: &mut dyn WeightLoader,
    key: &str,
) -> Result<(NodeId, Option<rlx_ir::quant::QuantScheme>)> {
    if let Some(id) = g.param_id(key) {
        return Ok((id, packed.get(key).map(|(s, _)| *s)));
    }
    if let Some((scheme, shape)) = weights.packed_meta(key) {
        let nbytes = weights
            .tensor_bytes_borrowed(key)
            .ok_or_else(|| anyhow!("dflash: packed {key} has no bytes"))?
            .len();
        let id = g.param(key, Shape::new(&[nbytes], DType::U8));
        packed.insert(key.to_string(), (scheme, shape));
        return Ok((id, Some(scheme)));
    }
    // F32 fallback: store transposed so a plain `mm` works.
    let (data, shape) = weights
        .take_transposed(key)
        .map_err(|e| anyhow!("dflash: missing weight {key}: {e}"))?;
    let id = g.param(key, Shape::new(&shape, DType::F32));
    params.insert(key.to_string(), data);
    Ok((id, None))
}

fn emit_proj(
    g: &mut Graph,
    x: NodeId,
    w: NodeId,
    scheme: Option<rlx_ir::quant::QuantScheme>,
    out: Shape,
) -> NodeId {
    match scheme {
        Some(s) => g.add_node(Op::DequantMatMul { scheme: s }, vec![x, w], out),
        None => g.mm(x, w),
    }
}

/// RMSNorm each head over `head_dim` with a shared `[head_dim]` gain.
#[allow(clippy::too_many_arguments)]
fn per_head_rms(
    g: &mut Graph,
    x: NodeId,
    gamma: NodeId,
    beta: NodeId,
    batch: usize,
    seq: usize,
    heads: usize,
    head_dim: usize,
    eps: f32,
) -> NodeId {
    let flat = (batch * seq * heads) as i64;
    let r = g.reshape_(x, vec![flat, head_dim as i64]);
    let n = g.rms_norm(r, gamma, beta, eps);
    g.reshape_(n, vec![batch as i64, seq as i64, (heads * head_dim) as i64])
}

/// Expand `n_kv` KV heads to `n_kv * group` by repeating each head's slice —
/// same narrow+concat idiom as rlx-llama32 (no `expand` op on `Graph`).
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

/// Build the full DFlash drafter graph.
///
/// Input `dflash_taps`: `[batch, seq, n_taps * hidden]` — the target model's
/// residual streams at `cfg.target_layers`, concatenated in that order.
/// Output: `[batch, seq, hidden]` draft hidden states, ready for the TARGET's
/// `lm_head`.
pub fn build_dflash_graph(
    cfg: &DflashConfig,
    weights: &mut dyn WeightLoader,
    batch: usize,
    seq: usize,
    packed: &mut Packed,
) -> Result<(Graph, Params)> {
    let mut g = Graph::new("dflash");
    let mut params: Params = HashMap::new();
    let f = DType::F32;

    let h = cfg.hidden_size;
    let dh = cfg.head_dim;
    let nh = cfg.num_attention_heads;
    let nkv = cfg.num_key_value_heads;
    let group = cfg.kv_group_size();
    let eps = cfg.rms_norm_eps as f32;

    let zero_h = {
        let id = g.param("dflash.zero_beta.hidden", Shape::new(&[h], f));
        params.insert("dflash.zero_beta.hidden".into(), vec![0f32; h]);
        id
    };
    let zero_dh = {
        let id = g.param("dflash.zero_beta.head_dim", Shape::new(&[dh], f));
        params.insert("dflash.zero_beta.head_dim".into(), vec![0f32; dh]);
        id
    };

    // RoPE tables. DFlash shares the target's rope_theta and head_dim.
    let half = dh / 2;
    let (mut cos, mut sin) = (vec![0f32; seq * half], vec![0f32; seq * half]);
    for p in 0..seq {
        for i in 0..half {
            let inv = 1.0f64 / cfg.rope_theta.powf(2.0 * i as f64 / dh as f64);
            let a = p as f64 * inv;
            cos[p * half + i] = a.cos() as f32;
            sin[p * half + i] = a.sin() as f32;
        }
    }
    let cos_id = g.param("dflash.rope.cos", Shape::new(&[seq, half], f));
    params.insert("dflash.rope.cos".into(), cos);
    let sin_id = g.param("dflash.rope.sin", Shape::new(&[seq, half], f));
    params.insert("dflash.rope.sin".into(), sin);

    // ── Encoder: concat(taps) -> fc -> enc.output_norm ──────────────────
    let taps = g.input(
        "dflash_taps",
        Shape::new(&[batch, seq, cfg.fused_input_dim()], f),
    );
    let (fc_w, fc_s) = load_proj(&mut g, &mut params, packed, weights, "fc.weight")?;
    let fused = emit_proj(&mut g, taps, fc_w, fc_s, Shape::new(&[batch, seq, h], f));
    let enc_n = load_norm(&mut g, &mut params, weights, "enc.output_norm.weight")?;
    let mut x = g.rms_norm(fused, enc_n, zero_h, eps);

    // ── Decoder blocks ──────────────────────────────────────────────────
    // Every DFlash layer is sliding-window (`pattern = [T,T,T,T,T]`), and the
    // window matches the target's.
    let mask = match cfg.sliding_window {
        Some(w) if w > 0 => MaskKind::SlidingWindow(w),
        _ => MaskKind::Causal,
    };

    for il in 0..cfg.num_hidden_layers {
        let p = format!("blk.{il}");
        let res = x;

        let an = load_norm(
            &mut g,
            &mut params,
            weights,
            &format!("{p}.attn_norm.weight"),
        )?;
        let xn = g.rms_norm(x, an, zero_h, eps);

        let (qw, qs) = load_proj(
            &mut g,
            &mut params,
            packed,
            weights,
            &format!("{p}.attn_q.weight"),
        )?;
        let (kw, ks) = load_proj(
            &mut g,
            &mut params,
            packed,
            weights,
            &format!("{p}.attn_k.weight"),
        )?;
        let (vw, vs) = load_proj(
            &mut g,
            &mut params,
            packed,
            weights,
            &format!("{p}.attn_v.weight"),
        )?;
        let q = emit_proj(
            &mut g,
            xn,
            qw,
            qs,
            Shape::new(&[batch, seq, cfg.q_proj_dim()], f),
        );
        let k = emit_proj(
            &mut g,
            xn,
            kw,
            ks,
            Shape::new(&[batch, seq, cfg.kv_proj_dim()], f),
        );
        let v = emit_proj(
            &mut g,
            xn,
            vw,
            vs,
            Shape::new(&[batch, seq, cfg.kv_proj_dim()], f),
        );

        // Per-head QK-norm, then RoPE — same order as the reference.
        let qn = load_norm(
            &mut g,
            &mut params,
            weights,
            &format!("{p}.attn_q_norm.weight"),
        )?;
        let kn = load_norm(
            &mut g,
            &mut params,
            weights,
            &format!("{p}.attn_k_norm.weight"),
        )?;
        let q = per_head_rms(&mut g, q, qn, zero_dh, batch, seq, nh, dh, eps);
        let k = per_head_rms(&mut g, k, kn, zero_dh, batch, seq, nkv, dh, eps);

        // GGUF checkpoints are converter-permuted for interleaved (GPT-J) RoPE.
        let q = g.rope_styled(q, cos_id, sin_id, dh, rlx_ir::RopeStyle::GptJ);
        let k = g.rope_styled(k, cos_id, sin_id, dh, rlx_ir::RopeStyle::GptJ);

        let k_rep = repeat_kv(&mut g, k, nkv, dh, group);
        let v_rep = repeat_kv(&mut g, v, nkv, dh, group);
        let attn_shape = Shape::new(&[batch, seq, cfg.q_proj_dim()], f);
        let attn = g.attention_kind_opts(q, k_rep, v_rep, nh, dh, mask, attn_shape, None, None);

        let (ow, os) = load_proj(
            &mut g,
            &mut params,
            packed,
            weights,
            &format!("{p}.attn_output.weight"),
        )?;
        let attn_out = emit_proj(&mut g, attn, ow, os, Shape::new(&[batch, seq, h], f));
        let ffn_inp = g.add(res, attn_out);

        let fn_ = load_norm(
            &mut g,
            &mut params,
            weights,
            &format!("{p}.ffn_norm.weight"),
        )?;
        let normed = g.rms_norm(ffn_inp, fn_, zero_h, eps);
        let inter = cfg.intermediate_size;
        let (gw, gs) = load_proj(
            &mut g,
            &mut params,
            packed,
            weights,
            &format!("{p}.ffn_gate.weight"),
        )?;
        let (uw, us) = load_proj(
            &mut g,
            &mut params,
            packed,
            weights,
            &format!("{p}.ffn_up.weight"),
        )?;
        let gate = emit_proj(&mut g, normed, gw, gs, Shape::new(&[batch, seq, inter], f));
        let up = emit_proj(&mut g, normed, uw, us, Shape::new(&[batch, seq, inter], f));
        let act = {
            let s = g.silu(gate);
            g.mul(s, up)
        };
        let (dw, ds) = load_proj(
            &mut g,
            &mut params,
            packed,
            weights,
            &format!("{p}.ffn_down.weight"),
        )?;
        let ffn_out = emit_proj(&mut g, act, dw, ds, Shape::new(&[batch, seq, h], f));
        x = g.add(ffn_inp, ffn_out);
    }

    let on = load_norm(&mut g, &mut params, weights, "output_norm.weight")?;
    let out = g.rms_norm(x, on, zero_h, eps);
    g.set_outputs(vec![out]);
    Ok((g, params))
}
