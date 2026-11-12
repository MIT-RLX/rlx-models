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

//! Compiled text-trunk prefill graph (batch 1, fixed `seq`, f32).
//!
//! `embed → embed_norm → N × (attn_norm → attention → attn_sconv → +res →
//! mlp_norm → (dense | MoE) → mlp_sconv → +res) → last row → norm → unembed · mup`.
//!
//! Weight names follow the eager schema ([`crate::eager::TextWeights`]); use
//! [`graph_weight_map`] to turn those tensors into the shapes this graph reads.
//!
//! What differs from the eager path, and why:
//! - **Relative bias** is built in-graph with `GatherElements` over host-built
//!   distance indices, then masked; causal and sliding-window masks are additive
//!   host tables (`-1e30`), so no backend needs a sliding-window attention kind.
//! - **Router weights** use `σ(z_i) / Σ σ(z_j)` over the selected routed logits
//!   and the shared logits, which is exactly `exp(logσ(z_i) − logsumexp(logσ(z)))`.
//! - **Expert banks** are stored pre-transposed (`[E, K, N]`) by
//!   [`graph_weight_map`], so the graph never makes a second copy of the bank.
//! - **KV heads** are replicated to the query head count in-graph.
//!
//! Verified on CPU only: `GatherElements` is the one op here that other backends
//! may not lower.

use crate::config::InklingTextConfig;
use crate::eager::{TextWeights, log_scaling_tau};
use anyhow::{Result, anyhow, bail};
use rlx_core::flow_util::WeightMapSource;
use rlx_core::weight_map::WeightMap;
use rlx_flow::{BuiltModel, CompileProfile, Emit, ModelFlow};
use rlx_ir::hir::HirMut;
use rlx_ir::op::{MaskKind, Op, PadMode};
use rlx_ir::{DType, HirGraphExt, HirNodeId, Shape};
use std::collections::HashMap;

pub const IN_MASK_LOCAL: &str = "mask_local";
pub const IN_MASK_GLOBAL: &str = "mask_global";
pub const IN_REL_IDX_LOCAL: &str = "rel_idx_local";
pub const IN_REL_IDX_GLOBAL: &str = "rel_idx_global";
pub const IN_REL_KEEP_GLOBAL: &str = "rel_keep_global";
pub const IN_LOG_TAU: &str = "log_tau";

const MASKED: f32 = -1e30;

/// Host-built per-`seq` tables the graph takes as inputs.
///
/// Only the tables the config needs exist: a config with no sliding layers has
/// no local tables, and `log_tau` exists only when the checkpoint sets
/// `log_scaling_n_floor`.
pub struct InklingTables {
    seq: usize,
    entries: Vec<(&'static str, Vec<f32>)>,
}

impl InklingTables {
    pub fn new(cfg: &InklingTextConfig, seq: usize) -> Self {
        let layers = 0..cfg.num_hidden_layers;
        let has_local = layers.clone().any(|l| cfg.is_sliding(l));
        let has_global = layers.clone().any(|l| !cfg.is_sliding(l));
        let mut entries = Vec::new();
        let dist = |tq: usize, tk: usize| tq as i64 - tk as i64;
        let table = |f: &dyn Fn(i64) -> f32| -> Vec<f32> {
            (0..seq * seq).map(|i| f(dist(i / seq, i % seq))).collect()
        };
        if has_local {
            let win = cfg.sliding_window_size as i64;
            entries.push((
                IN_MASK_LOCAL,
                table(&|d| if (0..win).contains(&d) { 0.0 } else { MASKED }),
            ));
            entries.push((IN_REL_IDX_LOCAL, table(&|d| d.clamp(0, win - 1) as f32)));
        }
        if has_global {
            let ext = cfg.rel_extent as i64;
            entries.push((
                IN_MASK_GLOBAL,
                table(&|d| if d >= 0 { 0.0 } else { MASKED }),
            ));
            entries.push((IN_REL_IDX_GLOBAL, table(&|d| d.clamp(0, ext - 1) as f32)));
            entries.push((
                IN_REL_KEEP_GLOBAL,
                table(&|d| if (0..ext).contains(&d) { 1.0 } else { 0.0 }),
            ));
            if let Some(tau) = log_scaling_tau(cfg, seq) {
                entries.push((IN_LOG_TAU, tau));
            }
        }
        Self { seq, entries }
    }

    pub fn seq(&self) -> usize {
        self.seq
    }

    /// `(name, data)` pairs to pass to `CompiledGraph::run` after `input_ids`.
    pub fn inputs(&self) -> Vec<(&str, &[f32])> {
        self.entries
            .iter()
            .map(|(n, d)| (*n, d.as_slice()))
            .collect()
    }
}

fn put(
    m: &mut HashMap<String, (Vec<f32>, Vec<usize>)>,
    key: String,
    data: &[f32],
    shape: &[usize],
) {
    m.insert(key, (data.to_vec(), shape.to_vec()));
}

/// Eager-schema tensors → the shapes this graph reads.
///
/// Differences from the eager layout: tensors carry shapes; depthwise conv
/// weights are `[C, K]`; routed expert banks are transposed to `[E, K, N]`;
/// shared experts are split per expert (`shared_gate.{s}`, …).
pub fn graph_weight_map(cfg: &InklingTextConfig, w: &TextWeights) -> Result<WeightMap> {
    let get = |k: &str| -> Result<&[f32]> {
        w.tensors
            .get(k)
            .map(Vec::as_slice)
            .ok_or_else(|| anyhow!("missing weight {k}"))
    };
    let (h, v, k) = (cfg.hidden_size, cfg.vocab_size, cfg.conv_kernel_size);
    let mut m = HashMap::new();
    put(&mut m, "embed".into(), get("embed")?, &[v, h]);
    put(&mut m, "embed_norm".into(), get("embed_norm")?, &[h]);
    put(&mut m, "norm".into(), get("norm")?, &[h]);
    put(&mut m, "unembed".into(), get("unembed")?, &[v, h]);

    for l in 0..cfg.num_hidden_layers {
        let (nh, nkv, hd) = cfg.attn_heads(l);
        let (q_dim, kv_dim) = (nh * hd, nkv * hd);
        let p = format!("layers.{l}");
        let mut add = |name: &str, shape: &[usize]| -> Result<()> {
            let key = format!("{p}.{name}");
            let data = get(&key)?;
            if data.len() != shape.iter().product::<usize>() {
                bail!("{key}: {} values, expected shape {shape:?}", data.len());
            }
            put(&mut m, key, data, shape);
            Ok(())
        };
        add("attn_norm", &[h])?;
        add("mlp_norm", &[h])?;
        add("attn_sconv", &[h, k])?;
        add("mlp_sconv", &[h, k])?;
        add("q_norm", &[hd])?;
        add("k_norm", &[hd])?;
        add("k_sconv", &[kv_dim, k])?;
        add("v_sconv", &[kv_dim, k])?;
        add("rel_proj", &[cfg.d_rel, cfg.rel_extent_for_layer(l)])?;
        add("wq", &[q_dim, h])?;
        add("wk", &[kv_dim, h])?;
        add("wv", &[kv_dim, h])?;
        add("wr", &[nh * cfg.d_rel, h])?;
        add("wo", &[h, q_dim])?;
        if cfg.is_dense_mlp(l) {
            let i = cfg.dense_intermediate_size;
            add("gate", &[i, h])?;
            add("up", &[i, h])?;
            add("down", &[h, i])?;
            add("mlp_global_scale", &[1])?;
            continue;
        }
        let i = cfg.moe_intermediate_size;
        let (ne, ns) = (cfg.n_routed_experts, cfg.n_shared_experts);
        add("gate_weight", &[ne + ns, h])?;
        add("gate_bias", &[ne])?;
        add("gate_global_scale", &[1])?;

        // [E, 2i, h] → [E, h, 2i]; [E, h, i] → [E, i, h].
        let w13 = get(&format!("{p}.expert_w13"))?;
        let w2 = get(&format!("{p}.expert_w2"))?;
        if w13.len() != ne * 2 * i * h || w2.len() != ne * h * i {
            bail!("{p}: expert bank sizes do not match config");
        }
        let (mut t13, mut t2) = (vec![0.0; w13.len()], vec![0.0; w2.len()]);
        for e in 0..ne {
            for n in 0..2 * i {
                for c in 0..h {
                    t13[e * h * 2 * i + c * 2 * i + n] = w13[e * 2 * i * h + n * h + c];
                }
            }
            for n in 0..h {
                for c in 0..i {
                    t2[e * i * h + c * h + n] = w2[e * h * i + n * i + c];
                }
            }
        }
        put(&mut m, format!("{p}.expert_w13"), &t13, &[ne, h, 2 * i]);
        put(&mut m, format!("{p}.expert_w2"), &t2, &[ne, i, h]);

        for (name, rows, cols) in [
            ("shared_gate", i, h),
            ("shared_up", i, h),
            ("shared_down", h, i),
        ] {
            let all = get(&format!("{p}.{name}"))?;
            if all.len() != ns * rows * cols {
                bail!("{p}.{name}: size does not match config");
            }
            for s in 0..ns {
                let part = &all[s * rows * cols..(s + 1) * rows * cols];
                put(&mut m, format!("{p}.{name}.{s}"), part, &[rows, cols]);
            }
        }
    }
    Ok(WeightMap::from_tensors(m))
}

fn rmsnorm(
    emit: &mut Emit<'_>,
    key: &str,
    x: HirNodeId,
    dim: usize,
    eps: f32,
) -> Result<HirNodeId> {
    let g = emit.load_param(key, false)?;
    let zb = emit.synth_param(
        &format!("{key}.zb"),
        vec![0.0; dim],
        Shape::new(&[dim], DType::F32),
    );
    Ok(HirMut::new(emit.hir()).rms_norm(x, g, zb, eps))
}

fn linear(emit: &mut Emit<'_>, key: &str, x: HirNodeId) -> Result<HirNodeId> {
    let w = emit.load_param(key, true)?;
    Ok(HirMut::new(emit.hir()).mm(x, w))
}

/// `y + depthwise_causal_conv(y)` on `[1, seq, channels]` (weights `[C, K]`).
fn short_conv(
    emit: &mut Emit<'_>,
    key: &str,
    y: HirNodeId,
    seq: usize,
    c: usize,
    kernel: usize,
) -> Result<HirNodeId> {
    let w = emit.load_param(key, false)?; // [C, K]
    let mut gb = HirMut::new(emit.hir());
    let wt = gb.transpose_(w, vec![1, 0]); // [K, C]
    let padded = gb.pad_(
        y,
        vec![[0, 0], [kernel - 1, 0], [0, 0]],
        PadMode::Constant(0.0),
    );
    let mut acc = y;
    for tap in 0..kernel {
        let shifted = gb.narrow_(padded, 1, tap, seq);
        let row = gb.narrow_(wt, 0, tap, 1);
        let row = gb.reshape_(row, vec![1, 1, c as i64]);
        let term = gb.mul(shifted, row);
        acc = gb.add(acc, term);
    }
    Ok(acc)
}

fn attention(
    emit: &mut Emit<'_>,
    cfg: &InklingTextConfig,
    l: usize,
    x: HirNodeId,
    seq: usize,
) -> Result<HirNodeId> {
    let f = DType::F32;
    let (nh, nkv, hd) = cfg.attn_heads(l);
    let (q_dim, kv_dim) = (nh * hd, nkv * hd);
    let sliding = cfg.is_sliding(l);
    let ext = cfg.rel_extent_for_layer(l);
    let p = format!("layers.{l}");
    let (s, nh_i, nkv_i, hd_i) = (seq as i64, nh as i64, nkv as i64, hd as i64);

    let q = linear(emit, &format!("{p}.wq"), x)?;
    let k = linear(emit, &format!("{p}.wk"), x)?;
    let v = linear(emit, &format!("{p}.wv"), x)?;
    let r = linear(emit, &format!("{p}.wr"), x)?;
    let k = short_conv(
        emit,
        &format!("{p}.k_sconv"),
        k,
        seq,
        kv_dim,
        cfg.conv_kernel_size,
    )?;
    let v = short_conv(
        emit,
        &format!("{p}.v_sconv"),
        v,
        seq,
        kv_dim,
        cfg.conv_kernel_size,
    )?;

    let q4 = HirMut::new(emit.hir()).reshape_(q, vec![1, s, nh_i, hd_i]);
    let k4 = HirMut::new(emit.hir()).reshape_(k, vec![1, s, nkv_i, hd_i]);
    let mut q4 = rmsnorm(emit, &format!("{p}.q_norm"), q4, hd, cfg.rms_norm_eps)?;
    let k4 = rmsnorm(emit, &format!("{p}.k_norm"), k4, hd, cfg.rms_norm_eps)?;

    let proj = emit.load_param(&format!("{p}.rel_proj"), false)?; // [d_rel, ext]
    let (mask_in, idx_in) = if sliding {
        (IN_MASK_LOCAL, IN_REL_IDX_LOCAL)
    } else {
        (IN_MASK_GLOBAL, IN_REL_IDX_GLOBAL)
    };
    let mask = emit.flow_input(mask_in)?.hir_id();
    let idx = emit.flow_input(idx_in)?.hir_id();
    let keep = if sliding {
        None
    } else {
        Some(emit.flow_input(IN_REL_KEEP_GLOBAL)?.hir_id())
    };
    let tau = if !sliding && cfg.log_scaling_n_floor.is_some() {
        Some(emit.flow_input(IN_LOG_TAU)?.hir_id())
    } else {
        None
    };

    let mut gb = HirMut::new(emit.hir());
    // Relative bias: mix the d_rel profiles per (token, head), then pick the
    // entry at each query/key distance.
    let r2 = gb.reshape_(r, vec![s * nh_i, cfg.d_rel as i64]);
    let mixed = gb.mm(r2, proj); // [seq*nh, ext]
    let mixed = gb.reshape_(mixed, vec![s, nh_i, ext as i64]);
    let mixed = gb.transpose_(mixed, vec![1, 0, 2]); // [nh, seq, ext]
    let idx = gb.expand_(idx, vec![nh_i, s, s]);
    let mut bias = gb.add_node(
        Op::GatherElements { axis: 2 },
        vec![mixed, idx],
        Shape::new(&[nh, seq, seq], f),
    );
    if let Some(keep) = keep {
        bias = gb.mul(bias, keep);
    }
    if let Some(tau) = tau {
        let tau_q = gb.reshape_(tau, vec![1, s, 1, 1]);
        q4 = gb.mul(q4, tau_q);
        let tau_b = gb.reshape_(tau, vec![1, s, 1]);
        bias = gb.mul(bias, tau_b);
    }
    let bias = gb.reshape_(bias, vec![1, nh_i, s, s]);
    let bias = gb.add(bias, mask);

    // Repeat KV heads to the query head count (head = kv * groups + g).
    let groups = (nh / nkv) as i64;
    let v4 = gb.reshape_(v, vec![1, s, nkv_i, 1, hd_i]);
    let k5 = gb.reshape_(k4, vec![1, s, nkv_i, 1, hd_i]);
    let k5 = gb.expand_(k5, vec![1, s, nkv_i, groups, hd_i]);
    let v5 = gb.expand_(v4, vec![1, s, nkv_i, groups, hd_i]);
    let qf = gb.reshape_(q4, vec![1, s, nh_i * hd_i]);
    let kf = gb.reshape_(k5, vec![1, s, nh_i * hd_i]);
    let vf = gb.reshape_(v5, vec![1, s, nh_i * hd_i]);

    let out = gb.add_node(
        rlx_ir::ops::attention::attention_kind_op(
            nh,
            hd,
            None,
            MaskKind::Bias,
            Some(1.0 / hd as f32), // q/k are RMS-normalised: 1/d, not 1/sqrt(d)
            None,
        ),
        vec![qf, kf, vf, bias],
        Shape::new(&[1, seq, q_dim], f),
    );
    let _ = ext;
    linear(emit, &format!("{p}.wo"), out)
}

fn swiglu(gb: &mut HirMut<'_>, gate: HirNodeId, up: HirNodeId) -> HirNodeId {
    let a = gb.silu(gate);
    gb.mul(a, up)
}

fn dense_mlp(emit: &mut Emit<'_>, l: usize, x: HirNodeId) -> Result<HirNodeId> {
    let p = format!("layers.{l}");
    let gate = linear(emit, &format!("{p}.gate"), x)?;
    let up = linear(emit, &format!("{p}.up"), x)?;
    let mid = swiglu(&mut HirMut::new(emit.hir()), gate, up);
    let down = linear(emit, &format!("{p}.down"), mid)?;
    let scale = emit.load_param(&format!("{p}.mlp_global_scale"), false)?;
    Ok(HirMut::new(emit.hir()).mul(down, scale))
}

fn moe_mlp(
    emit: &mut Emit<'_>,
    cfg: &InklingTextConfig,
    l: usize,
    x: HirNodeId,
    seq: usize,
) -> Result<HirNodeId> {
    let f = DType::F32;
    let p = format!("layers.{l}");
    let (h, i) = (cfg.hidden_size, cfg.moe_intermediate_size);
    let (ne, ns, top_k) = (
        cfg.n_routed_experts,
        cfg.n_shared_experts,
        cfg.num_experts_per_tok,
    );
    let rows = seq as i64;

    let router = emit.load_param(&format!("{p}.gate_weight"), true)?; // [h, ne+ns]
    let ebias = emit.load_param(&format!("{p}.gate_bias"), false)?;
    let gscale = emit.load_param(&format!("{p}.gate_global_scale"), false)?;
    let route_scale = emit.synth_param(
        &format!("{p}.route_scale"),
        vec![cfg.route_scale],
        Shape::new(&[1], f),
    );
    let w13 = emit.load_param(&format!("{p}.expert_w13"), false)?; // [E, h, 2i]
    let w2 = emit.load_param(&format!("{p}.expert_w2"), false)?; // [E, i, h]
    let mut shared = Vec::with_capacity(ns);
    for s in 0..ns {
        shared.push((
            emit.load_param(&format!("{p}.shared_gate.{s}"), true)?,
            emit.load_param(&format!("{p}.shared_up.{s}"), true)?,
            emit.load_param(&format!("{p}.shared_down.{s}"), true)?,
        ));
    }

    let mut gb = HirMut::new(emit.hir());
    let h2 = gb.reshape_(x, vec![rows, h as i64]);
    let logits = gb.mm(h2, router); // [seq, ne+ns]
    let routed = gb.narrow_(logits, 1, 0, ne);
    let shared_logits = gb.narrow_(logits, 1, ne, ns);
    let scores = gb.activation(
        rlx_ir::op::Activation::Sigmoid,
        routed,
        Shape::new(&[seq, ne], f),
    );
    let bias = gb.reshape_(ebias, vec![1, ne as i64]);
    let choice = gb.add(scores, bias);
    let top = gb.add_node(
        Op::TopK { k: top_k },
        vec![choice],
        Shape::new(&[seq, top_k], f),
    );

    let picked = gb.add_node(
        Op::GatherElements { axis: 1 },
        vec![routed, top],
        Shape::new(&[seq, top_k], f),
    );
    let sig_sel = gb.activation(
        rlx_ir::op::Activation::Sigmoid,
        picked,
        Shape::new(&[seq, top_k], f),
    );
    let sig_sh = gb.activation(
        rlx_ir::op::Activation::Sigmoid,
        shared_logits,
        Shape::new(&[seq, ns], f),
    );
    let total = {
        let a = gb.sum(sig_sel, vec![1], true);
        let b = gb.sum(sig_sh, vec![1], true);
        gb.add(a, b)
    };
    let scale = gb.mul(gscale, route_scale); // [1]
    let norm = gb.div(scale, total); // [seq, 1]
    let w_sel = gb.mul(sig_sel, norm); // [seq, top_k]
    let w_sh = gb.mul(sig_sh, norm); // [seq, ns]

    let mut acc: Option<HirNodeId> = None;
    for ki in 0..top_k {
        let col = gb.narrow_(top, 1, ki, 1);
        let eidx = gb.reshape_(col, vec![rows]);
        let gate_up = gb.grouped_matmul(h2, w13, eidx); // [seq, 2i]
        let g = gb.narrow_(gate_up, 1, 0, i);
        let u = gb.narrow_(gate_up, 1, i, i);
        let mid = swiglu(&mut gb, g, u);
        let down = gb.grouped_matmul(mid, w2, eidx); // [seq, h]
        let wk = gb.narrow_(w_sel, 1, ki, 1);
        let weighted = gb.mul(down, wk);
        acc = Some(match acc {
            Some(a) => gb.add(a, weighted),
            None => weighted,
        });
    }
    let mut out = acc.ok_or_else(|| anyhow!("num_experts_per_tok must be >= 1"))?;
    for (s, (sg, su, sd)) in shared.into_iter().enumerate() {
        let g = gb.mm(h2, sg);
        let u = gb.mm(h2, su);
        let mid = swiglu(&mut gb, g, u);
        let ws = gb.narrow_(w_sh, 1, s, 1);
        let mid = gb.mul(mid, ws);
        let y = gb.mm(mid, sd);
        out = gb.add(out, y);
    }
    Ok(gb.reshape_(out, vec![1, rows, h as i64]))
}

/// Build the prefill graph for exactly `seq` tokens; output `logits` is the last
/// position only, `[1, vocab]`.
pub fn build_text_flow(
    cfg: &InklingTextConfig,
    weights: &mut WeightMap,
    seq: usize,
) -> Result<BuiltModel> {
    if seq == 0 {
        bail!("rlx-inkling: seq must be > 0");
    }
    let f = DType::F32;
    let (h, eps) = (cfg.hidden_size, cfg.rms_norm_eps);
    let tables = InklingTables::new(cfg, seq);

    let mut flow = ModelFlow::new("inkling_text")
        .with_profile(CompileProfile::llama32_prefill())
        .input("input_ids", Shape::new(&[1, seq], f));
    for (name, data) in &tables.entries {
        let shape = match *name {
            IN_LOG_TAU => Shape::new(&[seq], f),
            IN_MASK_LOCAL | IN_MASK_GLOBAL => Shape::new(&[1, 1, seq, seq], f),
            _ => Shape::new(&[1, seq, seq], f),
        };
        debug_assert_eq!(data.len(), shape.num_elements().unwrap_or(0));
        flow = flow.input(*name, shape);
    }
    flow = flow
        .zero_beta_named("inkling.zero_beta.hidden", h)
        .embed("embed");

    let hs = Shape::new(&[1, seq, h], f);
    if cfg.use_embed_norm {
        let hs = hs.clone();
        flow = flow.plugin_named("embed_norm", move |emit, prev| {
            let x = prev
                .ok_or_else(|| anyhow!("embed_norm needs the embedding"))?
                .hir_id();
            let y = rmsnorm(emit, "embed_norm", x, h, eps)?;
            Ok(Some(emit.wrap(y, hs.clone())))
        });
    }
    for l in 0..cfg.num_hidden_layers {
        let cfg = cfg.clone();
        let hs = hs.clone();
        flow = flow.plugin_named(format!("layer{l}"), move |emit, prev| {
            let x = prev
                .ok_or_else(|| anyhow!("layer{l} needs a hidden input"))?
                .hir_id();
            let p = format!("layers.{l}");
            let n1 = rmsnorm(emit, &format!("{p}.attn_norm"), x, h, eps)?;
            let a = attention(emit, &cfg, l, n1, seq)?;
            let a = short_conv(
                emit,
                &format!("{p}.attn_sconv"),
                a,
                seq,
                h,
                cfg.conv_kernel_size,
            )?;
            let x1 = HirMut::new(emit.hir()).add(x, a);
            let n2 = rmsnorm(emit, &format!("{p}.mlp_norm"), x1, h, eps)?;
            let m = if cfg.is_dense_mlp(l) {
                dense_mlp(emit, l, n2)?
            } else {
                moe_mlp(emit, &cfg, l, n2, seq)?
            };
            let m = short_conv(
                emit,
                &format!("{p}.mlp_sconv"),
                m,
                seq,
                h,
                cfg.conv_kernel_size,
            )?;
            let out = HirMut::new(emit.hir()).add(x1, m);
            Ok(Some(emit.wrap(out, hs.clone())))
        });
    }

    let (out_vocab, mup) = (cfg.output_vocab_size(), cfg.logits_mup_width_multiplier);
    let vocab = cfg.vocab_size;
    flow = flow
        .gather_last_token_at(1, seq)
        .rms_norm("norm", eps)
        .plugin_named("unembed", move |emit, prev| {
            let x = prev
                .ok_or_else(|| anyhow!("unembed needs the final hidden"))?
                .hir_id();
            let w = emit.load_param("unembed", true)?; // [h, vocab]
            let mut gb = HirMut::new(emit.hir());
            let x = gb.reshape_(x, vec![1, h as i64]);
            let y = gb.mm(x, w); // [1, vocab]
            // transformers: lm_head(hidden / mup), padded vocab tail cut off.
            let y = if out_vocab < vocab {
                gb.narrow_(y, 1, 0, out_vocab)
            } else {
                y
            };
            let y = if mup != 1.0 {
                let m = emit.synth_param("unembed.inv_mup", vec![1.0 / mup], Shape::new(&[1], f));
                HirMut::new(emit.hir()).mul(y, m)
            } else {
                y
            };
            Ok(Some(emit.wrap(y, Shape::new(&[1, out_vocab], f))))
        })
        .output("logits");
    flow.build_with(&mut WeightMapSource(weights), None)
}

/// Prefill logits for the last token via the compiled graph (CPU).
pub fn forward_logits_compiled(
    cfg: &InklingTextConfig,
    weights: &TextWeights,
    input_ids: &[u32],
    device: rlx_runtime::Device,
) -> Result<Vec<f32>> {
    if input_ids.is_empty() {
        bail!("rlx-inkling: empty input_ids");
    }
    if let Some(&bad) = input_ids.iter().find(|&&t| t as usize >= cfg.vocab_size) {
        bail!("token id {bad} out of range (vocab={})", cfg.vocab_size);
    }
    let seq = input_ids.len();
    let mut wm = graph_weight_map(cfg, weights)?;
    let built = build_text_flow(cfg, &mut wm, seq)?;
    let mut compiled = rlx_core::flow_util::compile_built(built, device)?;
    let ids: Vec<f32> = input_ids.iter().map(|&t| t as f32).collect();
    let tables = InklingTables::new(cfg, seq);
    let mut inputs: Vec<(&str, &[f32])> = vec![("input_ids", ids.as_slice())];
    inputs.extend(tables.inputs());
    compiled
        .run(&inputs)
        .into_iter()
        .next()
        .ok_or_else(|| anyhow!("rlx-inkling: graph returned no output"))
}
