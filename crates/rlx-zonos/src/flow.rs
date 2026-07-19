// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0

//! Compiled Zonos backbone: LayerNorm + fused-QKV GQA + interleaved RoPE + silu MLP.
//!
//! Decode graphs use **batch=2** (cond + uncond CFG) per step.

use std::collections::HashMap;

use anyhow::{Context, Result};
use rlx_ir::infer::GraphExt;
use rlx_ir::op::{MaskKind, RopeStyle};
use rlx_ir::{DType, Graph, NodeId, Shape};

use crate::config::ZonosFileConfig;
use crate::weights::WeightMap;

/// CFG batch (conditioned + unconditioned).
pub const CFG_BATCH: usize = 2;

#[derive(Debug, Clone, Copy)]
pub struct ZonosDims {
    pub d_model: usize,
    pub n_heads: usize,
    pub n_kv: usize,
    pub head_dim: usize,
    pub n_layers: usize,
    pub mlp_inter: usize,
    pub eps: f32,
    pub rope_theta: f32,
}

impl ZonosDims {
    pub fn from_cfg(cfg: &ZonosFileConfig) -> Self {
        Self {
            d_model: cfg.backbone.d_model,
            n_heads: cfg.backbone.attn_cfg.num_heads,
            n_kv: cfg.backbone.attn_cfg.num_heads_kv,
            head_dim: cfg.head_dim(),
            n_layers: cfg.backbone.n_layer,
            mlp_inter: cfg.backbone.attn_mlp_d_intermediate,
            eps: cfg.backbone.norm_epsilon,
            rope_theta: 10_000.0,
        }
    }

    pub fn group(&self) -> usize {
        self.n_heads / self.n_kv
    }

    pub fn kv_width(&self) -> usize {
        self.n_kv * self.head_dim
    }
}

fn pname(li: usize, name: &str) -> String {
    format!("L{li}.{name}")
}

/// Metal: store large Linear weights as F16 so the arena stays under the
/// 4 GiB MPSGraph bind cliff; cast to F32 immediately before each `mm`.
pub fn use_f16_linear_weights(device: rlx_runtime::Device) -> bool {
    matches!(device, rlx_runtime::Device::Metal)
}

fn linear_w(g: &mut Graph, name: String, dims: &[usize], f16_w: bool) -> NodeId {
    let dt = if f16_w { DType::F16 } else { DType::F32 };
    // F16 weights stay half-sized in the arena. Metal MatMul / MPSGraph
    // cast to F32 inside the kernel (no IR Cast — those duplicated every
    // Linear into a full F32 buffer and blew past 4 GiB).
    g.param(name, Shape::new(dims, dt))
}

fn mm_linear(g: &mut Graph, x: NodeId, w: NodeId) -> NodeId {
    g.mm(x, w)
}

/// GPT-J / interleaved RoPE on packed `[B, S, n_heads * head_dim]`.
///
/// Prefer `Op::Rope` over the hand-rolled narrow/mul/concat graph — that
/// path dominated Metal decode thunk time (binary_broadcast + concat).
fn apply_rope_packed(
    g: &mut Graph,
    x_pack: NodeId,
    cos: NodeId,
    sin: NodeId,
    head_dim: usize,
) -> NodeId {
    g.rope_styled(x_pack, cos, sin, head_dim, RopeStyle::GptJ)
}

fn transformer_stack_decode(
    g: &mut Graph,
    mut x: NodeId,
    dims: &ZonosDims,
    upper: usize,
    cos: NodeId,
    sin: NodeId,
    mask: NodeId,
    f16_w: bool,
) -> (NodeId, Vec<NodeId>) {
    let ZonosDims {
        d_model: d,
        n_heads: nh,
        n_kv,
        head_dim: hd,
        n_layers,
        mlp_inter: inter,
        eps,
        ..
    } = *dims;
    let b = CFG_BATCH as i64;
    // Batch-major past `[B, U, n_kv, hd]` — concat on seq (axis 1), no
    // time-major transpose dance. `feed_kv_batch_major` writes one token
    // per batch into the resident pad.
    let kv_past = Shape::new(&[CFG_BATCH, upper, n_kv, hd], DType::F32);
    let heads_q = vec![b, 1, nh as i64, hd as i64];
    let heads_kv = vec![b, 1, n_kv as i64, hd as i64];
    let mut kv_outs = Vec::with_capacity(2 * n_layers);

    for li in 0..n_layers {
        let nw = g.param(pname(li, "nw"), Shape::new(&[d], DType::F32));
        let nb = g.param(pname(li, "nb"), Shape::new(&[d], DType::F32));
        let n1 = g.ln(x, nw, nb, eps);

        let qkv_w = linear_w(g, pname(li, "qkv"), &[d, nh * hd + 2 * n_kv * hd], f16_w);
        let qkv = mm_linear(g, n1, qkv_w);
        let q_mm = g.narrow_(qkv, 2, 0, nh * hd);
        let k_mm = g.narrow_(qkv, 2, nh * hd, n_kv * hd);
        let v_mm = g.narrow_(qkv, 2, nh * hd + n_kv * hd, n_kv * hd);
        // RoPE on packed `[B,1,H*D]` then reshape — one Metal `rope` thunk
        // instead of dozens of narrow/mul/concat ops per layer.
        let q = apply_rope_packed(g, q_mm, cos, sin, hd);
        let k = apply_rope_packed(g, k_mm, cos, sin, hd);
        let q = g.reshape_(q, heads_q.clone());
        let k = g.reshape_(k, heads_kv.clone());
        let v = g.reshape_(v_mm, heads_kv.clone());

        // Emit new K/V for resident feed (same nodes as attention inputs).
        kv_outs.push(k);
        kv_outs.push(v);

        let past_k = g.input(format!("past_k_{li}"), kv_past.clone());
        let past_v = g.input(format!("past_v_{li}"), kv_past.clone());
        let k_full = g.concat_(vec![past_k, k], 1);
        let v_full = g.concat_(vec![past_v, v], 1);

        // Native GQA: pass `[B,U+1,n_kv,hd]` — backends map Q heads → KV heads.
        let attn = g.attention_(q, k_full, v_full, mask, nh, hd);
        let attn = g.reshape_(attn, vec![b, 1, d as i64]);
        let ow = linear_w(g, pname(li, "o"), &[d, d], f16_w);
        let attn_o = mm_linear(g, attn, ow);
        x = g.add(x, attn_o);

        let nw2 = g.param(pname(li, "nw2"), Shape::new(&[d], DType::F32));
        let nb2 = g.param(pname(li, "nb2"), Shape::new(&[d], DType::F32));
        let n2 = g.ln(x, nw2, nb2, eps);
        let fc1 = linear_w(g, pname(li, "fc1"), &[d, 2 * inter], f16_w);
        let fused = mm_linear(g, n2, fc1);
        let y = g.narrow_(fused, 2, 0, inter);
        let gate = g.narrow_(fused, 2, inter, inter);
        let gate_s = g.silu(gate);
        let h = g.mul(y, gate_s);
        let fc2 = linear_w(g, pname(li, "fc2"), &[inter, d], f16_w);
        let mlp_o = mm_linear(g, h, fc2);
        x = g.add(x, mlp_o);
    }

    let nw = g.param("norm_f_w", Shape::new(&[d], DType::F32));
    let nb = g.param("norm_f_b", Shape::new(&[d], DType::F32));
    let hidden = g.ln(x, nw, nb, eps);
    (hidden, kv_outs)
}

/// Single-token decode, batch=2 CFG. Outputs `[hidden, new_k_0, new_v_0, …]`.
pub fn build_decode_graph(dims: &ZonosDims, upper: usize, f16_w: bool) -> Graph {
    let d = dims.d_model;
    let half = dims.head_dim / 2;
    let mut g = Graph::new("zonos_decode_cfg2");
    let x = g.input("inputs_embeds", Shape::new(&[CFG_BATCH, 1, d], DType::F32));
    let cos = g.input("rope_cos", Shape::new(&[1, half], DType::F32));
    let sin = g.input("rope_sin", Shape::new(&[1, half], DType::F32));
    let mask = g.input("attn_mask", Shape::new(&[CFG_BATCH, upper + 1], DType::F32));
    let (hidden, kv_outs) = transformer_stack_decode(&mut g, x, dims, upper, cos, sin, mask, f16_w);
    let mut outs = vec![hidden];
    outs.extend(kv_outs);
    g.set_outputs(outs);
    g
}

/// Multi-token causal prefill, batch=2. `seq` is the (padded) length.
///
/// Inputs: `inputs_embeds [2,seq,d]`, `rope_cos/sin [seq,half]`.
/// Outputs: `[hidden [2,seq,d], k_0, v_0, …]` — host crops last real token / KV when padded.
pub fn build_prefill_graph(dims: &ZonosDims, seq: usize, f16_w: bool) -> Graph {
    let ZonosDims {
        d_model: d,
        n_heads: nh,
        n_kv,
        head_dim: hd,
        n_layers,
        mlp_inter: inter,
        eps,
        ..
    } = *dims;
    let half = hd / 2;
    let b = CFG_BATCH as i64;
    let s = seq as i64;
    let mut g = Graph::new("zonos_prefill_cfg2");

    let mut x = g.input(
        "inputs_embeds",
        Shape::new(&[CFG_BATCH, seq, d], DType::F32),
    );
    let cos = g.input("rope_cos", Shape::new(&[seq, half], DType::F32));
    let sin = g.input("rope_sin", Shape::new(&[seq, half], DType::F32));
    let heads_q = vec![b, s, nh as i64, hd as i64];
    let heads_kv = vec![b, s, n_kv as i64, hd as i64];
    let mut kv_outs = Vec::with_capacity(2 * n_layers);

    for li in 0..n_layers {
        let nw = g.param(pname(li, "nw"), Shape::new(&[d], DType::F32));
        let nb = g.param(pname(li, "nb"), Shape::new(&[d], DType::F32));
        let n1 = g.ln(x, nw, nb, eps);

        let qkv_w = linear_w(
            &mut g,
            pname(li, "qkv"),
            &[d, nh * hd + 2 * n_kv * hd],
            f16_w,
        );
        let qkv = mm_linear(&mut g, n1, qkv_w);
        let q_mm = g.narrow_(qkv, 2, 0, nh * hd);
        let k_mm = g.narrow_(qkv, 2, nh * hd, n_kv * hd);
        let v_mm = g.narrow_(qkv, 2, nh * hd + n_kv * hd, n_kv * hd);
        let q = apply_rope_packed(&mut g, q_mm, cos, sin, hd);
        let k = apply_rope_packed(&mut g, k_mm, cos, sin, hd);
        let q = g.reshape_(q, heads_q.clone());
        let k = g.reshape_(k, heads_kv.clone());
        let v = g.reshape_(v_mm, heads_kv.clone());

        kv_outs.push(k);
        kv_outs.push(v);

        // Native GQA — K/V keep `n_kv` heads; attention kernels expand.
        let attn = g.attention_kind(
            q,
            k,
            v,
            nh,
            hd,
            MaskKind::Causal,
            Shape::new(&[CFG_BATCH, seq, nh, hd], DType::F32),
        );
        let attn = g.reshape_(attn, vec![b, s, d as i64]);
        let ow = linear_w(&mut g, pname(li, "o"), &[d, d], f16_w);
        let attn_o = mm_linear(&mut g, attn, ow);
        x = g.add(x, attn_o);

        let nw2 = g.param(pname(li, "nw2"), Shape::new(&[d], DType::F32));
        let nb2 = g.param(pname(li, "nb2"), Shape::new(&[d], DType::F32));
        let n2 = g.ln(x, nw2, nb2, eps);
        let fc1 = linear_w(&mut g, pname(li, "fc1"), &[d, 2 * inter], f16_w);
        let fused = mm_linear(&mut g, n2, fc1);
        let y = g.narrow_(fused, 2, 0, inter);
        let gate = g.narrow_(fused, 2, inter, inter);
        let gate_s = g.silu(gate);
        let h = g.mul(y, gate_s);
        let fc2 = linear_w(&mut g, pname(li, "fc2"), &[inter, d], f16_w);
        let mlp_o = mm_linear(&mut g, h, fc2);
        x = g.add(x, mlp_o);
    }

    let nw = g.param("norm_f_w", Shape::new(&[d], DType::F32));
    let nb = g.param("norm_f_b", Shape::new(&[d], DType::F32));
    let normed = g.ln(x, nw, nb, eps);
    // Full sequence hidden `[B,S,D]` — host crops to real seq when padded.
    let mut outs = vec![normed];
    outs.extend(kv_outs);
    g.set_outputs(outs);
    g
}

pub fn transpose(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = data[r * cols + c];
        }
    }
    out
}

pub fn graph_params(dims: &ZonosDims, w: &WeightMap) -> Result<HashMap<String, Vec<f32>>> {
    let d = dims.d_model;
    let nh = dims.n_heads;
    let n_kv = dims.n_kv;
    let hd = dims.head_dim;
    let q_size = nh * hd;
    let kv_size = n_kv * hd;
    let inter = dims.mlp_inter;
    let mut out = HashMap::new();

    for li in 0..dims.n_layers {
        let pref = format!("backbone.layers.{li}");
        out.insert(
            pname(li, "nw"),
            w.get(&format!("{pref}.norm.weight"))?.to_vec(),
        );
        out.insert(
            pname(li, "nb"),
            w.get(&format!("{pref}.norm.bias"))?.to_vec(),
        );
        out.insert(
            pname(li, "nw2"),
            w.get(&format!("{pref}.norm2.weight"))?.to_vec(),
        );
        out.insert(
            pname(li, "nb2"),
            w.get(&format!("{pref}.norm2.bias"))?.to_vec(),
        );

        let in_proj = w.get(&format!("{pref}.mixer.in_proj.weight"))?;
        anyhow::ensure!(
            in_proj.len() == (q_size + 2 * kv_size) * d,
            "in_proj size mismatch layer {li}"
        );
        // Keep fused QKV as one `[d, q+k+v]` Linear — one Metal GEMM per layer
        // instead of three (CFG decode is bandwidth-bound on weight traffic).
        out.insert(
            pname(li, "qkv"),
            transpose(in_proj, q_size + 2 * kv_size, d),
        );
        out.insert(
            pname(li, "o"),
            transpose(w.get(&format!("{pref}.mixer.out_proj.weight"))?, d, d),
        );

        let fc1 = w.get(&format!("{pref}.mlp.fc1.weight"))?;
        anyhow::ensure!(fc1.len() == 2 * inter * d, "fc1 size layer {li}");
        out.insert(pname(li, "fc1"), transpose(fc1, 2 * inter, d));
        out.insert(
            pname(li, "fc2"),
            transpose(w.get(&format!("{pref}.mlp.fc2.weight"))?, d, inter),
        );
    }

    out.insert("norm_f_w".into(), w.get("backbone.norm_f.weight")?.to_vec());
    out.insert("norm_f_b".into(), w.get("backbone.norm_f.bias")?.to_vec());
    Ok(out)
}

pub fn bucket_decode_mask(past_seq: usize, upper: usize) -> Vec<f32> {
    (0..=upper)
        .map(|i| if i < past_seq || i == upper { 1.0 } else { 0.0 })
        .collect()
}

/// Batch=2 mask: two identical keep-mask rows.
pub fn bucket_decode_mask_cfg2(past_seq: usize, upper: usize) -> Vec<f32> {
    let row = bucket_decode_mask(past_seq, upper);
    let mut out = Vec::with_capacity(CFG_BATCH * row.len());
    for _ in 0..CFG_BATCH {
        out.extend_from_slice(&row);
    }
    out
}

pub fn rope_cos_sin(dims: &ZonosDims, pos: usize) -> (Vec<f32>, Vec<f32>) {
    let half = dims.head_dim / 2;
    let mut cos = vec![0.0f32; half];
    let mut sin = vec![0.0f32; half];
    for i in 0..half {
        let inv = 1.0f32 / dims.rope_theta.powf(i as f32 / half as f32);
        let a = pos as f32 * inv;
        cos[i] = a.cos();
        sin[i] = a.sin();
    }
    (cos, sin)
}

pub fn rope_tables(dims: &ZonosDims, seq: usize) -> (Vec<f32>, Vec<f32>) {
    let half = dims.head_dim / 2;
    let mut cos = vec![0.0f32; seq * half];
    let mut sin = vec![0.0f32; seq * half];
    for t in 0..seq {
        for i in 0..half {
            let inv = 1.0f32 / dims.rope_theta.powf(i as f32 / half as f32);
            let a = t as f32 * inv;
            cos[t * half + i] = a.cos();
            sin[t * half + i] = a.sin();
        }
    }
    (cos, sin)
}

pub fn set_params(
    compiled: &mut rlx_runtime::CompiledGraph,
    params: &HashMap<String, Vec<f32>>,
) -> Result<()> {
    for (name, data) in params {
        compiled.set_param(name, data);
    }
    Ok(())
}

pub fn compile_decode(
    dims: &ZonosDims,
    params: &HashMap<String, Vec<f32>>,
    upper: usize,
    device: rlx_runtime::Device,
) -> Result<rlx_runtime::CompiledGraph> {
    let f16_w = use_f16_linear_weights(device);
    let g = build_decode_graph(dims, upper, f16_w);
    let mut compiled = rlx_runtime::Session::new(device).compile(g);
    set_params(&mut compiled, params).context("set zonos decode params")?;
    Ok(compiled)
}

pub fn compile_prefill(
    dims: &ZonosDims,
    params: &HashMap<String, Vec<f32>>,
    seq: usize,
    device: rlx_runtime::Device,
) -> Result<rlx_runtime::CompiledGraph> {
    let f16_w = use_f16_linear_weights(device);
    let g = build_prefill_graph(dims, seq, f16_w);
    let mut compiled = rlx_runtime::Session::new(device).compile(g);
    set_params(&mut compiled, params).context("set zonos prefill params")?;
    Ok(compiled)
}
