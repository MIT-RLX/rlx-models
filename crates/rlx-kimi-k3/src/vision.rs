// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! Kimi-K3 vision tower (MoonViT3d) + patchmergerv2 projector.
//!
//! A pre-norm ViT: each block is `x += wo(SDPA(rope(qkv(rms0(x)))))` then
//! `x += fc1(gelu_tanh(fc0(rms1(x))))`, all **bias-free**, attention in a wider
//! `qkv_hidden` space (`qkv_hidden != hidden`), interleaved 2D RoPE (GPT-J). After
//! `final_layernorm`, the **patchmergerv2** projector does a 2×2 spatial merge
//! (temporal mean is identity for images), then `proj.2(erf_gelu(proj.0(x)))` and
//! an RMSNorm `post_norm`, lifting `hidden → text_hidden`.
//!
//! Patch embedding + learned/interpolated positional embedding are host-side
//! preprocessing; this graph consumes the already-embedded patch hidden states
//! `[1, L, hidden]` plus precomputed 2D-RoPE `cos`/`sin` tables `[L, head_dim/2]`.

use crate::common::reg;
use anyhow::{Result, ensure};
use rlx_ir::hir::{HirMut, HirNodeId};
use rlx_ir::op::{MaskKind, RopeStyle};
use rlx_ir::{DType, HirGraphExt, Shape};
use std::collections::HashMap;

type Params = HashMap<String, Vec<f32>>;

#[derive(Debug, Clone, Copy)]
pub struct VisionDims {
    pub hidden: usize,      // residual-stream width (1024)
    pub qkv_hidden: usize,  // attention width = num_heads*head_dim (1536)
    pub num_heads: usize,   // 12
    pub head_dim: usize,    // 128
    pub inter: usize,       // FFN width (4096)
    pub merge: usize,       // spatial merge kernel (2 → 2×2)
    pub text_hidden: usize, // LM hidden the projector lifts to (7168)
    pub proj_mid: usize,    // projector hidden (4096)
    pub eps: f32,
    pub grid_h: usize,
    pub grid_w: usize,
}

impl VisionDims {
    pub fn seq_len(&self) -> usize {
        self.grid_h * self.grid_w
    }
    pub fn merge_in(&self) -> usize {
        self.merge * self.merge * self.hidden
    }
}

#[derive(Debug, Clone, Default)]
pub struct VisionBlockWeights {
    pub norm0: Vec<f32>, // [hidden]
    pub wqkv: Vec<f32>,  // [hidden, 3*qkv_hidden]
    pub wo: Vec<f32>,    // [qkv_hidden, hidden]
    pub norm1: Vec<f32>, // [hidden]
    pub fc0: Vec<f32>,   // [hidden, inter]
    pub fc1: Vec<f32>,   // [inter, hidden]
}

#[derive(Debug, Clone, Default)]
pub struct VisionWeights {
    pub blocks: Vec<VisionBlockWeights>,
    pub final_norm: Vec<f32>, // [hidden]
    pub proj0: Vec<f32>,      // [merge_in, proj_mid]
    pub proj2: Vec<f32>,      // [proj_mid, text_hidden]
    pub post_norm: Vec<f32>,  // [text_hidden]
}

fn rms(
    g: &mut HirMut,
    params: &mut Params,
    name: &str,
    x: HirNodeId,
    w: &[f32],
    n: usize,
    eps: f32,
) -> HirNodeId {
    let gamma = reg(g, params, name, w.to_vec(), &[n]);
    let zb = reg(g, params, &format!("{name}.zero_beta"), vec![0f32; n], &[n]);
    g.rms_norm(x, gamma, zb, eps)
}

fn lin(
    g: &mut HirMut,
    params: &mut Params,
    name: &str,
    x: HirNodeId,
    w: &[f32],
    i: usize,
    o: usize,
) -> HirNodeId {
    let wid = reg(g, params, name, w.to_vec(), &[i, o]);
    g.mm(x, wid)
}

/// One ViT encoder block on `x` `[1, L, hidden]`; returns `[1, L, hidden]`.
#[allow(clippy::too_many_arguments)]
fn block(
    g: &mut HirMut,
    params: &mut Params,
    prefix: &str,
    x: HirNodeId,
    cos: HirNodeId,
    sin: HirNodeId,
    w: &VisionBlockWeights,
    d: VisionDims,
) -> HirNodeId {
    let l = d.seq_len();
    let (hid, qh, nh, hd) = (d.hidden, d.qkv_hidden, d.num_heads, d.head_dim);
    let f = DType::F32;

    // attention sublayer
    let xn = rms(
        g,
        params,
        &format!("{prefix}.norm0"),
        x,
        &w.norm0,
        hid,
        d.eps,
    );
    let xn2d = g.reshape_(xn, vec![l as i64, hid as i64]);
    let qkv = lin(
        g,
        params,
        &format!("{prefix}.wqkv"),
        xn2d,
        &w.wqkv,
        hid,
        3 * qh,
    );
    let qkv3 = g.reshape_(qkv, vec![1, l as i64, (3 * qh) as i64]);
    let q = g.narrow_(qkv3, 2, 0, qh);
    let k = g.narrow_(qkv3, 2, qh, qh);
    let v = g.narrow_(qkv3, 2, 2 * qh, qh);
    // interleaved 2D RoPE on q,k (heads packed in the last axis).
    let q = g.rope_styled(q, cos, sin, hd, RopeStyle::GptJ);
    let k = g.rope_styled(k, cos, sin, hd, RopeStyle::GptJ);
    let attn = g.attention_kind(q, k, v, nh, hd, MaskKind::None, Shape::new(&[1, l, qh], f));
    let attn2d = g.reshape_(attn, vec![l as i64, qh as i64]);
    let ao = lin(g, params, &format!("{prefix}.wo"), attn2d, &w.wo, qh, hid);
    let ao = g.reshape_(ao, vec![1, l as i64, hid as i64]);
    let x = g.add(x, ao);

    // FFN sublayer: fc1(gelu_tanh(fc0(rms1(x))))
    let yn = rms(
        g,
        params,
        &format!("{prefix}.norm1"),
        x,
        &w.norm1,
        hid,
        d.eps,
    );
    let yn2d = g.reshape_(yn, vec![l as i64, hid as i64]);
    let h0 = lin(
        g,
        params,
        &format!("{prefix}.mlp.fc0"),
        yn2d,
        &w.fc0,
        hid,
        d.inter,
    );
    let h0 = g.gelu_approx(h0); // tanh-GELU
    let h1 = lin(
        g,
        params,
        &format!("{prefix}.mlp.fc1"),
        h0,
        &w.fc1,
        d.inter,
        hid,
    );
    let h1 = g.reshape_(h1, vec![1, l as i64, hid as i64]);
    g.add(x, h1)
}

/// Build the vision tower + patchmergerv2 on patch hidden states
/// `hidden` `[1, L, hidden]` and 2D-RoPE `cos`/`sin` `[L, head_dim/2]`.
/// Returns projected image tokens `[M, text_hidden]` with `M = (H/merge)*(W/merge)`.
pub fn build_vision(
    g: &mut HirMut,
    params: &mut Params,
    hidden: HirNodeId,
    cos: HirNodeId,
    sin: HirNodeId,
    w: &VisionWeights,
    d: VisionDims,
) -> Result<HirNodeId> {
    ensure!(
        d.grid_h.is_multiple_of(d.merge) && d.grid_w.is_multiple_of(d.merge),
        "grid not divisible by merge"
    );
    let (hid, m) = (d.hidden, d.merge);
    let (mh, mw) = (d.grid_h / m, d.grid_w / m);
    let n_merged = mh * mw;

    let mut x = hidden;
    for (i, bw) in w.blocks.iter().enumerate() {
        x = block(g, params, &format!("vision.blocks.{i}"), x, cos, sin, bw, d);
    }
    let x = rms(
        g,
        params,
        "vision.final_layernorm",
        x,
        &w.final_norm,
        hid,
        d.eps,
    );

    // patchmergerv2: 2×2 spatial merge (temporal mean = identity for images).
    // [1, H*W, C] → [H/m, m, W/m, m, C] → permute [H/m, W/m, m, m, C] → [n_merged, m*m*C]
    let g5 = g.reshape_(
        x,
        vec![mh as i64, m as i64, mw as i64, m as i64, hid as i64],
    );
    let g5 = g.transpose_(g5, vec![0, 2, 1, 3, 4]);
    let merged = g.reshape_(g5, vec![n_merged as i64, d.merge_in() as i64]);

    // proj.2(erf_gelu(proj.0(x))) → RMSNorm post_norm
    let z = lin(
        g,
        params,
        "mm_projector.proj.0",
        merged,
        &w.proj0,
        d.merge_in(),
        d.proj_mid,
    );
    let z = g.gelu(z); // erf-GELU
    let z = lin(
        g,
        params,
        "mm_projector.proj.2",
        z,
        &w.proj2,
        d.proj_mid,
        d.text_hidden,
    );
    Ok(rms(
        g,
        params,
        "mm_projector.post_norm",
        z,
        &w.post_norm,
        d.text_hidden,
        d.eps,
    ))
}
