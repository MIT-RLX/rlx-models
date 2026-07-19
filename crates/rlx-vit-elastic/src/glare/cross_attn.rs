// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! GLARE regional cross-attention module (Eqs. 5–6):
//! `CA(z, Z) = softmax(τ·(z·W_q)(Z·W_k)ᵀ)·(Z·W_v)`, extracting the semantic
//! context of region queries `z [R, H]` with respect to the view `Z [S, H]`.

use rlx_ir::infer::GraphExt;
use rlx_ir::{DType, Graph, NodeId, Shape};

use crate::vit::forward::ParamSpec;

const F: DType = DType::F32;

/// Emit `CA(q_in, kv_in) → [R, H]`, registering `W_q/W_k/W_v` into `params`.
pub fn build_cross_attention(
    g: &mut Graph,
    q_in: NodeId,
    kv_in: NodeId,
    h: usize,
    tau: f32,
    prefix: &str,
    params: &mut Vec<ParamSpec>,
) -> NodeId {
    let mut wp = |g: &mut Graph, name: String| -> NodeId {
        let node = g.param(name.clone(), Shape::new(&[h, h], F));
        params.push(ParamSpec {
            name,
            node,
            dims: vec![h, h],
        });
        node
    };
    let wq = wp(g, format!("{prefix}.q.weight"));
    let wk = wp(g, format!("{prefix}.k.weight"));
    let wv = wp(g, format!("{prefix}.v.weight"));

    let q = g.mm(q_in, wq); // [R, H]
    let k = g.mm(kv_in, wk); // [S, H]
    let v = g.mm(kv_in, wv); // [S, H]
    let kt = g.transpose_(k, vec![1, 0]); // [H, S]
    let scores = g.mm(q, kt); // [R, S]
    let tau_c = g.constant(tau as f64, F);
    let scores = g.mul(scores, tau_c);
    let attn = g.sm(scores, -1); // [R, S]
    g.mm(attn, v) // [R, H]
}

/// Initialize cross-attention `W_q/W_k/W_v` (small random, per-fan scale).
pub fn init_cross_attention_params(
    h: usize,
    prefix: &str,
    seed: u32,
) -> std::collections::HashMap<String, Vec<f32>> {
    let mut sd = seed;
    let fan = (h as f32).sqrt().recip();
    let mut rnd = |n: usize| -> Vec<f32> {
        sd = sd
            .wrapping_mul(1664525)
            .wrapping_add(1013904223)
            .wrapping_add(0x9E3779B9);
        let mut s = sd;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                ((s >> 8) as f32 / u32::MAX as f32 - 0.5) * 2.0 * fan
            })
            .collect()
    };
    let mut p = std::collections::HashMap::new();
    p.insert(format!("{prefix}.q.weight"), rnd(h * h));
    p.insert(format!("{prefix}.k.weight"), rnd(h * h));
    p.insert(format!("{prefix}.v.weight"), rnd(h * h));
    p
}
