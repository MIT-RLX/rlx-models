// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! DINO projection head `g_λ` (the shared head in GLARE; K = 8192).
//!
//! Structure: an `n_mlp_layers`-layer GELU MLP (`in_dim → hidden … → bottleneck`),
//! L2-normalize, then a final linear `bottleneck → out_k`. (DINO weight-norms
//! the last layer; here it is a plain linear — functionally a `bottleneck→K`
//! projection, trained/initialized directly.)

use std::collections::HashMap;

use rlx_ir::infer::GraphExt;
use rlx_ir::{DType, Graph, NodeId, Shape};

use super::loss::l2_normalize;
use crate::vit::forward::ParamSpec;

const F: DType = DType::F32;

/// DINO head geometry.
#[derive(Clone, Debug)]
pub struct DinoHeadConfig {
    pub in_dim: usize,
    pub hidden: usize,
    pub bottleneck: usize,
    pub out_k: usize,
    pub n_mlp_layers: usize,
}

impl DinoHeadConfig {
    /// The paper's head (`hidden=2048`, `bottleneck=256`, 3-layer MLP).
    pub fn dino(in_dim: usize, out_k: usize) -> Self {
        Self {
            in_dim,
            hidden: 2048,
            bottleneck: 256,
            out_k,
            n_mlp_layers: 3,
        }
    }
    /// A tiny head for tests.
    pub fn small(in_dim: usize, out_k: usize) -> Self {
        Self {
            in_dim,
            hidden: 64,
            bottleneck: 32,
            out_k,
            n_mlp_layers: 2,
        }
    }
}

fn hp(g: &mut Graph, params: &mut Vec<ParamSpec>, name: String, dims: &[usize]) -> NodeId {
    let node = g.param(name.clone(), Shape::new(dims, F));
    params.push(ParamSpec {
        name,
        node,
        dims: dims.to_vec(),
    });
    node
}

/// Emit the head over `x [N, in_dim]` → `[N, out_k]`, registering its params.
pub fn build_dino_head(
    g: &mut Graph,
    x: NodeId,
    cfg: &DinoHeadConfig,
    prefix: &str,
    params: &mut Vec<ParamSpec>,
) -> NodeId {
    let mut cur = x;
    let mut dim = cfg.in_dim;
    for i in 0..cfg.n_mlp_layers.saturating_sub(1) {
        let w = hp(
            g,
            params,
            format!("{prefix}.mlp.{i}.weight"),
            &[dim, cfg.hidden],
        );
        let b = hp(g, params, format!("{prefix}.mlp.{i}.bias"), &[cfg.hidden]);
        let z = g.mm(cur, w);
        let z = g.add(z, b);
        cur = g.gelu(z);
        dim = cfg.hidden;
    }
    let wp = hp(
        g,
        params,
        format!("{prefix}.proj.weight"),
        &[dim, cfg.bottleneck],
    );
    let bp = hp(g, params, format!("{prefix}.proj.bias"), &[cfg.bottleneck]);
    let z = g.mm(cur, wp);
    let z = g.add(z, bp);
    let z = l2_normalize(g, z, cfg.bottleneck);
    let wl = hp(
        g,
        params,
        format!("{prefix}.last.weight"),
        &[cfg.bottleneck, cfg.out_k],
    );
    g.mm(z, wl)
}

/// Deterministic small pseudo-random head init (checkpoints carry no DINO head).
pub fn init_head_params(
    cfg: &DinoHeadConfig,
    prefix: &str,
    seed: u32,
) -> HashMap<String, Vec<f32>> {
    let mut sd = seed;
    let mut rnd = |n: usize, scale: f32| -> Vec<f32> {
        sd = sd
            .wrapping_mul(1664525)
            .wrapping_add(1013904223)
            .wrapping_add(0x9E3779B9);
        let mut s = sd;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                ((s >> 8) as f32 / u32::MAX as f32 - 0.5) * 2.0 * scale
            })
            .collect()
    };
    let mut p = HashMap::new();
    let mut dim = cfg.in_dim;
    for i in 0..cfg.n_mlp_layers.saturating_sub(1) {
        let fan = (dim as f32).sqrt().recip();
        p.insert(
            format!("{prefix}.mlp.{i}.weight"),
            rnd(dim * cfg.hidden, fan),
        );
        p.insert(format!("{prefix}.mlp.{i}.bias"), vec![0.0; cfg.hidden]);
        dim = cfg.hidden;
    }
    let fanp = (dim as f32).sqrt().recip();
    p.insert(
        format!("{prefix}.proj.weight"),
        rnd(dim * cfg.bottleneck, fanp),
    );
    p.insert(format!("{prefix}.proj.bias"), vec![0.0; cfg.bottleneck]);
    let fanl = (cfg.bottleneck as f32).sqrt().recip();
    p.insert(
        format!("{prefix}.last.weight"),
        rnd(cfg.bottleneck * cfg.out_k, fanl),
    );
    p
}
