// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! The SnapViT self-supervised objective graph: a maskable ViT forward over
//! `n_crops` views → L2-normalized `[CLS]` embeddings → DINO cross-view loss.
//! Its gradient w.r.t. the block weights is the local Hessian-diagonal signal
//! (Eq. 2/3); the same graph, with different masks, is the xNES fitness
//! evaluator (embeddings as its output).

use rlx_ir::{DType, Graph, NodeId, Shape};

use crate::dino::{build_dino_loss, l2_normalize, pair_mask};
use crate::vit::config::VitConfig;
use crate::vit::forward::{ParamSpec, build_vit_graph};

const F: DType = DType::F32;

/// A built SnapViT loss graph (its single output is the scalar DINO loss).
pub struct SnapVitLoss {
    pub graph: Graph,
    pub params: Vec<ParamSpec>,
    pub loss: NodeId,
    /// L2-normalized `[CLS]` embeddings `[n_crops, H]` (the fitness output).
    pub embeddings: NodeId,
    pub hidden_input: NodeId,
    pub head_mask_input: NodeId,
    pub ffn_mask_input: NodeId,
    pub teacher_targets_input: NodeId,
    pub pair_mask_input: NodeId,
    pub n_crops: usize,
    pub n_global: usize,
    pub active_pairs: usize,
    pub cfg: VitConfig,
}

/// Build the SnapViT loss graph. `batch = n_crops`; the first `n_global` crops
/// are the teacher (global) views.
pub fn build_snapvit_loss(
    cfg: &VitConfig,
    n_crops: usize,
    n_global: usize,
    temp_s: f32,
) -> SnapVitLoss {
    let h = cfg.hidden_size;
    let vg = build_vit_graph(cfg, n_crops);
    let crate::vit::forward::VitGraph {
        mut graph,
        output,
        hidden_input,
        head_mask_input,
        ffn_mask_input,
        params,
        ..
    } = vg;

    // Student projections = L2-normalized CLS of every crop (embedding-as-logits,
    // per SnapViT's head-free SSL objective).
    let cls = crate::vit::forward::extract_cls(&mut graph, output, n_crops, h); // [n_crops, H]
    let embeddings = l2_normalize(&mut graph, cls, h);
    let teacher_targets_input = graph.input("teacher_targets", Shape::new(&[n_global, h], F));
    let pair_mask_input = graph.input("pair_mask", Shape::new(&[n_global, n_crops], F));
    let (_, active_pairs) = pair_mask(n_global, n_crops);
    let loss = build_dino_loss(
        &mut graph,
        embeddings,
        teacher_targets_input,
        pair_mask_input,
        temp_s,
        active_pairs,
    );
    graph.set_outputs(vec![loss]);

    SnapVitLoss {
        graph,
        params,
        loss,
        embeddings,
        hidden_input,
        head_mask_input,
        ffn_mask_input,
        teacher_targets_input,
        pair_mask_input,
        n_crops,
        n_global,
        active_pairs,
        cfg: cfg.clone(),
    }
}
