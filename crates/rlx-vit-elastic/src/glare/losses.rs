// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! The GLARE forward core (adapter ViT → CLS + patch + cross-attention region
//! representations → shared DINO head) and the student loss
//! `L = w_glob·L_glob + w_loc·L_loc + w_reg·L_reg` (Eq. 10).

use rlx_ir::infer::GraphExt;
use rlx_ir::{DType, Graph, NodeId, Shape};

use crate::dino::head::{DinoHeadConfig, build_dino_head};
use crate::dino::loss::dino_ce_aligned;
use crate::vit::config::VitConfig;
use crate::vit::forward::{ParamSpec, VitGraph, build_vit_graph_with};

use super::adapter::AdapterConfig;
use super::cross_attn::build_cross_attention;
use super::regions::RegionLayout;

const F: DType = DType::F32;

/// Loss weights for the three GLARE consistency terms.
#[derive(Clone, Copy, Debug)]
pub struct GlareWeights {
    pub glob: f32,
    pub loc: f32,
    pub reg: f32,
}

impl Default for GlareWeights {
    fn default() -> Self {
        Self {
            glob: 1.0,
            loc: 1.0,
            reg: 1.0,
        }
    }
}

/// The shared GLARE forward core: outputs the head logits `[M, K]` where
/// `M = 1 (CLS) + n_patch + n_regions`.
pub struct GlareCore {
    pub graph: Graph,
    pub all_logits: NodeId,
    pub m: usize,
    pub k: usize,
    pub n_patch: usize,
    pub n_regions: usize,
    pub hidden_input: NodeId,
    pub head_mask_input: NodeId,
    pub ffn_mask_input: NodeId,
    /// Frozen backbone block weights.
    pub backbone_params: Vec<ParamSpec>,
    /// Frozen constant region-pooling matrix.
    pub region_pool: ParamSpec,
    /// Trainable GLARE params (adapter + cross-attention + head).
    pub trainable_params: Vec<ParamSpec>,
    pub cfg: VitConfig,
}

/// Build the GLARE forward core (its output is `all_logits`).
pub fn build_glare_core(
    cfg: &VitConfig,
    head_cfg: &DinoHeadConfig,
    adapter: &AdapterConfig,
    region: &RegionLayout,
    tau: f32,
) -> GlareCore {
    let h = cfg.hidden_size;
    let n_patch = cfg.num_patches();
    let patch_row_base = cfg.patch_row_base();

    let vg = build_vit_graph_with(cfg, 1, Some(adapter.opts()));
    let VitGraph {
        mut graph,
        output,
        hidden_input,
        head_mask_input,
        ffn_mask_input,
        params: backbone_params,
        adapter_params,
        ..
    } = vg;
    let cls = crate::vit::forward::extract_cls(&mut graph, output, 1, h); // [1, H]

    // Patch tokens [n_patch, H] — extracted via a LAST-axis narrow (transpose
    // seq to the end) to avoid the Metal-NaN middle-axis narrow backward.
    let out_t = graph.transpose_(output, vec![0, 2, 1]); // [1, H, seq]
    let patches_t = graph.narrow_(out_t, 2, patch_row_base, n_patch); // [1, H, n_patch]
    let patches_hp = graph.transpose_(patches_t, vec![0, 2, 1]); // [1, n_patch, H]
    let patches = graph.reshape_(patches_hp, vec![n_patch as i64, h as i64]);

    // Region representations via a frozen block-average pooling matrix.
    let rp_node = graph.param(
        "glare.region_pool",
        Shape::new(&[region.n_regions, n_patch], F),
    );
    let region_pool = ParamSpec {
        name: "glare.region_pool".to_string(),
        node: rp_node,
        dims: vec![region.n_regions, n_patch],
    };
    let region_reps = graph.mm(rp_node, patches); // [R, H]

    // Cross-attention region context (trainable).
    let mut ca_params: Vec<ParamSpec> = Vec::new();
    let ca_reps = build_cross_attention(
        &mut graph,
        region_reps,
        patches,
        h,
        tau,
        "glare.ca",
        &mut ca_params,
    );

    // Shared head over [CLS ; patches ; region-context].
    let all_reps = graph.concat_(vec![cls, patches, ca_reps], 0); // [M, H]
    let m = 1 + n_patch + region.n_regions;
    let mut head_params: Vec<ParamSpec> = Vec::new();
    let all_logits = build_dino_head(
        &mut graph,
        all_reps,
        head_cfg,
        "glare.head",
        &mut head_params,
    );
    graph.set_outputs(vec![all_logits]);

    let mut trainable = adapter_params;
    trainable.extend(ca_params);
    trainable.extend(head_params);

    GlareCore {
        graph,
        all_logits,
        m,
        k: head_cfg.out_k,
        n_patch,
        n_regions: region.n_regions,
        hidden_input,
        head_mask_input,
        ffn_mask_input,
        backbone_params,
        region_pool,
        trainable_params: trainable,
        cfg: cfg.clone(),
    }
}

/// The student graph: the core plus the three DINO consistency losses against
/// teacher target inputs (`cls_target`, `patch_target`, `reg_target`).
pub struct GlareStudent {
    pub graph: Graph,
    pub total_loss: NodeId,
    pub trainable_params: Vec<ParamSpec>,
    pub backbone_params: Vec<ParamSpec>,
    pub region_pool: ParamSpec,
    pub k: usize,
    pub n_patch: usize,
    pub n_regions: usize,
    pub cfg: VitConfig,
}

/// Wrap a core into the student loss graph (its output is the scalar total loss).
pub fn build_glare_student(core: GlareCore, temp_s: f32, w: GlareWeights) -> GlareStudent {
    let GlareCore {
        mut graph,
        all_logits,
        k,
        n_patch,
        n_regions,
        backbone_params,
        region_pool,
        trainable_params,
        cfg,
        ..
    } = core;

    let cls_logits = graph.narrow_(all_logits, 0, 0, 1); // [1, K]
    let patch_logits = graph.narrow_(all_logits, 0, 1, n_patch); // [n_patch, K]
    let reg_logits = graph.narrow_(all_logits, 0, 1 + n_patch, n_regions); // [R, K]

    let cls_t = graph.input("cls_target", Shape::new(&[1, k], F));
    let patch_t = graph.input("patch_target", Shape::new(&[n_patch, k], F));
    let reg_t = graph.input("reg_target", Shape::new(&[n_regions, k], F));

    let l_glob = dino_ce_aligned(&mut graph, cls_logits, cls_t, 1, temp_s);
    let l_loc = dino_ce_aligned(&mut graph, patch_logits, patch_t, n_patch, temp_s);
    let l_reg = dino_ce_aligned(&mut graph, reg_logits, reg_t, n_regions, temp_s);

    let wg = graph.constant(w.glob as f64, F);
    let wl = graph.constant(w.loc as f64, F);
    let wr = graph.constant(w.reg as f64, F);
    let a = graph.mul(l_glob, wg);
    let b = graph.mul(l_loc, wl);
    let c = graph.mul(l_reg, wr);
    let ab = graph.add(a, b);
    let total_loss = graph.add(ab, c);
    graph.set_outputs(vec![total_loss]);

    GlareStudent {
        graph,
        total_loss,
        trainable_params,
        backbone_params,
        region_pool,
        k,
        n_patch,
        n_regions,
        cfg,
    }
}
