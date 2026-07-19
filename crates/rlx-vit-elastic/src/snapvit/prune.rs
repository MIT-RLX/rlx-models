// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! Prunability score (SnapViT Eq. 8) and elastic structured pruning (Eq. 9).
//!
//! `P_structure = local_structure · c_block(structure)`, where `c` is the
//! per-block scaling learned by xNES. Structures are ranked by their **per-
//! parameter** score and removed lowest-first until the target parameter
//! sparsity is met (heads and FFN channels compete on the same footing).

use std::collections::HashSet;

use crate::vit::config::{FfnKind, VitConfig};

use super::local::LocalScores;
use super::mask::masks_from_kept;

/// Number of xNES block-scaling coefficients: one per (layer, head) + one per
/// layer FFN. Layout: `[head blocks (L·nh)] ++ [ffn layer blocks (L)]`.
pub fn coeffs_len(cfg: &VitConfig) -> usize {
    cfg.num_hidden_layers * cfg.num_attention_heads + cfg.num_hidden_layers
}

/// Parameters removed when an attention head is pruned (qkv rows/cols + bias +
/// proj rows for that head).
pub fn head_param_count(cfg: &VitConfig) -> usize {
    let hd = cfg.head_dim();
    let h = cfg.hidden_size;
    4 * hd * h + 3 * hd
}

/// Parameters removed when one FFN inner channel is pruned.
pub fn ffn_param_count(cfg: &VitConfig) -> usize {
    let h = cfg.hidden_size;
    match cfg.ffn_kind {
        FfnKind::Gelu => 2 * h + 1,
        FfnKind::PackedSwiGLU => 3 * h + 2,
    }
}

/// Per-structure prunability `P = local ⊙ (M·c)` (`c` positive multipliers).
#[derive(Clone, Debug)]
pub struct Prunability {
    pub head: Vec<f32>,
    pub ffn: Vec<f32>,
}

/// Combine local scores with the block-scaling vector `c` (length
/// [`coeffs_len`]; already exponentiated to be positive).
pub fn prunability(cfg: &VitConfig, local: &LocalScores, c: &[f32]) -> Prunability {
    let nh = cfg.num_attention_heads;
    let inner = cfg.ffn_inner();
    let depth = cfg.num_hidden_layers;
    let ffn_block0 = depth * nh;

    let head = local
        .head
        .iter()
        .enumerate()
        .map(|(i, &s)| s * c.get(i).copied().unwrap_or(1.0).max(0.0))
        .collect();
    let ffn = local
        .ffn
        .iter()
        .enumerate()
        .map(|(j, &s)| {
            let layer = j / inner;
            s * c.get(ffn_block0 + layer).copied().unwrap_or(1.0).max(0.0)
        })
        .collect();
    Prunability { head, ffn }
}

/// A pruning solution at one sparsity: the masks + accounting.
#[derive(Clone, Debug)]
pub struct PruneResult {
    pub sparsity: f32,
    pub head_mask: Vec<f32>,
    pub ffn_mask: Vec<f32>,
    pub kept_heads: HashSet<usize>,
    pub kept_ffn: HashSet<usize>,
    pub heads_pruned: usize,
    pub ffn_pruned: usize,
    /// Fraction of prunable parameters removed.
    pub param_reduction: f32,
}

/// Rank structures by per-parameter prunability and remove lowest-first until
/// the removed parameter fraction reaches `sparsity` (in `[0, 1)`).
pub fn prune_at(cfg: &VitConfig, p: &Prunability, sparsity: f32) -> PruneResult {
    let n_heads = p.head.len();
    let n_ffn = p.ffn.len();
    let hp = head_param_count(cfg);
    let fp = ffn_param_count(cfg);
    let total: f64 = (n_heads * hp + n_ffn * fp) as f64;
    let budget = sparsity.clamp(0.0, 0.99) as f64 * total;

    // (key = per-param score, is_head, index, params). Lowest key removed first.
    let mut items: Vec<(f32, bool, usize, usize)> = Vec::with_capacity(n_heads + n_ffn);
    for (i, &s) in p.head.iter().enumerate() {
        items.push((s / hp as f32, true, i, hp));
    }
    for (j, &s) in p.ffn.iter().enumerate() {
        items.push((s / fp as f32, false, j, fp));
    }
    items.sort_by(|a, b| a.0.total_cmp(&b.0));

    let mut kept_heads: HashSet<usize> = (0..n_heads).collect();
    let mut kept_ffn: HashSet<usize> = (0..n_ffn).collect();
    let mut removed = 0.0f64;
    let mut heads_pruned = 0;
    let mut ffn_pruned = 0;
    for &(_, is_head, idx, params) in &items {
        if removed >= budget {
            break;
        }
        // Keep at least one head per layer and a floor of FFN channels to avoid
        // degenerate all-pruned layers.
        if is_head {
            if kept_heads.len() <= cfg.num_hidden_layers {
                continue;
            }
            kept_heads.remove(&idx);
            heads_pruned += 1;
        } else {
            if kept_ffn.len() <= cfg.num_hidden_layers {
                continue;
            }
            kept_ffn.remove(&idx);
            ffn_pruned += 1;
        }
        removed += params as f64;
    }

    let (head_mask, ffn_mask) = masks_from_kept(cfg, &kept_heads, &kept_ffn);
    PruneResult {
        sparsity,
        head_mask,
        ffn_mask,
        kept_heads,
        kept_ffn,
        heads_pruned,
        ffn_pruned,
        param_reduction: (removed / total.max(1.0)) as f32,
    }
}
