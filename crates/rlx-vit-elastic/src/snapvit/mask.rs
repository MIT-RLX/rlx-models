// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! Structure ↔ mask index maps.
//!
//! A **head structure** has index `l·num_heads + h` and occupies the per-channel
//! head-mask range `[l·H + h·hd, l·H + (h+1)·hd)`. An **FFN-channel structure**
//! has index `l·inner + c` and maps directly to `ffn_mask[l·inner + c]`.

use std::collections::HashSet;

use crate::vit::config::VitConfig;

/// All-ones head mask `[L·H]`.
pub fn ones_head_mask(cfg: &VitConfig) -> Vec<f32> {
    vec![1.0; cfg.num_hidden_layers * cfg.hidden_size]
}

/// All-ones FFN mask `[L·inner]`.
pub fn ones_ffn_mask(cfg: &VitConfig) -> Vec<f32> {
    vec![1.0; cfg.num_hidden_layers * cfg.ffn_inner()]
}

/// Total number of head structures.
pub fn num_head_structures(cfg: &VitConfig) -> usize {
    cfg.num_hidden_layers * cfg.num_attention_heads
}

/// Total number of FFN-channel structures.
pub fn num_ffn_structures(cfg: &VitConfig) -> usize {
    cfg.num_hidden_layers * cfg.ffn_inner()
}

/// Zero head structure `idx = l·num_heads + h` in a `[L·H]` head mask.
pub fn zero_head(cfg: &VitConfig, mask: &mut [f32], idx: usize) {
    let nh = cfg.num_attention_heads;
    let hd = cfg.head_dim();
    let h = cfg.hidden_size;
    let l = idx / nh;
    let head = idx % nh;
    let base = l * h + head * hd;
    for c in base..base + hd {
        mask[c] = 0.0;
    }
}

/// Zero FFN-channel structure `idx = l·inner + c` in a `[L·inner]` FFN mask.
pub fn zero_ffn(_cfg: &VitConfig, mask: &mut [f32], idx: usize) {
    mask[idx] = 0.0;
}

/// Build (head_mask, ffn_mask) that KEEP exactly the given structure indices.
pub fn masks_from_kept(
    cfg: &VitConfig,
    kept_heads: &HashSet<usize>,
    kept_ffn: &HashSet<usize>,
) -> (Vec<f32>, Vec<f32>) {
    let mut head_mask = ones_head_mask(cfg);
    let mut ffn_mask = ones_ffn_mask(cfg);
    for idx in 0..num_head_structures(cfg) {
        if !kept_heads.contains(&idx) {
            zero_head(cfg, &mut head_mask, idx);
        }
    }
    for idx in 0..num_ffn_structures(cfg) {
        if !kept_ffn.contains(&idx) {
            zero_ffn(cfg, &mut ffn_mask, idx);
        }
    }
    (head_mask, ffn_mask)
}
