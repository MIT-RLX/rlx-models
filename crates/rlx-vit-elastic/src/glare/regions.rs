// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! Region layout for GLARE's regional consistency.
//!
//! **Deviation (documented):** the paper uses *attention-aware* region sampling
//! (starting from the patch a last-block head attends to most) with per-step
//! random regions. To keep a single static graph, we use fixed contiguous
//! patch blocks and pool them with a constant averaging matrix. The
//! cross-attention module and the regional DINO objective are otherwise as in
//! the paper (Eqs. 5–7).

/// A fixed partition of the patch tokens into contiguous regions.
#[derive(Clone, Debug)]
pub struct RegionLayout {
    pub n_patch: usize,
    pub n_regions: usize,
}

impl RegionLayout {
    pub fn new(n_patch: usize, n_regions: usize) -> Self {
        Self {
            n_patch,
            n_regions: n_regions.max(1).min(n_patch.max(1)),
        }
    }

    /// The `[n_regions, n_patch]` block-average pooling matrix (row-major):
    /// region `r` averages its contiguous block of patches.
    pub fn pool_matrix(&self) -> Vec<f32> {
        let mut m = vec![0f32; self.n_regions * self.n_patch];
        let base = self.n_patch / self.n_regions;
        let rem = self.n_patch % self.n_regions;
        let mut start = 0usize;
        for r in 0..self.n_regions {
            let len = base + usize::from(r < rem);
            let w = 1.0 / len.max(1) as f32;
            for p in start..start + len {
                m[r * self.n_patch + p] = w;
            }
            start += len;
        }
        m
    }
}
