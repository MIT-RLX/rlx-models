// RLX models — fine-tuning.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! LoRA / DoRA adapter specs and host-side fuse.
//!
//! The forward pass uses rlx's first-class `LoraMatMul` op (so it lowers on
//! every backend); this module owns the configuration and the host-side
//! merge that bakes a trained adapter back into the base weights, producing
//! an adapter-free model with zero inference overhead.

use anyhow::{Result, bail};

/// How A is initialized. B is always zero so the initial delta is zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoraInit {
    /// A ~ N(0, std²); the standard LoRA init.
    Gaussian,
}

/// LoRA configuration. `scale = alpha / rank` is applied to the low-rank
/// delta, matching `LoraMatMul`'s `scale` field.
#[derive(Debug, Clone)]
pub struct LoraSpec {
    pub rank: usize,
    pub alpha: f32,
    pub dropout: f32,
    /// Substring patterns matched against weight (Param) names, e.g.
    /// `["q_proj", "v_proj"]`.
    pub target_modules: Vec<String>,
    pub init: LoraInit,
}

impl LoraSpec {
    pub fn new(rank: usize, alpha: f32, target_modules: Vec<String>) -> Self {
        Self {
            rank,
            alpha,
            dropout: 0.0,
            target_modules,
            init: LoraInit::Gaussian,
        }
    }
    /// The multiplicative scale applied to the low-rank delta.
    pub fn scale(&self) -> f32 {
        if self.rank == 0 {
            0.0
        } else {
            self.alpha / self.rank as f32
        }
    }
    /// Whether `weight_name` is targeted by this spec.
    pub fn targets(&self, weight_name: &str) -> bool {
        self.target_modules.iter().any(|m| weight_name.contains(m))
    }
}

/// DoRA = LoRA + a per-output-column magnitude vector. The forward composes
/// from first-class ops (no new op); this carries the extra config.
#[derive(Debug, Clone)]
pub struct DoraSpec {
    pub lora: LoraSpec,
}

/// Merge a LoRA adapter into a base weight: `W' = W + scale · (A · B)`.
///
/// `base` is `[k, n]` row-major; `a` is `[k, r]`; `b` is `[r, n]`. Returns the
/// fused `[k, n]` weight. This matches `LoraMatMul`'s semantics, so the fused
/// dense model is numerically equivalent to running the adapter.
pub fn fuse_lora(
    base: &[f32],
    a: &[f32],
    b: &[f32],
    k: usize,
    r: usize,
    n: usize,
    scale: f32,
) -> Result<Vec<f32>> {
    if base.len() != k * n {
        bail!("fuse_lora: base.len()={} != k*n={}", base.len(), k * n);
    }
    if a.len() != k * r {
        bail!("fuse_lora: a.len()={} != k*r={}", a.len(), k * r);
    }
    if b.len() != r * n {
        bail!("fuse_lora: b.len()={} != r*n={}", b.len(), r * n);
    }
    let mut out = base.to_vec();
    for i in 0..k {
        for j in 0..n {
            let mut acc = 0.0f32;
            for t in 0..r {
                acc += a[i * r + t] * b[t * n + j];
            }
            out[i * n + j] += scale * acc;
        }
    }
    Ok(out)
}

/// Column-wise L2 norms of a `[k, n]` row-major weight — one norm per output
/// column (over the `k` input rows). `out[j] = ‖W[:, j]‖₂`.
pub fn column_norms(w: &[f32], k: usize, n: usize) -> Vec<f32> {
    (0..n)
        .map(|j| {
            let s: f32 = (0..k).map(|i| w[i * n + j] * w[i * n + j]).sum();
            s.sqrt()
        })
        .collect()
}

/// Merge a DoRA adapter into a base weight (host-side):
/// `W' = m ⊙ (W + scale·A·B) / ‖W + scale·A·B‖_c`, where the magnitude `m [n]`
/// and column-norm broadcast over the `k` rows. Matches the DoRA forward, so
/// the fused dense model is adapter-free with zero inference overhead.
pub fn fuse_dora(
    base: &[f32],
    a: &[f32],
    b: &[f32],
    m: &[f32],
    scale: f32,
    k: usize,
    r: usize,
    n: usize,
) -> Result<Vec<f32>> {
    if base.len() != k * n || a.len() != k * r || b.len() != r * n || m.len() != n {
        bail!("fuse_dora: shape mismatch (k={k}, r={r}, n={n})");
    }
    // Wc = base + scale·(A·B)
    let mut wc = base.to_vec();
    for i in 0..k {
        for j in 0..n {
            let mut acc = 0.0;
            for t in 0..r {
                acc += a[i * r + t] * b[t * n + j];
            }
            wc[i * n + j] += scale * acc;
        }
    }
    let norms = column_norms(&wc, k, n);
    let mut out = vec![0.0; k * n];
    for i in 0..k {
        for j in 0..n {
            let nrm = if norms[j] > 0.0 { norms[j] } else { 1.0 };
            out[i * n + j] = m[j] * wc[i * n + j] / nrm;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn column_norms_are_per_output() {
        // W = [[3,0],[4,0]] (k=2,n=2): col 0 = ‖[3,4]‖ = 5, col 1 = 0.
        let w = vec![3.0, 0.0, 4.0, 0.0];
        assert_eq!(column_norms(&w, 2, 2), vec![5.0, 0.0]);
    }

    #[test]
    fn fuse_dora_with_b_zero_and_unit_magnitude_is_base() {
        // B=0, m=‖W‖_c → W' = ‖W‖_c · W/‖W‖_c = W.
        let base = vec![3.0, 1.0, 4.0, 2.0]; // [2,2]
        let m = column_norms(&base, 2, 2);
        let a = vec![0.1, 0.2]; // [2,1]
        let b = vec![0.0, 0.0]; // [1,2] = 0
        let fused = fuse_dora(&base, &a, &b, &m, 1.0, 2, 1, 2).unwrap();
        for (x, y) in fused.iter().zip(&base) {
            assert!((x - y).abs() < 1e-5, "{x} vs {y}");
        }
    }

    #[test]
    fn scale_is_alpha_over_rank() {
        let s = LoraSpec::new(8, 16.0, vec!["q_proj".into()]);
        assert_eq!(s.scale(), 2.0);
        assert!(s.targets("model.layers.0.self_attn.q_proj.weight"));
        assert!(!s.targets("model.layers.0.mlp.gate_proj.weight"));
    }

    #[test]
    fn fuse_adds_scaled_low_rank_delta() {
        // k=2, r=1, n=2. base = I. A=[1;2], B=[3,4]. delta = A@B = [[3,4],[6,8]].
        let base = vec![1.0, 0.0, 0.0, 1.0];
        let a = vec![1.0, 2.0]; // [2,1]
        let b = vec![3.0, 4.0]; // [1,2]
        let fused = fuse_lora(&base, &a, &b, 2, 1, 2, 0.5).unwrap();
        // W + 0.5 * [[3,4],[6,8]] = [[1+1.5, 2.0],[3.0, 1+4.0]]
        assert_eq!(fused, vec![2.5, 2.0, 3.0, 5.0]);
    }

    #[test]
    fn fuse_rejects_shape_mismatch() {
        assert!(fuse_lora(&[1.0], &[1.0], &[1.0], 2, 1, 2, 1.0).is_err());
    }
}
