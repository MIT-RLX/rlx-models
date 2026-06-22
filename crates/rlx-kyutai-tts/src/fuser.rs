//! Conditioner fuser — combines [`crate::conditioner`] outputs into the inputs
//! consumed by the backbone transformer.
//!
//! Driven by [`crate::config::FuserConfig`]:
//!
//! - **`sum`**: each named conditioner output is added to the per-step `[d_model]`
//!   hidden state before the temporal transformer step.
//! - **`prepend`**: each named conditioner output becomes a leading
//!   `[1, d_model]` row prepended to the temporal sequence (e.g. T5-style
//!   sentence-level conditioning). Empty in the published 1.6B-en/fr config.
//! - **`cross`**: each named conditioner output forms the K/V tensor consumed
//!   by [`crate::cross_attention::CrossAttention`] in every backbone layer.
//!
//! This module is plumbing only — the conditioner outputs are computed by
//! `LutConditioner` / `TensorConditioner`, and this module routes them.

use crate::config::FuserConfig;
use anyhow::Result;
use ndarray::{Array1, Array2, Axis};
use std::collections::HashMap;

/// Resolved per-conditioner output for one generation step.
#[derive(Debug, Clone)]
pub struct ConditionerOutputs {
    /// `name → [dim]` for scalar (sum / cross) signals.
    pub vectors: HashMap<String, Array1<f32>>,
    /// `name → [t, dim]` for sequence (prepend / cross) signals.
    /// Falls back to `vectors[name]` unsqueezed to `[1, dim]` when missing.
    pub sequences: HashMap<String, Array2<f32>>,
}

impl ConditionerOutputs {
    pub fn new() -> Self {
        Self {
            vectors: HashMap::new(),
            sequences: HashMap::new(),
        }
    }

    /// Register a vector output.
    pub fn insert_vector(&mut self, name: impl Into<String>, v: Array1<f32>) {
        self.vectors.insert(name.into(), v);
    }

    /// Register a sequence output.
    pub fn insert_sequence(&mut self, name: impl Into<String>, seq: Array2<f32>) {
        self.sequences.insert(name.into(), seq);
    }

    fn get_vector(&self, name: &str) -> Option<&Array1<f32>> {
        self.vectors.get(name)
    }

    fn get_sequence(&self, name: &str) -> Option<Array2<f32>> {
        if let Some(s) = self.sequences.get(name) {
            return Some(s.clone());
        }
        // Promote vector → 1-frame sequence for cross-attn / prepend use.
        self.vectors
            .get(name)
            .map(|v| v.clone().insert_axis(Axis(0)))
    }
}

impl Default for ConditionerOutputs {
    fn default() -> Self {
        Self::new()
    }
}

/// Sum-fused signal: pre-computed once per generation, added to every step's
/// `[d_model]` hidden state before the temporal transformer.
#[derive(Debug, Clone, Default)]
pub struct SumOffset {
    pub vector: Option<Array1<f32>>,
}

impl SumOffset {
    /// Apply: `hidden += vector` (in place).
    pub fn apply(&self, hidden: &mut Array1<f32>) {
        if let Some(v) = &self.vector {
            assert_eq!(
                v.len(),
                hidden.len(),
                "sum-fused conditioner dim {} != hidden dim {}",
                v.len(),
                hidden.len()
            );
            for (h, &x) in hidden.iter_mut().zip(v.iter()) {
                *h += x;
            }
        }
    }
}

/// Routed fuser outputs for one generation: sum offset, prepend rows, cross K/V.
#[derive(Debug, Clone, Default)]
pub struct FusedConditioning {
    /// Sum-fused vector (added to every step).
    pub sum: SumOffset,
    /// `[t_prepend, d_model]` rows prepended to the temporal sequence.
    pub prepend: Option<Array2<f32>>,
    /// `[t_cross, d_model]` K/V context for cross-attention.
    pub cross: Option<Array2<f32>>,
}

/// Apply `fuser` routing to per-conditioner outputs.
///
/// - `sum`: vectors are added together; dim must match `d_model`.
/// - `prepend`: sequences (or 1-row promoted vectors) are concatenated.
/// - `cross`: same as `prepend` but exposed under `cross` rather than `prepend`.
pub fn fuse(
    fuser: &FuserConfig,
    outputs: &ConditionerOutputs,
    d_model: usize,
) -> Result<FusedConditioning> {
    // Sum.
    let mut sum_acc: Option<Array1<f32>> = None;
    for name in &fuser.sum {
        let Some(v) = outputs.get_vector(name) else {
            continue;
        };
        if v.len() != d_model {
            // Skip silently — Kyutai's CFG / control LUTs use d=16 / d=2048 and
            // upstream projects them to d_model via per-conditioner output_proj.
            // The caller is responsible for that projection before calling fuse().
            anyhow::bail!(
                "sum-fused conditioner {name:?} has dim {} ≠ d_model {d_model}",
                v.len()
            );
        }
        sum_acc = Some(match sum_acc {
            None => v.clone(),
            Some(mut acc) => {
                for (a, b) in acc.iter_mut().zip(v.iter()) {
                    *a += *b;
                }
                acc
            }
        });
    }

    // Prepend / cross.
    let prepend = stack_sequences(&fuser.prepend, outputs)?;
    let cross = stack_sequences(&fuser.cross, outputs)?;

    Ok(FusedConditioning {
        sum: SumOffset { vector: sum_acc },
        prepend,
        cross,
    })
}

fn stack_sequences(names: &[String], outputs: &ConditionerOutputs) -> Result<Option<Array2<f32>>> {
    let mut frames: Vec<Array2<f32>> = Vec::new();
    let mut d: Option<usize> = None;
    for name in names {
        if let Some(seq) = outputs.get_sequence(name) {
            if let Some(d_set) = d {
                if seq.ncols() != d_set {
                    anyhow::bail!(
                        "fuser conditioner {name:?} has dim {} ≠ {d_set} (mixed-dim sequences not supported)",
                        seq.ncols()
                    );
                }
            } else {
                d = Some(seq.ncols());
            }
            frames.push(seq);
        }
    }
    if frames.is_empty() {
        return Ok(None);
    }
    let total_rows: usize = frames.iter().map(|f| f.nrows()).sum();
    let dim = d.unwrap();
    let mut out = Array2::<f32>::zeros((total_rows, dim));
    let mut row_off = 0;
    for f in frames {
        for r in 0..f.nrows() {
            for c in 0..dim {
                out[[row_off + r, c]] = f[[r, c]];
            }
        }
        row_off += f.nrows();
    }
    Ok(Some(out))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::FuserConfig;
    use ndarray::array;

    fn fuser_with(sum: &[&str], prepend: &[&str], cross: &[&str]) -> FuserConfig {
        FuserConfig {
            cross_attention_pos_emb: false,
            cross_attention_pos_emb_scale: 1.0,
            sum: sum.iter().map(|s| s.to_string()).collect(),
            prepend: prepend.iter().map(|s| s.to_string()).collect(),
            cross: cross.iter().map(|s| s.to_string()).collect(),
        }
    }

    #[test]
    fn sum_fuser_adds_vectors_of_matching_dim() {
        let mut outs = ConditionerOutputs::new();
        outs.insert_vector("a", array![1.0, 2.0, 3.0]);
        outs.insert_vector("b", array![10.0, 20.0, 30.0]);
        let cfg = fuser_with(&["a", "b"], &[], &[]);
        let fused = fuse(&cfg, &outs, 3).unwrap();
        assert_eq!(fused.sum.vector.unwrap(), array![11.0, 22.0, 33.0]);
    }

    #[test]
    fn sum_fuser_skips_missing_conditioners() {
        let mut outs = ConditionerOutputs::new();
        outs.insert_vector("a", array![1.0, 2.0]);
        let cfg = fuser_with(&["a", "missing"], &[], &[]);
        let fused = fuse(&cfg, &outs, 2).unwrap();
        assert_eq!(fused.sum.vector.unwrap(), array![1.0, 2.0]);
    }

    #[test]
    fn sum_offset_applies_in_place() {
        let s = SumOffset {
            vector: Some(array![0.1, 0.2, 0.3]),
        };
        let mut h = array![1.0, 1.0, 1.0];
        s.apply(&mut h);
        assert!((h[0] - 1.1).abs() < 1e-6);
        assert!((h[2] - 1.3).abs() < 1e-6);
    }

    #[test]
    fn cross_concatenates_promoted_vectors() {
        let mut outs = ConditionerOutputs::new();
        outs.insert_vector("a", array![1.0, 2.0, 3.0]);
        outs.insert_sequence("b", array![[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
        let cfg = fuser_with(&[], &[], &["a", "b"]);
        let fused = fuse(&cfg, &outs, 3).unwrap();
        let cross = fused.cross.unwrap();
        assert_eq!(cross.dim(), (3, 3));
        assert_eq!(cross.row(0), array![1.0, 2.0, 3.0]);
        assert_eq!(cross.row(2), array![7.0, 8.0, 9.0]);
    }

    #[test]
    fn fuse_rejects_mismatched_sum_dim() {
        let mut outs = ConditionerOutputs::new();
        outs.insert_vector("a", array![1.0, 2.0]);
        let cfg = fuser_with(&["a"], &[], &[]);
        // d_model = 3 vs vector dim 2 → error.
        assert!(fuse(&cfg, &outs, 3).is_err());
    }
}
