// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! **Validation only** — cross-checks the native graph against the community
//! ONNX export from AppleNeuralHash2ONNX.
//!
//! Gated behind the `onnx-parity` feature and never reachable from
//! [`crate::NeuralHasher`]. The default build has no ONNX dependency at all;
//! this module exists so the Espresso-derived architecture in [`crate::spec`]
//! can be shown to compute the same descriptor as the independently-produced
//! `model.onnx`, which is the strongest check available without an iOS device.
//!
//! Both sides run on rlx — the ONNX file is lowered to rlx-ir too — so a
//! disagreement isolates the *architecture derivation*, not backend numerics.
//!
//! ```text
//!   espresso.net ─▶ NeuralHashSpec ─▶ rlx-ir ─┐
//!                                             ├─▶ compare descriptors + hash bits
//!   model.onnx ────────────────────▶ rlx-ir ─┘
//! ```

use anyhow::{Context, Result, anyhow, ensure};
use rlx_onnx_import::{ImportOptions, build_hir_from_onnx_file};
use rlx_runtime::{CompileOptions, Device, Session};
use std::collections::HashMap;
use std::path::Path;

use crate::hash::NeuralHash;
use crate::seed::{EMBED_DIM, SeedMatrix};

/// How closely two descriptors agree.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParityReport {
    /// Largest absolute elementwise difference.
    pub max_abs_diff: f32,
    /// Mean absolute elementwise difference.
    pub mean_abs_diff: f32,
    /// Cosine similarity (1.0 = identical direction).
    pub cosine: f32,
    /// Differing hash bits after the seed projection, when a seed was supplied.
    pub hash_bits_differing: Option<u32>,
    /// Smallest `|score|` across both sides — how close the nearest bit was to
    /// flipping. A tiny value explains a 1–2 bit disagreement without implying
    /// the architecture is wrong.
    pub closest_score_to_zero: Option<f32>,
}

impl ParityReport {
    /// Whether the descriptors match to `tol` and the hashes (if computed) agree.
    pub fn passes(&self, tol: f32) -> bool {
        self.max_abs_diff <= tol && self.hash_bits_differing.is_none_or(|n| n == 0)
    }
}

/// Compare two 128-float descriptors, optionally through the seed projection.
pub fn compare(
    native: &[f32],
    reference: &[f32],
    seed: Option<&SeedMatrix>,
) -> Result<ParityReport> {
    ensure!(
        native.len() == reference.len(),
        "parity: descriptor lengths differ ({} vs {})",
        native.len(),
        reference.len()
    );
    ensure!(!native.is_empty(), "parity: empty descriptors");

    let mut max_abs = 0f32;
    let mut sum_abs = 0f64;
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (a, b) in native.iter().zip(reference.iter()) {
        let d = (a - b).abs();
        max_abs = max_abs.max(d);
        sum_abs += d as f64;
        dot += *a as f64 * *b as f64;
        na += (*a as f64).powi(2);
        nb += (*b as f64).powi(2);
    }
    let cosine = if na > 0.0 && nb > 0.0 {
        (dot / (na.sqrt() * nb.sqrt())) as f32
    } else {
        f32::NAN
    };

    let (bits, closest) = match seed {
        Some(s) => {
            let sa = s.project(native)?;
            let sb = s.project(reference)?;
            let ha = NeuralHash::from_scores(&sa)?;
            let hb = NeuralHash::from_scores(&sb)?;
            let closest = sa
                .iter()
                .chain(sb.iter())
                .fold(f32::INFINITY, |m, v| m.min(v.abs()));
            (Some(ha.hamming(&hb)), Some(closest))
        }
        None => (None, None),
    };

    Ok(ParityReport {
        max_abs_diff: max_abs,
        mean_abs_diff: (sum_abs / native.len() as f64) as f32,
        cosine,
        hash_bits_differing: bits,
        closest_score_to_zero: closest,
    })
}

/// Run the reference `model.onnx` (lowered to rlx-ir) on a preprocessed input.
///
/// Import is strict: a stubbed node would make the "reference" meaningless.
pub fn onnx_descriptor(onnx: &Path, input: &[f32], device: Device) -> Result<Vec<f32>> {
    ensure!(
        input.len() == crate::INPUT_ELEMS,
        "parity: expected a [3, {}, {}] tensor, got {} floats",
        crate::INPUT_SIZE,
        crate::INPUT_SIZE,
        input.len()
    );
    let named_lengths: HashMap<String, usize> = ["batch_size", "batch", "N", "n"]
        .iter()
        .map(|k| ((*k).to_string(), 1))
        .collect();
    let opts = ImportOptions {
        sequence_length: 1,
        named_lengths,
        strict: true,
        ..Default::default()
    };
    let (hir, mut params, report, manifest) = build_hir_from_onnx_file(onnx, opts)
        .with_context(|| format!("importing {} to rlx-ir", onnx.display()))?;
    ensure!(
        report.stubbed == 0 && report.unsupported.is_empty(),
        "parity: {} imported with {} stubbed node(s) and unsupported ops {:?}",
        onnx.display(),
        report.stubbed,
        report.unsupported
    );
    let input_name = manifest
        .inputs
        .first()
        .map(|i| i.name.clone())
        .ok_or_else(|| anyhow!("parity: {} declares no graph input", onnx.display()))?;

    let mut graph = Session::new(device)
        .compile_hir_with(hir, &CompileOptions::default())
        .map_err(|e| anyhow!("parity: compiling the reference graph for {device:?}: {e:?}"))?;
    for (name, data) in params.drain() {
        graph.set_param(&name, &data);
    }
    graph.finalize_params();

    let out = graph
        .run(&[(input_name.as_str(), input)])
        .into_iter()
        .next()
        .ok_or_else(|| anyhow!("parity: the ONNX graph produced no output"))?;
    ensure!(
        out.len() == EMBED_DIM,
        "parity: reference produced {} floats, expected {EMBED_DIM}",
        out.len()
    );
    Ok(out)
}

/// Number of differing bits between two hashes — a thin re-export so parity
/// harnesses do not need to reach into [`crate::hash`].
pub fn hash_bit_diff(a: &NeuralHash, b: &NeuralHash) -> u32 {
    a.hamming(b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hash::HASH_BITS;

    fn seed_identity() -> SeedMatrix {
        let mut rows = vec![0f32; HASH_BITS * EMBED_DIM];
        for r in 0..HASH_BITS {
            rows[r * EMBED_DIM + r] = 1.0;
        }
        SeedMatrix::from_rows(rows).unwrap()
    }

    #[test]
    fn identical_descriptors_pass() {
        let v: Vec<f32> = (0..EMBED_DIM).map(|i| i as f32 - 64.0).collect();
        let r = compare(&v, &v, Some(&seed_identity())).unwrap();
        assert_eq!(r.max_abs_diff, 0.0);
        assert_eq!(r.hash_bits_differing, Some(0));
        assert!((r.cosine - 1.0).abs() < 1e-6);
        assert!(r.passes(1e-5));
    }

    #[test]
    fn a_sign_flip_near_zero_shows_up_as_one_bit() {
        let mut a: Vec<f32> = (0..EMBED_DIM).map(|i| i as f32 - 64.0).collect();
        a[10] = 1e-7; // sits on the decision boundary
        let mut b = a.clone();
        b[10] = -1e-7;
        let r = compare(&a, &b, Some(&seed_identity())).unwrap();
        assert_eq!(r.hash_bits_differing, Some(1));
        assert!(r.closest_score_to_zero.unwrap() <= 1e-6);
        // Numerically near-identical, but not bit-identical — exactly the
        // "a few bits off" regime the upstream README warns about.
        assert!(r.max_abs_diff < 1e-6);
        assert!(!r.passes(1e-5), "a differing bit must fail the gate");
    }

    #[test]
    fn length_mismatch_is_an_error() {
        assert!(compare(&[1.0, 2.0], &[1.0], None).is_err());
        assert!(compare(&[], &[], None).is_err());
    }

    #[test]
    fn cosine_detects_a_scaled_descriptor() {
        let a: Vec<f32> = (1..=EMBED_DIM).map(|i| i as f32).collect();
        let b: Vec<f32> = a.iter().map(|v| v * 2.0).collect();
        let r = compare(&a, &b, None).unwrap();
        assert!((r.cosine - 1.0).abs() < 1e-6, "direction is unchanged");
        assert!(r.max_abs_diff > 1.0, "magnitudes differ");
    }
}
