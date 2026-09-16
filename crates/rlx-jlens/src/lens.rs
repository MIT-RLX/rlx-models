//! The fitted lens: per-layer `J_l`, on disk and back.
//!
//! Fitting is the expensive part — the reference uses 1000 sequences of 128
//! tokens — so the result is an artifact you save once and apply many times.
//! [`JacobianLens::merge`] exists for the same reason: fit disjoint slices of a
//! corpus on separate machines and combine them, which is the only practical way
//! to scale past one box.
//!
//! The file is **safetensors**, one tensor per layer keyed `J.{layer}`, so the
//! Python reference implementation can load a lens fitted here and vice versa.
//! Entries are normally stored as f16: for a language model they are O(1), so
//! the range is not a constraint and f16's extra mantissa beats bf16 at that
//! scale.
//!
//! That assumption is checked rather than trusted. A DINOv3 trunk's residual
//! stream grows ~72x through the stack, its `J` reaches `1e10`, and writing that
//! as f16 silently saturated every large entry to `inf` — the artifact looked
//! fine and every statistic computed from it came back `NaN`. Any layer with an
//! entry past f16's range is written as f32 instead; the loader already accepts
//! both, and safetensors records the dtype per tensor, so a mixed file is
//! self-describing.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use half::f16;
use safetensors::tensor::{Dtype, TensorView};

use crate::fit::Jacobian;

/// A fitted Jacobian lens.
#[derive(Debug, Clone)]
pub struct JacobianLens {
    jacobians: BTreeMap<usize, Jacobian>,
    /// Prompts averaged over. Carried so [`Self::merge`] can weight correctly.
    pub n_prompts: usize,
    pub d_model: usize,
    /// The layer whose residual `J_l` transports *into*.
    pub target_layer: usize,
}

impl JacobianLens {
    pub fn new(
        jacobians: BTreeMap<usize, Jacobian>,
        n_prompts: usize,
        target_layer: usize,
    ) -> Result<Self> {
        let d_model = jacobians
            .values()
            .next()
            .context("a lens needs at least one layer")?
            .d_model;
        for (layer, j) in &jacobians {
            ensure!(
                j.d_model == d_model,
                "layer {layer} is {}-wide, but the lens is {d_model}-wide",
                j.d_model
            );
        }
        Ok(Self {
            jacobians,
            n_prompts,
            d_model,
            target_layer,
        })
    }

    pub fn layers(&self) -> Vec<usize> {
        self.jacobians.keys().copied().collect()
    }

    pub fn get(&self, layer: usize) -> Option<&Jacobian> {
        self.jacobians.get(&layer)
    }

    pub fn iter(&self) -> impl Iterator<Item = (usize, &Jacobian)> {
        self.jacobians.iter().map(|(l, j)| (*l, j))
    }

    /// Combine lenses fitted on **disjoint** prompt slices.
    ///
    /// A prompt-count-weighted mean, so sharding a corpus and merging gives the
    /// same answer as one long run. Inputs must agree on layers, width and
    /// target; overlapping slices would double-count and are the caller's
    /// problem.
    pub fn merge(lenses: &[JacobianLens]) -> Result<JacobianLens> {
        let first = lenses.first().context("merge needs at least one lens")?;
        for other in &lenses[1..] {
            ensure!(
                other.layers() == first.layers(),
                "lenses disagree on layers: {:?} vs {:?}",
                first.layers(),
                other.layers()
            );
            ensure!(
                other.d_model == first.d_model && other.target_layer == first.target_layer,
                "lenses disagree on d_model or target layer"
            );
        }
        let total: usize = lenses.iter().map(|l| l.n_prompts).sum();
        ensure!(total > 0, "every lens has n_prompts = 0");

        let mut merged = BTreeMap::new();
        for layer in first.layers() {
            let mut acc = Jacobian::zeros(first.d_model);
            for lens in lenses {
                let j = lens.get(layer).expect("layers checked above");
                let w = lens.n_prompts as f32;
                for (a, b) in acc.values.iter_mut().zip(&j.values) {
                    *a += w * b;
                }
            }
            acc.scale(total as f32);
            merged.insert(layer, acc);
        }
        JacobianLens::new(merged, total, first.target_layer)
    }

    /// Write to `path` as safetensors, f16.
    ///
    /// Written to a temp file and renamed, so an interrupted save never leaves a
    /// half-written lens behind a valid-looking name.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        let d = self.d_model;
        // Keep the halved buffers alive until serialization has read them.
        // f16's largest finite value. An entry at or past it would round to
        // `inf`, so that layer goes out as f32.
        const F16_MAX: f32 = 65504.0;
        let encoded: Vec<(String, Dtype, Vec<u8>)> = self
            .jacobians
            .iter()
            .map(|(layer, j)| {
                let overflows = j.values.iter().any(|v| !v.is_finite() || v.abs() > F16_MAX);
                let (dtype, bytes): (Dtype, Vec<u8>) = if overflows {
                    (
                        Dtype::F32,
                        j.values.iter().flat_map(|v| v.to_le_bytes()).collect(),
                    )
                } else {
                    (
                        Dtype::F16,
                        j.values
                            .iter()
                            .flat_map(|v| f16::from_f32(*v).to_le_bytes())
                            .collect(),
                    )
                };
                (format!("J.{layer}"), dtype, bytes)
            })
            .collect();
        let views: Vec<(String, TensorView<'_>)> = encoded
            .iter()
            .map(|(name, dtype, bytes)| {
                TensorView::new(*dtype, vec![d, d], bytes)
                    .map(|v| (name.clone(), v))
                    .context("building tensor view")
            })
            .collect::<Result<_>>()?;

        let metadata = std::collections::HashMap::from([
            ("format".to_string(), "rlx-jlens.v1".to_string()),
            ("n_prompts".to_string(), self.n_prompts.to_string()),
            ("d_model".to_string(), d.to_string()),
            ("target_layer".to_string(), self.target_layer.to_string()),
        ]);
        let bytes =
            safetensors::tensor::serialize(views, Some(metadata)).context("serializing lens")?;

        let tmp = path.with_extension(format!("tmp{}", std::process::id()));
        std::fs::write(&tmp, &bytes).with_context(|| format!("writing {}", tmp.display()))?;
        std::fs::rename(&tmp, path).with_context(|| format!("renaming into {}", path.display()))?;
        Ok(())
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        let st = safetensors::SafeTensors::deserialize(&bytes)
            .with_context(|| format!("{} is not a safetensors file", path.display()))?;
        let meta = safetensors::SafeTensors::read_metadata(&bytes)
            .ok()
            .and_then(|(_, m)| m.metadata().clone());
        let get = |k: &str| -> Option<usize> {
            meta.as_ref()
                .and_then(|m| m.get(k))
                .and_then(|v| v.parse().ok())
        };
        let n_prompts = get("n_prompts").unwrap_or(0);
        let target_layer = get("target_layer").unwrap_or(0);

        let mut jacobians = BTreeMap::new();
        for (name, view) in st.tensors() {
            let Some(layer) = name
                .strip_prefix("J.")
                .and_then(|s| s.parse::<usize>().ok())
            else {
                continue;
            };
            let shape = view.shape();
            ensure!(
                shape.len() == 2 && shape[0] == shape[1],
                "{name} has shape {shape:?}, expected a square [d, d]"
            );
            let d = shape[0];
            let raw = view.data();
            let values: Vec<f32> = match view.dtype() {
                Dtype::F16 => raw
                    .chunks_exact(2)
                    .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32())
                    .collect(),
                Dtype::F32 => raw
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
                other => bail!("{name} has dtype {other:?}; expected F16 or F32"),
            };
            ensure!(
                values.len() == d * d,
                "{name} holds {} values, expected {}",
                values.len(),
                d * d
            );
            jacobians.insert(layer, Jacobian { values, d_model: d });
        }
        ensure!(
            !jacobians.is_empty(),
            "{} contains no `J.<layer>` tensors",
            path.display()
        );
        JacobianLens::new(jacobians, n_prompts, target_layer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lens(n_prompts: usize, fill: f32) -> JacobianLens {
        let mut m = BTreeMap::new();
        for layer in [0usize, 4] {
            let mut j = Jacobian::zeros(3);
            j.values.iter_mut().for_each(|v| *v = fill + layer as f32);
            m.insert(layer, j);
        }
        JacobianLens::new(m, n_prompts, 7).unwrap()
    }

    #[test]
    fn round_trips_through_a_file() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("rlx_jlens_test_{}.safetensors", std::process::id()));
        let original = lens(11, 0.25);
        original.save(&path).unwrap();
        let loaded = JacobianLens::load(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(loaded.layers(), original.layers());
        assert_eq!(loaded.n_prompts, 11);
        assert_eq!(loaded.d_model, 3);
        assert_eq!(loaded.target_layer, 7);
        for (layer, j) in original.iter() {
            let got = loaded.get(layer).unwrap();
            for (a, b) in j.values.iter().zip(&got.values) {
                // f16 storage: exact for these values, but allow its precision.
                assert!((a - b).abs() < 1e-3, "layer {layer}: {a} vs {b}");
            }
        }
    }

    /// Merging is prompt-weighted, so sharding a corpus must give the same
    /// answer as one run over the whole thing.
    #[test]
    fn merge_is_prompt_count_weighted() {
        let a = lens(3, 1.0); // layer 0 filled with 1.0
        let b = lens(1, 5.0); // layer 0 filled with 5.0
        let merged = JacobianLens::merge(&[a, b]).unwrap();
        assert_eq!(merged.n_prompts, 4);
        // (3·1 + 1·5) / 4 = 2
        for v in &merged.get(0).unwrap().values {
            assert!((v - 2.0).abs() < 1e-6, "got {v}");
        }
        for v in &merged.get(4).unwrap().values {
            assert!((v - 6.0).abs() < 1e-6, "got {v}");
        }
    }

    #[test]
    fn merging_mismatched_lenses_is_an_error() {
        let a = lens(1, 1.0);
        let mut m = BTreeMap::new();
        m.insert(0usize, Jacobian::zeros(3));
        let b = JacobianLens::new(m, 1, 7).unwrap();
        assert!(JacobianLens::merge(&[a, b]).is_err());
    }

    /// A ViT's `J` reaches 1e10; f16 would saturate it to `inf` and every
    /// statistic computed from the artifact would come back `NaN`.
    #[test]
    fn out_of_f16_range_survives_a_round_trip() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("rlx_jlens_big_{}.safetensors", std::process::id()));
        let mut m = BTreeMap::new();
        let mut j = Jacobian::zeros(3);
        j.values[0] = 1.5e10;
        j.values[4] = -3.2e7;
        j.values[8] = 0.25;
        m.insert(0usize, j);
        // A second, small layer must still take the compact f16 path.
        let mut small = Jacobian::zeros(3);
        small.values.iter_mut().for_each(|v| *v = 0.5);
        m.insert(1usize, small);

        JacobianLens::new(m, 1, 2).unwrap().save(&path).unwrap();
        let back = JacobianLens::load(&path).unwrap();
        std::fs::remove_file(&path).ok();

        let big = &back.get(0).unwrap().values;
        assert!(
            big[0].is_finite() && (big[0] - 1.5e10).abs() / 1.5e10 < 1e-6,
            "got {}",
            big[0]
        );
        assert!((big[4] + 3.2e7).abs() / 3.2e7 < 1e-6, "got {}", big[4]);
        assert!((big[8] - 0.25).abs() < 1e-6, "got {}", big[8]);
        for v in &back.get(1).unwrap().values {
            assert!((v - 0.5).abs() < 1e-3, "got {v}");
        }
    }

    #[test]
    fn merging_nothing_is_an_error() {
        assert!(JacobianLens::merge(&[]).is_err());
    }
}
