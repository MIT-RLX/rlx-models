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

//! Safetensors weight loading — standalone, no framework dependency.

use anyhow::{Context, Result, bail, ensure};
use std::collections::{HashMap, HashSet};
use std::path::Path;

use crate::gguf_support::{
    gguf_architecture_from_path, gguf_safetensors_only_hint, resolve_weights_file,
};
use crate::weight_loader::WeightLoader;
use crate::weight_registry::{LoadWeightsOptions, load_weight_map_resolved};
use rlx_ir::quant::QuantScheme;

/// Packed GGUF weight bytes + scheme + logical shape.
pub type PackedWeightTensor = (Vec<u8>, QuantScheme, Vec<usize>);
/// Named packed tensor (sidecar list from [`WeightMap::drain_loader`]).
pub type NamedPackedWeight = (String, Vec<u8>, QuantScheme, Vec<usize>);
/// F32 tensor snapshot (`name → (data, shape)`).
pub type F32WeightSnapshot = HashMap<String, (Vec<f32>, Vec<usize>)>;

/// How [`WeightMap::drain_loader`] / [`WeightMap::from_weight_loader`] handle leftovers.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum WeightDrainPolicy {
    #[default]
    AllF32,
    /// Log a warning when tensors remain after drain.
    AllF32WarnUnused,
    /// Fail if any tensor was not taken.
    AllF32StrictUnused,
}

/// Map of tensor name → (f32 data, shape).
pub struct WeightMap {
    tensors: HashMap<String, (Vec<f32>, Vec<usize>)>,
}

impl WeightMap {
    /// Drain every tensor from any [`WeightLoader`] (safetensors or GGUF).
    pub fn from_weight_loader(loader: &mut dyn WeightLoader) -> Result<Self> {
        Self::drain_loader(loader, WeightDrainPolicy::AllF32).map(|(m, _)| m)
    }

    /// Force-dequantize every tensor (including K-quants) into F32 and
    /// drop it in the map. Use when a family runner doesn't have a
    /// packed-matmul lowering yet but still wants to load GGUFs whose
    /// trunk weights are K-quant. Trades memory (4× larger than the
    /// packed bytes) for correctness — every tensor goes through
    /// `WeightLoader::take(...)` which dequantizes on the fly.
    pub fn from_weight_loader_dequant_all(loader: &mut dyn WeightLoader) -> Result<Self> {
        let keys = loader.remaining_keys();
        let mut tensors = HashMap::with_capacity(keys.len());
        for key in keys {
            let (data, shape) = loader.take(&key)?;
            tensors.insert(key, (data, shape));
        }
        Ok(Self { tensors })
    }

    /// Drain with policy; returns packed K-quants separately when the loader supports `take_packed`.
    pub fn drain_loader(
        loader: &mut dyn WeightLoader,
        policy: WeightDrainPolicy,
    ) -> Result<(Self, Vec<NamedPackedWeight>)> {
        let keys = loader.remaining_keys();
        let mut tensors = HashMap::with_capacity(keys.len());
        let mut packed = Vec::new();
        for key in keys {
            if let Some((bytes, scheme, shape)) = loader.take_packed(&key)? {
                packed.push((key, bytes, scheme, shape));
                continue;
            }
            let (data, shape) = loader.take(&key)?;
            tensors.insert(key, (data, shape));
        }
        let left = loader.remaining_keys();
        match policy {
            WeightDrainPolicy::AllF32 => {}
            WeightDrainPolicy::AllF32WarnUnused if !left.is_empty() => {
                eprintln!(
                    "[rlx-core] weight drain: {} unused tensors (format={})",
                    left.len(),
                    loader.format_id()
                );
                for k in left.iter().take(8) {
                    eprintln!("  unused: {k}");
                }
                if left.len() > 8 {
                    eprintln!("  … and {} more", left.len() - 8);
                }
            }
            WeightDrainPolicy::AllF32StrictUnused if !left.is_empty() => {
                bail!(
                    "weight drain left {} unused tensors (format={}): {:?}",
                    left.len(),
                    loader.format_id(),
                    &left[..left.len().min(5)]
                );
            }
            _ => {}
        }
        Ok((Self { tensors }, packed))
    }

    /// Resolve a file or weights directory, then load (safetensors or GGUF).
    pub fn from_resolved_path(path: &Path) -> Result<Self> {
        let file = resolve_weights_file(path)?;
        Self::from_resolved_file(&file)
    }

    /// Resolve path; reject `.gguf` with a hint naming the right runner.
    pub fn from_resolved_safetensors_only(path: &Path, runner: &str) -> Result<Self> {
        let file = resolve_weights_file(path)?;
        if file.extension().and_then(|s| s.to_str()) == Some("gguf") {
            let arch = gguf_architecture_from_path(&file)?;
            bail!(gguf_safetensors_only_hint(runner, &file, &arch));
        }
        Self::from_resolved_file(&file)
    }

    fn from_resolved_file(file: &Path) -> Result<Self> {
        load_weight_map_resolved(file, LoadWeightsOptions::map()).map(|(_, m)| m)
    }

    /// Load weights from a safetensors file. Auto-converts bf16/f16 to f32.
    pub fn from_file(path: &str) -> Result<Self> {
        Self::from_file_excluding(path, &HashSet::new())
    }

    /// Load weights, skipping tensor names present in `exclude` (saves RAM when
    /// bf16/NVFP4 linears are loaded separately for GPU upload).
    pub fn from_file_excluding(path: &str, exclude: &HashSet<String>) -> Result<Self> {
        let data = std::fs::read(path).with_context(|| format!("reading {path}"))?;
        let st =
            safetensors::SafeTensors::deserialize(&data).with_context(|| "parsing safetensors")?;

        let mut tensors = HashMap::new();
        for (name, view) in st.tensors() {
            if exclude.contains(name.as_str()) {
                continue;
            }
            let shape: Vec<usize> = view.shape().to_vec();
            let bytes = view.data();
            let f32_data = match view.dtype() {
                safetensors::Dtype::F32 => bytes_to_f32_vec(bytes),
                safetensors::Dtype::F16 => bytes
                    .chunks_exact(2)
                    .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                    .collect(),
                safetensors::Dtype::BF16 => bytes
                    .chunks_exact(2)
                    .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
                    .collect(),
                safetensors::Dtype::I64 => bytes
                    .chunks_exact(8)
                    .map(|c| i64::from_le_bytes(c.try_into().unwrap()) as f32)
                    .collect(),
                safetensors::Dtype::I32 => bytes
                    .chunks_exact(4)
                    .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32)
                    .collect(),
                safetensors::Dtype::C64 => {
                    // Some checkpoints (SAM3) include complex RoPE caches
                    // such as `freqs_cis`. Native code regenerates/handles
                    // those separately; keep loading usable for the real
                    // float weights instead of rejecting the entire file.
                    continue;
                }
                other => anyhow::bail!("unsupported dtype: {other:?}"),
            };
            tensors.insert(name.to_string(), (f32_data, shape));
        }

        Ok(Self { tensors })
    }

    /// Take a tensor by name (removes from map). Returns (data, shape).
    pub fn take(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        self.tensors
            .remove(key)
            .ok_or_else(|| anyhow::anyhow!("weight not found: {key}"))
    }

    /// Take and transpose a 2D weight: [out, in] → [in, out] for row-major matmul.
    pub fn take_transposed(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        let (data, shape) = self.take(key)?;
        if shape.len() != 2 {
            anyhow::bail!("transpose requires 2D, got {shape:?}");
        }
        let (rows, cols) = (shape[0], shape[1]);
        let mut transposed = vec![0f32; data.len()];
        for i in 0..rows {
            for j in 0..cols {
                transposed[j * rows + i] = data[i * cols + j];
            }
        }
        Ok((transposed, vec![cols, rows]))
    }

    /// Check if a key exists.
    pub fn has(&self, key: &str) -> bool {
        self.tensors.contains_key(key)
    }

    /// List all keys.
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(|s| s.as_str())
    }

    /// Number of tensors remaining.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Create from pre-built HashMap (for testing without safetensors files).
    pub fn from_tensors(tensors: HashMap<String, (Vec<f32>, Vec<usize>)>) -> Self {
        Self { tensors }
    }

    /// Drain all tensors into a snapshot map (for runners that rebuild graphs per shape).
    pub fn snapshot_from_path(path: &str) -> Result<F32WeightSnapshot> {
        let mut wm = Self::from_file(path)?;
        let keys: Vec<String> = wm.keys().map(|s| s.to_string()).collect();
        let mut out = HashMap::with_capacity(keys.len());
        for k in keys {
            out.insert(k.clone(), wm.take(&k)?);
        }
        Ok(out)
    }

    /// Load only tensors whose names appear in `want` (HF sharded checkpoints).
    pub fn from_safetensors_dir_selected(dir: &Path, want: &HashSet<String>) -> Result<Self> {
        crate::safetensors_checkpoint::SafetensorsCheckpoint::open(dir)?.load_selected(want)
    }

    /// Load and merge every `*.safetensors` file in `dir` (e.g. HF `text_encoder/`).
    pub fn from_safetensors_dir(dir: &Path) -> Result<Self> {
        let mut merged = HashMap::new();
        let mut any = false;
        for entry in std::fs::read_dir(dir).with_context(|| format!("read_dir {dir:?}"))? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) != Some("safetensors") {
                continue;
            }
            let part = Self::from_file(
                path.to_str()
                    .ok_or_else(|| anyhow::anyhow!("non-utf8 path {:?}", path))?,
            )?;
            for (k, v) in part.tensors {
                merged.insert(k, v);
            }
            any = true;
        }
        if !any {
            anyhow::bail!("no .safetensors files in {dir:?}");
        }
        Ok(Self { tensors: merged })
    }

    /// Rename keys in-place (e.g. strip `model.` HuggingFace prefix).
    pub fn remap_keys<F>(&mut self, mut f: F)
    where
        F: FnMut(String) -> String,
    {
        let keys: Vec<String> = self.tensors.keys().cloned().collect();
        for old in keys {
            if let Some(v) = self.tensors.remove(&old) {
                let new = f(old);
                self.tensors.insert(new, v);
            }
        }
    }

    /// Borrow tensor data + shape without removing from the map.
    pub fn get(&self, key: &str) -> Option<(&[f32], &[usize])> {
        self.tensors
            .get(key)
            .map(|(d, s)| (d.as_slice(), s.as_slice()))
    }

    /// Element-wise add `delta` into an existing rank-2 weight (PyTorch `[out, in]` layout).
    pub fn merge_add_weight(&mut self, key: &str, delta: &[f32]) -> Result<()> {
        let entry = self
            .tensors
            .get_mut(key)
            .with_context(|| format!("merge_add_weight: missing {key}"))?;
        let (data, shape) = entry;
        ensure!(
            shape.len() == 2,
            "merge_add_weight {key}: expected rank-2, got {shape:?}"
        );
        ensure!(
            data.len() == delta.len(),
            "merge_add_weight {key}: len {} != delta {}",
            data.len(),
            delta.len()
        );
        for (d, s) in data.iter_mut().zip(delta.iter()) {
            *d += s;
        }
        Ok(())
    }
}

/// Convert a raw byte slice to a `Vec<f32>`. Safetensors stores tensor
/// data at arbitrary byte offsets — when an f32 tensor doesn't land on
/// a 4-byte boundary, `bytemuck::cast_slice` panics with
/// `TargetAlignmentGreaterAndInputNotAligned`. SAM ViT-B is one such
/// file. Fall back to a manual little-endian decode in that case.
pub(crate) fn bytes_to_f32_vec(bytes: &[u8]) -> Vec<f32> {
    debug_assert!(
        bytes.len().is_multiple_of(4),
        "f32 byte slice length must be multiple of 4 (got {})",
        bytes.len()
    );
    if (bytes.as_ptr() as usize).is_multiple_of(std::mem::align_of::<f32>()) {
        let f32s: &[f32] = bytemuck::cast_slice(bytes);
        f32s.to_vec()
    } else {
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transpose_2d() {
        let mut wm = WeightMap {
            tensors: HashMap::from([(
                "w".to_string(),
                (vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]),
            )]),
        };
        let (data, shape) = wm.take_transposed("w").unwrap();
        assert_eq!(shape, vec![3, 2]);
        // Original: [[1,2,3],[4,5,6]] → Transposed: [[1,4],[2,5],[3,6]]
        assert_eq!(data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }
}
