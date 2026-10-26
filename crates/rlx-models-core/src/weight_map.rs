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
use rayon::prelude::*;
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

/// Cache-blocked, parallel `[rows, cols]` → `[cols, rows]` transpose.
///
/// The obvious nested loop writes down a column of the destination while
/// reading along a row of the source, so every store lands in a different cache
/// line. On a 2048×8192 Llama MLP weight that measured ~1 GB/s, and a 1 B-param
/// checkpoint spends about four seconds there — more than compiling and running
/// the graph combined. Tiling makes both sides cache-resident and rayon spreads
/// the tiles; the output is identical.
pub fn transpose_2d(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    /// One tile is 64×64 f32 = 16 KB per side, comfortably inside L1/L2.
    const TILE: usize = 64;
    let mut out = vec![0f32; rows * cols];
    if rows == 0 || cols == 0 {
        return out;
    }
    // Destination is `[cols, rows]`; hand each thread a band of its rows so the
    // writes never overlap.
    let band = TILE.max(cols.div_ceil(rayon::current_num_threads().max(1)));
    out.par_chunks_mut(band * rows)
        .enumerate()
        .for_each(|(chunk, dst)| {
            let j0 = chunk * band;
            let j1 = (j0 + band).min(cols);
            for jt in (j0..j1).step_by(TILE) {
                let jmax = (jt + TILE).min(j1);
                for it in (0..rows).step_by(TILE) {
                    let imax = (it + TILE).min(rows);
                    for i in it..imax {
                        let src = &data[i * cols + jt..i * cols + jmax];
                        for (o, v) in src.iter().enumerate() {
                            dst[(jt + o - j0) * rows + i] = *v;
                        }
                    }
                }
            }
        });
    out
}

/// Map of tensor name → (f32 data, shape).
pub struct WeightMap {
    tensors: HashMap<String, (Vec<f32>, Vec<usize>)>,
    /// The float dtype each tensor had on disk (safetensors loads record it),
    /// for [`crate::weight_storage::WeightStorage::Auto`]. Empty for maps built
    /// in memory.
    stored: HashMap<String, rlx_ir::DType>,
}

/// A safetensors dtype as the rlx float dtype it widens from, if it is one.
pub(crate) fn safetensors_float_dtype(d: safetensors::Dtype) -> Option<rlx_ir::DType> {
    Some(match d {
        safetensors::Dtype::F32 => rlx_ir::DType::F32,
        safetensors::Dtype::F16 => rlx_ir::DType::F16,
        safetensors::Dtype::BF16 => rlx_ir::DType::BF16,
        _ => return None,
    })
}

/// HF ↔ Hunyuan Q/K-norm name aliases (HY-MT `query_layernorm` / `key_layernorm`).
pub(crate) fn hunyuan_norm_aliases(key: &str) -> Vec<String> {
    let mut out = Vec::new();
    if let Some(prefix) = key.strip_suffix("self_attn.q_norm.weight") {
        out.push(format!("{prefix}self_attn.query_layernorm.weight"));
    }
    if let Some(prefix) = key.strip_suffix("self_attn.k_norm.weight") {
        out.push(format!("{prefix}self_attn.key_layernorm.weight"));
    }
    if let Some(prefix) = key.strip_suffix("self_attn.query_layernorm.weight") {
        out.push(format!("{prefix}self_attn.q_norm.weight"));
    }
    if let Some(prefix) = key.strip_suffix("self_attn.key_layernorm.weight") {
        out.push(format!("{prefix}self_attn.k_norm.weight"));
    }
    out
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
        Ok(Self {
            tensors,
            stored: HashMap::new(),
        })
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
        Ok((
            Self {
                tensors,
                stored: HashMap::new(),
            },
            packed,
        ))
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
        let mut stored = HashMap::new();
        for (name, view) in st.tensors() {
            if exclude.contains(name.as_str()) {
                continue;
            }
            if let Some(d) = safetensors_float_dtype(view.dtype()) {
                stored.insert(name.to_string(), d);
            }
            let shape: Vec<usize> = view.shape().to_vec();
            let bytes = view.data();
            let f32_data = match view.dtype() {
                safetensors::Dtype::F32 => bytes_to_f32_vec(bytes),
                safetensors::Dtype::F16 => bytes
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                    .collect(),
                safetensors::Dtype::BF16 => bytes
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
                    .collect(),
                safetensors::Dtype::I64 => bytes
                    .as_chunks::<8>()
                    .0
                    .iter()
                    .map(|c| i64::from_le_bytes(*c) as f32)
                    .collect(),
                safetensors::Dtype::I32 => bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
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

        Ok(Self { tensors, stored })
    }

    /// The float dtype `key` had on disk, when this map was loaded from a file.
    pub fn stored_dtype(&self, key: &str) -> Option<rlx_ir::DType> {
        self.stored.get(key).copied()
    }

    /// Attach recorded on-disk dtypes (runners that cache weights as f32 and
    /// rebuild maps per graph keep the dtypes beside the cache).
    pub fn with_stored_dtypes(mut self, dtypes: &HashMap<String, rlx_ir::DType>) -> Self {
        for k in self.tensors.keys() {
            if let Some(&d) = dtypes.get(k) {
                self.stored.insert(k.clone(), d);
            }
        }
        self
    }

    /// Record `key`'s on-disk dtype (loaders that build a map themselves).
    pub fn set_stored_dtype(&mut self, key: impl Into<String>, dtype: rlx_ir::DType) {
        self.stored.insert(key.into(), dtype);
    }

    /// Take a tensor by name (removes from map). Returns (data, shape).
    ///
    /// Also accepts Hunyuan dense / HY-MT aliases:
    /// `self_attn.query_layernorm` ↔ `self_attn.q_norm`,
    /// `self_attn.key_layernorm` ↔ `self_attn.k_norm`.
    pub fn take(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        if let Some(v) = self.tensors.remove(key) {
            return Ok(v);
        }
        for alt in hunyuan_norm_aliases(key) {
            if let Some(v) = self.tensors.remove(&alt) {
                return Ok(v);
            }
        }
        anyhow::bail!("weight not found: {key}")
    }

    /// Take and transpose a 2D weight: [out, in] → [in, out] for row-major matmul.
    pub fn take_transposed(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        let (data, shape) = self.take(key)?;
        if shape.len() != 2 {
            anyhow::bail!("transpose requires 2D, got {shape:?}");
        }
        let (rows, cols) = (shape[0], shape[1]);
        Ok((transpose_2d(&data, rows, cols), vec![cols, rows]))
    }

    /// Check if a key exists.
    pub fn has(&self, key: &str) -> bool {
        if self.tensors.contains_key(key) {
            return true;
        }
        hunyuan_norm_aliases(key)
            .into_iter()
            .any(|alt| self.tensors.contains_key(&alt))
    }

    /// Take every remaining tensor, leaving the map empty.
    ///
    /// 63 model crates spelled this as `for k in keys { p.insert(k, wm.take(&k)?) }`
    /// (some as `if let Ok(v) = wm.take(&k)`, which can never skip: a key from
    /// [`Self::keys`] is always present); this is the same drain.
    pub fn take_all(&mut self) -> HashMap<String, (Vec<f32>, Vec<usize>)> {
        std::mem::take(&mut self.tensors)
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
        Self {
            tensors,
            stored: HashMap::new(),
        }
    }

    /// Add (or replace) a tensor. For checkpoint preprocessors that synthesize
    /// tensors a builder expects — e.g. stacking per-expert MoE weights that the
    /// upstream checkpoint stores one tensor per expert. Returns the previous
    /// entry, if any.
    pub fn insert(
        &mut self,
        key: impl Into<String>,
        data: Vec<f32>,
        shape: Vec<usize>,
    ) -> Option<(Vec<f32>, Vec<usize>)> {
        self.tensors.insert(key.into(), (data, shape))
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
        let mut stored = HashMap::new();
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
            stored.extend(part.stored);
            any = true;
        }
        if !any {
            anyhow::bail!("no .safetensors files in {dir:?}");
        }
        Ok(Self {
            tensors: merged,
            stored,
        })
    }

    /// Rename keys in-place (e.g. strip `model.` HuggingFace prefix).
    pub fn remap_keys<F>(&mut self, mut f: F)
    where
        F: FnMut(String) -> String,
    {
        let keys: Vec<String> = self.tensors.keys().cloned().collect();
        for old in keys {
            if let Some(v) = self.tensors.remove(&old) {
                let dt = self.stored.remove(&old);
                let new = f(old);
                if let Some(d) = dt {
                    self.stored.insert(new.clone(), d);
                }
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
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }
}

#[cfg(test)]
mod transpose_tests {
    use super::transpose_2d;

    fn naive(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        let mut out = vec![0f32; data.len()];
        for i in 0..rows {
            for j in 0..cols {
                out[j * rows + i] = data[i * cols + j];
            }
        }
        out
    }

    #[test]
    fn tiled_transpose_matches_the_naive_one() {
        // Sizes that straddle the 64-wide tile and the rayon band split, plus
        // the degenerate single-row / single-column shapes.
        for &(r, c) in &[
            (1usize, 1usize),
            (1, 257),
            (257, 1),
            (64, 64),
            (65, 63),
            (127, 129),
            (256, 1024),
            (1024, 256),
            (300, 7),
        ] {
            let data: Vec<f32> = (0..r * c).map(|i| i as f32 * 0.5 - 3.0).collect();
            assert_eq!(transpose_2d(&data, r, c), naive(&data, r, c), "{r}x{c}");
        }
    }

    #[test]
    fn transposing_twice_is_the_identity() {
        let (r, c) = (193usize, 97usize);
        let data: Vec<f32> = (0..r * c).map(|i| (i % 251) as f32).collect();
        let once = transpose_2d(&data, r, c);
        assert_eq!(transpose_2d(&once, c, r), data);
    }

    #[test]
    fn empty_dimensions_do_not_panic() {
        assert!(transpose_2d(&[], 0, 8).is_empty());
        assert!(transpose_2d(&[], 8, 0).is_empty());
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
            stored: HashMap::new(),
        };
        let (data, shape) = wm.take_transposed("w").unwrap();
        assert_eq!(shape, vec![3, 2]);
        // Original: [[1,2,3],[4,5,6]] → Transposed: [[1,4],[2,5],[3,6]]
        assert_eq!(data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }
}
