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

//! Mmap-backed HuggingFace sharded safetensors checkpoints.
//!
//! Keeps shard files mapped once and reuses the mapping across selective
//! [`WeightMap`] loads (prefix subsets, single keys, etc.).

use crate::weight_map::WeightMap;
use anyhow::{Context, Result, bail};
use memmap2::{Mmap, MmapOptions};
use safetensors::SafeTensors;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

type F32TensorLoad = (String, Vec<f32>, Vec<usize>);

/// A shard's bytes: mapped from a file, or held in memory (a browser has no
/// files: [`SafetensorsCheckpoint::from_bytes`]).
pub enum ShardBytes {
    Mapped(Mmap),
    Owned(Vec<u8>),
}

impl std::ops::Deref for ShardBytes {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        match self {
            Self::Mapped(m) => m,
            Self::Owned(v) => v,
        }
    }
}

impl AsRef<[u8]> for ShardBytes {
    fn as_ref(&self) -> &[u8] {
        self
    }
}

/// Cached mmap views of a sharded (or flat) safetensors directory.
pub struct SafetensorsCheckpoint {
    dir: PathBuf,
    /// Tensor name → shard filename (relative to `dir`).
    index: HashMap<String, String>,
    shards: Mutex<HashMap<String, Arc<ShardBytes>>>,
}

impl SafetensorsCheckpoint {
    pub fn open(dir: &Path) -> Result<Self> {
        let dir = dir.to_path_buf();
        let index_path = dir.join("model.safetensors.index.json");
        let index = if index_path.is_file() {
            let raw = std::fs::read(&index_path).context("read model.safetensors.index.json")?;
            let parsed: serde_json::Value =
                serde_json::from_slice(&raw).context("parse model.safetensors.index.json")?;
            let weight_map = parsed
                .get("weight_map")
                .and_then(|m| m.as_object())
                .context("weight_map in index")?;
            weight_map
                .iter()
                .filter_map(|(k, v)| v.as_str().map(|shard| (k.clone(), shard.to_string())))
                .collect()
        } else {
            Self::index_from_dir(&dir)?
        };
        if index.is_empty() {
            bail!("no safetensors tensors found under {dir:?}");
        }
        Ok(Self {
            dir,
            index,
            shards: Mutex::new(HashMap::new()),
        })
    }

    /// One safetensors file (no index): only that file's tensors, mapped
    /// lazily — not every `*.safetensors` that happens to share its directory.
    pub fn open_file(file: &Path) -> Result<Self> {
        let dir = file
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));
        let name = file
            .file_name()
            .and_then(|s| s.to_str())
            .map(str::to_string)
            .context("non-utf8 safetensors filename")?;
        let mmap = map_file(file)?;
        let st = SafeTensors::deserialize(mmap.as_ref()).context("parse safetensors")?;
        let index: HashMap<String, String> = st
            .names()
            .into_iter()
            .map(|k| (k.to_string(), name.clone()))
            .collect();
        if index.is_empty() {
            bail!("no safetensors tensors in {file:?}");
        }
        Ok(Self {
            dir,
            index,
            shards: Mutex::new(HashMap::from([(name, mmap)])),
        })
    }

    /// Safetensors files held in memory, `(name, bytes)` each: what a page
    /// fetched. Every tensor of every file is indexed; nothing is copied.
    pub fn from_bytes(files: Vec<(String, Vec<u8>)>) -> Result<Self> {
        let mut index = HashMap::new();
        let mut shards = HashMap::new();
        for (name, bytes) in files {
            let st = SafeTensors::deserialize(&bytes)
                .with_context(|| format!("parse safetensors {name}"))?;
            for key in st.names() {
                index.insert(key.to_string(), name.clone());
            }
            shards.insert(name, Arc::new(ShardBytes::Owned(bytes)));
        }
        if index.is_empty() {
            bail!("no safetensors tensors in the given bytes");
        }
        Ok(Self {
            dir: PathBuf::new(),
            index,
            shards: Mutex::new(shards),
        })
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// The dtype a tensor is stored in, read from its shard's header.
    pub fn stored_dtype(&self, name: &str) -> Option<safetensors::Dtype> {
        let shard = self.index.get(name)?;
        let mmap = self.mmap_shard(shard).ok()?;
        let st = SafeTensors::deserialize(mmap.as_ref()).ok()?;
        st.tensor(name).ok().map(|v| v.dtype())
    }

    /// A tensor's shape, from its shard's header (no data read).
    pub fn shape_of(&self, name: &str) -> Option<Vec<usize>> {
        let shard = self.index.get(name)?;
        let bytes = self.mmap_shard(shard).ok()?;
        let st = SafeTensors::deserialize(bytes.as_ref()).ok()?;
        st.tensor(name).ok().map(|v| v.shape().to_vec())
    }

    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.index.keys().map(|s| s.as_str())
    }

    /// Whether a tensor name is present in the checkpoint index.
    pub fn contains(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }

    /// Raw storage bytes + dtype + shape for one tensor, copied out of the
    /// mmap. For custom codecs (e.g. integer-quant) that need the native
    /// bytes rather than an F32 conversion. Bytes are in the file's storage
    /// order (row-major, dtype-native).
    pub fn tensor_raw(&self, name: &str) -> Result<(Vec<u8>, safetensors::Dtype, Vec<usize>)> {
        let shard = self
            .index
            .get(name)
            .with_context(|| format!("tensor {name} not in checkpoint index"))?;
        let mmap = self.mmap_shard(shard)?;
        let st = SafeTensors::deserialize(mmap.as_ref()).context("parse safetensors")?;
        let view = st
            .tensor(name)
            .with_context(|| format!("tensor {name} missing in shard {shard}"))?;
        Ok((view.data().to_vec(), view.dtype(), view.shape().to_vec()))
    }

    /// Dequantize one tensor to F32 without loading the rest of the shard.
    pub fn load_tensor_f32(&self, name: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        let shard = self
            .index
            .get(name)
            .with_context(|| format!("tensor {name} not in checkpoint index"))?;
        let mmap = self.mmap_shard(shard)?;
        let st = SafeTensors::deserialize(mmap.as_ref()).context("parse safetensors")?;
        let view = st
            .tensor(name)
            .with_context(|| format!("tensor {name} missing in shard {shard}"))?;
        let shape = view.shape().to_vec();
        let fp8 = matches!(
            view.dtype(),
            safetensors::Dtype::F8_E4M3 | safetensors::Dtype::F8_E5M2
        );
        let mut data = tensor_view_to_f32(name, view)?;
        if fp8 && let Some(skey) = self.fp8_scale_key(name) {
            let (scale, sshape) = self.load_tensor_f32(&skey)?;
            rlx_ir::lowp_codec::apply_weight_scales(
                &mut data,
                &shape,
                &scale,
                &sshape,
                self.fp8_block(),
            )
            .map_err(|e| anyhow::anyhow!("{name}: {e}"))?;
        }
        Ok((data, shape))
    }

    /// The scale an HF FP8 checkpoint stores beside FP8 weight `name`
    /// (`<name>_scale_inv`, or compressed-tensors' `<name>_scale`).
    pub fn fp8_scale_key(&self, name: &str) -> Option<String> {
        ["_scale_inv", "_scale"]
            .iter()
            .map(|s| format!("{name}{s}"))
            .find(|k| self.index.contains_key(k))
    }

    /// Whether `name` is the stored scale of an FP8 weight — folded into that
    /// weight by [`Self::load_tensor_f32`], so not a tensor of its own.
    pub fn is_fp8_scale(&self, name: &str) -> bool {
        ["_scale_inv", "_scale"].iter().any(|s| {
            name.strip_suffix(s)
                .filter(|w| self.index.contains_key(*w))
                .and_then(|w| self.stored_dtype(w))
                .is_some_and(|d| {
                    matches!(d, safetensors::Dtype::F8_E4M3 | safetensors::Dtype::F8_E5M2)
                })
        })
    }

    /// `quantization_config.weight_block_size` from the checkpoint's
    /// `config.json`; `None` lets the scale shape decide.
    pub fn fp8_block(&self) -> Option<[usize; 2]> {
        let raw = std::fs::read(self.dir.join("config.json")).ok()?;
        let cfg: serde_json::Value = serde_json::from_slice(&raw).ok()?;
        match cfg
            .get("quantization_config")?
            .get("weight_block_size")?
            .as_array()?
            .as_slice()
        {
            [r, c] => Some([r.as_u64()? as usize, c.as_u64()? as usize]),
            _ => None,
        }
    }

    /// Load selected rows from a rank-2 tensor without materializing the full matrix.
    pub fn load_tensor_rows_f32(
        &self,
        name: &str,
        rows: &[u32],
        cols: usize,
    ) -> Result<Vec<Vec<f32>>> {
        let shard = self
            .index
            .get(name)
            .with_context(|| format!("tensor {name} not in checkpoint index"))?;
        let mmap = self.mmap_shard(shard)?;
        let st = SafeTensors::deserialize(mmap.as_ref()).context("parse safetensors")?;
        let view = st
            .tensor(name)
            .with_context(|| format!("tensor {name} missing in shard {shard}"))?;
        let shape: Vec<usize> = view.shape().to_vec();
        anyhow::ensure!(
            shape.len() == 2,
            "{name}: expected rank-2 embed, got shape {shape:?}"
        );
        anyhow::ensure!(
            shape[1] == cols,
            "{name}: cols {} != expected {cols}",
            shape[1]
        );
        let vocab = shape[0];
        let mut out = Vec::with_capacity(rows.len());
        for &row in rows {
            let r = row as usize;
            anyhow::ensure!(r < vocab, "{name}: row {row} >= vocab {vocab}");
            out.push(read_tensor_row_f32(&view, r, cols)?);
        }
        Ok(out)
    }

    pub fn load_selected(&self, want: &HashSet<String>) -> Result<WeightMap> {
        if want.is_empty() {
            bail!("load_selected: empty key set");
        }
        let mut shard_files: HashSet<String> = HashSet::new();
        for key in want {
            if let Some(shard) = self.index.get(key) {
                shard_files.insert(shard.clone());
            }
        }
        if shard_files.is_empty() {
            bail!("no requested tensors found under {:?}", self.dir);
        }

        let mut tensors = HashMap::with_capacity(want.len());
        if shard_files.len() == 1 {
            let shard = shard_files.into_iter().next().unwrap();
            let mmap = self.mmap_shard(&shard)?;
            ingest_selected_from_bytes(mmap.as_ref(), want, &mut tensors)?;
        } else {
            let mut handles = Vec::with_capacity(shard_files.len());
            for shard in shard_files {
                let mmap = self.mmap_shard(&shard)?;
                let want = want.clone();
                handles.push(std::thread::spawn(move || {
                    let mut local = HashMap::new();
                    ingest_selected_from_bytes(mmap.as_ref(), &want, &mut local)?;
                    Ok::<_, anyhow::Error>(local)
                }));
            }
            for handle in handles {
                let part = handle
                    .join()
                    .map_err(|_| anyhow::anyhow!("shard load thread panicked"))??;
                tensors.extend(part);
            }
        }
        if tensors.is_empty() {
            bail!("no requested tensors loaded from {:?}", self.dir);
        }
        let mut map = WeightMap::from_tensors(tensors);
        // On-disk dtypes, one header parse per shard.
        let shards: HashSet<&String> = want.iter().filter_map(|k| self.index.get(k)).collect();
        for shard in shards {
            let mmap = self.mmap_shard(shard)?;
            let st = SafeTensors::deserialize(mmap.as_ref()).context("parse safetensors")?;
            for (name, view) in st.tensors() {
                if want.contains(&name)
                    && let Some(d) = crate::weight_map::safetensors_float_dtype(view.dtype())
                {
                    map.set_stored_dtype(name, d);
                }
            }
        }
        Ok(map)
    }

    fn index_from_dir(dir: &Path) -> Result<HashMap<String, String>> {
        let mut index = HashMap::new();
        for entry in std::fs::read_dir(dir).with_context(|| format!("read_dir {dir:?}"))? {
            let path = entry?.path();
            if path.extension().and_then(|s| s.to_str()) != Some("safetensors") {
                continue;
            }
            let name = path
                .file_name()
                .and_then(|s| s.to_str())
                .map(str::to_string)
                .context("non-utf8 shard filename")?;
            let mmap = map_file(&path)?;
            let st = SafeTensors::deserialize(mmap.as_ref()).context("parse safetensors")?;
            for key in st.names() {
                index.insert(key.to_string(), name.clone());
            }
        }
        Ok(index)
    }

    fn mmap_shard(&self, shard: &str) -> Result<Arc<ShardBytes>> {
        let mut guard = self
            .shards
            .lock()
            .map_err(|_| anyhow::anyhow!("safetensors shard cache lock poisoned"))?;
        if let Some(m) = guard.get(shard) {
            return Ok(Arc::clone(m));
        }
        let path = self.dir.join(shard);
        let mmap = map_file(&path)?;
        guard.insert(shard.to_string(), Arc::clone(&mmap));
        Ok(mmap)
    }
}

fn map_file(path: &Path) -> Result<Arc<ShardBytes>> {
    let file = File::open(path).with_context(|| format!("open {path:?}"))?;
    let mmap = unsafe { MmapOptions::new().map(&file)? };
    Ok(Arc::new(ShardBytes::Mapped(mmap)))
}

fn ingest_selected_from_bytes(
    data: &[u8],
    want: &HashSet<String>,
    tensors: &mut HashMap<String, (Vec<f32>, Vec<usize>)>,
) -> Result<()> {
    use rayon::prelude::*;
    let st = SafeTensors::deserialize(data).context("parsing safetensors")?;
    // Collect wanted (name, view) pairs first so we can parallelize the
    // per-tensor BF16/F16 → F32 conversion (the dominant cost on bf16 ckpts).
    let selected: Vec<(String, safetensors::tensor::TensorView<'_>)> = st
        .tensors()
        .into_iter()
        .filter(|(name, _)| want.contains(name.as_str()))
        .map(|(name, view)| (name.to_string(), view))
        .collect();
    let converted: Vec<Result<F32TensorLoad>> = selected
        .into_par_iter()
        .map(|(name, view)| {
            let shape: Vec<usize> = view.shape().to_vec();
            let f32_data = tensor_bytes_to_f32(name.as_str(), view)?;
            Ok((name, f32_data, shape))
        })
        .collect();
    for r in converted {
        let (name, f32_data, shape) = r?;
        if f32_data.is_empty() {
            continue;
        }
        tensors.insert(name, (f32_data, shape));
    }
    Ok(())
}

fn read_tensor_row_f32(
    view: &safetensors::tensor::TensorView<'_>,
    row: usize,
    cols: usize,
) -> Result<Vec<f32>> {
    let bytes = view.data();
    match view.dtype() {
        safetensors::Dtype::F32 => {
            let row_bytes = cols * 4;
            let off = row * row_bytes;
            Ok(bytes[off..off + row_bytes]
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect())
        }
        safetensors::Dtype::F16 => {
            let row_bytes = cols * 2;
            let off = row * row_bytes;
            Ok(bytes[off..off + row_bytes]
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect())
        }
        safetensors::Dtype::BF16 => {
            let row_bytes = cols * 2;
            let off = row * row_bytes;
            Ok(bytes[off..off + row_bytes]
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect())
        }
        other => anyhow::bail!("row slice: unsupported dtype {other:?}"),
    }
}

fn tensor_bytes_to_f32(name: &str, view: safetensors::tensor::TensorView<'_>) -> Result<Vec<f32>> {
    tensor_view_to_f32(name, view)
}

/// Decode a safetensors tensor view into a contiguous f32 vector (C order).
pub fn tensor_view_to_f32(
    name: &str,
    view: safetensors::tensor::TensorView<'_>,
) -> Result<Vec<f32>> {
    raw_to_f32(name, view.dtype(), view.data())
}

/// Convert one tensor's raw little-endian bytes to F32.
pub fn raw_to_f32(name: &str, dtype: safetensors::Dtype, bytes: &[u8]) -> Result<Vec<f32>> {
    Ok(match dtype {
        safetensors::Dtype::F32 => super::weight_map::bytes_to_f32_vec(bytes),
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
        // HF FP8 checkpoints: the raw codes. `load_tensor_f32` applies the
        // stored scale.
        safetensors::Dtype::F8_E4M3 | safetensors::Dtype::F8_E5M2 => {
            let fmt = if dtype == safetensors::Dtype::F8_E4M3 {
                rlx_ir::quant::ScaledFormat::F8E4M3
            } else {
                rlx_ir::quant::ScaledFormat::F8E5M2
            };
            let mut out = vec![0f32; bytes.len()];
            rlx_ir::lowp_codec::decode_slice(fmt, bytes, &mut out);
            out
        }
        safetensors::Dtype::C64 => return Ok(vec![]),
        other => anyhow::bail!("{name}: unsupported dtype {other:?}"),
    })
}

/// `WeightLoader` that keeps shards mmap'd and only materializes F32 on
/// `WeightLoader::take`. Avoids the ~2× RAM spike of
/// [`WeightMap::from_safetensors_dir`] (full BF16→F32 HashMap before drain).
pub struct SafetensorsMmapLoader {
    ckpt: SafetensorsCheckpoint,
    taken: HashSet<String>,
    /// Hand `BF16` matrices to [`WeightLoader::take_packed`] as
    /// `QuantScheme::GgufBF16` (raw bf16 rows), like
    /// [`crate::weight_loader::GgufLoader::keep_bf16`].
    keep_bf16: bool,
}

impl SafetensorsMmapLoader {
    pub fn open(dir: &Path) -> Result<Self> {
        Ok(Self {
            ckpt: SafetensorsCheckpoint::open(dir)?,
            taken: HashSet::new(),
            keep_bf16: false,
        })
    }

    /// One unindexed safetensors file, mapped lazily.
    pub fn open_file(file: &Path) -> Result<Self> {
        Ok(Self {
            ckpt: SafetensorsCheckpoint::open_file(file)?,
            taken: HashSet::new(),
            keep_bf16: false,
        })
    }

    /// Report `BF16` `[out, in]` tensors to `take_packed` as
    /// `QuantScheme::GgufBF16`, so a builder keeps them at 2 bytes as a
    /// `DequantMatMul` weight — with `RLX_PACKED_LM_HEAD`, a tied bf16
    /// embedding serves as the LM head without an f32 `[hidden, vocab]` copy.
    pub fn keep_bf16(&mut self, on: bool) {
        self.keep_bf16 = on;
    }

    /// `key`, or the Hunyuan norm alias the eager [`WeightMap`] would accept.
    fn resolve(&self, key: &str) -> Option<String> {
        if self.ckpt.contains(key) {
            return Some(key.to_string());
        }
        crate::weight_map::hunyuan_norm_aliases(key)
            .into_iter()
            .find(|a| self.ckpt.contains(a))
    }

    pub fn checkpoint(&self) -> &SafetensorsCheckpoint {
        &self.ckpt
    }

    pub fn has(&self, key: &str) -> bool {
        self.ckpt.contains(key) && !self.taken.contains(key)
    }
}

impl crate::weight_loader::WeightLoader for SafetensorsMmapLoader {
    fn format_id(&self) -> &'static str {
        "safetensors-mmap"
    }

    fn len(&self) -> usize {
        self.remaining_keys().len()
    }

    fn take(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        let Some(real) = self.resolve(key) else {
            bail!("weight not found: {key}");
        };
        if self.taken.contains(&real) {
            bail!("weight already taken: {key}");
        }
        let (data, shape) = self.ckpt.load_tensor_f32(&real)?;
        self.taken.insert(real);
        Ok((data, shape))
    }

    fn take_transposed(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        let (data, shape) = self.take(key)?;
        if shape.len() != 2 {
            bail!("transpose requires 2D, got {shape:?}");
        }
        let (rows, cols) = (shape[0], shape[1]);
        Ok((
            crate::weight_map::transpose_2d(&data, rows, cols),
            vec![cols, rows],
        ))
    }

    /// HF FP8 linears (`F8_E4M3` / `F8_E5M2`, `[out, in]` or an `[E, out, in]`
    /// bank) as `Fp8E4m3B128` / `Fp8E5m2B128` blocks for `DequantMatMul`, the
    /// stored scale folded in. `None` for everything else, or an FP8 layout
    /// the blocks can't hold — the caller then takes it as f32.
    fn take_packed(&mut self, key: &str) -> Result<Option<crate::weight_map::PackedWeightTensor>> {
        use rlx_ir::lowp_codec::fp8_b128;
        let Some(real) = self.resolve(key).filter(|r| !self.taken.contains(r)) else {
            return Ok(None);
        };
        let scheme = match self.ckpt.stored_dtype(&real) {
            Some(safetensors::Dtype::F8_E4M3) => rlx_ir::quant::QuantScheme::Fp8E4m3B128,
            Some(safetensors::Dtype::F8_E5M2) => rlx_ir::quant::QuantScheme::Fp8E5m2B128,
            Some(safetensors::Dtype::BF16) if self.keep_bf16 => {
                let (bytes, _, shape) = self.ckpt.tensor_raw(&real)?;
                if shape.len() != 2 {
                    return Ok(None);
                }
                self.taken.insert(real);
                return Ok(Some((bytes, rlx_ir::quant::QuantScheme::GgufBF16, shape)));
            }
            _ => return Ok(None),
        };
        let (codes, _, shape) = self.ckpt.tensor_raw(&real)?;
        let scale = match self.ckpt.fp8_scale_key(&real) {
            Some(skey) => Some(self.ckpt.load_tensor_f32(&skey)?),
            None => None,
        };
        let packed = fp8_b128::pack_scaled(
            &codes,
            &shape,
            scale.as_ref().map(|(s, sh)| (s.as_slice(), sh.as_slice())),
            self.ckpt.fp8_block(),
        )
        .map_err(|e| anyhow::anyhow!("{key}: {e}"))?;
        Ok(packed.map(|w_q| {
            self.taken.insert(real);
            (w_q, scheme, shape)
        }))
    }

    fn stored_dtype(&self, key: &str) -> Option<rlx_ir::DType> {
        use safetensors::Dtype as S;
        Some(match self.ckpt.stored_dtype(&self.resolve(key)?)? {
            S::F32 => rlx_ir::DType::F32,
            S::F16 => rlx_ir::DType::F16,
            S::BF16 => rlx_ir::DType::BF16,
            _ => return None,
        })
    }

    /// Untaken tensors, less the scales of FP8 weights (part of their weight).
    fn remaining_keys(&self) -> Vec<String> {
        self.ckpt
            .keys()
            .filter(|k| !self.taken.contains(*k) && !self.ckpt.is_fp8_scale(k))
            .map(|s| s.to_string())
            .collect()
    }
}

#[cfg(test)]
mod fp8_tests {
    use super::*;
    use crate::weight_loader::WeightLoader;
    use safetensors::tensor::{Dtype, TensorView};

    /// An HF block-FP8 linear (`F8_E4M3` + bf16 `weight_scale_inv` 128×128
    /// grid, as Qwen3-FP8 ships): `take` gives the scaled f32, `take_packed`
    /// the `Fp8E4m3B128` blocks that dequantize to the same values, and the
    /// scale tensor is not a key of its own.
    #[test]
    fn fp8_linear_takes_scaled_and_packed() {
        let (n, k) = (130usize, 256usize);
        let codes: Vec<u8> = (0..n * k).map(|i| (i % 0x7e) as u8).collect();
        let grid: Vec<u8> = (0..2 * 2)
            .flat_map(|t| half::bf16::from_f32(0.25 * (t as f32 + 1.0)).to_le_bytes())
            .collect();
        let bias: Vec<u8> = [1.0f32, 2.0].iter().flat_map(|v| v.to_le_bytes()).collect();
        let tensors = [
            (
                "l.weight",
                TensorView::new(Dtype::F8_E4M3, vec![n, k], &codes).unwrap(),
            ),
            (
                "l.weight_scale_inv",
                TensorView::new(Dtype::BF16, vec![2, 2], &grid).unwrap(),
            ),
            (
                "l.bias",
                TensorView::new(Dtype::F32, vec![2], &bias).unwrap(),
            ),
        ];
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("model.safetensors");
        safetensors::serialize_to_file(tensors, None, &file).unwrap();

        let mut f32_loader = SafetensorsMmapLoader::open_file(&file).unwrap();
        let mut keys = f32_loader.remaining_keys();
        keys.sort();
        assert_eq!(keys, ["l.bias", "l.weight"]);
        let (w, shape) = f32_loader.take("l.weight").unwrap();
        assert_eq!(shape, vec![n, k]);
        for (t, v) in w.iter().enumerate() {
            let (i, j) = (t / k, t % k);
            let s = 0.25 * (((i / 128) * 2 + j / 128) as f32 + 1.0);
            let want = rlx_ir::quant::ScaledFormat::F8E4M3.decode(codes[t]) * s;
            assert_eq!(*v, want, "({i}, {j})");
        }

        let mut packed = SafetensorsMmapLoader::open_file(&file).unwrap();
        let (w_q, scheme, pshape) = packed.take_packed("l.weight").unwrap().expect("FP8 packs");
        assert_eq!(
            (scheme, pshape),
            (rlx_ir::quant::QuantScheme::Fp8E4m3B128, vec![n, k])
        );
        let back = rlx_ir::lowp_codec::fp8_b128::dequant_scheme(scheme, &w_q, n * k).unwrap();
        assert_eq!(back, w);
        assert!(
            packed.take_packed("l.bias").unwrap().is_none(),
            "f32 stays unpacked"
        );
        assert_eq!(packed.remaining_keys(), ["l.bias"]);
    }
}
