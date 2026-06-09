// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Safetensors → [`WeightMap`] for native RLX graph build.

use anyhow::{Context, Result, bail};
use memmap2::{Mmap, MmapOptions};
use rlx_core::weight_map::WeightMap;
use safetensors::SafeTensors;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use super::paths::is_rten_checkpoint;

/// Mmap-backed safetensors file; reuse across per-width graph builds.
pub struct SafetensorsFile {
    path: PathBuf,
    mmap: Mutex<Option<Mmap>>,
}

impl SafetensorsFile {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Ok(Self {
            path: path.as_ref().to_path_buf(),
            mmap: Mutex::new(None),
        })
    }

    fn with_mmap<R>(&self, f: impl FnOnce(&Mmap) -> Result<R>) -> Result<R> {
        let mut guard = self
            .mmap
            .lock()
            .map_err(|_| anyhow::anyhow!("safetensors mmap lock poisoned"))?;
        if guard.is_none() {
            let file = File::open(&self.path).with_context(|| format!("open {:?}", self.path))?;
            *guard = Some(unsafe { MmapOptions::new().map(&file)? });
        }
        f(guard.as_ref().unwrap())
    }

    /// Fresh [`WeightMap`] for graph build (drains keys from a parse of the mmap).
    pub fn weight_map(&self) -> Result<WeightMap> {
        self.with_mmap(load_safetensors_weights_from_mmap)
    }
}

/// Load a `.safetensors` file via mmap-backed read into f32 [`WeightMap`] tensors.
pub fn load_safetensors(path: &Path) -> Result<WeightMap> {
    SafetensorsFile::open(path)?.weight_map()
}

pub(crate) fn load_safetensors_weights_from_mmap(mmap: &Mmap) -> Result<WeightMap> {
    let mut wm = drain_safetensors_bytes(mmap)?;
    strip_graph_scalars(&mut wm);
    Ok(wm)
}

fn drain_safetensors_bytes(data: &[u8]) -> Result<WeightMap> {
    let st = SafeTensors::deserialize(data).context("parse safetensors")?;
    let mut tensors = std::collections::HashMap::new();
    for (name, view) in st.tensors() {
        let shape: Vec<usize> = view.shape().to_vec();
        let bytes = view.data();
        let f32_data = match view.dtype() {
            safetensors::Dtype::F32 => {
                if bytes.len() % 4 != 0 {
                    bail!("{name}: invalid f32 byte length");
                }
                bytes
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect()
            }
            safetensors::Dtype::F16 => bytes
                .chunks_exact(2)
                .map(|c| ::half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect(),
            safetensors::Dtype::BF16 => bytes
                .chunks_exact(2)
                .map(|c| ::half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect(),
            other => bail!("{name}: unsupported dtype {other:?}"),
        };
        tensors.insert(name.to_string(), (f32_data, shape));
    }
    Ok(WeightMap::from_tensors(tensors))
}

/// Drop RTen ONNX scalar nodes (slice/pad helpers) not used by the native graph.
fn strip_graph_scalars(wm: &mut WeightMap) {
    let remove: Vec<String> = wm
        .keys()
        .filter(|k| k.starts_with('/') || k.contains("Constant") || k.contains("Unsqueeze"))
        .map(str::to_string)
        .collect();
    for k in remove {
        let _ = wm.take(&k);
    }
}

/// Load weights for RLX graph build (safetensors only).
pub fn load_safetensors_weights(path: &Path) -> Result<WeightMap> {
    if is_rten_checkpoint(path) {
        bail!(
            "RLX graph weights require .safetensors ({:?}); run `rlx-ocr-convert` on .rten checkpoints",
            path
        );
    }
    SafetensorsFile::open(path)?.weight_map()
}

/// Alias for [`load_safetensors_weights`].
pub fn load_rlx_weights(path: &Path) -> Result<WeightMap> {
    load_safetensors_weights(path)
}
