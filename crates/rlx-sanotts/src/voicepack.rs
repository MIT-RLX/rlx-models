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

//! Voice-package loading: `manifest.json` + a flat little-endian fp16 blob whose
//! tensors are addressed by `offset_bytes`/`nbytes`.
//!
//! A package directory holds:
//! - `manifest.json` (format [`MANIFEST_FORMAT`])
//! - `weights.fp16.bin` (name from the manifest)
//! - `piper-phoneme-config.json` (codepoint → phoneme id table + espeak voice)

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::de::DeserializeOwned;
use sha2::{Digest, Sha256};

use crate::config::{MANIFEST_FORMAT, Manifest, PhonemeTable};

/// Name → (f32 data, shape). fp16/int weights are widened to f32 on load, which
/// is what every forward path here consumes.
#[derive(Debug, Default, Clone)]
pub struct TensorStore {
    map: HashMap<String, (Vec<f32>, Vec<usize>)>,
}

impl TensorStore {
    pub fn contains(&self, name: &str) -> bool {
        self.map.contains_key(name)
    }

    pub fn get(&self, name: &str) -> Result<&(Vec<f32>, Vec<usize>)> {
        self.map
            .get(name)
            .with_context(|| format!("missing tensor: {name}"))
    }

    pub fn data(&self, name: &str) -> Result<&[f32]> {
        Ok(self.get(name)?.0.as_slice())
    }

    pub fn shape(&self, name: &str) -> Result<&[usize]> {
        Ok(self.get(name)?.1.as_slice())
    }

    /// A `[1]`-shaped scalar (the residual blocks store their mixing scale this way).
    pub fn scalar(&self, name: &str) -> Result<f32> {
        let d = self.data(name)?;
        d.first()
            .copied()
            .with_context(|| format!("tensor {name} is empty"))
    }

    /// Insert or replace a tensor.
    ///
    /// Voice packages are read-only, so this exists for tools and tests that
    /// need to synthesize a component the shipped packs do not exercise — the
    /// decoder post-filter, for one.
    pub fn insert(&mut self, name: impl Into<String>, data: Vec<f32>, shape: Vec<usize>) {
        debug_assert_eq!(data.len(), shape.iter().product::<usize>());
        self.map.insert(name.into(), (data, shape));
    }

    /// Names present, sorted — used by the "no unexpected adapter" guards.
    pub fn names(&self) -> Vec<&str> {
        let mut v: Vec<&str> = self.map.keys().map(String::as_str).collect();
        v.sort_unstable();
        v
    }
}

/// A loaded voice package: manifest plus the raw weights blob.
pub struct VoicePack {
    pub name: String,
    pub directory: PathBuf,
    pub manifest: Manifest,
    weights: Vec<u8>,
}

impl std::fmt::Debug for VoicePack {
    /// Deliberately hand-written: the weights blob is megabytes, and a derived
    /// `Debug` would dump all of it into any `unwrap`/`assert` message.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VoicePack")
            .field("name", &self.name)
            .field("directory", &self.directory)
            .field("format", &self.manifest.format)
            .field("sample_rate", &self.manifest.sample_rate)
            .field("weights_bytes", &self.weights.len())
            .finish()
    }
}

impl VoicePack {
    /// Load from a directory containing `manifest.json` + the weights blob.
    ///
    /// Verifies the blob's declared size and (when present) its SHA-256, so a
    /// truncated or swapped file fails loudly instead of decoding into noise.
    pub fn load_from_dir(directory: impl AsRef<Path>) -> Result<Self> {
        let directory = directory.as_ref();
        let name = directory
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "voice".to_string());
        Self::load_named(name, directory)
    }

    /// Same as [`VoicePack::load_from_dir`] with an explicit voice name.
    pub fn load_named(name: impl Into<String>, directory: impl AsRef<Path>) -> Result<Self> {
        let directory = directory.as_ref().to_path_buf();
        let manifest_path = directory.join("manifest.json");
        let manifest_text = std::fs::read_to_string(&manifest_path)
            .with_context(|| format!("read {}", manifest_path.display()))?;
        let manifest: Manifest = serde_json::from_str(&manifest_text)
            .with_context(|| format!("parse {}", manifest_path.display()))?;
        if manifest.format != MANIFEST_FORMAT {
            bail!(
                "{}: unsupported manifest format {:?}, expected {MANIFEST_FORMAT:?}",
                manifest_path.display(),
                manifest.format
            );
        }

        let weights_path = directory.join(&manifest.weights_file);
        let weights = std::fs::read(&weights_path)
            .with_context(|| format!("read {}", weights_path.display()))?;
        if manifest.weights_size_bytes >= 0 && weights.len() as i64 != manifest.weights_size_bytes {
            bail!(
                "{}: size {} != manifest weights_size_bytes {}",
                weights_path.display(),
                weights.len(),
                manifest.weights_size_bytes
            );
        }
        if let Some(expected) = manifest.weights_sha256.as_deref() {
            let actual = hex_lower(&Sha256::digest(&weights));
            if actual != expected.to_ascii_lowercase() {
                bail!(
                    "{}: sha256 mismatch (manifest={expected}, actual={actual}); \
                     the voice package is corrupt or was tampered with",
                    weights_path.display()
                );
            }
        }

        Ok(Self {
            name: name.into(),
            directory,
            manifest,
            weights,
        })
    }

    pub fn sample_rate(&self) -> u32 {
        self.manifest.sample_rate
    }

    /// The voice's own default speaking-rate scale (larger = slower).
    pub fn duration_length_scale(&self) -> f32 {
        self.manifest.inference.duration_length_scale
    }

    /// Materialize one component's tensors as f32.
    pub fn component_tensors(&self, component: &str) -> Result<TensorStore> {
        let comp = self
            .manifest
            .components
            .get(component)
            .with_context(|| format!("manifest has no component {component:?}"))?;
        let mut map = HashMap::with_capacity(comp.tensors.len());
        for tensor in &comp.tensors {
            let end = tensor.offset_bytes.saturating_add(tensor.nbytes);
            if end > self.weights.len() {
                bail!(
                    "{component}.{}: truncated weights blob (wanted {} bytes at {}, have {})",
                    tensor.name,
                    tensor.nbytes,
                    tensor.offset_bytes,
                    self.weights.len()
                );
            }
            let raw = &self.weights[tensor.offset_bytes..end];
            let data = decode(raw, &tensor.dtype)
                .with_context(|| format!("{component}.{}", tensor.name))?;
            let want: usize = tensor.shape.iter().product();
            if data.len() != want {
                bail!(
                    "{component}.{}: {} elements decoded but shape {:?} needs {want}",
                    tensor.name,
                    data.len(),
                    tensor.shape
                );
            }
            map.insert(tensor.name.clone(), (data, tensor.shape.clone()));
        }
        Ok(TensorStore { map })
    }

    /// Deserialize one component's config into a typed struct.
    pub fn component_config<T: DeserializeOwned>(&self, component: &str) -> Result<T> {
        let comp = self
            .manifest
            .components
            .get(component)
            .with_context(|| format!("manifest has no component {component:?}"))?;
        serde_json::from_value(comp.config.clone())
            .with_context(|| format!("parse {component} config"))
    }

    /// Path to the bundled Piper phoneme config.
    pub fn phoneme_config_path(&self) -> PathBuf {
        let name = self
            .manifest
            .frontend
            .included_config
            .clone()
            .unwrap_or_else(|| "piper-phoneme-config.json".to_string());
        self.directory.join(name)
    }

    /// Parse the bundled Piper phoneme config.
    pub fn phoneme_table(&self) -> Result<PhonemeTable> {
        let path = self.phoneme_config_path();
        let text = std::fs::read_to_string(&path).with_context(|| {
            format!(
                "voice pack {:?} is missing its phoneme config: {}",
                self.name,
                path.display()
            )
        })?;
        PhonemeTable::from_json(&text).with_context(|| format!("parse {}", path.display()))
    }
}

fn decode(raw: &[u8], dtype: &str) -> Result<Vec<f32>> {
    let out = match dtype {
        "float16" => {
            if !raw.len().is_multiple_of(2) {
                bail!("float16 tensor has odd byte length {}", raw.len());
            }
            raw.chunks_exact(2)
                .map(|c| f32::from(half::f16::from_le_bytes([c[0], c[1]])))
                .collect()
        }
        "float32" => {
            if !raw.len().is_multiple_of(4) {
                bail!(
                    "float32 tensor has non-multiple-of-4 byte length {}",
                    raw.len()
                );
            }
            raw.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        }
        "int32" => raw
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32)
            .collect(),
        "int64" => raw
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32)
            .collect(),
        other => bail!("unsupported dtype {other:?}"),
    };
    Ok(out)
}

fn hex_lower(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        let _ = write!(s, "{b:02x}");
    }
    s
}
