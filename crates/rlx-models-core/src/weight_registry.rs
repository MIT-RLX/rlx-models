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

//! Extensible weight-format registry — register custom loaders for new extensions.

use anyhow::{Context, Result, anyhow};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use crate::gguf_support::{ResolveWeightsOptions, resolve_weights_file_with_options};
use crate::weight_loader::{GgufLoader, WeightLoader};
use crate::weight_map::{WeightDrainPolicy, WeightMap};

/// Opens a file path into a [`WeightLoader`].
pub type WeightLoaderFactory = fn(&Path) -> Result<Box<dyn WeightLoader>>;

/// Describes one on-disk weight format.
#[derive(Clone, Copy)]
pub struct WeightFormatRegistration {
    pub id: &'static str,
    pub extensions: &'static [&'static str],
    pub open: WeightLoaderFactory,
}

fn open_safetensors(path: &Path) -> Result<Box<dyn WeightLoader>> {
    // HuggingFace sharded checkpoints: prefer mmap index over a single shard file
    // so callers can pass either the directory or any shard path.
    let dir = if path.is_dir() {
        path.to_path_buf()
    } else {
        path.parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| path.to_path_buf())
    };
    if dir.join("model.safetensors.index.json").is_file() {
        return Ok(Box::new(
            crate::safetensors_checkpoint::SafetensorsMmapLoader::open(&dir)?,
        ));
    }
    let path_str = path
        .to_str()
        .ok_or_else(|| anyhow!("non-utf8 path {:?}", path))?;
    Ok(Box::new(WeightMap::from_file(path_str)?))
}

fn open_gguf(path: &Path) -> Result<Box<dyn WeightLoader>> {
    let path_str = path
        .to_str()
        .ok_or_else(|| anyhow!("non-utf8 path {:?}", path))?;
    Ok(Box::new(GgufLoader::from_file(path_str)?))
}

static BUILTIN_FORMATS: &[WeightFormatRegistration] = &[
    WeightFormatRegistration {
        id: "safetensors",
        extensions: &["safetensors"],
        open: open_safetensors,
    },
    WeightFormatRegistration {
        id: "gguf",
        extensions: &["gguf"],
        open: open_gguf,
    },
];

static CUSTOM_FORMATS: Mutex<Vec<WeightFormatRegistration>> = Mutex::new(Vec::new());
static REGISTRY_INIT: OnceLock<()> = OnceLock::new();

/// Register a custom weight format (call before the first load). Later entries override
/// built-ins when the same extension is registered twice.
pub fn register_weight_format(reg: WeightFormatRegistration) {
    CUSTOM_FORMATS
        .lock()
        .expect("weight format registry lock")
        .push(reg);
}

fn formats() -> Vec<WeightFormatRegistration> {
    REGISTRY_INIT.get_or_init(|| ());
    let mut out: Vec<WeightFormatRegistration> = BUILTIN_FORMATS.to_vec();
    let custom = CUSTOM_FORMATS.lock().expect("weight format registry lock");
    out.extend(custom.iter().copied());
    out
}

/// One registered on-disk format (built-in or custom).
#[derive(Debug, Clone, Copy)]
pub struct RegisteredFormat {
    pub id: &'static str,
    pub extensions: &'static [&'static str],
}

/// All registered formats (built-ins first, then custom registrations).
pub fn list_registered_formats() -> Vec<RegisteredFormat> {
    formats()
        .into_iter()
        .map(|r| RegisteredFormat {
            id: r.id,
            extensions: r.extensions,
        })
        .collect()
}

/// Comma-separated extension list for error messages.
pub fn registered_extensions_hint() -> String {
    let mut exts: Vec<&str> = Vec::new();
    for reg in list_registered_formats() {
        for e in reg.extensions {
            if !exts.contains(e) {
                exts.push(e);
            }
        }
    }
    exts.join(", ")
}

/// Extension → format id (last registration wins).
pub fn format_for_extension(ext: &str) -> Option<&'static str> {
    let ext = ext.to_ascii_lowercase();
    let mut found = None;
    for reg in formats() {
        if reg.extensions.iter().any(|e| *e == ext) {
            found = Some(reg.id);
        }
    }
    found
}

/// Open a single file via the format registry.
pub fn open_weight_loader(path: &Path) -> Result<Box<dyn WeightLoader>> {
    // HF model directories (sharded safetensors) have no extension — detect by index.
    if path.is_dir() {
        if path.join("model.safetensors.index.json").is_file()
            || path.join("model.safetensors").is_file()
        {
            return open_safetensors(path)
                .with_context(|| format!("opening {:?} as format safetensors", path));
        }
        // Prefer a lone .gguf in the directory.
        if let Ok(rd) = std::fs::read_dir(path) {
            let ggufs: Vec<PathBuf> = rd
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| {
                    p.extension()
                        .and_then(|s| s.to_str())
                        .is_some_and(|e| e.eq_ignore_ascii_case("gguf"))
                })
                .collect();
            if ggufs.len() == 1 {
                return open_gguf(&ggufs[0])
                    .with_context(|| format!("opening {:?} as format gguf", ggufs[0]));
            }
        }
    }

    let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");
    for reg in formats() {
        if reg.extensions.contains(&ext) {
            return (reg.open)(path)
                .with_context(|| format!("opening {:?} as format {}", path, reg.id));
        }
    }
    let known = registered_extensions_hint();
    Err(anyhow!(
        "unsupported weight extension `.{ext}` for {path:?}\n\
         Registered extensions: .{known}\n\
         Register a custom loader before the first open:\n\
           use rlx_core::weights::WeightFormatRegistration;\n\
           WeightFormatRegistration::new(\"myfmt\", &[\"mybin\"], my_open).register();\n\
         Docs: rlx_core::weights module, README → GGUF → Custom formats"
    ))
}

/// Options for [`load_weights_resolved`] — prefer [`crate::weights::LoadOpts`] presets at call sites.
#[derive(Debug, Clone, Default)]
pub struct LoadWeightsOptions<'a> {
    pub resolve: ResolveWeightsOptions<'a>,
    /// How to drain into a [`WeightMap`] when `into_map` is true.
    pub drain: WeightDrainPolicy,
    /// If true, return a drained [`WeightMap`]; if false, return the live loader.
    pub into_map: bool,
}

impl<'a> LoadWeightsOptions<'a> {
    pub fn map() -> Self {
        Self {
            into_map: true,
            ..Default::default()
        }
    }

    pub fn loader() -> Self {
        Self {
            into_map: false,
            ..Default::default()
        }
    }

    pub fn prefer_q4_k_m(self) -> Self {
        self.prefer_substring("Q4_K_M")
    }

    pub fn prefer_substring(mut self, sub: &'a str) -> Self {
        self.resolve.prefer_gguf_substring = Some(sub);
        self
    }

    pub fn gguf_index(mut self, idx: usize) -> Self {
        self.resolve.gguf_index = Some(idx);
        self
    }

    pub fn drain(mut self, policy: WeightDrainPolicy) -> Self {
        self.drain = policy;
        self
    }

    pub fn warn_unused(self) -> Self {
        self.drain(WeightDrainPolicy::AllF32WarnUnused)
    }
}

/// Result of resolving and opening weights.
pub enum LoadedWeights {
    /// Live loader (supports packed `take`, MTP, mmap borrow).
    Loader {
        path: PathBuf,
        format_id: &'static str,
        loader: Box<dyn WeightLoader>,
    },
    /// Drained map (F32 tensors + optional packed sidecar).
    Map {
        path: PathBuf,
        format_id: &'static str,
        map: WeightMap,
        packed: Vec<crate::weight_map::NamedPackedWeight>,
    },
}

impl LoadedWeights {
    /// Drained map, if this load used `into_map: true`.
    pub fn as_map(&self) -> Option<&WeightMap> {
        match self {
            Self::Map { map, .. } => Some(map),
            Self::Loader { .. } => None,
        }
    }

    /// Live loader, if this load used `into_map: false`.
    pub fn as_loader_mut(&mut self) -> Option<&mut dyn WeightLoader> {
        match self {
            Self::Loader { loader, .. } => Some(loader.as_mut()),
            Self::Map { .. } => None,
        }
    }

    /// Packed K-quant tensors when returned as [`LoadedWeights::Map`].
    pub fn packed_tensors(&self) -> Option<&[crate::weight_map::NamedPackedWeight]> {
        match self {
            Self::Map { packed, .. } => Some(packed.as_slice()),
            Self::Loader { .. } => None,
        }
    }

    pub fn path(&self) -> &Path {
        match self {
            Self::Loader { path, .. } | Self::Map { path, .. } => path,
        }
    }

    pub fn format_id(&self) -> &'static str {
        match self {
            Self::Loader { format_id, .. } | Self::Map { format_id, .. } => format_id,
        }
    }

    pub fn into_map(self) -> Result<WeightMap> {
        match self {
            Self::Map { map, packed, .. } => {
                if !packed.is_empty() {
                    anyhow::bail!(
                        "into_map: {} packed tensors were not merged (use Loader path for packed mode)",
                        packed.len()
                    );
                }
                Ok(map)
            }
            Self::Loader { mut loader, .. } => Ok(WeightMap::from_weight_loader(loader.as_mut())?),
        }
    }
}

/// Resolve a file or directory, enforce GGUF arch policy, open via registry, optionally drain.
pub fn load_weights_resolved(path: &Path, opts: LoadWeightsOptions<'_>) -> Result<LoadedWeights> {
    let file = resolve_weights_file_with_options(path, &opts.resolve)?;
    // Split GGUF merge happens inside GgufLoader::from_file / load_gguf_file.
    let ext = file.extension().and_then(|s| s.to_str()).unwrap_or("");
    if ext == "gguf" && opts.into_map {
        let arch = crate::gguf_support::gguf_architecture_from_path(&file)?;
        if arch == "laguna" && !crate::gguf_support::laguna_allow_f32_expand() {
            anyhow::bail!(
                "F32 expand disabled for Laguna GGUF ({file:?}): \
                 WeightMap drain widens every quant to host F32. \
                 Prefer `rlx-laguna --packed-load`, or opt in with \
                 `RLX_LAGUNA_ALLOW_F32_EXPAND=1` / `--allow-f32-expand` \
                 (see `--weights` sniff for `F32-expand≈`)."
            );
        }
    }
    let format_id = format_for_extension(ext)
        .ok_or_else(|| anyhow!("no registered loader for extension `.{ext}`"))?;
    let mut loader = open_weight_loader(&file)?;
    if opts.into_map {
        let (map, packed) = WeightMap::drain_loader(loader.as_mut(), opts.drain)?;
        Ok(LoadedWeights::Map {
            path: file,
            format_id,
            map,
            packed,
        })
    } else {
        Ok(LoadedWeights::Loader {
            path: file,
            format_id,
            loader,
        })
    }
}

/// Convenience: resolve + drain to F32 [`WeightMap`].
pub fn load_weight_map_resolved(
    path: &Path,
    opts: LoadWeightsOptions<'_>,
) -> Result<(PathBuf, WeightMap)> {
    let mut o = opts;
    o.into_map = true;
    let loaded = load_weights_resolved(path, o)?;
    Ok((loaded.path().to_path_buf(), loaded.into_map()?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_extension_errors() {
        let path = std::env::temp_dir().join("rlx_weight_registry_test.noext");
        match open_weight_loader(&path) {
            Err(e) => assert!(
                e.to_string().contains("unsupported weight extension"),
                "{e}"
            ),
            Ok(_) => panic!("expected unsupported extension for {path:?}"),
        }
    }

    #[test]
    fn format_for_extension_builtin() {
        assert_eq!(format_for_extension("gguf"), Some("gguf"));
        assert_eq!(format_for_extension("safetensors"), Some("safetensors"));
        assert_eq!(format_for_extension("bin"), None);
    }

    #[test]
    fn list_formats_includes_builtins() {
        let ids: Vec<_> = list_registered_formats().iter().map(|r| r.id).collect();
        assert!(ids.contains(&"gguf"));
        assert!(ids.contains(&"safetensors"));
    }
}
