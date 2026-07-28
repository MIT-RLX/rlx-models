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

//! Model-agnostic weight I/O — paths, formats, drain policy only.
//!
//! Architecture checks and tensor renaming live in **model crates** ([`gguf_validate_arch`],
//! [`register_gguf_tensor_resolver`]), not here.
//!
//! ```no_run
//! use rlx_models_core::weights::{self, LoadOpts};
//!
//! let (path, map) = weights::open_map_with(LoadOpts::map().prefer_q4_k_m(), "weights/")?;
//! let loaded = weights::open_with(LoadOpts::loader(), "model.gguf")?;
//! # Ok::<(), anyhow::Error>(())
//! ```

use std::path::{Path, PathBuf};

use anyhow::Result;

pub use crate::gguf_resolve::{
    GgufTensorNameResolver, LlamaFamilyGgufResolver, PassThroughGgufResolver,
    PrefixStripGgufResolver, Qwen35NativeGgufResolver, register_gguf_tensor_resolver,
};
pub use crate::gguf_support::{
    ResolveWeightsOptions, gguf_split_siblings, gguf_validate_arch, list_gguf_files_in_dir,
    load_gguf_file, resolve_weights_file, resolve_weights_file_with_options,
};
pub use crate::weight_loader::{GgufLoader, WeightLoader};
pub use crate::weight_map::{WeightDrainPolicy, WeightMap};
pub use crate::weight_registry::{
    LoadWeightsOptions, LoadedWeights, RegisteredFormat, WeightFormatRegistration,
    format_for_extension, list_registered_formats, load_weight_map_resolved, load_weights_resolved,
    open_weight_loader, register_weight_format,
};

/// Alias for [`LoadWeightsOptions`].
pub type LoadOpts<'a> = LoadWeightsOptions<'a>;

/// Alias for [`ResolveWeightsOptions`].
pub type ResolveOpts<'a> = ResolveWeightsOptions<'a>;

/// Default directory resolve: prefer [`DEFAULT_GGUF_PREFER_SUBSTR`](crate::gguf_support::DEFAULT_GGUF_PREFER_SUBSTR).
pub fn default_resolve_opts<'a>() -> ResolveWeightsOptions<'a> {
    ResolveWeightsOptions::default()
        .prefer_substring(crate::gguf_support::DEFAULT_GGUF_PREFER_SUBSTR)
}

/// Resolve a file or weights directory to one on-disk path.
pub fn pick(path: impl AsRef<Path>, resolve: &ResolveWeightsOptions<'_>) -> Result<PathBuf> {
    resolve_weights_file_with_options(path.as_ref(), resolve)
}

/// [`pick`] with [`default_resolve_opts`].
pub fn pick_default(path: impl AsRef<Path>) -> Result<PathBuf> {
    pick(path, &default_resolve_opts())
}

/// Resolve a weights path, validate GGUF `general.architecture` when applicable, drain to F32.
///
/// Call from model runners with that family's arch list (e.g. [`crate::DINOV2_GGUF_ARCHES`]).
pub fn load_weight_map(path: impl AsRef<Path>, gguf_arches: &[&str]) -> Result<WeightMap> {
    let path = path.as_ref();
    let file = pick_default(path)?;
    if file.extension().and_then(|s| s.to_str()) == Some("gguf") {
        gguf_validate_arch(&file, gguf_arches)?;
    }
    WeightMap::from_resolved_path(path)
}

/// Idempotent: ensure built-in GGUF tensor resolvers and model-family
/// registry entries are registered (safe to call from `main`).
pub fn init() {
    crate::gguf_resolve::ensure_builtin_resolvers();
    crate::model_registry::ensure_builtin_gguf_models();
}

impl WeightFormatRegistration {
    /// Describe a custom on-disk format.
    pub const fn new(
        id: &'static str,
        extensions: &'static [&'static str],
        open: crate::weight_registry::WeightLoaderFactory,
    ) -> Self {
        Self {
            id,
            extensions,
            open,
        }
    }

    /// [`register_weight_format`] shorthand.
    pub fn register(self) {
        register_weight_format(self);
    }
}

/// Resolve + open (live [`WeightLoader`]).
pub fn open(path: impl AsRef<Path>) -> Result<LoadedWeights> {
    open_with(LoadOpts::loader(), path)
}

/// Resolve + open with options.
pub fn open_with(opts: LoadOpts<'_>, path: impl AsRef<Path>) -> Result<LoadedWeights> {
    load_weights_resolved(path.as_ref(), opts)
}

/// Resolve + drain to F32 [`WeightMap`].
pub fn open_map(path: impl AsRef<Path>) -> Result<(PathBuf, WeightMap)> {
    open_map_with(LoadOpts::map(), path)
}

/// Resolve + drain with options.
pub fn open_map_with(opts: LoadOpts<'_>, path: impl AsRef<Path>) -> Result<(PathBuf, WeightMap)> {
    load_weight_map_resolved(path.as_ref(), opts)
}

/// Numbered `.gguf` listing + resolve hints for a directory (CLI / errors).
pub fn gguf_dir_guide(dir: &Path) -> Result<GgufDirGuide> {
    let files = list_gguf_files_in_dir(dir)?;
    let mut lines = Vec::new();
    if files.is_empty() {
        lines.push(format!("No .gguf files in {dir:?}"));
        return Ok(GgufDirGuide { files, lines });
    }
    lines.push(format!("{} .gguf file(s) in {dir:?}:", files.len()));
    for (i, p) in files.iter().enumerate() {
        let name = p.file_name().and_then(|s| s.to_str()).unwrap_or("?");
        lines.push(format!("  [{i}] {name}"));
    }
    if files.len() > 1 {
        lines.push(String::new());
        lines.push("Pick one:".into());
        lines.push(format!("  pass exact path: {:?}", files[0]));
        lines.push("  Rust: LoadOpts::map().prefer_q4_k_m()".into());
        lines.push("  Rust: LoadOpts::map().gguf_index(0)".into());
        lines.push("  CLI:  rlx-inspect dir/ --prefer Q4_K_M".into());
    }
    Ok(GgufDirGuide { files, lines })
}

#[derive(Debug, Clone)]
pub struct GgufDirGuide {
    pub files: Vec<PathBuf>,
    pub lines: Vec<String>,
}

impl GgufDirGuide {
    pub fn print(&self) {
        for line in &self.lines {
            println!("{line}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn load_opts_format_only() {
        assert!(LoadOpts::map().into_map);
        assert!(!LoadOpts::loader().into_map);
    }
}
