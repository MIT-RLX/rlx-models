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

use crate::format::WeightFormat;
use anyhow::{Result, anyhow};
use rlx_core::gguf_support::{ResolveWeightsOptions, resolve_weights_file_with_options};
use rlx_core::weight_loader::{GgufLoader, WeightLoader, hf_to_gguf_name};
use rlx_core::weight_map::WeightMap;
use std::path::{Path, PathBuf};

pub use rlx_core::gguf_resolve::{GgufTensorNameResolver, register_gguf_tensor_resolver};
pub use rlx_core::weight_registry::{
    RegisteredFormat, list_registered_formats, load_weight_map_resolved, load_weights_resolved,
    open_weight_loader,
};
pub use rlx_core::weights::{
    self, GgufDirGuide, LoadOpts, ResolveOpts, gguf_dir_guide, open as open_weights, open_map,
    open_map_with, open_with,
};
pub use rlx_core::{
    LoadWeightsOptions, LoadedWeights, WeightFormatRegistration, register_weight_format,
};

pub fn open_loader(path: &Path) -> Result<Box<dyn WeightLoader>> {
    open_loader_with_format(path, None)
}

/// Resolve a file or weights directory, then open the right loader.
pub fn open_loader_resolved(path: &Path) -> Result<(PathBuf, Box<dyn WeightLoader>)> {
    open_loader_resolved_with_options(path, &ResolveWeightsOptions::default())
}

pub fn open_loader_resolved_with_options(
    path: &Path,
    resolve: &ResolveWeightsOptions<'_>,
) -> Result<(PathBuf, Box<dyn WeightLoader>)> {
    let file = resolve_weights_file_with_options(path, resolve)?;
    let loader = open_loader_with_format(&file, None)?;
    Ok((file, loader))
}

pub fn open_loader_with_format(
    path: &Path,
    format: Option<WeightFormat>,
) -> Result<Box<dyn WeightLoader>> {
    let fmt = WeightFormat::resolve(path, format)?;
    let path_str = path.to_str().ok_or_else(|| anyhow!("non-utf8 path"))?;
    match fmt {
        WeightFormat::Safetensors => Ok(Box::new(WeightMap::from_file(path_str)?)),
        WeightFormat::Gguf => Ok(Box::new(GgufLoader::from_file(path_str)?)),
    }
}

/// GGUF loader with optional MTP-head visibility (LM families).
pub fn open_gguf_loader(path: &Path, include_mtp: bool) -> Result<GgufLoader> {
    let path_str = path.to_str().ok_or_else(|| anyhow!("non-utf8 path"))?;
    let mut loader = GgufLoader::from_file(path_str)?;
    loader.include_mtp(include_mtp);
    Ok(loader)
}

/// Resolved path → drained [`WeightMap`] (safetensors or GGUF).
pub fn open_weight_map_resolved(path: &Path) -> Result<(PathBuf, WeightMap)> {
    open_map_with(LoadOpts::map().warn_unused(), path)
}

/// Full resolved load (loader or map, per options).
pub fn open_weights_resolved(path: &Path, opts: LoadWeightsOptions<'_>) -> Result<LoadedWeights> {
    load_weights_resolved(path, opts)
}

pub fn debug_resolve_name(hf: &str) -> Option<String> {
    hf_to_gguf_name(hf)
}
