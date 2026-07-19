// RLX models — model resolution & HuggingFace download.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! High-level model resolution.
//!
//! [`Resolver`] turns a user-supplied string (a local path, or a
//! [`ModelRef`]) into a [`ResolvedModel`] — a concrete set of local files on
//! disk plus the detected [`ModelFormat`]. Local paths resolve directly;
//! everything else is downloaded from HuggingFace into a configurable cache
//! directory.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use hf_hub::api::sync::{Api, ApiBuilder};
use hf_hub::{Repo, RepoType};

use crate::model_ref::ModelRef;
use crate::quant::{gguf_shard_group, is_gguf_file, select_gguf};

/// The on-disk format of a resolved model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFormat {
    /// A GGUF (llama.cpp) file, possibly split across shards.
    Gguf,
    /// A HuggingFace `safetensors` model directory (weights + `config.json`).
    Safetensors,
}

/// A fully resolved model: the concrete local files, their format, and the
/// reference they were resolved from.
#[derive(Debug, Clone)]
pub struct ResolvedModel {
    /// Local file paths on disk. For GGUF this is the shard group (entry point
    /// first); for safetensors it is the weight shards plus `config.json` and,
    /// when present, `tokenizer.json`.
    pub files: Vec<PathBuf>,
    /// The detected format.
    pub format: ModelFormat,
    /// The reference this model was resolved from. `None` for bare local
    /// paths that were not expressed as a `ModelRef`.
    pub ref_: Option<ModelRef>,
}

impl ResolvedModel {
    /// The primary weight file: the first GGUF shard, or the first safetensors
    /// shard. Non-weight sidecar files (`config.json`, `tokenizer.json`) sort
    /// after weights, so `files[0]` is always a weight file when present.
    pub fn primary_file(&self) -> Option<&Path> {
        self.files.first().map(PathBuf::as_path)
    }
}

/// Resolves model references to local files, downloading from HuggingFace when
/// necessary.
#[derive(Clone)]
pub struct Resolver {
    cache_dir: PathBuf,
    token: Option<String>,
}

impl Default for Resolver {
    fn default() -> Self {
        Self::new()
    }
}

impl Resolver {
    /// Create a resolver using the default HuggingFace cache directory
    /// (respecting `HF_HOME`, else `~/.cache/huggingface`).
    pub fn new() -> Self {
        Self {
            cache_dir: default_cache_dir(),
            token: hf_token_from_env(),
        }
    }

    /// Override the cache directory.
    pub fn with_cache_dir(mut self, cache_dir: impl Into<PathBuf>) -> Self {
        self.cache_dir = cache_dir.into();
        self
    }

    /// Override the HuggingFace access token (for gated/private repos).
    pub fn with_token(mut self, token: impl Into<String>) -> Self {
        self.token = Some(token.into());
        self
    }

    /// The cache directory in use.
    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }

    /// Resolve an input string. If it names an existing local path, resolve it
    /// locally; otherwise parse it as a [`ModelRef`] and download from HF.
    pub fn resolve(&self, input: &str) -> Result<ResolvedModel> {
        let path = Path::new(input);
        if path.exists() {
            return resolve_local(path, None);
        }

        let model_ref = ModelRef::parse(input)
            .map_err(|err| anyhow!("{err}"))
            .with_context(|| format!("resolve model input {input:?}"))?;
        self.resolve_ref(&model_ref)
    }

    /// Resolve a parsed [`ModelRef`] by downloading from HuggingFace.
    pub fn resolve_ref(&self, model_ref: &ModelRef) -> Result<ResolvedModel> {
        // A repo whose `repo` field is itself a local directory (rare, but
        // supported for symmetry) resolves locally.
        let as_path = Path::new(&model_ref.repo);
        if as_path.exists() {
            return resolve_local(as_path, Some(model_ref.clone()));
        }

        let api = self.build_api()?;
        let revision = model_ref.revision_or_main();
        let repo = api.repo(Repo::with_revision(
            model_ref.repo.clone(),
            RepoType::Model,
            revision.to_string(),
        ));

        let info = repo
            .info()
            .map_err(|err| anyhow!(err.to_string()))
            .with_context(|| format!("list HuggingFace repo {}@{revision}", model_ref.repo))?;
        let siblings: Vec<String> = info.siblings.into_iter().map(|s| s.rfilename).collect();

        // Prefer GGUF when the repo ships GGUF files or a quant selector was
        // requested; otherwise treat it as a safetensors repo.
        let wants_gguf =
            model_ref.selector.is_some() || siblings.iter().any(|file| is_gguf_file(file));

        if wants_gguf {
            self.download_gguf(&repo, model_ref, &siblings)
        } else {
            self.download_safetensors(&repo, model_ref, &siblings)
        }
    }

    fn build_api(&self) -> Result<Api> {
        ApiBuilder::new()
            .with_cache_dir(self.cache_dir.clone())
            .with_token(self.token.clone())
            .build()
            .map_err(|err| anyhow!(err.to_string()))
            .context("build HuggingFace API client")
    }

    fn download_gguf(
        &self,
        repo: &hf_hub::api::sync::ApiRepo,
        model_ref: &ModelRef,
        siblings: &[String],
    ) -> Result<ResolvedModel> {
        let selected =
            select_gguf(siblings, model_ref.selector.as_deref()).ok_or_else(|| match model_ref
                .selector
                .as_deref()
            {
                Some(selector) => anyhow!(
                    "no GGUF file matching selector {selector:?} in {}",
                    model_ref.repo
                ),
                None => anyhow!("no GGUF file found in {}", model_ref.repo),
            })?;

        let group = gguf_shard_group(siblings, selected);
        let mut files = Vec::with_capacity(group.len());
        for file in group {
            files.push(get_file(repo, file, model_ref)?);
        }

        Ok(ResolvedModel {
            files,
            format: ModelFormat::Gguf,
            ref_: Some(model_ref.clone()),
        })
    }

    fn download_safetensors(
        &self,
        repo: &hf_hub::api::sync::ApiRepo,
        model_ref: &ModelRef,
        siblings: &[String],
    ) -> Result<ResolvedModel> {
        let weights: Vec<&String> = siblings
            .iter()
            .filter(|file| file.to_ascii_lowercase().ends_with(".safetensors"))
            .collect();
        if weights.is_empty() {
            bail!("no safetensors or GGUF weights found in {}", model_ref.repo);
        }

        let mut files = Vec::new();
        for file in &weights {
            files.push(get_file(repo, file.as_str(), model_ref)?);
        }
        // Required + optional sidecars, in a stable order after the weights.
        for sidecar in ["config.json", "tokenizer.json"] {
            if siblings.iter().any(|file| file.as_str() == sidecar) {
                files.push(get_file(repo, sidecar, model_ref)?);
            }
        }

        Ok(ResolvedModel {
            files,
            format: ModelFormat::Safetensors,
            ref_: Some(model_ref.clone()),
        })
    }
}

fn get_file(
    repo: &hf_hub::api::sync::ApiRepo,
    file: &str,
    model_ref: &ModelRef,
) -> Result<PathBuf> {
    repo.get(file)
        .map_err(|err| anyhow!(err.to_string()))
        .with_context(|| {
            format!(
                "download {file} from HuggingFace repo {}@{}",
                model_ref.repo,
                model_ref.revision_or_main()
            )
        })
}

/// Resolve a local file or directory into a [`ResolvedModel`].
///
/// - A `.gguf` file resolves to its shard group.
/// - A directory is scanned for GGUF files (picked by `ref_`'s selector) or a
///   safetensors model (weights + `config.json`).
fn resolve_local(path: &Path, ref_: Option<ModelRef>) -> Result<ResolvedModel> {
    if path.is_file() {
        return resolve_local_file(path, ref_);
    }
    if path.is_dir() {
        return resolve_local_dir(path, ref_);
    }
    bail!(
        "local model path is neither a file nor a directory: {}",
        path.display()
    )
}

fn resolve_local_file(path: &Path, ref_: Option<ModelRef>) -> Result<ResolvedModel> {
    let name = file_name_str(path)?;
    if is_gguf_file(name) {
        let parent = path.parent().unwrap_or_else(|| Path::new("."));
        let files = local_gguf_shard_group(parent, name)?;
        return Ok(ResolvedModel {
            files,
            format: ModelFormat::Gguf,
            ref_,
        });
    }
    if name.to_ascii_lowercase().ends_with(".safetensors") {
        // A lone safetensors file: pair it with a sibling config.json if any.
        let mut files = vec![path.to_path_buf()];
        if let Some(parent) = path.parent() {
            let config = parent.join("config.json");
            if config.is_file() {
                files.push(config);
            }
        }
        return Ok(ResolvedModel {
            files,
            format: ModelFormat::Safetensors,
            ref_,
        });
    }
    bail!(
        "unrecognized local model file (expected .gguf or .safetensors): {}",
        path.display()
    )
}

fn resolve_local_dir(dir: &Path, ref_: Option<ModelRef>) -> Result<ResolvedModel> {
    let entries: Vec<String> = std::fs::read_dir(dir)
        .with_context(|| format!("read model directory {}", dir.display()))?
        .filter_map(|entry| entry.ok())
        .filter_map(|entry| entry.file_name().into_string().ok())
        .collect();

    let selector = ref_.as_ref().and_then(|r| r.selector.as_deref());
    let has_gguf = entries.iter().any(|file| is_gguf_file(file));

    if has_gguf {
        let selected = select_gguf(&entries, selector)
            .ok_or_else(|| anyhow!("no matching GGUF file in directory {}", dir.display()))?;
        let files = local_gguf_shard_group(dir, selected)?;
        return Ok(ResolvedModel {
            files,
            format: ModelFormat::Gguf,
            ref_,
        });
    }

    let mut weights: Vec<PathBuf> = entries
        .iter()
        .filter(|file| file.to_ascii_lowercase().ends_with(".safetensors"))
        .map(|file| dir.join(file))
        .collect();
    if weights.is_empty() {
        bail!(
            "no GGUF or safetensors weights found in directory {}",
            dir.display()
        );
    }
    weights.sort();

    let mut files = weights;
    for sidecar in ["config.json", "tokenizer.json"] {
        let candidate = dir.join(sidecar);
        if candidate.is_file() {
            files.push(candidate);
        }
    }
    if !files
        .iter()
        .any(|p| p.file_name().is_some_and(|n| n == "config.json"))
    {
        bail!(
            "safetensors model directory {} is missing config.json",
            dir.display()
        );
    }

    Ok(ResolvedModel {
        files,
        format: ModelFormat::Safetensors,
        ref_,
    })
}

/// Gather the on-disk shard group for a GGUF file living in `dir`.
fn local_gguf_shard_group(dir: &Path, primary: &str) -> Result<Vec<PathBuf>> {
    let entries: Vec<String> = std::fs::read_dir(dir)
        .with_context(|| format!("read directory {}", dir.display()))?
        .filter_map(|entry| entry.ok())
        .filter_map(|entry| entry.file_name().into_string().ok())
        .collect();
    let mut group: Vec<PathBuf> = gguf_shard_group(&entries, primary)
        .into_iter()
        .map(|file| dir.join(file))
        .collect();
    if group.is_empty() {
        group.push(dir.join(primary));
    }
    // Keep the entry-point (00001 / non-split) first, remaining shards sorted.
    group.sort_by(|a, b| {
        let ap = a
            .file_name()
            .and_then(|n| n.to_str())
            .map(is_first_shard)
            .unwrap_or(false);
        let bp = b
            .file_name()
            .and_then(|n| n.to_str())
            .map(is_first_shard)
            .unwrap_or(false);
        bp.cmp(&ap).then_with(|| a.cmp(b))
    });
    Ok(group)
}

fn is_first_shard(file: &str) -> bool {
    match crate::model_ref::split_gguf_shard_info(file) {
        Some(shard) => shard.part == "00001",
        None => true,
    }
}

fn file_name_str(path: &Path) -> Result<&str> {
    path.file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| anyhow!("path has no file name: {}", path.display()))
}

/// The default HuggingFace hub cache directory.
///
/// Precedence: `HF_HUB_CACHE` > `HUGGINGFACE_HUB_CACHE` > `HF_HOME/hub` >
/// `~/.cache/huggingface/hub`.
pub fn default_cache_dir() -> PathBuf {
    if let Some(path) = env_path("HF_HUB_CACHE") {
        return path;
    }
    if let Some(path) = env_path("HUGGINGFACE_HUB_CACHE") {
        return path;
    }
    if let Some(path) = env_path("HF_HOME") {
        return path.join("hub");
    }
    if let Some(path) = env_path("XDG_CACHE_HOME") {
        return path.join("huggingface").join("hub");
    }
    // Default: `~/.cache/huggingface/hub`, matching the HuggingFace hub layout.
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".cache")
        .join("huggingface")
        .join("hub")
}

fn hf_token_from_env() -> Option<String> {
    for key in ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"] {
        if let Ok(token) = std::env::var(key) {
            let token = token.trim();
            if !token.is_empty() {
                return Some(token.to_string());
            }
        }
    }
    None
}

fn env_path(key: &str) -> Option<PathBuf> {
    let value = std::env::var(key).ok()?;
    let trimmed = value.trim();
    (!trimmed.is_empty()).then(|| PathBuf::from(trimmed))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_cache_dir_respects_hf_home() {
        // Snapshot & clear the higher-precedence vars for a deterministic check.
        // SAFETY: single-threaded test; we restore below.
        let prev_hub = std::env::var("HF_HUB_CACHE").ok();
        let prev_hf_home = std::env::var("HF_HOME").ok();
        unsafe {
            std::env::remove_var("HF_HUB_CACHE");
            std::env::remove_var("HUGGINGFACE_HUB_CACHE");
            std::env::set_var("HF_HOME", "/tmp/hf-home-test");
        }

        assert_eq!(default_cache_dir(), PathBuf::from("/tmp/hf-home-test/hub"));

        unsafe {
            std::env::remove_var("HF_HOME");
            if let Some(v) = prev_hub {
                std::env::set_var("HF_HUB_CACHE", v);
            }
            if let Some(v) = prev_hf_home {
                std::env::set_var("HF_HOME", v);
            }
        }
    }

    #[test]
    fn resolved_model_primary_file() {
        let model = ResolvedModel {
            files: vec![PathBuf::from("/m/model.gguf")],
            format: ModelFormat::Gguf,
            ref_: None,
        };
        assert_eq!(model.primary_file(), Some(Path::new("/m/model.gguf")));
    }
}
