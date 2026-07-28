//! Scan local LLM app caches and resolve short weight queries to concrete paths.
//!
//! # Overview
//!
//! Many tools download the same GGUF / safetensors checkpoints into different
//! directories. This module walks those caches (no network) and either **lists**
//! them ([`scan_weights`] / [`scan_weights_in_roots`]) or **picks one path**
//! ([`resolve_weight_query`] / [`resolve_weights_path_or_query`]).
//!
//! | Source | Typical roots (override with env) |
//! |--------|-----------------------------------|
//! | LM Studio | `~/.lmstudio/models` (`LMS_MODELS` / `LM_STUDIO_MODELS`) |
//! | Ollama | `~/.ollama/models` (`OLLAMA_MODELS`); Linux service path |
//! | Hugging Face | `HF_HUB_CACHE` / `$HF_HOME/hub` / `~/.cache/huggingface/hub` |
//! | MLX | HF hub repos under `mlx-community`; optional `MLX_CACHE` |
//! | vLLM | Same HF hub; optional `VLLM_CACHE_ROOT` |
//! | Lemonade | `LEMONADE_CACHE_DIR` / `~/.cache/lemonade` (+ platform installs) |
//! | RLX local | `weights/`, `.cache/`, temp `rlx-weights`, `RLX_WEIGHTS_DIR` |
//! | Extra | `RLX_WEIGHTS_PATHS` (`;` on Windows, `:` elsewhere) |
//!
//! Windows uses `%USERPROFILE%`, `%LOCALAPPDATA%`, `%TEMP%` / `%TMP%` where
//! appropriate; path lists accept `;` separators.
//!
//! # Library usage
//!
//! ```no_run
//! use rlx_models_core::weights_discover::{DiscoverOpts, scan_weights, resolve_weight_query};
//!
//! let hits = scan_weights(&DiscoverOpts::default().with_query("qwen3"))?;
//! let path = resolve_weight_query("qwen3-0.6b", &DiscoverOpts::default())?;
//! # Ok::<(), anyhow::Error>(())
//! ```
//!
//! Downstream crates that depend on workspace `rlx-core` import the same API as
//! `rlx_core::weights_discover` (also re-exported from `rlx_models` and `rlx_cli`).
//!
//! Custom roots (tests, embedded apps) use [`scan_weights_in_roots`] and
//! [`resolve_weight_query_in_roots`] so host caches are not touched.
//!
//! CLI: `rlx-inspect scan` / `rlx-inspect resolve` / `just weights-scan`.

use crate::gguf_support::{
    DEFAULT_GGUF_PREFER_SUBSTR, ResolveWeightsOptions, gguf_architecture_from_path,
    resolve_weights_file_with_options,
};
use anyhow::{Context, Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};

/// Where a discovered weight file was found.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WeightSourceKind {
    LmStudio,
    Ollama,
    HuggingFace,
    Mlx,
    Vllm,
    Lemonade,
    RlxLocal,
    Extra,
}

impl WeightSourceKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::LmStudio => "lmstudio",
            Self::Ollama => "ollama",
            Self::HuggingFace => "hf",
            Self::Mlx => "mlx",
            Self::Vllm => "vllm",
            Self::Lemonade => "lemonade",
            Self::RlxLocal => "rlx",
            Self::Extra => "extra",
        }
    }

    /// Parse a CLI `--source` token (`lmstudio`, `hf`, `huggingface`, …).
    pub fn parse(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "lmstudio" | "lms" | "lm-studio" => Ok(Self::LmStudio),
            "ollama" => Ok(Self::Ollama),
            "hf" | "huggingface" | "hub" => Ok(Self::HuggingFace),
            "mlx" => Ok(Self::Mlx),
            "vllm" => Ok(Self::Vllm),
            "lemonade" => Ok(Self::Lemonade),
            "rlx" | "local" => Ok(Self::RlxLocal),
            "extra" => Ok(Self::Extra),
            other => bail!("unknown weight source `{other}`"),
        }
    }

    /// Prefer earlier sources when ranking ambiguous query matches.
    pub fn resolve_priority(self) -> u8 {
        match self {
            Self::RlxLocal => 0,
            Self::LmStudio => 1,
            Self::Lemonade => 2,
            Self::Ollama => 3,
            Self::Extra => 4,
            Self::Mlx => 5,
            Self::Vllm => 6,
            Self::HuggingFace => 7,
        }
    }

    /// Every source kind (stable order for CLI help / filters).
    pub fn all() -> &'static [Self] {
        &[
            Self::LmStudio,
            Self::Ollama,
            Self::HuggingFace,
            Self::Mlx,
            Self::Vllm,
            Self::Lemonade,
            Self::RlxLocal,
            Self::Extra,
        ]
    }
}

impl std::fmt::Display for WeightSourceKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Detected on-disk weight format (cheap extension / magic check).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DiscoveredFormat {
    Gguf,
    Safetensors,
    Unknown,
}

impl DiscoveredFormat {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Gguf => "gguf",
            Self::Safetensors => "safetensors",
            Self::Unknown => "unknown",
        }
    }
}

/// One runnable weight file (or safetensors model directory) found locally.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveredWeight {
    pub path: PathBuf,
    pub sources: Vec<WeightSourceKind>,
    pub display_name: String,
    pub format: DiscoveredFormat,
    pub size_bytes: Option<u64>,
    pub quant_hint: Option<String>,
    pub arch_hint: Option<String>,
}

impl DiscoveredWeight {
    /// Best (lowest) resolve priority among attached sources.
    pub fn best_priority(&self) -> u8 {
        self.sources
            .iter()
            .map(|s| s.resolve_priority())
            .min()
            .unwrap_or(u8::MAX)
    }
}

/// Options for [`scan_weights`] / [`resolve_weight_query`].
#[derive(Debug, Clone)]
pub struct DiscoverOpts {
    /// Restrict to these sources; `None` = all present roots.
    pub sources: Option<Vec<WeightSourceKind>>,
    /// Case-insensitive substring filter on display name + path.
    pub query: Option<String>,
    /// Prefer GGUF filenames containing this substring (default `Q4_K_M` for resolve).
    pub prefer_quant: Option<String>,
    /// Open GGUF headers for `general.architecture` (slower).
    pub sniff_arch: bool,
    /// Max directory depth when walking recursive roots.
    pub max_depth: usize,
    /// Extra roots to scan as [`WeightSourceKind::Extra`] (in addition to env).
    pub extra_roots: Vec<PathBuf>,
}

impl Default for DiscoverOpts {
    fn default() -> Self {
        Self {
            sources: None,
            query: None,
            prefer_quant: None,
            sniff_arch: false,
            max_depth: 8,
            extra_roots: Vec::new(),
        }
    }
}

impl DiscoverOpts {
    pub fn with_query(mut self, q: impl Into<String>) -> Self {
        self.query = Some(q.into());
        self
    }

    pub fn with_prefer_quant(mut self, q: impl Into<String>) -> Self {
        self.prefer_quant = Some(q.into());
        self
    }

    pub fn with_sniff_arch(mut self, yes: bool) -> Self {
        self.sniff_arch = yes;
        self
    }

    pub fn with_sources(mut self, sources: impl Into<Vec<WeightSourceKind>>) -> Self {
        self.sources = Some(sources.into());
        self
    }

    pub fn with_max_depth(mut self, depth: usize) -> Self {
        self.max_depth = depth;
        self
    }

    pub fn with_extra_roots(mut self, roots: impl Into<Vec<PathBuf>>) -> Self {
        self.extra_roots = roots.into();
        self
    }

    fn allows(&self, kind: WeightSourceKind) -> bool {
        match &self.sources {
            None => true,
            Some(list) => list.contains(&kind),
        }
    }
}

/// Existing roots for each known source (missing dirs omitted).
pub fn default_source_roots() -> Vec<(WeightSourceKind, PathBuf)> {
    let mut out = Vec::new();
    for (kind, path) in candidate_source_roots() {
        if path.is_dir() {
            out.push((kind, path));
        }
    }
    out
}

/// True when `s` looks like a filesystem path (not a short discovery query).
///
/// Recognizes `/`, `\`, Windows drive letters (`C:…`), and relative prefixes
/// (`./`, `.\`, `../`, `..\`). Bare names like `qwen3-0.6b` or `model.gguf`
/// are treated as queries.
pub fn looks_like_filesystem_path(s: &str) -> bool {
    let s = s.trim();
    if s.is_empty() {
        return false;
    }
    if s.contains('/') || s.contains('\\') {
        return true;
    }
    let b = s.as_bytes();
    if b.len() >= 2 && b[0].is_ascii_alphabetic() && b[1] == b':' {
        return true;
    }
    matches!(s, "." | "..")
        || s.starts_with("./")
        || s.starts_with(".\\")
        || s.starts_with("../")
        || s.starts_with("..\\")
}

fn candidate_source_roots() -> Vec<(WeightSourceKind, PathBuf)> {
    let mut out = Vec::new();

    // LM Studio
    if let Some(p) = env_path(&["LMS_MODELS", "LM_STUDIO_MODELS"]) {
        out.push((WeightSourceKind::LmStudio, p));
    } else {
        for p in lmstudio_model_candidates() {
            out.push((WeightSourceKind::LmStudio, p));
        }
    }

    // Ollama
    if let Some(p) = env_path(&["OLLAMA_MODELS"]) {
        out.push((WeightSourceKind::Ollama, p));
    } else {
        for p in ollama_model_candidates() {
            out.push((WeightSourceKind::Ollama, p));
        }
    }

    // Hugging Face hub (also used by MLX / vLLM)
    for hub in hf_hub_cache_candidates() {
        out.push((WeightSourceKind::HuggingFace, hub.clone()));
        out.push((WeightSourceKind::Mlx, hub.clone()));
        out.push((WeightSourceKind::Vllm, hub));
    }

    if let Some(mlx) = env_path(&["MLX_CACHE"]) {
        out.push((WeightSourceKind::Mlx, mlx));
    }

    if let Some(vllm) = env_path(&["VLLM_CACHE_ROOT"]) {
        out.push((WeightSourceKind::Vllm, vllm));
    }

    for p in lemonade_cache_candidates() {
        out.push((WeightSourceKind::Lemonade, p));
    }

    if let Some(p) = env_path(&["RLX_WEIGHTS_DIR"]) {
        out.push((WeightSourceKind::RlxLocal, p));
    }
    out.push((WeightSourceKind::RlxLocal, PathBuf::from("weights")));
    out.push((WeightSourceKind::RlxLocal, PathBuf::from(".cache")));
    out.push((WeightSourceKind::RlxLocal, rlx_temp_weights_dir()));

    for p in split_path_list(std::env::var_os("RLX_WEIGHTS_PATHS")) {
        out.push((WeightSourceKind::Extra, p));
    }

    out
}

fn home_dir() -> Option<PathBuf> {
    std::env::var_os("USERPROFILE")
        .or_else(|| std::env::var_os("HOME"))
        .map(PathBuf::from)
}

fn local_app_data() -> Option<PathBuf> {
    std::env::var_os("LOCALAPPDATA").map(PathBuf::from)
}

fn env_path(keys: &[&str]) -> Option<PathBuf> {
    for k in keys {
        if let Ok(v) = std::env::var(k) {
            let t = v.trim();
            if !t.is_empty() {
                return Some(PathBuf::from(t));
            }
        }
    }
    None
}

fn split_path_list(raw: Option<std::ffi::OsString>) -> Vec<PathBuf> {
    let Some(raw) = raw else {
        return Vec::new();
    };
    let s = raw.to_string_lossy();
    // Prefer `;` when present (Windows / mixed). On Windows without `;`, treat
    // the whole value as one path so `C:\Models` is not split on `:`.
    // Elsewhere without `;`, split on `:` (Unix PATH style).
    let parts: Vec<&str> = if s.contains(';') {
        s.split(';').collect()
    } else if cfg!(windows) {
        vec![s.as_ref()]
    } else {
        s.split(':').collect()
    };
    parts
        .into_iter()
        .map(str::trim)
        .filter(|p| !p.is_empty())
        .map(PathBuf::from)
        .collect()
}

fn rlx_temp_weights_dir() -> PathBuf {
    let base = std::env::var_os("TEMP")
        .or_else(|| std::env::var_os("TMP"))
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            if cfg!(windows) {
                PathBuf::from(r"C:\Temp")
            } else {
                PathBuf::from("/tmp")
            }
        });
    base.join("rlx-weights")
}

fn lmstudio_model_candidates() -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Some(home) = home_dir() {
        out.push(home.join(".lmstudio").join("models"));
    }
    if let Some(local) = local_app_data() {
        out.push(local.join("LM Studio").join("models"));
        out.push(local.join(".lmstudio").join("models"));
    }
    out
}

fn ollama_model_candidates() -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Some(home) = home_dir() {
        out.push(home.join(".ollama").join("models"));
    }
    if let Some(local) = local_app_data() {
        out.push(local.join("Ollama").join("models"));
        out.push(local.join(".ollama").join("models"));
    }
    #[cfg(target_os = "linux")]
    {
        out.push(PathBuf::from("/usr/share/ollama/.ollama/models"));
    }
    out
}

fn hf_hub_cache_candidates() -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Some(p) = env_path(&["HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"]) {
        out.push(p);
        return out;
    }
    if let Some(home) = env_path(&["HF_HOME"]) {
        out.push(home.join("hub"));
        return out;
    }
    if let Some(xdg) = env_path(&["XDG_CACHE_HOME"]) {
        out.push(xdg.join("huggingface").join("hub"));
    }
    if let Some(home) = home_dir() {
        out.push(home.join(".cache").join("huggingface").join("hub"));
    }
    if let Some(local) = local_app_data() {
        out.push(local.join("huggingface").join("hub"));
    }
    out
}

/// Primary HF hub cache used when resolving Lemonade `org/repo:file` specs.
fn hf_hub_cache() -> Option<PathBuf> {
    let cands = hf_hub_cache_candidates();
    cands
        .iter()
        .find(|p| p.is_dir())
        .cloned()
        .or_else(|| cands.into_iter().next())
}

fn lemonade_cache_candidates() -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Some(p) = env_path(&["LEMONADE_CACHE_DIR"]) {
        out.push(p);
        return out;
    }
    if let Some(home) = home_dir() {
        out.push(home.join(".cache").join("lemonade"));
        #[cfg(target_os = "macos")]
        {
            out.push(PathBuf::from(
                "/Library/Application Support/lemonade/.cache",
            ));
            out.push(
                home.join("Library")
                    .join("Application Support")
                    .join("lemonade")
                    .join(".cache"),
            );
        }
    }
    if let Some(local) = local_app_data() {
        out.push(local.join("lemonade").join(".cache"));
        out.push(local.join("Lemonade").join(".cache"));
    }
    #[cfg(target_os = "linux")]
    {
        out.push(PathBuf::from("/var/lib/lemonade/.cache/lemonade"));
        out.push(PathBuf::from("/opt/var/lib/lemonade/.cache/lemonade"));
    }
    out
}

/// Scan configured local caches for weight files.
pub fn scan_weights(opts: &DiscoverOpts) -> Result<Vec<DiscoveredWeight>> {
    let roots = candidate_source_roots();
    scan_weights_in_roots(&roots, opts)
}

/// Scan only the given `(source, root)` pairs (ignores host default roots).
///
/// Still honors [`DiscoverOpts::extra_roots`], `query`, `sniff_arch`, and
/// `max_depth`. Use this from libraries/tests that inject fixture directories.
pub fn scan_weights_in_roots(
    roots: &[(WeightSourceKind, PathBuf)],
    opts: &DiscoverOpts,
) -> Result<Vec<DiscoveredWeight>> {
    let mut acc = DiscoveryAccum::new(opts);
    let mut seen_root: BTreeSet<(WeightSourceKind, PathBuf)> = BTreeSet::new();

    for (kind, root) in roots {
        if !opts.allows(*kind) {
            continue;
        }
        if !root.is_dir() {
            continue;
        }
        let key = (*kind, root.clone());
        if !seen_root.insert(key) {
            continue;
        }
        match kind {
            WeightSourceKind::Ollama => scan_ollama(root, &mut acc)?,
            WeightSourceKind::HuggingFace | WeightSourceKind::Mlx | WeightSourceKind::Vllm => {
                scan_hf_style(root, *kind, &mut acc)?;
            }
            WeightSourceKind::Lemonade => scan_lemonade(root, &mut acc)?,
            WeightSourceKind::LmStudio | WeightSourceKind::RlxLocal | WeightSourceKind::Extra => {
                walk_weight_files(root, *kind, 0, opts.max_depth, &mut acc)?;
            }
        }
    }

    for root in &opts.extra_roots {
        if opts.allows(WeightSourceKind::Extra) && root.is_dir() {
            walk_weight_files(root, WeightSourceKind::Extra, 0, opts.max_depth, &mut acc)?;
        }
    }

    let mut items = acc.finish()?;
    if let Some(q) = &opts.query {
        let q = q.to_ascii_lowercase();
        items.retain(|w| matches_query(w, &q));
    }
    items.sort_by(|a, b| {
        a.best_priority()
            .cmp(&b.best_priority())
            .then_with(|| a.display_name.cmp(&b.display_name))
            .then_with(|| a.path.cmp(&b.path))
    });
    Ok(items)
}

/// Resolve a short query to a single weight path, or error with candidates.
pub fn resolve_weight_query(query: &str, opts: &DiscoverOpts) -> Result<PathBuf> {
    let roots = candidate_source_roots();
    resolve_weight_query_in_roots(query, &roots, opts)
}

/// Like [`resolve_weight_query`], but only searches `roots`.
pub fn resolve_weight_query_in_roots(
    query: &str,
    roots: &[(WeightSourceKind, PathBuf)],
    opts: &DiscoverOpts,
) -> Result<PathBuf> {
    let q = query.trim();
    if q.is_empty() {
        bail!("empty weight query");
    }
    let mut discover = opts.clone();
    discover.query = Some(q.to_string());
    let mut hits = scan_weights_in_roots(roots, &discover)?;
    if hits.is_empty() {
        bail!("no local weights matched query `{q}` (try `rlx-inspect scan --query {q}`)");
    }

    let prefer = discover
        .prefer_quant
        .as_deref()
        .unwrap_or(DEFAULT_GGUF_PREFER_SUBSTR);
    let preferred: Vec<_> = hits
        .iter()
        .filter(|w| {
            w.format == DiscoveredFormat::Gguf
                && w.path
                    .file_name()
                    .and_then(|s| s.to_str())
                    .is_some_and(|n| n.contains(prefer))
        })
        .cloned()
        .collect();
    if preferred.len() == 1 {
        return Ok(preferred[0].path.clone());
    }
    if !preferred.is_empty() {
        hits = preferred;
    }

    hits.sort_by(|a, b| {
        a.best_priority()
            .cmp(&b.best_priority())
            .then_with(|| a.path.cmp(&b.path))
    });
    let best_pri = hits[0].best_priority();
    let top: Vec<_> = hits
        .into_iter()
        .filter(|w| w.best_priority() == best_pri)
        .collect();
    if top.len() == 1 {
        return Ok(top[0].path.clone());
    }

    let listing: Vec<String> = top
        .iter()
        .map(|w| {
            let srcs: Vec<&str> = w.sources.iter().map(|s| s.as_str()).collect();
            format!(
                "  - [{}] {} → {}",
                srcs.join("+"),
                w.display_name,
                w.path.display()
            )
        })
        .collect();
    bail!(
        "ambiguous weight query `{q}` ({} matches); pass an exact path or narrow with --prefer / --query:\n{}",
        top.len(),
        listing.join("\n")
    );
}

/// If `path_or_query` exists on disk, resolve like [`resolve_weights_file_with_options`];
/// otherwise treat it as a discovery query (when it does not look like a path).
pub fn resolve_weights_path_or_query(
    path_or_query: &Path,
    resolve: &ResolveWeightsOptions<'_>,
    discover: &DiscoverOpts,
) -> Result<PathBuf> {
    if path_or_query.exists() {
        return resolve_weights_file_with_options(path_or_query, resolve);
    }
    let q = path_or_query
        .to_str()
        .ok_or_else(|| anyhow!("non-utf8 weight query"))?;
    if looks_like_filesystem_path(q) {
        bail!("weights path not found: {path_or_query:?}");
    }
    let mut opts = discover.clone();
    if opts.prefer_quant.is_none() {
        opts.prefer_quant = resolve
            .prefer_gguf_substring
            .map(str::to_string)
            .or_else(|| Some(DEFAULT_GGUF_PREFER_SUBSTR.to_string()));
    }
    resolve_weight_query(q, &opts)
}

fn matches_query(w: &DiscoveredWeight, q_lower: &str) -> bool {
    if w.display_name.to_ascii_lowercase().contains(q_lower) {
        return true;
    }
    if w.path
        .to_string_lossy()
        .to_ascii_lowercase()
        .contains(q_lower)
    {
        return true;
    }
    w.path
        .file_stem()
        .and_then(|s| s.to_str())
        .is_some_and(|s| s.to_ascii_lowercase().contains(q_lower))
}

// --- accumulator / dedup ---------------------------------------------------

struct DiscoveryAccum {
    by_key: HashMap<String, DiscoveredWeight>,
    sniff_arch: bool,
}

impl DiscoveryAccum {
    fn new(opts: &DiscoverOpts) -> Self {
        Self {
            by_key: HashMap::new(),
            sniff_arch: opts.sniff_arch,
        }
    }

    fn push(
        &mut self,
        path: PathBuf,
        source: WeightSourceKind,
        display_name: String,
        format: DiscoveredFormat,
    ) {
        if !path.exists() {
            return;
        }
        let key = dedup_key(&path);
        let size_bytes = fs::metadata(&path).ok().map(|m| m.len());
        let quant_hint = quant_from_name(&path);
        let entry = self.by_key.entry(key).or_insert_with(|| DiscoveredWeight {
            path: path.clone(),
            sources: Vec::new(),
            display_name: display_name.clone(),
            format,
            size_bytes,
            quant_hint: quant_hint.clone(),
            arch_hint: None,
        });
        if !entry.sources.contains(&source) {
            entry.sources.push(source);
            entry.sources.sort();
        }
        // Prefer registry-style names (lemonade:MyModel) over raw filenames.
        if better_display_name(&entry.display_name, &display_name) {
            entry.display_name = display_name;
        }
        if entry.format == DiscoveredFormat::Unknown && format != DiscoveredFormat::Unknown {
            entry.format = format;
        }
        if entry.size_bytes.is_none() {
            entry.size_bytes = size_bytes;
        }
        if entry.quant_hint.is_none() {
            entry.quant_hint = quant_hint;
        }
    }

    fn finish(mut self) -> Result<Vec<DiscoveredWeight>> {
        if self.sniff_arch {
            for w in self.by_key.values_mut() {
                if w.format == DiscoveredFormat::Gguf && w.arch_hint.is_none() {
                    if let Ok(arch) = gguf_architecture_from_path(&w.path) {
                        w.arch_hint = Some(arch);
                    }
                }
            }
        }
        Ok(self.by_key.into_values().collect())
    }
}

fn dedup_key(path: &Path) -> String {
    path.canonicalize()
        .unwrap_or_else(|_| path.to_path_buf())
        .to_string_lossy()
        .into_owned()
}

/// Prefer registry-style names over path/filename leaves when merging duplicates.
fn better_display_name(old: &str, new: &str) -> bool {
    let old_leaf = old.rsplit(':').next().unwrap_or(old);
    let new_leaf = new.rsplit(':').next().unwrap_or(new);
    let old_fileish = old_leaf.contains('.') || old_leaf.contains('/');
    let new_fileish = new_leaf.contains('.') || new_leaf.contains('/');
    match (old_fileish, new_fileish) {
        (true, false) => true,
        (false, true) => false,
        _ => new.len() > old.len(),
    }
}

fn quant_from_name(path: &Path) -> Option<String> {
    let name = path.file_name()?.to_str()?.to_ascii_uppercase();
    // Common GGUF quant tags, longest first.
    const TAGS: &[&str] = &[
        "Q8_0", "Q6_K", "Q5_K_M", "Q5_K_S", "Q5_0", "Q4_K_M", "Q4_K_S", "Q4_0", "Q3_K_M", "Q3_K_S",
        "Q2_K", "IQ4_XS", "IQ3_XXS", "IQ2_XXS", "F16", "BF16", "F32",
    ];
    for t in TAGS {
        if name.contains(t) {
            return Some((*t).to_string());
        }
    }
    None
}

fn format_from_path(path: &Path) -> DiscoveredFormat {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    match ext.as_str() {
        "gguf" => DiscoveredFormat::Gguf,
        "safetensors" => DiscoveredFormat::Safetensors,
        _ => {
            if looks_like_gguf_magic(path) {
                DiscoveredFormat::Gguf
            } else {
                DiscoveredFormat::Unknown
            }
        }
    }
}

fn looks_like_gguf_magic(path: &Path) -> bool {
    let Ok(mut f) = File::open(path) else {
        return false;
    };
    let mut magic = [0u8; 4];
    if f.read_exact(&mut magic).is_err() {
        return false;
    }
    &magic == b"GGUF"
}

fn skip_dir_name(name: &str) -> bool {
    matches!(
        name,
        ".git" | "node_modules" | ".locks" | "__pycache__" | ".cache"
    )
}

// --- walkers ---------------------------------------------------------------

fn walk_weight_files(
    root: &Path,
    source: WeightSourceKind,
    depth: usize,
    max_depth: usize,
    acc: &mut DiscoveryAccum,
) -> Result<()> {
    if depth > max_depth {
        return Ok(());
    }
    let entries = match fs::read_dir(root) {
        Ok(e) => e,
        Err(_) => return Ok(()),
    };
    for ent in entries.flatten() {
        let path = ent.path();
        let file_type = ent.file_type().ok();
        let is_dir = file_type.as_ref().is_some_and(|t| t.is_dir()) || path.is_dir();
        let is_file = file_type.as_ref().is_some_and(|t| t.is_file()) || path.is_file();
        if is_dir {
            let name = path
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or_default();
            if skip_dir_name(name) || name == "blobs" {
                continue;
            }
            // HF-style: prefer snapshots walk from scan_hf_style; if we hit
            // models--* here (RLX local), still recurse.
            walk_weight_files(&path, source, depth + 1, max_depth, acc)?;
            // Safetensors model dir: directory containing model.safetensors
            let st = path.join("model.safetensors");
            if st.is_file() {
                let display = display_for_path(source, &path);
                acc.push(path.clone(), source, display, DiscoveredFormat::Safetensors);
            }
        } else if is_file {
            let fmt = format_from_path(&path);
            if matches!(fmt, DiscoveredFormat::Gguf | DiscoveredFormat::Safetensors) {
                let display = display_for_path(source, &path);
                acc.push(path, source, display, fmt);
            }
        }
    }
    Ok(())
}

fn display_for_path(source: WeightSourceKind, path: &Path) -> String {
    let leaf = path.file_name().and_then(|s| s.to_str()).unwrap_or("model");
    match source {
        WeightSourceKind::LmStudio => {
            // publisher/model/file.gguf
            let parent = path.parent();
            let model = parent.and_then(|p| p.file_name()).and_then(|s| s.to_str());
            let pub_ = parent
                .and_then(|p| p.parent())
                .and_then(|p| p.file_name())
                .and_then(|s| s.to_str());
            match (pub_, model) {
                (Some(a), Some(b)) => format!("lmstudio:{a}/{b}/{leaf}"),
                _ => format!("lmstudio:{leaf}"),
            }
        }
        WeightSourceKind::RlxLocal => format!("rlx:{leaf}"),
        WeightSourceKind::Extra => format!("extra:{leaf}"),
        other => format!("{}:{leaf}", other.as_str()),
    }
}

fn scan_hf_style(hub: &Path, source: WeightSourceKind, acc: &mut DiscoveryAccum) -> Result<()> {
    let entries = match fs::read_dir(hub) {
        Ok(e) => e,
        Err(_) => return Ok(()),
    };
    for ent in entries.flatten() {
        let repo_dir = ent.path();
        if !repo_dir.is_dir() {
            continue;
        }
        let repo_name = repo_dir
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or_default();
        if !repo_name.starts_with("models--") {
            continue;
        }
        let repo_id = hf_repo_id_from_dir(repo_name);
        let is_mlx = repo_id.contains("mlx-community")
            || repo_id.to_ascii_lowercase().contains("/mlx")
            || repo_name.to_ascii_lowercase().contains("mlx");
        let label_source = match source {
            WeightSourceKind::Mlx if !is_mlx => continue,
            WeightSourceKind::Mlx => WeightSourceKind::Mlx,
            WeightSourceKind::HuggingFace if is_mlx => WeightSourceKind::Mlx,
            WeightSourceKind::HuggingFace => WeightSourceKind::HuggingFace,
            WeightSourceKind::Vllm => WeightSourceKind::Vllm,
            WeightSourceKind::Lemonade => WeightSourceKind::Lemonade,
            other => other,
        };

        let snapshots = repo_dir.join("snapshots");
        if !snapshots.is_dir() {
            continue;
        }
        let snap_entries = match fs::read_dir(&snapshots) {
            Ok(e) => e,
            Err(_) => continue,
        };
        for snap in snap_entries.flatten() {
            let snap_path = snap.path();
            if !snap_path.is_dir() {
                continue;
            }
            collect_snapshot_weights(&snap_path, label_source, &repo_id, acc)?;
        }
    }
    Ok(())
}

fn hf_repo_id_from_dir(dir_name: &str) -> String {
    // models--org--repo → org/repo (repo may contain --)
    let rest = dir_name.strip_prefix("models--").unwrap_or(dir_name);
    let parts: Vec<&str> = rest.splitn(2, "--").collect();
    if parts.len() == 2 {
        format!("{}/{}", parts[0], parts[1].replace("--", "/"))
    } else {
        rest.replace("--", "/")
    }
}

fn collect_snapshot_weights(
    snap: &Path,
    source: WeightSourceKind,
    repo_id: &str,
    acc: &mut DiscoveryAccum,
) -> Result<()> {
    let mut ggufs = Vec::new();
    let mut has_model_st = false;
    let mut has_any_st = false;
    let entries = match fs::read_dir(snap) {
        Ok(e) => e,
        Err(_) => return Ok(()),
    };
    for ent in entries.flatten() {
        let path = ent.path();
        if !path.is_file() {
            continue;
        }
        let name = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or_default();
        if name.ends_with(".gguf") {
            ggufs.push(path);
        } else if name == "model.safetensors" {
            has_model_st = true;
            has_any_st = true;
        } else if name.ends_with(".safetensors") {
            has_any_st = true;
        }
    }
    for g in ggufs {
        let leaf = g
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("model.gguf");
        let display = format!("{}:{repo_id}/{leaf}", source.as_str());
        acc.push(g, source, display, DiscoveredFormat::Gguf);
    }
    if has_model_st || has_any_st {
        let display = format!("{}:{repo_id}", source.as_str());
        acc.push(
            snap.to_path_buf(),
            source,
            display,
            DiscoveredFormat::Safetensors,
        );
    }
    Ok(())
}

fn scan_ollama(models_dir: &Path, acc: &mut DiscoveryAccum) -> Result<()> {
    let manifests = models_dir.join("manifests");
    let blobs = models_dir.join("blobs");
    if !manifests.is_dir() || !blobs.is_dir() {
        return Ok(());
    }
    walk_ollama_manifests(&manifests, &blobs, &manifests, acc)?;
    Ok(())
}

fn walk_ollama_manifests(
    manifests_root: &Path,
    blobs: &Path,
    dir: &Path,
    acc: &mut DiscoveryAccum,
) -> Result<()> {
    let entries = match fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return Ok(()),
    };
    for ent in entries.flatten() {
        let path = ent.path();
        if path.is_dir() {
            walk_ollama_manifests(manifests_root, blobs, &path, acc)?;
            continue;
        }
        if !path.is_file() {
            continue;
        }
        let text = match fs::read_to_string(&path) {
            Ok(t) => t,
            Err(_) => continue,
        };
        let Ok(manifest) = serde_json::from_str::<OllamaManifest>(&text) else {
            continue;
        };
        let Some(layer) = manifest
            .layers
            .iter()
            .find(|l| l.media_type == "application/vnd.ollama.image.model")
        else {
            continue;
        };
        let blob_name = layer.digest.replace(':', "-");
        let blob_path = blobs.join(&blob_name);
        if !blob_path.is_file() {
            continue;
        }
        if !looks_like_gguf_magic(&blob_path) {
            // Non-GGUF Ollama layers are not usable by RLX runners yet.
            continue;
        }
        let rel = path
            .strip_prefix(manifests_root)
            .unwrap_or(&path)
            .to_string_lossy()
            .replace('\\', "/");
        // registry.ollama.ai/library/name/tag → ollama:name:tag
        let display = ollama_display_name(&rel);
        acc.push(
            blob_path,
            WeightSourceKind::Ollama,
            display,
            DiscoveredFormat::Gguf,
        );
    }
    Ok(())
}

fn ollama_display_name(rel: &str) -> String {
    let parts: Vec<&str> = rel.split('/').collect();
    if parts.len() >= 2 {
        let tag = parts[parts.len() - 1];
        let name = parts[parts.len() - 2];
        format!("ollama:{name}:{tag}")
    } else {
        format!("ollama:{rel}")
    }
}

#[derive(Debug, Deserialize)]
struct OllamaManifest {
    #[serde(default)]
    layers: Vec<OllamaLayer>,
}

#[derive(Debug, Deserialize)]
struct OllamaLayer {
    digest: String,
    #[serde(rename = "mediaType")]
    media_type: String,
}

fn scan_lemonade(cache_dir: &Path, acc: &mut DiscoveryAccum) -> Result<()> {
    // config.json → models_dir / extra_models_dir
    let config_path = cache_dir.join("config.json");
    let mut models_dirs: Vec<PathBuf> = Vec::new();
    let mut extra_dir: Option<PathBuf> = None;
    if config_path.is_file() {
        if let Ok(text) = fs::read_to_string(&config_path) {
            if let Ok(cfg) = serde_json::from_str::<Value>(&text) {
                if let Some(md) = cfg.get("models_dir").and_then(|v| v.as_str()) {
                    if md != "auto" && !md.is_empty() {
                        models_dirs.push(PathBuf::from(md));
                    }
                }
                if let Some(ed) = cfg.get("extra_models_dir").and_then(|v| v.as_str()) {
                    if !ed.is_empty() {
                        extra_dir = Some(PathBuf::from(ed));
                    }
                }
            }
        }
    }
    // Default: walk the lemonade cache for dropped GGUFs. HF hub is scanned
    // separately as HuggingFace/Mlx/Vllm; Lemonade named models come from
    // user_models.json via resolve_lemonade_checkpoint.
    models_dirs.push(cache_dir.to_path_buf());
    for dir in models_dirs {
        if dir.is_dir() {
            // Prefer HF snapshot layout if this looks like a hub cache.
            if fs::read_dir(&dir)
                .ok()
                .map(|mut i| {
                    i.any(|e| {
                        e.ok()
                            .map(|e| e.file_name().to_string_lossy().starts_with("models--"))
                            .unwrap_or(false)
                    })
                })
                .unwrap_or(false)
            {
                scan_hf_style(&dir, WeightSourceKind::Lemonade, acc)?;
            } else {
                walk_weight_files(&dir, WeightSourceKind::Lemonade, 0, 6, acc)?;
            }
        }
    }
    if let Some(ed) = extra_dir {
        if ed.is_dir() {
            walk_weight_files(&ed, WeightSourceKind::Lemonade, 0, 6, acc)?;
        }
    }

    // user_models.json named checkpoints
    let user_models = cache_dir.join("user_models.json");
    if user_models.is_file() {
        index_lemonade_user_models(&user_models, acc)?;
    }
    Ok(())
}

fn index_lemonade_user_models(path: &Path, acc: &mut DiscoveryAccum) -> Result<()> {
    let text = fs::read_to_string(path)
        .with_context(|| format!("read lemonade user_models {}", path.display()))?;
    let map: BTreeMap<String, Value> = match serde_json::from_str(&text) {
        Ok(m) => m,
        Err(_) => return Ok(()),
    };
    for (name, entry) in map {
        let checkpoint = entry
            .get("checkpoint")
            .and_then(|v| v.as_str())
            .map(str::to_string);
        let checkpoints = entry.get("checkpoints").cloned();
        let mut specs = Vec::new();
        if let Some(c) = checkpoint {
            specs.push(c);
        }
        if let Some(Value::Object(obj)) = checkpoints {
            for (_k, v) in obj {
                if let Some(s) = v.as_str() {
                    specs.push(s.to_string());
                }
            }
        }
        for spec in specs {
            let as_path = PathBuf::from(&spec);
            let resolved = if as_path.exists() {
                Some(as_path)
            } else {
                resolve_lemonade_checkpoint(&spec)
            };
            if let Some(resolved) = resolved {
                let fmt = format_from_path(&resolved);
                if matches!(fmt, DiscoveredFormat::Gguf | DiscoveredFormat::Safetensors)
                    || resolved.is_dir()
                {
                    let display = format!("lemonade:{name}");
                    let fmt = if resolved.is_dir() {
                        DiscoveredFormat::Safetensors
                    } else {
                        fmt
                    };
                    acc.push(resolved, WeightSourceKind::Lemonade, display, fmt);
                }
            }
        }
    }
    Ok(())
}

/// Resolve `org/repo:file.gguf` or `org/repo:QUANT` against the HF hub cache.
fn resolve_lemonade_checkpoint(spec: &str) -> Option<PathBuf> {
    let hub = hf_hub_cache()?;
    let (repo, variant) = match spec.split_once(':') {
        Some((r, v)) => (r, Some(v)),
        None => (spec, None),
    };
    let dir_name = format!("models--{}", repo.replace('/', "--"));
    let snapshots = hub.join(&dir_name).join("snapshots");
    if !snapshots.is_dir() {
        return None;
    }
    // Prefer refs/main → snapshot
    let mut snap_dirs = Vec::new();
    if let Ok(main) = fs::read_to_string(hub.join(&dir_name).join("refs/main")) {
        let hash = main.trim();
        let p = snapshots.join(hash);
        if p.is_dir() {
            snap_dirs.push(p);
        }
    }
    if snap_dirs.is_empty() {
        if let Ok(rd) = fs::read_dir(&snapshots) {
            for e in rd.flatten() {
                if e.path().is_dir() {
                    snap_dirs.push(e.path());
                }
            }
        }
    }
    for snap in snap_dirs {
        if let Some(v) = variant {
            let exact = snap.join(v);
            if exact.is_file() {
                return Some(exact);
            }
            // Quant shorthand: find *.gguf containing QUANT
            if let Ok(rd) = fs::read_dir(&snap) {
                let mut matches: Vec<PathBuf> = rd
                    .flatten()
                    .map(|e| e.path())
                    .filter(|p| {
                        p.extension()
                            .and_then(|e| e.to_str())
                            .is_some_and(|e| e.eq_ignore_ascii_case("gguf"))
                            && p.file_name().and_then(|s| s.to_str()).is_some_and(|n| {
                                n.to_ascii_uppercase().contains(&v.to_ascii_uppercase())
                            })
                    })
                    .collect();
                matches.sort();
                if let Some(p) = matches.into_iter().next() {
                    return Some(p);
                }
            }
        } else {
            let st = snap.join("model.safetensors");
            if st.is_file() {
                return Some(snap);
            }
            if let Ok(rd) = fs::read_dir(&snap) {
                let mut ggufs: Vec<_> = rd
                    .flatten()
                    .map(|e| e.path())
                    .filter(|p| {
                        p.extension()
                            .and_then(|e| e.to_str())
                            .is_some_and(|e| e.eq_ignore_ascii_case("gguf"))
                    })
                    .collect();
                ggufs.sort();
                if let Some(p) = ggufs.into_iter().next() {
                    return Some(p);
                }
            }
            return Some(snap);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Temp fixture dir removed on drop.
    struct TempDir(PathBuf);

    impl TempDir {
        fn new(name: &str) -> Self {
            let base = std::env::temp_dir().join(format!(
                "rlx-weights-discover-{}-{}-{}",
                name,
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            fs::create_dir_all(&base).unwrap();
            Self(base)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn looks_like_path_windows_and_unix() {
        assert!(looks_like_filesystem_path(r"C:\Models\qwen.gguf"));
        assert!(looks_like_filesystem_path(r"D:"));
        assert!(looks_like_filesystem_path("/home/u/model.gguf"));
        assert!(looks_like_filesystem_path("./weights"));
        assert!(looks_like_filesystem_path(r".\weights"));
        assert!(looks_like_filesystem_path("../cache"));
        assert!(!looks_like_filesystem_path("qwen3-0.6b"));
        assert!(!looks_like_filesystem_path("model.gguf"));
        assert!(!looks_like_filesystem_path("Q4_K_M"));
    }

    #[test]
    fn split_path_list_preserves_windows_drives() {
        let paths = split_path_list(Some(r"C:\Models;D:\More".into()));
        assert_eq!(paths.len(), 2);
        assert_eq!(paths[0], PathBuf::from(r"C:\Models"));
        assert_eq!(paths[1], PathBuf::from(r"D:\More"));
    }

    #[test]
    fn resolve_in_roots_prefers_quant() {
        let dir = TempDir::new("resolve-api");
        write_minimal_gguf(&dir.path().join("Thing-Q4_K_M.gguf"));
        write_minimal_gguf(&dir.path().join("Thing-Q8_0.gguf"));
        let roots = [(WeightSourceKind::RlxLocal, dir.path().to_path_buf())];
        let path = resolve_weight_query_in_roots(
            "Thing",
            &roots,
            &DiscoverOpts::default().with_prefer_quant("Q8_0"),
        )
        .unwrap();
        assert!(path.to_string_lossy().contains("Q8_0"));
    }

    fn write_minimal_gguf(path: &Path) {
        // GGUF magic + version + empty counts is enough for magic sniff / listing.
        let mut f = File::create(path).unwrap();
        f.write_all(b"GGUF").unwrap();
        f.write_all(&3u32.to_le_bytes()).unwrap(); // version
        f.write_all(&0u64.to_le_bytes()).unwrap(); // tensor_count
        f.write_all(&0u64.to_le_bytes()).unwrap(); // kv_count
    }

    #[test]
    fn lmstudio_layout_scan() {
        let dir = TempDir::new("lms");
        let model = dir.path().join("publisher").join("cool-model");
        fs::create_dir_all(&model).unwrap();
        let gguf = model.join("cool-model-Q4_K_M.gguf");
        write_minimal_gguf(&gguf);
        let hits = scan_weights_in_roots(
            &[(WeightSourceKind::LmStudio, dir.path().to_path_buf())],
            &DiscoverOpts::default(),
        )
        .unwrap();
        assert_eq!(hits.len(), 1);
        assert!(
            hits[0]
                .display_name
                .contains("lmstudio:publisher/cool-model")
        );
        assert_eq!(hits[0].format, DiscoveredFormat::Gguf);
        assert_eq!(hits[0].quant_hint.as_deref(), Some("Q4_K_M"));
    }

    #[test]
    fn ollama_manifest_to_blob() {
        let dir = TempDir::new("ollama");
        let manifests = dir
            .path()
            .join("manifests")
            .join("registry.ollama.ai")
            .join("library")
            .join("qwen2.5");
        fs::create_dir_all(&manifests).unwrap();
        let blobs = dir.path().join("blobs");
        fs::create_dir_all(&blobs).unwrap();
        let digest = "sha256:deadbeefcafebabe";
        let blob = blobs.join("sha256-deadbeefcafebabe");
        write_minimal_gguf(&blob);
        let manifest = serde_json::json!({
            "layers": [{
                "digest": digest,
                "mediaType": "application/vnd.ollama.image.model"
            }]
        });
        fs::write(
            manifests.join("32b"),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();
        let hits = scan_weights_in_roots(
            &[(WeightSourceKind::Ollama, dir.path().to_path_buf())],
            &DiscoverOpts::default(),
        )
        .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].display_name, "ollama:qwen2.5:32b");
        assert_eq!(hits[0].path, blob);
    }

    #[test]
    fn hf_snapshot_gguf() {
        let hub = TempDir::new("hf");
        let snap = hub
            .path()
            .join("models--org--Cool-Model")
            .join("snapshots")
            .join("abc123");
        fs::create_dir_all(&snap).unwrap();
        let gguf = snap.join("Cool-Model-Q4_K_M.gguf");
        write_minimal_gguf(&gguf);
        let hits = scan_weights_in_roots(
            &[(WeightSourceKind::HuggingFace, hub.path().to_path_buf())],
            &DiscoverOpts::default(),
        )
        .unwrap();
        assert_eq!(hits.len(), 1);
        assert!(hits[0].display_name.contains("org/Cool-Model"));
    }

    #[test]
    fn lemonade_user_models_checkpoint() {
        let cache = TempDir::new("lemonade");
        let models = cache.path().join("extra-models");
        fs::create_dir_all(&models).unwrap();
        let gguf = models.join("LemModel-Q4_K_M.gguf");
        write_minimal_gguf(&gguf);
        let user = serde_json::json!({
            "MyLem": {
                "checkpoint": gguf.to_string_lossy(),
                "recipe": "llamacpp"
            }
        });
        fs::write(
            cache.path().join("user_models.json"),
            serde_json::to_vec_pretty(&user).unwrap(),
        )
        .unwrap();
        let hits = scan_weights_in_roots(
            &[(WeightSourceKind::Lemonade, cache.path().to_path_buf())],
            &DiscoverOpts::default(),
        )
        .unwrap();
        assert!(
            hits.iter().any(|h| h.display_name == "lemonade:MyLem"),
            "hits={hits:?}"
        );
    }

    #[test]
    fn resolve_unique_and_ambiguous() {
        let dir = TempDir::new("resolve");
        let a = dir.path().join("alpha");
        let b = dir.path().join("beta");
        fs::create_dir_all(&a).unwrap();
        fs::create_dir_all(&b).unwrap();
        write_minimal_gguf(&a.join("UniqueWidget-Q4_K_M.gguf"));
        write_minimal_gguf(&b.join("SharedThing-Q4_K_M.gguf"));
        write_minimal_gguf(&b.join("SharedThing-Q8_0.gguf"));

        let roots = [(WeightSourceKind::RlxLocal, dir.path().to_path_buf())];
        let hits =
            scan_weights_in_roots(&roots, &DiscoverOpts::default().with_query("UniqueWidget"))
                .unwrap();
        assert_eq!(hits.len(), 1);

        let amb = scan_weights_in_roots(&roots, &DiscoverOpts::default().with_query("SharedThing"))
            .unwrap();
        assert_eq!(amb.len(), 2);

        let preferred: Vec<_> = amb
            .iter()
            .filter(|w| {
                w.format == DiscoveredFormat::Gguf
                    && w.path
                        .file_name()
                        .and_then(|s| s.to_str())
                        .is_some_and(|n| n.contains("Q8_0"))
            })
            .collect();
        assert_eq!(preferred.len(), 1);
        assert!(preferred[0].path.to_string_lossy().contains("Q8_0"));

        let err = resolve_weights_path_or_query(
            Path::new("/no/such/model.gguf"),
            &ResolveWeightsOptions::default(),
            &DiscoverOpts::default(),
        )
        .unwrap_err();
        assert!(err.to_string().contains("not found"));
    }

    #[test]
    fn path_or_query_existing_file() {
        let dir = TempDir::new("porq");
        let gguf = dir.path().join("x.gguf");
        write_minimal_gguf(&gguf);
        let got = resolve_weights_path_or_query(
            &gguf,
            &ResolveWeightsOptions::default(),
            &DiscoverOpts::default(),
        )
        .unwrap();
        assert_eq!(got, gguf);
    }
}
