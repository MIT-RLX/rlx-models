//! Versatile model-asset loading.
//!
//! A model "bundle" is a set of named assets — a `config.json`, one or more
//! weight files (safetensors / GGUF / nested `.rlxp` graphs), a tokenizer,
//! frontend data, etc. Historically every crate loaded these with
//! `std::fs::read(dir.join(name))`, which hard-wires a filesystem directory.
//! [`AssetSource`] abstracts *where the bytes come from* so the exact same model
//! can be loaded from:
//!
//! * a **directory** — `AssetSource::dir("weights/my-model")`
//! * a single **packed file** — `AssetSource::pack_file("my-model.rlxp")`
//!   (official [`.rlxp`](https://github.com/MIT-RLX/rlx/blob/main/docs/rlxp.md)
//!   with the `rlxp` feature; legacy `.rlxpack` still opens)
//! * **in-memory** bytes — `AssetSource::memory(map)` or an in-memory pack
//! * a **config spec** (JSON) — `AssetSource::from_spec(&spec)` for
//!   `{"source":"dir","path":"…"}` / `{"source":"pack","path":"…"}`
//! * a **custom provider** — anything implementing [`AssetProvider`]
//!   (an HTTP cache, an embedded VFS, a zip, …)
//!
//! With feature **`native-pack`**, [`native_pack`] bakes local ONNX subgraphs into
//! nested `graphs/*.rlxp` and builds outer Hub packs that ship **no** `.onnx`.
//!
//! DX: `AssetSource` is `Clone` + `From<&Path>/From<PathBuf>/From<&str>` (which
//! auto-detects dir vs. `*.rlxp` / `*.rlxpack` file) and `From<HashMap<String,Vec<u8>>>`,
//! so a model's entry point can be `load(src: impl Into<AssetSource>)` and
//! callers just pass a path, a file, or a byte map. For sub-loaders that insist
//! on a real path (tokenizers, the g2p frontend), [`AssetSource::local_dir`]
//! returns the backing directory when there is one, or transparently
//! materializes the needed subtree to a self-cleaning temp directory.

use std::borrow::Cow;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context, Result, anyhow, bail};
use serde::Deserialize;

#[cfg(feature = "native-pack")]
pub mod native_pack;

/// A source of named model assets. Read bytes by relative, `/`-separated name.
///
/// Implementors must be cheap to share (`Send + Sync`); [`AssetSource`] wraps
/// one in an `Arc`. Names are normalized to `/` separators and never start
/// with `/` or contain `..`.
pub trait AssetProvider: Send + Sync {
    /// Read one asset's bytes, borrowing from the provider when possible.
    fn read_bytes(&self, name: &str) -> Result<Cow<'_, [u8]>>;

    /// Whether the named asset exists (no read).
    fn exists(&self, name: &str) -> bool;

    /// Every asset name this provider can serve (used by `local_dir` /
    /// `materialize` and for good "asset not found, have: …" errors).
    fn names(&self) -> Result<Vec<String>>;

    /// The backing filesystem directory, if this provider is one — lets
    /// path-based sub-loaders skip materialization.
    fn as_dir(&self) -> Option<&Path> {
        None
    }
}

impl AssetProvider for Box<dyn AssetProvider> {
    fn read_bytes(&self, name: &str) -> Result<Cow<'_, [u8]>> {
        (**self).read_bytes(name)
    }
    fn exists(&self, name: &str) -> bool {
        (**self).exists(name)
    }
    fn names(&self) -> Result<Vec<String>> {
        (**self).names()
    }
    fn as_dir(&self) -> Option<&Path> {
        (**self).as_dir()
    }
}

/// A cloneable handle to a bundle of model assets. See the module docs.
#[derive(Clone)]
pub struct AssetSource {
    provider: Arc<dyn AssetProvider>,
    /// Prefix applied to every name (for [`AssetSource::subdir`] scoping).
    prefix: String,
}

impl std::fmt::Debug for AssetSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AssetSource")
            .field("dir", &self.provider.as_dir())
            .field("prefix", &self.prefix)
            .finish()
    }
}

fn normalize(name: &str) -> String {
    name.trim_start_matches('/').replace('\\', "/")
}

impl AssetSource {
    /// Wrap any [`AssetProvider`].
    pub fn new(provider: impl AssetProvider + 'static) -> Self {
        Self {
            provider: Arc::new(provider),
            prefix: String::new(),
        }
    }

    /// Load assets from a filesystem directory (the classic layout).
    pub fn dir(path: impl Into<PathBuf>) -> Self {
        Self::new(DirProvider { root: path.into() })
    }

    /// Load assets from an in-memory name → bytes map.
    pub fn memory(map: HashMap<String, Vec<u8>>) -> Self {
        let map = map.into_iter().map(|(k, v)| (normalize(&k), v)).collect();
        Self::new(MemoryProvider { map })
    }

    /// Load assets from a packed container on disk (`.rlxp` with the `rlxp`
    /// feature, or legacy `.rlxpack`).
    pub fn pack_file(path: impl AsRef<Path>) -> Result<Self> {
        Ok(Self::new(pack::open_pack_provider(path.as_ref())?))
    }

    /// Load assets from an in-memory packed container (`.rlxp` / `.rlxpack`).
    pub fn pack_bytes(bytes: impl Into<Arc<[u8]>>) -> Result<Self> {
        Ok(Self::new(pack::open_pack_bytes(bytes.into())?))
    }

    /// Build from a declarative [`SourceSpec`] (e.g. parsed from a config file).
    pub fn from_spec(spec: &SourceSpec) -> Result<Self> {
        match spec {
            SourceSpec::Dir { path } => Ok(Self::dir(path.clone())),
            SourceSpec::Pack { path } => Self::pack_file(path),
        }
    }

    /// Auto-detect a path: an existing directory → [`AssetSource::dir`]; a file
    /// with a pack extension (`.rlxp` / `.rlxpack` / `.pack`) → packed provider;
    /// otherwise treat it as a directory (the historical default). Pack parse
    /// errors surface lazily on first read, so this never fails.
    ///
    /// Bare `*.rlx` (bake `RLXBAKE1`) is **not** treated as a sidecar pack —
    /// use [`pack::classify_bundle_file`] / runtime bake loaders for those.
    pub fn from_path(path: impl Into<PathBuf>) -> Self {
        let path = path.into();
        let looks_packed = path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| matches!(e, "rlxp" | "rlxpack" | "pack"));
        if looks_packed || (path.is_file() && !path.is_dir() && !path.extension().is_some_and(|e| e.eq_ignore_ascii_case("rlx"))) {
            Self::new(pack::LazyPackProvider::new(path))
        } else {
            Self::dir(path)
        }
    }

    /// Read one asset's bytes.
    pub fn read(&self, name: &str) -> Result<Cow<'_, [u8]>> {
        let full = self.full_name(name);
        self.provider
            .read_bytes(&full)
            .with_context(|| self.not_found_context(&full))
    }

    /// Read one asset as owned bytes.
    pub fn read_vec(&self, name: &str) -> Result<Vec<u8>> {
        Ok(self.read(name)?.into_owned())
    }

    /// Read one asset as a UTF-8 string.
    pub fn read_to_string(&self, name: &str) -> Result<String> {
        let bytes = self.read(name)?;
        String::from_utf8(bytes.into_owned())
            .with_context(|| format!("asset {name:?} is not valid UTF-8"))
    }

    /// Read + parse one asset as JSON.
    pub fn read_json<T: serde::de::DeserializeOwned>(&self, name: &str) -> Result<T> {
        let s = self.read_to_string(name)?;
        serde_json::from_str(&s).with_context(|| format!("parse asset {name:?} as JSON"))
    }

    /// Whether the named asset exists.
    pub fn exists(&self, name: &str) -> bool {
        self.provider.exists(&self.full_name(name))
    }

    /// All asset names (already scoped to this source's prefix).
    pub fn names(&self) -> Result<Vec<String>> {
        let all = self.provider.names()?;
        if self.prefix.is_empty() {
            return Ok(all);
        }
        let pfx = format!("{}/", self.prefix);
        Ok(all
            .into_iter()
            .filter_map(|n| n.strip_prefix(&pfx).map(str::to_string))
            .collect())
    }

    /// A source scoped to a sub-prefix (e.g. `src.subdir("frontend")`), so a
    /// sub-loader reads `tokenizer.json` and gets `frontend/tokenizer.json`.
    pub fn subdir(&self, prefix: &str) -> AssetSource {
        let prefix = normalize(prefix);
        let combined = if self.prefix.is_empty() {
            prefix
        } else if prefix.is_empty() {
            self.prefix.clone()
        } else {
            format!("{}/{}", self.prefix, prefix)
        };
        AssetSource {
            provider: self.provider.clone(),
            prefix: combined,
        }
    }

    /// The backing directory for this source (scoped by prefix), if it is
    /// directory-backed. `None` for pack / memory / custom sources.
    pub fn dir_path(&self) -> Option<PathBuf> {
        self.provider.as_dir().map(|d| {
            if self.prefix.is_empty() {
                d.to_path_buf()
            } else {
                d.join(&self.prefix)
            }
        })
    }

    /// Return a real local directory for a subtree, for sub-loaders that require
    /// a `&Path` (tokenizers, the g2p frontend). Directory-backed sources return
    /// the real path with zero copies; every other source materializes the
    /// subtree into a self-cleaning temp directory (deleted when the returned
    /// [`LocalDir`] drops).
    ///
    /// `prefix` selects a subtree (e.g. `Some("frontend")`); `None` is the whole
    /// bundle.
    pub fn local_dir(&self, prefix: Option<&str>) -> Result<LocalDir> {
        let scoped = match prefix {
            Some(p) => self.subdir(p),
            None => self.clone(),
        };
        if let Some(dir) = scoped.dir_path() {
            return Ok(LocalDir::Real(dir));
        }
        // Materialize the subtree to a temp dir.
        let root = temp_dir_unique();
        std::fs::create_dir_all(&root)
            .with_context(|| format!("create temp asset dir {}", root.display()))?;
        let guard = TempDirGuard(root.clone());
        for name in scoped.names()? {
            let bytes = scoped.read(&name)?;
            let dest = root.join(&name);
            if let Some(parent) = dest.parent() {
                std::fs::create_dir_all(parent).ok();
            }
            std::fs::write(&dest, &bytes)
                .with_context(|| format!("materialize asset {name:?} → {}", dest.display()))?;
        }
        Ok(LocalDir::Temp(guard))
    }

    fn full_name(&self, name: &str) -> String {
        let name = normalize(name);
        if self.prefix.is_empty() {
            name
        } else {
            format!("{}/{}", self.prefix, name)
        }
    }

    fn not_found_context(&self, full: &str) -> String {
        match self.provider.names() {
            Ok(mut names) if !names.contains(&full.to_string()) => {
                names.sort();
                names.truncate(24);
                format!("asset {full:?} not found; available: {names:?}")
            }
            _ => format!("read asset {full:?}"),
        }
    }
}

/// Adopt versatile loading in one line. A model keeps its existing
/// `load_from_dir(&Path)` loader and writes:
///
/// ```ignore
/// pub fn load(src: impl Into<AssetSource>) -> Result<Self> {
///     let (mut m, keep) = load_materialized(src, Self::load_from_dir)?;
///     m.assets = keep; // stash the guard if the model reads files lazily
///     Ok(m)
/// }
/// ```
///
/// giving it dir / packed-file / in-memory / spec / custom-provider loading for
/// free. `keep` is `Some(guard)` only when the source had to be materialized to
/// a temp directory (pack / memory); models that read every file eagerly during
/// `load_from_dir` can ignore it (the temp dir is cleaned when it drops).
pub fn load_materialized<T>(
    src: impl Into<AssetSource>,
    from_dir: impl FnOnce(&Path) -> Result<T>,
) -> Result<(T, Option<LocalDir>)> {
    let local = src.into().local_dir(None)?;
    let model = from_dir(local.path())?;
    let keep = matches!(local, LocalDir::Temp(_)).then_some(local);
    Ok((model, keep))
}

impl From<PathBuf> for AssetSource {
    fn from(p: PathBuf) -> Self {
        AssetSource::from_path(p)
    }
}
impl From<&Path> for AssetSource {
    fn from(p: &Path) -> Self {
        AssetSource::from_path(p.to_path_buf())
    }
}
impl From<&PathBuf> for AssetSource {
    fn from(p: &PathBuf) -> Self {
        AssetSource::from_path(p.clone())
    }
}
impl From<&str> for AssetSource {
    fn from(p: &str) -> Self {
        AssetSource::from_path(PathBuf::from(p))
    }
}
impl From<String> for AssetSource {
    fn from(p: String) -> Self {
        AssetSource::from_path(PathBuf::from(p))
    }
}
impl From<HashMap<String, Vec<u8>>> for AssetSource {
    fn from(m: HashMap<String, Vec<u8>>) -> Self {
        AssetSource::memory(m)
    }
}
impl From<&AssetSource> for AssetSource {
    fn from(s: &AssetSource) -> Self {
        s.clone()
    }
}

/// Declarative source description for config files. Deserializes from
/// `{"source":"dir","path":"…"}` or `{"source":"pack","path":"…"}`.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "source", rename_all = "snake_case")]
pub enum SourceSpec {
    Dir { path: PathBuf },
    Pack { path: PathBuf },
}

// ── Directory provider ──────────────────────────────────────────────────────

struct DirProvider {
    root: PathBuf,
}

impl AssetProvider for DirProvider {
    fn read_bytes(&self, name: &str) -> Result<Cow<'_, [u8]>> {
        let path = self.root.join(name);
        Ok(Cow::Owned(
            std::fs::read(&path).with_context(|| format!("read {}", path.display()))?,
        ))
    }
    fn exists(&self, name: &str) -> bool {
        self.root.join(name).exists()
    }
    fn names(&self) -> Result<Vec<String>> {
        let mut out = Vec::new();
        walk_dir(&self.root, &self.root, &mut out)?;
        Ok(out)
    }
    fn as_dir(&self) -> Option<&Path> {
        Some(&self.root)
    }
}

fn walk_dir(base: &Path, dir: &Path, out: &mut Vec<String>) -> Result<()> {
    let rd = match std::fs::read_dir(dir) {
        Ok(rd) => rd,
        Err(_) => return Ok(()), // absent dir → no assets, not an error
    };
    for entry in rd {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            walk_dir(base, &path, out)?;
        } else if let Ok(rel) = path.strip_prefix(base) {
            out.push(rel.to_string_lossy().replace('\\', "/"));
        }
    }
    Ok(())
}

// ── In-memory provider ──────────────────────────────────────────────────────

struct MemoryProvider {
    map: HashMap<String, Vec<u8>>,
}

impl AssetProvider for MemoryProvider {
    fn read_bytes(&self, name: &str) -> Result<Cow<'_, [u8]>> {
        self.map
            .get(name)
            .map(|v| Cow::Borrowed(v.as_slice()))
            .ok_or_else(|| anyhow!("asset {name:?} not in memory bundle"))
    }
    fn exists(&self, name: &str) -> bool {
        self.map.contains_key(name)
    }
    fn names(&self) -> Result<Vec<String>> {
        Ok(self.map.keys().cloned().collect())
    }
}

// ── Temp-dir materialization guard ──────────────────────────────────────────

/// A real local directory for a bundle subtree — either the source's own
/// directory or a temp directory that is deleted when this value drops. Hand
/// [`LocalDir::path`] to any sub-loader that needs a `&Path`.
pub enum LocalDir {
    /// The source is directory-backed; this is its real path (no cleanup).
    Real(PathBuf),
    /// A temp directory holding the materialized subtree (auto-deleted on drop).
    Temp(TempDirGuard),
}

impl LocalDir {
    pub fn path(&self) -> &Path {
        match self {
            LocalDir::Real(p) => p,
            LocalDir::Temp(g) => &g.0,
        }
    }
}

/// Deletes its directory on drop.
pub struct TempDirGuard(PathBuf);
impl Drop for TempDirGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn temp_dir_unique() -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("rlx-asset-{}-{}", std::process::id(), n))
}

// ── Pack containers (`.rlxp` + legacy `.rlxpack`) ───────────────────────────

/// Pack / unpack model asset trees.
///
/// Prefer the official [`.rlxp`](https://github.com/MIT-RLX/rlx/blob/main/docs/rlxp.md)
/// flat package (`RLXPFLAT`, cold-zstd sidecars) when the `rlxp` feature is on.
/// Legacy `RLXPACK1` (`.rlxpack`) remains readable for older Hub dumps.
pub mod pack {
    use super::*;
    use std::io::Write;

    const MAGIC_RLXPACK: &[u8; 8] = b"RLXPACK1";
    #[cfg(feature = "rlxp")]
    const MAGIC_RLXPFLAT: &[u8; 8] = b"RLXPFLAT";

    /// Open a packed file: `.rlxp` (feature `rlxp`) or legacy `.rlxpack`.
    pub fn open_pack_provider(path: &Path) -> Result<Box<dyn AssetProvider>> {
        #[cfg(feature = "rlxp")]
        {
            if looks_rlxp_path(path) || peek_magic(path)? == Some(*MAGIC_RLXPFLAT) {
                return Ok(Box::new(RlxpProvider::open(path)?));
            }
        }
        Ok(Box::new(PackProvider::open(path)?))
    }

    /// Open packed bytes (magic-dispatched).
    pub fn open_pack_bytes(bytes: Arc<[u8]>) -> Result<Box<dyn AssetProvider>> {
        #[cfg(feature = "rlxp")]
        {
            if bytes.len() >= 8 && &bytes[..8] == MAGIC_RLXPFLAT {
                return Ok(Box::new(RlxpProvider::from_bytes(bytes)?));
            }
        }
        Ok(Box::new(PackProvider::from_bytes(bytes)?))
    }

    fn peek_magic(path: &Path) -> Result<Option<[u8; 8]>> {
        use std::io::Read;
        let mut f = std::fs::File::open(path)
            .with_context(|| format!("open {}", path.display()))?;
        let mut buf = [0u8; 8];
        match f.read(&mut buf) {
            Ok(8) => Ok(Some(buf)),
            Ok(_) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    fn looks_rlxp_path(path: &Path) -> bool {
        path.extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case("rlxp"))
    }

    fn media_type_for(name: &str) -> String {
        match Path::new(name)
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_ascii_lowercase()
            .as_str()
        {
            "json" => "application/json".into(),
            "onnx" => "application/octet-stream".into(),
            "txt" | "md" | "cfg" => "text/plain".into(),
            "safetensors" => "application/octet-stream".into(),
            "data" => "application/octet-stream".into(),
            _ => "application/octet-stream".into(),
        }
    }

    /// Pack every file under `dir` into `out`.
    ///
    /// With feature `rlxp`, writes an official flat `.rlxp` (sidecar-only,
    /// `include_graph = false`). Without it, writes legacy `.rlxpack`.
    pub fn write_dir(dir: impl AsRef<Path>, out: impl AsRef<Path>) -> Result<()> {
        #[cfg(feature = "rlxp")]
        {
            return write_dir_rlxp(dir, out);
        }
        #[cfg(not(feature = "rlxp"))]
        {
            write_dir_rlxpack(dir, out)
        }
    }

    /// Always write legacy `.rlxpack` (even when `rlxp` is enabled).
    pub fn write_dir_rlxpack(dir: impl AsRef<Path>, out: impl AsRef<Path>) -> Result<()> {
        let dir = dir.as_ref();
        let mut names = Vec::new();
        walk_dir(dir, dir, &mut names)?;
        names.sort();
        let mut blobs = Vec::with_capacity(names.len());
        for name in &names {
            blobs.push((name.clone(), std::fs::read(dir.join(name))?));
        }
        let mut f = std::fs::File::create(out.as_ref())
            .with_context(|| format!("create {}", out.as_ref().display()))?;
        write_rlxpack(blobs.iter().map(|(n, b)| (n.clone(), b.as_slice())), &mut f)
    }

    /// Pack `dir` into an official flat `.rlxp` (cold-zstd sidecars, no graph).
    #[cfg(feature = "rlxp")]
    pub fn write_dir_rlxp(dir: impl AsRef<Path>, out: impl AsRef<Path>) -> Result<()> {
        write_named_files_rlxp(collect_dir_files(dir.as_ref())?, out, None)
    }

    /// Pack an explicit `(relative_path, bytes)` list into a flat `.rlxp`.
    #[cfg(feature = "rlxp")]
    pub fn write_named_files_rlxp(
        files: Vec<(String, Vec<u8>)>,
        out: impl AsRef<Path>,
        package_name: Option<&str>,
    ) -> Result<()> {
        use rlx_ir::Graph;
        use rlx_pkg::{ContainerKind, WriteOptions, write_package};

        let out = out.as_ref();
        let mut sidecars = Vec::with_capacity(files.len());
        for (name, bytes) in files {
            sidecars.push((name.clone(), media_type_for(&name), bytes));
        }
        let name = package_name
            .map(str::to_string)
            .unwrap_or_else(|| {
                out.file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("assets")
                    .to_string()
            });
        let opts = WriteOptions {
            name,
            producer: Some("rlx-assets".into()),
            features: vec!["asset_bundle".into()],
            container: ContainerKind::Flat,
            sidecars,
            include_graph: false,
            compress_sidecars: true,
            ..WriteOptions::default()
        };
        let graph = Graph::new("assets");
        write_package(out, &graph, &[], &opts)
            .with_context(|| format!("write rlxp {}", out.display()))?;
        Ok(())
    }

    /// Collect every file under `dir` as `(relpath, bytes)`.
    #[cfg(feature = "rlxp")]
    pub fn collect_dir_files(dir: &Path) -> Result<Vec<(String, Vec<u8>)>> {
        let mut names = Vec::new();
        walk_dir(dir, dir, &mut names)?;
        names.sort();
        let mut out = Vec::with_capacity(names.len());
        for name in names {
            let bytes = std::fs::read(dir.join(&name))
                .with_context(|| format!("read {}/{}", dir.display(), name))?;
            out.push((name, bytes));
        }
        Ok(out)
    }

    /// Materialize all sidecars from an `.rlxp` into `extract_dir` (persistent cache).
    #[cfg(feature = "rlxp")]
    pub fn materialize_rlxp(rlxp_path: &Path, extract_dir: &Path) -> Result<()> {
        let marker = extract_dir.join(".rlxp_ok");
        if marker.is_file() {
            return Ok(());
        }
        if extract_dir.exists() {
            std::fs::remove_dir_all(extract_dir)
                .with_context(|| format!("clear stale extract {}", extract_dir.display()))?;
        }
        std::fs::create_dir_all(extract_dir)?;
        let pack = rlx_pkg::Package::open(rlxp_path)
            .with_context(|| format!("open rlxp {}", rlxp_path.display()))?;
        for sc in &pack.manifest().sidecars {
            let bytes = pack.sidecar(&sc.id)?;
            let path = extract_dir.join(&sc.id);
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&path, bytes)
                .with_context(|| format!("write {}", path.display()))?;
        }
        std::fs::write(&marker, b"ok")?;
        Ok(())
    }

    /// Hashed extract dir beside a pack file (same idea as GGUF materialize caches).
    pub fn extract_dir_for(pack_path: &Path, prefix: &str) -> PathBuf {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut h = DefaultHasher::new();
        pack_path.hash(&mut h);
        if let Ok(meta) = std::fs::metadata(pack_path) {
            meta.len().hash(&mut h);
            if let Ok(modified) = meta.modified() {
                modified.hash(&mut h);
            }
        }
        let stem = pack_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("pack");
        std::env::temp_dir().join(format!("{prefix}-{stem}-{:x}", h.finish()))
    }

    /// Options for [`resolve_bundle_path`] / [`resolve_bundle_load_path`].
    #[derive(Debug, Clone, Default)]
    pub struct ResolveBundleOpts<'a> {
        /// Preferred pack filename inside a directory (e.g. `"rlx-tts.rlxp"`).
        pub default_rlxp: Option<&'a str>,
        /// Preferred GGUF filename inside a directory (e.g. `"rlx-tts.gguf"`).
        pub default_gguf: Option<&'a str>,
        /// Temp-dir prefix when extracting a pack (e.g. `"rlx-tts-rlxp"`).
        pub extract_prefix: Option<&'a str>,
    }

    /// Classified bundle location before optional extraction.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub enum ResolvedBundlePath {
        /// `.rlxp` / `.rlxpack` / `.pack` sidecar or weight pack file.
        Pack(PathBuf),
        /// Bake artifact (`RLXBAKE1` / encrypted `RLXENC01`, usually `*.rlx`).
        Bake(PathBuf),
        /// Single `.gguf` archive.
        Gguf(PathBuf),
        /// Loose on-disk directory bundle.
        Directory(PathBuf),
    }

    /// Path ready for directory- or GGUF-based loaders (packs are extracted).
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub enum BundleLoadPath {
        Directory(PathBuf),
        Gguf(PathBuf),
    }

    /// Whether `path` looks like an extractable sidecar pack by extension.
    ///
    /// Note: bare `*.rlx` is **not** included — that extension is used by bake
    /// (`RLXBAKE1`). Use [`classify_bundle_file`] which sniffs magic.
    pub fn is_pack_extension(path: &Path) -> bool {
        path.extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| {
                matches!(
                    e.to_ascii_lowercase().as_str(),
                    "rlxp" | "rlxpack" | "pack"
                )
            })
    }

    fn is_gguf_extension(path: &Path) -> bool {
        path.extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case("gguf"))
    }

    fn is_rlx_extension(path: &Path) -> bool {
        path.extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case("rlx"))
    }

    /// True when `path` is an official `.rlxp` (extension or `RLXPFLAT` magic).
    pub fn is_rlxp_file(path: &Path) -> bool {
        if path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case("rlxp"))
        {
            return true;
        }
        #[cfg(feature = "rlxp")]
        {
            return peek_magic(path)
                .ok()
                .flatten()
                .is_some_and(|m| m == *MAGIC_RLXPFLAT);
        }
        #[cfg(not(feature = "rlxp"))]
        {
            false
        }
    }

    /// True when `path` is a bake artifact (`RLXBAKE1` / `RLXENC01`, usually `*.rlx`).
    pub fn is_bake_file(path: &Path) -> bool {
        match peek_magic(path).ok().flatten() {
            Some(m) if &m == b"RLXBAKE1" || &m == b"RLXENC01" => true,
            _ => is_rlx_extension(path) && !is_rlxp_file(path) && !is_rlxpack_magic(path),
        }
    }

    fn is_rlxpack_magic(path: &Path) -> bool {
        peek_magic(path)
            .ok()
            .flatten()
            .is_some_and(|m| m == *MAGIC_RLXPACK)
    }

    /// Classify a single on-disk file (extension + magic).
    pub fn classify_bundle_file(path: &Path) -> Result<ResolvedBundlePath> {
        if is_gguf_extension(path) {
            return Ok(ResolvedBundlePath::Gguf(path.to_path_buf()));
        }
        let magic = peek_magic(path)?;
        #[cfg(feature = "rlxp")]
        {
            if looks_rlxp_path(path)
                || magic == Some(*MAGIC_RLXPFLAT)
            {
                return Ok(ResolvedBundlePath::Pack(path.to_path_buf()));
            }
        }
        #[cfg(not(feature = "rlxp"))]
        {
            if looks_rlxp_path(path) {
                bail!(
                    "{} looks like .rlxp but the rlx-assets `rlxp` feature is disabled",
                    path.display()
                );
            }
        }
        if magic == Some(*MAGIC_RLXPACK)
            || path
                .extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| {
                    matches!(e.to_ascii_lowercase().as_str(), "rlxpack" | "pack")
                })
        {
            return Ok(ResolvedBundlePath::Pack(path.to_path_buf()));
        }
        if magic == Some(*b"RLXBAKE1") || magic == Some(*b"RLXENC01") || is_rlx_extension(path) {
            return Ok(ResolvedBundlePath::Bake(path.to_path_buf()));
        }
        if is_pack_extension(path) {
            return Ok(ResolvedBundlePath::Pack(path.to_path_buf()));
        }
        bail!("bundle file has unknown extension: {}", path.display())
    }

    fn find_in_dir(dir: &Path, default_name: Option<&str>, extensions: &[&str]) -> Option<PathBuf> {
        if let Some(name) = default_name {
            let cand = dir.join(name);
            if cand.is_file() {
                return Some(cand);
            }
        }
        // Prefer earlier extensions in `extensions` (e.g. rlxp before rlxpack).
        for ext in extensions {
            let mut matches = Vec::new();
            let entries = std::fs::read_dir(dir).ok()?;
            for entry in entries.flatten() {
                let path = entry.path();
                if !path.is_file() {
                    continue;
                }
                if path
                    .extension()
                    .and_then(|e| e.to_str())
                    .is_some_and(|e| e.eq_ignore_ascii_case(ext))
                {
                    matches.push(path);
                }
            }
            matches.sort();
            if let Some(p) = matches.into_iter().next() {
                return Some(p);
            }
        }
        None
    }

    /// Resolve a user-supplied model path without extracting packs.
    ///
    /// **Preference order** when `path` is a **directory**:
    /// 1. Pack file — named [`ResolveBundleOpts::default_rlxp`] if present, else the
    ///    first `*.rlxp`, then `*.rlxpack`, then `*.pack`
    /// 2. Bake file — first `*.rlx` (`RLXBAKE1` / encrypted)
    /// 3. GGUF — named [`ResolveBundleOpts::default_gguf`] if present, else the
    ///    first `*.gguf`
    /// 4. Loose directory (unchanged)
    ///
    /// When `path` is a **file**: magic/extension → [`ResolvedBundlePath`].
    pub fn resolve_bundle_path(path: &Path, opts: &ResolveBundleOpts<'_>) -> Result<ResolvedBundlePath> {
        if path.is_file() {
            return classify_bundle_file(path);
        }
        if path.is_dir() {
            const PACK_EXTS: &[&str] = &["rlxp", "rlxpack", "pack"];
            if let Some(pack) = find_in_dir(path, opts.default_rlxp, PACK_EXTS) {
                return Ok(ResolvedBundlePath::Pack(pack));
            }
            // `*.rlx` may be bake or (rarely) a misnamed pack — sniff magic.
            if let Some(rlx) = find_in_dir(path, None, &["rlx"]) {
                return classify_bundle_file(&rlx);
            }
            if let Some(gguf) = find_in_dir(path, opts.default_gguf, &["gguf"]) {
                return Ok(ResolvedBundlePath::Gguf(gguf));
            }
            return Ok(ResolvedBundlePath::Directory(path.to_path_buf()));
        }
        bail!("bundle path not found: {}", path.display())
    }

    /// Like [`resolve_bundle_path`], but materializes pack files to a persistent
    /// cache directory (see [`extract_dir_for`]) and returns a load-ready path.
    ///
    /// Bake (`.rlx`) artifacts are not extractable sidecars — callers that only
    /// understand loose dirs / GGUF should handle [`ResolvedBundlePath::Bake`]
    /// via [`resolve_bundle_path`] instead.
    pub fn resolve_bundle_load_path(
        path: &Path,
        opts: &ResolveBundleOpts<'_>,
    ) -> Result<BundleLoadPath> {
        match resolve_bundle_path(path, opts)? {
            ResolvedBundlePath::Pack(p) => {
                let prefix = opts.extract_prefix.unwrap_or("rlx-pack");
                let dir = extract_pack_to_dir(&p, prefix)?;
                Ok(BundleLoadPath::Directory(dir))
            }
            ResolvedBundlePath::Bake(p) => bail!(
                "bake artifact {} (.rlx / RLXBAKE1) cannot be materialized as a sidecar pack; \
                 convert with `rlx-bake convert … -o model.rlxp` or pass a .rlxp/.gguf/dir",
                p.display()
            ),
            ResolvedBundlePath::Gguf(p) => Ok(BundleLoadPath::Gguf(p)),
            ResolvedBundlePath::Directory(d) => Ok(BundleLoadPath::Directory(d)),
        }
    }

    /// Extract any supported pack (`.rlxp`, `.rlxpack`, …) into `extract_dir`.
    pub fn materialize_pack(pack_path: &Path, extract_dir: &Path) -> Result<()> {
        #[cfg(feature = "rlxp")]
        if is_rlxp_file(pack_path) {
            return materialize_rlxp(pack_path, extract_dir);
        }
        materialize_rlxpack(pack_path, extract_dir)
    }

    /// Alias for [`materialize_rlxp`] (extract sidecars to a directory).
    #[cfg(feature = "rlxp")]
    pub fn extract_rlxp_to(rlxp_path: &Path, extract_dir: &Path) -> Result<()> {
        materialize_rlxp(rlxp_path, extract_dir)
    }

    /// Materialize every entry from a legacy `.rlxpack` into `extract_dir`.
    pub fn materialize_rlxpack(rlxpack_path: &Path, extract_dir: &Path) -> Result<()> {
        let marker = extract_dir.join(".rlxpack_ok");
        if marker.is_file() {
            return Ok(());
        }
        if extract_dir.exists() {
            std::fs::remove_dir_all(extract_dir)
                .with_context(|| format!("clear stale extract {}", extract_dir.display()))?;
        }
        std::fs::create_dir_all(extract_dir)?;
        let provider = PackProvider::open(rlxpack_path)?;
        for name in provider.names()? {
            let bytes = provider.read_bytes(&name)?;
            let path = extract_dir.join(&name);
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&path, &bytes)
                .with_context(|| format!("write {}", path.display()))?;
        }
        std::fs::write(&marker, b"ok")?;
        Ok(())
    }

    /// Extract a pack beside its source (content-addressed temp dir) and return that dir.
    pub fn extract_pack_to_dir(pack_path: &Path, prefix: &str) -> Result<PathBuf> {
        let extract_dir = extract_dir_for(pack_path, prefix);
        materialize_pack(pack_path, &extract_dir)
            .with_context(|| format!("extract pack {}", pack_path.display()))?;
        Ok(extract_dir)
    }

    /// Serialize `(name, bytes)` entries into a legacy `.rlxpack`.
    pub fn write<'a>(
        entries: impl IntoIterator<Item = (String, &'a [u8])>,
        w: &mut impl Write,
    ) -> Result<()> {
        write_rlxpack(entries, w)
    }

    fn write_rlxpack<'a>(
        entries: impl IntoIterator<Item = (String, &'a [u8])>,
        w: &mut impl Write,
    ) -> Result<()> {
        let mut files = std::collections::BTreeMap::new();
        let mut data = Vec::new();
        for (name, bytes) in entries {
            let off = data.len() as u64;
            data.extend_from_slice(bytes);
            files.insert(normalize(&name), (off, bytes.len() as u64));
        }
        let header = serde_json::to_vec(&Header { files })?;
        w.write_all(MAGIC_RLXPACK)?;
        w.write_all(&(header.len() as u64).to_le_bytes())?;
        w.write_all(&header)?;
        w.write_all(&data)?;
        Ok(())
    }

    #[derive(serde::Serialize, serde::Deserialize)]
    struct Header {
        files: std::collections::BTreeMap<String, (u64, u64)>,
    }

    fn parse_rlxpack_header(bytes: &[u8]) -> Result<(Header, usize)> {
        if bytes.len() < 16 || &bytes[..8] != MAGIC_RLXPACK {
            bail!("not an rlxpack (bad magic)");
        }
        let hdr_len = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
        let hdr_end = 16 + hdr_len;
        let hdr = bytes
            .get(16..hdr_end)
            .ok_or_else(|| anyhow!("rlxpack header truncated"))?;
        let header: Header = serde_json::from_slice(hdr).context("parse rlxpack header")?;
        Ok((header, hdr_end))
    }

    enum Backing {
        Mmap(memmap2::Mmap),
        Bytes(Arc<[u8]>),
    }
    impl Backing {
        fn as_slice(&self) -> &[u8] {
            match self {
                Backing::Mmap(m) => m,
                Backing::Bytes(b) => b,
            }
        }
    }

    /// Legacy `.rlxpack` provider.
    pub struct PackProvider {
        backing: Backing,
        files: std::collections::BTreeMap<String, (u64, u64)>,
        data_start: usize,
    }

    impl PackProvider {
        pub fn open(path: &Path) -> Result<Self> {
            let file =
                std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
            // SAFETY: the pack file is treated as immutable for the provider's life.
            let mmap = unsafe { memmap2::Mmap::map(&file) }
                .with_context(|| format!("mmap {}", path.display()))?;
            let (header, data_start) = parse_rlxpack_header(&mmap)?;
            Ok(Self {
                backing: Backing::Mmap(mmap),
                files: header.files,
                data_start,
            })
        }

        pub fn from_bytes(bytes: Arc<[u8]>) -> Result<Self> {
            let (header, data_start) = parse_rlxpack_header(&bytes)?;
            Ok(Self {
                backing: Backing::Bytes(bytes),
                files: header.files,
                data_start,
            })
        }

        fn slice(&self, name: &str) -> Option<&[u8]> {
            let (off, len) = *self.files.get(name)?;
            let start = self.data_start + off as usize;
            self.backing.as_slice().get(start..start + len as usize)
        }
    }

    impl AssetProvider for PackProvider {
        fn read_bytes(&self, name: &str) -> Result<Cow<'_, [u8]>> {
            self.slice(name)
                .map(Cow::Borrowed)
                .ok_or_else(|| anyhow!("asset {name:?} not in pack"))
        }
        fn exists(&self, name: &str) -> bool {
            self.files.contains_key(name)
        }
        fn names(&self) -> Result<Vec<String>> {
            Ok(self.files.keys().cloned().collect())
        }
    }

    /// Official `.rlxp` provider (sidecars as named assets).
    #[cfg(feature = "rlxp")]
    pub struct RlxpProvider {
        pack: rlx_pkg::Package,
        /// When opened from in-memory bytes, keep the temp file alive.
        _tmp: Option<TmpRlxpFile>,
    }

    #[cfg(feature = "rlxp")]
    struct TmpRlxpFile(PathBuf);
    #[cfg(feature = "rlxp")]
    impl Drop for TmpRlxpFile {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    #[cfg(feature = "rlxp")]
    impl RlxpProvider {
        pub fn open(path: &Path) -> Result<Self> {
            let pack = rlx_pkg::Package::open(path)
                .with_context(|| format!("open rlxp {}", path.display()))?;
            Ok(Self { pack, _tmp: None })
        }

        pub fn from_bytes(bytes: Arc<[u8]>) -> Result<Self> {
            // Package::open needs a path; materialize a short-lived file.
            let path = super::temp_dir_unique().with_extension("rlxp");
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&path, &bytes)
                .with_context(|| format!("write temp rlxp {}", path.display()))?;
            let pack = rlx_pkg::Package::open(&path).context("open in-memory rlxp")?;
            Ok(Self {
                pack,
                _tmp: Some(TmpRlxpFile(path)),
            })
        }
    }

    #[cfg(feature = "rlxp")]
    impl AssetProvider for RlxpProvider {
        fn read_bytes(&self, name: &str) -> Result<Cow<'_, [u8]>> {
            let bytes = self
                .pack
                .sidecar(name)
                .with_context(|| format!("sidecar {name:?}"))?;
            Ok(Cow::Owned(bytes))
        }
        fn exists(&self, name: &str) -> bool {
            self.pack
                .manifest()
                .sidecars
                .iter()
                .any(|s| s.id == name)
        }
        fn names(&self) -> Result<Vec<String>> {
            Ok(self
                .pack
                .manifest()
                .sidecars
                .iter()
                .map(|s| s.id.clone())
                .collect())
        }
    }

    /// Lazy path → pack; opens `.rlxp` or `.rlxpack` on first use.
    pub struct LazyPackProvider {
        path: PathBuf,
        inner: std::sync::OnceLock<Result<Box<dyn AssetProvider>, String>>,
    }
    impl LazyPackProvider {
        pub fn new(path: PathBuf) -> Self {
            Self {
                path,
                inner: std::sync::OnceLock::new(),
            }
        }
        fn get(&self) -> Result<&dyn AssetProvider> {
            self.inner
                .get_or_init(|| open_pack_provider(&self.path).map_err(|e| format!("{e:#}")))
                .as_ref()
                .map(|p| p.as_ref())
                .map_err(|e| anyhow!("{e}"))
        }
    }
    impl AssetProvider for LazyPackProvider {
        fn read_bytes(&self, name: &str) -> Result<Cow<'_, [u8]>> {
            self.get()?.read_bytes(name)
        }
        fn exists(&self, name: &str) -> bool {
            self.get().map(|p| p.exists(name)).unwrap_or(false)
        }
        fn names(&self) -> Result<Vec<String>> {
            self.get()?.names()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> HashMap<String, Vec<u8>> {
        let mut m = HashMap::new();
        m.insert("config.json".into(), br#"{"model":"x"}"#.to_vec());
        m.insert("onnx/decoder.onnx".into(), vec![1, 2, 3, 4]);
        m.insert("frontend/vocab.txt".into(), b"a\nb\n".to_vec());
        m
    }

    #[test]
    fn memory_source_reads_and_lists() {
        let src = AssetSource::memory(sample());
        assert!(src.exists("config.json"));
        assert!(!src.exists("missing"));
        assert_eq!(&*src.read("onnx/decoder.onnx").unwrap(), &[1, 2, 3, 4]);
        assert_eq!(src.read_to_string("frontend/vocab.txt").unwrap(), "a\nb\n");
        let mut names = src.names().unwrap();
        names.sort();
        assert_eq!(
            names,
            ["config.json", "frontend/vocab.txt", "onnx/decoder.onnx"]
        );
        assert!(src.dir_path().is_none());
    }

    #[test]
    fn subdir_scopes_reads_and_names() {
        let src = AssetSource::memory(sample());
        let fe = src.subdir("frontend");
        assert_eq!(fe.read_to_string("vocab.txt").unwrap(), "a\nb\n");
        assert_eq!(fe.names().unwrap(), ["vocab.txt"]);
    }

    #[test]
    fn pack_round_trips_in_memory() {
        let entries = sample();
        let mut buf = Vec::new();
        pack::write(
            entries.iter().map(|(k, v)| (k.clone(), v.as_slice())),
            &mut buf,
        )
        .unwrap();
        let src = AssetSource::pack_bytes(Arc::<[u8]>::from(buf)).unwrap();
        assert_eq!(&*src.read("onnx/decoder.onnx").unwrap(), &[1, 2, 3, 4]);
        assert_eq!(
            src.read_to_string("config.json").unwrap(),
            r#"{"model":"x"}"#
        );
        assert!(src.exists("frontend/vocab.txt"));
        assert_eq!(src.names().unwrap().len(), 3);
    }

    #[test]
    fn missing_asset_lists_available() {
        let src = AssetSource::memory(sample());
        let err = format!("{:#}", src.read("nope.bin").unwrap_err());
        assert!(err.contains("nope.bin"), "{err}");
    }

    #[test]
    fn materialize_writes_subtree_to_temp_dir() {
        let src = AssetSource::memory(sample());
        let local = src.local_dir(Some("frontend")).unwrap();
        assert!(matches!(local, LocalDir::Temp(_)));
        let p = local.path().join("vocab.txt");
        assert_eq!(std::fs::read_to_string(&p).unwrap(), "a\nb\n");
        let dir = local.path().to_path_buf();
        drop(local);
        assert!(!dir.exists(), "temp dir should be cleaned on drop");
    }

    #[test]
    fn dir_source_is_dir_backed() {
        let tmp = temp_dir_unique();
        std::fs::create_dir_all(tmp.join("onnx")).unwrap();
        std::fs::write(tmp.join("config.json"), b"{}").unwrap();
        std::fs::write(tmp.join("onnx/a.onnx"), b"xyz").unwrap();
        let src = AssetSource::dir(&tmp);
        assert_eq!(src.dir_path().as_deref(), Some(tmp.as_path()));
        assert_eq!(&*src.read("onnx/a.onnx").unwrap(), b"xyz");
        // dir-backed local_dir returns the real path (no temp copy).
        assert!(matches!(src.local_dir(None).unwrap(), LocalDir::Real(_)));
        std::fs::remove_dir_all(&tmp).ok();
    }
}
