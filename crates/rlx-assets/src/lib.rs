//! Versatile model-asset loading.
//!
//! A model "bundle" is a set of named assets — a `config.json`, one or more
//! weight files (safetensors / ONNX / GGUF), a tokenizer, frontend data, etc.
//! Historically every crate loaded these with `std::fs::read(dir.join(name))`,
//! which hard-wires a filesystem directory. [`AssetSource`] abstracts *where the
//! bytes come from* so the exact same model can be loaded from:
//!
//! * a **directory** — `AssetSource::dir("weights/my-model")`
//! * a single **packed file** — `AssetSource::pack_file("my-model.rlxpack")`
//!   (see [`pack`] for the trivial container format + a `dir → pack` writer)
//! * **in-memory** bytes — `AssetSource::memory(map)` or an in-memory pack
//! * a **config spec** (JSON) — `AssetSource::from_spec(&spec)` for
//!   `{"source":"dir","path":"…"}` / `{"source":"pack","path":"…"}`
//! * a **custom provider** — anything implementing [`AssetProvider`]
//!   (an HTTP cache, an embedded VFS, a zip, …)
//!
//! DX: `AssetSource` is `Clone` + `From<&Path>/From<PathBuf>/From<&str>` (which
//! auto-detects dir vs. `*.rlxpack` file) and `From<HashMap<String,Vec<u8>>>`,
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
    let n = name.trim_start_matches('/').replace('\\', "/");
    n
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

    /// Load assets from an `.rlxpack` container on disk (memory-mapped).
    pub fn pack_file(path: impl AsRef<Path>) -> Result<Self> {
        Ok(Self::new(pack::PackProvider::open(path.as_ref())?))
    }

    /// Load assets from an in-memory `.rlxpack` container.
    pub fn pack_bytes(bytes: impl Into<Arc<[u8]>>) -> Result<Self> {
        Ok(Self::new(pack::PackProvider::from_bytes(bytes.into())?))
    }

    /// Build from a declarative [`SourceSpec`] (e.g. parsed from a config file).
    pub fn from_spec(spec: &SourceSpec) -> Result<Self> {
        match spec {
            SourceSpec::Dir { path } => Ok(Self::dir(path.clone())),
            SourceSpec::Pack { path } => Self::pack_file(path),
        }
    }

    /// Auto-detect a path: an existing directory → [`AssetSource::dir`]; a file
    /// with a pack extension → [`AssetSource::pack_file`]; otherwise treat it as
    /// a directory (the historical default). Pack parse errors surface lazily on
    /// first read, so this never fails.
    pub fn from_path(path: impl Into<PathBuf>) -> Self {
        let path = path.into();
        let looks_packed = path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| matches!(e, "rlxpack" | "pack"));
        if looks_packed || (path.is_file() && !path.is_dir()) {
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

// ── Pack container (`.rlxpack`) ─────────────────────────────────────────────

/// A trivial, dependency-free bundle container: pack a directory of assets into
/// one file (or in-memory blob) and load it with [`AssetSource::pack_file`] /
/// [`AssetSource::pack_bytes`].
///
/// Layout (all integers little-endian):
/// ```text
/// magic     : 8 bytes  = "RLXPACK1"
/// hdr_len   : u64      = byte length of the JSON header
/// header    : JSON     = { "files": { name: [offset, len], … } }
/// data      : bytes    = concatenated asset payloads (offset is from data start)
/// ```
pub mod pack {
    use super::*;
    use std::io::Write;

    const MAGIC: &[u8; 8] = b"RLXPACK1";

    #[derive(serde::Serialize, serde::Deserialize)]
    struct Header {
        files: std::collections::BTreeMap<String, (u64, u64)>,
    }

    /// Serialize `(name, bytes)` entries into a pack, writing to `w`.
    pub fn write<'a>(
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
        w.write_all(MAGIC)?;
        w.write_all(&(header.len() as u64).to_le_bytes())?;
        w.write_all(&header)?;
        w.write_all(&data)?;
        Ok(())
    }

    /// Pack every file under `dir` (recursively) into `out` (`.rlxpack`). A DX
    /// helper so `dir → single file` is one call.
    pub fn write_dir(dir: impl AsRef<Path>, out: impl AsRef<Path>) -> Result<()> {
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
        write(blobs.iter().map(|(n, b)| (n.clone(), b.as_slice())), &mut f)
    }

    fn parse_header(bytes: &[u8]) -> Result<(Header, usize)> {
        if bytes.len() < 16 || &bytes[..8] != MAGIC {
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

    /// An opened pack (memory-mapped file or in-memory bytes).
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
            let (header, data_start) = parse_header(&mmap)?;
            Ok(Self {
                backing: Backing::Mmap(mmap),
                files: header.files,
                data_start,
            })
        }

        pub fn from_bytes(bytes: Arc<[u8]>) -> Result<Self> {
            let (header, data_start) = parse_header(&bytes)?;
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

    /// A pack referenced by path but opened lazily on first use, so
    /// `AssetSource::from(path)` can stay infallible.
    pub struct LazyPackProvider {
        path: PathBuf,
        inner: std::sync::OnceLock<Result<PackProvider, String>>,
    }
    impl LazyPackProvider {
        pub fn new(path: PathBuf) -> Self {
            Self {
                path,
                inner: std::sync::OnceLock::new(),
            }
        }
        fn get(&self) -> Result<&PackProvider> {
            self.inner
                .get_or_init(|| PackProvider::open(&self.path).map_err(|e| format!("{e:#}")))
                .as_ref()
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
