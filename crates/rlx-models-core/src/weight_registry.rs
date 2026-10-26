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

use anyhow::{Context, Result, anyhow, bail};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use crate::gguf_support::{ResolveWeightsOptions, resolve_weights_file_with_options};
use crate::weight_loader::{GgufLoader, MlxLoader, WeightLoader};
use crate::weight_map::{WeightDrainPolicy, WeightMap};

/// True when `dir/config.json` declares an mlx-lm `quantization` block
/// (`{group_size, bits[, mode]}`). This is the mlx-community discriminator;
/// GPTQ/AWQ use `quantization_config` (with `quant_method`) instead and are
/// left to the safetensors path.
pub fn dir_is_mlx_quant(dir: &Path) -> bool {
    let Ok(bytes) = std::fs::read(dir.join("config.json")) else {
        return false;
    };
    let Ok(v) = serde_json::from_slice::<serde_json::Value>(&bytes) else {
        return false;
    };
    v.get("quantization")
        .and_then(|q| q.as_object())
        .is_some_and(|q| q.contains_key("bits") || q.contains_key("group_size"))
}

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
    // A directory holding a single `model.safetensors` and no index. The
    // comment above promises a directory is acceptable and only the sharded
    // case delivered on it; the unsharded one passed the directory down to a
    // reader of files, which fails with "is a directory" several frames from
    // here, naming the directory and no reason. Anyone handed a model
    // directory passes the directory.
    let file = if path.is_dir() {
        dir.join("model.safetensors")
    } else {
        path.to_path_buf()
    };
    // Lazily mapped, like the sharded case: the eager `WeightMap` converted
    // the WHOLE checkpoint to f32 up front (12.5 GB for a 3B bf16 model) before
    // any builder asked for a tensor, and builders that keep weights at their
    // stored width (`WeightStorage`) need the dtype it erased.
    if file.is_file() {
        return Ok(Box::new(
            crate::safetensors_checkpoint::SafetensorsMmapLoader::open_file(&file)?,
        ));
    }
    let path_str = file
        .to_str()
        .ok_or_else(|| anyhow!("non-utf8 path {:?}", file))?;
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
/// Which registered format a weights path means.
///
/// One answer, because there used to be three and they disagreed. A
/// checkpoint directory holding both `model.safetensors` and a quantised
/// `.gguf` of the same weights is two different models as far as the output
/// is concerned, and the format detector preferred GGUF while the loader
/// registry preferred safetensors. A model loaded whole and the same model
/// loaded in blocks could therefore run different weights, and the only
/// symptom was an answer that was plausible and different — which is how
/// this was found, by a parity test that was looking for something else.
///
/// Returns `Err` for a directory holding more than one format: there is no
/// choice to make here that is not a guess at what somebody meant, and
/// guessing is what caused the problem. A directory holding one format, or
/// a path naming a file, resolves as it always did.
///
/// `mlx` is reported for an mlx-community quantised directory, whose
/// `.safetensors` files carry packed `uint32` tensors the plain safetensors
/// reader cannot dequantise.
pub fn resolve_format(path: &Path) -> Result<&'static str> {
    if path.is_dir() {
        if dir_is_mlx_quant(path) {
            return Ok("mlx");
        }
        let mut present: Vec<(&'static str, PathBuf)> = Vec::new();
        for reg in formats() {
            if let Some(f) = first_file_of(path, reg.extensions) {
                present.push((reg.id, f));
            }
        }
        return match present.len() {
            0 => bail!(
                "no weights in {path:?}: nothing with a registered weight \
                 extension. Registered extensions: .{}",
                crate::registered_extensions_hint()
            ),
            1 => Ok(present[0].0),
            _ => {
                let names: Vec<String> = present
                    .iter()
                    .map(|(id, f)| {
                        format!(
                            "{id} ({})",
                            f.file_name().unwrap_or_default().to_string_lossy()
                        )
                    })
                    .collect();
                bail!(
                    "{path:?} holds more than one weight format — {}. These are \
                     different quantisations and decode to different tokens, so \
                     pass the file you mean rather than the directory.",
                    names.join(" and ")
                )
            }
        };
    }
    let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");
    format_for_extension(ext).ok_or_else(|| {
        anyhow!(
            "unsupported weight extension {ext:?} on {path:?}; registered: .{}",
            crate::registered_extensions_hint()
        )
    })
}

/// The first file in `dir` with any of `exts`, in a stable order.
fn first_file_of(dir: &Path, exts: &[&str]) -> Option<PathBuf> {
    let mut found: Vec<PathBuf> = std::fs::read_dir(dir)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .and_then(|s| s.to_str())
                .is_some_and(|e| exts.iter().any(|w| e.eq_ignore_ascii_case(w)))
        })
        .collect();
    found.sort();
    found.into_iter().next()
}

pub fn open_weight_loader(path: &Path) -> Result<Box<dyn WeightLoader>> {
    // What a path means is decided in one place, so that this and
    // `block_weights::weight_files` and the format detector cannot disagree
    // about a directory and load different weights for the same model.
    let format = resolve_format(path)?;
    if format == "mlx" {
        // mlx-community quantised dirs are safetensors dirs whose `.weight`
        // tensors are packed uint32 with `.scales`/`.biases` sidecars — the
        // plain safetensors reader cannot dequantise them.
        let p = path
            .to_str()
            .ok_or_else(|| anyhow!("non-utf8 path {:?}", path))?;
        return Ok(Box::new(MlxLoader::open(p)?));
    }
    if path.is_dir() {
        // `open_safetensors` takes the directory and finds the shards or the
        // single file; the others want the file itself.
        if format == "safetensors" {
            return open_safetensors(path)
                .with_context(|| format!("opening {:?} as format safetensors", path));
        }
        let reg = formats()
            .into_iter()
            .find(|r| r.id == format)
            .ok_or_else(|| anyhow!("format {format} is registered but has no opener"))?;
        let file = first_file_of(path, reg.extensions)
            .ok_or_else(|| anyhow!("no {format} file in {path:?}"))?;
        return (reg.open)(&file).with_context(|| format!("opening {:?} as format {format}", file));
    }

    for reg in formats() {
        if reg.id == format {
            return (reg.open)(path)
                .with_context(|| format!("opening {:?} as format {}", path, reg.id));
        }
    }
    let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");
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

    fn tmpdir(name: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("rlx-resolve-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn one_format_in_a_directory_resolves_to_it() {
        let d = tmpdir("st");
        std::fs::write(d.join("model.safetensors"), b"").unwrap();
        assert_eq!(resolve_format(&d).unwrap(), "safetensors");
        let g = tmpdir("gg");
        std::fs::write(g.join("model.gguf"), b"").unwrap();
        assert_eq!(resolve_format(&g).unwrap(), "gguf");
        let _ = std::fs::remove_dir_all(&d);
        let _ = std::fs::remove_dir_all(&g);
    }

    #[test]
    fn a_file_resolves_by_its_extension() {
        assert_eq!(
            resolve_format(Path::new("/m/model.safetensors")).unwrap(),
            "safetensors"
        );
        assert_eq!(resolve_format(Path::new("/m/model.gguf")).unwrap(), "gguf");
        assert!(resolve_format(Path::new("/m/model.bin")).is_err());
    }

    #[test]
    fn two_formats_in_a_directory_is_an_error_naming_both() {
        // The disagreement this function exists to end: a directory holding
        // `model.safetensors` and a Q4 GGUF of the same weights answered
        // differently depending on which path loaded it, because the
        // detector preferred GGUF and the loader preferred safetensors.
        let d = tmpdir("both");
        std::fs::write(d.join("model.safetensors"), b"").unwrap();
        std::fs::write(d.join("model.gguf"), b"").unwrap();
        let err = resolve_format(&d).unwrap_err().to_string();
        assert!(err.contains("more than one weight format"), "{err}");
        assert!(err.contains("model.safetensors"), "{err}");
        assert!(err.contains("model.gguf"), "{err}");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn the_loader_and_the_block_reader_agree_on_every_directory() {
        // Not that they return the same thing — they return different
        // things — but that neither resolves a directory the other would
        // refuse, and both act on the same format. One policy, asked twice.
        for (name, files) in [
            ("agree-st", vec!["model.safetensors"]),
            ("agree-gg", vec!["model.gguf"]),
            (
                "agree-shards",
                vec![
                    "model-00001-of-00002.safetensors",
                    "model-00002-of-00002.safetensors",
                ],
            ),
            ("agree-both", vec!["model.safetensors", "model.gguf"]),
            ("agree-none", vec!["config.json"]),
        ] {
            let d = tmpdir(name);
            for f in &files {
                std::fs::write(d.join(f), b"").unwrap();
            }
            let resolved = resolve_format(&d);
            let listed = crate::block_weights::weight_files(&d);
            assert_eq!(
                resolved.is_ok(),
                listed.is_ok(),
                "{name}: resolve_format ok={} but weight_files ok={}",
                resolved.is_ok(),
                listed.is_ok()
            );
            if let (Ok(fmt), Ok(fs)) = (&resolved, &listed) {
                let exts: Vec<&str> = list_registered_formats()
                    .into_iter()
                    .find(|r| &r.id == fmt)
                    .map(|r| r.extensions.to_vec())
                    .unwrap_or_default();
                for f in fs {
                    let e = f.extension().and_then(|s| s.to_str()).unwrap_or("");
                    assert!(exts.contains(&e), "{name}: {f:?} is not a {fmt} file");
                }
            }
            let _ = std::fs::remove_dir_all(&d);
        }
    }

    #[test]
    fn format_for_extension_builtin() {
        assert_eq!(format_for_extension("gguf"), Some("gguf"));
        assert_eq!(format_for_extension("safetensors"), Some("safetensors"));
        assert_eq!(format_for_extension("bin"), None);
    }

    /// A safetensors file with one tiny tensor in it.
    fn one_tensor_file() -> Vec<u8> {
        let header = br#"{"t":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#;
        let mut out = (header.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(header);
        out.extend_from_slice(&1.0f32.to_le_bytes());
        out.extend_from_slice(&2.0f32.to_le_bytes());
        out
    }

    #[test]
    fn a_directory_holding_one_safetensors_file_opens_like_the_file_does() {
        // Sharded checkpoints are named by their directory and unsharded
        // ones by their file, but anyone holding a model directory passes
        // the directory for both — and this used to reach a reader of files
        // and fail with "is a directory", several frames from the caller and
        // naming no reason.
        let dir = std::env::temp_dir().join(format!("rlx-wr-dir-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("model.safetensors"), one_tensor_file()).unwrap();

        let from_dir = open_weight_loader(&dir).expect("opening the directory");
        let from_file =
            open_weight_loader(&dir.join("model.safetensors")).expect("opening the file");
        let mut a = from_dir.remaining_keys();
        let mut b = from_file.remaining_keys();
        a.sort();
        b.sort();
        assert_eq!(a, b, "a directory and the file in it must load the same");
        assert!(a.iter().any(|k| k == "t"), "{a:?}");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn list_formats_includes_builtins() {
        let ids: Vec<_> = list_registered_formats().iter().map(|r| r.id).collect();
        assert!(ids.contains(&"gguf"));
        assert!(ids.contains(&"safetensors"));
    }
}
