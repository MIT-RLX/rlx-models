//! Pack / open runnable moss-nano bundles.
//!
//! Prefer a single **`.rlxp`** (official RLXPFLAT) with nested
//! `graphs/*.rlxp`. Runtime materializes to a hashed temp dir, then
//! [`MossNative::load_loose`] lowers via `rlx-onnx-import` (no ORT; Hub has no
//! `.onnx`).
//!
//! Legacy `moss-nano.gguf` still opens. Pack from a local ONNX tree with
//! [`pack_rlxp`] / `just export-moss-nano-rlxp`.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use rlx_assets::pack::{
    BundleLoadPath, ResolveBundleOpts, ResolvedBundlePath, extract_dir_for, is_rlxp_file,
    materialize_rlxp, resolve_bundle_load_path, resolve_bundle_path,
};
use rlx_assets::native_pack::{OnnxGraphSpec, pack_native_from_onnx_dir};
use rlx_gguf::{GgmlType, GgufFile, GgufWriter, MetaValue};

use rlx_runtime::Device;

use crate::native::MossNative;

pub const ARCH: &str = "rlx-moss-nano";
pub const FORMAT: &str = "rlx-moss-nano-gguf-v1";
pub const DEFAULT_GGUF_NAME: &str = "moss-nano.gguf";
pub const DEFAULT_RLXP_NAME: &str = "moss-nano.rlxp";
pub const HF_REPO: &str = "eugenehp/moss-nano";

/// Files required at runtime after native pack materialize.
pub const REQUIRED_RELPATHS: &[&str] = &[
    "browser_poc_manifest.json",
    "tokenizer.json",
    "graphs/moss_tts_prefill.rlxp",
    "graphs/moss_tts_local_fixed_sampled_frame.rlxp",
    "graphs/moss_audio_tokenizer_decode_full.rlxp",
];

/// Legacy ONNX + external-data sources for packing / GGUF.
pub const LEGACY_ONNX_RELPATHS: &[&str] = &[
    "browser_poc_manifest.json",
    "tokenizer.json",
    "moss_tts_prefill.onnx",
    "moss_tts_global_shared.data",
    "moss_tts_local_fixed_sampled_frame.onnx",
    "moss_tts_local_shared.data",
    "codec/moss_audio_tokenizer_decode_full.onnx",
    "codec/moss_audio_tokenizer_decode_shared.data",
];

const TEXT_SUFFIXES: &[&str] = &["json", "txt", "md", "cfg"];

#[derive(Debug, Clone)]
pub struct PackReport {
    pub path: PathBuf,
    pub bytes: u64,
    pub file_kv: u32,
    pub blob_count: u32,
}

/// Resolve `path` if it is an `.rlxp` file, or `dir/moss-nano.rlxp`.
pub fn resolve_rlxp_path(path: &Path) -> Option<PathBuf> {
    match resolve_bundle_path(
        path,
        &ResolveBundleOpts {
            default_rlxp: Some(DEFAULT_RLXP_NAME),
            default_gguf: None,
            extract_prefix: None,
        },
    )
    .ok()?
    {
        ResolvedBundlePath::Pack(p) if is_rlxp_file(&p) => Some(p),
        _ => None,
    }
}

/// Resolve `path` if it is a GGUF file, or `dir/moss-nano.gguf`.
pub fn resolve_gguf_path(path: &Path) -> Option<PathBuf> {
    match resolve_bundle_path(
        path,
        &ResolveBundleOpts {
            default_rlxp: None,
            default_gguf: Some(DEFAULT_GGUF_NAME),
            extract_prefix: None,
        },
    )
    .ok()?
    {
        ResolvedBundlePath::Gguf(p) => Some(p),
        _ => None,
    }
}

fn bundle_resolve_opts() -> ResolveBundleOpts<'static> {
    ResolveBundleOpts {
        default_rlxp: Some(DEFAULT_RLXP_NAME),
        default_gguf: Some(DEFAULT_GGUF_NAME),
        extract_prefix: Some("rlx-moss-rlxp"),
    }
}

/// Prefer `.rlxp`, then legacy GGUF, then a loose directory bundle.
pub fn open_path(path: &Path, device: Device) -> Result<MossNative> {
    match resolve_bundle_load_path(path, &bundle_resolve_opts())? {
        BundleLoadPath::Gguf(p) => open_gguf(&p, device),
        BundleLoadPath::Directory(d) => MossNative::load_loose(&d, device),
    }
}

pub fn open_rlxp(rlxp_path: &Path, device: Device) -> Result<MossNative> {
    let extract_dir = extract_dir_for(rlxp_path, "rlx-moss-rlxp");
    materialize_rlxp(rlxp_path, &extract_dir)
        .with_context(|| format!("materialize {}", rlxp_path.display()))?;
    for rel in REQUIRED_RELPATHS {
        ensure!(
            extract_dir.join(rel).is_file(),
            "rlxp missing required file after extract: {rel}"
        );
    }
    MossNative::load_loose(&extract_dir, device)
}

pub fn open_gguf(gguf_path: &Path, device: Device) -> Result<MossNative> {
    let file = GgufFile::from_path(gguf_path)
        .with_context(|| format!("open GGUF {}", gguf_path.display()))?;
    let arch = file
        .metadata
        .get("general.architecture")
        .and_then(MetaValue::as_str)
        .unwrap_or("");
    ensure!(
        arch == ARCH,
        "expected general.architecture={ARCH}, got {arch:?} in {}",
        gguf_path.display()
    );

    let extract_dir = extract_dir_for(gguf_path, "rlx-moss-gguf");
    materialize_gguf(&file, &extract_dir)
        .with_context(|| format!("materialize beside {}", gguf_path.display()))?;
    MossNative::load_loose(&extract_dir, device)
}

fn materialize_gguf(file: &GgufFile, extract_dir: &Path) -> Result<()> {
    let marker = extract_dir.join(".gguf_ok");
    if marker.is_file() && extract_dir.join("browser_poc_manifest.json").is_file() {
        return Ok(());
    }
    if extract_dir.exists() {
        std::fs::remove_dir_all(extract_dir)
            .with_context(|| format!("clear stale extract {}", extract_dir.display()))?;
    }
    std::fs::create_dir_all(extract_dir)?;

    for (key, val) in &file.metadata {
        let Some(rel) = key.strip_prefix("rlx_moss.file.") else {
            continue;
        };
        let Some(text) = val.as_str() else {
            continue;
        };
        let path = extract_dir.join(rel.replace('|', "/"));
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&path, text).with_context(|| format!("write {}", path.display()))?;
    }

    for name in file.keys() {
        let Some(rel) = name.strip_prefix("blob.") else {
            continue;
        };
        let t = file.get(name).context("blob tensor")?;
        ensure!(
            t.dtype == GgmlType::I8,
            "blob {name}: expected I8, got {:?}",
            t.dtype
        );
        let bytes = file.tensor_bytes(t)?;
        let path = extract_dir.join(rel.replace('|', "/"));
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&path, bytes).with_context(|| format!("write blob {}", path.display()))?;
    }

    for rel in LEGACY_ONNX_RELPATHS {
        ensure!(
            extract_dir.join(rel).is_file(),
            "GGUF missing required file after extract: {rel}"
        );
    }
    std::fs::write(&marker, b"ok")?;
    Ok(())
}

/// Pack required loose-dir files into one `moss-nano.rlxp` (native nested graphs).
pub fn pack_rlxp(bundle: &Path, out: &Path) -> Result<PackReport> {
    ensure!(bundle.is_dir(), "bundle dir not found: {}", bundle.display());
    ensure!(
        bundle.join("browser_poc_manifest.json").is_file() && bundle.join("tokenizer.json").is_file(),
        "missing manifest/tokenizer under {}",
        bundle.display()
    );
    let specs = [
        OnnxGraphSpec {
            name: "moss_tts_prefill".into(),
            onnx_path: bundle.join("moss_tts_prefill.onnx"),
        },
        OnnxGraphSpec {
            name: "moss_tts_local_fixed_sampled_frame".into(),
            onnx_path: bundle.join("moss_tts_local_fixed_sampled_frame.onnx"),
        },
        OnnxGraphSpec {
            name: "moss_audio_tokenizer_decode_full".into(),
            onnx_path: bundle.join("codec/moss_audio_tokenizer_decode_full.onnx"),
        },
    ];
    for s in &specs {
        ensure!(s.onnx_path.is_file(), "missing pack source {}", s.onnx_path.display());
    }
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    pack_native_from_onnx_dir(bundle, &specs, out, "moss-nano")
        .with_context(|| format!("write {}", out.display()))?;
    let bytes = std::fs::metadata(out)?.len();
    Ok(PackReport {
        path: out.to_path_buf(),
        bytes,
        file_kv: 0,
        blob_count: specs.len() as u32,
    })
}

/// Pack the required loose-dir files into one legacy `moss-nano.gguf`.
pub fn pack_directory(bundle: &Path, out: &Path) -> Result<PackReport> {
    ensure!(bundle.is_dir(), "bundle dir not found: {}", bundle.display());
    for rel in LEGACY_ONNX_RELPATHS {
        ensure!(
            bundle.join(rel).is_file(),
            "missing required file {} (under {})",
            rel,
            bundle.display()
        );
    }

    let mut w = GgufWriter::new();
    w.set_arch(ARCH);
    w.set_meta("general.name", MetaValue::String("moss-nano".into()));
    w.set_meta("rlx_moss.format", MetaValue::String(FORMAT.into()));
    w.set_meta("rlx_moss.sample_rate_hz", MetaValue::U32(48_000));
    w.set_meta("rlx_moss.channels", MetaValue::U32(2));
    w.set_meta(
        "rlx_moss.hf_repo",
        MetaValue::String(HF_REPO.into()),
    );

    let mut names = BTreeSet::new();
    let mut file_kv = 0u32;
    let mut blob_count = 0u32;

    for rel in LEGACY_ONNX_RELPATHS {
        let path = bundle.join(rel);
        let data = std::fs::read(&path).with_context(|| format!("read {}", path.display()))?;
        let key = rel.replace('/', "|");
        let is_text = path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| TEXT_SUFFIXES.iter().any(|s| s.eq_ignore_ascii_case(e)))
            && data.len() < 8 * 1024 * 1024;
        if is_text {
            if let Ok(text) = std::str::from_utf8(&data) {
                w.set_meta(
                    format!("rlx_moss.file.{key}"),
                    MetaValue::String(text.to_string()),
                );
                file_kv += 1;
                continue;
            }
        }
        let name = format!("blob.{key}");
        ensure!(names.insert(name.clone()), "duplicate tensor {name}");
        w.add_tensor_bytes(&name, vec![data.len()], GgmlType::I8, data)?;
        blob_count += 1;
    }

    w.set_meta("rlx_moss.file_kv_count", MetaValue::U32(file_kv));
    w.set_meta("rlx_moss.blob_count", MetaValue::U32(blob_count));

    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    w.write_to_path(out)
        .with_context(|| format!("write {}", out.display()))?;
    let bytes = std::fs::metadata(out)?.len();
    Ok(PackReport {
        path: out.to_path_buf(),
        bytes,
        file_kv,
        blob_count,
    })
}
