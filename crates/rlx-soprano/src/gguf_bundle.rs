//! Pack / open runnable soprano bundles.
//!
//! Prefer a single **`.rlxp`** (official RLXPFLAT asset pack) with nested
//! `graphs/*.rlxp`. Runtime materializes the outer pack, then
//! [`NativeSoprano::open_loose`] lowers nested graphs via `rlx-onnx-import`
//! (no ORT; Hub has no `.onnx`).
//!
//! Legacy `soprano.gguf` still opens. Pack-time source is local ONNX under
//! `onnx/` (`just export-soprano-rlxp` → [`pack_rlxp`]).

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use rlx_assets::pack::{
    BundleLoadPath, ResolveBundleOpts, ResolvedBundlePath, extract_dir_for, is_rlxp_file,
    materialize_rlxp, resolve_bundle_load_path, resolve_bundle_path,
};
use rlx_gguf::{GgmlType, GgufFile, GgufWriter, MetaValue};
use rlx_runtime::Device;

use crate::native::NativeSoprano;

pub const ARCH: &str = "rlx-soprano";
pub const FORMAT: &str = "rlx-soprano-gguf-v1";
pub const DEFAULT_GGUF_NAME: &str = "soprano.gguf";
pub const DEFAULT_RLXP_NAME: &str = "soprano.rlxp";
pub const HF_REPO: &str = "eugenehp/soprano";

pub const REQUIRED_RELPATHS: &[&str] = &[
    "tokenizer.json",
    "graphs/soprano_backbone_kv_fp32.rlxp",
    "graphs/soprano_decoder_fp32.rlxp",
];

/// Legacy ONNX sources used only when packing (`just export-soprano-rlxp`).
pub const PACK_ONNX_SOURCES: &[&str] = &[
    "onnx/soprano_backbone_kv_fp32.onnx",
    "onnx/soprano_decoder_fp32.onnx",
];

/// Files embedded in legacy GGUF packs (still materializable locally).
pub const LEGACY_GGUF_RELPATHS: &[&str] = &[
    "tokenizer.json",
    "onnx/soprano_backbone_kv_fp32.onnx",
    "onnx/soprano_decoder_fp32.onnx",
    "onnx/soprano_decoder_fp32.onnx.data",
];

const TEXT_SUFFIXES: &[&str] = &["json", "txt", "md", "cfg"];

#[derive(Debug, Clone)]
pub struct PackReport {
    pub path: PathBuf,
    pub bytes: u64,
    pub file_kv: u32,
    pub blob_count: u32,
}

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
        extract_prefix: Some("rlx-soprano-rlxp"),
    }
}

pub fn open_path(path: &Path, device: Device) -> Result<NativeSoprano> {
    match resolve_bundle_load_path(path, &bundle_resolve_opts())? {
        BundleLoadPath::Gguf(p) => open_gguf(&p, device),
        BundleLoadPath::Directory(d) => NativeSoprano::open_loose(&d, device),
    }
}

pub fn open_rlxp(rlxp_path: &Path, device: Device) -> Result<NativeSoprano> {
    let extract_dir = extract_dir_for(rlxp_path, "rlx-soprano-rlxp");
    materialize_rlxp(rlxp_path, &extract_dir)
        .with_context(|| format!("materialize {}", rlxp_path.display()))?;
    for rel in REQUIRED_RELPATHS {
        ensure!(
            extract_dir.join(rel).is_file(),
            "rlxp missing required file after extract: {rel}"
        );
    }
    NativeSoprano::open_loose(&extract_dir, device)
}

pub fn open_gguf(gguf_path: &Path, device: Device) -> Result<NativeSoprano> {
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
    let extract_dir = extract_dir_for(gguf_path, "rlx-soprano-gguf");
    materialize_gguf(&file, &extract_dir)
        .with_context(|| format!("materialize beside {}", gguf_path.display()))?;
    NativeSoprano::open_loose(&extract_dir, device)
}

fn materialize_gguf(file: &GgufFile, extract_dir: &Path) -> Result<()> {
    let marker = extract_dir.join(".gguf_ok");
    if marker.is_file() && extract_dir.join("tokenizer.json").is_file() {
        return Ok(());
    }
    if extract_dir.exists() {
        std::fs::remove_dir_all(extract_dir)
            .with_context(|| format!("clear stale extract {}", extract_dir.display()))?;
    }
    std::fs::create_dir_all(extract_dir)?;

    for (key, val) in &file.metadata {
        let Some(rel) = key.strip_prefix("rlx_soprano.file.") else {
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

    for rel in LEGACY_GGUF_RELPATHS {
        ensure!(
            extract_dir.join(rel).is_file(),
            "GGUF missing required file after extract: {rel}"
        );
    }
    std::fs::write(&marker, b"ok")?;
    Ok(())
}

/// Pack required loose-dir files into one `soprano.rlxp` (native nested graphs).
pub fn pack_rlxp(bundle: &Path, out: &Path) -> Result<PackReport> {
    ensure!(
        bundle.is_dir(),
        "bundle dir not found: {}",
        bundle.display()
    );
    ensure!(
        bundle.join("tokenizer.json").is_file(),
        "missing tokenizer.json under {}",
        bundle.display()
    );
    for rel in PACK_ONNX_SOURCES {
        ensure!(
            bundle.join(rel).is_file(),
            "missing pack source {} (under {})",
            rel,
            bundle.display()
        );
    }
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let specs = rlx_assets::native_pack::specs_from_root(
        bundle,
        &["soprano_backbone_kv_fp32", "soprano_decoder_fp32"],
    );
    rlx_assets::native_pack::pack_native_from_onnx_dir(bundle, &specs, out, "soprano")
        .with_context(|| format!("write {}", out.display()))?;
    let bytes = std::fs::metadata(out)?.len();
    Ok(PackReport {
        path: out.to_path_buf(),
        bytes,
        file_kv: 0,
        blob_count: specs.len() as u32,
    })
}

pub fn pack_directory(bundle: &Path, out: &Path) -> Result<PackReport> {
    ensure!(
        bundle.is_dir(),
        "bundle dir not found: {}",
        bundle.display()
    );
    for rel in LEGACY_GGUF_RELPATHS {
        ensure!(
            bundle.join(rel).is_file(),
            "missing required file {} (under {})",
            rel,
            bundle.display()
        );
    }

    let mut w = GgufWriter::new();
    w.set_arch(ARCH);
    w.set_meta("general.name", MetaValue::String("soprano".into()));
    w.set_meta("rlx_soprano.format", MetaValue::String(FORMAT.into()));
    w.set_meta("rlx_soprano.sample_rate_hz", MetaValue::U32(32_000));
    w.set_meta("rlx_soprano.hf_repo", MetaValue::String(HF_REPO.into()));

    let mut names = BTreeSet::new();
    let mut file_kv = 0u32;
    let mut blob_count = 0u32;

    for rel in LEGACY_GGUF_RELPATHS {
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
                    format!("rlx_soprano.file.{key}"),
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

    w.set_meta("rlx_soprano.file_kv_count", MetaValue::U32(file_kv));
    w.set_meta("rlx_soprano.blob_count", MetaValue::U32(blob_count));

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
