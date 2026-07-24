//! Open / pack `rlx-tts.rlxp` (preferred) or legacy `rlx-tts.gguf`.

use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use rlx_assets::pack::{
    ResolveBundleOpts, ResolvedBundlePath, is_rlxp_file, resolve_bundle_path,
};
use rlx_gguf::{GgmlType, GgufFile, GgufWriter, MetaValue};
use rlx_ir::Graph;
use rlx_pkg::{ContainerKind, Package, PackedWeight, WriteOptions, write_package};

use crate::frontend::HydraLite;
use crate::native::{
    BundleFiles, BundleManifest, PredictorBiases, RlxTts, load_output_post, open_dir_bundle,
};
use crate::weights::Weights;

const ARCH: &str = "rlx-tts";
const FORMAT: &str = "rlx-tts-gguf-v1";
const RLXP_FORMAT: &str = "rlx-tts-rlxp-v1";
pub const DEFAULT_RLXP_NAME: &str = "rlx-tts.rlxp";
const NEURAL_STEMS: &[&str] = &["encoder", "decoder", "wavernn"];
const NEURAL_PREFIXES: &[&str] = &["encoder.", "decoder.", "wavernn."];
const TEXT_SUFFIXES: &[&str] = &["json", "txt", "cfg", "md"];
const SKIP_NAMES: &[&str] = &[
    ".DS_Store",
    "rlx-tts.gguf",
    "frontend.cfg",
    "gryphon.cfg",
    // Legacy / unused at runtime (kept out of packed GGUF).
    "gprm",
    "g2p_seq2seq.bin",
    "g2p_seq2seq.arch.json",
    "g2p_seq2seq.stack.json",
    "g2p_seq2seq.inventory.json",
    "g2p_seq2seq.meta.json",
    "phbk",
    "to_xsampa.json",
    "symbols.json",
];

/// Resolve an `.rlxp` path: either the file itself or `dir/rlx-tts.rlxp`.
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

/// Resolve a GGUF path: either the file itself or `dir/rlx-tts.gguf`.
pub fn resolve_gguf_path(path: &Path) -> Option<PathBuf> {
    match resolve_bundle_path(
        path,
        &ResolveBundleOpts {
            default_rlxp: None,
            default_gguf: Some("rlx-tts.gguf"),
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
        default_gguf: Some("rlx-tts.gguf"),
        extract_prefix: Some("rlx-tts-pack"),
    }
}

pub fn open_gguf(gguf_path: &Path) -> Result<RlxTts> {
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

    let extract_dir = rlx_assets::pack::extract_dir_for(gguf_path, "rlx-tts-gguf");
    materialize_sidecar(&file, &extract_dir)
        .with_context(|| format!("materialize frontend beside {}", gguf_path.display()))?;

    let mut encoder = HashMap::new();
    let mut decoder = HashMap::new();
    let mut wavernn = HashMap::new();
    for name in file.keys() {
        let (bucket, stripped) = if let Some(rest) = name.strip_prefix("encoder.") {
            (&mut encoder, rest)
        } else if let Some(rest) = name.strip_prefix("decoder.") {
            (&mut decoder, rest)
        } else if let Some(rest) = name.strip_prefix("wavernn.") {
            (&mut wavernn, rest)
        } else {
            continue;
        };
        let (data, shape) = file
            .dequant_f32(name)
            .with_context(|| format!("dequant {name}"))?;
        bucket.insert(stripped.to_string(), (data, shape));
    }
    ensure!(!encoder.is_empty(), "GGUF missing encoder.* tensors");
    ensure!(!decoder.is_empty(), "GGUF missing decoder.* tensors");
    ensure!(!wavernn.is_empty(), "GGUF missing wavernn.* tensors");

    let mut encoder = Weights::from_map(encoder);
    let mut decoder = Weights::from_map(decoder);
    if std::env::var_os("RLX_FS2_F16_PARAMS").is_some() {
        encoder.f16_round_params();
        decoder.f16_round_params();
    }
    let wavernn = Weights::from_map(wavernn);

    let manifest = manifest_from_gguf(&file, &extract_dir)?;
    let frontend = HydraLite::open(&extract_dir)?;
    let post = load_output_post(&extract_dir);

    Ok(RlxTts::from_parts(
        extract_dir,
        manifest,
        encoder,
        decoder,
        wavernn,
        frontend,
        post,
    ))
}

pub fn open_rlxp(rlxp_path: &Path) -> Result<RlxTts> {
    let pack = Package::open(rlxp_path)
        .with_context(|| format!("open RLXP {}", rlxp_path.display()))?;
    let extract_dir = rlx_assets::pack::extract_dir_for(rlxp_path, "rlx-tts-rlxp");
    materialize_rlxp_sidecars(&pack, &extract_dir)
        .with_context(|| format!("materialize frontend beside {}", rlxp_path.display()))?;

    let idx = pack
        .weights_index()
        .context("rlxp missing weights index")?;
    let mut encoder = HashMap::new();
    let mut decoder = HashMap::new();
    let mut wavernn = HashMap::new();
    for name in idx.names() {
        let (bucket, stripped) = if let Some(rest) = name.strip_prefix("encoder.") {
            (&mut encoder, rest)
        } else if let Some(rest) = name.strip_prefix("decoder.") {
            (&mut decoder, rest)
        } else if let Some(rest) = name.strip_prefix("wavernn.") {
            (&mut wavernn, rest)
        } else {
            continue;
        };
        let entry = pack.weight_entry(name)?;
        let data = pack.tensor_f32(name).with_context(|| format!("tensor {name}"))?;
        bucket.insert(stripped.to_string(), (data, entry.shape.clone()));
    }
    ensure!(!encoder.is_empty(), "RLXP missing encoder.* tensors");
    ensure!(!decoder.is_empty(), "RLXP missing decoder.* tensors");
    ensure!(!wavernn.is_empty(), "RLXP missing wavernn.* tensors");

    let mut encoder = Weights::from_map(encoder);
    let mut decoder = Weights::from_map(decoder);
    if std::env::var_os("RLX_FS2_F16_PARAMS").is_some() {
        encoder.f16_round_params();
        decoder.f16_round_params();
    }
    let wavernn = Weights::from_map(wavernn);

    let manifest = manifest_from_extract(&extract_dir)?;
    let frontend = HydraLite::open(&extract_dir)?;
    let post = load_output_post(&extract_dir);

    Ok(RlxTts::from_parts(
        extract_dir,
        manifest,
        encoder,
        decoder,
        wavernn,
        frontend,
        post,
    ))
}

fn materialize_rlxp_sidecars(pack: &Package, extract_dir: &Path) -> Result<()> {
    let marker = extract_dir.join(".rlxp_ok");
    if marker.is_file() && extract_dir.join("manifest.json").is_file() {
        return Ok(());
    }
    if extract_dir.exists() {
        std::fs::remove_dir_all(extract_dir)
            .with_context(|| format!("clear stale extract {}", extract_dir.display()))?;
    }
    std::fs::create_dir_all(extract_dir)?;

    for sc in &pack.manifest().sidecars {
        let bytes = pack
            .sidecar(&sc.id)
            .with_context(|| format!("sidecar {}", sc.id))?;
        let path = extract_dir.join(&sc.id);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&path, bytes).with_context(|| format!("write {}", path.display()))?;
    }

    let man_path = extract_dir.join("manifest.json");
    if !man_path.is_file() {
        std::fs::write(&man_path, default_manifest_json_rlxp())?;
    }

    std::fs::write(&marker, b"ok")?;
    Ok(())
}

fn default_manifest_json_rlxp() -> String {
    format!(
        r#"{{
  "format": "{RLXP_FORMAT}",
  "voice_identifier": "rlx-tts",
  "voice_name": "rlx-tts",
  "sample_rate_hz": 24000,
  "mel_bins": 80,
  "hop_length": 240,
  "phone_vocab": 84,
  "files": {{
    "encoder": "encoder.safetensors",
    "decoder": "decoder.safetensors",
    "wavernn": "wavernn.safetensors"
  }}
}}
"#
    )
}

fn manifest_from_extract(extract_dir: &Path) -> Result<BundleManifest> {
    let man_path = extract_dir.join("manifest.json");
    if man_path.is_file() {
        let text = std::fs::read_to_string(&man_path)?;
        if let Ok(mut m) = serde_json::from_str::<BundleManifest>(&text) {
            if m.format.is_empty() {
                m.format = RLXP_FORMAT.into();
            }
            return Ok(m);
        }
    }
    Ok(BundleManifest {
        format: RLXP_FORMAT.into(),
        voice_identifier: "rlx-tts".into(),
        sample_rate_hz: 24_000,
        mel_bins: 80,
        hop_length: 240,
        phone_vocab: 84,
        files: BundleFiles {
            encoder: "encoder.safetensors".into(),
            decoder: "decoder.safetensors".into(),
            wavernn: "wavernn.safetensors".into(),
        },
        predictor_biases: PredictorBiases::default(),
        source_asset_dir: None,
    })
}

fn materialize_sidecar(file: &GgufFile, extract_dir: &Path) -> Result<()> {
    let marker = extract_dir.join(".gguf_ok");
    if marker.is_file() && extract_dir.join("manifest.json").is_file() {
        return Ok(());
    }
    if extract_dir.exists() {
        std::fs::remove_dir_all(extract_dir)
            .with_context(|| format!("clear stale extract {}", extract_dir.display()))?;
    }
    std::fs::create_dir_all(extract_dir)?;

    for (key, val) in &file.metadata {
        let Some(rel) = key.strip_prefix("rlx_tts.file.") else {
            continue;
        };
        let Some(text) = val.as_str() else {
            continue;
        };
        let path = extract_dir.join(rel.replace('|', "/"));
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&path, text)
            .with_context(|| format!("write {}", path.display()))?;
    }

    for name in file.keys() {
        let Some(rel) = name.strip_prefix("blob.") else {
            continue;
        };
        // Skip accidental neural-shaped names.
        if NEURAL_PREFIXES.iter().any(|p| name.starts_with(p)) {
            continue;
        }
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
        std::fs::write(&path, bytes)
            .with_context(|| format!("write blob {}", path.display()))?;
    }

    // Ensure a minimal manifest exists even if the KV was missing.
    let man_path = extract_dir.join("manifest.json");
    if !man_path.is_file() {
        let man = default_manifest_json(file);
        std::fs::write(&man_path, man)?;
    }

    std::fs::write(&marker, b"ok")?;
    Ok(())
}

fn default_manifest_json(file: &GgufFile) -> String {
    let rate = meta_u32(file, "rlx_tts.sample_rate_hz").unwrap_or(24_000);
    let voice = file
        .metadata
        .get("rlx_tts.voice_identifier")
        .and_then(MetaValue::as_str)
        .unwrap_or("rlx-tts");
    format!(
        r#"{{
  "format": "rlx-tts-gguf-v1",
  "voice_identifier": "{voice}",
  "voice_name": "{voice}",
  "sample_rate_hz": {rate},
  "mel_bins": 80,
  "hop_length": 240,
  "phone_vocab": 84,
  "files": {{
    "encoder": "encoder.safetensors",
    "decoder": "decoder.safetensors",
    "wavernn": "wavernn.safetensors"
  }}
}}
"#
    )
}

fn meta_u32(file: &GgufFile, key: &str) -> Option<u32> {
    file.metadata.get(key).and_then(|v| match v {
        MetaValue::U32(x) => Some(*x),
        MetaValue::U64(x) => u32::try_from(*x).ok(),
        MetaValue::I32(x) if *x >= 0 => Some(*x as u32),
        _ => v.as_u32(),
    })
}

fn manifest_from_gguf(file: &GgufFile, extract_dir: &Path) -> Result<BundleManifest> {
    let man_path = extract_dir.join("manifest.json");
    if man_path.is_file() {
        let text = std::fs::read_to_string(&man_path)?;
        if let Ok(mut m) = serde_json::from_str::<BundleManifest>(&text) {
            m.format = file
                .metadata
                .get("rlx_tts.format")
                .and_then(MetaValue::as_str)
                .unwrap_or("rlx-tts-gguf-v1")
                .to_string();
            return Ok(m);
        }
    }
    Ok(BundleManifest {
        format: "rlx-tts-gguf-v1".into(),
        voice_identifier: file
            .metadata
            .get("rlx_tts.voice_identifier")
            .and_then(MetaValue::as_str)
            .unwrap_or("rlx-tts")
            .to_string(),
        sample_rate_hz: meta_u32(file, "rlx_tts.sample_rate_hz").unwrap_or(24_000),
        mel_bins: meta_u32(file, "rlx_tts.mel_bins").unwrap_or(80),
        hop_length: meta_u32(file, "rlx_tts.hop_length").unwrap_or(240),
        phone_vocab: meta_u32(file, "rlx_tts.phone_vocab").unwrap_or(84),
        files: BundleFiles {
            encoder: "encoder.safetensors".into(),
            decoder: "decoder.safetensors".into(),
            wavernn: "wavernn.safetensors".into(),
        },
        predictor_biases: PredictorBiases::default(),
        source_asset_dir: None,
    })
}

/// Sanitize a loose-bundle `manifest.json` (strip Apple source fields, set voice ids).
pub fn sanitize_manifest(path: &Path) -> Result<()> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("read {}", path.display()))?;
    let mut man: serde_json::Value =
        serde_json::from_str(&text).with_context(|| format!("parse {}", path.display()))?;
    let obj = man
        .as_object_mut()
        .with_context(|| format!("manifest root not object: {}", path.display()))?;
    obj.insert(
        "format".into(),
        serde_json::Value::String("rlx-tts-bundle-v1".into()),
    );
    obj.insert(
        "voice_identifier".into(),
        serde_json::Value::String("rlx-tts".into()),
    );
    obj.insert(
        "voice_name".into(),
        serde_json::Value::String("rlx-tts".into()),
    );
    obj.remove("source_asset_dir");
    obj.insert(
        "notes".into(),
        serde_json::json!(["Private local RLX TTS bundle. Do not commit or publish weights."]),
    );
    let out = serde_json::to_string_pretty(&man)? + "\n";
    std::fs::write(path, out).with_context(|| format!("write {}", path.display()))?;
    Ok(())
}

/// Pack a loose directory bundle into one runnable `rlx-tts.gguf`.
pub fn pack_directory(bundle: &Path, out: &Path) -> Result<PackReport> {
    ensure!(bundle.is_dir(), "bundle dir not found: {}", bundle.display());

    let man_path = bundle.join("manifest.json");
    let man: serde_json::Value = if man_path.is_file() {
        serde_json::from_str(&std::fs::read_to_string(&man_path)?)?
    } else {
        serde_json::json!({})
    };
    let voice = man
        .get("voice_name")
        .or_else(|| man.get("voice_identifier"))
        .and_then(|v| v.as_str())
        .unwrap_or("rlx-tts");
    let sample_rate = man
        .get("sample_rate_hz")
        .and_then(|v| v.as_u64())
        .unwrap_or(24_000) as u32;
    let mel_bins = man
        .get("mel_bins")
        .and_then(|v| v.as_u64())
        .unwrap_or(80) as u32;
    let hop_length = man
        .get("hop_length")
        .and_then(|v| v.as_u64())
        .unwrap_or(240) as u32;
    let phone_vocab = man
        .get("phone_vocab")
        .and_then(|v| v.as_u64())
        .unwrap_or(84) as u32;

    let mut w = GgufWriter::new();
    w.set_arch(ARCH);
    w.set_meta("general.name", MetaValue::String(voice.into()));
    w.set_meta("rlx_tts.format", MetaValue::String(FORMAT.into()));
    w.set_meta(
        "rlx_tts.voice_identifier",
        MetaValue::String(
            man.get("voice_identifier")
                .and_then(|v| v.as_str())
                .unwrap_or("rlx-tts")
                .into(),
        ),
    );
    w.set_meta("rlx_tts.sample_rate_hz", MetaValue::U32(sample_rate));
    w.set_meta("rlx_tts.mel_bins", MetaValue::U32(mel_bins));
    w.set_meta("rlx_tts.hop_length", MetaValue::U32(hop_length));
    w.set_meta("rlx_tts.phone_vocab", MetaValue::U32(phone_vocab));

    let mut names = BTreeSet::new();
    let mut tensor_count = 0usize;
    for stem in NEURAL_STEMS {
        let path = bundle.join(format!("{stem}.safetensors"));
        ensure!(path.is_file(), "missing {}", path.display());
        let weights = Weights::load(&path)?;
        let mut keys: Vec<_> = weights.names().map(str::to_string).collect();
        keys.sort();
        for key in keys {
            let (data, shape) = weights.get(&key)?;
            let name = format!("{stem}.{key}");
            add_f32(&mut w, &mut names, &name, shape.clone(), data)?;
            tensor_count += 1;
        }
    }

    let mut file_kv = 0u32;
    let mut blob_count = 0u32;
    let mut files = Vec::new();
    collect_files(bundle, bundle, &mut files)?;
    files.sort();
    for path in files {
        let rel = path.strip_prefix(bundle).unwrap_or(path.as_path());
        if skip_rel(rel) {
            continue;
        }
        // Neural safetensors expanded above.
        if NEURAL_STEMS.iter().any(|s| {
            rel == Path::new(&format!("{s}.safetensors"))
        }) {
            continue;
        }
        let data = std::fs::read(&path).with_context(|| format!("read {}", path.display()))?;
        let key = rel_key(rel);
        let is_text = path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| TEXT_SUFFIXES.iter().any(|s| s.eq_ignore_ascii_case(e)))
            && data.len() < 8 * 1024 * 1024;
        if is_text {
            if let Ok(text) = std::str::from_utf8(&data) {
                w.set_meta(
                    format!("rlx_tts.file.{key}"),
                    MetaValue::String(text.to_string()),
                );
                file_kv += 1;
                continue;
            }
        }
        let name = format!("blob.{key}");
        add_i8(&mut w, &mut names, &name, vec![data.len()], &data)?;
        blob_count += 1;
        tensor_count += 1;
    }

    w.set_meta("rlx_tts.tensor_count", MetaValue::U32(tensor_count as u32));
    w.set_meta("rlx_tts.file_kv_count", MetaValue::U32(file_kv));
    w.set_meta("rlx_tts.blob_count", MetaValue::U32(blob_count));

    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    w.write_to_path(out)
        .with_context(|| format!("write {}", out.display()))?;
    let bytes = std::fs::metadata(out)?.len();
    Ok(PackReport {
        path: out.to_path_buf(),
        bytes,
        tensor_count,
        file_kv,
        blob_count,
    })
}

/// Pack a loose directory or re-pack an existing GGUF into `rlx-tts.rlxp`.
pub fn pack_rlxp(source: &Path, out: &Path) -> Result<PackReport> {
    if source.is_file()
        && source
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("gguf"))
    {
        return pack_rlxp_from_gguf(source, out);
    }
    ensure!(source.is_dir(), "pack source not found: {}", source.display());
    if let Some(gguf) = resolve_gguf_path(source) {
        let loose = source.join("manifest.json").is_file()
            && source.join("encoder.safetensors").is_file();
        if !loose {
            return pack_rlxp_from_gguf(&gguf, out);
        }
    }
    pack_rlxp_from_directory(source, out)
}

fn pack_rlxp_from_gguf(gguf_path: &Path, out: &Path) -> Result<PackReport> {
    let file = GgufFile::from_path(gguf_path)
        .with_context(|| format!("open GGUF {}", gguf_path.display()))?;
    let voice = file
        .metadata
        .get("rlx_tts.voice_identifier")
        .and_then(MetaValue::as_str)
        .unwrap_or("rlx-tts");

    let mut weights = Vec::new();
    let mut sidecars = Vec::new();
    let mut tensor_count = 0usize;
    let mut file_kv = 0u32;
    let mut blob_count = 0u32;

    for name in file.keys() {
        if NEURAL_PREFIXES.iter().any(|p| name.starts_with(p)) {
            let t = file.get(name).with_context(|| format!("tensor {name}"))?;
            let (data, _) = file.dequant_f32(name).with_context(|| format!("dequant {name}"))?;
            let bytes: Vec<u8> = data.iter().flat_map(|x| x.to_le_bytes()).collect();
            weights.push(PackedWeight::hot(
                name,
                t.shape.clone(),
                "f32",
                "row_major",
                bytes,
            ));
            tensor_count += 1;
        } else if let Some(rel) = name.strip_prefix("blob.") {
            let t = file.get(name).context("blob tensor")?;
            let bytes = file.tensor_bytes(t)?;
            let path = rel.replace('|', "/");
            sidecars.push((path.clone(), media_type_for(&path), bytes.to_vec()));
            blob_count += 1;
        }
    }

    for (key, val) in &file.metadata {
        let Some(rel) = key.strip_prefix("rlx_tts.file.") else {
            continue;
        };
        let Some(text) = val.as_str() else {
            continue;
        };
        let path = rel.replace('|', "/");
        sidecars.push((
            path.clone(),
            media_type_for(&path),
            text.as_bytes().to_vec(),
        ));
        file_kv += 1;
    }

    if !sidecars.iter().any(|(id, _, _)| id == "manifest.json") {
        sidecars.push((
            "manifest.json".into(),
            "application/json".into(),
            default_manifest_json(&file).into_bytes(),
        ));
    }

    write_rlxp_pack(out, voice, weights, sidecars)?;
    let bytes = std::fs::metadata(out)?.len();
    Ok(PackReport {
        path: out.to_path_buf(),
        bytes,
        tensor_count,
        file_kv,
        blob_count,
    })
}

fn pack_rlxp_from_directory(bundle: &Path, out: &Path) -> Result<PackReport> {
    ensure!(bundle.is_dir(), "bundle dir not found: {}", bundle.display());

    let man_path = bundle.join("manifest.json");
    let man: serde_json::Value = if man_path.is_file() {
        serde_json::from_str(&std::fs::read_to_string(&man_path)?)?
    } else {
        serde_json::json!({})
    };
    let voice = man
        .get("voice_name")
        .or_else(|| man.get("voice_identifier"))
        .and_then(|v| v.as_str())
        .unwrap_or("rlx-tts");

    let mut weights = Vec::new();
    let mut tensor_count = 0usize;
    for stem in NEURAL_STEMS {
        let path = bundle.join(format!("{stem}.safetensors"));
        ensure!(path.is_file(), "missing {}", path.display());
        let wts = Weights::load(&path)?;
        let mut keys: Vec<_> = wts.names().map(str::to_string).collect();
        keys.sort();
        for key in keys {
            let (data, shape) = wts.get(&key)?;
            let name = format!("{stem}.{key}");
            let bytes: Vec<u8> = data.iter().flat_map(|x| x.to_le_bytes()).collect();
            weights.push(PackedWeight::hot(
                name,
                shape.clone(),
                "f32",
                "row_major",
                bytes,
            ));
            tensor_count += 1;
        }
    }

    let mut sidecars = Vec::new();
    let mut file_kv = 0u32;
    let mut blob_count = 0u32;
    let mut files = Vec::new();
    collect_files(bundle, bundle, &mut files)?;
    files.sort();
    for path in files {
        let rel = path.strip_prefix(bundle).unwrap_or(path.as_path());
        if skip_rel(rel) {
            continue;
        }
        if NEURAL_STEMS.iter().any(|s| rel == Path::new(&format!("{s}.safetensors"))) {
            continue;
        }
        let data = std::fs::read(&path).with_context(|| format!("read {}", path.display()))?;
        let rel_s = rel.to_string_lossy().replace('\\', "/");
        let is_text = path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| TEXT_SUFFIXES.iter().any(|s| s.eq_ignore_ascii_case(e)))
            && data.len() < 8 * 1024 * 1024;
        if is_text {
            if std::str::from_utf8(&data).is_ok() {
                file_kv += 1;
            }
        } else {
            blob_count += 1;
        }
        sidecars.push((rel_s.clone(), media_type_for(&rel_s), data));
    }

    if !sidecars.iter().any(|(id, _, _)| id == "manifest.json") {
        let rate = man
            .get("sample_rate_hz")
            .and_then(|v| v.as_u64())
            .unwrap_or(24_000) as u32;
        sidecars.push((
            "manifest.json".into(),
            "application/json".into(),
            format!(
                r#"{{
  "format": "{RLXP_FORMAT}",
  "voice_identifier": "{voice}",
  "voice_name": "{voice}",
  "sample_rate_hz": {rate},
  "mel_bins": 80,
  "hop_length": 240,
  "phone_vocab": 84,
  "files": {{
    "encoder": "encoder.safetensors",
    "decoder": "decoder.safetensors",
    "wavernn": "wavernn.safetensors"
  }}
}}
"#
            )
            .into_bytes(),
        ));
    }

    write_rlxp_pack(out, voice, weights, sidecars)?;
    let bytes = std::fs::metadata(out)?.len();
    Ok(PackReport {
        path: out.to_path_buf(),
        bytes,
        tensor_count,
        file_kv,
        blob_count,
    })
}

fn write_rlxp_pack(
    out: &Path,
    voice: &str,
    weights: Vec<PackedWeight>,
    sidecars: Vec<(String, String, Vec<u8>)>,
) -> Result<()> {
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let opts = WriteOptions {
        name: voice.into(),
        producer: Some("rlx-tts".into()),
        features: vec!["rlx-tts".into()],
        container: ContainerKind::Flat,
        sidecars,
        include_graph: false,
        compress_sidecars: true,
        ..WriteOptions::default()
    };
    let graph = Graph::new(voice);
    write_package(out, &graph, &weights, &opts)
        .with_context(|| format!("write {}", out.display()))?;
    Ok(())
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
        "txt" | "md" | "cfg" => "text/plain".into(),
        "safetensors" => "application/octet-stream".into(),
        _ => "application/octet-stream".into(),
    }
}

#[derive(Debug, Clone)]
pub struct PackReport {
    pub path: PathBuf,
    pub bytes: u64,
    pub tensor_count: usize,
    pub file_kv: u32,
    pub blob_count: u32,
}

fn rel_key(rel: &Path) -> String {
    rel.to_string_lossy().replace('\\', "/").replace('/', "|")
}

fn skip_rel(rel: &Path) -> bool {
    if rel.components().any(|c| {
        let s = c.as_os_str().to_string_lossy();
        s == "fixtures" || s.starts_with(".rlx-extracted")
    }) {
        return true;
    }
    let name = rel.file_name().and_then(|n| n.to_str()).unwrap_or("");
    SKIP_NAMES.iter().any(|s| *s == name)
}

fn collect_files(_root: &Path, dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    for entry in std::fs::read_dir(dir).with_context(|| format!("read_dir {}", dir.display()))? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect_files(_root, &path, out)?;
        } else if path.is_file() {
            out.push(path);
        }
    }
    Ok(())
}

fn add_f32(
    w: &mut GgufWriter,
    names: &mut BTreeSet<String>,
    name: &str,
    shape: Vec<usize>,
    data: &[f32],
) -> Result<()> {
    ensure!(names.insert(name.to_string()), "duplicate tensor {name}");
    let bytes: Vec<u8> = data.iter().flat_map(|x| x.to_le_bytes()).collect();
    w.add_tensor_bytes(name, shape, GgmlType::F32, bytes)?;
    Ok(())
}

fn add_i8(
    w: &mut GgufWriter,
    names: &mut BTreeSet<String>,
    name: &str,
    shape: Vec<usize>,
    data: &[u8],
) -> Result<()> {
    ensure!(names.insert(name.to_string()), "duplicate tensor {name}");
    w.add_tensor_bytes(name, shape, GgmlType::I8, data.to_vec())?;
    Ok(())
}

/// Prefer packed `.rlxp`, then legacy GGUF, then a loose directory bundle.
pub fn open_path(path: &Path) -> Result<RlxTts> {
    match resolve_bundle_path(path, &bundle_resolve_opts())? {
        ResolvedBundlePath::Pack(p) if is_rlxp_file(&p) => open_rlxp(&p),
        ResolvedBundlePath::Pack(p) => {
            let dir = rlx_assets::pack::extract_pack_to_dir(&p, "rlx-tts-pack")?;
            open_path(&dir)
        }
        ResolvedBundlePath::Bake(p) => bail!(
            "bake artifact {} is not a rlx-tts weight pack; convert to .rlxp \
             (`rlx-bake convert … -o rlx-tts.rlxp`) or use .gguf / loose dir",
            p.display()
        ),
        ResolvedBundlePath::Gguf(p) => open_gguf(&p),
        ResolvedBundlePath::Directory(d) => {
            if d.join("manifest.json").is_file() && d.join("encoder.safetensors").is_file() {
                open_dir_bundle(&d)
            } else {
                bail!(
                    "bundle dir {} has neither {DEFAULT_RLXP_NAME} nor rlx-tts.gguf nor safetensors+manifest",
                    d.display()
                );
            }
        }
    }
}

/// Materialize the default bundle GGUF (if present) and return its extract root.
/// Used by unit tests that need on-disk frontend tables.
#[cfg(test)]
pub fn default_extract_dir() -> Option<PathBuf> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let candidates = [
        PathBuf::from(crate::native::DEFAULT_BUNDLE_DIR).join(DEFAULT_RLXP_NAME),
        PathBuf::from(crate::native::DEFAULT_BUNDLE_DIR).join("rlx-tts.gguf"),
        manifest_dir.join("../../weights/tts/rlx-tts/rlx-tts.rlxp"),
        manifest_dir.join("../../weights/tts/rlx-tts/rlx-tts.gguf"),
        PathBuf::from(".cache/rlx-tts/rlx-tts.rlxp"),
        PathBuf::from(".cache/rlx-tts/rlx-tts.gguf"),
    ];
    for g in &candidates {
        if g.is_file() {
            return open_path(g).ok().map(|m| m.bundle_dir().to_path_buf());
        }
    }
    // Loose frontend/ still on disk (dev layout before packing).
    for root in [
        PathBuf::from(crate::native::DEFAULT_BUNDLE_DIR),
        manifest_dir.join("../../weights/tts/rlx-tts"),
    ] {
        if root.join("frontend").is_dir() {
            return Some(root);
        }
    }
    None
}
