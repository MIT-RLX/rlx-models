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

//! Model-compatibility discovery for rlx-models.
//!
//! Answers the same question llama.cpp and HuggingFace answer for their own
//! ecosystems: *can this runtime load and run this weights file?* — but for
//! rlx-models.
//!
//! Mirrors two upstream behaviors so reports stay aligned with what users
//! already expect from llama.cpp / HF:
//!
//! * **GGUF required fields.** llama.cpp's loader (`src/llama-model.cpp::
//!   load_hparams`, `src/llama-vocab.cpp`) treats `general.architecture`,
//!   `<arch>.context_length`, `<arch>.embedding_length`,
//!   `<arch>.block_count`, `tokenizer.ggml.model`, and
//!   `tokenizer.ggml.tokens` as mandatory. If any are missing the file
//!   isn't runnable, regardless of arch support. The same predicate drives
//!   HuggingFace's "compatible with llama.cpp" badge
//!   (`huggingface.js/packages/tasks/src/local-apps.ts:
//!   isLlamaCppGgufModel = !!model.gguf?.context_length`).
//!
//! * **HF safetensors model_type.** `transformers` resolves the model class
//!   from `config.json::model_type` (primary); `architectures[]` only
//!   disambiguates which head within that class. We follow the same
//!   precedence in [`crate::model_type_runner_name`].

use anyhow::{Context, Result, anyhow};
use rlx_core::gguf_support::resolve_weights_file;
use rlx_gguf::{GgufFile, MetaValue};
use std::path::{Path, PathBuf};

use crate::auto_dispatch::{
    UnimplementedArch, arch_runner_name, known_unimplemented_arch, model_type_runner_name,
};

/// Where the arch identification came from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompatSource {
    /// `general.architecture` value from a `.gguf` file.
    GgufArch(String),
    /// `model_type` value from a sidecar `config.json`.
    SafetensorsConfig(String),
}

impl CompatSource {
    pub fn arch(&self) -> &str {
        match self {
            Self::GgufArch(s) | Self::SafetensorsConfig(s) => s.as_str(),
        }
    }
}

/// llama.cpp's load-time required GGUF metadata fields. All are read by
/// `load_hparams`; missing any throws at load time.
#[derive(Debug, Clone, Default)]
pub struct GgufRequiredFields {
    pub context_length: Option<u64>,
    pub embedding_length: Option<u64>,
    pub block_count: Option<u64>,
    pub tokenizer_model: Option<String>,
    pub has_tokens: bool,
}

impl GgufRequiredFields {
    /// Which required field is missing, if any. Order mirrors the order
    /// llama.cpp reads them (helps line up against upstream errors).
    pub fn missing(&self) -> Vec<&'static str> {
        let mut out = Vec::new();
        if self.context_length.is_none() {
            out.push("<arch>.context_length");
        }
        if self.embedding_length.is_none() {
            out.push("<arch>.embedding_length");
        }
        if self.block_count.is_none() {
            out.push("<arch>.block_count");
        }
        if self.tokenizer_model.is_none() {
            out.push("tokenizer.ggml.model");
        }
        if !self.has_tokens {
            out.push("tokenizer.ggml.tokens");
        }
        out
    }

    pub fn is_complete(&self) -> bool {
        self.missing().is_empty()
    }
}

/// Top-level compatibility verdict.
#[derive(Debug, Clone)]
pub enum CompatibilityStatus {
    /// rlx has a registered runner for this arch and all required GGUF
    /// metadata is present (or the file is safetensors, where the
    /// per-arch loader does the field check itself).
    Supported { runner: &'static str },
    /// Required GGUF metadata is missing — the file would not load even
    /// if the arch were implemented. Lists the missing field names.
    MissingMetadata { missing: Vec<&'static str> },
    /// Arch is on PLAN.md but not yet implemented.
    KnownUnimplemented(UnimplementedArch),
    /// Arch isn't recognized.
    Unknown,
}

impl CompatibilityStatus {
    pub fn is_runnable(&self) -> bool {
        matches!(self, Self::Supported { .. })
    }
}

/// Structured report from [`check_path`]. Print directly via [`Display`]
/// or serialize as JSON via [`CompatibilityReport::to_json`].
#[derive(Debug, Clone)]
pub struct CompatibilityReport {
    pub path: PathBuf,
    pub source: CompatSource,
    pub status: CompatibilityStatus,
    /// Present for `.gguf` files only.
    pub gguf_fields: Option<GgufRequiredFields>,
}

impl CompatibilityReport {
    /// Render as a single-line JSON object — useful for `--json` output and
    /// for piping into shell tooling.
    pub fn to_json(&self) -> String {
        let (status_tag, status_detail) = match &self.status {
            CompatibilityStatus::Supported { runner } => {
                ("supported", serde_json::json!({ "runner": runner }))
            }
            CompatibilityStatus::MissingMetadata { missing } => (
                "missing_metadata",
                serde_json::json!({ "missing": missing }),
            ),
            CompatibilityStatus::KnownUnimplemented(u) => (
                "known_unimplemented",
                serde_json::json!({
                    "family": u.family,
                    "milestone": u.milestone,
                    "note": u.note,
                }),
            ),
            CompatibilityStatus::Unknown => ("unknown", serde_json::Value::Null),
        };
        let (source_kind, arch) = match &self.source {
            CompatSource::GgufArch(s) => ("gguf", s.as_str()),
            CompatSource::SafetensorsConfig(s) => ("safetensors_config", s.as_str()),
        };
        let gguf_fields = self.gguf_fields.as_ref().map(|f| {
            serde_json::json!({
                "context_length": f.context_length,
                "embedding_length": f.embedding_length,
                "block_count": f.block_count,
                "tokenizer_model": f.tokenizer_model,
                "has_tokens": f.has_tokens,
            })
        });
        serde_json::json!({
            "path": self.path.display().to_string(),
            "source": source_kind,
            "arch": arch,
            "status": status_tag,
            "detail": status_detail,
            "gguf_fields": gguf_fields,
        })
        .to_string()
    }
}

impl std::fmt::Display for CompatibilityReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "path:     {}", self.path.display())?;
        match &self.source {
            CompatSource::GgufArch(a) => {
                writeln!(f, "source:   GGUF general.architecture = `{a}`")?;
            }
            CompatSource::SafetensorsConfig(a) => {
                writeln!(f, "source:   safetensors config.json model_type = `{a}`")?;
            }
        }
        if let Some(fields) = &self.gguf_fields {
            writeln!(f, "fields:")?;
            writeln!(
                f,
                "  context_length:   {}",
                opt_display(fields.context_length)
            )?;
            writeln!(
                f,
                "  embedding_length: {}",
                opt_display(fields.embedding_length)
            )?;
            writeln!(f, "  block_count:      {}", opt_display(fields.block_count))?;
            writeln!(
                f,
                "  tokenizer.model:  {}",
                fields.tokenizer_model.as_deref().unwrap_or("<missing>")
            )?;
            writeln!(
                f,
                "  tokens:           {}",
                if fields.has_tokens {
                    "present"
                } else {
                    "<missing>"
                }
            )?;
        }
        match &self.status {
            CompatibilityStatus::Supported { runner } => {
                writeln!(f, "status:   SUPPORTED")?;
                writeln!(f, "runner:   {runner}")?;
                writeln!(
                    f,
                    "          rlx-run {runner} --weights {}",
                    self.path.display()
                )?;
            }
            CompatibilityStatus::MissingMetadata { missing } => {
                writeln!(f, "status:   INCOMPATIBLE (missing required GGUF metadata)")?;
                writeln!(f, "missing:  {}", missing.join(", "))?;
                writeln!(
                    f,
                    "note:     llama.cpp would also reject this file at load time"
                )?;
            }
            CompatibilityStatus::KnownUnimplemented(u) => {
                writeln!(f, "status:   NOT YET IMPLEMENTED")?;
                writeln!(f, "family:   {}", u.family)?;
                writeln!(f, "blocked by: PLAN.md {}", u.milestone)?;
                writeln!(f, "note:     {}", u.note)?;
            }
            CompatibilityStatus::Unknown => {
                writeln!(f, "status:   UNKNOWN ARCH")?;
                writeln!(
                    f,
                    "note:     arch `{}` is not in rlx-models's recognized set or on PLAN.md",
                    self.source.arch()
                )?;
            }
        }
        Ok(())
    }
}

fn opt_display<T: std::fmt::Display>(v: Option<T>) -> String {
    match v {
        Some(v) => v.to_string(),
        None => "<missing>".to_string(),
    }
}

/// Read a `<arch>.<suffix>` u64 from GGUF metadata.
fn meta_arch_u64(raw: &GgufFile, arch: &str, suffix: &str) -> Option<u64> {
    let k = format!("{arch}.{suffix}");
    raw.metadata.get(&k).and_then(MetaValue::as_u64)
}

fn extract_gguf_fields(raw: &GgufFile, arch: &str) -> GgufRequiredFields {
    let tokenizer_model = raw
        .metadata
        .get("tokenizer.ggml.model")
        .and_then(MetaValue::as_str)
        .map(str::to_owned);
    let has_tokens = matches!(
        raw.metadata.get("tokenizer.ggml.tokens"),
        Some(MetaValue::Array(arr)) if !arr.is_empty()
    );
    GgufRequiredFields {
        context_length: meta_arch_u64(raw, arch, "context_length"),
        embedding_length: meta_arch_u64(raw, arch, "embedding_length"),
        block_count: meta_arch_u64(raw, arch, "block_count"),
        tokenizer_model,
        has_tokens,
    }
}

fn classify(source: &CompatSource, fields: Option<&GgufRequiredFields>) -> CompatibilityStatus {
    let arch = source.arch();
    if let Some(runner) = match source {
        CompatSource::GgufArch(_) => arch_runner_name(arch),
        CompatSource::SafetensorsConfig(_) => model_type_runner_name(arch),
    } {
        // For GGUF we also enforce llama.cpp's load-time field check.
        if let Some(f) = fields {
            let missing = f.missing();
            if !missing.is_empty() {
                return CompatibilityStatus::MissingMetadata { missing };
            }
        }
        return CompatibilityStatus::Supported { runner };
    }
    if let Some(u) = known_unimplemented_arch(arch) {
        return CompatibilityStatus::KnownUnimplemented(u);
    }
    CompatibilityStatus::Unknown
}

/// Compatibility report for a local weights path (file or directory).
pub fn check_path(path: &Path) -> Result<CompatibilityReport> {
    let file = resolve_weights_file(path)?;
    let ext = file.extension().and_then(|s| s.to_str()).unwrap_or("");
    match ext {
        "gguf" => {
            let raw =
                GgufFile::from_path(&file).with_context(|| format!("opening GGUF {file:?}"))?;
            let arch = raw
                .metadata
                .get("general.architecture")
                .and_then(MetaValue::as_str)
                .ok_or_else(|| anyhow!("{file:?}: GGUF has no general.architecture"))?
                .to_string();
            let fields = extract_gguf_fields(&raw, &arch);
            let source = CompatSource::GgufArch(arch);
            let status = classify(&source, Some(&fields));
            Ok(CompatibilityReport {
                path: file,
                source,
                status,
                gguf_fields: Some(fields),
            })
        }
        "safetensors" => {
            let dir = file
                .parent()
                .ok_or_else(|| anyhow!("safetensors path {file:?} has no parent dir"))?;
            let cfg = dir.join("config.json");
            if !cfg.is_file() {
                return Err(anyhow!(
                    "{file:?}: no sidecar config.json — cannot determine model_type"
                ));
            }
            let bytes = std::fs::read(&cfg).with_context(|| format!("reading {cfg:?}"))?;
            let v: serde_json::Value =
                serde_json::from_slice(&bytes).with_context(|| format!("parsing {cfg:?}"))?;
            let model_type = v
                .get("model_type")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| anyhow!("{cfg:?}: missing `model_type`"))?
                .to_string();
            let source = CompatSource::SafetensorsConfig(model_type);
            let status = classify(&source, None);
            Ok(CompatibilityReport {
                path: file,
                source,
                status,
                gguf_fields: None,
            })
        }
        other => Err(anyhow!(
            "{file:?}: unsupported extension `.{other}` (expected .gguf or .safetensors)"
        )),
    }
}

/// Heuristic: does `s` look like a HuggingFace repo id (`org/name`) rather
/// than a local filesystem path? We use the same rule HF Hub clients do:
/// exactly one `/`, no path separators leading the string, no extension
/// suffix.
pub fn looks_like_hf_repo(s: &str) -> bool {
    if s.starts_with('/') || s.starts_with('.') || s.starts_with('~') {
        return false;
    }
    let slashes = s.bytes().filter(|b| *b == b'/').count();
    if slashes != 1 {
        return false;
    }
    let last = s.rsplit_once('/').map(|(_, t)| t).unwrap_or("");
    // Reject obvious file-extension suffixes — `org/model.safetensors` is a
    // path, not a repo.
    !matches!(
        last.rsplit_once('.').map(|(_, ext)| ext),
        Some("gguf") | Some("safetensors") | Some("bin") | Some("pt") | Some("onnx"),
    )
}

/// CLI entry point for `rlx-run check <path-or-repo> [--json]`.
pub fn run_check(args: &[String]) -> Result<()> {
    let mut json = false;
    let mut input: Option<&str> = None;
    for a in args {
        match a.as_str() {
            "--json" => json = true,
            "-h" | "--help" | "help" => {
                println!(
                    "rlx-run check — report whether rlx-models can run a model\n\
                     \n\
                     USAGE:\n  rlx-run check <path-or-repo> [--json]\n\
                     \n\
                     Accepts a local weights path or a HuggingFace repo id\n\
                     (e.g. `unsloth/Qwen3-7B-GGUF`). Mirrors llama.cpp's load-time\n\
                     GGUF field check + HuggingFace's compatibility predicate, so\n\
                     the verdict matches what users see upstream.\n\
                     \n\
                     HF-repo checks require the `compat-net` cargo feature."
                );
                return Ok(());
            }
            other => {
                if input.is_some() {
                    return Err(anyhow!("check: unexpected extra arg `{other}`"));
                }
                input = Some(other);
            }
        }
    }
    let input = input.ok_or_else(|| {
        anyhow!("check: expected a weights path or HF repo id\nusage: rlx-run check <path-or-repo> [--json]")
    })?;
    let report = if looks_like_hf_repo(input) {
        check_hf_repo(input)?
    } else {
        check_path(Path::new(input))?
    };
    if json {
        println!("{}", report.to_json());
    } else {
        print!("{report}");
    }
    if !report.status.is_runnable() {
        return Err(anyhow!("model is not runnable by rlx-models"));
    }
    Ok(())
}

/// Stub returned when the binary was built without the `compat-net` feature.
#[cfg(not(feature = "compat-net"))]
pub fn check_hf_repo(repo: &str) -> Result<CompatibilityReport> {
    Err(anyhow!(
        "{repo}: HF-repo checks require building rlx-cli with the `compat-net` feature \
         (`cargo build -p rlx-cli --features compat-net`)"
    ))
}

#[cfg(feature = "compat-net")]
mod hf_fetch {
    use super::*;
    use std::io::Read;

    /// URL for fetching a file's raw bytes at `main`. Public to support
    /// testing URL construction without a live network.
    pub fn resolve_url(repo: &str, file: &str) -> String {
        format!("https://huggingface.co/{repo}/resolve/main/{file}")
    }

    /// HF Hub API endpoint for listing a repo's files at `main`.
    pub fn tree_api_url(repo: &str) -> String {
        format!("https://huggingface.co/api/models/{repo}/tree/main")
    }

    /// First 4 MB of a `.gguf` is enough to parse the header (magic +
    /// metadata + tensor descriptors) for the catalog models we care
    /// about. The data segment is intentionally truncated.
    pub const GGUF_HEADER_FETCH_BYTES: usize = 4 * 1024 * 1024;

    fn get(url: &str) -> Result<ureq::Response> {
        ureq::get(url)
            .timeout(std::time::Duration::from_secs(30))
            .call()
            .with_context(|| format!("GET {url}"))
    }

    fn get_range(url: &str, end_inclusive: usize) -> Result<Vec<u8>> {
        let resp = ureq::get(url)
            .timeout(std::time::Duration::from_secs(60))
            .set("Range", &format!("bytes=0-{end_inclusive}"))
            .call()
            .with_context(|| format!("GET (range) {url}"))?;
        let mut buf = Vec::with_capacity(end_inclusive + 1);
        resp.into_reader()
            .take((end_inclusive + 1) as u64)
            .read_to_end(&mut buf)
            .with_context(|| format!("reading range body from {url}"))?;
        Ok(buf)
    }

    /// Try `config.json` first (covers safetensors and most modern repos
    /// even when they also ship GGUF). Fall back to GGUF header sniffing.
    pub fn check(repo: &str) -> Result<CompatibilityReport> {
        let cfg_url = resolve_url(repo, "config.json");
        if let Ok(resp) = get(&cfg_url) {
            if resp.status() == 200 {
                let mut bytes = Vec::new();
                resp.into_reader().read_to_end(&mut bytes).ok();
                if let Ok(v) = serde_json::from_slice::<serde_json::Value>(&bytes) {
                    if let Some(model_type) =
                        v.get("model_type").and_then(serde_json::Value::as_str)
                    {
                        let source = CompatSource::SafetensorsConfig(model_type.to_string());
                        let status = classify(&source, None);
                        return Ok(CompatibilityReport {
                            path: PathBuf::from(format!("hf://{repo}/config.json")),
                            source,
                            status,
                            gguf_fields: None,
                        });
                    }
                }
            }
        }

        // No usable config.json — look for a .gguf in the repo tree.
        let tree_url = tree_api_url(repo);
        let resp = get(&tree_url)?;
        if resp.status() != 200 {
            return Err(anyhow!(
                "{repo}: HF tree API returned status {} (is the repo public?)",
                resp.status()
            ));
        }
        let listing: serde_json::Value = resp
            .into_json()
            .with_context(|| format!("parsing HF tree JSON for {repo}"))?;
        let arr = listing
            .as_array()
            .ok_or_else(|| anyhow!("{repo}: HF tree API did not return a JSON array"))?;
        let gguf_path = arr
            .iter()
            .filter_map(|v| v.get("path").and_then(serde_json::Value::as_str))
            .find(|p| p.ends_with(".gguf"))
            .ok_or_else(|| {
                anyhow!(
                    "{repo}: no config.json with model_type and no .gguf file at root — \
                     cannot determine architecture"
                )
            })?
            .to_owned();

        let gguf_url = resolve_url(repo, &gguf_path);
        let bytes = get_range(&gguf_url, GGUF_HEADER_FETCH_BYTES - 1)?;
        let mut cursor = std::io::Cursor::new(bytes);
        let raw = GgufFile::from_reader(&mut cursor)
            .with_context(|| format!("parsing GGUF header from {gguf_url}"))?;
        let arch = raw
            .metadata
            .get("general.architecture")
            .and_then(MetaValue::as_str)
            .ok_or_else(|| anyhow!("{gguf_url}: GGUF has no general.architecture"))?
            .to_string();
        let fields = extract_gguf_fields(&raw, &arch);
        let source = CompatSource::GgufArch(arch);
        let status = classify(&source, Some(&fields));
        Ok(CompatibilityReport {
            path: PathBuf::from(format!("hf://{repo}/{gguf_path}")),
            source,
            status,
            gguf_fields: Some(fields),
        })
    }
}

/// Check whether a HuggingFace repo is runnable by rlx-models.
///
/// Fetches `config.json` first (one HTTPS GET); falls back to listing the
/// repo tree and Range-fetching the first 4 MB of any `.gguf` file to
/// parse the header — the same approach `huggingface.js/packages/gguf`
/// uses to drive HF Hub's compatibility badge.
///
/// Requires the `compat-net` cargo feature.
#[cfg(feature = "compat-net")]
pub fn check_hf_repo(repo: &str) -> Result<CompatibilityReport> {
    hf_fetch::check(repo)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a GGUF file in a temp path with arbitrary metadata; missing
    /// arch fields are simply omitted so the test can probe the
    /// MissingMetadata path.
    fn write_test_gguf(arch: &str, fields: &[(&str, MetaValueOwned)]) -> PathBuf {
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(&rlx_gguf::GGUF_MAGIC.to_le_bytes());
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&1u64.to_le_bytes()); // tensor count
        let kv_count = 1 + fields.len(); // architecture + the rest
        buf.extend_from_slice(&(kv_count as u64).to_le_bytes());

        write_string_kv(&mut buf, "general.architecture", arch);
        for (k, v) in fields {
            match v {
                MetaValueOwned::Str(s) => write_string_kv(&mut buf, k, s),
                MetaValueOwned::U64(n) => {
                    buf.extend_from_slice(&(k.len() as u64).to_le_bytes());
                    buf.extend_from_slice(k.as_bytes());
                    buf.extend_from_slice(&10u32.to_le_bytes()); // U64 type
                    buf.extend_from_slice(&n.to_le_bytes());
                }
                MetaValueOwned::StringArray(items) => {
                    buf.extend_from_slice(&(k.len() as u64).to_le_bytes());
                    buf.extend_from_slice(k.as_bytes());
                    buf.extend_from_slice(&9u32.to_le_bytes()); // Array
                    buf.extend_from_slice(&8u32.to_le_bytes()); // of String
                    buf.extend_from_slice(&(items.len() as u64).to_le_bytes());
                    for s in items {
                        buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
                        buf.extend_from_slice(s.as_bytes());
                    }
                }
            }
        }
        // one tiny f32 tensor so the GGUF loader accepts the file
        let name = "w";
        buf.extend_from_slice(&(name.len() as u64).to_le_bytes());
        buf.extend_from_slice(name.as_bytes());
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&4u64.to_le_bytes());
        buf.extend_from_slice(&(rlx_gguf::GgmlType::F32 as u32).to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());
        while !buf
            .len()
            .is_multiple_of(rlx_gguf::DEFAULT_ALIGNMENT as usize)
        {
            buf.push(0);
        }
        for _ in 0..4 {
            buf.extend_from_slice(&1.0f32.to_le_bytes());
        }
        use std::sync::atomic::{AtomicU64, Ordering};
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "rlx_compat_{}_{}_{}.gguf",
            arch,
            std::process::id(),
            SEQ.fetch_add(1, Ordering::Relaxed),
        ));
        std::fs::write(&path, &buf).unwrap();
        path
    }

    enum MetaValueOwned {
        Str(String),
        U64(u64),
        StringArray(Vec<String>),
    }

    fn write_string_kv(buf: &mut Vec<u8>, k: &str, v: &str) {
        buf.extend_from_slice(&(k.len() as u64).to_le_bytes());
        buf.extend_from_slice(k.as_bytes());
        buf.extend_from_slice(&8u32.to_le_bytes());
        buf.extend_from_slice(&(v.len() as u64).to_le_bytes());
        buf.extend_from_slice(v.as_bytes());
    }

    #[test]
    fn supported_when_arch_known_and_all_required_fields_present() {
        let path = write_test_gguf(
            "qwen3",
            &[
                ("qwen3.context_length", MetaValueOwned::U64(8192)),
                ("qwen3.embedding_length", MetaValueOwned::U64(4096)),
                ("qwen3.block_count", MetaValueOwned::U64(32)),
                ("tokenizer.ggml.model", MetaValueOwned::Str("gpt2".into())),
                (
                    "tokenizer.ggml.tokens",
                    MetaValueOwned::StringArray(vec!["a".into(), "b".into()]),
                ),
            ],
        );
        let r = check_path(&path).unwrap();
        match r.status {
            CompatibilityStatus::Supported { runner } => assert_eq!(runner, "qwen3"),
            other => panic!("expected Supported, got {other:?}"),
        }
        assert!(r.status.is_runnable());
        assert_eq!(r.gguf_fields.as_ref().unwrap().context_length, Some(8192));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn missing_metadata_when_required_field_absent() {
        // Qwen3 arch is supported, but no block_count → not runnable.
        let path = write_test_gguf(
            "qwen3",
            &[
                ("qwen3.context_length", MetaValueOwned::U64(8192)),
                ("qwen3.embedding_length", MetaValueOwned::U64(4096)),
                ("tokenizer.ggml.model", MetaValueOwned::Str("gpt2".into())),
                (
                    "tokenizer.ggml.tokens",
                    MetaValueOwned::StringArray(vec!["a".into()]),
                ),
            ],
        );
        let r = check_path(&path).unwrap();
        match &r.status {
            CompatibilityStatus::MissingMetadata { missing } => {
                assert!(missing.contains(&"<arch>.block_count"));
            }
            other => panic!("expected MissingMetadata, got {other:?}"),
        }
        assert!(!r.status.is_runnable());
        std::fs::remove_file(&r.path).ok();
    }

    #[test]
    fn known_unimplemented_when_arch_in_plan_but_not_implemented() {
        let path = write_test_gguf(
            "minimax-m2",
            &[
                ("minimax-m2.context_length", MetaValueOwned::U64(8192)),
                ("minimax-m2.embedding_length", MetaValueOwned::U64(4096)),
                ("minimax-m2.block_count", MetaValueOwned::U64(32)),
                ("tokenizer.ggml.model", MetaValueOwned::Str("gpt2".into())),
                (
                    "tokenizer.ggml.tokens",
                    MetaValueOwned::StringArray(vec!["a".into()]),
                ),
            ],
        );
        let r = check_path(&path).unwrap();
        match &r.status {
            CompatibilityStatus::KnownUnimplemented(u) => {
                assert_eq!(u.milestone, "M5");
                assert!(u.family.contains("MiniMax"));
            }
            other => panic!("expected KnownUnimplemented, got {other:?}"),
        }
        assert!(!r.status.is_runnable());
        std::fs::remove_file(&r.path).ok();
    }

    #[test]
    fn unknown_when_arch_not_recognized() {
        let path = write_test_gguf("totally-fake-arch", &[]);
        let r = check_path(&path).unwrap();
        assert!(matches!(r.status, CompatibilityStatus::Unknown));
        std::fs::remove_file(&r.path).ok();
    }

    #[test]
    fn json_round_trip_emits_status_tag() {
        let path = write_test_gguf("totally-fake-arch", &[]);
        let r = check_path(&path).unwrap();
        let j = r.to_json();
        let v: serde_json::Value = serde_json::from_str(&j).unwrap();
        assert_eq!(v["status"], "unknown");
        assert_eq!(v["source"], "gguf");
        assert_eq!(v["arch"], "totally-fake-arch");
        std::fs::remove_file(&r.path).ok();
    }

    #[test]
    fn looks_like_hf_repo_distinguishes_repos_from_paths() {
        assert!(looks_like_hf_repo("unsloth/Qwen3-7B-GGUF"));
        assert!(looks_like_hf_repo("bartowski/something"));
        // Local paths should NOT be flagged as repos
        assert!(!looks_like_hf_repo("/Users/me/model.gguf"));
        assert!(!looks_like_hf_repo("./model.gguf"));
        assert!(!looks_like_hf_repo("~/models/qwen3"));
        assert!(!looks_like_hf_repo("model.gguf"));
        // Multi-segment paths aren't repos
        assert!(!looks_like_hf_repo("models/qwen3/file.gguf"));
        // File-extension suffix → path even if exactly one slash
        assert!(!looks_like_hf_repo("org/file.safetensors"));
        assert!(!looks_like_hf_repo("org/file.gguf"));
    }

    #[cfg(feature = "compat-net")]
    #[test]
    fn hf_url_construction() {
        use super::hf_fetch::{resolve_url, tree_api_url};
        assert_eq!(
            resolve_url("unsloth/Qwen3-7B-GGUF", "config.json"),
            "https://huggingface.co/unsloth/Qwen3-7B-GGUF/resolve/main/config.json"
        );
        assert_eq!(
            tree_api_url("unsloth/Qwen3-7B-GGUF"),
            "https://huggingface.co/api/models/unsloth/Qwen3-7B-GGUF/tree/main"
        );
    }

    #[test]
    fn safetensors_uses_sidecar_model_type() {
        let dir = std::env::temp_dir().join("rlx_compat_st_sidecar");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), br#"{"model_type":"llama"}"#).unwrap();
        let st = dir.join("model.safetensors");
        std::fs::write(&st, b"").unwrap();
        let r = check_path(&st).unwrap();
        match r.status {
            CompatibilityStatus::Supported { runner } => assert_eq!(runner, "llama32"),
            other => panic!("expected Supported, got {other:?}"),
        }
        std::fs::remove_dir_all(&dir).ok();
    }
}
