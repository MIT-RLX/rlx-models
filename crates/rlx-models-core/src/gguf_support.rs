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

//! Shared GGUF helpers for LM runners (architecture checks, path resolution).

use anyhow::{Context, Result, bail};
use rlx_gguf::{GgufFile, MetaValue};
use std::path::{Path, PathBuf};

/// LM families in this workspace that load `.gguf` weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufModelFamily {
    Qwen3,
    Qwen35,
    /// Qwen3-VL / Qwen3-VL-MoE (`qwen3vl` / `qwen3vlmoe`) — LM + mmproj.
    Qwen3Vl,
    Llama32,
    /// Mistral 3+ / Ministral (`mistral3` / `mistral4`) — Llama-shaped via `rlx-mistral`.
    Mistral,
    Gemma,
    /// LiquidAI LFM2.5 text variants (`lfm2` / `lfm` / `lfm25` / `lfm2_5`).
    Lfm,
    /// Thinking Machines Inkling (`inkling` — Unsloth / llama.cpp converters).
    Inkling,
    /// Poolside Laguna MoE (`laguna` — Unsloth / llama.cpp converters).
    Laguna,
}

impl GgufModelFamily {
    pub fn cli_name(self) -> &'static str {
        match self {
            Self::Qwen3 => "rlx-qwen3",
            Self::Qwen35 => "rlx-qwen35",
            Self::Qwen3Vl => "rlx-qwen3-vl",
            Self::Llama32 => "rlx-llama32",
            Self::Mistral => "rlx-mistral",
            Self::Gemma => "rlx-gemma",
            Self::Lfm => "rlx-lfm",
            Self::Inkling => "rlx-inkling",
            Self::Laguna => "rlx-laguna",
        }
    }

    /// Short name this family registers as with `register_cli` in
    /// `rlx_run` (e.g. `"qwen3"`, `"qwen35"`). Used by `auto_runner` to
    /// look up a `ModelRunner` in the registry.
    pub fn runner_name(self) -> &'static str {
        match self {
            Self::Qwen3 => "qwen3",
            Self::Qwen35 => "qwen35",
            Self::Qwen3Vl => "qwen3-vl",
            Self::Llama32 => "llama32",
            Self::Mistral => "mistral",
            Self::Gemma => "gemma",
            Self::Lfm => "lfm",
            Self::Inkling => "inkling",
            Self::Laguna => "laguna",
        }
    }

    pub fn runner_hint(self) -> &'static str {
        match self {
            Self::Qwen3 => "`rlx_models::Qwen3Runner::builder()...build()`",
            Self::Qwen35 => "`rlx_models::Qwen35Runner::builder()...build()`",
            Self::Qwen3Vl => "`rlx_qwen3_vl::Qwen3VlRunner::builder()...build()`",
            Self::Llama32 => "`rlx_models::Llama32Runner::builder()...build()`",
            Self::Mistral => "`rlx_mistral::MistralRunner::builder()...build()`",
            Self::Gemma => "`rlx_models::GemmaRunner::builder()...build()`",
            Self::Lfm => "`rlx_lfm::LfmRunner::builder()...build()`",
            Self::Inkling => "`rlx_inkling::InklingRunner::builder()...build()`",
            Self::Laguna => "`rlx_laguna::LagunaRunner::builder()...build()`",
        }
    }

    fn accepts_arch(self, arch: &str) -> bool {
        match self {
            // Qwen3 family accepts qwen2 (handled via attention_bias +
            // qk_norm flags in qwen3_cfg_from_gguf), plus the qwen3
            // tags. qwen25 / qwen2_5 (Qwen 2.5 / 2.5.1) ship as the
            // `qwen2` arch tag in llama.cpp; we also accept the
            // explicit short tags for safetensors sidecars that name
            // them differently. Used by rlx-omnicoder which delegates
            // here. PLAN.md M4.
            // Also accept qwen3vl* so `Qwen3Runner` can load the LM half of a
            // Qwen3-VL GGUF when composed by `Qwen3VlRunner` (auto_sniff still
            // routes those arches to `Qwen3Vl` via `gguf_family_for_arch`).
            Self::Qwen3 => matches!(
                arch,
                "qwen3"
                    | "qwen2"
                    | "qwen25"
                    | "qwen2_5"
                    | "qwen2vl"
                    | "qwen3vl"
                    | "qwen3vlmoe"
                    | "qwen3_vl"
                    | "qwen3-vl"
            ),
            // Qwen3.6 reuses the Qwen3.5 trunk; `Qwen35Config::from_gguf`
            // reads its metadata keys under the `qwen36.*` prefix. Routes
            // through the same runner. PLAN.md M1.
            Self::Qwen35 => matches!(arch, "qwen35" | "qwen35moe" | "qwen36" | "qwen36moe"),
            Self::Qwen3Vl => matches!(arch, "qwen3vl" | "qwen3vlmoe" | "qwen3_vl" | "qwen3-vl"),
            // Llama32 family accepts the Llama-shaped M4 stub archs
            // (`mistral3`/`mistral4`, `phi3`, `granite`, `command-r`/`cohere2`)
            // so the per-family stub wrappers in
            // `crates/rlx-{mistral,phi,granite,cohere}` can delegate
            // through it. The per-arch *quality* deltas (sliding window,
            // attention.scale, parallel_residual, etc.) are PLAN.md M4
            // follow-up — runner will produce SOME output but it may
            // not match the upstream reference yet.
            // Only `llama` is parity-tested today. The other arch
            // stubs need per-arch tensor-name remap + quality deltas;
            // tracked as M4 follow-up. accepts_arch is checked late by
            // the per-family runner — keeping it lenient lets the
            // per-family stub crates delegate here for compile-only
            // quick-check tests, but `auto_runner` only routes plain `llama`.
            Self::Llama32 => matches!(
                arch,
                "llama"
                    | "phi3"
                    | "phi4"
                    | "granite"
                    | "granitemoe"
                    | "granitehybrid"
                    | "command-r"
                    | "cohere2"
                    // Muse Glimmer runs the full arch deltas on the packed
                    // path (`DenseArch::MuseGlimmer`), not a lenient stub.
                    | "muse-glimmer"
            ),
            Self::Mistral => matches!(arch, "mistral3" | "mistral4"),
            // Gemma 3 / 3n / 4 route through the same family — share
            // the V3+ layer style, RoPE base, and 4-norm sandwich
            // wired in `rlx-gemma`. Gemma 4 unified additionally
            // ships `attention_k_eq_v`, split RoPE per layer kind,
            // and per-layer KV shape; all dispatched off the
            // per-layer accessors on `GemmaConfig`.
            Self::Gemma => matches!(
                arch,
                "gemma"
                    | "gemma2"
                    | "gemma3"
                    | "gemma3n"
                    | "gemma4"
                    | "gemma4moe"
                    | "gemma4_unified"
            ),
            Self::Lfm => matches!(arch, "lfm2" | "lfm" | "lfm25" | "lfm2_5"),
            // Unsloth / llama.cpp: `general.architecture = inkling`
            Self::Inkling => matches!(arch, "inkling" | "inkling_mm_model"),
            // Unsloth / llama.cpp: `general.architecture = laguna` (PR #25165)
            Self::Laguna => matches!(arch, "laguna"),
        }
    }
}

/// `general.architecture` string from GGUF metadata, if present.
pub fn gguf_architecture_str(file: &GgufFile) -> Option<&str> {
    file.metadata
        .get("general.architecture")
        .and_then(MetaValue::as_str)
}

/// Rough F32 dequant footprint (every tensor × 4 bytes).
pub fn gguf_f32_bytes_estimate(file: &GgufFile) -> u64 {
    file.tensors
        .values()
        .map(|t| (t.n_elements() as u64) * 4)
        .sum()
}

/// Ensure a GGUF file's `general.architecture` is in `allowed` (call from model runners, not the loader).
pub fn gguf_validate_arch(path: &Path, allowed: &[&str]) -> Result<()> {
    let arch = gguf_architecture_from_path(path)?;
    if allowed.contains(&arch.as_str()) {
        return Ok(());
    }
    bail!(
        "{path:?}: GGUF architecture `{arch}` not in [{}]",
        allowed.join(", ")
    );
}

/// `general.architecture` for a GGUF file on disk.
///
/// Header-only — never slurps the tensor-data segment (safe for 20GB+ Laguna
/// / large K-quant files when callers only need the arch tag).
pub fn gguf_architecture_from_path(path: &Path) -> Result<String> {
    let raw = GgufFile::header_from_path(path)
        .with_context(|| format!("opening GGUF header {path:?}"))?;
    Ok(gguf_architecture_str(&raw).unwrap_or("unknown").to_string())
}

/// User-facing hint when a runner only accepts safetensors but a GGUF path was given.
pub fn gguf_safetensors_only_hint(runner: &str, path: &Path, arch: &str) -> String {
    let mut msg = format!(
        "{runner}: {path:?} is GGUF (architecture `{arch}`); this runner expects safetensors"
    );
    if crate::gguf_config::is_embed_gguf_arch(arch) {
        msg.push_str(". Load embedding GGUF with `rlx-embed` or `RlxEmbed::from_weights`");
    } else if crate::gguf_config::is_flux_gguf_arch(arch) {
        msg.push_str(
            ". Load FLUX denoiser GGUF with `rlx-flux2` (`Flux2Runner::builder().weights(path)`) \
             — e.g. unsloth/FLUX.2-klein-9B-GGUF; VAE/text encoder stay separate safetensors",
        );
    } else if crate::gguf_config::is_dinov2_gguf_arch(arch) {
        msg.push_str(
            ". Load DINOv2 GGUF with `rlx-dinov2` (`DinoV2Runner::builder().weights(path)`)",
        );
    } else if crate::gguf_config::is_sam3_gguf_arch(arch) {
        msg.push_str(". Load SAM3 GGUF with `rlx-sam3` (`Sam3::from_checkpoint_on`)");
    } else if crate::gguf_config::is_sam2_gguf_arch(arch) {
        msg.push_str(". Load SAM2 GGUF with `rlx-sam2`");
    } else if crate::gguf_config::is_sam_gguf_arch(arch) {
        msg.push_str(". Load SAM / MobileSAM GGUF with `rlx-sam`");
    } else if crate::gguf_config::is_vjepa2_gguf_arch(arch) {
        msg.push_str(". Load V-JEPA2 GGUF with `rlx-vjepa2`");
    } else if crate::gguf_config::is_w2v_bert_gguf_arch(arch) {
        msg.push_str(". Load with `rlx-wav2vec2-bert` (sidecar `config.json` still required)");
    } else if let Some(fam) = gguf_family_for_arch(arch) {
        msg.push_str(&format!(
            ". Use `{}` ({}) instead",
            fam.cli_name(),
            fam.runner_hint()
        ));
    } else {
        msg.push_str(". See README Status → Weights for GGUF coverage per family");
    }
    msg
}

/// Map a GGUF architecture tag to the runner family that should load it.
///
/// Delegates to [`crate::model_registry`] (built-ins + any third-party
/// [`crate::model_registry::register_gguf_model`] calls).
pub fn gguf_family_for_arch(arch: &str) -> Option<GgufModelFamily> {
    crate::model_registry::family_for_gguf_arch(arch)
}

/// Open the file and ensure `general.architecture` matches `expected`.
///
/// For [`GgufModelFamily::Laguna`], opens **header-only** (F32 expand and
/// full payload slurp are forbidden for that family). Other families keep
/// the historical full open.
pub fn assert_gguf_family(path: &Path, expected: GgufModelFamily) -> Result<GgufFile> {
    let raw = if expected == GgufModelFamily::Laguna {
        GgufFile::header_from_path(path)
            .with_context(|| format!("opening Laguna GGUF header {path:?}"))?
    } else {
        GgufFile::from_path(path).with_context(|| format!("opening GGUF {path:?}"))?
    };
    let arch = gguf_architecture_str(&raw).unwrap_or("unknown");
    if expected.accepts_arch(arch) {
        return Ok(raw);
    }
    if let Some(actual) = gguf_family_for_arch(arch) {
        bail!(
            "{path:?} is a {arch} GGUF (family {actual:?}). Use `{}` or {} instead.",
            actual.cli_name(),
            actual.runner_hint()
        );
    }
    bail!(
        "{path:?} has general.architecture={arch:?}; this runner expects a {} GGUF",
        expected.cli_name()
    );
}

/// Default quant substring when resolving a directory of `.gguf` files (e.g. unsloth layouts).
pub const DEFAULT_GGUF_PREFER_SUBSTR: &str = "Q4_K_M";

/// Options for [`resolve_weights_file_with_options`].
#[derive(Debug, Clone, Default)]
pub struct ResolveWeightsOptions<'a> {
    /// Prefer a `.gguf` whose file name contains this substring (e.g. `Q4_K_M`).
    pub prefer_gguf_substring: Option<&'a str>,
    /// Pick the N-th `.gguf` after sorting paths (0-based).
    pub gguf_index: Option<usize>,
}

impl<'a> ResolveWeightsOptions<'a> {
    pub fn prefer_substring(mut self, sub: &'a str) -> Self {
        self.prefer_gguf_substring = Some(sub);
        self
    }

    pub fn index(mut self, idx: usize) -> Self {
        self.gguf_index = Some(idx);
        self
    }
}

/// Resolve `--weights` to a single file: pass-through for files, or pick one
/// `.gguf` / `model.safetensors` inside a directory.
pub fn resolve_weights_file(path: &Path) -> Result<PathBuf> {
    resolve_weights_file_with_options(path, &ResolveWeightsOptions::default())
}

/// Resolve with optional GGUF file selection inside a directory.
///
/// When `prefer_gguf_substring` is set and no shallow `.gguf` matches (Unsloth /
/// Laguna-style layouts with `UD-Q4_K_M/*.gguf` under the repo root), scans
/// **one level** of child directories. Among matches, prefers `*-00001-of-*.gguf`
/// (first multi-part shard).
pub fn resolve_weights_file_with_options(
    path: &Path,
    opts: &ResolveWeightsOptions<'_>,
) -> Result<PathBuf> {
    if path.is_file() {
        return Ok(path.to_path_buf());
    }
    if !path.is_dir() {
        bail!("weights path not found: {path:?}");
    }
    let mut ggufs = list_gguf_files_in_dir(path)?;
    if let Some(sub) = opts.prefer_gguf_substring {
        let preferred = filter_gguf_by_substring(&ggufs, sub);
        if !preferred.is_empty() {
            ggufs = prefer_first_split_shard(preferred);
        } else {
            let nested = list_gguf_files_in_child_dirs(path, Some(sub))?;
            if !nested.is_empty() {
                ggufs = prefer_first_split_shard(nested);
            }
        }
    }
    match ggufs.len() {
        1 => return Ok(ggufs[0].clone()),
        n if n > 1 => {
            if let Some(idx) = opts.gguf_index {
                return ggufs.get(idx).cloned().ok_or_else(|| {
                    anyhow::anyhow!(
                        "gguf_index={idx} out of range; directory {path:?} has {n} .gguf files"
                    )
                });
            }
            let listing: Vec<String> = ggufs
                .iter()
                .map(|p| format!("  - {}", p.display()))
                .collect();
            bail!(
                "directory {path:?} contains {n} .gguf files; pass the exact file path, \
                 use LoadOpts::map().prefer_q4_k_m() / prefer_substring(\"Q4_K_M\"), \
                 gguf_index(0), or run `rlx-inspect {path:?} --prefer Q4_K_M`:\n{}",
                listing.join("\n")
            );
        }
        _ => {}
    }
    let st = path.join("model.safetensors");
    if st.is_file() {
        return Ok(st);
    }
    let index = path.join("model.safetensors.index.json");
    if index.is_file() {
        return Ok(path.to_path_buf());
    }
    bail!(
        "directory {path:?} has no .gguf file and no model.safetensors \
         (or model.safetensors.index.json); pass a .gguf or .safetensors path"
    );
}

/// Sorted `.gguf` paths in a directory (non-recursive).
pub fn list_gguf_files_in_dir(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut ggufs = Vec::new();
    for entry in std::fs::read_dir(dir).with_context(|| format!("reading dir {dir:?}"))? {
        let entry = entry?;
        let p = entry.path();
        if p.extension().and_then(|s| s.to_str()) == Some("gguf") && p.is_file() {
            ggufs.push(p);
        }
    }
    ggufs.sort();
    Ok(ggufs)
}

fn filter_gguf_by_substring(ggufs: &[PathBuf], sub: &str) -> Vec<PathBuf> {
    ggufs
        .iter()
        .filter(|p| {
            p.file_name()
                .and_then(|s| s.to_str())
                .is_some_and(|n| n.contains(sub))
                || p.to_string_lossy().contains(sub)
        })
        .cloned()
        .collect()
}

/// `.gguf` files in immediate child directories of `dir`, optionally filtered by
/// substring in the file name or relative path (Unsloth quant folders).
fn list_gguf_files_in_child_dirs(dir: &Path, prefer: Option<&str>) -> Result<Vec<PathBuf>> {
    let mut ggufs = Vec::new();
    for entry in std::fs::read_dir(dir).with_context(|| format!("reading dir {dir:?}"))? {
        let entry = entry?;
        let child = entry.path();
        if !child.is_dir() {
            continue;
        }
        for p in list_gguf_files_in_dir(&child)? {
            if let Some(sub) = prefer {
                let name_ok = p
                    .file_name()
                    .and_then(|s| s.to_str())
                    .is_some_and(|n| n.contains(sub));
                let path_ok = p.to_string_lossy().contains(sub);
                if !(name_ok || path_ok) {
                    continue;
                }
            }
            ggufs.push(p);
        }
    }
    ggufs.sort();
    Ok(ggufs)
}

fn is_first_split_shard_name(name: &str) -> bool {
    name.contains("-00001-of-") || name.contains("-0001-of-")
}

/// If any path looks like a multi-part first shard, keep only those; a single
/// `00001` wins immediately (typical Unsloth 3-way split under one quant dir).
fn prefer_first_split_shard(ggufs: Vec<PathBuf>) -> Vec<PathBuf> {
    let firsts: Vec<PathBuf> = ggufs
        .iter()
        .filter(|p| {
            p.file_name()
                .and_then(|s| s.to_str())
                .is_some_and(is_first_split_shard_name)
        })
        .cloned()
        .collect();
    if firsts.is_empty() { ggufs } else { firsts }
}

/// Other parts of the same multi-file GGUF split in `path`'s directory (sorted by `split.no`).
pub fn gguf_split_siblings(path: &Path) -> Result<Option<Vec<PathBuf>>> {
    let raw = GgufFile::from_path(path).with_context(|| format!("opening GGUF {path:?}"))?;
    let count = raw
        .metadata
        .get("split.count")
        .and_then(MetaValue::as_u32)
        .unwrap_or(1);
    if count <= 1 {
        return Ok(None);
    }
    let dir = path.parent().unwrap_or_else(|| Path::new("."));
    let mut parts: Vec<(u32, PathBuf)> = Vec::new();
    for candidate in list_gguf_files_in_dir(dir)? {
        let other = GgufFile::from_path(&candidate)
            .with_context(|| format!("opening split candidate {candidate:?}"))?;
        let other_count = other
            .metadata
            .get("split.count")
            .and_then(MetaValue::as_u32)
            .unwrap_or(1);
        if other_count != count {
            continue;
        }
        let no = other
            .metadata
            .get("split.no")
            .and_then(MetaValue::as_u32)
            .unwrap_or(0);
        parts.push((no, candidate));
    }
    parts.sort_by_key(|(no, _)| *no);
    parts.dedup_by_key(|(no, _)| *no);
    Ok(Some(parts.into_iter().map(|(_, p)| p).collect()))
}

/// Open a GGUF file, merging multi-part splits when all siblings are present in the directory.
/// True when the low-footprint compile/load mode is requested via
/// `RLX_LOW_MEM_COMPILE`. In this mode large GGUFs are `mmap`-ed instead
/// of slurped, weights are stream-uploaded and dropped, and redundant
/// F32 param copies are elided — trading a little speed for a much lower
/// peak RSS (targets constrained machines / iOS-class RAM).
pub fn low_mem_compile() -> bool {
    matches!(
        std::env::var("RLX_LOW_MEM_COMPILE").ok().as_deref(),
        Some("1") | Some("true") | Some("on") | Some("yes")
    )
}

/// Opt-in Laguna quant→F32 expand. **Off by default.**
///
/// Set `RLX_LAGUNA_ALLOW_F32_EXPAND` to `1` / `true` / `on` / `yes`, or pass
/// `rlx-laguna --allow-f32-expand`. Prefer packed mmap; use `--weights` sniff
/// for the per-file `F32-expand≈` estimate before draining.
pub fn laguna_allow_f32_expand() -> bool {
    matches!(
        std::env::var("RLX_LAGUNA_ALLOW_F32_EXPAND").ok().as_deref(),
        Some("1") | Some("true") | Some("on") | Some("yes")
    )
}

pub fn load_gguf_file(path: &Path) -> Result<GgufFile> {
    // Laguna: never Owned-slurp Q4/IQ payloads into RSS. Always mmap (or
    // header-only for split discovery). F32 expand is separately forbidden
    // in `GgufLoader::take`.
    let header = GgufFile::header_from_path(path)
        .with_context(|| format!("opening GGUF header {path:?}"))?;
    let arch = gguf_architecture_str(&header).unwrap_or("unknown");

    // Always mmap the data segment (paged on demand, reclaimable) instead of
    // slurping the whole multi-GB payload into RSS — the file bytes then back
    // `tensor_bytes_borrowed` zero-copy and never count as resident anon memory.
    // Laguna *requires* mmap (F32 expand / full slurp forbidden); every other
    // arch prefers it and falls back to an owned read only when the platform
    // can't mmap. Multi-part splits fall through to the owned merge path below
    // (mmap is single-file); `low_mem_compile()` no longer gates this — mmap is
    // now the default everywhere.
    let raw = match GgufFile::from_path_mmap(path) {
        Ok(f) => f,
        Err(e) if arch == "laguna" => {
            bail!(
                "Laguna GGUF must be mmap'd (F32 expand / full RSS slurp forbidden): \
                 {path:?}: {e:#}"
            );
        }
        Err(_e) => GgufFile::from_path(path).with_context(|| format!("opening GGUF {path:?}"))?,
    };
    let count = raw
        .metadata
        .get("split.count")
        .and_then(MetaValue::as_u32)
        .unwrap_or(1);
    if count <= 1 {
        return Ok(raw);
    }
    if arch == "laguna" {
        bail!(
            "Laguna multi-part GGUF merge into an owned buffer is FORBIDDEN \
             ({path:?} is split.count={count}): that would RSS-slurp Q4 payloads. \
             Use a single-file quant (e.g. XS Q4_K_M) or a future per-shard mmap packed path."
        );
    }
    let siblings = gguf_split_siblings(path)?;
    match siblings {
        Some(parts) if parts.len() as u32 == count => {
            eprintln!(
                "[rlx-core] merging {count} GGUF split parts from {:?}",
                path.parent().unwrap_or(path)
            );
            GgufFile::from_split_paths(&parts)
        }
        _ => {
            let hint = gguf_split_hint(path)?.unwrap_or_else(|| {
                format!("{path:?} is a split GGUF but sibling parts are missing")
            });
            bail!("{hint}");
        }
    }
}

/// If the file is a multi-part GGUF split, return an error hint when merge is not possible.
pub fn gguf_split_hint(path: &Path) -> Result<Option<String>> {
    let raw = GgufFile::from_path(path).with_context(|| format!("opening GGUF {path:?}"))?;
    let count = raw
        .metadata
        .get("split.count")
        .and_then(MetaValue::as_u32)
        .unwrap_or(1);
    if count <= 1 {
        return Ok(None);
    }
    let no = raw
        .metadata
        .get("split.no")
        .and_then(MetaValue::as_u32)
        .unwrap_or(0);
    let mut msg = format!(
        "{path:?} is part {no} of a {count}-file GGUF split; \
         place all parts in the same directory (auto-merge) or use a single-file quant"
    );
    if let Some(siblings) = gguf_split_siblings(path)? {
        msg.push_str("\nSplit parts in this directory:");
        for (i, p) in siblings.iter().enumerate() {
            msg.push_str(&format!("\n  [{i}] {}", p.display()));
        }
    }
    Ok(Some(msg))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embed_arch_list_includes_bert_and_nomic() {
        assert!(crate::gguf_config::is_embed_gguf_arch("bert"));
        assert!(crate::gguf_config::is_embed_gguf_arch("nomic-bert"));
        assert!(!crate::gguf_config::is_embed_gguf_arch("sam3"));
    }

    #[test]
    fn vision_arch_tags() {
        assert!(crate::gguf_config::is_sam3_gguf_arch("sam3"));
        assert!(crate::gguf_config::is_dinov2_gguf_arch("dinov2"));
        assert!(crate::gguf_config::is_sam_gguf_arch("mobile-sam"));
        assert!(crate::gguf_config::is_w2v_bert_gguf_arch("wav2vec2"));
        assert!(!crate::gguf_config::is_sam3_gguf_arch("dinov2"));
    }

    #[test]
    fn gguf_validate_arch_accepts_and_rejects() {
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(&rlx_gguf::GGUF_MAGIC.to_le_bytes());
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&1u64.to_le_bytes());
        buf.extend_from_slice(&1u64.to_le_bytes());
        let write_string = |buf: &mut Vec<u8>, k: &str, v: &str| {
            buf.extend_from_slice(&(k.len() as u64).to_le_bytes());
            buf.extend_from_slice(k.as_bytes());
            buf.extend_from_slice(&8u32.to_le_bytes());
            buf.extend_from_slice(&(v.len() as u64).to_le_bytes());
            buf.extend_from_slice(v.as_bytes());
        };
        write_string(&mut buf, "general.architecture", "dinov2");
        let write_tensor = |buf: &mut Vec<u8>, name: &str, shape: &[usize], off: u64| {
            buf.extend_from_slice(&(name.len() as u64).to_le_bytes());
            buf.extend_from_slice(name.as_bytes());
            buf.extend_from_slice(&(shape.len() as u32).to_le_bytes());
            for &d in shape {
                buf.extend_from_slice(&(d as u64).to_le_bytes());
            }
            buf.extend_from_slice(&(rlx_gguf::GgmlType::F32 as u32).to_le_bytes());
            buf.extend_from_slice(&off.to_le_bytes());
        };
        write_tensor(&mut buf, "w", &[4], 0);
        while !buf
            .len()
            .is_multiple_of(rlx_gguf::DEFAULT_ALIGNMENT as usize)
        {
            buf.push(0);
        }
        for _ in 0..4 {
            buf.extend_from_slice(&1.0f32.to_le_bytes());
        }
        let path = std::env::temp_dir().join("rlx_gguf_validate_arch_test.gguf");
        std::fs::write(&path, &buf).unwrap();
        gguf_validate_arch(&path, crate::gguf_config::DINOV2_GGUF_ARCHES).expect("dinov2 ok");
        let err = gguf_validate_arch(&path, crate::gguf_config::SAM3_GGUF_ARCHES)
            .expect_err("wrong family");
        assert!(format!("{err:#}").contains("dinov2"));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn family_for_arch_maps_known_tags() {
        assert_eq!(
            gguf_family_for_arch("qwen35"),
            Some(GgufModelFamily::Qwen35)
        );
        assert_eq!(
            gguf_family_for_arch("qwen35moe"),
            Some(GgufModelFamily::Qwen35)
        );
        assert_eq!(
            gguf_family_for_arch("qwen36"),
            Some(GgufModelFamily::Qwen35)
        );
        assert_eq!(
            gguf_family_for_arch("qwen36moe"),
            Some(GgufModelFamily::Qwen35)
        );
        assert_eq!(
            gguf_family_for_arch("llama"),
            Some(GgufModelFamily::Llama32)
        );
        assert_eq!(gguf_family_for_arch("gemma"), Some(GgufModelFamily::Gemma));
        assert_eq!(gguf_family_for_arch("gemma2"), Some(GgufModelFamily::Gemma));
        assert_eq!(gguf_family_for_arch("gemma3"), Some(GgufModelFamily::Gemma));
        assert_eq!(
            gguf_family_for_arch("gemma3n"),
            Some(GgufModelFamily::Gemma)
        );
        assert_eq!(gguf_family_for_arch("gemma4"), Some(GgufModelFamily::Gemma));
        assert_eq!(
            gguf_family_for_arch("gemma4moe"),
            Some(GgufModelFamily::Gemma)
        );
        assert_eq!(
            gguf_family_for_arch("gemma4_unified"),
            Some(GgufModelFamily::Gemma)
        );
        assert_eq!(
            gguf_family_for_arch("inkling"),
            Some(GgufModelFamily::Inkling)
        );
        assert_eq!(
            gguf_family_for_arch("laguna"),
            Some(GgufModelFamily::Laguna)
        );
        assert!(gguf_family_for_arch("clip").is_none());
    }

    #[test]
    fn gemma_family_accepts_gemma4_variants() {
        let fam = GgufModelFamily::Gemma;
        assert!(fam.accepts_arch("gemma"));
        assert!(fam.accepts_arch("gemma2"));
        assert!(fam.accepts_arch("gemma3"));
        assert!(fam.accepts_arch("gemma3n"));
        assert!(fam.accepts_arch("gemma4"));
        assert!(fam.accepts_arch("gemma4moe"));
        assert!(fam.accepts_arch("gemma4_unified"));
        assert!(!fam.accepts_arch("llama"));
        assert!(!fam.accepts_arch("qwen3"));
    }

    #[test]
    fn resolve_nested_unsloth_prefer_first_shard() {
        let root = std::env::temp_dir().join(format!(
            "rlx-gguf-nested-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let quant = root.join("UD-Q4_K_M");
        std::fs::create_dir_all(&quant).unwrap();
        // Top-level IQ (should be ignored when preferring Q4_K_M).
        std::fs::write(root.join("Laguna-S-2.1-UD-IQ1_M.gguf"), b"GGUF").unwrap();
        let shard1 = quant.join("Laguna-S-2.1-UD-Q4_K_M-00001-of-00003.gguf");
        let shard2 = quant.join("Laguna-S-2.1-UD-Q4_K_M-00002-of-00003.gguf");
        let shard3 = quant.join("Laguna-S-2.1-UD-Q4_K_M-00003-of-00003.gguf");
        for p in [&shard1, &shard2, &shard3] {
            std::fs::write(p, b"GGUF").unwrap();
        }
        let got = resolve_weights_file_with_options(
            &root,
            &ResolveWeightsOptions::default().prefer_substring("Q4_K_M"),
        )
        .expect("resolve nested");
        assert_eq!(got, shard1);
        let _ = std::fs::remove_dir_all(&root);
    }
}
