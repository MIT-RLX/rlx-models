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

//! Converts an installed bundle into the three open formats at once.
//!
//! | format | what it holds | who reads it |
//! | --- | --- | --- |
//! | `.safetensors` | one file per graph, f32 | `rlx_core::weight_map::WeightMap` |
//! | `.gguf` | one file per bundle, every graph's tensors + the graph JSON as metadata | the llama.cpp family, `rlx-gguf` |
//! | `.rlxp` | everything above plus the tokenizer, normalizer and configs | `rlx_assets::pack` |
//!
//! All three come from the same `export::stage_graph` staging, so they
//! cannot drift apart in content — only in encoding.
//!
//! # Why F16 is not lossy here
//!
//! Every linear weight is `W_int8 / w_quantization_scale`: at most 255 distinct
//! values spanning one scalar range. F16 carries 11 significant bits, so its
//! relative error (~5e-4) sits an order of magnitude below the int8 step the
//! weights were already quantized to (1/127 ≈ 7.9e-3). Storing them as F32
//! records precision the shipped model never had, so GGUF defaults to F16 and
//! halves the file. The embedding table is the one genuinely-f32 tensor (it
//! comes from the 4-percentile curve, not a scalar scale) and is written at
//! whatever dtype is asked for.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use rlx_gguf::{GgmlType, GgufWriter, MetaValue};

use crate::espresso::Manifest;
use crate::export::{ExportedGraph, Staged, stage_graph, write_safetensors};
use crate::net::Graph;

/// What one converted bundle produced.
#[derive(Debug, Clone, serde::Serialize)]
pub struct BundleReport {
    /// Directory name the bundle was written under.
    pub name: String,
    /// Target languages the bundle covers.
    pub languages: Vec<String>,
    /// Graphs converted.
    pub graphs: Vec<String>,
    /// Graphs named by the manifest but not installed on this machine.
    pub missing: Vec<String>,
    pub tensors: usize,
    pub elements: usize,
    pub safetensors_bytes: u64,
    pub gguf_bytes: u64,
    pub rlxp_bytes: u64,
}

/// Files carried alongside the weights: everything needed to run the pipeline.
const SIDECARS: &[&str] = &[
    "spm.model",
    "normalizer.pat",
    "tokenizer.pat",
    "gender_defaults_list",
    "pyespresso.mdl.bin",
];

/// Resolves `name` in `home` first, then in any other installed directory.
///
/// Per-language graphs ship in a separate `partial-<lang>` asset, so they are
/// legitimately elsewhere; the *shared* graphs must come from `home`, because
/// every bundle has a file called `embedding.espresso.net` and picking the
/// wrong one pairs a decoder with a different vocabulary's embedding.
fn locate(home: &Path, dirs: &[PathBuf], name: &str) -> Option<PathBuf> {
    if home.join(name).exists() {
        return Some(home.to_path_buf());
    }
    dirs.iter().find(|d| d.join(name).exists()).cloned()
}

/// Converts one bundle into `<out>/<name>/` and packs it.
///
/// `home` is the directory holding the bundle's own `pyespresso.mdl.bin`;
/// `dirs` is every installed MT directory, used only to find the per-language
/// graphs. `dtype` is the GGUF encoding.
pub fn convert_bundle(
    home: &Path,
    dirs: &[PathBuf],
    out: &Path,
    name: &str,
    dtype: GgmlType,
) -> Result<BundleReport> {
    let manifest = Manifest::load(home.join("pyespresso.mdl.bin"))
        .with_context(|| format!("reading the manifest in {}", home.display()))?;
    let languages: Vec<String> = manifest.languages().iter().map(|s| s.to_string()).collect();

    let dir = out.join(name);
    std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;

    // Every graph the manifest names, for every language it covers. Languages
    // whose `partial-` asset is not installed are reported, not skipped
    // silently — a bundle missing a decoder cannot translate into it.
    let mut wanted: Vec<String> = Vec::new();
    for lang in &languages {
        for f in manifest.graph_files_for(lang) {
            if !wanted.contains(&f) {
                wanted.push(f);
            }
        }
    }

    let mut gguf = GgufWriter::new();
    gguf.set_arch("quasar-nmt");
    let mut descs: BTreeMap<String, ExportedGraph> = BTreeMap::new();
    let mut graphs_done: Vec<String> = Vec::new();
    let mut missing: Vec<String> = Vec::new();
    let (mut n_tensors, mut n_elems) = (0usize, 0usize);
    let mut st_bytes = 0u64;

    for net in &wanted {
        let Some(src) = locate(home, dirs, net) else {
            missing.push(net.clone());
            continue;
        };
        let g = Graph::load(&src, net).with_context(|| format!("loading {net}"))?;
        let (tensors, desc) = stage_graph(&g).with_context(|| format!("staging {net}"))?;

        let st = dir.join(format!("{}.safetensors", g.name));
        write_safetensors(&tensors, &st)?;
        st_bytes += std::fs::metadata(&st).map(|m| m.len()).unwrap_or(0);

        add_to_gguf(&mut gguf, &g.name, &tensors, dtype)?;
        n_tensors += tensors.len();
        n_elems += tensors.values().map(|t| t.data.len()).sum::<usize>();
        descs.insert(g.name.clone(), desc);
        graphs_done.push(g.name.clone());
    }
    if graphs_done.is_empty() {
        bail!("no graphs of {name} are installed");
    }

    // The graph description travels inside the GGUF as metadata, so a `.gguf`
    // on its own is enough to rebuild the model — the tensors alone would not
    // say how they connect.
    let model = serde_json::json!({
        "format": "quasar-nmt",
        "bundle": name,
        "languages": languages,
        "state_width": manifest.int("StateWidth"),
        // Inferred from `StateStrings`, not stored: there is no
        // `DecoderLayers` key, so asking for one wrote `null` into every
        // descriptor this crate has ever exported.
        "decoder_layers": manifest.decoder_layers(),
        "graphs": descs,
    });
    let model_json = serde_json::to_string_pretty(&model)?;
    std::fs::write(dir.join("model.json"), &model_json)?;
    gguf.set_meta("quasar.model_json", MetaValue::String(model_json));
    gguf.set_meta("quasar.bundle", MetaValue::String(name.to_string()));
    gguf.set_meta(
        "quasar.languages",
        MetaValue::Array(
            languages
                .iter()
                .map(|l| MetaValue::String(l.clone()))
                .collect(),
        ),
    );

    for f in SIDECARS {
        if let Some(src) = locate(home, dirs, f) {
            std::fs::copy(src.join(f), dir.join(f)).with_context(|| format!("copying {f}"))?;
        }
    }
    // Shortlists gate decoding quality, so a bundle without them translates
    // into the wrong variant rather than failing. Copy every table beside the
    // graphs rather than deriving names: `shortlist-lang-pair` is usually
    // `all-<lang>` but not always, and building the name from the language
    // silently dropped `all-zh_TW` — the one table whose name is different,
    // and the one whose absence costs the most.
    let sl = dir.join("shortlists");
    let mut tables: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for d in std::iter::once(home).chain(dirs.iter().map(PathBuf::as_path)) {
        let Ok(rd) = std::fs::read_dir(d.join("shortlists")) else {
            continue;
        };
        for ent in rd.flatten() {
            let p = ent.path();
            if p.extension().and_then(|e| e.to_str()) != Some("shortlist") {
                continue;
            }
            let Some(name) = p.file_name().and_then(|n| n.to_str()) else {
                continue;
            };
            // `all-<lang>`, and any regional variant of it. Building the name
            // as `all-<lang>` alone dropped `all-zh_TW`; taking every table in
            // sight instead put all twenty into all seven bundles and added
            // 2.3 GB of tables no bundle can use.
            let stem = name.trim_end_matches(".shortlist");
            let Some(tag) = stem.strip_prefix("all-") else {
                continue;
            };
            let base = tag.split(['_', '-']).next().unwrap_or(tag);
            if !languages.iter().any(|l| l == base) {
                continue;
            }
            if !tables.insert(name.to_string()) {
                continue;
            }
            std::fs::create_dir_all(&sl)?;
            std::fs::copy(&p, sl.join(name))?;
        }
    }

    let gguf_path = dir.join(format!("{name}.gguf"));
    gguf.write_to_path(&gguf_path)
        .with_context(|| format!("writing {}", gguf_path.display()))?;
    let gguf_bytes = std::fs::metadata(&gguf_path).map(|m| m.len()).unwrap_or(0);

    // Pack last, so the pack contains every format.
    let rlxp = out.join(format!("{name}.rlxp"));
    rlx_assets::pack::write_dir_rlxp(&dir, &rlxp)
        .with_context(|| format!("packing {}", rlxp.display()))?;
    let rlxp_bytes = std::fs::metadata(&rlxp).map(|m| m.len()).unwrap_or(0);

    Ok(BundleReport {
        name: name.to_string(),
        languages,
        graphs: graphs_done,
        missing,
        tensors: n_tensors,
        elements: n_elems,
        safetensors_bytes: st_bytes,
        gguf_bytes,
        rlxp_bytes,
    })
}

/// Adds one graph's tensors to `w`, namespaced by graph.
///
/// GGUF is a flat namespace and several graphs use the same tensor names, so
/// the graph name is prefixed. Anything that is not 1-D or 2-D, or whose row
/// length is not a multiple of the block size, falls back to F32: block
/// quantizers cannot encode a partial block.
fn add_to_gguf(
    w: &mut GgufWriter,
    graph: &str,
    tensors: &BTreeMap<String, Staged>,
    dtype: GgmlType,
) -> Result<()> {
    for (name, t) in tensors {
        let row = t.dims.last().copied().unwrap_or(t.data.len());
        let blocked = !matches!(dtype, GgmlType::F32 | GgmlType::F16 | GgmlType::BF16);
        let use_dtype = if blocked && row % 32 != 0 {
            GgmlType::F32
        } else {
            dtype
        };
        let bytes = rlx_gguf::quantize(&t.data, use_dtype)
            .with_context(|| format!("quantizing {graph}/{name} as {use_dtype:?}"))?;
        w.add_tensor_bytes(format!("{graph}.{name}"), t.dims.clone(), use_dtype, bytes)?;
    }
    Ok(())
}
