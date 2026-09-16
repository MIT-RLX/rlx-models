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

//! Bundle every rlx-ocr2 asset into one `.rlxp` package.
//!
//! The pipeline has no single executable graph to embed — the detector is interpreted
//! from a recipe and the recognizer is rebuilt per line width — so this writes a
//! **weight-only** package (`include_graph: false`, manifest `graph.encoding = "none"`)
//! and carries the non-tensor assets as sidecars:
//!
//! ```text
//! weights   detector/<name>     270 conv tensors
//!           recognizer/<name>    30 CRNN tensors
//! sidecars  detector_recipe.json  the 265-op detector recipe
//!           codemap.txt           class index -> Unicode codepoint
//!           lexicon.tsv           correction lexicon        (optional)
//!           ngram.bin             correction n-gram model   (optional)
//! ```
//!
//! Tensors are stored hot (mmap-resident, uncompressed) so opening a package is a
//! memory-map plus a per-tensor copy rather than a decompress; the sidecars are zstd'd
//! since they are read once at load.

use crate::detection::Detector;
use crate::pipeline::Ocr2;
use crate::rescore::{Lexicon, Rescorer};
use crate::runner::Recognizer;
use crate::weights::{Tensors, WeightSource, parse_codemap};
use anyhow::{Context, Result, bail};
use rlx_pkg::{Package, PackedWeight, StorageTier, WriteOptions, write_package};
use rlx_runtime::Device;
use std::path::Path;

/// Which `.rlxp` container to emit: a single flat mmap file (default), a ZIP, or an
/// unpacked directory. Re-exported so callers need not depend on `rlx-pkg` directly.
pub use rlx_pkg::ContainerKind;

/// Sidecar ids.
pub const SIDECAR_RECIPE: &str = "detector_recipe.json";
pub const SIDECAR_CODEMAP: &str = "codemap.txt";
pub const SIDECAR_LEXICON: &str = "lexicon.tsv";
pub const SIDECAR_NGRAM: &str = "ngram.bin";

/// Weight-name prefixes keeping the two networks apart inside one index.
const DET_PREFIX: &str = "detector/";
const REC_PREFIX: &str = "recognizer/";

/// The files [`write_pack`] looks for in an asset directory.
pub const REQUIRED_FILES: [&str; 4] = [
    SIDECAR_RECIPE,
    "detector.safetensors",
    "recognizer.safetensors",
    SIDECAR_CODEMAP,
];
pub const OPTIONAL_FILES: [&str; 2] = [SIDECAR_LEXICON, SIDECAR_NGRAM];

/// Read a `.safetensors` file into `(name, values, shape)` rows.
fn read_safetensors(path: &Path) -> Result<Vec<(String, Vec<f32>, Vec<usize>)>> {
    let s = path
        .to_str()
        .with_context(|| format!("path is not UTF-8: {}", path.display()))?;
    let mut wm = rlx_core::weight_map::WeightMap::from_file(s)
        .with_context(|| format!("reading {}", path.display()))?;
    let names: Vec<String> = wm.keys().map(str::to_string).collect();
    names
        .into_iter()
        .map(|n| {
            let (data, shape) = wm.take(&n)?;
            Ok((n, data, shape))
        })
        .collect()
}

fn packed(prefix: &str, name: &str, data: Vec<f32>, shape: Vec<usize>) -> PackedWeight {
    PackedWeight {
        name: format!("{prefix}{name}"),
        shape,
        scheme: "f32".into(),
        layout: "row_major".into(),
        data: bytemuck::cast_slice(&data).to_vec(),
        rank: None,
        tier: StorageTier::Hot,
    }
}

/// Pack an asset directory into `out`. Missing optional assets are simply omitted;
/// the resulting package then loads without the correction stack.
pub fn write_pack(dir: &Path, out: &Path, container: ContainerKind) -> Result<()> {
    for f in REQUIRED_FILES {
        if !dir.join(f).is_file() {
            bail!("{} is missing from {}", f, dir.display());
        }
    }

    let mut weights = Vec::new();
    for (file, prefix) in [
        ("detector.safetensors", DET_PREFIX),
        ("recognizer.safetensors", REC_PREFIX),
    ] {
        for (name, data, shape) in read_safetensors(&dir.join(file))? {
            weights.push(packed(prefix, &name, data, shape));
        }
    }

    let mut sidecars = vec![
        (
            SIDECAR_RECIPE.to_string(),
            "application/json".to_string(),
            std::fs::read(dir.join(SIDECAR_RECIPE))?,
        ),
        (
            SIDECAR_CODEMAP.to_string(),
            "text/plain".to_string(),
            std::fs::read(dir.join(SIDECAR_CODEMAP))?,
        ),
    ];
    for (file, media) in [
        (SIDECAR_LEXICON, "text/tab-separated-values"),
        (SIDECAR_NGRAM, "application/octet-stream"),
    ] {
        let p = dir.join(file);
        if p.is_file() {
            sidecars.push((file.to_string(), media.to_string(), std::fs::read(&p)?));
        }
    }

    let opts = WriteOptions {
        name: "rlx-ocr2".into(),
        producer: Some(format!("rlx-ocr2 {}", env!("CARGO_PKG_VERSION"))),
        features: vec!["ocr2_detector".into(), "ocr2_recognizer".into()],
        container,
        sidecars,
        include_graph: false, // two graphs, both built at load time — nothing to embed
        ..Default::default()
    };
    write_package(out, &rlx_ir::Graph::new("rlx-ocr2"), &weights, &opts)
        .with_context(|| format!("writing {}", out.display()))
}

/// An opened `.rlxp`, with the assets pulled out ready to build the pipeline from.
pub struct Ocr2Pack {
    pub recipe_json: String,
    pub codemap: Vec<u32>,
    detector: Tensors,
    recognizer: Tensors,
    lexicon: Option<String>,
    ngram: Option<Vec<u8>>,
}

impl Ocr2Pack {
    /// Open a package written by [`write_pack`] (flat file, zip, or directory).
    pub fn open(path: &Path) -> Result<Self> {
        let pack = Package::open(path).with_context(|| format!("opening {}", path.display()))?;
        let index = pack
            .weights_index()
            .context("package has no weights index")?;

        let (mut detector, mut recognizer) = (Tensors::new(), Tensors::new());
        for entry in &index.tensors {
            let Some((prefix, name)) = split_prefix(&entry.name) else {
                continue; // not ours — leave it alone
            };
            let bytes = pack
                .tensor_bytes(&entry.name)
                .with_context(|| format!("reading tensor {}", entry.name))?;
            if bytes.len() != entry.shape.iter().product::<usize>() * 4 {
                bail!(
                    "tensor {} has {} bytes, shape {:?}",
                    entry.name,
                    bytes.len(),
                    entry.shape
                );
            }
            let values = bytemuck::cast_slice::<u8, f32>(&bytes).to_vec();
            let dst = if prefix == DET_PREFIX {
                &mut detector
            } else {
                &mut recognizer
            };
            dst.insert(name.to_string(), (values, entry.shape.clone()));
        }
        if detector.is_empty() || recognizer.is_empty() {
            bail!(
                "package is missing weights (detector {}, recognizer {})",
                detector.len(),
                recognizer.len()
            );
        }

        let text = |id: &str| -> Result<String> {
            String::from_utf8(pack.sidecar(id)?)
                .with_context(|| format!("sidecar {id} is not UTF-8"))
        };
        Ok(Self {
            recipe_json: text(SIDECAR_RECIPE)?,
            codemap: parse_codemap(&text(SIDECAR_CODEMAP)?)?,
            detector,
            recognizer,
            lexicon: text(SIDECAR_LEXICON).ok(),
            ngram: pack.sidecar(SIDECAR_NGRAM).ok(),
        })
    }

    /// True when the package carries the n-gram and/or lexicon correction assets.
    pub fn has_rescorer(&self) -> bool {
        self.lexicon.is_some() || self.ngram.is_some()
    }

    /// Detector over `heads` (empty = all heads the recipe declares).
    pub fn detector(&self, device: Device, heads: Vec<String>) -> Result<Detector> {
        Detector::from_parts(
            self.recipe_json.clone(),
            WeightSource::memory(self.detector.clone()),
            device,
            heads,
        )
    }

    pub fn recognizer(&self, device: Device) -> Result<Recognizer> {
        Recognizer::from_parts(
            WeightSource::memory(self.recognizer.clone()),
            self.codemap.clone(),
            device,
        )
    }

    /// The correction stack, or `None` if the package carries neither asset.
    pub fn rescorer(&self) -> Result<Option<Rescorer>> {
        if !self.has_rescorer() {
            return Ok(None);
        }
        let ngram = self
            .ngram
            .as_deref()
            .map(crate::ngram::NgramModel::from_bytes)
            .transpose()?;
        let lexicon = self.lexicon.as_deref().map(Lexicon::from_text);
        Ok(Some(Rescorer::new(ngram, None, lexicon)))
    }

    /// The full pipeline, with correction attached when the package carries it.
    pub fn into_pipeline(self, device: Device) -> Result<Ocr2> {
        let mut ocr = Ocr2::new(
            self.detector(device, Ocr2::pipeline_heads())?,
            self.recognizer(device)?,
        );
        if let Some(r) = self.rescorer()? {
            ocr = ocr.with_rescorer(r);
        }
        Ok(ocr)
    }
}

fn split_prefix(name: &str) -> Option<(&'static str, &str)> {
    if let Some(rest) = name.strip_prefix(DET_PREFIX) {
        Some((DET_PREFIX, rest))
    } else {
        name.strip_prefix(REC_PREFIX).map(|rest| (REC_PREFIX, rest))
    }
}

impl Ocr2 {
    /// Load the whole pipeline from a single `.rlxp`.
    pub fn load_pack(path: &Path, device: Device) -> Result<Self> {
        Ocr2Pack::open(path)?.into_pipeline(device)
    }
}

impl Recognizer {
    /// Load just the recognizer stage from a `.rlxp`.
    pub fn load_pack(path: &Path, device: Device) -> Result<Self> {
        Ocr2Pack::open(path)?.recognizer(device)
    }
}
