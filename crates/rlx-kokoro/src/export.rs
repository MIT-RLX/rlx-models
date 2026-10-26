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

//! Export Kokoro-82M as an `.rlxp` package rlx runs alone — text in, 24 kHz
//! speech out (`Model::synthesize`) — through the one exporter
//! ([`rlx_core::export`]).
//!
//! Its two graphs, `encoder` and `decoder`, are rlx graphs symbolic in
//! the phonemes and the frames: the native runner's split bundle
//! (`encoder.onnx`, `decoder_raw.onnx`) imported once, here, at three
//! lengths each, the constants the import bakes from the length (index
//! tables, aranges, an upsampler's staircases and interpolation weights)
//! rebuilt from it ([`symbolic_in_lifted`]), and the uniform params baked
//! as the runner's compile bakes them. The rest of
//! what [`crate::native::NativeKokoro`] does — the text preprocessor,
//! espeak-ng, the phoneme vocabulary, the voice's style row, the length
//! regulator between the graphs, the ISTFT normalization — is the
//! package's program (`program/`, wasm32-wasip1), with espeak-ng's English
//! data, the vocabulary, the voices and the ISTFT window as its files.

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use rlx_core::export::graph::symbolic_in_lifted;
use rlx_core::export::{Checkpoint, ExportOptions, ExportReport, Package, program};
use rlx_ir::Graph;
use serde_json::json;

use crate::native::SPLIT_SUBDIR;

/// The voice a package speaks in unless asked for another.
const DEFAULT_VOICE: &str = "af_heart";

/// The symbols: the phonemes (the encoder's length) and the frames (the
/// decoder's).
const PHONEMES: u32 = 0;
const FRAMES: u32 = 1;

/// The ONNX graph `file` imported at `len` as the rlx graph `name`
/// ([`rlx_tiny_tts::model::import_rlx_graph`]), its params kept the first
/// time.
fn import(
    file: &Path,
    name: &str,
    len: usize,
    named: &[(&str, usize)],
    params: &RefCell<Option<HashMap<String, Vec<f32>>>>,
) -> Result<Graph> {
    let (g, weights) = rlx_tiny_tts::model::import_rlx_graph(file, name, len, named)?;
    params.borrow_mut().get_or_insert(weights);
    Ok(g)
}

/// Every file under `dir`, by its path relative to it.
fn files(dir: &Path, base: &Path, out: &mut Vec<(String, PathBuf)>) -> Result<()> {
    for e in std::fs::read_dir(dir)? {
        let p = e?.path();
        if p.is_dir() {
            files(&p, base, out)?;
        } else {
            let rel = p.strip_prefix(base)?.to_string_lossy().replace('\\', "/");
            out.push((rel, p));
        }
    }
    Ok(())
}

/// Export the Kokoro checkpoint in `dir` (`config.json`, `tokenizer.json`,
/// `voices/`, the split bundle under `onnx/rlx-split/`) as the package `out`.
pub fn export_rlxp(dir: &Path, out: &Path, opts: &ExportOptions) -> Result<ExportReport> {
    let ck = Checkpoint::open(dir)?;
    let split = dir.join("onnx").join(SPLIT_SUBDIR);
    let mut pkg = Package::for_checkpoint(&ck, opts, "kokoro", "text-to-speech", "rlx-kokoro");

    // The graphs, symbolic in their lengths.
    let held = RefCell::new(None);
    let (encoder, tables) = symbolic_in_lifted(PHONEMES, (5, 7, 9), 512, |n| {
        import(
            &split.join("encoder.onnx"),
            "encoder",
            n,
            &[("sequence_length", n)],
            &held,
        )
    })?;
    let mut params = held.take().context("the encoder's params")?;
    params.extend(tables);
    let (decoder, tables) = symbolic_in_lifted(FRAMES, (20, 28, 36), 4096, |n| {
        import(
            &split.join("decoder_raw.onnx"),
            "decoder",
            n,
            &[("unk__357", 1), ("unk__368", n)],
            &held,
        )
    })?;
    params.extend(held.take().context("the decoder's params")?);
    params.extend(tables);
    pkg.function(encoder);
    pkg.function(decoder);
    pkg.dims = vec![("phonemes".into(), PHONEMES), ("frames".into(), FRAMES)];
    let dims = pkg.param_dims();
    let mut names: Vec<&String> = params.keys().collect();
    names.sort();
    for n in names {
        let d = dims
            .get(n)
            .cloned()
            .unwrap_or_else(|| vec![params[n].len()]);
        pkg.hold_f32(n.clone(), d, &params[n]);
    }

    // The program's files: the vocabulary, the voices, the ISTFT window,
    // espeak-ng's English data.
    let vocab = pkg.sidecar(
        "tokenizer.json",
        "application/json",
        std::fs::read(dir.join("tokenizer.json"))?,
    );
    let window = pkg.sidecar(
        "window_sum.f32",
        "application/octet-stream",
        std::fs::read(split.join("window_sum.f32"))?,
    );
    let mut voices = serde_json::Map::new();
    for e in std::fs::read_dir(dir.join("voices"))? {
        let p = e?.path();
        if p.extension().and_then(|x| x.to_str()) != Some("bin") {
            continue;
        }
        let name = p
            .file_stem()
            .and_then(|s| s.to_str())
            .context("voice name")?;
        let id = pkg.sidecar(
            &format!("voices/{name}.bin"),
            "application/octet-stream",
            std::fs::read(&p)?,
        );
        voices.insert(name.to_string(), json!(id));
    }
    let espeak = tempfile::tempdir()?;
    espeak_ng::install_bundled_language(espeak.path(), "en")
        .map_err(|e| anyhow::anyhow!("espeak-ng's English data: {e}"))?;
    let mut data = Vec::new();
    files(espeak.path(), espeak.path(), &mut data)?;
    for (rel, p) in data {
        pkg.sidecar(
            &format!("espeak/{rel}"),
            "application/octet-stream",
            std::fs::read(p)?,
        );
    }

    let module = program::build_for(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("program/Cargo.toml"),
        "wasm32-wasip1",
    )?;
    program::entry(
        &mut pkg,
        "speak",
        module,
        &["synthesize"],
        "text-to-speech",
        json!({
            "vocab": vocab,
            "voices": voices,
            "default_voice": DEFAULT_VOICE,
            "window_sum": window,
            "espeak": format!("/package/espeak"),
        }),
    );
    pkg.metadata("sample_rate", json!(crate::config::SAMPLE_RATE));
    pkg.metadata("voices", json!(voices.keys().collect::<Vec<_>>()));
    pkg.write(out, opts)
}

/// Register this crate's exporter ([`rlx_core::export::register_exporter`]).
pub fn register() {
    use rlx_core::export::{Exporter, register_exporter};
    register_exporter(Exporter::new("kokoro", "text-to-speech", export_rlxp).claims(&["kokoro"]));
}
