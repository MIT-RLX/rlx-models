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

//! Export Supertonic-3 as an `.rlxp` package rlx runs alone — text in,
//! 44.1 kHz speech out (`Model::synthesize`) — through the one exporter
//! ([`rlx_core::export`]).
//!
//! Its four graphs (`duration_predictor`, `text_encoder`,
//! `vector_estimator`, `vocoder`) are the checkpoint's ONNX imported once,
//! here, as rlx graphs symbolic in the text and latent lengths — the
//! constants the import bakes from them rebuilt from them
//! ([`symbolic_in_lifted_sizes`]). The rest of what
//! [`crate::Supertonic::synthesize`] does — the text preprocessor, the
//! tokenizer, the voice styles, the noise, the flow-matching loop — is the
//! package's program (`program/`, wasm32-wasip1), the tokenizer's table,
//! the config and the voices its files.

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use rlx_core::export::graph::symbolic_in_lifted_sizes;
use rlx_core::export::{Checkpoint, ExportOptions, ExportReport, Package, program};
use rlx_ir::Graph;
use serde_json::json;

/// The symbols: the text's length and the latent's.
const TEXT: u32 = 0;
const LATENT: u32 = 1;
/// The three builds' text and latent lengths (a text is at least its
/// `<en></en>` and a period, past the relative attention's window).
const TEXTS: (usize, usize, usize) = (11, 15, 19);
const LATENTS: (usize, usize, usize) = (23, 31, 41);

/// The ONNX graph `name` in `onnx` imported at `sizes` (`TEXT`, `LATENT`
/// as it has them), its params kept the first time.
fn import(
    onnx: &Path,
    name: &str,
    text: Option<usize>,
    latent: Option<usize>,
    params: &RefCell<HashMap<String, Vec<f32>>>,
) -> Result<Graph> {
    let mut named = Vec::new();
    named.extend(text.map(|t| ("text_length", t)));
    named.extend(latent.map(|l| ("latent_length", l)));
    let length = latent.or(text).context("a length")?;
    let (g, weights) = rlx_tiny_tts::model::import_rlx_graph(
        &onnx.join(format!("{name}.onnx")),
        name,
        length,
        &named,
    )?;
    let mut held = params.borrow_mut();
    for (k, v) in weights {
        held.entry(k).or_insert(v);
    }
    Ok(g)
}

/// Every file under `dir` as a sidecar named `prefix` + its name.
fn sidecars(pkg: &mut Package, dir: &Path, prefix: &str) -> Result<()> {
    for e in std::fs::read_dir(dir)? {
        let p = e?.path();
        let name = p
            .file_name()
            .and_then(|s| s.to_str())
            .context("file name")?;
        if p.is_file() {
            pkg.sidecar(
                &format!("{prefix}{name}"),
                "application/json",
                std::fs::read(&p)?,
            );
        }
    }
    Ok(())
}

/// Export the Supertonic-3 checkpoint in `dir` (`onnx/`, `voice_styles/`)
/// as the package `out`.
pub fn export_rlxp(dir: &Path, out: &Path, opts: &ExportOptions) -> Result<ExportReport> {
    let ck = Checkpoint::open(dir)?;
    let onnx = dir.join("onnx");
    let cfg = crate::config::StConfig::load(&onnx)?;
    let mut pkg =
        Package::for_checkpoint(&ck, opts, "supertonic", "text-to-speech", "rlx-supertonic");

    let held = RefCell::new(HashMap::new());
    let mut params = HashMap::new();
    let (ta, tb, tc) = TEXTS;
    let (la, lb, lc) = LATENTS;
    let text_sizes = [(TEXT, ta, tb, tc)];
    let latent_sizes = [(LATENT, la, lb, lc)];
    let both = [(TEXT, ta, tb, tc), (LATENT, la, lb, lc)];
    for (name, sizes, longest) in [
        ("duration_predictor", &text_sizes[..], &[1024][..]),
        ("text_encoder", &text_sizes[..], &[1024][..]),
        ("vector_estimator", &both[..], &[1024, 4096][..]),
        ("vocoder", &latent_sizes[..], &[4096][..]),
    ] {
        let syms: Vec<u32> = sizes.iter().map(|s| s.0).collect();
        let (g, tables) = symbolic_in_lifted_sizes(sizes, longest, |n| {
            let at = |s: u32| syms.iter().position(|x| *x == s).map(|i| n[i]);
            import(&onnx, name, at(TEXT), at(LATENT), &held)
        })?;
        params.extend(tables);
        pkg.function(g);
    }
    params.extend(held.take());
    pkg.dims = vec![("text".into(), TEXT), ("latent".into(), LATENT)];
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

    // The program's files where the runner's modules read them.
    for f in ["unicode_indexer.json", "tts.json"] {
        pkg.sidecar(f, "application/json", std::fs::read(onnx.join(f))?);
    }
    sidecars(&mut pkg, &dir.join("voice_styles"), "voice_styles/")?;
    let voices = crate::voices::list_voices(&dir.join("voice_styles"))?;
    let default_voice = voices.first().context("no voice styles")?.clone();

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
            "default_voice": default_voice,
            "total_step": crate::model::DEFAULT_TOTAL_STEP,
            "speed": crate::model::DEFAULT_SPEED,
        }),
    );
    pkg.metadata("sample_rate", json!(cfg.sample_rate));
    pkg.metadata("voices", json!(voices));
    pkg.metadata("languages", json!(crate::config::AVAILABLE_LANGS));
    pkg.write(out, opts)
}

/// Register this crate's exporter ([`rlx_core::export::register_exporter`]).
pub fn register() {
    use rlx_core::export::{Exporter, register_exporter};
    register_exporter(
        Exporter::new("supertonic", "text-to-speech", export_rlxp).claims(&["supertonic"]),
    );
}
