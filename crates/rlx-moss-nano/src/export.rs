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

//! Export MOSS-TTS-Nano as an `.rlxp` package rlx runs alone — text in,
//! 48 kHz stereo speech out (`Model::synthesize`, a builtin voice
//! `options.voice`; interleaved samples) — through the one exporter
//! ([`rlx_core::export`]).
//!
//! Its graphs are the checkpoint's ONNX imported once, here: the global
//! transformer's prefill symbolic in the rows (its hidden states only, the
//! caches it also makes left out), the sampled frame (its random draws
//! inputs), and the audio tokenizer's decoder symbolic in the frames. The
//! frame loop is the package's program (`program/`), the manifest its file.

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use rlx_core::export::graph::symbolic_in_lifted_many;
use rlx_core::export::{ExportOptions, ExportReport, Package, program};
use rlx_ir::Graph;
use serde_json::json;

/// The symbols: the prefill's rows and the codec's frames.
const ROWS: u32 = 0;
const FRAMES: u32 = 1;

/// The ONNX graph `file` imported at `len` as `name`, its first output alone
/// if `first` (the prefill's hidden states), its params kept.
fn import(
    file: &Path,
    name: &str,
    len: usize,
    named: &[(&str, usize)],
    first: bool,
    params: &RefCell<HashMap<String, Vec<f32>>>,
) -> Result<Graph> {
    let (mut g, weights) = rlx_tiny_tts::model::import_rlx_graph(file, name, len, named)?;
    if first {
        let out = g.outputs[0];
        g.set_outputs(vec![out]);
    }
    let mut held = params.borrow_mut();
    for (k, v) in weights {
        held.entry(k).or_insert(v);
    }
    Ok(g)
}

/// Export the MOSS-TTS-Nano checkpoint in `dir` (its ONNX, `codec/`,
/// `browser_poc_manifest.json`, `tokenizer.json`) as the package `out`.
pub fn export_rlxp(dir: &Path, out: &Path, opts: &ExportOptions) -> Result<ExportReport> {
    let manifest = crate::config::Manifest::load(&dir.join("browser_poc_manifest.json"))?;
    let mut pkg = Package::new(
        "moss-tts-nano",
        &opts.version,
        "moss-tts-nano",
        "text-to-speech",
        "rlx-moss-nano",
    );
    if let Ok(card) = std::fs::read_to_string(dir.join("README.md")) {
        rlx_core::export::add_huggingface_card(&mut pkg.model, &card);
    }
    let held = RefCell::new(HashMap::new());
    let mut params = HashMap::new();
    let prefill = "moss_tts_prefill";
    let (g, tables) = symbolic_in_lifted_many(&[(ROWS, vec![37, 46, 61, 96, 131])], &[4096], |n| {
        import(
            &dir.join(format!("{prefill}.onnx")),
            prefill,
            n[0],
            &[("batch", 1), ("prefill_seq", n[0])],
            true,
            &held,
        )
    })
    .context(prefill)?;
    params.extend(tables);
    pkg.function(g);
    let local = "moss_tts_local_fixed_sampled_frame";
    pkg.function(import(
        &dir.join(format!("{local}.onnx")),
        local,
        1,
        &[("batch", 1)],
        false,
        &held,
    )?);
    let codec = "moss_audio_tokenizer_decode_full";
    let (g, tables) =
        symbolic_in_lifted_many(&[(FRAMES, vec![37, 46, 61, 96, 131])], &[4096], |n| {
            import(
                &dir.join("codec").join(format!("{codec}.onnx")),
                codec,
                n[0],
                &[("batch", 1), ("code_length", n[0])],
                false,
                &held,
            )
        })
        .context(codec)?;
    params.extend(tables);
    pkg.function(g);
    params.extend(held.take());
    pkg.dims = vec![("rows".into(), ROWS), ("frames".into(), FRAMES)];
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
    for f in ["browser_poc_manifest.json", "tokenizer.json"] {
        pkg.sidecar(f, "application/json", std::fs::read(dir.join(f))?);
    }
    let voices = manifest.voice_names();
    let module = program::build_for(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("program/Cargo.toml"),
        "wasm32-wasip1",
    )?;
    program::entry(
        &mut pkg,
        "speak",
        module,
        &["synthesize", "run"],
        "text-to-speech",
        json!({
            "voice": voices.first().context("no builtin voices")?,
            "sample_rate": 48_000,
            "channels": 2,
            "max_frames": crate::native::NativeOpts::default().max_frames,
        }),
    );
    pkg.metadata("sample_rate", json!(48_000));
    pkg.metadata("channels", json!(2));
    pkg.metadata("voices", json!(voices));
    pkg.write(out, opts)
}

/// Register this crate's exporter ([`rlx_core::export::register_exporter`]).
pub fn register() {
    use rlx_core::export::{Exporter, register_exporter};
    register_exporter(
        Exporter::new("moss-tts-nano", "text-to-speech", export_rlxp)
            .claims(&["moss-tts-nano", "moss_tts_nano"]),
    );
}
