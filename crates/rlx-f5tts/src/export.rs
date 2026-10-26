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

//! Export F5-TTS as an `.rlxp` package rlx runs alone — text and a voice
//! to clone in (`options.prompt`), 24 kHz speech out — through the one
//! exporter ([`rlx_core::export`]).
//!
//! Its three graphs (`F5_Preprocess`, `F5_Transformer`, `F5_Decode`) are
//! the checkpoint's ONNX imported once, here, as rlx graphs symbolic in
//! the reference audio's samples, the ids and the frames to make
//! ([`symbolic_in_lifted_sizes`]): the preprocessor's STFT frames
//! (`(audio)/256+1`) and the frames past the reference's are quotients of
//! those. The rest of what [`crate::native::F5Native::synthesize`] does is
//! the package's program (`program/`, wasm32-wasip1), the vocabulary its
//! file.

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use rlx_core::export::graph::symbolic_in_lifted_sizes;
use rlx_core::export::{Checkpoint, ExportOptions, ExportReport, Package, program};
use rlx_ir::Graph;
use serde_json::json;

use crate::config::{DEFAULT_MAX_DURATION, DEFAULT_NFE, Layout, SAMPLE_RATE};

/// The symbols: the reference audio's samples, the ids, the frames.
const AUDIO: u32 = 0;
const IDS: u32 = 1;
const FRAMES: u32 = 2;
/// `cat_mel_text`'s width: 100 mels and 512 of text embedding.
const TEXT_EMBED_LEN: usize = 612;

/// The ONNX graph `name` imported at `n` (one size per symbol of `syms`),
/// its params kept.
fn import(
    dir: &Path,
    name: &str,
    syms: &[u32],
    n: &[usize],
    params: &RefCell<HashMap<String, Vec<f32>>>,
) -> Result<Graph> {
    let at = |s: u32| syms.iter().position(|x| *x == s).map(|i| n[i]);
    let frames = at(FRAMES).context("every F5 graph has the frames")?;
    let mut named = vec![("max_duration", frames)];
    named.extend(at(AUDIO).map(|v| ("audio_len", v)));
    named.extend(at(IDS).map(|v| ("text_ids_len", v)));
    if name != "F5_Decode" {
        named.push(("text_embed_len", TEXT_EMBED_LEN));
    }
    let (g, weights) = rlx_tiny_tts::model::import_rlx_graph(
        &dir.join(format!("{name}.onnx")),
        name,
        frames,
        &named,
    )?;
    let mut held = params.borrow_mut();
    for (k, v) in weights {
        held.entry(k).or_insert(v);
    }
    Ok(g)
}

/// Export the F5-TTS checkpoint in `dir` (`F5_*.onnx`, `vocab.txt`) as the
/// package `out`.
pub fn export_rlxp(dir: &Path, out: &Path, opts: &ExportOptions) -> Result<ExportReport> {
    let ck = Checkpoint::open(dir)?;
    let layout = Layout::resolve(dir)?;
    let mut pkg = Package::for_checkpoint(&ck, opts, "f5tts", "text-to-speech", "rlx-f5tts");

    let held = RefCell::new(HashMap::new());
    let mut params = HashMap::new();
    // Sizes of different remainders by the STFT's hop, so its frames are
    // a quotient one fit pins; no two symbols' in proportion.
    let audio = (0, 256 * 47, 256 * 59 + 255, 256 * 71 + 100);
    let ids = (IDS, 37, 45, 61);
    let frames = (FRAMES, 301, 333, 349);
    for (name, sizes) in [
        (
            "F5_Preprocess",
            vec![(AUDIO, audio.1, audio.2, audio.3), ids, frames],
        ),
        ("F5_Transformer", vec![frames]),
        ("F5_Decode", vec![frames]),
    ] {
        let syms: Vec<u32> = sizes.iter().map(|s| s.0).collect();
        let longest = vec![8192; sizes.len()];
        let (g, tables) = symbolic_in_lifted_sizes(&sizes, &longest, |n| {
            import(&layout.dir, name, &syms, n, &held)
        })
        .with_context(|| name.to_string())?;
        params.extend(tables);
        pkg.function(g);
    }
    params.extend(held.take());
    pkg.dims = vec![
        ("audio".into(), AUDIO),
        ("ids".into(), IDS),
        ("max_duration".into(), FRAMES),
    ];
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
    pkg.sidecar(
        "vocab.txt",
        "text/plain",
        std::fs::read(layout.dir.join("vocab.txt"))?,
    );

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
        json!({"nfe": DEFAULT_NFE, "max_duration": DEFAULT_MAX_DURATION}),
    );
    pkg.metadata("sample_rate", json!(SAMPLE_RATE));
    pkg.metadata(
        "prompt",
        json!("options.prompt: {audio: [24 kHz mono samples], sample_rate: 24000, text: its transcript}"),
    );
    pkg.write(out, opts)
}

/// Register this crate's exporter ([`rlx_core::export::register_exporter`]).
pub fn register() {
    use rlx_core::export::{Exporter, register_exporter};
    register_exporter(
        Exporter::new("f5tts", "text-to-speech", export_rlxp).claims(&["f5tts", "f5-tts"]),
    );
}
