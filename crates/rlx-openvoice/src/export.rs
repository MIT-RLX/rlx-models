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

//! Export OpenVoice v2 as an `.rlxp` package rlx runs alone — text and a
//! voice to clone in (`options.prompt`), 22.05 kHz speech out — through the
//! one exporter ([`rlx_core::export`]).
//!
//! It is MeloTTS' package ([`rlx_tiny_tts::export::export_with`]) and the
//! tone converter's graphs: the extractor and the converter the
//! checkpoint's ONNX imported once, here, symbolic in the spectrogram's
//! frames ([`symbolic_in_lifted`]), and the spectrogram the runner computes
//! in Rust ([`crate::dsp`]) [`spectrogram`]. Its program is MeloTTS' with
//! the conversion's glue (`program/`).

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use rlx_core::export::audio::{spectrogram, stft_bases};
use rlx_core::export::graph::{symbolic_in, symbolic_in_lifted_many};
use rlx_core::export::{ExportOptions, ExportReport};
use serde_json::json;

use crate::dsp::{HOP, N_FFT, SR, hann_periodic};

/// The symbol for the spectrogram's frames.
const SPEC_FRAMES: u32 = 2;

/// The MeloTTS bundle beside the OpenVoice checkpoint in `dir`: its
/// `melotts/`, else a sibling `melotts/`.
fn melotts_dir(dir: &Path) -> Result<PathBuf> {
    let beside = dir.parent().map(|p| p.join("melotts"));
    for p in std::iter::once(dir.join("melotts")).chain(beside) {
        if p.join("onnx/decoder.onnx").is_file() {
            return Ok(p);
        }
    }
    bail!(
        "OpenVoice speaks with MeloTTS: no MeloTTS bundle at {}/melotts or beside it",
        dir.display()
    )
}

/// Export the OpenVoice checkpoint in `dir` (`tone_extract.onnx`,
/// `tone_color.onnx`, a MeloTTS bundle beside it) as the package `out`.
pub fn export_rlxp(dir: &Path, out: &Path, opts: &ExportOptions) -> Result<ExportReport> {
    let melo = melotts_dir(dir)?;
    let program = Path::new(env!("CARGO_MANIFEST_DIR")).join("program/Cargo.toml");
    rlx_tiny_tts::export::export_with(
        &melo,
        out,
        opts,
        "openvoice",
        "rlx-openvoice",
        |pkg, params| {
            pkg.model["name"] = json!("openvoice");
            let held = RefCell::new(HashMap::new());
            // Sizes of different remainders by each power of two to 64 —
            // the extractor's stride-2 convs make quotients of them — and
            // seven of them: three fit `⌊n / 7⌋` where a dim is `⌈n / 8⌉`.
            let frames = [(SPEC_FRAMES, vec![37, 46, 61, 100, 131, 190, 257])];
            for name in ["tone_extract", "tone_color"] {
                let (g, tables) = symbolic_in_lifted_many(&frames, &[8192], |n| {
                    let (g, weights) = rlx_tiny_tts::model::import_rlx_graph(
                        &dir.join(format!("{name}.onnx")),
                        name,
                        n[0],
                        &[],
                    )?;
                    let mut h = held.borrow_mut();
                    for (k, v) in weights {
                        h.entry(k).or_insert(v);
                    }
                    Ok(g)
                })
                .with_context(|| name.to_string())?;
                params.extend(tables);
                pkg.function(g);
            }
            params.extend(held.take());
            pkg.function(symbolic_in(SPEC_FRAMES, (37, 46, 61), |t| {
                spectrogram(N_FFT, HOP, 1e-6, t)
            })?);
            params.extend(stft_bases(N_FFT, &hann_periodic(N_FFT)));
            pkg.dims.push(("spectrogram_frames".into(), SPEC_FRAMES));
            Ok(json!({"tau": crate::DEFAULT_TAU, "tone_rate": SR}))
        },
        &program,
    )
}

/// Register this crate's exporter ([`rlx_core::export::register_exporter`]).
pub fn register() {
    use rlx_core::export::{Exporter, register_exporter};
    register_exporter(
        Exporter::new("openvoice", "text-to-speech", export_rlxp).claims(&["openvoice"]),
    );
}
