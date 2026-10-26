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

//! Export the WavTokenizer decoder as an `.rlxp` package rlx runs alone,
//! through the one exporter ([`rlx_core::export`]): features `[1, 512,
//! frames, 1]` in, the 24 kHz waveform out, for any number of frames. The
//! inverse STFT the runner does on the host ([`crate::head_to_wav`]) is the
//! package's second stage ([`rlx_core::export::audio::vocos_istft`]).

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use rlx_core::export::audio::{istft_bases, vocos_istft};
use rlx_core::export::graph::{Pipeline, Stage, broadcast_params, symbolic_in};
use rlx_core::export::{Checkpoint, ExportOptions, ExportReport};
use serde_json::json;

use crate::graph::build_decode_graph;
use crate::model::{HOP, INPUT_CH, N_FFT, WavtokWeights};

/// The symbol for the number of frames.
const FRAMES: u32 = 0;

fn decoder(w: &WavtokWeights, t: usize) -> Result<(rlx_ir::Graph, HashMap<String, Vec<f32>>)> {
    let (mut g, params, _) = build_decode_graph(w, t)?;
    g.name = "decode".into();
    Ok((g, params.into_iter().collect()))
}

/// Export the WavTokenizer decoder in `path` (its `.safetensors`) as the
/// package `out`.
pub fn export_rlxp(path: &Path, out: &Path, opts: &ExportOptions) -> Result<ExportReport> {
    let ck = Checkpoint::open(path)?;
    let file = ck
        .weight_files()?
        .into_iter()
        .next()
        .context("no safetensors file")?;
    let w = WavtokWeights::from_safetensors(&std::fs::read(&file)?)?;
    let window = w.head.window.clone();
    let decode = symbolic_in(FRAMES, (5, 7, 3), |t| Ok(decoder(&w, t)?.0))?;
    let (at, mut params) = decoder(&w, 3)?;
    let decode = broadcast_params(&decode, &at, &mut params, &[])?;
    let istft = symbolic_in(FRAMES, (5, 7, 3), |t| vocos_istft(N_FFT, HOP, &window, t))?;
    params.extend(istft_bases(N_FFT, &window));
    let stages = vec![
        Stage::new(decode).outputs(&["spec"]),
        Stage::new(istft).outputs(&["audio"]),
    ];
    let mut p = Pipeline::new("wavtokenizer", "audio-decoding", stages);
    p.inputs = vec![json!({"name": "feats", "kind": "tensor", "dtype": "f32",
                           "shape": [1, INPUT_CH, null, 1]})];
    p.dims = vec![("frames".into(), FRAMES)];
    p.metadata = vec![
        ("sample_rate".into(), json!(24_000)),
        ("hop".into(), json!(HOP)),
    ];
    p.exported_by = "rlx-wavtokenizer".into();
    p.write(params, Some(&ck), out, opts)
}

/// Register this crate's exporter ([`rlx_core::export::register_exporter`]).
pub fn register() {
    use rlx_core::export::{Exporter, register_exporter};
    register_exporter(
        Exporter::new("wavtokenizer", "audio-decoding", export_rlxp).claims(&["wavtokenizer"]),
    );
}
