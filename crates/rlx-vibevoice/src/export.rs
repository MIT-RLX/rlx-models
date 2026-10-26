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

//! Export the VibeVoice acoustic decoder as an `.rlxp` package rlx runs
//! alone, through the one exporter ([`rlx_core::export`]): the latent
//! `[64, frames]` in, the 24 kHz waveform out, for any number of frames.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use rlx_core::export::graph::{Pipeline, Stage, broadcast_params, symbolic_in};
use rlx_core::export::{Checkpoint, ExportOptions, ExportReport};
use serde_json::json;

use crate::graph::build_decode_graph;
use crate::model::{DIM_IN, UP_STRIDES, VibeWeights};

/// The symbol for the number of frames.
const FRAMES: u32 = 0;

fn decoder(w: &VibeWeights, t: usize) -> Result<(rlx_ir::Graph, HashMap<String, Vec<f32>>)> {
    let (mut g, params, _) = build_decode_graph(w, t)?;
    g.name = "decode".into();
    Ok((g, params.into_iter().collect()))
}

/// Export the VibeVoice decoder in `path` (its `.safetensors`) as the
/// package `out`.
pub fn export_rlxp(path: &Path, out: &Path, opts: &ExportOptions) -> Result<ExportReport> {
    let ck = Checkpoint::open(path)?;
    let file = ck
        .weight_files()?
        .into_iter()
        .next()
        .context("no safetensors file")?;
    let w = VibeWeights::from_safetensors(&std::fs::read(&file)?)?;
    let decode = symbolic_in(FRAMES, (5, 7, 3), |t| Ok(decoder(&w, t)?.0))?;
    let (at, mut params) = decoder(&w, 3)?;
    let decode = broadcast_params(&decode, &at, &mut params, &[])?;
    let mut p = Pipeline::new(
        "vibevoice",
        "audio-decoding",
        vec![Stage::new(decode).outputs(&["audio"])],
    );
    p.inputs = vec![json!({"name": "latent", "kind": "tensor", "dtype": "f32",
                           "shape": [DIM_IN, null]})];
    p.dims = vec![("frames".into(), FRAMES)];
    p.metadata = vec![
        ("sample_rate".into(), json!(24_000)),
        (
            "samples_per_frame".into(),
            json!(UP_STRIDES.iter().product::<usize>()),
        ),
    ];
    p.exported_by = "rlx-vibevoice".into();
    p.write(params, Some(&ck), out, opts)
}

/// Register this crate's exporter ([`rlx_core::export::register_exporter`]).
pub fn register() {
    use rlx_core::export::{Exporter, register_exporter};
    register_exporter(
        Exporter::new("vibevoice", "audio-decoding", export_rlxp).claims(&["vibevoice"]),
    );
}
