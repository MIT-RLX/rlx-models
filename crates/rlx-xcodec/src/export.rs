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

//! Export the XCodec2 decoder (RoFormer + Vocos head) as an `.rlxp` package
//! rlx runs alone, through the one exporter ([`rlx_core::export`]): the
//! embeddings `[frames, 1024]` in, the 16 kHz waveform out, for up to
//! [`MAX_FRAMES`] frames. Its RoPE tables are held whole and read up to the
//! length; the inverse STFT the runner does on the host is the second stage.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use rlx_core::audio_ops_ir::rope_tables;
use rlx_core::export::audio::{istft_bases, vocos_istft};
use rlx_core::export::graph::{Pipeline, Stage, broadcast_params, symbolic_in};
use rlx_core::export::{Checkpoint, ExportOptions, ExportReport};
use serde_json::json;

use crate::graph::build_decode_graph;
use crate::model::{DIM, HEAD_DIM, HOP, ROPE_THETA, XcodecWeights};

/// The symbol for the number of frames.
const FRAMES: u32 = 0;

/// The longest input the package's RoPE tables reach: 8192 frames, 164 s.
pub const MAX_FRAMES: usize = 8192;

fn decoder(w: &XcodecWeights, t: usize) -> Result<(rlx_ir::Graph, HashMap<String, Vec<f32>>)> {
    let (mut g, params) = build_decode_graph(w, t)?;
    g.name = "decode".into();
    Ok((g, params.into_iter().collect()))
}

/// The decoder `w` as two functions symbolic in the frames (symbol
/// `frames`) — `decode` (`emb` `[t, 1024]` → the head's `spec`) and `istft`
/// (`spec` → `audio`, `n_fft` the window's length, a quarter of it the hop)
/// — with their params: for a package whose model ends in this decoder
/// (NeuTTS's NeuCodec).
pub fn decoder_functions(
    w: &XcodecWeights,
    frames: u32,
) -> Result<(rlx_ir::Graph, rlx_ir::Graph, HashMap<String, Vec<f32>>)> {
    let decode = symbolic_in(frames, (5, 7, 3), |t| Ok(decoder(w, t)?.0))?;
    // The rotary tables as the builder makes them, as long as the package
    // reaches.
    let inv_freq: Vec<f32> = (0..HEAD_DIM / 2)
        .map(|i| 1.0 / ROPE_THETA.powf(2.0 * i as f32 / HEAD_DIM as f32))
        .collect();
    let (cos, sin) = rope_tables(&inv_freq, MAX_FRAMES);
    let (at, mut params) = decoder(w, 3)?;
    let decode = broadcast_params(
        &decode,
        &at,
        &mut params,
        &[("rope.cos", &cos, HEAD_DIM), ("rope.sin", &sin, HEAD_DIM)],
    )?;
    let n_fft = w.window.len();
    let istft = symbolic_in(frames, (5, 7, 3), |t| {
        vocos_istft(n_fft, n_fft / 4, &w.window, t)
    })?;
    params.extend(istft_bases(n_fft, &w.window));
    Ok((decode, istft, params))
}

/// Export the XCodec2 decoder in `path` (its `.safetensors`) as the package
/// `out`.
pub fn export_rlxp(path: &Path, out: &Path, opts: &ExportOptions) -> Result<ExportReport> {
    let ck = Checkpoint::open(path)?;
    let file = ck
        .weight_files()?
        .into_iter()
        .next()
        .context("no safetensors file")?;
    let w = XcodecWeights::from_safetensors(&std::fs::read(&file)?)?;
    let (decode, istft, params) = decoder_functions(&w, FRAMES)?;
    let stages = vec![
        Stage::new(decode).outputs(&["spec"]),
        Stage::new(istft).outputs(&["audio"]),
    ];
    let mut p = Pipeline::new("xcodec2", "audio-decoding", stages);
    p.inputs = vec![json!({"name": "emb", "kind": "tensor", "dtype": "f32",
                           "shape": [null, DIM]})];
    p.dims = vec![("frames".into(), FRAMES)];
    p.metadata = vec![
        ("sample_rate".into(), json!(16_000)),
        ("hop".into(), json!(HOP)),
        ("max_frames".into(), json!(MAX_FRAMES)),
    ];
    p.exported_by = "rlx-xcodec".into();
    p.write(params, Some(&ck), out, opts)
}

/// Register this crate's exporter ([`rlx_core::export::register_exporter`]).
pub fn register() {
    use rlx_core::export::{Exporter, register_exporter};
    register_exporter(Exporter::new("xcodec2", "audio-decoding", export_rlxp).claims(&["xcodec2"]));
}
