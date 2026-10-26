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

//! Export the NanoCodec decoder as an `.rlxp` package rlx runs alone,
//! through the one exporter ([`rlx_core::export`]): Group-FSQ codes
//! `[groups, frames]` in, the 22.05 kHz waveform out, for any number of
//! frames. The dequantization the runner does on the host
//! ([`crate::fsq_decode`]: each code's digits in mixed radix, centred and
//! scaled) is the first stage, exact in f32; the decoder the second.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use rlx_core::export::graph::{Pipeline, Stage, broadcast_params, symbolic_in};
use rlx_core::export::{Checkpoint, ExportOptions, ExportReport};
use rlx_ir::op::BinaryOp;
use rlx_ir::{DType, Graph, GraphExt, NodeId, Op, Shape};
use serde_json::json;

use crate::graph::build_decode_graph;
use crate::model::{INPUT_DIM, LEVELS, NUM_GROUPS, NanoWeights};

/// The symbol for the number of frames.
const FRAMES: u32 = 0;

fn f32s(d: &[usize]) -> Shape {
    Shape::new(d, DType::F32)
}

fn scalar(g: &mut Graph, v: f32) -> NodeId {
    g.add_node(
        Op::Constant {
            data: v.to_le_bytes().to_vec(),
        },
        vec![],
        f32s(&[1]),
    )
}

/// `codes` `[groups, t]` → `latent` `[16, t]`, channel `group · 4 + digit`
/// the digit's `((code / base) mod levels − levels / 2) / (levels / 2)`.
fn fsq(t: usize) -> Graph {
    let mut g = Graph::new("fsq");
    let codes = g.input("codes", Shape::new(&[NUM_GROUPS, t], DType::I32));
    let codes = g.add_node(
        Op::Cast { to: DType::F32 },
        vec![codes],
        f32s(&[NUM_GROUPS, t]),
    );
    let mut digits = Vec::new();
    let mut base = 1i64;
    for &levels in &LEVELS {
        let half = levels / 2;
        let b = scalar(&mut g, base as f32);
        let q = g.binary(BinaryOp::Div, codes, b, f32s(&[NUM_GROUPS, t]));
        let q = g.add_node(
            Op::Activation(rlx_ir::op::Activation::Floor),
            vec![q],
            f32s(&[NUM_GROUPS, t]),
        );
        let l = scalar(&mut g, levels as f32);
        let r = g.binary(BinaryOp::Mod, q, l, f32s(&[NUM_GROUPS, t]));
        let h = scalar(&mut g, half as f32);
        let c = g.binary(BinaryOp::Sub, r, h, f32s(&[NUM_GROUPS, t]));
        let v = g.binary(BinaryOp::Div, c, h, f32s(&[NUM_GROUPS, t]));
        digits.push(g.reshape(
            v,
            vec![NUM_GROUPS as i64, 1, t as i64],
            f32s(&[NUM_GROUPS, 1, t]),
        ));
        base *= levels;
    }
    let stacked = g.concat_(digits, 1);
    let latent = g.reshape(
        stacked,
        vec![INPUT_DIM as i64, t as i64],
        f32s(&[INPUT_DIM, t]),
    );
    g.node_mut(latent).name = Some("latent".into());
    g.set_outputs(vec![latent]);
    g
}

fn decoder(w: &NanoWeights, t: usize) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    let (mut g, params, _) = build_decode_graph(w, t)?;
    g.name = "decode".into();
    Ok((g, params.into_iter().collect()))
}

/// Export the NanoCodec decoder in `path` (its `.safetensors`) as the
/// package `out`.
pub fn export_rlxp(path: &Path, out: &Path, opts: &ExportOptions) -> Result<ExportReport> {
    let ck = Checkpoint::open(path)?;
    let file = ck
        .weight_files()?
        .into_iter()
        .next()
        .context("no safetensors file")?;
    let w = NanoWeights::from_safetensors(&std::fs::read(&file)?)?;
    let lookup = symbolic_in(FRAMES, (5, 7, 3), |t| Ok(fsq(t)))?;
    let decode = symbolic_in(FRAMES, (5, 7, 3), |t| Ok(decoder(&w, t)?.0))?;
    // The decoder's padding blocks, materialized at the length, as one
    // slice expanded.
    let (at, mut params) = decoder(&w, 3)?;
    let decode = broadcast_params(&decode, &at, &mut params, &[])?;
    let stages = vec![
        Stage::new(lookup).outputs(&["latent"]),
        Stage::new(decode).outputs(&["audio"]),
    ];
    let mut p = Pipeline::new("nanocodec", "audio-decoding", stages);
    p.inputs = vec![json!({"name": "codes", "kind": "tensor", "dtype": "i32",
                           "shape": [NUM_GROUPS, null]})];
    p.dims = vec![("frames".into(), FRAMES)];
    p.metadata = vec![
        ("sample_rate".into(), json!(22_050)),
        ("levels".into(), json!(LEVELS)),
    ];
    p.exported_by = "rlx-nanocodec".into();
    p.write(params, Some(&ck), out, opts)
}

/// Register this crate's exporter ([`rlx_core::export::register_exporter`]).
pub fn register() {
    use rlx_core::export::{Exporter, register_exporter};
    register_exporter(
        Exporter::new("nanocodec", "audio-decoding", export_rlxp).claims(&["nanocodec"]),
    );
}
