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

//! Export the FACodec decoder as an `.rlxp` package rlx runs alone, through
//! the one exporter ([`rlx_core::export`]): the latent `emb` `[256,
//! frames]` and the speaker embedding `spk` `[256]` in, the 16 kHz waveform
//! out, for any number of frames and any speaker. The timbre the runner
//! computes on the host and builds into its graph
//! ([`FacodecWeights::timbre_affine`]: `W·spk + b`, split into the AdaIN's
//! scale and shift) is computed in the package's graph from `spk`.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, bail};
use rlx_core::export::graph::{Pipeline, Stage, broadcast_params, symbolic_in};
use rlx_core::export::{Checkpoint, ExportOptions, ExportReport};
use rlx_ir::{DType, Graph, GraphExt, Op, Shape};
use serde_json::json;

use crate::graph::build_decode_graph;
use crate::model::{FacodecWeights, IN_CH};

/// The symbol for the number of frames.
const FRAMES: u32 = 0;

/// Timbre values no weight has, so the params holding them are found.
fn sentinels() -> (Vec<f32>, Vec<f32>) {
    let g = (0..IN_CH).map(|i| 1.0e6 + i as f32).collect();
    let b = (0..IN_CH).map(|i| 2.0e6 + i as f32).collect();
    (g, b)
}

fn decoder(w: &FacodecWeights, t: usize) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    let (gamma, beta) = sentinels();
    let (mut g, params, _) = build_decode_graph(w, &gamma, &beta, t)?;
    g.name = "decode".into();
    Ok((g, params.into_iter().collect()))
}

/// `graph` with the params holding `gamma` and `beta` computed from the
/// input `spk`: `style = spk · Wᵀ + b`, its halves.
fn timbre_from_input(
    graph: &Graph,
    params: &mut HashMap<String, Vec<f32>>,
    w: &FacodecWeights,
) -> Result<Graph> {
    let (gamma, beta) = sentinels();
    let find = |v: &Vec<f32>| params.iter().find(|(_, p)| *p == v).map(|(k, _)| k.clone());
    let (Some(gn), Some(bn)) = (find(&gamma), find(&beta)) else {
        bail!("the decoder holds no timbre params to take from the speaker");
    };
    params.remove(&gn);
    params.remove(&bn);
    params.insert("timbre.weight".into(), w.timbre_w.clone());
    params.insert("timbre.bias".into(), w.timbre_b.clone());
    let f32s = |d: &[usize]| Shape::new(d, DType::F32);
    let mut style = None;
    rlx_ir::stateful::rebuild(graph, |out, _, node, _| {
        let Op::Param { name } = &node.op else {
            return Ok(None);
        };
        let half = if *name == gn {
            0
        } else if *name == bn {
            IN_CH
        } else {
            return Ok(None);
        };
        let s = *style.get_or_insert_with(|| {
            let spk = out.input("spk", f32s(&[1, IN_CH]));
            let wt = out.param("timbre.weight", f32s(&[2 * IN_CH, IN_CH]));
            let wt = out.transpose_(wt, vec![1, 0]);
            let b = out.param("timbre.bias", f32s(&[2 * IN_CH]));
            let s = out.mm(spk, wt);
            out.add(s, b)
        });
        let part = out.narrow_(s, 1, half, IN_CH);
        Ok(Some(out.reshape(
            part,
            vec![1, IN_CH as i64],
            node.shape.clone(),
        )))
    })
    .map_err(anyhow::Error::msg)
}

/// Export the FACodec decoder in `path` (its `.safetensors`) as the package
/// `out`.
pub fn export_rlxp(path: &Path, out: &Path, opts: &ExportOptions) -> Result<ExportReport> {
    let ck = Checkpoint::open(path)?;
    let file = ck
        .weight_files()?
        .into_iter()
        .next()
        .context("no safetensors file")?;
    let w = FacodecWeights::from_safetensors(&std::fs::read(&file)?)?;
    let decode = symbolic_in(FRAMES, (5, 7, 3), |t| Ok(decoder(&w, t)?.0))?;
    let (at, mut params) = decoder(&w, 3)?;
    let decode = broadcast_params(&decode, &at, &mut params, &[])?;
    let decode = timbre_from_input(&decode, &mut params, &w)?;
    let stage = Stage::new(decode).outputs(&["audio"]);
    let mut p = Pipeline::new("facodec", "audio-decoding", vec![stage]);
    p.inputs = vec![
        json!({"name": "emb", "kind": "tensor", "dtype": "f32", "shape": [IN_CH, null]}),
        json!({"name": "spk", "kind": "tensor", "dtype": "f32", "shape": [1, IN_CH]}),
    ];
    p.dims = vec![("frames".into(), FRAMES)];
    p.metadata = vec![("sample_rate".into(), json!(16_000))];
    p.exported_by = "rlx-facodec".into();
    p.write(params, Some(&ck), out, opts)
}

/// Register this crate's exporter ([`rlx_core::export::register_exporter`]).
pub fn register() {
    use rlx_core::export::{Exporter, register_exporter};
    register_exporter(Exporter::new("facodec", "audio-decoding", export_rlxp).claims(&["facodec"]));
}
