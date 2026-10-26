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

//! Export a SNAC decoder as an `.rlxp` package rlx runs alone, through the
//! one exporter ([`rlx_core::export`]): each level's codes in (`codes_0`
//! the coarsest), the waveform out, for any number of codes.
//!
//! The codebook lookup the codec does on the host ([`crate::eager::from_codes`]
//! — each level's rows through its 1×1 projection, repeated to the finest
//! rate, summed) is the package's first stage; the decoder is the second,
//! its noise planes zero (the codec's deterministic `decode(codes, None)`).
//! Both are symbolic in the coarsest level's length.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use rlx_core::export::graph::{Pipeline, Stage, broadcast_params, symbolic_in};
use rlx_core::export::{Checkpoint, ExportOptions, ExportReport};
use rlx_ir::op::BinaryOp;
use rlx_ir::{DType, Graph, GraphExt, NodeId, Op, Shape};
use serde_json::json;

use crate::config::SnacConfig;
use crate::graph::build_decode_graph;
use crate::model::SnacWeights;

/// The symbol for the coarsest level's length.
const FRAMES: u32 = 0;

fn f32s(d: &[usize]) -> Shape {
    Shape::new(d, DType::F32)
}

/// Each level's codes (level `i` `n · s₀ / sᵢ` long) → the latent `z`
/// `[1, latent, n · s₀, 1]`, as `from_codes` builds it.
fn codes_to_latent(w: &SnacWeights, n: usize) -> (Graph, HashMap<String, Vec<f32>>) {
    let cfg = &w.config;
    let latent = w.latent();
    let cb = cfg.codebook_dim;
    let t = n * cfg.vq_strides[0];
    let mut g = Graph::new("codes_to_latent");
    let mut params = HashMap::new();
    let mut sum: Option<NodeId> = None;
    for (i, q) in w.quantizers.iter().enumerate() {
        let tl = t / q.stride;
        let codes = g.input(format!("codes_{i}"), Shape::new(&[tl], DType::I32));
        let book = format!("quantizer.{i}.codebook");
        let proj = format!("quantizer.{i}.out_proj.weight");
        let bias = format!("quantizer.{i}.out_proj.bias");
        params.insert(book.clone(), q.codebook.clone());
        params.insert(proj.clone(), q.out_proj_w.clone());
        params.insert(bias.clone(), q.out_proj_b.clone());
        let book = g.param(book, f32s(&[cfg.codebook_size, cb]));
        let proj = g.param(proj, f32s(&[latent, cb, 1]));
        let bias = g.param(bias, f32s(&[latent]));
        let rows = g.gather(book, codes, 0, f32s(&[tl, cb]));
        let w2 = g.reshape_(proj, vec![latent as i64, cb as i64]);
        let wt = g.transpose_(w2, vec![1, 0]);
        let x = g.mm(rows, wt);
        let x = g.add(x, bias);
        // Each frame repeated `stride` times: [tl, latent] → [t, latent].
        let x = g.reshape(x, vec![tl as i64, 1, latent as i64], f32s(&[tl, 1, latent]));
        let x = g.add_node(
            Op::Expand {
                target_shape: vec![tl as i64, q.stride as i64, latent as i64],
            },
            vec![x],
            f32s(&[tl, q.stride, latent]),
        );
        let x = g.reshape(x, vec![t as i64, latent as i64], f32s(&[t, latent]));
        sum = Some(match sum {
            Some(s) => g.binary(BinaryOp::Add, s, x, f32s(&[t, latent])),
            None => x,
        });
    }
    let z = g.transpose_(sum.expect("a quantizer"), vec![1, 0]);
    let z = g.reshape(
        z,
        vec![1, latent as i64, t as i64, 1],
        f32s(&[1, latent, t, 1]),
    );
    g.node_mut(z).name = Some("latent".into());
    g.set_outputs(vec![z]);
    (g, params)
}

/// The decoder at `n` coarse frames, its noise planes zero.
fn decoder(w: &SnacWeights, n: usize) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    let t = n * w.config.vq_strides[0];
    let (g, params, _, noise) = build_decode_graph(w, t)?;
    let silent: Vec<&str> = noise.iter().map(|(name, _)| name.as_str()).collect();
    let mut g = rlx_ir::stateful::rebuild(&g, |out, _, node, _| {
        let Op::Input { name } = &node.op else {
            return Ok(None);
        };
        if !silent.contains(&name.as_str()) {
            return Ok(None);
        }
        let zero = out.add_node(
            Op::Constant {
                data: 0f32.to_le_bytes().to_vec(),
            },
            vec![],
            f32s(&[1]),
        );
        let target = node
            .shape
            .dims()
            .iter()
            .map(|d| match d {
                rlx_ir::Dim::Static(n) => *n as i64,
                rlx_ir::Dim::Dynamic(_) => -1,
            })
            .collect();
        Ok(Some(out.add_node(
            Op::Expand {
                target_shape: target,
            },
            vec![zero],
            node.shape.clone(),
        )))
    })
    .map_err(anyhow::Error::msg)?;
    g.name = "decode".into();
    Ok((g, params.into_iter().collect()))
}

/// The lookup and the decoder, symbolic in the coarsest level's length
/// (symbol `frames`), with their params.
fn functions(w: &SnacWeights, frames: u32) -> Result<(Graph, Graph, HashMap<String, Vec<f32>>)> {
    let lookup = symbolic_in(frames, (5, 7, 3), |n| Ok(codes_to_latent(w, n).0))?;
    let decode = symbolic_in(frames, (5, 7, 3), |n| Ok(decoder(w, n)?.0))?;
    let mut params = codes_to_latent(w, 1).1;
    // The decoder's padding blocks, materialized at the length, as one
    // slice expanded.
    let (at, mut decoder_params) = decoder(w, 3)?;
    let decode = broadcast_params(&decode, &at, &mut decoder_params, &[])?;
    params.extend(decoder_params);
    Ok((lookup, decode, params))
}

/// The SNAC decoder in the `.safetensors` `file` (the 24 kHz model's
/// config) as two functions symbolic in the coarsest level's length
/// (symbol `frames`) — `snac_latent` (`codes_i` i32 → `latent`) and
/// `snac_decode` (`z` → `audio`) — with their params and the config: for a
/// package whose model ends in SNAC (Orpheus).
pub fn snac_functions(
    file: &Path,
    frames: u32,
) -> Result<(Vec<Graph>, HashMap<String, Vec<f32>>, SnacConfig)> {
    let cfg = SnacConfig::snac_24khz();
    let w = SnacWeights::from_safetensors(&std::fs::read(file)?, cfg.clone())?;
    let (mut lookup, mut decode, params) = functions(&w, frames)?;
    lookup.name = "snac_latent".into();
    decode.name = "snac_decode".into();
    Ok((vec![lookup, decode], params, cfg))
}

/// Export the SNAC decoder in `path` (its folded-weight-norm
/// `.safetensors`, beside an optional `config.json` of SNAC's fields; the
/// 24 kHz model otherwise) as the package `out`.
pub fn export_rlxp(path: &Path, out: &Path, opts: &ExportOptions) -> Result<ExportReport> {
    let ck = Checkpoint::open(path)?;
    let cfg = serde_json::from_value::<SnacConfig>(ck.config.clone())
        .unwrap_or_else(|_| SnacConfig::snac_24khz());
    let file = ck
        .weight_files()?
        .into_iter()
        .next()
        .context("no safetensors file")?;
    let w = SnacWeights::from_safetensors(&std::fs::read(&file)?, cfg)?;
    let (lookup, decode, params) = functions(&w, FRAMES)?;

    let levels = w.quantizers.len();
    let mut lookup_stage = Stage::new(lookup).outputs(&["latent"]);
    for i in 0..levels {
        let name = format!("codes_{i}");
        lookup_stage = lookup_stage.input(&name, &name);
    }
    let decode_stage = Stage::new(decode).input("z", "latent").outputs(&["audio"]);
    let mut p = Pipeline::new("snac", "audio-decoding", vec![lookup_stage, decode_stage]);
    p.inputs = (0..levels)
        .map(|i| {
            json!({"name": format!("codes_{i}"), "kind": "tensor", "dtype": "i32",
                   "shape": [null], "stride": w.quantizers[i].stride})
        })
        .collect();
    p.dims = vec![("frames".into(), FRAMES)];
    p.metadata = vec![
        ("sample_rate".into(), json!(w.config.sampling_rate)),
        ("levels".into(), json!(levels)),
        (
            "codes".into(),
            json!("codes_i holds frames · strides[0] / strides[i] ids; codes_0 the coarsest"),
        ),
    ];
    p.exported_by = "rlx-snac".into();
    p.write(params, Some(&ck), out, opts)
}

/// Register this crate's exporter ([`rlx_core::export::register_exporter`]).
pub fn register() {
    use rlx_core::export::{Exporter, register_exporter};
    register_exporter(Exporter::new("snac", "audio-decoding", export_rlxp).claims(&["snac"]));
}
