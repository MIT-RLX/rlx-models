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

//! Export the EnCodec decoder as an `.rlxp` package rlx runs alone, through
//! the one exporter ([`rlx_core::export`]): codes `[codebooks, frames]` in
//! — any number of each, so any bandwidth — the waveform out.
//!
//! What [`crate::EncodecCodec::decode`] does on the host is in the
//! package: the residual codebooks' rows summed ([`crate::eager::rvq_decode`],
//! one gather from the codebooks stacked), and the LSTM between its two
//! graphs ([`crate::lstm::forward`], `Op::Lstm` plus the residual).

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, bail};
use rlx_core::export::graph::{Pipeline, Stage, broadcast_params, symbolic_in};
use rlx_core::export::{Checkpoint, ExportOptions, ExportReport};
use rlx_ir::{DType, Graph, GraphExt, Op, Shape};
use serde_json::json;

use crate::config::EncodecConfig;
use crate::graph::{build_decode_post_lstm_graph, build_decode_pre_lstm_graph};
use crate::model::{DecoderW, EncodecWeights};

/// The symbols: the number of frames and of codebooks.
const FRAMES: u32 = 0;
const CODEBOOKS: u32 = 1;

fn f32s(d: &[usize]) -> Shape {
    Shape::new(d, DType::F32)
}

/// `codes` `[n, t]` → `zq` `[1, dim, t, 1]`: codebook `k`'s row for each
/// code (`codes[k] + k · bins` in the codebooks stacked), summed over `k`.
fn lookup(cfg: &EncodecConfig, books: usize, n: usize, t: usize) -> Graph {
    let (bins, dim) = (cfg.codebook_size, cfg.codebook_dim);
    let mut g = Graph::new("lookup");
    let codes = g.input("codes", Shape::new(&[n, t], DType::I32));
    let codes = g.add_node(Op::Cast { to: DType::F32 }, vec![codes], f32s(&[n, t]));
    let offsets = g.param("codebook.offsets", f32s(&[books]));
    let offsets = g.narrow_(offsets, 0, 0, n);
    let offsets = g.reshape(offsets, vec![n as i64, 1], f32s(&[n, 1]));
    let offsets = g.add_node(
        Op::Expand {
            target_shape: vec![n as i64, t as i64],
        },
        vec![offsets],
        f32s(&[n, t]),
    );
    let rows = g.add(codes, offsets);
    let rows = g.reshape(rows, vec![(n * t) as i64], f32s(&[n * t]));
    let rows = g.add_node(
        Op::Cast { to: DType::I32 },
        vec![rows],
        Shape::new(&[n * t], DType::I32),
    );
    let table = g.param("codebooks", f32s(&[books * bins, dim]));
    let picked = g.gather(table, rows, 0, f32s(&[n * t, dim]));
    let picked = g.reshape(
        picked,
        vec![n as i64, t as i64, dim as i64],
        f32s(&[n, t, dim]),
    );
    let sum = g.sum(picked, vec![0], false);
    let z = g.transpose_(sum, vec![1, 0]);
    let z = g.reshape(z, vec![1, dim as i64, t as i64, 1], f32s(&[1, dim, t, 1]));
    g.node_mut(z).name = Some("zq".into());
    g.set_outputs(vec![z]);
    g
}

/// `x` `[1, dim, t, 1]` → `lstm(x) + x`, the same shape, as
/// [`crate::lstm::forward`] computes it.
fn lstm(dec: &DecoderW, t: usize) -> Graph {
    let dim = dec.lstm.dim;
    let layers = dec.lstm.layers.len();
    let mut g = Graph::new("lstm");
    let x = g.input("x", f32s(&[1, dim, t, 1]));
    let seq = g.reshape(x, vec![dim as i64, t as i64], f32s(&[dim, t]));
    let seq = g.transpose_(seq, vec![1, 0]);
    let seq = g.reshape(seq, vec![1, t as i64, dim as i64], f32s(&[1, t, dim]));
    let w_ih = g.param("lstm.w_ih", f32s(&[layers * 4 * dim, dim]));
    let w_hh = g.param("lstm.w_hh", f32s(&[layers * 4 * dim, dim]));
    let bias = g.param("lstm.bias", f32s(&[layers * 4 * dim]));
    let y = g.lstm(
        seq,
        w_ih,
        w_hh,
        bias,
        dim,
        layers,
        false,
        f32s(&[1, t, dim]),
    );
    let y = g.add(y, seq);
    let y = g.reshape(y, vec![t as i64, dim as i64], f32s(&[t, dim]));
    let y = g.transpose_(y, vec![1, 0]);
    let y = g.reshape(y, vec![1, dim as i64, t as i64, 1], f32s(&[1, dim, t, 1]));
    g.node_mut(y).name = Some("z".into());
    g.set_outputs(vec![y]);
    g
}

/// The LSTM's layers packed as `Op::Lstm` takes them, the two biases summed.
fn lstm_params(dec: &DecoderW) -> HashMap<String, Vec<f32>> {
    let (mut ih, mut hh, mut b) = (Vec::new(), Vec::new(), Vec::new());
    for l in &dec.lstm.layers {
        ih.extend(&l.w_ih);
        hh.extend(&l.w_hh);
        b.extend(l.b_ih.iter().zip(&l.b_hh).map(|(x, y)| x + y));
    }
    HashMap::from([
        ("lstm.w_ih".to_string(), ih),
        ("lstm.w_hh".to_string(), hh),
        ("lstm.bias".to_string(), b),
    ])
}

fn graph_of(
    built: Result<(Graph, Vec<(String, Vec<f32>)>, usize)>,
    name: &str,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    let (mut g, params, _) = built?;
    g.name = name.into();
    Ok((g, params.into_iter().collect()))
}

/// The EnCodec decoder of the checkpoint `ck` as four functions, symbolic
/// in the frames (`frames`) and the codebooks (`books_sym`): `lookup`
/// (`codes` `[n, t]` i32 → `zq`), `pre` (`zq` → `pre`), `lstm` (`x` → `z`)
/// and `post` (`z` → `audio`), with their params and the config — for a
/// package whose model ends in EnCodec (MetaVoice).
pub fn encodec_functions(
    ck: &Checkpoint,
    frames: u32,
    books_sym: u32,
) -> Result<(Vec<Graph>, HashMap<String, Vec<f32>>, EncodecConfig)> {
    let cfg = EncodecConfig::encodec_24khz();
    let file = ck
        .weight_files()?
        .into_iter()
        .next()
        .context("no safetensors file")?;
    let w = EncodecWeights::from_safetensors(&std::fs::read(&file)?, cfg.clone())?;
    let dec = w
        .decoder
        .as_ref()
        .context("the checkpoint has no decoder")?;
    let books = w.codebooks.len();
    if books < 3 {
        bail!("{books} codebooks: too few to tell the codebook count from the frames");
    }

    // The lookup, symbolic in the frames and the codebooks; checked at a
    // third size of each.
    let at = |n: usize, t: usize| lookup(&cfg, books, n, t);
    let mut lookup_g =
        rlx_ir::dynamic::generalize(&at(2, 5), &at(3, 7), &[(frames, 5, 7), (books_sym, 2, 3)])
            .map_err(|e| anyhow::anyhow!("the lookup is not symbolic: {e}"))?;
    let mut bind = rlx_ir::DimBinding::new();
    bind.set(frames, 4);
    bind.set(books_sym, books);
    if format!("{}", rlx_ir::dynamic::specialize(&lookup_g, &bind)) != format!("{}", at(books, 4)) {
        bail!("the lookup generalized from two sizes is not the one built at a third");
    }
    lookup_g.name = "lookup".into();
    let mut params: HashMap<String, Vec<f32>> = HashMap::new();
    params.insert(
        "codebook.offsets".into(),
        (0..books).map(|k| (k * cfg.codebook_size) as f32).collect(),
    );
    params.insert("codebooks".into(), w.codebooks.concat());

    let pre = symbolic_in(frames, (5, 7, 3), |t| {
        Ok(graph_of(build_decode_pre_lstm_graph(dec, t), "pre")?.0)
    })?;
    let (pre_at, mut pre_params) = graph_of(build_decode_pre_lstm_graph(dec, 3), "pre")?;
    let pre = broadcast_params(&pre, &pre_at, &mut pre_params, &[])?;
    let lstm_dim = cfg.lstm_dim();
    let post = symbolic_in(frames, (5, 7, 3), |t| {
        Ok(graph_of(build_decode_post_lstm_graph(dec, lstm_dim, t), "post")?.0)
    })?;
    let (post_at, mut post_params) =
        graph_of(build_decode_post_lstm_graph(dec, lstm_dim, 3), "post")?;
    let post = broadcast_params(&post, &post_at, &mut post_params, &[])?;
    // The audio graphs name their params `w0`, `w1`, … each from 0: keep
    // the two graphs' apart.
    let (pre, pre_params) = prefix(pre, pre_params, "pre.");
    let (post, post_params) = prefix(post, post_params, "post.");
    params.extend(pre_params);
    params.extend(post_params);
    params.extend(lstm_params(dec));
    let lstm_g = symbolic_in(frames, (5, 7, 3), |t| Ok(lstm(dec, t)))?;
    Ok((vec![lookup_g, pre, lstm_g, post], params, cfg))
}

/// Export the 24 kHz EnCodec checkpoint in `path` (its `.safetensors`) as
/// the package `out`.
pub fn export_rlxp(path: &Path, out: &Path, opts: &ExportOptions) -> Result<ExportReport> {
    let ck = Checkpoint::open(path)?;
    let (graphs, params, cfg) = encodec_functions(&ck, FRAMES, CODEBOOKS)?;
    let books = params["codebook.offsets"].len();
    let [lookup_g, pre, lstm_g, post]: [Graph; 4] = graphs
        .try_into()
        .map_err(|_| anyhow::anyhow!("four EnCodec functions"))?;

    let stages = vec![
        Stage::new(lookup_g).outputs(&["zq"]),
        Stage::new(pre).input("zq", "zq").outputs(&["pre"]),
        Stage::new(lstm_g).input("x", "pre").outputs(&["z"]),
        Stage::new(post).input("z", "z").outputs(&["audio"]),
    ];
    let mut p = Pipeline::new("encodec", "audio-decoding", stages);
    p.inputs = vec![json!({"name": "codes", "kind": "tensor", "dtype": "i32",
                           "shape": [null, null], "codebooks_at_most": books})];
    p.dims = vec![("frames".into(), FRAMES), ("codebooks".into(), CODEBOOKS)];
    p.metadata = vec![
        ("sample_rate".into(), json!(cfg.sampling_rate)),
        ("hop".into(), json!(cfg.hop_length())),
    ];
    p.exported_by = "rlx-encodec".into();
    p.write(params, Some(&ck), out, opts)
}

/// Each param of `graph` (and its value) renamed `prefix` + its name.
fn prefix(
    graph: Graph,
    params: HashMap<String, Vec<f32>>,
    prefix: &str,
) -> (Graph, HashMap<String, Vec<f32>>) {
    let mut graph = graph;
    for node in graph.nodes_mut() {
        if let Op::Param { name } = &mut node.op {
            *name = format!("{prefix}{name}");
        }
    }
    let params = params
        .into_iter()
        .map(|(k, v)| (format!("{prefix}{k}"), v))
        .collect();
    (graph, params)
}

/// Register this crate's exporter ([`rlx_core::export::register_exporter`]).
pub fn register() {
    use rlx_core::export::{Exporter, register_exporter};
    register_exporter(Exporter::new("encodec", "audio-decoding", export_rlxp).claims(&["encodec"]));
}
