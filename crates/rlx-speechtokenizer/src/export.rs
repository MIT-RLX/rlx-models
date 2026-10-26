// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
//! Export the SpeechTokenizer decoder as an `.rlxp` package rlx runs alone,
//! through the one exporter ([`rlx_core::export`]): codes `[codebooks,
//! frames]` in — any number of each (codebook 0 alone is the semantic
//! tokens) — the waveform out.
//!
//! What [`crate::codec::SpeechTokenizerCodec::decode`] does on the host is
//! in the package: the residual codebooks' rows summed
//! ([`crate::eager::rvq_decode`], one gather from the codebooks stacked),
//! and the LSTM between its two graphs ([`crate::lstm::lstm`], `Op::Lstm`
//! plus the skip).

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, bail};
use rlx_core::export::graph::{Pipeline, Stage, broadcast_params, symbolic_in};
use rlx_core::export::{Checkpoint, ExportOptions, ExportReport};
use rlx_ir::{DType, Graph, GraphExt, Op, Shape};
use serde_json::json;

use crate::config::SpeechTokenizerConfig;
use crate::graph::{build_dec_post, build_dec_pre};
use crate::model::{DecoderW, StWeights};

/// The symbols: the number of frames and of codebooks.
const FRAMES: u32 = 0;
const CODEBOOKS: u32 = 1;

fn f32s(d: &[usize]) -> Shape {
    Shape::new(d, DType::F32)
}

/// `codes` `[n, t]` → `zq` `[1, dim, t, 1]`: codebook `k`'s row for each
/// code (`codes[k] + k · bins` in the codebooks stacked), summed over `k`.
fn lookup(cfg: &SpeechTokenizerConfig, books: usize, n: usize, t: usize) -> Graph {
    let (bins, dim) = (cfg.codebook_size, cfg.dimension);
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
/// [`crate::lstm::lstm`] computes it.
fn lstm(dec: &DecoderW, dim: usize, t: usize) -> Graph {
    let layers = dec.lstm.len();
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
    for l in &dec.lstm {
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

type Built = Result<(Graph, Vec<(String, Vec<f32>)>, usize)>;

fn graph_of(built: Built, name: &str) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    let (mut g, params, _) = built?;
    g.name = name.into();
    Ok((g, params.into_iter().collect()))
}

/// One of the decoder's two graphs, symbolic in the frames, its params
/// prefixed `name.` (the audio graphs name theirs `w0`, `w1`, … each from
/// 0). Built at 4 frames and more: reflect padding of 3 needs 4.
fn half(name: &str, build: impl Fn(usize) -> Built) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    let g = symbolic_in(FRAMES, (5, 7, 4), |t| Ok(graph_of(build(t), name)?.0))?;
    let (at, mut params) = graph_of(build(4), name)?;
    let mut g = broadcast_params(&g, &at, &mut params, &[])?;
    for node in g.nodes_mut() {
        if let Op::Param { name: p } = &mut node.op {
            *p = format!("{name}.{p}");
        }
    }
    let params = params
        .into_iter()
        .map(|(k, v)| (format!("{name}.{k}"), v))
        .collect();
    Ok((g, params))
}

/// Export the SpeechTokenizer checkpoint in `path` (its `.safetensors`;
/// the 16 kHz `speechtokenizer_hubert_avg` model) as the package `out`.
pub fn export_rlxp(path: &Path, out: &Path, opts: &ExportOptions) -> Result<ExportReport> {
    let ck = Checkpoint::open(path)?;
    let cfg = SpeechTokenizerConfig::default_16khz();
    let file = ck
        .weight_files()?
        .into_iter()
        .next()
        .context("no safetensors file")?;
    let w = StWeights::from_safetensors(&std::fs::read(&file)?, cfg.clone())?;
    let dec = &w.decoder;
    let books = w.codebooks.len();
    if books < 3 {
        bail!("{books} codebooks: too few to tell the codebook count from the frames");
    }

    // The lookup, symbolic in the frames and the codebooks; checked at a
    // third size of each.
    let at = |n: usize, t: usize| lookup(&cfg, books, n, t);
    let mut lookup_g =
        rlx_ir::dynamic::generalize(&at(2, 5), &at(3, 7), &[(FRAMES, 5, 7), (CODEBOOKS, 2, 3)])
            .map_err(|e| anyhow::anyhow!("the lookup is not symbolic: {e}"))?;
    let mut bind = rlx_ir::DimBinding::new();
    bind.set(FRAMES, 4);
    bind.set(CODEBOOKS, books);
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

    let dim = cfg.dimension;
    let (pre, pre_params) = half("pre", |t| build_dec_pre(dec, t))?;
    let (post, post_params) = half("post", |t| build_dec_post(dec, dim, t))?;
    params.extend(pre_params);
    params.extend(post_params);
    params.extend(lstm_params(dec));
    let lstm_g = symbolic_in(FRAMES, (5, 7, 3), |t| Ok(lstm(dec, dim, t)))?;

    let stages = vec![
        Stage::new(lookup_g).outputs(&["zq"]),
        Stage::new(pre).input("zq", "zq").outputs(&["pre"]),
        Stage::new(lstm_g).input("x", "pre").outputs(&["z"]),
        Stage::new(post).input("z", "z").outputs(&["audio"]),
    ];
    let mut p = Pipeline::new("speechtokenizer", "audio-decoding", stages);
    p.inputs = vec![json!({"name": "codes", "kind": "tensor", "dtype": "i32",
                           "shape": [null, null], "codebooks_at_most": books})];
    p.dims = vec![("frames".into(), FRAMES), ("codebooks".into(), CODEBOOKS)];
    p.metadata = vec![
        ("sample_rate".into(), json!(cfg.sampling_rate)),
        ("hop".into(), json!(cfg.hop_length())),
        (
            "codes".into(),
            json!("codebook 0 holds the semantic tokens; the rest add acoustic detail"),
        ),
    ];
    p.exported_by = "rlx-speechtokenizer".into();
    p.write(params, Some(&ck), out, opts)
}

/// Register this crate's exporter ([`rlx_core::export::register_exporter`]).
pub fn register() {
    use rlx_core::export::{Exporter, register_exporter};
    register_exporter(
        Exporter::new("speechtokenizer", "audio-decoding", export_rlxp)
            .claims(&["speechtokenizer"]),
    );
}
