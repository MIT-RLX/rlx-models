// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
//! Export a DAC decoder as an `.rlxp` package rlx runs alone, through the
//! one exporter ([`rlx_core::export`]): codes `[codebooks, frames]` in —
//! any number of each — the waveform out.
//!
//! What [`crate::DacCodec::decode_codes`] does on the host is in the
//! package: each codebook's rows through its output projection, summed
//! ([`crate::quantize::ResidualVectorQuantizer::from_codes`]; one gather
//! from the codebooks stacked and one batched matmul), then the decoder.
//! The weight-normalized convolutions are folded, so their weights are the
//! package's own.

use std::path::Path;

use anyhow::{Context, Result, bail};
use rlx_core::export::graph::{Pipeline, Stage, broadcast_params, symbolic_in};
use rlx_core::export::{Checkpoint, ExportOptions, ExportReport};
use rlx_ir::{DType, Graph, GraphExt, Op, Shape};
use serde_json::json;
use std::collections::HashMap;

use crate::build::build_decoder;
use crate::config::DacConfig;
use crate::graph::build_decoder_graph;
use crate::quantize::build_quantizer;
use crate::weights::WeightStore;

/// The symbols: the number of frames and of codebooks.
const FRAMES: u32 = 0;
const CODEBOOKS: u32 = 1;

fn f32s(d: &[usize]) -> Shape {
    Shape::new(d, DType::F32)
}

/// `codes` `[n, t]` → `z` `[1, latent, t, 1]`: codebook `k`'s row for each
/// code (`codes[k] + k · bins` in the codebooks stacked), through
/// projection `k`, summed over `k`.
fn lookup(cfg: &DacConfig, books: usize, n: usize, t: usize) -> Graph {
    let (bins, dim, latent) = (cfg.codebook_size, cfg.codebook_dim, cfg.latent_dim);
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
    // Codebook `k`'s rows through projection `k`: `[n, t, dim] @ [n, dim, latent]`.
    let proj = g.param("out_proj.weight", f32s(&[books, dim, latent]));
    let proj = g.narrow_(proj, 0, 0, n);
    let x = g.mm(picked, proj);
    let bias = g.param("out_proj.bias", f32s(&[books, latent]));
    let bias = g.narrow_(bias, 0, 0, n);
    let bias = g.reshape(
        bias,
        vec![n as i64, 1, latent as i64],
        f32s(&[n, 1, latent]),
    );
    let bias = g.add_node(
        Op::Expand {
            target_shape: vec![n as i64, t as i64, latent as i64],
        },
        vec![bias],
        f32s(&[n, t, latent]),
    );
    let x = g.add(x, bias);
    let sum = g.sum(x, vec![0], false);
    let z = g.transpose_(sum, vec![1, 0]);
    let z = g.reshape(
        z,
        vec![1, latent as i64, t as i64, 1],
        f32s(&[1, latent, t, 1]),
    );
    g.node_mut(z).name = Some("z".into());
    g.set_outputs(vec![z]);
    g
}

/// Export the DAC checkpoint in `dir` (`model.safetensors`, beside a
/// `config.json` of DAC's fields; the 24 kHz model otherwise) as the
/// package `out`.
pub fn export_rlxp(dir: &Path, out: &Path, opts: &ExportOptions) -> Result<ExportReport> {
    let ck = Checkpoint::open(dir)?;
    let (graphs, params, cfg) = dac_functions(&ck, FRAMES, CODEBOOKS)?;
    let books = cfg.n_codebooks;
    let mut graphs = graphs.into_iter();
    let (lookup_g, decode) = (
        graphs.next().context("lookup")?,
        graphs.next().context("decode")?,
    );
    let stages = vec![
        Stage::new(lookup_g).outputs(&["z"]),
        Stage::new(decode).input("z", "z").outputs(&["audio"]),
    ];
    let mut p = Pipeline::new("dac", "audio-decoding", stages);
    p.inputs = vec![json!({"name": "codes", "kind": "tensor", "dtype": "i32",
                           "shape": [null, null], "codebooks_at_most": books})];
    p.dims = vec![("frames".into(), FRAMES), ("codebooks".into(), CODEBOOKS)];
    p.metadata = vec![
        ("sample_rate".into(), json!(cfg.sample_rate)),
        ("hop".into(), json!(cfg.hop_length)),
    ];
    p.exported_by = "rlx-dac".into();
    p.write(params, Some(&ck), out, opts)
}

/// The DAC decoder in `ck` as two functions, symbolic in the frames and
/// the codebooks (their symbols `frames`, `books`) — `lookup` (`codes`
/// `[n, t]` i32 → `z`) and `decode` (`z` → `audio`) — with their params
/// and the config: for a package whose model ends in DAC (Parler-TTS).
pub fn dac_functions(
    ck: &Checkpoint,
    frames: u32,
    books_sym: u32,
) -> Result<(Vec<Graph>, HashMap<String, Vec<f32>>, DacConfig)> {
    let cfg = match serde_json::from_value::<DacConfig>(ck.config.clone()) {
        Ok(c) => c,
        Err(_) => DacConfig::config_24khz(),
    };
    cfg.validate()?;
    let file = ck
        .weight_files()?
        .into_iter()
        .find(|f| f.extension().is_some_and(|e| e == "safetensors"))
        .context("no safetensors file")?;
    let store = WeightStore::open(&file)?;
    let quantizer = build_quantizer(&cfg, &store)?;
    let decoder = build_decoder(&cfg, &store)?;
    let books = cfg.n_codebooks;
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
    let (mut table, mut proj, mut bias) = (Vec::new(), Vec::new(), Vec::new());
    for (codebook, out_proj) in quantizer.decoding() {
        table.extend(codebook.iter());
        // `[latent, dim, 1]` → `[dim, latent]`.
        let w = out_proj.weight.index_axis(ndarray::Axis(2), 0);
        proj.extend(w.t().iter());
        match &out_proj.bias {
            Some(b) => bias.extend(b.iter()),
            None => bias.extend(std::iter::repeat_n(0.0, cfg.latent_dim)),
        }
    }
    params.insert("codebooks".into(), table);
    params.insert("out_proj.weight".into(), proj);
    params.insert("out_proj.bias".into(), bias);

    let latent = cfg.latent_dim;
    let build = |t: usize| -> Result<(Graph, HashMap<String, Vec<f32>>)> {
        let (mut g, p, _) = build_decoder_graph(&decoder, latent, t)?;
        g.name = "decode".into();
        Ok((g, p.into_iter().collect()))
    };
    let decode = symbolic_in(frames, (5, 7, 4), |t| Ok(build(t)?.0))?;
    let (at4, mut decoder_params) = build(4)?;
    let decode = broadcast_params(&decode, &at4, &mut decoder_params, &[])?;
    params.extend(decoder_params);
    Ok((vec![lookup_g, decode], params, cfg))
}

/// Register this crate's exporter ([`rlx_core::export::register_exporter`]).
pub fn register() {
    use rlx_core::export::{Exporter, register_exporter};
    register_exporter(Exporter::new("dac", "audio-decoding", export_rlxp).claims(&["dac"]));
}
