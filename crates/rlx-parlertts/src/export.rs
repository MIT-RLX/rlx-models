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

//! Export Parler-TTS as an `.rlxp` package rlx runs alone — a transcript
//! and a voice description in, 44.1 kHz speech out (`Model::synthesize`,
//! the description `options.description`) — through the one exporter
//! ([`rlx_core::export`]).
//!
//! Its graphs: the T5 encoder and the decoder, the checkpoint's ONNX
//! imported once, here, as the runner imports them, symbolic in the
//! description's ids and the transcript's (the decoder at the runner's
//! fixed `max_steps`, its causal mask rebuilt from the length); `pick`, one
//! step's logits; and DAC's lookup and decoder ([`rlx_dac::export::dac_functions`]).
//! The sampling loop is the package's program (`program/`).

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use rlx_core::export::graph::{symbolic_in, symbolic_in_lifted_many};
use rlx_core::export::{Checkpoint, ExportOptions, ExportReport, Package, program};
use rlx_ir::{DType, Graph, GraphExt, Shape};
use rlx_onnx_import::{ImportOptions, build_hir_from_onnx_file};
use serde_json::json;

/// The symbols: DAC's frames and codebooks, the description's ids, the
/// transcript's, the decoder's logits' positions.
const DAC_FRAMES: u32 = 0;
const DAC_BOOKS: u32 = 1;
const DESCRIPTION: u32 = 2;
const PROMPT: u32 = 3;
const POSITIONS: u32 = 4;

/// The ONNX graph `name` imported as the runner imports it, its params
/// renamed `prefix` + theirs and kept.
fn import(
    dir: &Path,
    name: &str,
    prefix: &str,
    seq: usize,
    named: &[(&str, usize)],
    params: &RefCell<HashMap<String, Vec<f32>>>,
) -> Result<Graph> {
    let mut named_lengths: HashMap<String, usize> =
        named.iter().map(|(k, v)| (k.to_string(), *v)).collect();
    named_lengths.insert("batch_size".into(), 1);
    let opts = ImportOptions {
        sequence_length: seq,
        named_lengths,
        strict: false,
        ..Default::default()
    };
    let path = dir.join("onnx").join(format!("{name}.onnx"));
    let (hir, weights, _, _) =
        build_hir_from_onnx_file(&path, opts).with_context(|| format!("import {name}"))?;
    let g = hir
        .lower_to_mir()
        .map_err(|e| anyhow::anyhow!("lower {name}: {e:?}"))?
        .into_graph();
    // A param that holds one value throughout (a `ConstantOfShape` the
    // import makes a param, as long as the length) is that value as a
    // constant: the same numbers, and rebuilt from the length.
    let uniform = |v: &[f32]| !v.is_empty() && v.iter().all(|x| x.to_bits() == v[0].to_bits());
    let mut g = rlx_ir::stateful::rebuild(&g, |out, _, node, _| {
        let rlx_ir::Op::Param { name: p } = &node.op else {
            return Ok(None);
        };
        Ok(Some(match weights.get(p) {
            Some(v) if uniform(v) => out.add_node(
                rlx_ir::Op::Constant {
                    data: v.iter().flat_map(|x| x.to_le_bytes()).collect(),
                },
                vec![],
                node.shape.clone(),
            ),
            _ => out.param(format!("{prefix}{p}"), node.shape.clone()),
        }))
    })
    .map_err(anyhow::Error::msg)?;
    rlx_ir::dynamic::sync_graph_shapes(&mut g);
    g.name = name.into();
    let mut held = params.borrow_mut();
    for (k, v) in weights {
        if !uniform(&v) {
            held.entry(format!("{prefix}{k}")).or_insert(v);
        }
    }
    Ok(g)
}

/// `logits` (`dims`, its positions at `axis`) at `position` → `[…, vocab]`
/// without that axis: one step's rows.
fn pick(dims: &[usize], axis: usize) -> Graph {
    let mut g = Graph::new("pick");
    let logits = g.input("logits", Shape::new(dims, DType::F32));
    let at = g.input("position", Shape::new(&[1], DType::I64));
    let row = g.gather_(logits, at, axis);
    let mut out: Vec<i64> = dims.iter().map(|d| *d as i64).collect();
    out.remove(axis);
    let row = g.reshape_(row, out);
    g.node_mut(row).name = Some("rows".into());
    g.set_outputs(vec![row]);
    g
}

/// DAC beside the checkpoint in `dir`: its `dac/`, else a sibling `parler-dac/`.
fn dac_dir(dir: &Path) -> Result<PathBuf> {
    let beside = dir.parent().map(|p| p.join("parler-dac"));
    for p in std::iter::once(dir.join("dac")).chain(beside) {
        if p.join("model.safetensors").is_file() {
            return Ok(p);
        }
    }
    bail!(
        "Parler-TTS decodes with DAC: no DAC at {}/dac or beside it",
        dir.display()
    )
}

/// Export the Parler-TTS checkpoint in `dir` (`onnx/text_encoder.onnx`,
/// `onnx/decoder.onnx`, `tokenizer.json`, DAC beside it) as the package `out`.
pub fn export_rlxp(dir: &Path, out: &Path, opts: &ExportOptions) -> Result<ExportReport> {
    let ck = Checkpoint::open(dir)?;
    let mut pkg =
        Package::for_checkpoint(&ck, opts, "parler-tts", "text-to-speech", "rlx-parlertts");
    let defaults = crate::InferOpts::default();
    let t = defaults.max_steps;

    let held = RefCell::new(HashMap::new());
    let mut params = HashMap::new();
    let ids = (DESCRIPTION, vec![9, 13, 22, 31]);
    let (encoder, tables) = symbolic_in_lifted_many(std::slice::from_ref(&ids), &[1024], |n| {
        import(
            dir,
            "text_encoder",
            "enc.",
            n[0],
            &[("sequence_length", n[0]), ("t", n[0])],
            &held,
        )
    })
    .context("text_encoder")?;
    params.extend(tables);
    let (mut decoder, tables) =
        symbolic_in_lifted_many(&[ids, (PROMPT, vec![7, 15, 12, 26])], &[1024, 1024], |n| {
            import(
                dir,
                "decoder",
                "dec.",
                t,
                &[
                    ("t", t),
                    ("et", n[0]),
                    ("pt", n[1]),
                    ("sequence_length", t),
                    ("encoder_sequence_length", n[0]),
                ],
                &held,
            )
        })
        .context("decoder")?;
    params.extend(tables);
    params.extend(held.take());
    let first = decoder.outputs[0];
    decoder.node_mut(first).name = Some("logits".into());
    // `pick` over the logits' positions (the axis before the vocabulary).
    let dims: Vec<usize> = decoder
        .node(first)
        .shape
        .dims()
        .iter()
        .map(|d| match d {
            rlx_ir::Dim::Static(n) => *n,
            rlx_ir::Dim::Dynamic(_) => 0,
        })
        .collect();
    let axis = dims.len() - 2;
    let pick = symbolic_in(POSITIONS, (431, 437, 445), |s| {
        let mut d = dims.clone();
        d[axis] = s;
        Ok(pick(&d, axis))
    })?;

    let dac = Checkpoint::open(&dac_dir(dir)?)?;
    let (dac_graphs, dac_params, dac_cfg) =
        rlx_dac::export::dac_functions(&dac, DAC_FRAMES, DAC_BOOKS)?;
    params.extend(dac_params);
    for g in [encoder, decoder, pick].into_iter().chain(dac_graphs) {
        pkg.function(g);
    }
    pkg.dims = vec![
        ("dac_frames".into(), DAC_FRAMES),
        ("codebooks".into(), DAC_BOOKS),
        ("description_ids".into(), DESCRIPTION),
        ("transcript_ids".into(), PROMPT),
        ("positions".into(), POSITIONS),
    ];
    let pdims = pkg.param_dims();
    let mut names: Vec<&String> = params.keys().collect();
    names.sort();
    for n in names {
        let d = pdims
            .get(n)
            .cloned()
            .unwrap_or_else(|| vec![params[n].len()]);
        pkg.hold_f32(n.clone(), d, &params[n]);
    }
    pkg.sidecar(
        "tokenizer.json",
        "application/json",
        std::fs::read(dir.join("tokenizer.json"))?,
    );

    let module = program::build(&Path::new(env!("CARGO_MANIFEST_DIR")).join("program/Cargo.toml"))?;
    program::entry(
        &mut pkg,
        "speak",
        module,
        &["synthesize"],
        "text-to-speech",
        json!({
            "codebooks": 9,
            "bos": 1025,
            "pad": 1024,
            "vocab": 1088,
            "dac_max": 1024,
            "max_steps": t,
            "sample_rate": dac_cfg.sample_rate,
            "description": crate::native::DEFAULT_DESCRIPTION,
            "temperature": defaults.temperature,
            "top_k": defaults.top_k,
            "seed": defaults.seed,
        }),
    );
    pkg.metadata("sample_rate", json!(dac_cfg.sample_rate));
    pkg.write(out, opts)
}

/// Register this crate's exporter ([`rlx_core::export::register_exporter`]).
pub fn register() {
    use rlx_core::export::{Exporter, register_exporter};
    register_exporter(
        Exporter::new("parler-tts", "text-to-speech", export_rlxp).claims(&[
            "parler_tts",
            "parler-tts",
            "parlertts",
        ]),
    );
}
