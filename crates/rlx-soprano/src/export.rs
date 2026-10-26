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

//! Export Soprano 1.1 as an `.rlxp` package rlx runs alone: text in,
//! 32 kHz speech out (`Model::synthesize`), through the one exporter
//! ([`rlx_core::export`]).
//!
//! Its two ONNX graphs (`onnx/`, the export-time reference) are converted
//! to rlx IR once: the backbone (`soprano_backbone`, the whole context
//! recomputed each token as [`NativeSoprano`](crate::NativeSoprano) runs it,
//! its past caches empty; symbolic in the tokens, out the logits and the
//! last hidden states) and the Vocos decoder (`soprano_decoder`, the
//! hidden states → PCM, symbolic in the frames). The program (`program/`)
//! is the runner's host logic: the prompt, the sampler (top-k, top-p, the
//! repetition penalty, its xorshift stream), the long-text chunking and the
//! offline crop.

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use rlx_core::export::graph::symbolic_in_lifted;
use rlx_core::export::{Checkpoint, ExportOptions, ExportReport, Package, program};
use rlx_ir::{Graph, Op};
use serde_json::json;

use crate::native::{EOS, HIDDEN, InferOpts, N_LAYERS, SAMPLE_RATE, TOKEN_SIZE, VOCAB};

/// The tokens' and the frames' symbols.
const TOKENS: u32 = 0;
const FRAMES: u32 = 1;
/// The longest context the runner reaches: 128 tokens.
const MAX_TOKENS: usize = 128;
/// The most frames a context makes.
const MAX_FRAMES: usize = 128;

/// `file`'s ONNX graph `name` at `len` as rlx IR, its params kept in
/// `params` (once: every size's are the same).
fn import(
    file: &Path,
    name: &str,
    len: usize,
    named: &[(&str, usize)],
    params: &RefCell<Option<HashMap<String, Vec<f32>>>>,
) -> Result<Graph> {
    let (g, weights) = rlx_tiny_tts::model::import_rlx_graph(file, name, len, named)?;
    params.borrow_mut().get_or_insert(weights);
    Ok(g)
}

/// The backbone at `seq` tokens and no past: its past caches empty
/// constants, its outputs the logits `[1, seq, vocab]` and the last hidden
/// states `[1, seq, hidden]` (its caches dropped).
fn backbone(
    file: &Path,
    seq: usize,
    params: &RefCell<Option<HashMap<String, Vec<f32>>>>,
) -> Result<Graph> {
    let named = [
        ("batch_size", 1),
        ("sequence_length", seq),
        ("past_sequence_length", 0),
        ("past_sequence_length + sequence_length", seq),
    ];
    let g = import(file, "soprano_backbone", seq, &named, params)?;
    let outs = g.outputs.clone();
    anyhow::ensure!(
        outs.len() > 1 + 2 * N_LAYERS,
        "the backbone has {} outputs",
        outs.len()
    );
    let mut g = rlx_ir::stateful::rebuild(&g, |out, _, node, _| match &node.op {
        Op::Input { name } if name.starts_with("past_key_values.") => Ok(Some(out.add_node(
            Op::Constant { data: Vec::new() },
            vec![],
            node.shape.clone(),
        ))),
        _ => Ok(None),
    })
    .map_err(anyhow::Error::msg)?;
    let keep = vec![g.outputs[0], g.outputs[1 + 2 * N_LAYERS]];
    g.node_mut(keep[0]).name = Some("logits".into());
    g.node_mut(keep[1]).name = Some("hidden".into());
    g.set_outputs(keep);
    g.name = "soprano_backbone".into();
    Ok(g)
}

/// The decoder at `t` frames: `hidden_states` `[1, 512, t]` → PCM
/// `[(t − 1) · 2048 + …]`, as the runner imports it.
fn decoder(
    file: &Path,
    t: usize,
    params: &RefCell<Option<HashMap<String, Vec<f32>>>>,
) -> Result<Graph> {
    let named = [
        ("s53", t),
        ("2048*s53", TOKEN_SIZE * t),
        ("2048*s53 - 2048", TOKEN_SIZE * t.saturating_sub(1)),
        ("4*s53 - 3", 4 * t - 3),
        ("8192*s53 - 6144", 8192 * t - 6144),
    ];
    let mut g = import(file, "soprano_decoder", t, &named, params)?;
    let out = g.outputs[0];
    g.node_mut(out).name = Some("audio".into());
    g.name = "soprano_decoder".into();
    Ok(g)
}

/// Export the Soprano directory `dir` (`tokenizer.json`, `onnx/`'s fp32
/// backbone and decoder) as the package `out`.
pub fn export_rlxp(dir: &Path, out: &Path, opts: &ExportOptions) -> Result<ExportReport> {
    let ck = Checkpoint::open(dir)?;
    let onnx = dir.join("onnx");
    let bb = onnx.join("soprano_backbone_kv_fp32.onnx");
    let dec = onnx.join("soprano_decoder_fp32.onnx");
    let mut pkg = Package::for_checkpoint(&ck, opts, "soprano", "text-to-speech", "rlx-soprano");

    let held = RefCell::new(None);
    let (bb_graph, tables) =
        symbolic_in_lifted(TOKENS, (5, 7, 9), MAX_TOKENS, |n| backbone(&bb, n, &held))?;
    let mut params = held.take().context("the backbone's params")?;
    params.extend(tables);
    let (dec_graph, tables) =
        symbolic_in_lifted(FRAMES, (5, 7, 9), MAX_FRAMES, |n| decoder(&dec, n, &held))?;
    params.extend(held.take().context("the decoder's params")?);
    params.extend(tables);
    pkg.function(bb_graph);
    pkg.function(dec_graph);
    pkg.dims = vec![("tokens".into(), TOKENS), ("frames".into(), FRAMES)];
    let dims = pkg.param_dims();
    let mut names: Vec<&String> = params.keys().collect();
    names.sort();
    for n in names {
        let d = dims
            .get(n)
            .cloned()
            .unwrap_or_else(|| vec![params[n].len()]);
        pkg.hold_f32(n.clone(), d, &params[n]);
    }

    let tokenizer = pkg.sidecar(
        "tokenizer.json",
        "application/json",
        std::fs::read(dir.join("tokenizer.json"))?,
    );
    let o = InferOpts::default();
    let module = program::build(&Path::new(env!("CARGO_MANIFEST_DIR")).join("program/Cargo.toml"))?;
    program::entry(
        &mut pkg,
        "speak",
        module,
        &["synthesize", "run"],
        "text-to-speech",
        json!({
            "tokenizer": tokenizer,
            "eos": EOS,
            "vocab": VOCAB,
            "hidden": HIDDEN,
            "token_size": TOKEN_SIZE,
            "max_tokens": MAX_TOKENS,
            "max_prompt_tokens": 80,
            "max_new_tokens": o.max_new_tokens,
            "temperature": o.temperature,
            "top_p": o.top_p,
            "top_k": o.top_k,
            "repetition_penalty": o.repetition_penalty,
            "seed": o.seed,
            "greedy": o.greedy,
            "sample_rate": SAMPLE_RATE,
        }),
    );
    pkg.metadata("sample_rate", json!(SAMPLE_RATE));
    pkg.write(out, opts)
}

/// Register this crate's exporter ([`rlx_core::export::register_exporter`]).
pub fn register() {
    use rlx_core::export::{Exporter, register_exporter};
    register_exporter(Exporter::new("soprano", "text-to-speech", export_rlxp));
}
