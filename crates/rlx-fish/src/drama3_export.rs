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

//! Export Fish Audio Drama 3 as an `.rlxp` package rlx runs alone: text in,
//! 44.1 kHz speech out, through the one exporter ([`rlx_core::export`]).
//!
//! Its graphs are [`crate::drama3::DramaModel`]'s: `embed` (rows: the token
//! embedding, plus an audio row's first codebook row, over `√2`); each
//! backbone layer's `layer{l}.prefill` (the prompt's rows, symbolic in them
//! and in the caches' rows) and `layer{l}.step` (one row: its key and value
//! written in at `pos`, or its DeltaNet window and state carried), each
//! giving the layer's output but for the routed experts, the post norm and
//! the router logits; `experts` (eight routed experts on the rows that
//! picked them, their FP8 weights and block scales inputs the program has
//! the host read from the checkpoint for that run, dequantized in the
//! graph); `final` (the final norm, the main token's logits); each tower
//! layer's `tower{i}.prefill` / `tower{i}.step` (its code embeddings
//! gathered in the graph, a residual layer's head logits); and the codec
//! S2-Pro's package has (`dequant`, `codec`). The checkpoint's 39 GB are
//! referenced by name, none read at export: the dense weights bind as the
//! functions compile, the 32 GB of experts are read eight at a time. The
//! program (`drama3_program/`) builds the prompt (the crate's
//! `drama3_text.rs` and `prompt.rs`), routes the experts, samples (its
//! `sampling.rs`) and decodes.

use std::collections::{BTreeSet, HashMap};
use std::path::Path;

use anyhow::{Context, Result};
use rlx_core::export::graph::{symbolic_in, symbolic_in_lifted_many};
use rlx_core::export::{ExportOptions, ExportReport, Package, program};
use rlx_ir::{DType, Graph, GraphExt, NodeId, Op, Shape};
use serde_json::json;

use crate::drama3::{CODEBOOK_EMBED, DramaModel, DramaOptions, LM, Mixer, Mode, RESIDUAL, SHARED};
use crate::export::{codec_functions, package_graph, rope_rows_of};

/// Experts one `experts` run applies.
pub(crate) const EXPERTS_PER_RUN: usize = 8;
const FP8_BLOCK: usize = 128;
/// The symbols: the codec's frames, the prompt's rows, the caches' rows,
/// an `experts` run's picks per expert.
const FRAMES: u32 = 0;
const PROMPT: u32 = 2;
const CAP: u32 = 3;
const PICKS: u32 = 4;
/// RoPE rows the tables hold: the longest prompt and a chunk's frames.
const MAX_ROWS: usize = crate::drama3::MAX_PROMPT_ROWS + 2048;
const EMBED: &str = "model.language_model.embed_tokens.weight";

fn f32s(d: &[usize]) -> Shape {
    Shape::new(d, DType::F32)
}

fn constant(g: &mut Graph, data: &[f32], dims: &[usize]) -> NodeId {
    g.add_node(
        Op::Constant {
            data: data.iter().flat_map(|v| v.to_le_bytes()).collect(),
        },
        vec![],
        f32s(dims),
    )
}

/// `w` (`[n, rows, cols]` FP8 e4m3 bytes) with its `[n, rows/128, cols/128]`
/// block scales `s`, as f32.
fn dequant_fp8(g: &mut Graph, w: NodeId, s: NodeId, n: usize, rows: usize, cols: usize) -> NodeId {
    let lut: Vec<f32> = (0..=255u8).map(crate::store::e4m3).collect();
    let lut = constant(g, &lut, &[256]);
    let w = g.reshape_(w, vec![(n * rows * cols) as i64]);
    let idx = g.add_node(
        Op::Cast { to: DType::I32 },
        vec![w],
        Shape::new(&[n * rows * cols], DType::I32),
    );
    let v = g.gather_(lut, idx, 0);
    let (rb, cb) = (rows / FP8_BLOCK, cols / FP8_BLOCK);
    let v = g.reshape_(
        v,
        vec![
            n as i64,
            rb as i64,
            FP8_BLOCK as i64,
            cb as i64,
            FP8_BLOCK as i64,
        ],
    );
    let s = g.reshape_(s, vec![n as i64, rb as i64, 1, cb as i64, 1]);
    let v = g.mul(v, s);
    g.reshape_(v, vec![n as i64, rows as i64, cols as i64])
}

/// `experts` at `rows` rows and `q` picks per expert: [`EXPERTS_PER_RUN`]
/// routed experts applied to the rows that picked them, their outputs
/// added to the routed sum so far.
///
/// Inputs: `x` `[1, rows, hidden]` (the post norm), `h` `[1, rows, hidden]`
/// (the layer's output before the routed experts), `acc` `[rows, hidden]`
/// (the routed sum so far), `src` `[E·q]` i32 (the row each pick reads),
/// `dest` `[E·q]` (the same, f32), `prob` `[E, q, 1]` (each pick's weight;
/// 0 pads), the experts' FP8 `gate` / `up` `[E, ffn, hidden]` and `down`
/// `[E, hidden, ffn]` with their block scales `*_s`. Outputs: `acc.next`
/// (`acc` plus the run's picks, each row's summed in order) and `out` (`h`
/// plus it).
fn experts(hidden: usize, ffn: usize, rows: usize, q: usize) -> Graph {
    let e = EXPERTS_PER_RUN;
    let mut g = Graph::new("experts");
    let x = g.input("x", f32s(&[1, rows, hidden]));
    let h = g.input("h", f32s(&[1, rows, hidden]));
    let acc = g.input("acc", f32s(&[rows, hidden]));
    let src = g.input("src", Shape::new(&[e * q], DType::I32));
    let dest = g.input("dest", f32s(&[e * q]));
    let prob = g.input("prob", f32s(&[e, q, 1]));
    let w = |g: &mut Graph, name: &str, r: usize, c: usize| {
        let b = g.input(name, Shape::new(&[e, r, c], DType::U8));
        let s = g.input(
            format!("{name}_s"),
            f32s(&[e, r / FP8_BLOCK, c / FP8_BLOCK]),
        );
        dequant_fp8(g, b, s, e, r, c)
    };
    let wg = w(&mut g, "gate", ffn, hidden);
    let wu = w(&mut g, "up", ffn, hidden);
    let wd = w(&mut g, "down", hidden, ffn);
    let x2 = g.reshape_(x, vec![rows as i64, hidden as i64]);
    let xs = g.gather_(x2, src, 0);
    let xs = g.reshape_(xs, vec![e as i64, q as i64, hidden as i64]);
    let wg = g.transpose_(wg, vec![0, 2, 1]);
    let wu = g.transpose_(wu, vec![0, 2, 1]);
    let wd = g.transpose_(wd, vec![0, 2, 1]);
    let gate = g.mm(xs, wg);
    let up = g.mm(xs, wu);
    let a = g.silu(gate);
    let a = g.mul(a, up);
    let y = g.mm(a, wd);
    let y = g.mul(y, prob);
    let y = g.reshape_(y, vec![(e * q) as i64, hidden as i64]);
    let picked = g.add_node(
        Op::ScatterAdd { axis: 0 },
        vec![y, dest],
        f32s(&[rows, hidden]),
    );
    let sum = g.add(acc, picked);
    g.node_mut(sum).name = Some("acc.next".into());
    let s3 = g.reshape_(sum, vec![1, rows as i64, hidden as i64]);
    let out = g.add(h, s3);
    g.node_mut(out).name = Some("out".into());
    g.set_outputs(vec![sum, out]);
    g
}

fn zero(g: &mut Graph, dims: &[usize]) -> NodeId {
    let z = constant(g, &[0.0], &vec![1; dims.len()]);
    g.add_node(
        Op::Expand {
            target_shape: dims.iter().map(|&d| d as i64).collect(),
        },
        vec![z],
        f32s(dims),
    )
}

/// `rows` (`[1, n, w]`) written at row `at` of `cache` (`[1, cap, w]`).
fn write_at(g: &mut Graph, cache: NodeId, rows: NodeId, at: NodeId, name: &str) -> NodeId {
    let shape = g.node(cache).shape.clone();
    let full = g.add_node(
        Op::DynamicUpdateSlice { axis: 1 },
        vec![cache, rows, at],
        shape,
    );
    g.node_mut(full).name = Some(name.into());
    full
}

/// `rows` (`[1, n, w]`) at the start of zero caches of `cap` rows.
fn into_cache(g: &mut Graph, rows: NodeId, cap: usize, name: &str) -> NodeId {
    let w = g.node(rows).shape.dims()[2].unwrap_static();
    let z = zero(g, &[1, cap, w]);
    let origin = constant(g, &[0.0], &[1]);
    write_at(g, z, rows, origin, name)
}

fn name(g: &mut Graph, id: NodeId, n: &str) {
    g.node_mut(id).name = Some(n.into());
}

/// `layer{l}.prefill` at `t` rows and `cap` cache rows.
fn layer_prefill(
    m: &DramaModel,
    l: usize,
    t: usize,
    cap: usize,
    held: &mut HashMap<String, Vec<f32>>,
    refs: &mut BTreeSet<String>,
) -> Result<Graph> {
    let (b, outs) = m.layer_builder(l, Mode::Prefill(t))?;
    let mut g = package_graph(b, outs, held, refs)?;
    let kind = m.cfg.layers[l];
    if kind == Mixer::Full {
        g = rope_rows_of(&g, "backbone.rope", None, MAX_ROWS)?;
    }
    let o = g.outputs.clone();
    for (i, n) in ["h", "xn", "router"].iter().enumerate() {
        name(&mut g, o[i], n);
    }
    let (a, b) = match kind {
        Mixer::Full => (
            into_cache(&mut g, o[3], cap, "past_k"),
            into_cache(&mut g, o[4], cap, "past_v"),
        ),
        Mixer::Linear => {
            // The window: the prompt's last `kernel - 1` conv inputs.
            let hist = m.cfg.conv_kernel - 1;
            let w = g.narrow_(o[3], 1, t - hist, hist);
            name(&mut g, w, "conv");
            name(&mut g, o[4], "ssm");
            (w, o[4])
        }
    };
    g.set_outputs(vec![o[0], o[1], o[2], a, b]);
    g.name = format!("layer{l}.prefill");
    Ok(g)
}

/// `layer{l}.step` at `cap` cache rows.
fn layer_step(
    m: &DramaModel,
    l: usize,
    cap: usize,
    held: &mut HashMap<String, Vec<f32>>,
    refs: &mut BTreeSet<String>,
) -> Result<Graph> {
    let (b, outs) = m.layer_builder(l, Mode::Decode(cap))?;
    let mut g = package_graph(b, outs, held, refs)?;
    let kind = m.cfg.layers[l];
    if kind == Mixer::Full {
        g = rope_rows_of(&g, "backbone.rope", Some("pos"), MAX_ROWS)?;
    }
    let o = g.outputs.clone();
    for (i, n) in ["h", "xn", "router"].iter().enumerate() {
        name(&mut g, o[i], n);
    }
    let (a, b) = match kind {
        Mixer::Full => {
            let pos = g.input_id("pos").context("no pos")?;
            let pk = g.input_id("past_k").context("no past_k")?;
            let pv = g.input_id("past_v").context("no past_v")?;
            (
                write_at(&mut g, pk, o[3], pos, "past_k.next"),
                write_at(&mut g, pv, o[4], pos, "past_v.next"),
            )
        }
        Mixer::Linear => {
            let hist = m.cfg.conv_kernel - 1;
            let st = g.input_id("conv_state").context("no conv_state")?;
            let kept = g.narrow_(st, 1, 1, hist - 1);
            let w = g.concat_(vec![kept, o[3]], 1);
            name(&mut g, w, "conv.next");
            name(&mut g, o[4], "ssm.next");
            (w, o[4])
        }
    };
    g.set_outputs(vec![o[0], o[1], o[2], a, b]);
    g.name = format!("layer{l}.step");
    Ok(g)
}

/// `final` at `t` rows: `x` `[1, t, hidden]` → `hidden` (each row's final
/// norm), `last` (the last row's) and `logits` `[1, 1 + band]` (the main
/// token's over `[<|im_end|>, band…]`).
fn final_fn(m: &DramaModel, t: usize) -> Graph {
    let d = m.cfg.hidden;
    let n = m.head.len() / d;
    let mut g = Graph::new("final");
    let x = g.input("x", f32s(&[1, t, d]));
    let sq = g.mul(x, x);
    let ms = g.mean(sq, vec![2], true);
    let eps = constant(&mut g, &[m.cfg.eps], &[1]);
    let ms = g.add(ms, eps);
    let rms = g.sqrt(ms);
    let inv = g.recip(rms);
    let xn = g.mul(x, inv);
    let w = g.param("drama3.final_norm", f32s(&[d]));
    let hidden = g.mul(xn, w);
    name(&mut g, hidden, "hidden");
    let last = g.narrow_(hidden, 1, t - 1, 1);
    name(&mut g, last, "last");
    let row = g.reshape_(last, vec![1, d as i64]);
    let head = g.param("drama3.head", f32s(&[d, n]));
    let logits = g.mm(row, head);
    name(&mut g, logits, "logits");
    g.set_outputs(vec![hidden, last, logits]);
    g
}

/// `embed`: `tokens` `[t]` i32, `codes` `[t, k]` i32 (an audio row's
/// `i · vocab + code_i`, `i` in `1..=k`), `audio` `[t, 1]` (1 for an audio
/// row) → `embeds` `[1, t, hidden]`.
fn embed(m: &DramaModel, vocab: usize, cb_rows: usize, t: usize) -> Graph {
    let d = m.cfg.hidden;
    let k = m.cfg.tower.vq_backbone_codebooks;
    let mut g = Graph::new("embed");
    let tokens = g.input("tokens", Shape::new(&[t], DType::I32));
    let codes = g.input("codes", Shape::new(&[t, k], DType::I32));
    let audio = g.input("audio", f32s(&[t, 1]));
    let table = g.param(EMBED, f32s(&[vocab, d]));
    let tok = g.gather_(table, tokens, 0);
    let cb = g.param(CODEBOOK_EMBED, f32s(&[cb_rows, d]));
    let flat = g.reshape_(codes, vec![(t * k) as i64]);
    let rows = g.gather_(cb, flat, 0);
    let rows = g.reshape_(rows, vec![t as i64, k as i64, d as i64]);
    let sum = g.sum(rows, vec![1], false);
    let sum = g.mul(sum, audio);
    let x = g.add(tok, sum);
    // An audio row over √(k + 1), a text row as it is.
    let s = 1.0 / ((k + 1) as f32).sqrt();
    let less = constant(&mut g, &[s - 1.0], &[1]);
    let one = constant(&mut g, &[1.0], &[1]);
    let gain = g.mul(audio, less);
    let gain = g.add(gain, one);
    let x = g.mul(x, gain);
    let x = g.reshape_(x, vec![1, t as i64, d as i64]);
    name(&mut g, x, "embeds");
    g.set_outputs(vec![x]);
    g
}

/// Tower layer `i`'s input `e` (`[1, s, dim]`) its code embeddings:
/// `code` `[s]` i32 (the row's label code), `on` `[s, 1]` (1 when the
/// label is audio); the first layer adds `codes` `[s, n]` i32 (the row's
/// own codes past the backbone's, `i · vocab + code_i`) when `cur_on`.
fn tower_codes(
    m: &DramaModel,
    g: &Graph,
    i: usize,
    s: usize,
    cb_rows: usize,
    refs: &mut BTreeSet<String>,
) -> Result<Graph> {
    let t = &m.cfg.tower;
    let d = t.dim;
    let ns = t.shared_layers;
    let key = if i < ns {
        format!("{SHARED}{i}.semantic_embedding.weight")
    } else {
        format!("{RESIDUAL}{}.codebook_embedding.weight", i - ns)
    };
    refs.insert(key.clone());
    let first = 1 + t.vq_backbone_codebooks;
    let extra = t.num_codebooks - first;
    if i == 0 && extra > 0 {
        refs.insert(CODEBOOK_EMBED.into());
    }
    rlx_ir::stateful::rebuild(g, |out, _, node, _| {
        let Op::Input { name } = &node.op else {
            return Ok(None);
        };
        if name != "e" {
            return Ok(None);
        }
        let code = out.input("code", Shape::new(&[s], DType::I32));
        let on = out.input("on", f32s(&[s, 1]));
        let table = out.param(key.clone(), f32s(&[t.vocab, d]));
        let rows = out.gather_(table, code, 0);
        let mut e = out.mul(rows, on);
        if i == 0 && extra > 0 {
            let codes = out.input("codes", Shape::new(&[s, extra], DType::I32));
            let cur_on = out.input("cur_on", f32s(&[s, 1]));
            let cb = out.param(CODEBOOK_EMBED, f32s(&[cb_rows, d]));
            let flat = out.reshape_(codes, vec![(s * extra) as i64]);
            let r = out.gather_(cb, flat, 0);
            let r = out.reshape_(r, vec![s as i64, extra as i64, d as i64]);
            let sum = out.sum(r, vec![1], false);
            let sum = out.mul(sum, cur_on);
            e = out.add(e, sum);
        }
        Ok(Some(out.reshape_(e, vec![1, s as i64, d as i64])))
    })
    .map_err(anyhow::Error::msg)
}

/// `tower{i}.prefill` at `s` rows (a prompt's rows but its last, each
/// labelled with the next row's frame) and `cap` cache rows.
fn tower_prefill(
    m: &DramaModel,
    i: usize,
    s: usize,
    cap: usize,
    cb_rows: usize,
    held: &mut HashMap<String, Vec<f32>>,
    refs: &mut BTreeSet<String>,
) -> Result<Graph> {
    let (b, outs) = m.tower_builder(i, Mode::Prefill(s));
    let mut g = package_graph(b, outs, held, refs)?;
    g = rope_rows_of(&g, "tower.rope", None, MAX_ROWS)?;
    g = tower_codes(m, &g, i, s, cb_rows, refs)?;
    let o = g.outputs.clone();
    name(&mut g, o[0], "out");
    let k = into_cache(&mut g, o[1], cap, "past_k");
    let v = into_cache(&mut g, o[2], cap, "past_v");
    g.set_outputs(vec![o[0], k, v]);
    g.name = format!("tower{i}.prefill");
    Ok(g)
}

/// `rows` at `t` rows: `x` `[1, t, dim]` → `rows`, its first `t - 1` (the
/// tower's prompt).
fn rows_fn(d: usize, t: usize) -> Graph {
    let mut g = Graph::new("rows");
    let x = g.input("x", f32s(&[1, t, d]));
    let r = g.narrow_(x, 1, 0, t - 1);
    name(&mut g, r, "rows");
    g.set_outputs(vec![r]);
    g
}

/// `tower{i}.step` at `cap` cache rows: `out`, a residual layer's head
/// `logits`, its key and value written in at `pos`.
fn tower_step(
    m: &DramaModel,
    i: usize,
    cap: usize,
    cb_rows: usize,
    held: &mut HashMap<String, Vec<f32>>,
    refs: &mut BTreeSet<String>,
) -> Result<Graph> {
    let (b, outs) = m.tower_builder(i, Mode::Decode(cap));
    let mut g = package_graph(b, outs, held, refs)?;
    g = rope_rows_of(&g, "tower.rope", Some("pos"), MAX_ROWS)?;
    g = tower_codes(m, &g, i, 1, cb_rows, refs)?;
    let mut o = g.outputs.clone();
    name(&mut g, o[0], "out");
    let kv = o.len() - 2;
    if kv == 2 {
        let logits = g.reshape_(o[1], vec![1, m.cfg.tower.vocab as i64]);
        name(&mut g, logits, "logits");
        o[1] = logits;
    }
    let pos = g.input_id("pos").context("no pos")?;
    let pk = g.input_id("past_k").context("no past_k")?;
    let pv = g.input_id("past_v").context("no past_v")?;
    o[kv] = write_at(&mut g, pk, o[kv], pos, "past_k.next");
    o[kv + 1] = write_at(&mut g, pv, o[kv + 1], pos, "past_v.next");
    g.set_outputs(o);
    g.name = format!("tower{i}.step");
    Ok(g)
}

/// Export the Drama 3 directory `dir` (`config.json`, `model*.safetensors`
/// and their index, `tokenizer.json`, `codec.pth`) as the package `out`.
pub fn export_rlxp(dir: &Path, out: &Path, opts: &ExportOptions) -> Result<ExportReport> {
    // Expanded K/V heads: every backend's attention takes them.
    rlx_ir::env::set("RLX_FISH_GQA_EXPAND", "1");
    let m = DramaModel::open(
        dir,
        rlx_runtime::Device::Cpu,
        DramaOptions {
            weight_storage: Some(rlx_flow::storage::WeightStorage::F32),
            ..DramaOptions::default()
        },
    )?;
    let c = m.cfg.clone();
    let mut pkg = Package::new(
        "fish-drama-3",
        &opts.version,
        "fish-drama3",
        "text-to-speech",
        "rlx-fish",
    );
    pkg.model["license"] = json!("fish-audio-research");
    let mut held: HashMap<String, Vec<f32>> = HashMap::new();
    let mut refs: BTreeSet<String> = BTreeSet::new();
    let shape = |k: &str| -> Result<Vec<usize>> { Ok(m.store.info(k)?.shape.clone()) };
    let vocab = shape(EMBED)?[0];
    let cb_rows = shape(CODEBOOK_EMBED)?[0];
    let sizes = [(PROMPT, vec![5, 7, 3]), (CAP, vec![37, 53, 20])];
    let scratch = || (HashMap::new(), BTreeSet::new());

    // The backbone.
    for l in 0..c.layers.len() {
        // A DeltaNet's prompt graph has no cache rows.
        let p = if c.layers[l] == Mixer::Full {
            let (p, lifted) = symbolic_in_lifted_many(&sizes, &[MAX_ROWS, MAX_ROWS], |n| {
                let (mut h, mut r) = scratch();
                layer_prefill(&m, l, n[0], n[1], &mut h, &mut r)
            })?;
            anyhow::ensure!(lifted.is_empty(), "layer {l}'s prompt graph lifted tables");
            p
        } else {
            symbolic_in(PROMPT, (5, 7, 3), |t| {
                let (mut h, mut r) = scratch();
                layer_prefill(&m, l, t, 20, &mut h, &mut r)
            })?
        };
        layer_prefill(&m, l, 3, 20, &mut held, &mut refs)?;
        pkg.function(p);
        pkg.function(symbolic_in(CAP, (37, 53, 20), |cap| {
            let (mut h, mut r) = scratch();
            layer_step(&m, l, cap, &mut h, &mut r)
        })?);
        layer_step(&m, l, 20, &mut held, &mut refs)?;
    }
    let (eg, lifted) = symbolic_in_lifted_many(
        &[(PROMPT, vec![5, 7, 3]), (PICKS, vec![2, 3, 5])],
        &[MAX_ROWS, MAX_ROWS],
        |n| Ok(experts(c.hidden, c.moe_ffn, n[0], n[1])),
    )?;
    anyhow::ensure!(lifted.is_empty(), "the experts graph lifted tables");
    pkg.function(eg);
    pkg.function(symbolic_in(PROMPT, (5, 7, 3), |t| Ok(final_fn(&m, t)))?);
    pkg.function(symbolic_in(PROMPT, (5, 7, 3), |t| {
        Ok(rows_fn(c.hidden, t))
    })?);
    pkg.function(symbolic_in(PROMPT, (5, 7, 3), |t| {
        Ok(embed(&m, vocab, cb_rows, t))
    })?);
    refs.extend([EMBED.to_string(), CODEBOOK_EMBED.to_string()]);
    held.insert("drama3.final_norm".into(), m.final_norm.clone());
    let n_head = m.head.len() / c.hidden;
    held.insert(
        "drama3.head".into(),
        crate::nn::transpose(&m.head, n_head, c.hidden),
    );

    // The tower.
    let n_tower = c.tower.shared_layers + c.tower.num_codebooks - 1;
    for i in 0..n_tower {
        let (p, lifted) = symbolic_in_lifted_many(&sizes, &[MAX_ROWS, MAX_ROWS], |n| {
            let (mut h, mut r) = scratch();
            tower_prefill(&m, i, n[0], n[1], cb_rows, &mut h, &mut r)
        })?;
        anyhow::ensure!(
            lifted.is_empty(),
            "tower layer {i}'s prompt graph lifted tables"
        );
        tower_prefill(&m, i, 3, 20, cb_rows, &mut held, &mut refs)?;
        pkg.function(p);
        pkg.function(symbolic_in(CAP, (37, 53, 20), |cap| {
            let (mut h, mut r) = scratch();
            tower_step(&m, i, cap, cb_rows, &mut h, &mut r)
        })?);
        tower_step(&m, i, 20, cb_rows, &mut held, &mut refs)?;
    }
    let positions: Vec<usize> = (0..MAX_ROWS).collect();
    for (table, (cos, sin)) in [
        ("backbone.rope", m.rope_backbone(&positions)),
        ("tower.rope", m.rope_tower(&positions)),
    ] {
        held.insert(format!("{table}.cos"), cos);
        held.insert(format!("{table}.sin"), sin);
    }

    codec_functions(&mut pkg, dir, &mut held)?;
    pkg.dims = vec![
        ("frames".into(), FRAMES),
        ("prompt".into(), PROMPT),
        ("cap".into(), CAP),
        ("picks".into(), PICKS),
    ];
    let pdims = pkg.param_dims();
    let mut names: Vec<&String> = held.keys().collect();
    names.sort();
    for n in names {
        let d = pdims.get(n).cloned().unwrap_or_else(|| vec![held[n].len()]);
        pkg.hold_f32(n.clone(), d, &held[n]);
    }
    drop(held);

    // The checkpoint, by name: every shard a source.
    let index: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("model.safetensors.index.json"))?)?;
    let map = index["weight_map"]
        .as_object()
        .context("the index has no weight_map")?;
    let mut shards: Vec<&str> = map.values().filter_map(|v| v.as_str()).collect();
    shards.sort();
    shards.dedup();
    for shard in shards {
        let count = refs
            .iter()
            .filter(|k| map.get(k.as_str()).and_then(|v| v.as_str()) == Some(shard))
            .count();
        pkg.reference_named("safetensors", &dir.join(shard), out, count)?;
    }

    pkg.sidecar(
        "tokenizer.json",
        "application/json",
        std::fs::read(dir.join("tokenizer.json"))?,
    );
    let module =
        program::build(&Path::new(env!("CARGO_MANIFEST_DIR")).join("drama3_program/Cargo.toml"))
            .context("build Drama 3's program")?;
    let kinds: Vec<&str> = c
        .layers
        .iter()
        .map(|k| match k {
            Mixer::Full => "full",
            Mixer::Linear => "linear",
        })
        .collect();
    let t = &c.tower;
    program::entry(
        &mut pkg,
        "speak",
        module,
        &["synthesize", "run"],
        "text-to-speech",
        json!({
            "tokenizer": "sidecar:tokenizer.json",
            "layers": kinds,
            "hidden": c.hidden,
            "kv": c.kv_width(),
            "conv_kernel": c.conv_kernel,
            "conv_dim": c.conv_dim(),
            "lin_v_heads": c.lin_v_heads,
            "lin_k_dim": c.lin_k_dim,
            "lin_v_dim": c.lin_v_dim,
            "num_experts": c.num_experts,
            "top_k": c.top_k,
            "norm_topk": c.norm_topk,
            "moe_ffn": c.moe_ffn,
            "experts": format!("{LM}layers."),
            "tower_layers": n_tower,
            "tower_shared": t.shared_layers,
            "tower_dim": t.dim,
            "tower_kv": t.n_kv * t.head_dim,
            "tower_vocab": t.vocab,
            "num_codebooks": t.num_codebooks,
            "backbone_codebooks": t.vq_backbone_codebooks,
            "semantic_begin": c.semantic_begin,
            "semantic_end": c.semantic_end,
            "eos": c.eos,
            "max_rows": MAX_ROWS,
            "max_prompt_rows": crate::drama3::MAX_PROMPT_ROWS,
            "max_new_tokens": crate::drama3::DEFAULT_MAX_NEW_TOKENS,
            "temperature": 0.7,
            "top_p": 0.7,
            "top_k_sample": 30,
            "seed": 0,
            "semantic_size": crate::codec::SEMANTIC_SIZE,
            "residual_size": crate::codec::RESIDUAL_SIZE,
            "sample_rate": crate::codec::SAMPLE_RATE,
        }),
    );
    pkg.metadata("sample_rate", json!(crate::codec::SAMPLE_RATE));
    pkg.write(out, opts)
}

/// Register this crate's Drama 3 exporter ([`rlx_core::export::register_exporter`]).
pub fn register() {
    use rlx_core::export::{Exporter, register_exporter};
    register_exporter(
        Exporter::new("fish-drama3", "text-to-speech", export_rlxp).claims(&["fish_omni"]),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    /// One layer's routed experts as the graph applies them, against the
    /// runner's [`crate::moe::ExpertBank`] on the checkpoint.
    #[test]
    fn the_experts_graph_applies_the_routed_experts() {
        let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../weights/tts/fish-drama-3");
        if !dir.join("config.json").is_file() {
            eprintln!("skip: no Drama 3 checkpoint");
            return;
        }
        let cfg = crate::drama3::DramaConfig::from_dir(&dir).unwrap();
        let store = Arc::new(crate::store::ShardStore::open(&dir).unwrap());
        let (hd, ffn) = (cfg.hidden, cfg.moe_ffn);
        let bank = crate::moe::ExpertBank::new(
            store.clone(),
            "model.language_model.layers.",
            hd,
            ffn,
            cfg.top_k,
            cfg.norm_topk,
            0,
        );
        let rows = 3;
        let mut seed = 7u32;
        let mut rnd = || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed >> 8) as f32 / (1u32 << 24) as f32 - 0.5
        };
        let x: Vec<f32> = (0..rows * hd).map(|_| rnd()).collect();
        let router: Vec<f32> = (0..rows * cfg.num_experts).map(|_| 4.0 * rnd()).collect();
        let layer = 5;
        let want = bank.forward(layer, &x, &router, rows).unwrap();

        // The picks, by expert; runs of EXPERTS_PER_RUN experts.
        let mut by: std::collections::BTreeMap<usize, Vec<(usize, f32)>> = Default::default();
        for r in 0..rows {
            for (e, w) in bank.route(&router[r * cfg.num_experts..(r + 1) * cfg.num_experts]) {
                by.entry(e).or_default().push((r, w));
            }
        }
        let chosen: Vec<(usize, Vec<(usize, f32)>)> = by.into_iter().collect();
        let mut acc = vec![0f32; rows * hd];
        let mut out = Vec::new();
        let h = vec![0.25f32; rows * hd];
        for run in chosen.chunks(EXPERTS_PER_RUN) {
            let q = run.iter().map(|(_, p)| p.len()).max().unwrap();
            let g = experts(hd, ffn, rows, q);
            let mut c = rlx_runtime::Session::new(rlx_runtime::Device::Cpu).compile(g);
            let mut src = vec![0i32; EXPERTS_PER_RUN * q];
            let mut dest = vec![0f32; EXPERTS_PER_RUN * q];
            let mut prob = vec![0f32; EXPERTS_PER_RUN * q];
            let mut bytes: std::collections::HashMap<&str, Vec<u8>> = Default::default();
            for slot in 0..EXPERTS_PER_RUN {
                // An unused slot reads expert 0 at weight 0.
                let (ex, picks) = run.get(slot).map_or((0, &[][..]), |(e, p)| (*e, &p[..]));
                for (i, &(r, w)) in picks.iter().enumerate() {
                    src[slot * q + i] = r as i32;
                    dest[slot * q + i] = r as f32;
                    prob[slot * q + i] = w;
                }
                for (name, scales, proj) in [
                    ("gate", "gate_s", "gate_proj"),
                    ("up", "up_s", "up_proj"),
                    ("down", "down_s", "down_proj"),
                ] {
                    let key = format!(
                        "model.language_model.layers.{layer}.mlp.experts.{ex}.{proj}.weight"
                    );
                    bytes
                        .entry(name)
                        .or_default()
                        .extend(store.bytes(&key).unwrap());
                    bytes
                        .entry(scales)
                        .or_default()
                        .extend(store.bytes(&format!("{key}_scale_inv")).unwrap());
                }
            }
            let le = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
            let src_b: Vec<u8> = src.iter().flat_map(|x| x.to_le_bytes()).collect();
            let feeds: Vec<(&str, Vec<u8>, DType)> = vec![
                ("x", le(&x), DType::F32),
                ("h", le(&h), DType::F32),
                ("acc", le(&acc), DType::F32),
                ("src", src_b, DType::I32),
                ("dest", le(&dest), DType::F32),
                ("prob", le(&prob), DType::F32),
                ("gate", bytes["gate"].clone(), DType::U8),
                ("up", bytes["up"].clone(), DType::U8),
                ("down", bytes["down"].clone(), DType::U8),
                ("gate_s", bytes["gate_s"].clone(), DType::F32),
                ("up_s", bytes["up_s"].clone(), DType::F32),
                ("down_s", bytes["down_s"].clone(), DType::F32),
            ];
            let refs: Vec<(&str, &[u8], DType)> = feeds
                .iter()
                .map(|(n, d, t)| (*n, d.as_slice(), *t))
                .collect();
            let outs = c.run_typed(&refs);
            let f = |b: &[u8]| -> Vec<f32> {
                b.chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect()
            };
            acc = f(&outs[0].0);
            out = f(&outs[1].0);
        }
        let diff = acc
            .iter()
            .zip(&want)
            .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        let scale = want.iter().fold(0f32, |m, v| m.max(v.abs()));
        eprintln!(
            "{} runs; max |d| {diff} of {scale}",
            chosen.len().div_ceil(EXPERTS_PER_RUN)
        );
        assert!(diff <= 1e-4 * scale.max(1.0), "max |d| {diff}");
        let d2 = out
            .iter()
            .zip(&acc)
            .fold(0f32, |m, (o, a)| m.max((o - a - 0.25).abs()));
        assert!(d2 < 1e-5);
    }
}
