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

//! Export Sesame CSM-1B as an `.rlxp` package rlx runs alone: text in,
//! 24 kHz speech out, through the one exporter ([`rlx_core::export`]).
//!
//! Its graphs, as `generate_codes` computes: `frame_embed` (each frame's
//! codebook and text rows, masked, summed), the backbone (rlx-llama32's
//! Llama over embedded frames, its last hidden states; symbolic in the
//! frames), `c0_head`, `depth_project` (a backbone state into the depth
//! decoder), `depth_embed` (a codebook's code, projected), the depth decoder
//! (over its 33 positions) and `codebook_head`; then Mimi's `lookup` and
//! `decode` ([`rlx_mimi::export::mimi_functions`]). The weights are the
//! checkpoint's, referenced. The program (`program/`) builds the prompt's
//! frames, samples each frame's 32 codes (top-k, the runner's `fastrand`
//! stream), feeds them back, and decodes the frames with Mimi.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use rlx_core::export::graph::{broadcast_params, symbolic_in};
use rlx_core::export::{Checkpoint, ExportOptions, ExportReport, Package, program};
use rlx_core::weight_map::WeightMap;
use rlx_ir::{DType, Graph, GraphExt, Op, Shape};
use rlx_llama32::{Llama32Config, Llama32Flow};
use serde_json::json;

use crate::config::SesameConfig;

/// The frames' symbol (the backbone's and `frame_embed`'s), and the
/// longest context the backbone's RoPE tables reach.
const FRAMES: u32 = 0;
const MAX_FRAMES: usize = 2048;
/// The depth decoder's positions: the backbone state and 32 codes.
const DEPTH_LEN: usize = 33;

fn f32s(d: &[usize]) -> Shape {
    Shape::new(d, DType::F32)
}

/// A Llama config from CSM's fields (rlx-llama32's HF reader).
fn llama_config(v: serde_json::Value) -> Result<Llama32Config> {
    rlx_core::config::from_hf_json(&v.to_string())
}

/// `tokens` `[n, 33]` i32 and `mask` `[n, 33]` → each frame's rows summed:
/// codebook `k`'s row of the audio table (`k · vocab + token`) where its
/// mask is set, and the text table's row of the last slot.
fn frame_embed(cfg: &SesameConfig, n: usize) -> Graph {
    let (k, h, vocab) = (cfg.num_codebooks, cfg.hidden_size, cfg.vocab_size);
    let mut g = Graph::new("frame_embed");
    let tokens = g.input("tokens", Shape::new(&[n, k + 1], DType::I32));
    let mask = g.input("mask", f32s(&[n, k + 1]));
    let tok = g.add_node(Op::Cast { to: DType::F32 }, vec![tokens], f32s(&[n, k + 1]));
    let audio = g.narrow_(tok, 1, 0, k);
    let offsets = g.param("frame_embed.offsets", f32s(&[1, k]));
    let rows = g.add(audio, offsets);
    let rows = g.reshape_(rows, vec![(n * k) as i64]);
    let rows = g.add_node(
        Op::Cast { to: DType::I32 },
        vec![rows],
        Shape::new(&[n * k], DType::I32),
    );
    let table = g.param(
        "depth_decoder.model.embed_tokens.weight",
        f32s(&[k * vocab, h]),
    );
    let picked = g.gather(table, rows, 0, f32s(&[n * k, h]));
    let picked = g.reshape_(picked, vec![n as i64, k as i64, h as i64]);
    let m_audio = g.narrow_(mask, 1, 0, k);
    let m_audio = g.reshape_(m_audio, vec![n as i64, k as i64, 1]);
    let picked = g.mul(picked, m_audio);
    let audio_sum = g.sum(picked, vec![1], false);
    let text_id = g.narrow_(tokens, 1, k, 1);
    let text_id = g.reshape_(text_id, vec![n as i64]);
    let text_table = g.param("embed_text_tokens.weight", f32s(&[cfg.text_vocab_size, h]));
    let text = g.gather(text_table, text_id, 0, f32s(&[n, h]));
    let m_text = g.narrow_(mask, 1, k, 1);
    let text = g.mul(text, m_text);
    let sum = g.add(audio_sum, text);
    g.node_mut(sum).name = Some("embeds".into());
    g.set_outputs(vec![sum]);
    g
}

/// `x` `[d_in]` through `weight` (`[d_out, d_in]`, `name`) → `[d_out]`.
fn linear_graph(fname: &str, input: &str, weight: &str, d_in: usize, d_out: usize) -> Graph {
    let mut g = Graph::new(fname);
    let x = g.input(input, f32s(&[d_in]));
    let x = g.reshape_(x, vec![1, d_in as i64]);
    let w = g.param(weight, f32s(&[d_out, d_in]));
    let wt = g.transpose_(w, vec![1, 0]);
    let y = g.mm(x, wt);
    let y = g.reshape_(y, vec![d_out as i64]);
    g.set_outputs(vec![y]);
    g
}

/// `codebook` `[1]` and `code` `[1]` i32 → codebook's row of the audio
/// table, through the depth projection (`[depth_hidden]`).
fn depth_embed(cfg: &SesameConfig, dh: usize) -> Graph {
    let (h, vocab, k) = (cfg.hidden_size, cfg.vocab_size, cfg.num_codebooks);
    let mut g = Graph::new("depth_embed");
    let cb = g.input("codebook", Shape::new(&[1], DType::I32));
    let code = g.input("code", Shape::new(&[1], DType::I32));
    let cbf = g.add_node(Op::Cast { to: DType::F32 }, vec![cb], f32s(&[1]));
    let codef = g.add_node(Op::Cast { to: DType::F32 }, vec![code], f32s(&[1]));
    let v = g.constant(vocab as f64, DType::F32);
    let base = g.mul(cbf, v);
    let row = g.add(base, codef);
    let row = g.add_node(
        Op::Cast { to: DType::I32 },
        vec![row],
        Shape::new(&[1], DType::I32),
    );
    let table = g.param(
        "depth_decoder.model.embed_tokens.weight",
        f32s(&[k * vocab, h]),
    );
    let e = g.gather(table, row, 0, f32s(&[1, h]));
    let w = g.param(
        "depth_decoder.model.inputs_embeds_projector.weight",
        f32s(&[dh, h]),
    );
    let wt = g.transpose_(w, vec![1, 0]);
    let y = g.mm(e, wt);
    let y = g.reshape_(y, vec![dh as i64]);
    g.set_outputs(vec![y]);
    g
}

/// `h` `[depth_hidden]` and `codebook` `[1]` (1 … 31) → that codebook's
/// logits: `h @ codebooks_head[codebook − 1]`.
fn codebook_head(cfg: &SesameConfig, dh: usize) -> Graph {
    let (vocab, k) = (cfg.vocab_size, cfg.num_codebooks);
    let mut g = Graph::new("codebook_head");
    let x = g.input("h", f32s(&[dh]));
    let cb = g.input("codebook", Shape::new(&[1], DType::I32));
    let x = g.reshape_(x, vec![1, dh as i64]);
    let heads = g.param(
        "depth_decoder.codebooks_head.weight",
        f32s(&[k - 1, dh, vocab]),
    );
    let one = g.add_node(
        Op::Constant {
            data: 1i32.to_le_bytes().to_vec(),
        },
        vec![],
        Shape::new(&[1], DType::I32),
    );
    let idx = g.sub(cb, one);
    let w = g.gather(heads, idx, 0, f32s(&[1, dh, vocab]));
    let w = g.reshape_(w, vec![dh as i64, vocab as i64]);
    let y = g.mm(x, w);
    let y = g.reshape_(y, vec![vocab as i64]);
    g.set_outputs(vec![y]);
    g
}

/// A Llama of `prefix`'s weights (renamed `model.*` for rlx-llama32's
/// flow, the graph's params named back) over `inputs_embeds` `[1, seq, h]`
/// → its final-normed hidden states.
fn llama_graph(
    cfg: &Llama32Config,
    weights: &HashMap<String, (Vec<f32>, Vec<usize>)>,
    prefix: &str,
    name: &str,
    seq: usize,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    let mut wm = WeightMap::from_tensors(
        weights
            .iter()
            .filter_map(|(k, v)| {
                k.strip_prefix(prefix)
                    .map(|rest| (format!("model.{rest}"), v.clone()))
            })
            .collect(),
    );
    let built = Llama32Flow::new(cfg)
        .prefill()
        .batch(1)
        .seq(seq)
        .inputs_embeds()
        .hidden_only()
        .build(&mut wm)
        .with_context(|| format!("build {name}"))?;
    let (g, params) = rlx_core::flow_util::graph_from_built(built)?;
    let rename = |p: &str| match p.strip_prefix("model.") {
        Some(rest) => format!("{prefix}{rest}"),
        None => format!("{name}.{p}"),
    };
    let mut g = rlx_ir::stateful::rebuild(&g, |out, _, node, _| match &node.op {
        Op::Param { name: p } => Ok(Some(out.param(rename(p), node.shape.clone()))),
        _ => Ok(None),
    })
    .map_err(anyhow::Error::msg)?;
    let out = g.outputs[0];
    g.node_mut(out).name = Some("hidden".into());
    g.name = name.into();
    Ok((
        g,
        params.into_iter().map(|(k, v)| (rename(&k), v)).collect(),
    ))
}

/// Export the CSM checkpoint in `dir` (`config.json`, `model.safetensors`,
/// `tokenizer.json`; Mimi from `RLX_MIMI_DIR`) as the package `out`.
pub fn export_rlxp(dir: &Path, out: &Path, opts: &ExportOptions) -> Result<ExportReport> {
    let ck = Checkpoint::open(dir)?;
    let cfg = SesameConfig::from_file(dir.join("config.json"))?;
    let raw: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("config.json"))?)?;
    let st = dir.join("model.safetensors");
    let source = WeightMap::from_file(st.to_str().context("weights path")?)?;
    let weights: HashMap<String, (Vec<f32>, Vec<usize>)> = source
        .keys()
        .map(|k| {
            let (d, s) = source.get(k).expect("listed");
            (k.to_string(), (d.to_vec(), s.to_vec()))
        })
        .collect();
    let mut pkg = Package::for_checkpoint(&ck, opts, "sesame", "text-to-speech", "rlx-sesame");

    // The backbone, symbolic in the frames, its RoPE tables as long as the
    // package reaches.
    let mut bb_cfg = raw.clone();
    bb_cfg["model_type"] = json!("llama");
    bb_cfg["vocab_size"] = json!(cfg.vocab_size);
    let bb_cfg = llama_config(bb_cfg)?;
    let backbone = symbolic_in(FRAMES, (5, 7, 3), |n| {
        Ok(llama_graph(&bb_cfg, &weights, "backbone_model.", "backbone", n)?.0)
    })?;
    let (at, mut params) = llama_graph(&bb_cfg, &weights, "backbone_model.", "backbone", 3)?;
    let inv = rlx_llama32::rope::resolve_inv_freq(&bb_cfg, None);
    let half = inv.len();
    let (cos, sin) = rlx_llama32::rope::build_rope_tables(&inv, MAX_FRAMES);
    let backbone = broadcast_params(
        &backbone,
        &at,
        &mut params,
        &[
            ("backbone.rope.cos", &cos, half),
            ("backbone.rope.sin", &sin, half),
        ],
    )?;
    pkg.function(backbone);

    // The depth decoder, at its 33 positions.
    let mut d_cfg = raw["depth_decoder_config"].clone();
    d_cfg["model_type"] = json!("llama");
    let d_cfg = llama_config(d_cfg)?;
    let dh = d_cfg.hidden_size;
    let (depth, p) = llama_graph(&d_cfg, &weights, "depth_decoder.model.", "depth", DEPTH_LEN)?;
    pkg.function(depth);
    params.extend(p);

    // The embeddings, the heads and the projection.
    let fe = rlx_ir::dynamic::generalize(
        &frame_embed(&cfg, 37),
        &frame_embed(&cfg, 53),
        &[(FRAMES, 37, 53)],
    )
    .map_err(anyhow::Error::msg)?;
    pkg.function(fe);
    params.insert(
        "frame_embed.offsets".into(),
        (0..cfg.num_codebooks)
            .map(|k| (k * cfg.vocab_size) as f32)
            .collect(),
    );
    let mut c0 = linear_graph(
        "c0_head",
        "h",
        "lm_head.weight",
        cfg.hidden_size,
        cfg.vocab_size,
    );
    c0.name = "c0_head".into();
    pkg.function(c0);
    pkg.function(linear_graph(
        "depth_project",
        "h",
        "depth_decoder.model.inputs_embeds_projector.weight",
        cfg.hidden_size,
        dh,
    ));
    pkg.function(depth_embed(&cfg, dh));
    pkg.function(codebook_head(&cfg, dh));
    for name in [
        "embed_text_tokens.weight",
        "depth_decoder.model.embed_tokens.weight",
        "lm_head.weight",
        "depth_decoder.model.inputs_embeds_projector.weight",
        "depth_decoder.codebooks_head.weight",
    ] {
        let (d, _) = weights.get(name).with_context(|| format!("no {name}"))?;
        params.insert(name.into(), d.clone());
    }

    // Mimi.
    let mimi_dir = rlx_mimi::default_mimi_dir();
    let (graphs, mimi_params, mimi_cfg, _) = rlx_mimi::export::mimi_functions(&mimi_dir)
        .with_context(|| format!("Mimi from {}", mimi_dir.display()))?;
    for mut g in graphs {
        g.name = format!("mimi_{}", g.name);
        pkg.function(g);
    }
    let mimi_held: HashMap<String, Vec<f32>> = mimi_params;
    // Mimi's symbols are the frames' (0) and the codebooks' (1).
    pkg.dims = vec![("frames".into(), FRAMES), ("codebooks".into(), 1)];

    // The checkpoint's tensors referenced, the rest held.
    let dims = pkg.param_dims();
    let mut names: Vec<&String> = mimi_held.keys().collect();
    names.sort();
    for n in names {
        let d = dims
            .get(n)
            .cloned()
            .unwrap_or_else(|| vec![mimi_held[n].len()]);
        pkg.hold_f32(n.clone(), d, &mimi_held[n]);
    }
    pkg.reference_map(&params, &source, "safetensors", &st, out, false)?;

    let tokenizer = pkg.tokenizer(&ck, None)?;
    let module = program::build(&Path::new(env!("CARGO_MANIFEST_DIR")).join("program/Cargo.toml"))?;
    program::entry(
        &mut pkg,
        "speak",
        module,
        &["synthesize", "run"],
        "text-to-speech",
        json!({
            "tokenizer": tokenizer,
            "bos": 128_000,
            "eos": 128_001,
            "num_codebooks": cfg.num_codebooks,
            "depth_len": DEPTH_LEN,
            "hidden": cfg.hidden_size,
            "depth_hidden": dh,
            "max_audio_frames": 1125,
            "temperature": 0.9,
            "topk": 50,
            "seed": 42,
            "sample_rate": mimi_cfg.sampling_rate,
        }),
    );
    pkg.metadata("sample_rate", json!(mimi_cfg.sampling_rate));
    pkg.write(out, opts)
}

/// Register this crate's exporter ([`rlx_core::export::register_exporter`]).
pub fn register() {
    use rlx_core::export::{Exporter, register_exporter};
    register_exporter(Exporter::new("sesame", "text-to-speech", export_rlxp).claims(&["csm"]));
}
