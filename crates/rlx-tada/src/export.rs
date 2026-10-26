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

//! Export HumeAI TADA as an `.rlxp` package rlx runs alone: text in, 24 kHz
//! speech out in the voice of a prompt, through the one exporter
//! ([`rlx_core::export`]).
//!
//! Its graphs are the runner's: `embed` (each row's token embedding plus
//! its acoustic conditioning — the previous latent through
//! `acoustic_proj`, its mask and frame-gap embeddings — for both guidance
//! branches); `prefill` (the Llama-3.2 backbone over the prompt's rows,
//! symbolic in them and in the caches' rows, its keys and values kept) and
//! `step` (one row, its key and value written in at `position`; the hidden
//! state of both branches); `solver` (the flow-matching head's whole ODE
//! solve, every Euler step and the guidance blend); and the codec decoder
//! as `codec_attn` (its local-attention stack, the segment mask computed
//! from the frames' token mask) and `codec_conv` (the DAC stack), symbolic
//! in the frames. The program (`program/`) builds the prompt from the text
//! and a voice (`voice.json`, the voice prompt the export found beside the
//! checkpoint; `options.voice` another), runs the autoregressive loop (the
//! crate's `rng.rs` noise, `gray.rs` durations, `text.rs` normalization),
//! lays the latents on the frame grid and trims as the runner does.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use rlx_core::export::graph::{broadcast_params, symbolic_in, symbolic_in_lifted_many};
use rlx_core::export::{ExportOptions, ExportReport, Package, program};
use rlx_core::weight_map::WeightMap;
use rlx_ir::hir::{HirModule, HirMut};
use rlx_ir::{DType, Graph, GraphExt, HirGraphExt, NodeId, Op, Shape};
use rlx_llama32::{Llama32Config, Llama32Flow};
use serde_json::json;

use crate::builder::{Ctx, lower};
use crate::codec::CodecDecoder;
use crate::config::{DecoderConfig, FRAME_RATE, SAMPLE_RATE, TadaConfig};
use crate::head::{DiffusionHead, SolveOptions};
use crate::prompt::VoicePrompt;
use crate::tokenizer::{PREFIX_TEMPLATE, TadaTokenizer};
use crate::weights::TensorStore;

/// The symbols: the codec's frames, the prompt's rows, the caches' rows.
const FRAMES: u32 = 0;
const PROMPT: u32 = 2;
const CAP: u32 = 3;
const EMBED: &str = "model.embed_tokens.weight";

fn f32s(d: &[usize]) -> Shape {
    Shape::new(d, DType::F32)
}

fn constant(g: &mut Graph, v: f32) -> NodeId {
    g.add_node(
        Op::Constant {
            data: v.to_le_bytes().to_vec(),
        },
        vec![],
        f32s(&[1]),
    )
}

fn expand(g: &mut Graph, x: NodeId, dims: &[usize]) -> NodeId {
    g.add_node(
        Op::Expand {
            target_shape: dims.iter().map(|&d| d as i64).collect(),
        },
        vec![x],
        f32s(dims),
    )
}

fn name(g: &mut Graph, id: NodeId, n: &str) {
    g.node_mut(id).name = Some(n.into());
}

/// `g` with each param `p` named `{prefix}{p}`, and `params` likewise: a
/// builder's `w0`, `w1`, … are unique only within its own graph.
fn prefixed(
    g: &Graph,
    params: Vec<(String, Vec<f32>)>,
    prefix: &str,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    let g = rlx_ir::stateful::rebuild(g, |out, _, node, _| match &node.op {
        Op::Param { name } => {
            let id = out.param(format!("{prefix}{name}"), node.shape.clone());
            Ok(Some(id))
        }
        _ => Ok(None),
    })
    .map_err(anyhow::Error::msg)?;
    Ok((
        g,
        params
            .into_iter()
            .map(|(n, v)| (format!("{prefix}{n}"), v))
            .collect(),
    ))
}

/// `rows` (`[b, n, w]`) written at row `at` of `cache` (`[b, cap, w]`).
fn write_at(g: &mut Graph, cache: NodeId, rows: NodeId, at: NodeId, n: &str) -> NodeId {
    let shape = g.node(cache).shape.clone();
    let full = g.add_node(
        Op::DynamicUpdateSlice { axis: 1 },
        vec![cache, rows, at],
        shape,
    );
    name(g, full, n);
    full
}

/// `embed` at `t` rows: `tokens` `[2, t]` i32 (each guidance branch's),
/// `latent` `[t, acoustic]` (the latent each row feeds back), `mask` /
/// `before` / `after` `[t]` i32 (whether it has one, its frame gaps) →
/// `embeds` `[2, t, hidden]`: `InputEmbedder::conditioning` plus the token's
/// row.
fn embed(cfg: &TadaConfig, t: usize) -> Graph {
    let (h, a, c) = (cfg.hidden_size, cfg.acoustic_dim, cfg.num_time_classes);
    let mut g = Graph::new("embed");
    let tokens = g.input("tokens", Shape::new(&[2, t], DType::I32));
    let latent = g.input("latent", f32s(&[t, a]));
    let mask = g.input("mask", Shape::new(&[t], DType::I32));
    let before = g.input("before", Shape::new(&[t], DType::I32));
    let after = g.input("after", Shape::new(&[t], DType::I32));
    let w = g.param("acoustic_proj.weight", f32s(&[h, a]));
    let w = g.transpose_(w, vec![1, 0]);
    let x = g.mm(latent, w);
    let b = g.param("acoustic_proj.bias", f32s(&[h]));
    let b = g.reshape_(b, vec![1, h as i64]);
    let b = expand(&mut g, b, &[t, h]);
    let mut cond = g.add(x, b);
    for (table, rows, at) in [
        ("acoustic_mask_emb.weight", 2, mask),
        ("time_start_embed.weight", c, before),
        ("time_end_embed.weight", c, after),
    ] {
        let p = g.param(table, f32s(&[rows, h]));
        let e = g.gather_(p, at, 0);
        cond = g.add(cond, e);
    }
    let table = g.param(EMBED, f32s(&[cfg.vocab_size, h]));
    let flat = g.reshape_(tokens, vec![(2 * t) as i64]);
    let tok = g.gather_(table, flat, 0);
    let tok = g.reshape_(tok, vec![2, t as i64, h as i64]);
    let cond = g.reshape_(cond, vec![1, t as i64, h as i64]);
    let cond = expand(&mut g, cond, &[2, t, h]);
    let x = g.add(tok, cond);
    name(&mut g, x, "embeds");
    g.set_outputs(vec![x]);
    g
}

/// The backbone's decoder tensors as f32, for one graph build.
fn backbone_weights(store: &TensorStore) -> Result<WeightMap> {
    let mut tensors = HashMap::new();
    for n in store.keys() {
        if n.starts_with("model.layers.") || n == "model.norm.weight" {
            tensors.insert(n.to_string(), (store.get(n)?, store.shape(n)?.to_vec()));
        }
    }
    Ok(WeightMap::from_tensors(tensors))
}

/// `prefill` at `t` rows and `cap` cache rows: both branches' prompt
/// through the backbone, each layer's keys and values at the start of its
/// caches (`past_k_{l}`, `past_v_{l}` `[2, cap, kv]`).
fn prefill(cfg: &Llama32Config, store: &TensorStore, t: usize, cap: usize) -> Result<Built> {
    let built = Llama32Flow::new(cfg)
        .prefill()
        .batch(2)
        .seq(t)
        .inputs_embeds()
        .hidden_only()
        .export_kv()
        .build(&mut backbone_weights(store)?)?;
    let (mut g, params) = rlx_core::flow_util::graph_from_built(built)?;
    let o = g.outputs.clone();
    let kv = cfg.num_key_value_heads * cfg.head_dim();
    let origin = constant(&mut g, 0.0);
    let zero = constant(&mut g, 0.0);
    let zero = g.reshape_(zero, vec![1, 1, 1]);
    let mut outs = Vec::new();
    for (i, &row) in o[1..].iter().enumerate() {
        let z = expand(&mut g, zero, &[2, cap, kv]);
        let kind = if i % 2 == 0 { "k" } else { "v" };
        outs.push(write_at(
            &mut g,
            z,
            row,
            origin,
            &format!("past_{kind}_{}", i / 2),
        ));
    }
    g.set_outputs(outs);
    g.name = "prefill".into();
    Ok((g, params))
}

/// `step` at `cap` cache rows: both branches' row at `position`, its
/// `hidden` `[2, 1, hidden]` and each layer's caches with its key and value
/// in (`past_k_{l}.next`, `past_v_{l}.next`).
fn step(cfg: &Llama32Config, store: &TensorStore, cap: usize) -> Result<Built> {
    let built = Llama32Flow::for_decode(cfg, 2, cap)
        .custom_mask()
        .inputs_embeds()
        .hidden_only()
        .export_kv()
        .build(&mut backbone_weights(store)?)?;
    let (mut g, params) = rlx_core::flow_util::graph_from_built(built)?;
    let o = g.outputs.clone();
    name(&mut g, o[0], "hidden");
    let pos = g.input_id("position").context("no position")?;
    let mut outs = vec![o[0]];
    for (i, &full) in o[1..].iter().enumerate() {
        let kind = if i % 2 == 0 { "k" } else { "v" };
        let input = format!("past_{kind}_{}", i / 2);
        let cache = g.input_id(&input).with_context(|| format!("no {input}"))?;
        let rows = g.node(full).shape.dims()[1].unwrap_static();
        let row = g.narrow_(full, 1, rows - 1, 1);
        outs.push(write_at(&mut g, cache, row, pos, &format!("{input}.next")));
    }
    g.set_outputs(outs);
    g.name = "step".into();
    Ok((g, params))
}

type Built = (Graph, HashMap<String, Vec<f32>>);

/// `solver`: the flow-matching head's ODE solve at `opts` — `noise` `[1,
/// latent]`, `cond` and `neg_cond` `[1, hidden]` → `speech` `[1, latent]`.
fn solver(head: &DiffusionHead, opts: &SolveOptions) -> Result<Built> {
    let (hir, params) = head.build_solver(opts)?;
    let g = lower(hir, "diffusion head solver")?;
    let (mut g, params) = prefixed(&g, params, "head.")?;
    let out = g.outputs[0];
    name(&mut g, out, "speech");
    g.name = "solver".into();
    Ok((g, params))
}

/// `codec_attn` at `f` frames: `z` `[1, f, embed]` and its `mask` `[f]`
/// (1 on a token's frame) → `x` `[1, hidden, f, 1]`, the decoder's
/// projection and local-attention stack, under the segment mask
/// (`mask::decoder_segment_bias`: a frame sees its block and the one before,
/// blocks ending at a token's frame) computed from `mask`.
fn codec_attn(dec: &CodecDecoder, f: usize) -> Result<Built> {
    let cfg: &DecoderConfig = &dec.cfg;
    let (e, h, heads) = (cfg.embed_dim, cfg.hidden_dim, cfg.num_attn_heads);
    let mut hir = HirModule::new("codec_attn");
    let mut hg = HirMut::new(&mut hir);
    let mut ctx = Ctx::new(&mut hg);
    let z = ctx.g.input("z", f32s(&[1, f, e]));
    let x = ctx.linear(z, &dec.proj.weight, h, e, dec.proj.bias.as_deref());
    let bias = ctx.g.input("bias", f32s(&[1, 1, f, f]));
    let bias = ctx
        .g
        .expand_(bias, vec![1, heads as i64, f as i64, f as i64]);
    let st = &dec.stack;
    let half = st.head_dim() / 2;
    let cos = ctx.param(st.cos.clone(), &[st.max_seq, half]);
    let sin = ctx.param(st.sin.clone(), &[st.max_seq, half]);
    let cos = ctx.g.narrow_(cos, 0, 0, f);
    let sin = ctx.g.narrow_(sin, 0, 0, f);
    let y = st.build_with(&mut ctx, x, bias, f, cos, sin)?;
    let y = ctx.g.transpose_(y, vec![0, 2, 1]);
    let y = ctx.g.reshape_(y, vec![1, h as i64, f as i64, 1]);
    let params = ctx.into_params();
    hir.set_outputs(vec![y]);
    let g = lower(hir, "codec attention")?;
    let (g, params) = prefixed(&g, params, "codec.attn.")?;
    // The bias from the mask: block = cumsum(mask) − mask; frame i reads
    // frame j when block_i − block_j is 0 or 1.
    let mut g = rlx_ir::stateful::rebuild(&g, |out, _, node, _| match &node.op {
        Op::Input { name } if name == "bias" => {
            let m = out.input("mask", f32s(&[f]));
            let block = out.add_node(
                Op::Cumsum {
                    axis: 0,
                    exclusive: true,
                },
                vec![m],
                f32s(&[f]),
            );
            let bi = out.reshape_(block, vec![f as i64, 1]);
            let bi = expand(out, bi, &[f, f]);
            let bj = out.reshape_(block, vec![1, f as i64]);
            let bj = expand(out, bj, &[f, f]);
            let d = out.sub(bi, bj);
            let half_c = constant(out, 0.5);
            let d = out.sub(d, half_c);
            let d = out.abs(d);
            let one = constant(out, 1.0);
            let one = expand(out, one, &[f, f]);
            let near = out.lt(d, one);
            let near = out.add_node(Op::Cast { to: DType::F32 }, vec![near], f32s(&[f, f]));
            let one = constant(out, 1.0);
            let off = out.sub(near, one);
            let big = constant(out, 1e30);
            let bias = out.mul(off, big);
            Ok(Some(out.reshape_(bias, vec![1, 1, f as i64, f as i64])))
        }
        _ => Ok(None),
    })
    .map_err(anyhow::Error::msg)?;
    let out = g.outputs[0];
    name(&mut g, out, "x");
    g.name = "codec_attn".into();
    Ok((g, params))
}

/// `codec_conv` at `f` frames: `x` `[1, hidden, f, 1]` → `audio` (the DAC
/// stack's waveform, flat).
fn codec_conv(dec: &CodecDecoder, f: usize) -> Result<Built> {
    let h = dec.cfg.hidden_dim;
    let (g, params, (_, samples)) = rlx_dac::graph::build_decoder_graph(&dec.wav_layers()?, h, f)?;
    let params: Vec<(String, Vec<f32>)> = params.into_iter().collect();
    let g = rlx_ir::stateful::rebuild(&g, |out, _, node, _| match &node.op {
        Op::Input { name } if name == "z" => Ok(Some(out.input("x", node.shape.clone()))),
        _ => Ok(None),
    })
    .map_err(anyhow::Error::msg)?;
    let (mut g, params) = prefixed(&g, params, "codec.conv.")?;
    let out = g.outputs[0];
    let audio = g.reshape_(out, vec![samples as i64]);
    name(&mut g, audio, "audio");
    g.set_outputs(vec![audio]);
    g.name = "codec_conv".into();
    Ok((g, params))
}

/// The voice prompt beside the checkpoint: `harness.tadaprompt`, else the
/// only `.tadaprompt` there.
fn find_voice(root: &Path) -> Result<std::path::PathBuf> {
    let harness = root.join("harness.tadaprompt");
    if harness.is_file() {
        return Ok(harness);
    }
    let found: Vec<_> = std::fs::read_dir(root)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "tadaprompt"))
        .collect();
    match found.as_slice() {
        [one] => Ok(one.clone()),
        _ => bail!(
            "{}: no voice — build one with `rlx-tada prompt` (`just tada-harness-prompt`) \
             as harness.tadaprompt",
            root.display()
        ),
    }
}

/// A voice prompt as the program reads it: its token ids, their latents
/// (`[tokens, acoustic]`, flat) and frame gaps.
pub fn voice_json(prompt: &VoicePrompt, cfg: &TadaConfig) -> serde_json::Value {
    let (before, after) = prompt.frame_gaps(cfg.num_time_classes as u32);
    json!({
        "token_ids": prompt.token_ids,
        "token_values": prompt.token_values.iter().copied().collect::<Vec<f32>>(),
        "before": before,
        "after": after,
    })
}

/// The BOOL tensors of the safetensors file `path`, by its header.
fn bool_tensors(path: &Path) -> Result<std::collections::HashSet<String>> {
    use std::io::Read;
    let mut f = std::fs::File::open(path)?;
    let mut len = [0u8; 8];
    f.read_exact(&mut len)?;
    let mut header = vec![0u8; u64::from_le_bytes(len) as usize];
    f.read_exact(&mut header)?;
    let header: serde_json::Value = serde_json::from_slice(&header)?;
    Ok(header
        .as_object()
        .context("a safetensors header is an object")?
        .iter()
        .filter(|(_, v)| v["dtype"] == "BOOL")
        .map(|(k, _)| k.clone())
        .collect())
}

/// Export the TADA root `dir` (`tada-1b/`, `tada-codec/decoder/`,
/// `tokenizer/`, a voice prompt `harness.tadaprompt`) as the package `out`.
pub fn export_rlxp(dir: &Path, out: &Path, opts: &ExportOptions) -> Result<ExportReport> {
    let backbone_dir = crate::model::find_backbone(dir)?;
    let config = backbone_dir.join("config.json");
    let cfg = TadaConfig::from_file(&config)?;
    let llama = Llama32Config::from_file(&config)?;
    let weights = backbone_dir.join("model.safetensors");
    let store = Arc::new(TensorStore::open(&weights)?);
    let solve = SolveOptions::default();
    let mut pkg = Package::new(
        "tada-1b",
        &opts.version,
        "tada",
        "text-to-speech",
        "rlx-tada",
    );
    pkg.model["license"] = json!("llama3.2");
    let mut params: HashMap<String, Vec<f32>> = HashMap::new();

    // The backbone and its rows' embedding.
    let (p, lifted) = symbolic_in_lifted_many(
        &[(PROMPT, vec![5, 7, 3]), (CAP, vec![37, 53, 20])],
        &[llama.max_position_embeddings, llama.max_position_embeddings],
        |n| Ok(prefill(&llama, &store, n[0], n[1])?.0),
    )?;
    anyhow::ensure!(lifted.is_empty(), "the prompt graph lifted tables");
    pkg.function(p);
    params.extend(prefill(&llama, &store, 3, 20)?.1);
    pkg.function(symbolic_in(CAP, (37, 53, 20), |c| {
        Ok(step(&llama, &store, c)?.0)
    })?);
    params.extend(step(&llama, &store, 20)?.1);
    pkg.function(symbolic_in(PROMPT, (5, 7, 3), |t| Ok(embed(&cfg, t)))?);
    let (sg, sp) = solver(
        &DiffusionHead::load(store.clone(), &cfg, "prediction_head.")?,
        &solve,
    )?;
    pkg.function(sg);
    params.extend(sp);

    // The codec decoder.
    let dec_path = dir.join("tada-codec/decoder/model.safetensors");
    let dec_store = Arc::new(TensorStore::open(&dec_path)?);
    let dec = CodecDecoder::load(dec_store, "", DecoderConfig::default())?;
    let (ag, lifted) =
        symbolic_in_lifted_many(&[(FRAMES, vec![5, 7, 4])], &[dec.stack.max_seq], |n| {
            Ok(codec_attn(&dec, n[0])?.0)
        })?;
    anyhow::ensure!(lifted.is_empty(), "the codec attention lifted tables");
    pkg.function(ag);
    params.extend(codec_attn(&dec, 4)?.1);
    let conv = symbolic_in(FRAMES, (5, 7, 4), |f| Ok(codec_conv(&dec, f)?.0))?;
    let (at4, mut conv_params) = codec_conv(&dec, 4)?;
    let conv = broadcast_params(&conv, &at4, &mut conv_params, &[])?;
    pkg.function(conv);
    params.extend(conv_params);
    pkg.dims = vec![
        ("frames".into(), FRAMES),
        ("prompt".into(), PROMPT),
        ("cap".into(), CAP),
    ];

    // Checkpoint tensors referenced (a fingerprint match, transposed or
    // not), the rest held: the codec's folded convolutions, the solver's
    // schedule constants.
    // Its BOOL tensors (attention-mask buffers) are no weight.
    let skip = bool_tensors(&weights)?;
    let source = WeightMap::from_file_excluding(weights.to_str().context("path")?, &skip)?;
    pkg.reference_map(&params, &source, "safetensors", &weights, out, false)?;
    drop(source);

    // The text front end and the voice.
    let tok = TadaTokenizer::load(&dir.join("tokenizer"))?;
    let prefix = tok.encode(PREFIX_TEMPLATE)?;
    let s = tok.special;
    pkg.sidecar(
        "tokenizer.json",
        "application/json",
        std::fs::read(dir.join("tokenizer/tokenizer.json"))?,
    );
    let voice = VoicePrompt::load(&find_voice(dir)?)?;
    voice.validate()?;
    pkg.sidecar(
        "voice.json",
        "application/json",
        serde_json::to_vec(&voice_json(&voice, &cfg))?,
    );
    let module = program::build(&Path::new(env!("CARGO_MANIFEST_DIR")).join("program/Cargo.toml"))
        .context("build TADA's program")?;
    program::entry(
        &mut pkg,
        "speak",
        module,
        &["synthesize", "run"],
        "text-to-speech",
        json!({
            "tokenizer": "sidecar:tokenizer.json",
            "voice": "voice.json",
            "prefix": prefix,
            "bos": s.bos,
            "eos": s.eos,
            "eot": s.eot,
            "start_header": s.start_header,
            "end_header": s.end_header,
            "pad": s.pad,
            "layers": llama.num_hidden_layers,
            "hidden": cfg.hidden_size,
            "acoustic_dim": cfg.acoustic_dim,
            "time_bits": cfg.num_time_bits(),
            "time_classes": cfg.num_time_classes,
            "latent": cfg.latent_dim(),
            "shift": cfg.shift_acoustic,
            "num_eos": cfg.num_eos_tokens(),
            "acoustic_mean": cfg.acoustic_mean,
            "acoustic_std": cfg.acoustic_std,
            "noise_temperature": solve.noise_temperature,
            "seed": crate::synth::SynthOptions::default().seed,
            "sample_rate": SAMPLE_RATE,
            "frame_rate": FRAME_RATE,
        }),
    );
    pkg.metadata("sample_rate", json!(SAMPLE_RATE));
    pkg.write(out, opts)
}

/// Register this crate's exporter ([`rlx_core::export::register_exporter`]).
pub fn register() {
    use rlx_core::export::{Exporter, register_exporter};
    register_exporter(Exporter::new("tada", "text-to-speech", export_rlxp));
}
