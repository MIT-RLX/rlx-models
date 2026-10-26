// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: Apache-2.0

//! Export Gepard as an `.rlxp` package rlx runs alone: text in, 22.05 kHz
//! speech out, through the one exporter ([`rlx_core::export`]).
//!
//! Its graphs, as [`crate::GepardSynthesizer`]'s eager loop computes:
//! `prefill` (the prompt's tokens through the Qwen3.5 backbone, symbolic in
//! their number and the caches' rows; the last row's heads) and `step` (one
//! frame's 32 codes embedded — per-channel rows, the GELU MLP, the
//! affine-free norm, the scale — through the backbone after the caches,
//! its row written in place, symbolic in the rows; its heads): each ends in
//! the 32 codebook heads' logits and the stop head's. Then the NanoCodec
//! HiFi-GAN decoder, symbolic in the frames (`decode`). The backbone and
//! heads are the checkpoint's tensors, referenced. The program
//! (`program/`) builds the prompt (the host's tokenizer, the text
//! repeated), samples each frame's channels (the runner's `fastrand`
//! stream), stops on the stop head, dequantizes and decodes.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use rlx_core::export::graph::{broadcast_params, symbolic_in, symbolic_in_lifted_many};
use rlx_core::export::{ExportOptions, ExportReport, Package, program};
use rlx_core::weight_map::WeightMap;
use rlx_ir::op::MaskKind;
use rlx_ir::{DType, Graph, GraphExt, NodeId, Op, Shape};
use serde_json::json;

use crate::codec_ops::{FSQ_LEVELS, NUM_CHANNELS};
use crate::config::GepardConfig;
use crate::gepard_decoder::{build_gepard_graph, load_gepard_nano_weights};

/// The symbols: the decoder's frames, the prompt's length, the caches' rows.
const FRAMES: u32 = 0;
const PROMPT: u32 = 2;
const CAP: u32 = 3;
/// The RoPE tables' rows: the longest context a request reaches.
const MAX_CONTEXT: usize = 4096;
const SAMPLE_RATE: u32 = 22_050;

fn f32s(d: &[usize]) -> Shape {
    Shape::new(d, DType::F32)
}

fn scalar(g: &mut Graph, v: f32, rank: usize) -> NodeId {
    g.add_node(
        Op::Constant {
            data: v.to_le_bytes().to_vec(),
        },
        vec![],
        f32s(&vec![1; rank]),
    )
}

/// `x @ wᵀ (+ b)` for a `[out, in]` weight.
fn linear(g: &mut Graph, x: NodeId, w: &str, b: Option<&str>, out: usize, inp: usize) -> NodeId {
    let wp = g.param(w, f32s(&[out, inp]));
    let wt = g.transpose_(wp, vec![1, 0]);
    let y = g.mm(x, wt);
    match b {
        Some(b) => {
            let bp = g.param(b, f32s(&[out]));
            g.add(y, bp)
        }
        None => y,
    }
}

/// Qwen3.5's RMSNorm, `x / rms(x) · (1 + w)`, over the last axis.
fn norm_1p(g: &mut Graph, x: NodeId, w: &str, width: usize, eps: f32) -> NodeId {
    let wp = g.param(w, f32s(&[width]));
    let one = scalar(g, 1.0, 1);
    let gamma = g.add(wp, one);
    let zero = g.param(format!("gepard.zero.{width}"), f32s(&[width]));
    g.rms_norm(x, gamma, zero, eps)
}

/// A step's cache read: its position `pos` `[1]` (f32) and the keys it may
/// see, `mask` `[1, cap]`.
struct StepAt {
    pos: NodeId,
    mask: NodeId,
}

/// The backbone over `x` `[1, t, hidden]` at RoPE rows `cos` / `sin`
/// `[t, head_dim / 2]`: each layer's keys and values as caches of `cap`
/// rows — a prompt's (`step` none) its rows written over zeros from row 0,
/// a step's its row written into `k{i}` / `v{i}` at `pos`. The final norm's
/// output and the caches (a step's named `k{i}.next`, `v{i}.next`).
#[allow(clippy::too_many_arguments)]
fn backbone(
    g: &mut Graph,
    cfg: &GepardConfig,
    mut x: NodeId,
    cos: NodeId,
    sin: NodeId,
    t: usize,
    cap: usize,
    step: Option<&StepAt>,
) -> (NodeId, Vec<NodeId>) {
    let bb = &cfg.backbone;
    let (hidden, heads, kv_heads) = (
        bb.hidden_size,
        bb.num_attention_heads,
        bb.num_key_value_heads,
    );
    let hd = bb.effective_head_dim();
    let eps = bb.rms_norm_eps as f32;
    let (q_dim, kv_dim) = (heads * hd, kv_heads * hd);
    let group = heads / kv_heads;
    let cache = f32s(&[1, cap, kv_dim]);
    let origin = scalar(g, 0.0, 1);
    let nought = scalar(g, 0.0, 3);
    let mut caches = Vec::new();
    for i in 0..bb.num_hidden_layers {
        let p = format!("model.layers.{i}");
        let n1 = norm_1p(g, x, &format!("{p}.input_layernorm.weight"), hidden, eps);
        // q_proj is per head `[Q | gate]`.
        let qg = linear(
            g,
            n1,
            &format!("{p}.self_attn.q_proj.weight"),
            None,
            2 * q_dim,
            hidden,
        );
        let qg = g.reshape_(qg, vec![1, t as i64, heads as i64, 2 * hd as i64]);
        let q = g.narrow_(qg, 3, 0, hd);
        let gate = g.narrow_(qg, 3, hd, hd);
        let gate = g.reshape_(gate, vec![1, t as i64, q_dim as i64]);
        let q = norm_1p(g, q, &format!("{p}.self_attn.q_norm.weight"), hd, eps);
        let q = g.reshape_(q, vec![1, t as i64, q_dim as i64]);
        let q = g.rope(q, cos, sin, hd);
        let k = linear(
            g,
            n1,
            &format!("{p}.self_attn.k_proj.weight"),
            None,
            kv_dim,
            hidden,
        );
        let k = g.reshape_(k, vec![1, t as i64, kv_heads as i64, hd as i64]);
        let k = norm_1p(g, k, &format!("{p}.self_attn.k_norm.weight"), hd, eps);
        let k = g.reshape_(k, vec![1, t as i64, kv_dim as i64]);
        let k = g.rope(k, cos, sin, hd);
        let v = linear(
            g,
            n1,
            &format!("{p}.self_attn.v_proj.weight"),
            None,
            kv_dim,
            hidden,
        );
        let mut kv = Vec::new();
        for (name, row) in [(format!("k{i}"), k), (format!("v{i}"), v)] {
            let (base, at, out) = match step {
                Some(s) => (
                    g.input(name.clone(), cache.clone()),
                    s.pos,
                    format!("{name}.next"),
                ),
                None => {
                    let z = g.add_node(
                        Op::Expand {
                            target_shape: vec![1, cap as i64, kv_dim as i64],
                        },
                        vec![nought],
                        cache.clone(),
                    );
                    (z, origin, name)
                }
            };
            let full = g.add_node(
                Op::DynamicUpdateSlice { axis: 1 },
                vec![base, row, at],
                cache.clone(),
            );
            g.node_mut(full).name = Some(out);
            caches.push(full);
            kv.push(full);
        }
        // Each key / value head repeated for its group of query heads.
        let rows = if step.is_some() { cap } else { t };
        let repeat = |g: &mut Graph, x: NodeId| -> NodeId {
            let x = g.reshape_(x, vec![1, rows as i64, kv_heads as i64, 1, hd as i64]);
            let x = g.add_node(
                Op::Expand {
                    target_shape: vec![1, rows as i64, kv_heads as i64, group as i64, hd as i64],
                },
                vec![x],
                f32s(&[1, rows, kv_heads, group, hd]),
            );
            g.reshape_(x, vec![1, rows as i64, q_dim as i64])
        };
        let a = match step {
            Some(s) => {
                let kr = repeat(g, kv[0]);
                let vr = repeat(g, kv[1]);
                g.attention(q, kr, vr, s.mask, heads, hd, f32s(&[1, t, q_dim]))
            }
            None => {
                let kr = repeat(g, k);
                let vr = repeat(g, v);
                g.attention_kind(q, kr, vr, heads, hd, MaskKind::Causal, f32s(&[1, t, q_dim]))
            }
        };
        let gate = g.sigmoid(gate);
        let a = g.mul(a, gate);
        let a = linear(
            g,
            a,
            &format!("{p}.self_attn.o_proj.weight"),
            None,
            hidden,
            q_dim,
        );
        x = g.add(x, a);
        let n2 = norm_1p(
            g,
            x,
            &format!("{p}.post_attention_layernorm.weight"),
            hidden,
            eps,
        );
        let inter = bb.intermediate_size;
        let gt = linear(
            g,
            n2,
            &format!("{p}.mlp.gate_proj.weight"),
            None,
            inter,
            hidden,
        );
        let gt = g.silu(gt);
        let up = linear(
            g,
            n2,
            &format!("{p}.mlp.up_proj.weight"),
            None,
            inter,
            hidden,
        );
        let h = g.mul(gt, up);
        let m = linear(
            g,
            h,
            &format!("{p}.mlp.down_proj.weight"),
            None,
            hidden,
            inter,
        );
        x = g.add(x, m);
    }
    (norm_1p(g, x, "model.norm.weight", hidden, eps), caches)
}

/// The heads on `h` `[1, hidden]`: the 32 codebook heads' logits, then the
/// stop head's, as `logits` `[1, Σ levels + 1]`.
fn heads(g: &mut Graph, cfg: &GepardConfig, h: NodeId) -> NodeId {
    let hidden = cfg.hidden_size();
    let vocabs = cfg.codec.channel_vocabs();
    let mut outs = Vec::new();
    for (ch, &v) in vocabs.iter().enumerate() {
        outs.push(linear(
            g,
            h,
            &format!("codebook_heads.{ch}.weight"),
            Some(&format!("codebook_heads.{ch}.bias")),
            v as usize,
            hidden,
        ));
    }
    outs.push(linear(
        g,
        h,
        "stop_head.weight",
        Some("stop_head.bias"),
        1,
        hidden,
    ));
    let logits = g.concat_(outs, 1);
    g.node_mut(logits).name = Some("logits".into());
    logits
}

/// `prefill`: `ids` `[t]` i32 → the last row's `logits` and each layer's
/// `k{i}`, `v{i}` `[1, cap, kv_dim]`.
fn prefill(cfg: &GepardConfig, t: usize, cap: usize) -> Graph {
    let bb = &cfg.backbone;
    let (hidden, half) = (bb.hidden_size, bb.effective_head_dim() / 2);
    let mut g = Graph::new("prefill");
    let ids = g.input("ids", Shape::new(&[t], DType::I32));
    let table = g.param("model.embed_tokens.weight", f32s(&[bb.vocab_size, hidden]));
    let x = g.gather_(table, ids, 0);
    let x = g.reshape_(x, vec![1, t as i64, hidden as i64]);
    let cos = g.param("rope.cos", f32s(&[MAX_CONTEXT, half]));
    let sin = g.param("rope.sin", f32s(&[MAX_CONTEXT, half]));
    let cos = g.narrow_(cos, 0, 0, t);
    let sin = g.narrow_(sin, 0, 0, t);
    let (xf, caches) = backbone(&mut g, cfg, x, cos, sin, t, cap, None);
    let last = g.narrow_(xf, 1, t - 1, 1);
    let last = g.reshape_(last, vec![1, hidden as i64]);
    let logits = heads(&mut g, cfg, last);
    let mut outs = vec![logits];
    outs.extend(caches);
    g.set_outputs(outs);
    g
}

/// `step`: the last frame's `codes` `[32]` i32 at `pos` `[1]` (f32), the
/// keys it sees `mask` `[1, cap]`, each layer's caches → its `logits` and
/// the caches with its row written at `pos`.
fn step(cfg: &GepardConfig, cap: usize) -> Graph {
    let bb = &cfg.backbone;
    let (hidden, half) = (bb.hidden_size, bb.effective_head_dim() / 2);
    let dim = cfg.audio_embed_dim;
    let vocabs = cfg.codec.channel_vocabs();
    let mut g = Graph::new("step");
    let codes = g.input("codes", Shape::new(&[NUM_CHANNELS], DType::I32));
    let at = g.input("pos", f32s(&[1]));
    let mask = g.input("mask", f32s(&[1, cap]));
    // The frame's embedding: each channel's row, the MLP, the affine-free
    // norm, the scale.
    let mut rows = Vec::new();
    for (ch, &v) in vocabs.iter().enumerate() {
        let table = g.param(
            format!("audio_embeddings.{ch}.weight"),
            f32s(&[v as usize, dim]),
        );
        let code = g.narrow_(codes, 0, ch, 1);
        rows.push(g.gather_(table, code, 0));
    }
    let x = g.concat_(rows, 1);
    let x = linear(
        &mut g,
        x,
        "audio_embed_proj.0.weight",
        Some("audio_embed_proj.0.bias"),
        hidden,
        hidden,
    );
    let x = g.gelu_approx(x);
    let x = linear(
        &mut g,
        x,
        "audio_embed_proj.2.weight",
        Some("audio_embed_proj.2.bias"),
        hidden,
        hidden,
    );
    let ones = g.param("gepard.one", f32s(&[hidden]));
    let zero = g.param(format!("gepard.zero.{hidden}"), f32s(&[hidden]));
    let x = g.ln(x, ones, zero, 1e-5);
    let scale = g.param("gepard.audio_embed_scale", f32s(&[1]));
    let x = g.mul(x, scale);
    let x = g.reshape_(x, vec![1, 1, hidden as i64]);
    let cos = g.param("rope.cos", f32s(&[MAX_CONTEXT, half]));
    let sin = g.param("rope.sin", f32s(&[MAX_CONTEXT, half]));
    let cos = g.gather_(cos, at, 0);
    let sin = g.gather_(sin, at, 0);
    let s = StepAt { pos: at, mask };
    let (xf, caches) = backbone(&mut g, cfg, x, cos, sin, 1, cap, Some(&s));
    let h = g.reshape_(xf, vec![1, hidden as i64]);
    let logits = heads(&mut g, cfg, h);
    let mut outs = vec![logits];
    outs.extend(caches);
    g.set_outputs(outs);
    g
}

/// The eager backbone's RoPE angles, `pos / θ^(2i / head_dim)` in f32.
fn rope_tables(theta: f32, head_dim: usize) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;
    let (mut cos, mut sin) = (
        Vec::with_capacity(MAX_CONTEXT * half),
        Vec::with_capacity(MAX_CONTEXT * half),
    );
    for pos in 0..MAX_CONTEXT {
        for i in 0..half {
            let angle = pos as f32 / theta.powf((2 * i) as f32 / head_dim as f32);
            let (s, c) = angle.sin_cos();
            cos.push(c);
            sin.push(s);
        }
    }
    (cos, sin)
}

/// Export the Gepard directory `dir` (`model.safetensors`,
/// `gepard_config.json`, `tokenizer.json`, `nano_dec_1.89kbps.safetensors`)
/// as the package `out`.
pub fn export_rlxp(dir: &Path, out: &Path, opts: &ExportOptions) -> Result<ExportReport> {
    let cfg = GepardConfig::from_path(dir)?;
    let hidden = cfg.hidden_size();
    let mut pkg = Package::new(
        "gepard-1.0",
        &opts.version,
        "gepard",
        "text-to-speech",
        "rlx-gepard",
    );
    pkg.model["license"] = json!("apache-2.0");

    // The backbone and heads, their weights referenced.
    let (p, lifted) = symbolic_in_lifted_many(
        &[(PROMPT, vec![5, 7, 3]), (CAP, vec![37, 53, 20])],
        &[MAX_CONTEXT, MAX_CONTEXT],
        |n| Ok(prefill(&cfg, n[0], n[1])),
    )?;
    anyhow::ensure!(lifted.is_empty(), "the prompt graph lifted tables");
    pkg.function(p);
    pkg.function(symbolic_in(CAP, (37, 53, 20), |n| Ok(step(&cfg, n)))?);
    let st = dir.join("model.safetensors");
    let source = WeightMap::from_file(st.to_str().context("weights path")?)?;
    let mut params: HashMap<String, Vec<f32>> = source
        .keys()
        .filter(|k| {
            k.starts_with("model.")
                || k.starts_with("codebook_heads.")
                || k.starts_with("stop_head.")
                || k.starts_with("audio_embed")
        })
        .filter(|k| *k != "audio_embed_scale")
        .map(|k| (k.to_string(), source.get(k).expect("listed").0.to_vec()))
        .collect();
    let scale = source
        .get("audio_embed_scale")
        .context("no audio_embed_scale")?
        .0
        .to_vec();

    // The decoder, symbolic in the frames, its params held.
    let nano =
        load_gepard_nano_weights(&std::fs::read(dir.join("nano_dec_1.89kbps.safetensors"))?)?;
    let build = |t: usize| -> Result<(Graph, HashMap<String, Vec<f32>>)> {
        let (mut g, p, _) = build_gepard_graph(&nano, t)?;
        g.name = "decode".into();
        Ok((g, p.into_iter().collect()))
    };
    let decode = symbolic_in(FRAMES, (5, 7, 3), |t| Ok(build(t)?.0))?;
    let (at, mut dec_params) = build(3)?;
    let decode = broadcast_params(&decode, &at, &mut dec_params, &[])?;
    let mut decode = decode;
    for node in decode.nodes_mut() {
        if let Op::Param { name } = &mut node.op {
            *name = format!("decode.{name}");
        }
    }
    pkg.function(decode);
    let mut held: HashMap<String, Vec<f32>> = dec_params
        .into_iter()
        .map(|(k, v)| (format!("decode.{k}"), v))
        .collect();
    let (cos, sin) = rope_tables(
        cfg.backbone.rope_theta as f32,
        cfg.backbone.effective_head_dim(),
    );
    held.insert("rope.cos".into(), cos);
    held.insert("rope.sin".into(), sin);
    for w in [cfg.backbone.effective_head_dim(), hidden] {
        held.insert(format!("gepard.zero.{w}"), vec![0.0; w]);
    }
    held.insert("gepard.one".into(), vec![1.0; hidden]);
    held.insert("gepard.audio_embed_scale".into(), scale);
    pkg.dims = vec![
        ("frames".into(), FRAMES),
        ("prompt".into(), PROMPT),
        ("cap".into(), CAP),
    ];
    let dims = pkg.param_dims();
    let mut names: Vec<&String> = held.keys().collect();
    names.sort();
    for n in names {
        let d = dims.get(n).cloned().unwrap_or_else(|| vec![held[n].len()]);
        pkg.hold_f32(n.clone(), d, &held[n]);
    }
    pkg.reference_map(&params, &source, "safetensors", &st, out, false)?;
    params.clear();

    pkg.sidecar(
        "tokenizer.json",
        "application/json",
        std::fs::read(dir.join("tokenizer.json"))?,
    );
    let sp = &cfg.special_tokens;
    let tl = &cfg.text_layout;
    let module = program::build(&Path::new(env!("CARGO_MANIFEST_DIR")).join("program/Cargo.toml"))
        .context("build Gepard's program")?;
    program::entry(
        &mut pkg,
        "speak",
        module,
        &["synthesize", "run"],
        "text-to-speech",
        json!({
            "tokenizer": "sidecar:tokenizer.json",
            "layers": cfg.backbone.num_hidden_layers,
            "start_of_text": sp.start_of_text,
            "end_of_text": sp.end_of_text,
            "start_of_speech": sp.start_of_speech,
            "repeat": {
                "enabled": tl.enabled,
                "target_text_tokens": tl.target_text_tokens,
                "apply_below": tl.apply_below,
                "max_repeats": tl.max_repeats,
            },
            "vocabs": cfg.codec.channel_vocabs(),
            "levels": FSQ_LEVELS,
            "max_context": MAX_CONTEXT,
            "max_frames": 2000,
            "min_frames": 8,
            "frame_rate": 21.5,
            "stop_threshold": 0.5,
            "temperature": 0.4,
            "seed": 54,
            "repetition_penalty": 1.0,
            "repetition_window": 32,
            "sample_rate": SAMPLE_RATE,
        }),
    );
    pkg.metadata("sample_rate", json!(SAMPLE_RATE));
    pkg.write(out, opts)
}

/// Register this crate's exporter ([`rlx_core::export::register_exporter`]).
pub fn register() {
    use rlx_core::export::{Exporter, register_exporter};
    register_exporter(Exporter::new("gepard", "text-to-speech", export_rlxp));
}
