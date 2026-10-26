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

//! Export Voxtral-4B-TTS as an `.rlxp` package rlx runs alone: text and a
//! preset voice in, 24 kHz speech out, through the one exporter
//! ([`rlx_core::export`]).
//!
//! Its graphs, as [`crate::NativeTtsEngine`] computes: `embed` (the prompt's
//! token rows; the program puts the voice's rows at its audio slots);
//! `prefill` (the Ministral backbone over the prompt's rows, symbolic in
//! their number and the caches' rows; its last hidden state) and `step` (one
//! frame's 37 codes embedded — each codebook's row, summed — through the
//! backbone after the caches, its row written in place); `acoustic` (the
//! semantic head's logits, and the flow-matching transformer's 7 Euler steps
//! from the given noise, conditioned and unconditioned rows mixed by
//! `cfg_alpha`); `codec` (the audio tokenizer's decoder over the frames'
//! codes, symbolic in their number: the semantic codebook's rows and the
//! rescaled acoustic levels, causal convs, sliding-window ALiBi
//! transformers, the patch projection). The backbone, embeddings, heads and
//! codec transformers are the checkpoint's tensors, referenced. The program
//! (`program/`) tokenizes (`kitoken` over `tekken.json`), builds the prompt,
//! draws each frame's noise (the runner's `StdRng` stream), picks the
//! semantic code, quantizes the acoustic ones, stops at end of audio and
//! decodes.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use rlx_core::audio_ops_ir::{AudioGraph, Conv1d, ConvTranspose1d, PadMode, finish_graph};
use rlx_core::export::graph::{symbolic_in, symbolic_in_lifted, symbolic_in_lifted_many};
use rlx_core::export::{ExportOptions, ExportReport, Package, program};
use rlx_core::weight_map::WeightMap;
use rlx_ir::hir::{HirGraphExt, HirModule, HirMut, HirNodeId};
use rlx_ir::op::{CmpOp, MaskKind};
use rlx_ir::{DType, Graph, GraphExt, NodeId, Op, RopeStyle, Shape};
use serde_json::json;

use crate::config::{CodecArgs, VoxtralTtsConfig};
use crate::tokens::{
    AUDIO_TOKEN_OFFSET, DEFAULT_CFG_ALPHA, DEFAULT_EULER_STEPS, END_AUDIO, PRESET_VOICES,
};

/// The symbols: the codec's frames, the prompt's rows, the caches' rows.
const FRAMES: u32 = 0;
const PROMPT: u32 = 2;
const CAP: u32 = 3;
/// RoPE rows: the longest context a request reaches.
const MAX_CONTEXT: usize = 4096;
const N_CODEBOOKS: usize = 37;
/// The codec builder's tensors this long and longer are weights (its convs,
/// the semantic table); shorter ones are its own tables, baked.
const WEIGHT_MIN: usize = 1 << 18;

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

/// `x @ wᵀ` for a `[out, in]` weight.
fn linear(g: &mut Graph, x: NodeId, w: &str, out: usize, inp: usize) -> NodeId {
    let wp = g.param(w, f32s(&[out, inp]));
    let wt = g.transpose_(wp, vec![1, 0]);
    g.mm(x, wt)
}

fn norm(g: &mut Graph, x: NodeId, w: &str, width: usize, eps: f32) -> NodeId {
    let wp = g.param(w, f32s(&[width]));
    let zero = g.param(format!("voxtral.zero.{width}"), f32s(&[width]));
    g.rms_norm(x, wp, zero, eps)
}

/// `[1, rows, kv]` with each key / value head repeated for its group.
fn repeat_kv(
    g: &mut Graph,
    x: NodeId,
    rows: usize,
    kv_heads: usize,
    group: usize,
    hd: usize,
) -> NodeId {
    if group == 1 {
        return x;
    }
    let x = g.reshape_(x, vec![1, rows as i64, kv_heads as i64, 1, hd as i64]);
    let x = g.add_node(
        Op::Expand {
            target_shape: vec![1, rows as i64, kv_heads as i64, group as i64, hd as i64],
        },
        vec![x],
        f32s(&[1, rows, kv_heads, group, hd]),
    );
    g.reshape_(x, vec![1, rows as i64, (kv_heads * group * hd) as i64])
}

/// A step's cache read: its position and the keys it may see.
struct StepAt {
    pos: NodeId,
    mask: NodeId,
}

/// The Ministral backbone over `x` `[1, t, hidden]` at RoPE rows `cos` /
/// `sin` `[t, head_dim / 2]`, its caches `cap` rows (a prompt's written over
/// zeros from row 0, a step's into `k{i}` / `v{i}` at `pos`). The final
/// norm's output and the caches.
#[allow(clippy::too_many_arguments)]
fn backbone(
    g: &mut Graph,
    cfg: &crate::config::TextConfig,
    mut x: NodeId,
    cos: NodeId,
    sin: NodeId,
    t: usize,
    cap: usize,
    step: Option<&StepAt>,
) -> (NodeId, Vec<NodeId>) {
    let (hidden, heads, kv_heads, hd) = (
        cfg.hidden_size,
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.head_dim,
    );
    let inter = cfg.intermediate_size.unwrap_or(hidden * 3);
    let eps = cfg.rms_norm_eps as f32;
    let (q_dim, kv_dim) = (heads * hd, kv_heads * hd);
    let cache = f32s(&[1, cap, kv_dim]);
    let origin = scalar(g, 0.0, 1);
    let nought = scalar(g, 0.0, 3);
    let mut caches = Vec::new();
    for i in 0..cfg.num_hidden_layers {
        let p = format!("layers.{i}");
        let n1 = norm(g, x, &format!("{p}.attention_norm.weight"), hidden, eps);
        let q = linear(g, n1, &format!("{p}.attention.wq.weight"), q_dim, hidden);
        let q = g.rope_styled(q, cos, sin, hd, RopeStyle::GptJ);
        let k = linear(g, n1, &format!("{p}.attention.wk.weight"), kv_dim, hidden);
        let k = g.rope_styled(k, cos, sin, hd, RopeStyle::GptJ);
        let v = linear(g, n1, &format!("{p}.attention.wv.weight"), kv_dim, hidden);
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
        let group = heads / kv_heads;
        let a = match step {
            Some(s) => {
                let kr = repeat_kv(g, kv[0], cap, kv_heads, group, hd);
                let vr = repeat_kv(g, kv[1], cap, kv_heads, group, hd);
                g.attention(q, kr, vr, s.mask, heads, hd, f32s(&[1, t, q_dim]))
            }
            None => {
                let kr = repeat_kv(g, k, t, kv_heads, group, hd);
                let vr = repeat_kv(g, v, t, kv_heads, group, hd);
                g.attention_kind(q, kr, vr, heads, hd, MaskKind::Causal, f32s(&[1, t, q_dim]))
            }
        };
        let a = linear(g, a, &format!("{p}.attention.wo.weight"), hidden, q_dim);
        x = g.add(x, a);
        let n2 = norm(g, x, &format!("{p}.ffn_norm.weight"), hidden, eps);
        let gt = linear(g, n2, &format!("{p}.feed_forward.w1.weight"), inter, hidden);
        let gt = g.silu(gt);
        let up = linear(g, n2, &format!("{p}.feed_forward.w3.weight"), inter, hidden);
        let h = g.mul(gt, up);
        let m = linear(g, h, &format!("{p}.feed_forward.w2.weight"), hidden, inter);
        x = g.add(x, m);
    }
    (norm(g, x, "norm.weight", hidden, eps), caches)
}

const TOK: &str = "mm_audio_embeddings.tok_embeddings.weight";
const CODEBOOK: &str = "mm_audio_embeddings.audio_codebook_embeddings.embeddings.weight";

/// `embed`: `ids` `[t]` i32 → their rows `x` `[1, t, hidden]`.
fn embed(cfg: &VoxtralTtsConfig, t: usize) -> Graph {
    let tc = &cfg.text_config;
    let mut g = Graph::new("embed");
    let ids = g.input("ids", Shape::new(&[t], DType::I32));
    let table = g.param(TOK, f32s(&[tc.vocab_size, tc.hidden_size]));
    let x = g.gather_(table, ids, 0);
    let x = g.reshape_(x, vec![1, t as i64, tc.hidden_size as i64]);
    g.node_mut(x).name = Some("x".into());
    g.set_outputs(vec![x]);
    g
}

/// `prefill`: the prompt's rows `x` `[1, t, hidden]` → the last row's
/// hidden state `h` `[1, hidden]` and each layer's caches.
fn prefill(cfg: &VoxtralTtsConfig, t: usize, cap: usize) -> Graph {
    let tc = &cfg.text_config;
    let half = tc.head_dim / 2;
    let mut g = Graph::new("prefill");
    let x = g.input("x", f32s(&[1, t, tc.hidden_size]));
    let cos = g.param("rope.cos", f32s(&[MAX_CONTEXT, half]));
    let sin = g.param("rope.sin", f32s(&[MAX_CONTEXT, half]));
    let cos = g.narrow_(cos, 0, 0, t);
    let sin = g.narrow_(sin, 0, 0, t);
    let (xf, caches) = backbone(&mut g, tc, x, cos, sin, t, cap, None);
    let h = g.narrow_(xf, 1, t - 1, 1);
    let h = g.reshape_(h, vec![1, tc.hidden_size as i64]);
    g.node_mut(h).name = Some("h".into());
    let mut outs = vec![h];
    outs.extend(caches);
    g.set_outputs(outs);
    g
}

/// `step`: the last frame's `codes` `[37]` i32 (each codebook's id, its
/// special tokens counted) at `pos` `[1]`, the keys it sees `mask`
/// `[1, cap]` → its hidden state `h` and the caches with its row written.
fn step(cfg: &VoxtralTtsConfig, cap: usize, rows: usize) -> Graph {
    let tc = &cfg.text_config;
    let half = tc.head_dim / 2;
    let hidden = tc.hidden_size;
    let mut g = Graph::new("step");
    let codes = g.input("codes", Shape::new(&[N_CODEBOOKS], DType::I32));
    let at = g.input("pos", f32s(&[1]));
    let mask = g.input("mask", f32s(&[1, cap]));
    let codes = g.add_node(
        Op::Cast { to: DType::F32 },
        vec![codes],
        f32s(&[N_CODEBOOKS]),
    );
    let offsets = g.param("codebook.offsets", f32s(&[N_CODEBOOKS]));
    let ids = g.add(codes, offsets);
    let ids = g.add_node(
        Op::Cast { to: DType::I32 },
        vec![ids],
        Shape::new(&[N_CODEBOOKS], DType::I32),
    );
    let table = g.param(CODEBOOK, f32s(&[rows, hidden]));
    let picked = g.gather_(table, ids, 0);
    let x = g.sum(picked, vec![0], false);
    let x = g.reshape_(x, vec![1, 1, hidden as i64]);
    let cos = g.param("rope.cos", f32s(&[MAX_CONTEXT, half]));
    let sin = g.param("rope.sin", f32s(&[MAX_CONTEXT, half]));
    let cos = g.gather_(cos, at, 0);
    let sin = g.gather_(sin, at, 0);
    let s = StepAt { pos: at, mask };
    let (xf, caches) = backbone(&mut g, tc, x, cos, sin, 1, cap, Some(&s));
    let h = g.reshape_(xf, vec![1, hidden as i64]);
    g.node_mut(h).name = Some("h".into());
    let mut outs = vec![h];
    outs.extend(caches);
    g.set_outputs(outs);
    g
}

/// The flow-matching transformer's velocity for `x_t` `[2, 36]` beside the
/// projected time `t_proj` `[1, dim]` and hidden states `llm` `[2, dim]`.
fn velocity(
    g: &mut Graph,
    cfg: &VoxtralTtsConfig,
    x_t: NodeId,
    t_proj: NodeId,
    llm: NodeId,
) -> NodeId {
    let a = &cfg.audio_config.audio_model_args;
    let at = &a.acoustic_transformer_args;
    let (dim, n) = (at.dim, a.n_acoustic_codebook);
    let eps = at.norm_eps as f32;
    let pre = "acoustic_transformer.";
    let xp = linear(g, x_t, &format!("{pre}input_projection.weight"), dim, n);
    let xp = g.reshape_(xp, vec![2, 1, dim as i64]);
    let tp = g.reshape_(t_proj, vec![1, 1, dim as i64]);
    let tp = g.concat_(vec![tp, tp], 0);
    let lp = g.reshape_(llm, vec![2, 1, dim as i64]);
    let mut h = g.concat_(vec![xp, tp, lp], 1); // [2, 3, dim]
    let (heads, kv_heads, hd) = (at.n_heads, at.n_kv_heads, at.head_dim);
    for i in 0..at.n_layers {
        let p = format!("{pre}layers.{i}");
        let n1 = norm(g, h, &format!("{p}.attention_norm.weight"), dim, eps);
        let q = linear(g, n1, &format!("{p}.attention.wq.weight"), heads * hd, dim);
        let k = linear(
            g,
            n1,
            &format!("{p}.attention.wk.weight"),
            kv_heads * hd,
            dim,
        );
        let v = linear(
            g,
            n1,
            &format!("{p}.attention.wv.weight"),
            kv_heads * hd,
            dim,
        );
        let group = heads / kv_heads;
        let rep = |g: &mut Graph, x: NodeId| -> NodeId {
            let x = g.reshape_(x, vec![2, 3, kv_heads as i64, 1, hd as i64]);
            let x = g.add_node(
                Op::Expand {
                    target_shape: vec![2, 3, kv_heads as i64, group as i64, hd as i64],
                },
                vec![x],
                f32s(&[2, 3, kv_heads, group, hd]),
            );
            g.reshape_(x, vec![2, 3, (heads * hd) as i64])
        };
        let kr = rep(g, k);
        let vr = rep(g, v);
        let att = g.attention_kind(
            q,
            kr,
            vr,
            heads,
            hd,
            MaskKind::None,
            f32s(&[2, 3, heads * hd]),
        );
        let o = linear(g, att, &format!("{p}.attention.wo.weight"), dim, heads * hd);
        h = g.add(h, o);
        let n2 = norm(g, h, &format!("{p}.ffn_norm.weight"), dim, eps);
        let gt = linear(
            g,
            n2,
            &format!("{p}.feed_forward.w1.weight"),
            at.hidden_dim,
            dim,
        );
        let gt = g.silu(gt);
        let up = linear(
            g,
            n2,
            &format!("{p}.feed_forward.w3.weight"),
            at.hidden_dim,
            dim,
        );
        let m = g.mul(gt, up);
        let m = linear(
            g,
            m,
            &format!("{p}.feed_forward.w2.weight"),
            dim,
            at.hidden_dim,
        );
        h = g.add(h, m);
    }
    let h = norm(g, h, &format!("{pre}norm.weight"), dim, eps);
    let first = g.narrow_(h, 1, 0, 1);
    let first = g.reshape_(first, vec![2, dim as i64]);
    linear(
        g,
        first,
        &format!("{pre}acoustic_codebook_output.weight"),
        n,
        dim,
    )
}

/// `acoustic`: the hidden state `h` `[1, hidden]`, the noise `x0` `[1, 36]`
/// and `cfg_alpha` `[1]` → the semantic head's `semantic` `[1, vocab]` and
/// the flow's `sampled` `[1, 36]` after the Euler steps (the time rows a
/// param, `acoustic.t_emb`).
fn acoustic(cfg: &VoxtralTtsConfig, semantic_vocab: usize, steps: usize) -> Graph {
    let a = &cfg.audio_config.audio_model_args;
    let at = &a.acoustic_transformer_args;
    let (dim, n) = (at.dim, a.n_acoustic_codebook);
    let pre = "acoustic_transformer.";
    let mut g = Graph::new("acoustic");
    let h = g.input("h", f32s(&[1, at.input_dim]));
    let x0 = g.input("x0", f32s(&[1, n]));
    let alpha = g.input("cfg_alpha", f32s(&[1]));
    let sem = linear(
        &mut g,
        h,
        &format!("{pre}semantic_codebook_output.weight"),
        semantic_vocab,
        dim,
    );
    g.node_mut(sem).name = Some("semantic".into());
    // The hidden state's projection and the unconditioned (zero) row's.
    let lp = linear(
        &mut g,
        h,
        &format!("{pre}llm_projection.weight"),
        dim,
        at.input_dim,
    );
    let zero = g.param(format!("voxtral.zero.row{dim}"), f32s(&[1, dim]));
    let llm = g.concat_(vec![lp, zero], 0);
    let t_emb = g.param("acoustic.t_emb", f32s(&[steps, dim]));
    let one = scalar(&mut g, 1.0, 1);
    let beta = g.sub(one, alpha);
    let mut sampled = x0;
    for i in 0..steps {
        let te = g.narrow_(t_emb, 0, i, 1);
        let tp = linear(
            &mut g,
            te,
            &format!("{pre}time_projection.weight"),
            dim,
            dim,
        );
        let xs = g.concat_(vec![sampled, sampled], 0);
        let v = velocity(&mut g, cfg, xs, tp, llm);
        let vc = g.narrow_(v, 0, 0, 1);
        let vu = g.narrow_(v, 0, 1, 1);
        let vc = g.mul(vc, alpha);
        let vu = g.mul(vu, beta);
        let v = g.add(vc, vu);
        let dt = (i + 1) as f32 / steps as f32 - i as f32 / steps as f32;
        let dt = scalar(&mut g, dt, 1);
        let dv = g.mul(v, dt);
        sampled = g.add(sampled, dv);
    }
    g.node_mut(sampled).name = Some("sampled".into());
    g.set_outputs(vec![sem, sampled]);
    g
}

/// The runner's `time_embed` at each Euler step's time: `[steps, dim]`.
fn time_rows(dim: usize, theta: f64, steps: usize) -> Vec<f32> {
    let half = dim / 2;
    let inv: Vec<f32> = (0..half)
        .map(|i| (-(i as f64) * theta.ln() / half as f64).exp() as f32)
        .collect();
    let mut out = Vec::with_capacity(steps * dim);
    for s in 0..steps {
        let t = s as f32 / steps as f32;
        let ang: Vec<f32> = inv.iter().map(|f| t * f).collect();
        out.extend(ang.iter().map(|a| a.cos()));
        out.extend(ang.iter().map(|a| a.sin()));
    }
    out
}

// ── codec ────────────────────────────────────────────────────────────────────

fn alibi_slopes(n: usize) -> Vec<f32> {
    let p2 = |m: usize| -> Vec<f32> {
        let r = 2f32.powf(-8.0 / m as f32);
        (0..m).map(|i| r.powi(i as i32)).collect()
    };
    if n.is_power_of_two() {
        p2(n)
    } else {
        let m = 1usize << n.ilog2();
        let mut s = p2(m);
        s.extend(p2(2 * m).into_iter().step_by(2).take(n - m));
        s
    }
}

/// `w = g · v / ‖v‖` per output row (torch `weight_norm`), or the plain
/// weight.
fn conv_weight(w: &WeightMap, prefix: &str) -> Result<(Vec<f32>, Vec<usize>)> {
    let gk = format!("{prefix}.conv.parametrizations.weight.original0");
    let vk = format!("{prefix}.conv.parametrizations.weight.original1");
    if let (Some((g, _)), Some((v, vs))) = (w.get(&gk), w.get(&vk)) {
        let out_ch = vs[0];
        let fan = v.len() / out_ch;
        let mut data = vec![0f32; v.len()];
        for oc in 0..out_ch {
            let row = &v[oc * fan..(oc + 1) * fan];
            let n = row.iter().map(|x| x * x).sum::<f32>().sqrt() + 1e-12;
            let s = g[oc] / n;
            for (d, x) in data[oc * fan..(oc + 1) * fan].iter_mut().zip(row) {
                *d = x * s;
            }
        }
        return Ok((data, vs.to_vec()));
    }
    let (d, s) = w
        .get(&format!("{prefix}.conv.weight"))
        .with_context(|| format!("no conv {prefix}"))?;
    Ok((d.to_vec(), s.to_vec()))
}

struct CodecWeights {
    first: Vec<f32>,
    ups: Vec<Vec<f32>>,
    out: Vec<f32>,
    semantic: Vec<f32>,
}

/// One sliding-window ALiBi transformer layer over `x` `[1, t, dim]`
/// (`iota` `[t]`), its weights the checkpoint's (`p.…`).
fn codec_layer(
    g: &mut HirMut,
    a: &CodecArgs,
    x: HirNodeId,
    p: &str,
    t: usize,
    bias: HirNodeId,
) -> HirNodeId {
    let (dim, heads, hd) = (a.dim, a.n_heads, a.head_dim);
    let lin = |g: &mut HirMut, x: HirNodeId, w: &str, out: usize, inp: usize| -> HirNodeId {
        let wp = g.param(format!("{p}.{w}"), f32s(&[out, inp]));
        let wt = g.transpose_(wp, vec![1, 0]);
        g.mm(x, wt)
    };
    let rms = |g: &mut HirMut, x: HirNodeId, w: &str, width: usize, eps: f32| -> HirNodeId {
        let wp = g.param(format!("{p}.{w}"), f32s(&[width]));
        let zero = g.param(format!("voxtral.zero.{width}"), f32s(&[width]));
        g.rms_norm(x, wp, zero, eps)
    };
    let h = rms(g, x, "attention_norm.weight", dim, 1e-2);
    let q = lin(g, h, "attention.wq.weight", heads * hd, dim);
    let k = lin(g, h, "attention.wk.weight", a.n_kv_heads * hd, dim);
    let v = lin(g, h, "attention.wv.weight", a.n_kv_heads * hd, dim);
    let q = rms(g, q, "attention.q_norm.weight", heads * hd, 1e-6);
    let k = rms(g, k, "attention.k_norm.weight", a.n_kv_heads * hd, 1e-6);
    let att = g.attention_bias(q, k, v, bias, heads, hd, f32s(&[1, t, heads * hd]));
    let o = lin(g, att, "attention.wo.weight", dim, heads * hd);
    let s = g.param(format!("{p}.attention_scale"), f32s(&[dim]));
    let o = g.mul(o, s);
    let x = g.add(x, o);
    let h2 = rms(g, x, "ffn_norm.weight", dim, 1e-2);
    let w1 = lin(g, h2, "feed_forward.w1.weight", a.hidden_dim, dim);
    let w1 = g.silu(w1);
    let w3 = lin(g, h2, "feed_forward.w3.weight", a.hidden_dim, dim);
    let m = g.mul(w1, w3);
    let ff = lin(g, m, "feed_forward.w2.weight", dim, a.hidden_dim);
    let s = g.param(format!("{p}.ffn_scale"), f32s(&[dim]));
    let ff = g.mul(ff, s);
    g.add(x, ff)
}

/// The causal sliding-window ALiBi bias `[1, heads, t, t]`: `slope · (k − q)`
/// for `q − window ≤ k ≤ q`, `-1e9` elsewhere.
fn alibi_bias(ag: &mut AudioGraph, heads: usize, t: usize, window: usize) -> HirNodeId {
    let iota = ag.param((0..t).map(|i| i as f32).collect(), &[t]);
    let rows = ag.g.reshape_(iota, vec![t as i64, 1]);
    let cols = ag.g.reshape_(iota, vec![1, t as i64]);
    let rows = ag.expand(rows, &[t, t]);
    let cols = ag.expand(cols, &[t, t]);
    let rel = ag.g.sub(cols, rows); // k − q
    let zero = ag.scalar(0.0, &[t, t]);
    let low = ag.scalar(-(window as f32), &[t, t]);
    let past = ag.g.add_node(
        Op::Compare(CmpOp::Le),
        vec![rel, zero],
        Shape::new(&[t, t], DType::Bool),
    );
    let near = ag.g.add_node(
        Op::Compare(CmpOp::Ge),
        vec![rel, low],
        Shape::new(&[t, t], DType::Bool),
    );
    let past =
        ag.g.add_node(Op::Cast { to: DType::F32 }, vec![past], f32s(&[t, t]));
    let near =
        ag.g.add_node(Op::Cast { to: DType::F32 }, vec![near], f32s(&[t, t]));
    let ok = ag.g.mul(past, near);
    let one = ag.scalar(1.0, &[t, t]);
    let off = ag.g.sub(ok, one);
    let big = ag.scalar(1e9, &[t, t]);
    let pen = ag.g.mul(off, big);
    let rel = ag.g.reshape_(rel, vec![1, 1, t as i64, t as i64]);
    let rel = ag.expand(rel, &[1, heads, t, t]);
    let slopes = ag.param(alibi_slopes(heads), &[1, heads, 1, 1]);
    let slopes = ag.expand(slopes, &[1, heads, t, t]);
    let b = ag.g.mul(rel, slopes);
    let pen = ag.g.reshape_(pen, vec![1, 1, t as i64, t as i64]);
    let pen = ag.expand(pen, &[1, heads, t, t]);
    ag.g.add(b, pen)
}

/// `codec`: `semantic` `[f]` i32 (codebook ids) and `acoustic` `[f, 36]`
/// (levels) → `audio` `[f · 1920]`. Its convs' weights (weight-normed) and
/// the semantic table are the graph's own params; the transformer layers'
/// the checkpoint's (`audio_tokenizer.…`).
fn codec(a: &CodecArgs, w: &CodecWeights, f: usize) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    let mut hir = HirModule::new("codec");
    let mut g = HirMut::new(&mut hir);
    let mut ag = AudioGraph::new(&mut g);
    let (sd, ad) = (a.semantic_dim, a.acoustic_dim);
    let sem =
        ag.g.input("semantic".to_string(), Shape::new(&[f], DType::I32));
    let ac = ag.input("acoustic", &[f, ad]);
    let table = ag.param(w.semantic.clone(), &[w.semantic.len() / sd, sd]);
    let se = ag.g.gather_(table, sem, 0); // [f, 256]
    let two = ag.scalar(2.0 / (a.acoustic_codebook_size as f32 - 1.0), &[f, ad]);
    let one = ag.scalar(1.0, &[f, ad]);
    let av = ag.g.mul(ac, two);
    let av = ag.g.sub(av, one);
    let emb = ag.g.concat_(vec![se, av], 1); // [f, 292]
    let emb = ag.g.transpose_(emb, vec![1, 0]);
    let latent = sd + ad;
    let mut x = ag.g.reshape_(emb, vec![1, latent as i64, f as i64, 1]);
    let kernels = a.decoder_convs_kernels();
    let strides = a.decoder_convs_strides();
    let lengths = a.decoder_transformer_lengths();
    let halvings = a.encoder_convs_strides().iter().filter(|&&s| s > 1).count();
    let mut window = a.attn_sliding_window_size >> halvings;
    let (y, _, mut t) = ag.conv1d(
        x,
        f,
        &Conv1d {
            weight: &w.first,
            bias: None,
            c_out: a.dim,
            c_in: latent,
            k: kernels[0],
            stride: strides[0],
            dilation: 1,
            groups: 1,
            pad_left: kernels[0] - strides[0],
            pad_right: 0,
            pad_mode: PadMode::Replicate,
        },
    );
    x = y;
    if strides[0] > 1 {
        window *= 2;
    }
    let mut bi = 1;
    let mut up = 0;
    for (idx, &n) in lengths.iter().enumerate() {
        let h = ag.g.reshape_(x, vec![a.dim as i64, t as i64]);
        let h = ag.g.transpose_(h, vec![1, 0]);
        let mut h = ag.g.reshape_(h, vec![1, t as i64, a.dim as i64]);
        let bias = alibi_bias(&mut ag, a.n_heads, t, window);
        for li in 0..n {
            h = codec_layer(
                ag.g,
                a,
                h,
                &format!("audio_tokenizer.decoder_blocks.{bi}.layers.{li}"),
                t,
                bias,
            );
        }
        let h = ag.g.reshape_(h, vec![t as i64, a.dim as i64]);
        let h = ag.g.transpose_(h, vec![1, 0]);
        x = ag.g.reshape_(h, vec![1, a.dim as i64, t as i64, 1]);
        bi += 1;
        if idx + 1 != lengths.len() && (kernels[idx + 1] != 1 || strides[idx + 1] != 1) {
            let (k, s) = (kernels[idx + 1], strides[idx + 1]);
            let (y, _, tu) = ag.conv_transpose1d(
                x,
                t,
                &ConvTranspose1d {
                    weight: &w.ups[up],
                    bias: None,
                    c_in: a.dim,
                    c_out_per_group: a.dim,
                    k,
                    stride: s,
                    groups: 1,
                    trim_left: 0,
                    trim_right: k - s,
                },
            );
            x = y;
            t = tu;
            up += 1;
            bi += 1;
            if s > 1 {
                window *= 2;
            }
        }
    }
    let kp = a.patch_proj_kernel_size;
    let patch = a.pretransform_patch_size;
    let (y, _, t) = ag.conv1d(
        x,
        t,
        &Conv1d {
            weight: &w.out,
            bias: None,
            c_out: patch,
            c_in: a.dim,
            k: kp,
            stride: 1,
            dilation: 1,
            groups: 1,
            pad_left: kp - 1,
            pad_right: 0,
            pad_mode: PadMode::Reflect,
        },
    );
    let y = ag.g.reshape_(y, vec![patch as i64, t as i64]);
    let y = ag.g.transpose_(y, vec![1, 0]);
    let audio = ag.g.reshape_(y, vec![(t * patch) as i64]);
    let mut params: HashMap<String, Vec<f32>> =
        std::mem::take(&mut ag.params).into_iter().collect();
    let graph = finish_graph(hir, audio)?;
    // The builder's own small tensors (iotas, zero pads, scalars, slopes)
    // constants, so the fit makes those as long as a length symbolic; its
    // weights (the semantic table, the convs) params, renamed `codec.w<n>`.
    let built = |p: &str| {
        p.strip_prefix('w')
            .is_some_and(|n| n.parse::<usize>().is_ok())
    };
    let mut graph = rlx_ir::stateful::rebuild(&graph, |out, _, node, _| {
        let Op::Param { name: p } = &node.op else {
            return Ok(None);
        };
        if !built(p) {
            return Ok(None);
        }
        let data = params.get(p).ok_or_else(|| format!("no data for {p}"))?;
        if data.len() >= WEIGHT_MIN {
            return Ok(Some(out.param(format!("codec.{p}"), node.shape.clone())));
        }
        Ok(Some(out.add_node(
            Op::Constant {
                data: data.iter().flat_map(|x| x.to_le_bytes()).collect(),
            },
            vec![],
            node.shape.clone(),
        )))
    })
    .map_err(anyhow::Error::msg)?;
    params.retain(|k, v| built(k) && v.len() >= WEIGHT_MIN);
    let params: HashMap<String, Vec<f32>> = params
        .into_iter()
        .map(|(k, v)| (format!("codec.{k}"), v))
        .collect();
    graph.name = "codec".into();
    let out = graph.outputs[0];
    graph.node_mut(out).name = Some("audio".into());
    Ok((graph, params))
}

/// Export the Voxtral-4B-TTS directory `dir` (`consolidated.safetensors`,
/// `params.json`, `tekken.json`, `voice_embedding/*.f32`) as the package
/// `out`.
pub fn export_rlxp(dir: &Path, out: &Path, opts: &ExportOptions) -> Result<ExportReport> {
    let cfg = VoxtralTtsConfig::from_model_dir(dir)?;
    let tc = &cfg.text_config;
    let am = &cfg.audio_config.audio_model_args;
    let ca = &cfg.audio_config.codec_args;
    let steps = am
        .acoustic_transformer_args
        .n_decoding_steps
        .unwrap_or(DEFAULT_EULER_STEPS);
    let mut pkg = Package::new(
        "voxtral-4b-tts",
        &opts.version,
        "voxtral-tts",
        "text-to-speech",
        "rlx-voxtral-tts",
    );
    pkg.model["license"] = json!("cc-by-nc-4.0");
    let st = dir.join("consolidated.safetensors");
    let source = WeightMap::from_file(st.to_str().context("weights path")?)?;
    let dims = |k: &str| -> Result<Vec<usize>> {
        Ok(source.get(k).with_context(|| format!("no {k}"))?.1.to_vec())
    };
    let semantic_vocab = dims("acoustic_transformer.semantic_codebook_output.weight")?[0];
    let cb_rows = dims(CODEBOOK)?[0];

    pkg.function(symbolic_in(PROMPT, (5, 7, 3), |t| Ok(embed(&cfg, t)))?);
    let (p, lifted) = symbolic_in_lifted_many(
        &[(PROMPT, vec![5, 7, 3]), (CAP, vec![37, 53, 20])],
        &[MAX_CONTEXT, MAX_CONTEXT],
        |n| Ok(prefill(&cfg, n[0], n[1])),
    )?;
    anyhow::ensure!(lifted.is_empty(), "the prompt graph lifted tables");
    pkg.function(p);
    pkg.function(symbolic_in(CAP, (37, 53, 20), |n| {
        Ok(step(&cfg, n, cb_rows))
    })?);
    pkg.function(acoustic(&cfg, semantic_vocab, steps));

    // The codec, symbolic in the frames.
    let first = conv_weight(&source, "audio_tokenizer.decoder_blocks.0")?.0;
    let mut ups = Vec::new();
    let lengths = ca.decoder_transformer_lengths();
    let (kernels, strides) = (ca.decoder_convs_kernels(), ca.decoder_convs_strides());
    let mut bi = 1;
    for idx in 0..lengths.len() {
        bi += 1;
        if idx + 1 != lengths.len() && (kernels[idx + 1] != 1 || strides[idx + 1] != 1) {
            ups.push(conv_weight(&source, &format!("audio_tokenizer.decoder_blocks.{bi}"))?.0);
            bi += 1;
        }
    }
    let out_w = conv_weight(&source, "audio_tokenizer.output_proj")?.0;
    let (sum, _) = source
        .get("audio_tokenizer.quantizer.semantic_codebook.embedding_sum")
        .context("no semantic embedding_sum")?;
    let (usage, _) = source
        .get("audio_tokenizer.quantizer.semantic_codebook.cluster_usage")
        .context("no semantic cluster_usage")?;
    let sd = ca.semantic_dim;
    let semantic: Vec<f32> = sum
        .chunks(sd)
        .zip(usage.iter())
        .flat_map(|(row, &u)| row.iter().map(move |x| x / u.max(1e-5)).collect::<Vec<_>>())
        .collect();
    let cw = CodecWeights {
        first,
        ups,
        out: out_w,
        semantic,
    };
    let (cg, lifted) = symbolic_in_lifted(FRAMES, (5, 7, 3), 1024, |f| Ok(codec(ca, &cw, f)?.0))?;
    let (_, cparams) = codec(ca, &cw, 3)?;
    pkg.function(cg);

    let mut held: HashMap<String, Vec<f32>> = cparams;
    held.extend(lifted);
    let mut offsets = Vec::with_capacity(N_CODEBOOKS);
    let mut acc = 0usize;
    for c in 0..N_CODEBOOKS {
        offsets.push(acc as f32);
        acc += if c == 0 {
            am.semantic_codebook_size + 2
        } else {
            am.acoustic_codebook_size + 2
        };
    }
    held.insert("codebook.offsets".into(), offsets);
    let inv = crate::backbone::rope::build_inv_freq(tc.rope_theta, tc.head_dim);
    let (cos, sin) = crate::backbone::rope::build_rope_tables(&inv, MAX_CONTEXT);
    held.insert("rope.cos".into(), cos);
    held.insert("rope.sin".into(), sin);
    held.insert(
        "acoustic.t_emb".into(),
        time_rows(
            am.acoustic_transformer_args.dim,
            am.acoustic_transformer_args.time_theta(),
            steps,
        ),
    );
    for w in [
        tc.hidden_size,
        ca.dim,
        ca.n_heads * ca.head_dim,
        am.acoustic_transformer_args.dim,
    ] {
        held.insert(format!("voxtral.zero.{w}"), vec![0.0; w]);
    }
    held.insert(
        format!("voxtral.zero.row{}", am.acoustic_transformer_args.dim),
        vec![0.0; am.acoustic_transformer_args.dim],
    );
    pkg.dims = vec![
        ("frames".into(), FRAMES),
        ("prompt".into(), PROMPT),
        ("cap".into(), CAP),
    ];
    let pdims = pkg.param_dims();
    let mut names: Vec<&String> = held.keys().collect();
    names.sort();
    for n in names {
        let d = pdims.get(n).cloned().unwrap_or_else(|| vec![held[n].len()]);
        pkg.hold_f32(n.clone(), d, &held[n]);
    }
    drop(held);

    // Every other param the graphs read is a checkpoint tensor.
    let mut wanted: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for g in &pkg.graphs {
        for node in g.nodes() {
            if let Op::Param { name } = &node.op
                && source.get(name).is_some()
            {
                wanted.insert(name.clone());
            }
        }
    }
    let params: HashMap<String, Vec<f32>> = wanted
        .into_iter()
        .map(|k| {
            let d = source.get(&k).expect("listed").0.to_vec();
            (k, d)
        })
        .collect();
    pkg.reference_map(&params, &source, "safetensors", &st, out, false)?;
    drop(params);

    // The voices and the tokenizer, read by the program.
    let mut voices = serde_json::Map::new();
    for &v in PRESET_VOICES {
        let p = dir.join("voice_embedding").join(format!("{v}.f32"));
        if !p.is_file() {
            continue;
        }
        let bytes = std::fs::read(&p)?;
        voices.insert(v.into(), json!(bytes.len() / 4 / tc.hidden_size));
        pkg.sidecar(
            &format!("voices/{v}.f32"),
            "application/octet-stream",
            bytes,
        );
    }
    pkg.sidecar(
        "tekken.json",
        "application/json",
        std::fs::read(dir.join(crate::config::TEKKEN_FILE))?,
    );

    let module = program::build_for(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("program/Cargo.toml"),
        "wasm32-wasip1",
    )
    .context("build Voxtral TTS' program")?;
    program::entry(
        &mut pkg,
        "speak",
        module,
        &["synthesize", "run"],
        "text-to-speech",
        json!({
            "files": "/package",
            "voices": voices,
            "voice": "neutral_female",
            "layers": tc.num_hidden_layers,
            "hidden": tc.hidden_size,
            "kv_dim": tc.num_key_value_heads * tc.head_dim,
            "audio_token": 24,
            "begin_audio": 25,
            "text_to_audio": 36,
            "audio_to_text": 35,
            "bos": 1,
            "end_audio": END_AUDIO,
            "empty_audio": crate::tokens::EMPTY_AUDIO,
            "special": AUDIO_TOKEN_OFFSET,
            "semantic_limit": 2 + am.semantic_codebook_size,
            "acoustic_levels": am.acoustic_codebook_size,
            "n_acoustic": am.n_acoustic_codebook,
            "cfg_alpha": DEFAULT_CFG_ALPHA,
            "seed": 42,
            "max_frames": 2500,
            "max_context": MAX_CONTEXT,
            "sample_rate": ca.sampling_rate,
        }),
    );
    pkg.metadata("sample_rate", json!(ca.sampling_rate));
    pkg.write(out, opts)
}

/// Register this crate's exporter ([`rlx_core::export::register_exporter`]).
pub fn register() {
    use rlx_core::export::{Exporter, register_exporter};
    register_exporter(Exporter::new("voxtral-tts", "text-to-speech", export_rlxp));
}
