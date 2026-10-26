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

//! Export Qwen3-TTS CustomVoice (`Qwen/Qwen3-TTS-12Hz-*-CustomVoice`) as an
//! `.rlxp` package rlx runs alone: text in, a built-in speaker's 24 kHz
//! speech out.
//!
//! Its graphs: `text_rows` (the text embedding through `text_projection`)
//! and `codec_rows` (codec embedding rows), from which the program lays the
//! prompt out as [`crate::prompt::build_custom_voice_prompt`] does;
//! `prompt` (the talker over the prompt, its last row's hidden state and
//! each layer's keys and values at the start of caches of `cap` rows) and
//! `step` (one row at `position`, its state inside), the talker's RoPE the
//! text section's 1-D rows; `head` (the codec head); the code predictor as
//! `cp_prompt` (its two rows: the talker's hidden state and the first
//! code's embedding), `cp_step`, a head and an embedding a group
//! (`cp_head_{k}`, `cp_embed_{k}`); and `speech` (the 12 Hz tokenizer's
//! decoder: RVQ, the causal pre-convolution, the sliding-window
//! pre-transformer, the ConvNeXt upsampling, the SnakeBeta decoder),
//! symbolic in the frames. The program (`program/`) runs the codec loop as
//! [`crate::fused_e2e::codec_frame_step`] does: repetition penalty, greedy
//! first code (the reserved range masked but end of speech), the code
//! predictor's greedy groups, the frame's embedding sum, the talker step.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use rlx_core::audio_ops_ir::{AudioGraph, Conv1d, ConvTranspose1d, PadMode};
use rlx_core::export::graph::{broadcast_params, symbolic_in, symbolic_in_lifted_many};
use rlx_core::export::{ExportOptions, ExportReport, Package, program};
use rlx_core::flow_util::{built_from_hir, graph_from_built};
use rlx_core::weight_map::WeightMap;
use rlx_ir::hir::{HirModule, HirMut};
use rlx_ir::{DType, Graph, GraphExt, HirGraphExt, NodeId, Op, Shape};
use rlx_qwen3::Qwen3Config;
use serde_json::json;

use crate::codec_frame::{
    Qwen3TtsGraphProfiles, Qwen3TtsGraphRole, build_qwen3_tts_decode_built,
    build_qwen3_tts_prefill_built,
};
use crate::config::{Qwen3TtsConfig, TalkerConfig};
use crate::load::{Qwen3TtsWeightStore, remap_code_predictor_weights, remap_talker_weights};
use crate::weights::SnapshotLoader;

/// The symbols: the talker's caches, the code predictor's, the prompt's
/// rows, the speech frames, the embedding rows.
const CAP: u32 = 0;
const CP_CAP: u32 = 2;
const PROMPT: u32 = 3;
const FRAMES: u32 = 4;
const ROWS_SYM: u32 = 5;
/// Positions the talker's RoPE tables reach.
const ROWS: usize = 8192;
/// The code predictor's caches: its two prompt rows and the groups.
const CP_ROWS: usize = 20;

type Built = (Graph, HashMap<String, Vec<f32>>);

fn f32s(d: &[usize]) -> Shape {
    Shape::new(d, DType::F32)
}

fn name(g: &mut Graph, id: NodeId, n: &str) {
    g.node_mut(id).name = Some(n.into());
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

/// `g` with each param `p` named `{prefix}{p}`, and `params` likewise.
fn prefixed(g: &Graph, params: HashMap<String, Vec<f32>>, prefix: &str) -> Result<Built> {
    let g = rlx_ir::stateful::rebuild(g, |out, _, node, _| match &node.op {
        Op::Param { name } => Ok(Some(
            out.param(format!("{prefix}{name}"), node.shape.clone()),
        )),
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

/// `more` into `params`: a name both hold must be one tensor (the talker's
/// and the code predictor's weights share names, `model.layers.*`).
fn merge(params: &mut HashMap<String, Vec<f32>>, more: HashMap<String, Vec<f32>>) -> Result<()> {
    for (n, v) in more {
        if let Some(had) = params.get(&n) {
            anyhow::ensure!(had == &v, "two graphs' params named '{n}' differ");
        }
        params.insert(n, v);
    }
    Ok(())
}

/// The talker's 1-D RoPE rows for every position (`mrope::talker_rope_half`,
/// the text section: all three axes at the position).
fn talker_rope(talker: &TalkerConfig) -> (Vec<f32>, Vec<f32>) {
    let half = talker.head_dim / 2;
    let (mut cos, mut sin) = (
        Vec::with_capacity(ROWS * half),
        Vec::with_capacity(ROWS * half),
    );
    for p in 0..ROWS {
        let (c, s) = crate::mrope::talker_rope_half(talker, p, half);
        cos.extend(c);
        sin.extend(s);
    }
    (cos, sin)
}

/// `rows`: `ids` `[n]` i32 → `rows` `[n, h]`: the codec embedding's rows
/// (`codec`) or the text embedding through `text_projection` (`text`).
fn rows(talker: &TalkerConfig, text: bool, n: usize, text_hidden: usize) -> Graph {
    let h = talker.hidden_size;
    let mut g = Graph::new(if text { "text_rows" } else { "codec_rows" });
    let ids = g.input("ids", Shape::new(&[n], DType::I32));
    let out = if text {
        let t = g.param(
            "talker.model.text_embedding.weight",
            f32s(&[talker.text_vocab_size, text_hidden]),
        );
        let x = g.gather_(t, ids, 0);
        let lin = |g: &mut Graph, x: NodeId, key: &str, o: usize, i: usize| {
            let w = g.param(format!("{key}.weight"), f32s(&[o, i]));
            let w = g.transpose_(w, vec![1, 0]);
            let y = g.mm(x, w);
            let b = g.param(format!("{key}.bias"), f32s(&[o]));
            let b = g.reshape_(b, vec![1, o as i64]);
            g.add(y, b)
        };
        let x = lin(
            &mut g,
            x,
            "talker.text_projection.linear_fc1",
            text_hidden,
            text_hidden,
        );
        let x = g.silu(x);
        lin(
            &mut g,
            x,
            "talker.text_projection.linear_fc2",
            h,
            text_hidden,
        )
    } else {
        let t = g.param(
            "talker.model.codec_embedding.weight",
            f32s(&[talker.vocab_size, h]),
        );
        g.gather_(t, ids, 0)
    };
    name(&mut g, out, "rows");
    g.set_outputs(vec![out]);
    g
}

/// `head`: `hidden` `[1, 1, h]` → `logits` `[1, vocab]` (the codec head).
fn head(talker: &TalkerConfig) -> Graph {
    let h = talker.hidden_size;
    let mut g = Graph::new("head");
    let x = g.input("hidden", f32s(&[1, 1, h]));
    let x = g.reshape_(x, vec![1, h as i64]);
    let w = g.param("talker.codec_head.weight", f32s(&[talker.vocab_size, h]));
    let w = g.transpose_(w, vec![1, 0]);
    let y = g.mm(x, w);
    name(&mut g, y, "logits");
    g.set_outputs(vec![y]);
    g
}

/// `cp_head_{k}` / `cp_embed_{k}`: the code predictor's group-`k` head
/// (`hidden` → `logits`) and embedding (`token` `[1]` i32 → `row` `[1, 1,
/// h]`).
fn cp_group(h: usize, vocab: usize, k: usize, embed: bool) -> Graph {
    let mut g = Graph::new(&format!("cp_{}_{k}", if embed { "embed" } else { "head" }));
    if embed {
        let t = g.input("token", Shape::new(&[1], DType::I32));
        let w = g.param(
            format!("talker.code_predictor.model.codec_embedding.{k}.weight"),
            f32s(&[vocab, h]),
        );
        let r = g.gather_(w, t, 0);
        let r = g.reshape_(r, vec![1, 1, h as i64]);
        name(&mut g, r, "row");
        g.set_outputs(vec![r]);
    } else {
        let x = g.input("hidden", f32s(&[1, 1, h]));
        let x = g.reshape_(x, vec![1, h as i64]);
        let w = g.param(
            format!("talker.code_predictor.lm_head.{k}.weight"),
            f32s(&[vocab, h]),
        );
        let w = g.transpose_(w, vec![1, 0]);
        let y = g.mm(x, w);
        name(&mut g, y, "logits");
        g.set_outputs(vec![y]);
    }
    g
}

/// A Qwen3 prompt graph at `t` rows writing its keys and values into zero
/// caches of `cap` rows (`past_k_{l}`, `past_v_{l}`), its last row's
/// `hidden` `[1, 1, h]`.
fn prompt_graph(
    qwen3: &Qwen3Config,
    weights: &HashMap<String, (Vec<f32>, Vec<usize>)>,
    profile: &rlx_flow::CompileProfile,
    rope: Option<(Vec<f32>, Vec<f32>)>,
    t: usize,
    cap: usize,
    gname: &str,
) -> Result<Built> {
    let mut loader = SnapshotLoader::new(weights.clone());
    let (cos, sin) = match rope {
        Some((c, s)) => (Some(c), Some(s)),
        None => (None, None),
    };
    let built = build_qwen3_tts_prefill_built(qwen3, &mut loader, t, profile, cos, sin)?;
    let (mut g, params) = graph_from_built(built)?;
    let o = g.outputs.clone();
    let h = qwen3.hidden_size;
    let last = g.narrow_(o[0], 1, t - 1, 1);
    let last = g.reshape_(last, vec![1, 1, h as i64]);
    name(&mut g, last, "hidden");
    let kv = qwen3.num_key_value_heads * qwen3.head_dim;
    let origin = constant(&mut g, 0.0);
    let zero = constant(&mut g, 0.0);
    let zero = g.reshape_(zero, vec![1, 1, 1]);
    let mut outs = vec![last];
    for (i, &rowsn) in o[1..].iter().enumerate() {
        let z = g.add_node(
            Op::Expand {
                target_shape: vec![1, cap as i64, kv as i64],
            },
            vec![zero],
            f32s(&[1, cap, kv]),
        );
        let full = g.add_node(
            Op::DynamicUpdateSlice { axis: 1 },
            vec![z, rowsn, origin],
            f32s(&[1, cap, kv]),
        );
        let kind = if i % 2 == 0 { "k" } else { "v" };
        name(&mut g, full, &format!("past_{kind}_{}", i / 2));
        outs.push(full);
    }
    g.set_outputs(outs);
    g.name = gname.into();
    Ok((g, params))
}

/// A Qwen3 decode step from `inputs_embeds`, symbolic in its caches
/// (`cap_sym`), with its state inside: RoPE rows from `tables` at
/// `position`, the mask from it, each key and value written there.
fn step_graph(
    qwen3: &Qwen3Config,
    weights: &HashMap<String, (Vec<f32>, Vec<usize>)>,
    profile: &rlx_flow::CompileProfile,
    cap_sym: u32,
    rows: usize,
    gname: &str,
) -> Result<Built> {
    let at = |upper: usize| -> Result<Built> {
        let mut loader = SnapshotLoader::new(weights.clone());
        let built = build_qwen3_tts_decode_built(qwen3, &mut loader, upper, profile)?;
        Ok(graph_from_built(built)?)
    };
    let (a, params) = at(36)?;
    let (b, _) = at(52)?;
    let g = rlx_ir::dynamic::generalize(&a, &b, &[(cap_sym, 37, 53), (cap_sym + 1, 36, 52)])
        .map_err(|e| anyhow::anyhow!("{gname} is not symbolic in its cache: {e}"))?;
    let spec = rlx_ir::stateful::StatefulDecode {
        position: "position".into(),
        tables: vec![
            ("rope_cos".into(), format!("{gname}.rope_cos.table"), rows),
            ("rope_sin".into(), format!("{gname}.rope_sin.table"), rows),
        ],
        mask: Some(("mask".into(), format!("{gname}.iota"), rows)),
        cache_symbol: cap_sym,
        tokens: 1,
        starts: None,
        rope_offset: None,
        windows: Vec::new(),
    };
    let mut g = rlx_ir::stateful::stateful_decode(&g, &spec)
        .map_err(|e| anyhow::anyhow!("{gname}'s state: {e}"))?;
    let o = g.outputs.clone();
    let h = qwen3.hidden_size;
    let hidden = g.reshape_(o[0], vec![1, 1, h as i64]);
    name(&mut g, hidden, "hidden");
    let mut outs = vec![hidden];
    for (i, &id) in o[1..].iter().enumerate() {
        let kind = if i % 2 == 0 { "k" } else { "v" };
        name(&mut g, id, &format!("past_{kind}_{}.next", i / 2));
        outs.push(id);
    }
    g.set_outputs(outs);
    g.name = gname.into();
    Ok((g, params))
}

/// The 12 Hz decoder's tensors.
struct Speech<'a> {
    w: &'a HashMap<String, (Vec<f32>, Vec<usize>)>,
}

impl Speech<'_> {
    fn d(&self, key: &str) -> Result<&[f32]> {
        Ok(&self
            .w
            .get(&format!("decoder.{key}"))
            .with_context(|| format!("no decoder.{key}"))?
            .0)
    }

    fn shape(&self, key: &str) -> Result<&[usize]> {
        Ok(&self
            .w
            .get(&format!("decoder.{key}"))
            .with_context(|| format!("no decoder.{key}"))?
            .1)
    }

    /// A causal conv (`{key}.conv.weight` / `.bias`) on `[1, c, t, 1]`.
    fn conv(
        &self,
        ag: &mut AudioGraph,
        x: NodeIdH,
        t: usize,
        key: &str,
        dilation: usize,
        groups: usize,
    ) -> Result<(NodeIdH, usize)> {
        let s = self.shape(&format!("{key}.conv.weight"))?.to_vec();
        let (c_out, c_in_g, k) = (s[0], s[1], s[2]);
        let (y, c, _) = ag.conv1d(
            x,
            t,
            &Conv1d {
                weight: self.d(&format!("{key}.conv.weight"))?,
                bias: Some(self.d(&format!("{key}.conv.bias"))?),
                c_out,
                c_in: c_in_g * groups,
                k,
                stride: 1,
                dilation,
                groups,
                pad_left: (k - 1) * dilation,
                pad_right: 0,
                pad_mode: PadMode::Constant,
            },
        );
        Ok((y, c))
    }

    /// A transposed conv (stride `stride`, the last `k - stride` columns
    /// trimmed).
    fn up(
        &self,
        ag: &mut AudioGraph,
        x: NodeIdH,
        t: usize,
        key: &str,
        stride: usize,
    ) -> Result<(NodeIdH, usize, usize)> {
        let s = self.shape(&format!("{key}.conv.weight"))?.to_vec();
        let (c_in, c_out, k) = (s[0], s[1], s[2]);
        Ok(ag.conv_transpose1d(
            x,
            t,
            &ConvTranspose1d {
                weight: self.d(&format!("{key}.conv.weight"))?,
                bias: Some(self.d(&format!("{key}.conv.bias"))?),
                c_in,
                c_out_per_group: c_out,
                k,
                stride,
                groups: 1,
                trim_left: 0,
                trim_right: k.saturating_sub(stride),
            },
        ))
    }

    fn snake(
        &self,
        ag: &mut AudioGraph,
        x: NodeIdH,
        c: usize,
        t: usize,
        key: &str,
    ) -> Result<NodeIdH> {
        Ok(ag.snake_beta(
            x,
            c,
            t,
            self.d(&format!("{key}.alpha"))?,
            self.d(&format!("{key}.beta"))?,
        ))
    }
}

type NodeIdH = rlx_ir::hir::HirNodeId;

/// `speech` at `f` frames: `codes` `[16, f]` i32 and `positions` `[f]` i32
/// (`0..f`) → `audio` `[1920 · f]`, clamped to [−1, 1]: what
/// [`crate::speech_tokenizer::St12HzDecoder::decode`] does.
fn speech(
    w: &HashMap<String, (Vec<f32>, Vec<usize>)>,
    cfg: &crate::speech_tokenizer::decode::DecoderConfig,
    f: usize,
) -> Result<Built> {
    let s = Speech { w };
    let mut hir = HirModule::new("speech");
    let (out, params) = {
        let mut m = HirMut::new(&mut hir);
        let mut ag = AudioGraph::new(&mut m);
        let codes = ag.g.input("codes", Shape::new(&[16, f], DType::I32));
        let pos = ag.g.input("positions", Shape::new(&[f], DType::I32));
        // RVQ: each branch's codebooks summed, through its output projection.
        let vq = cfg.codebook_dim / 2;
        let mut total = None;
        for (branch, first, n) in [("rvq_first", 0usize, 1usize), ("rvq_rest", 1, 15)] {
            let mut acc = None;
            for l in 0..n {
                let p = format!("quantizer.{branch}.vq.layers.{l}._codebook");
                let sum = s.d(&format!("{p}.embedding_sum"))?;
                let usage = s.d(&format!("{p}.cluster_usage"))?;
                let rowsn = usage.len();
                let mut table = vec![0f32; rowsn * vq];
                for r in 0..rowsn {
                    let den = usage[r].max(1e-5);
                    for c in 0..vq {
                        table[r * vq + c] = sum[r * vq + c] / den;
                    }
                }
                let tp = ag.param(table, &[rowsn, vq]);
                let at = ag.g.narrow_(codes, 0, first + l, 1);
                let at = ag.g.reshape_(at, vec![f as i64]);
                let e = ag.g.gather_(tp, at, 0);
                acc = Some(match acc {
                    None => e,
                    Some(a) => ag.g.add(a, e),
                });
            }
            let proj = s.d(&format!("quantizer.{branch}.output_proj.weight"))?;
            let y = ag.linear(acc.expect("a codebook"), vq, cfg.codebook_dim, proj);
            total = Some(match total {
                None => y,
                Some(t) => ag.g.add(t, y),
            });
        }
        let x = ag.tc_to_ct(total.expect("two branches"), f, cfg.codebook_dim);
        let (x, c) = s.conv(&mut ag, x, f, "pre_conv", 1, 1)?;
        // The pre-transformer, time-major.
        let mut x = ag.ct_to_tc(x, c, f);
        let hd = cfg.head_dim;
        let nh = cfg.num_attention_heads;
        let hs = cfg.hidden_size;
        let inner = nh * hd;
        let pt = "pre_transformer";
        x = ag.linear_bias(
            x,
            f,
            c,
            hs,
            s.d(&format!("{pt}.input_proj.weight"))?,
            s.d(&format!("{pt}.input_proj.bias"))?,
        );
        // RoPE rows gathered at the positions; the sliding causal mask from
        // them (−1e9 times how far outside the window a key is: the
        // softmax gives it nothing, as the reference's −∞).
        let half = hd / 2;
        let steps = 8000;
        let (mut tc, mut ts) = (vec![0f32; steps * half], vec![0f32; steps * half]);
        for p in 0..steps {
            for i in 0..half {
                let inv = 1.0 / cfg.rope_theta.powf(2.0 * i as f64 / hd as f64);
                let a = p as f64 * inv;
                tc[p * half + i] = a.cos() as f32;
                ts[p * half + i] = a.sin() as f32;
            }
        }
        let ctab = ag.param(tc, &[steps, half]);
        let stab = ag.param(ts, &[steps, half]);
        let cos = ag.g.gather_(ctab, pos, 0);
        let sin = ag.g.gather_(stab, pos, 0);
        let cos = ag.g.reshape_(cos, vec![1, f as i64, half as i64]);
        let sin = ag.g.reshape_(sin, vec![1, f as i64, half as i64]);
        let cos = ag.expand(cos, &[nh, f, half]);
        let sin = ag.expand(sin, &[nh, f, half]);
        let pf =
            ag.g.add_node(Op::Cast { to: DType::F32 }, vec![pos], f32s(&[f]));
        let col = ag.g.reshape_(pf, vec![f as i64, 1]);
        let col = ag.expand(col, &[f, f]);
        let row = ag.g.reshape_(pf, vec![1, f as i64]);
        let row = ag.expand(row, &[f, f]);
        let d = ag.g.sub(col, row);
        let neg = ag.g.neg(d);
        let before = ag.g.relu(neg);
        let off = ag.scalar((cfg.sliding_window - 1) as f32, &[f, f]);
        let far = ag.g.sub(d, off);
        let far = ag.g.relu(far);
        let outside = ag.g.add(before, far);
        let big = ag.scalar(-1e9, &[f, f]);
        let mask = ag.g.mul(outside, big);
        let mask = ag.g.reshape_(mask, vec![1, f as i64, f as i64]);
        let mask = ag.expand(mask, &[nh, f, f]);
        let scale = ag.scalar(1.0 / (hd as f32).sqrt(), &[nh, f, f]);
        for l in 0..cfg.num_hidden_layers {
            let lp = format!("{pt}.layers.{l}");
            let h = ag.rms_norm(
                x,
                hs,
                s.d(&format!("{lp}.input_layernorm.weight"))?,
                cfg.rms_norm_eps,
            );
            let heads = |ag: &mut AudioGraph, key: &str| -> Result<NodeIdH> {
                let y = ag.linear(h, hs, inner, s.d(key)?);
                let y = ag.g.reshape_(y, vec![f as i64, nh as i64, hd as i64]);
                Ok(ag.g.transpose_(y, vec![1, 0, 2]))
            };
            let q = heads(&mut ag, &format!("{lp}.self_attn.q_proj.weight"))?;
            let k = heads(&mut ag, &format!("{lp}.self_attn.k_proj.weight"))?;
            let v = heads(&mut ag, &format!("{lp}.self_attn.v_proj.weight"))?;
            let rot = |ag: &mut AudioGraph, t: NodeIdH| {
                let a = ag.g.narrow_(t, 2, 0, half);
                let b = ag.g.narrow_(t, 2, half, half);
                let ac = ag.g.mul(a, cos);
                let bs = ag.g.mul(b, sin);
                let first = ag.g.sub(ac, bs);
                let as_ = ag.g.mul(a, sin);
                let bc = ag.g.mul(b, cos);
                let second = ag.g.add(as_, bc);
                ag.g.concat_(vec![first, second], 2)
            };
            let q = rot(&mut ag, q);
            let k = rot(&mut ag, k);
            let kt = ag.g.transpose_(k, vec![0, 2, 1]);
            let sc = ag.g.mm(q, kt);
            let sc = ag.g.mul(sc, scale);
            let sc = ag.g.add(sc, mask);
            let p = ag.g.sm(sc, -1);
            let o = ag.g.mm(p, v);
            let o = ag.g.transpose_(o, vec![1, 0, 2]);
            let o = ag.g.reshape_(o, vec![f as i64, inner as i64]);
            let o = ag.linear(o, inner, hs, s.d(&format!("{lp}.self_attn.o_proj.weight"))?);
            let o = ag.scale_rows(o, f, hs, s.d(&format!("{lp}.self_attn_layer_scale.scale"))?);
            x = ag.g.add(x, o);
            let h2 = ag.rms_norm(
                x,
                hs,
                s.d(&format!("{lp}.post_attention_layernorm.weight"))?,
                cfg.rms_norm_eps,
            );
            let ffn = cfg.intermediate_size;
            let gate = ag.linear(h2, hs, ffn, s.d(&format!("{lp}.mlp.gate_proj.weight"))?);
            let up = ag.linear(h2, hs, ffn, s.d(&format!("{lp}.mlp.up_proj.weight"))?);
            let gate = ag.silu(gate);
            let m = ag.g.mul(gate, up);
            let m = ag.linear(m, ffn, hs, s.d(&format!("{lp}.mlp.down_proj.weight"))?);
            let m = ag.scale_rows(m, f, hs, s.d(&format!("{lp}.mlp_layer_scale.scale"))?);
            x = ag.g.add(x, m);
        }
        x = ag.rms_norm(x, hs, s.d(&format!("{pt}.norm.weight"))?, cfg.rms_norm_eps);
        x = ag.linear_bias(
            x,
            f,
            hs,
            cfg.latent_dim,
            s.d(&format!("{pt}.output_proj.weight"))?,
            s.d(&format!("{pt}.output_proj.bias"))?,
        );
        let mut x = ag.tc_to_ct(x, f, cfg.latent_dim);
        let mut t = f;
        // The ConvNeXt upsampling.
        for (i, &ratio) in cfg.upsampling_ratios.iter().enumerate() {
            let (y, cy, ty) = s.up(&mut ag, x, t, &format!("upsample.{i}.0"), ratio)?;
            let (xx, cc, tt) = (y, cy, ty);
            let cn = format!("upsample.{i}.1");
            let (dw, _) = s.conv(&mut ag, xx, tt, &format!("{cn}.dwconv"), 1, cc)?;
            let seq = ag.ct_to_tc(dw, cc, tt);
            let seq = ag.layer_norm(
                seq,
                cc,
                s.d(&format!("{cn}.norm.weight"))?,
                s.d(&format!("{cn}.norm.bias"))?,
                1e-6,
            );
            let p1 = s.shape(&format!("{cn}.pwconv1.weight"))?[0];
            let seq = ag.linear_bias(
                seq,
                tt,
                cc,
                p1,
                s.d(&format!("{cn}.pwconv1.weight"))?,
                s.d(&format!("{cn}.pwconv1.bias"))?,
            );
            let seq = ag.gelu(seq);
            let seq = ag.linear_bias(
                seq,
                tt,
                p1,
                cc,
                s.d(&format!("{cn}.pwconv2.weight"))?,
                s.d(&format!("{cn}.pwconv2.bias"))?,
            );
            let seq = ag.scale_rows(seq, tt, cc, s.d(&format!("{cn}.gamma"))?);
            let seq = ag.tc_to_ct(seq, tt, cc);
            x = ag.g.add(xx, seq);
            t = tt;
        }
        // The decoder.
        let (y, mut c) = s.conv(&mut ag, x, t, "decoder.0", 1, 1)?;
        x = y;
        for (bi, &rate) in cfg.upsample_rates.iter().enumerate() {
            let p = format!("decoder.{}", bi + 1);
            let a = s.snake(&mut ag, x, c, t, &format!("{p}.block.0"))?;
            let (y, cy, ty) = s.up(&mut ag, a, t, &format!("{p}.block.1"), rate)?;
            x = y;
            c = cy;
            t = ty;
            for (ri, dil) in (2..=4).zip([1usize, 3, 9]) {
                let rp = format!("{p}.block.{ri}");
                let a = s.snake(&mut ag, x, c, t, &format!("{rp}.act1"))?;
                let (y, _) = s.conv(&mut ag, a, t, &format!("{rp}.conv1"), dil, 1)?;
                let a = s.snake(&mut ag, y, c, t, &format!("{rp}.act2"))?;
                let (y, _) = s.conv(&mut ag, a, t, &format!("{rp}.conv2"), 1, 1)?;
                x = ag.g.add(x, y);
            }
        }
        let last = cfg.upsample_rates.len() + 1;
        let a = s.snake(&mut ag, x, c, t, &format!("decoder.{last}"))?;
        let (y, _) = s.conv(&mut ag, a, t, &format!("decoder.{}", last + 1), 1, 1)?;
        let y = ag.clamp(y, 1, t, -1.0, 1.0);
        let y = ag.g.reshape_(y, vec![t as i64]);
        (y, std::mem::take(&mut ag.params))
    };
    hir.set_outputs(vec![out]);
    let (mut g, p) = graph_from_built(built_from_hir(hir, params.into_iter().collect())?)?;
    let o = g.outputs[0];
    name(&mut g, o, "audio");
    g.name = "speech".into();
    prefixed(&g, p, "speech.")
}

/// Export the Qwen3-TTS CustomVoice directory `dir` (`config.json`,
/// `model.safetensors`, `tokenizer.json`, `speech_tokenizer/`) as the
/// package `out`.
pub fn export_rlxp(dir: &Path, out: &Path, opts: &ExportOptions) -> Result<ExportReport> {
    let cfg = Qwen3TtsConfig::from_model_dir(dir)?;
    let talker = cfg.talker();
    let store = Qwen3TtsWeightStore::open(dir)?;
    let mut pkg = Package::new(
        "qwen3-tts-customvoice",
        &opts.version,
        "qwen3-tts",
        "text-to-speech",
        "rlx-qwen3-tts",
    );
    pkg.model["license"] = json!("apache-2.0");
    let mut params: HashMap<String, Vec<f32>> = HashMap::new();
    let text_hidden = store
        .tensor_snapshot(&["talker.model.text_embedding.weight"])?
        .get("talker.model.text_embedding.weight")
        .context("text embedding")?
        .1[1];

    // Embedding rows, the codec head, the code predictor's groups.
    for text in [true, false] {
        let (g, _) = symbolic_in_lifted_many(&[(ROWS_SYM, vec![7, 9, 5])], &[0], |n| {
            Ok(rows(talker, text, n[0], text_hidden))
        })?;
        pkg.function(g);
    }
    pkg.function(head(talker));
    let cpc = cfg.code_predictor();
    let groups = cpc.num_code_groups - 1;
    for k in 0..groups {
        pkg.function(cp_group(cpc.hidden_size, cpc.vocab_size, k, false));
        pkg.function(cp_group(cpc.hidden_size, cpc.vocab_size, k, true));
    }

    // The talker.
    let tq = talker.to_qwen3_config();
    let tw = remap_talker_weights(&mut store.load_talker_backbone()?)?;
    let tprof =
        Qwen3TtsGraphProfiles::for_role(dir, Qwen3TtsGraphRole::Talker, rlx_runtime::Device::Cpu);
    let mut tq_rows = tq.clone();
    tq_rows.max_position_embeddings = ROWS;
    let rope = talker_rope(talker);
    let (pg, lifted) = symbolic_in_lifted_many(
        &[(PROMPT, vec![37, 53, 20]), (CAP, vec![70, 101, 44])],
        &[0, 0],
        |n| {
            Ok(prompt_graph(
                &tq_rows,
                &tw,
                &tprof.prefill,
                Some(rope.clone()),
                n[0],
                n[1],
                "prompt",
            )?
            .0)
        },
    )?;
    anyhow::ensure!(lifted.is_empty(), "the talker prompt lifted tables");
    pkg.function(pg);
    merge(
        &mut params,
        prompt_graph(
            &tq_rows,
            &tw,
            &tprof.prefill,
            Some(rope.clone()),
            20,
            44,
            "prompt",
        )?
        .1,
    )?;
    let (sg, sp) = step_graph(&tq_rows, &tw, &tprof.decode, CAP, ROWS, "step")?;
    pkg.function(sg);
    merge(&mut params, sp)?;
    params.insert("step.rope_cos.table".into(), rope.0.clone());
    params.insert("step.rope_sin.table".into(), rope.1.clone());
    params.insert("step.iota".into(), (0..ROWS).map(|i| i as f32).collect());
    drop(tw);

    // The code predictor.
    let mut cq = cpc.to_qwen3_config();
    cq.max_position_embeddings = CP_ROWS;
    let cw = remap_code_predictor_weights(&mut store.load_code_predictor_backbone()?)?;
    let cprof = Qwen3TtsGraphProfiles::for_role(
        dir,
        Qwen3TtsGraphRole::CodePredictor,
        rlx_runtime::Device::Cpu,
    );
    // Its weights and RoPE tables are the talker's names (`model.layers.*`,
    // `rope.*`) with other values: its own, `cp.`.
    let cp_prompt = |cap: usize| -> Result<Built> {
        let (g, p) = prompt_graph(&cq, &cw, &cprof.prefill, None, 2, cap, "cp_prompt")?;
        prefixed(&g, p, "cp.")
    };
    let (cg, lifted) = symbolic_in_lifted_many(&[(CP_CAP, vec![21, 29, 25])], &[0], |n| {
        Ok(cp_prompt(n[0])?.0)
    })?;
    anyhow::ensure!(
        lifted.is_empty(),
        "the code predictor's prompt lifted tables"
    );
    pkg.function(cg);
    merge(&mut params, cp_prompt(25)?.1)?;
    let (csg, csp) = step_graph(&cq, &cw, &cprof.decode, CP_CAP, CP_ROWS, "cp_step")?;
    let (csg, csp) = prefixed(&csg, csp, "cp.")?;
    pkg.function(csg);
    merge(&mut params, csp)?;
    let half = cq.head_dim / 2;
    let (mut cc, mut cs) = (Vec::new(), Vec::new());
    for p in 0..CP_ROWS {
        let (c, s) = rlx_qwen3::generator::compute_rope_slice(&cq, p);
        anyhow::ensure!(c.len() == half, "the code predictor's RoPE row");
        cc.extend(c);
        cs.extend(s);
    }
    params.insert("cp.cp_step.rope_cos.table".into(), cc);
    params.insert("cp.cp_step.rope_sin.table".into(), cs);
    params.insert(
        "cp.cp_step.iota".into(),
        (0..CP_ROWS).map(|i| i as f32).collect(),
    );
    drop(cw);

    // The speech decoder.
    let st_path = dir.join("speech_tokenizer/model.safetensors");
    let dcfg = crate::speech_tokenizer::decode::load_decoder_config(&dir.join("speech_tokenizer"))?;
    let sw: HashMap<String, (Vec<f32>, Vec<usize>)> = {
        let mut wm = WeightMap::from_file(st_path.to_str().context("path")?)?;
        let keys: Vec<String> = wm
            .keys()
            .filter(|k| k.starts_with("decoder."))
            .map(String::from)
            .collect();
        keys.into_iter()
            .map(|k| {
                let v = wm.take(&k)?;
                Ok((k, v))
            })
            .collect::<Result<_>>()?
    };
    let speech_g = symbolic_in(FRAMES, (5, 7, 4), |f| Ok(speech(&sw, &dcfg, f)?.0))?;
    let (at4, mut speech_p) = speech(&sw, &dcfg, 4)?;
    let speech_g = broadcast_params(&speech_g, &at4, &mut speech_p, &[])?;
    pkg.function(speech_g);
    pkg.dims = vec![
        ("cap".into(), CAP),
        ("cp_cap".into(), CP_CAP),
        ("prompt".into(), PROMPT),
        ("frames".into(), FRAMES),
        ("rows".into(), ROWS_SYM),
    ];

    // The checkpoint's tensors referenced; the rest (the decoder's
    // codebooks, the tables) held.
    {
        let src = WeightMap::from_file(dir.join("model.safetensors").to_str().context("path")?)?;
        pkg.reference_map(
            &params,
            &src,
            "safetensors",
            &dir.join("model.safetensors"),
            out,
            false,
        )?;
    }
    {
        let src = WeightMap::from_file(st_path.to_str().context("path")?)?;
        pkg.reference_map(&speech_p, &src, "safetensors", &st_path, out, false)?;
    }

    pkg.sidecar(
        "tokenizer.json",
        "application/json",
        std::fs::read(dir.join("tokenizer.json")).context("tokenizer.json")?,
    );
    let module = program::build(&Path::new(env!("CARGO_MANIFEST_DIR")).join("program/Cargo.toml"))
        .context("build Qwen3-TTS' program")?;
    let defaults = crate::config::GenerationConfig::default();
    program::entry(
        &mut pkg,
        "speak",
        module,
        &["synthesize", "run"],
        "text-to-speech",
        json!({
            "tokenizer": "sidecar:tokenizer.json",
            "layers": tq.num_hidden_layers,
            "cp_layers": cq.num_hidden_layers,
            "groups": groups,
            "hidden": tq.hidden_size,
            "vocab": talker.vocab_size,
            "cp_vocab": cpc.vocab_size,
            "rows": ROWS,
            "cp_rows": CP_ROWS,
            "eos": talker.codec_eos_token_id,
            "codec_pad": talker.codec_pad_id,
            "codec_bos": talker.codec_bos_id,
            "codec_think": talker.codec_think_id,
            "codec_nothink": talker.codec_nothink_id,
            "codec_think_bos": talker.codec_think_bos_id,
            "codec_think_eos": talker.codec_think_eos_id,
            "speakers": talker.spk_id,
            "languages": talker.codec_language_id,
            "tts_bos": cfg.tts_bos_token_id,
            "tts_eos": cfg.tts_eos_token_id,
            "tts_pad": cfg.tts_pad_token_id,
            "speaker": "vivian",
            "language": "auto",
            "repetition_penalty": defaults.repetition_penalty,
            "max_new_tokens": defaults.max_new_tokens,
            "min_new_tokens": defaults.min_new_tokens,
            "sample_rate": crate::tokens::SAMPLE_RATE_HZ,
        }),
    );
    pkg.metadata("sample_rate", json!(crate::tokens::SAMPLE_RATE_HZ));
    pkg.write(out, opts)
}

/// Register this crate's exporter ([`rlx_core::export::register_exporter`]),
/// taken by name (`--family qwen3-tts`).
pub fn register() {
    use rlx_core::export::{Exporter, register_exporter};
    register_exporter(Exporter::new("qwen3-tts", "text-to-speech", export_rlxp));
}

#[cfg(test)]
mod stages {
    use super::*;
    use std::path::PathBuf;

    fn run(
        g: Graph,
        params: &HashMap<String, Vec<f32>>,
        inputs: &[(&str, &[f32])],
    ) -> Vec<Vec<f32>> {
        let mut c = rlx_runtime::Session::new(rlx_runtime::Device::Cpu).compile(g);
        for (n, v) in params {
            c.set_param(n, v);
        }
        c.finalize_params();
        c.run(inputs)
    }

    fn max_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs()))
    }

    /// The package's talker graphs against the runner's on one prompt.
    #[test]
    fn talker_stages_match_the_runner() {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../weights/tts/qwen3-tts-0.6b-customvoice");
        if !dir.join("tokenizer.json").is_file() {
            return;
        }
        let cfg = Qwen3TtsConfig::from_model_dir(&dir).unwrap();
        let talker = cfg.talker();
        let store = Qwen3TtsWeightStore::open(&dir).unwrap();
        let tok = crate::prompt::load_text_tokenizer(&dir).unwrap();
        let te = crate::text_embed::TextEmbedder::open(&store).unwrap();
        let prompt = crate::prompt::build_custom_voice_prompt(
            &cfg,
            &store,
            &te,
            &tok,
            "Hello from native Rust.",
            "vivian",
            "English",
        )
        .unwrap();
        let (t, h) = prompt.embeds.dim();
        // The text rows graph against the embedder.
        let ids = [cfg.tts_pad_token_id];
        let text_hidden = store
            .tensor_snapshot(&["talker.model.text_embedding.weight"])
            .unwrap()["talker.model.text_embedding.weight"]
            .1[1];
        let g = rows(talker, true, 1, text_hidden);
        let snap = store
            .tensor_snapshot(&[
                "talker.model.text_embedding.weight",
                "talker.text_projection.linear_fc1.weight",
                "talker.text_projection.linear_fc1.bias",
                "talker.text_projection.linear_fc2.weight",
                "talker.text_projection.linear_fc2.bias",
            ])
            .unwrap();
        let p: HashMap<String, Vec<f32>> = snap.into_iter().map(|(k, v)| (k, v.0)).collect();
        let idf: Vec<f32> = ids.iter().map(|&i| i as f32).collect();
        let got = run(g, &p, &[("ids", &idf)]).remove(0);
        let d = max_diff(&got, &prompt.tts_pad_embed);
        eprintln!("text row max |d| {d}");
        assert!(d < 1e-5);

        // The prompt graph against the runner's prefill.
        let tq = talker.to_qwen3_config();
        let mut tq_rows = tq.clone();
        tq_rows.max_position_embeddings = ROWS;
        let tw = remap_talker_weights(&mut store.load_talker_backbone().unwrap()).unwrap();
        let tprof = Qwen3TtsGraphProfiles::for_role(
            &dir,
            Qwen3TtsGraphRole::Talker,
            rlx_runtime::Device::Cpu,
        );
        let (g, p) = prompt_graph(
            &tq_rows,
            &tw,
            &tprof.prefill,
            Some(talker_rope(talker)),
            t,
            64,
            "prompt",
        )
        .unwrap();
        let flat: Vec<f32> = prompt.embeds.iter().copied().collect();
        let got = run(g, &p, &[("inputs_embeds", &flat)]).remove(0);
        let mut mk = crate::megakernel::Qwen3TtsMegakernel::open(
            &store,
            talker,
            cfg.code_predictor(),
            rlx_runtime::Device::Cpu,
        )
        .unwrap();
        mk.talker_prefill_core(prompt.embeds.view()).unwrap();
        let want = mk.talker_hidden_row().to_vec();
        let d = max_diff(&got, &want);
        eprintln!("prompt hidden ({t} rows, {h}) max |d| {d}");
        assert!(d < 1e-4);
        // The head.
        let (hw, _) = store.take_codec_head().unwrap();
        let hp = HashMap::from([("talker.codec_head.weight".to_string(), hw.clone())]);
        let logits = run(head(talker), &hp, &[("hidden", &want)]).remove(0);
        let mut ref_logits = vec![0f32; talker.vocab_size];
        for (i, l) in ref_logits.iter_mut().enumerate() {
            *l = (0..h).map(|j| want[j] * hw[i * h + j]).sum();
        }
        let d = max_diff(&logits, &ref_logits);
        eprintln!("head max |d| {d}");
        assert!(d < 1e-3);
    }
}
