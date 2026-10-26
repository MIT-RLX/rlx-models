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

//! Export Moonshine v2 streaming as an `.rlxp` package with a `transcribe`
//! entry, through the one exporter ([`rlx_core::export`]): 16 kHz audio of
//! any length in, its text out, as [`crate::MoonshineStreaming::transcribe`]
//! gives it.
//!
//! What the runner does on the host is in the package:
//!
//! - the frontend ([`crate::frontend`]): each 80-sample frame's CMVN and
//!   `asinh` compression, and the validity masks of the frames and of both
//!   convolutions, from the clip's real length (the entry pads the audio to
//!   whole groups of four frames, which leaves the encoder's length as the
//!   runner's);
//! - each layer's sliding-window attention bias over the live positions;
//! - the decoder's token embedding and LM head around the runner's own
//!   decoder graph, one function per 16-token bucket, each symbolic in the
//!   encoder's length and run with its states as context — greedy, the
//!   whole sequence each token, as the runner decodes.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, anyhow};
use rlx_core::export::graph::{Pipeline, Stage, broadcast_params, symbolic_in};
use rlx_core::export::{Checkpoint, ExportOptions, ExportReport};
use rlx_core::flow_util::graph_from_built;
use rlx_ir::op::{Activation, BinaryOp, CmpOp};
use rlx_ir::{DType, Graph, GraphExt, NodeId, Op, Shape};
use rlx_moonshine::MoonshineWeightPrefix;
use rlx_moonshine::weight_source::CloningWeightSource;
use serde_json::json;

use crate::{
    CMVN_EPS, CONV_K, CONV_STRIDE, DECODER_BUCKET, MASKED, StreamingConfig, build_encoder,
    window_input,
};

/// The symbols: groups of four frames (the encoder's length), and the
/// encoder's length a decoder function reads.
const GROUPS: u32 = 0;
const ENCODED: u32 = 1;

/// Frames to a group: the encoder halves its length twice.
const GROUP: usize = CONV_STRIDE * CONV_STRIDE;

/// The longest decoder bucket the package holds when the options name none.
const LONGEST: usize = 256;

fn f32s(d: &[usize]) -> Shape {
    Shape::new(d, DType::F32)
}

fn k(g: &mut Graph, v: f32) -> NodeId {
    g.add_node(
        Op::Constant {
            data: v.to_le_bytes().to_vec(),
        },
        vec![],
        f32s(&[1]),
    )
}

fn act(g: &mut Graph, a: Activation, x: NodeId) -> NodeId {
    let s = g.node(x).shape.clone();
    g.add_node(Op::Activation(a), vec![x], s)
}

fn bin(g: &mut Graph, op: BinaryOp, a: NodeId, b: NodeId) -> NodeId {
    let s = g
        .node(a)
        .shape
        .broadcast_with(&g.node(b).shape)
        .expect("broadcastable");
    g.binary(op, a, b, s)
}

/// `a op b` as 0 / 1.
fn flag(g: &mut Graph, op: CmpOp, a: NodeId, b: NodeId) -> NodeId {
    let s = g
        .node(a)
        .shape
        .broadcast_with(&g.node(b).shape)
        .expect("broadcastable");
    let dims = s.dims().to_vec();
    let c = g.add_node(
        Op::Compare(op),
        vec![a, b],
        Shape::from_dims(&dims, DType::Bool),
    );
    g.add_node(
        Op::Cast { to: DType::F32 },
        vec![c],
        Shape::from_dims(&dims, DType::F32),
    )
}

/// Which of `len` conv outputs see a valid input, the inputs valid below
/// `valid` (`[1]`): output `t` reads inputs `max(0, 2t − K + 1) ..= 2t`.
/// Returns the mask `[len]` and its count.
fn conv_valid(g: &mut Graph, iota: NodeId, len: usize, valid: NodeId) -> (NodeId, NodeId) {
    let t = g.narrow_(iota, 0, 0, len);
    let stride = k(g, CONV_STRIDE as f32);
    let back = k(g, (CONV_K - 1) as f32);
    let start = bin(g, BinaryOp::Mul, t, stride);
    let start = bin(g, BinaryOp::Sub, start, back);
    let zero = k(g, 0.0);
    let start = bin(g, BinaryOp::Max, start, zero);
    let m = flag(g, CmpOp::Lt, start, valid);
    let count = g.reduce(m, rlx_ir::op::ReduceOp::Sum, vec![0], true, f32s(&[1]));
    (m, count)
}

/// The frontend at `groups` groups of four frames: `audio` `[groups · 320]`
/// (zero-padded) and `audio_samples` `[1]` (its real length) → the
/// encoder's inputs.
fn frontend(cfg: &StreamingConfig, log_k: f32, groups: usize) -> Graph {
    let e = &cfg.encoder;
    let fl = e.frame_len;
    let frames = groups * GROUP;
    let (t1, t2) = (groups * CONV_STRIDE, groups);
    let heads = e.num_attention_heads;
    let mut g = Graph::new("frontend");
    let audio = g.input("audio", f32s(&[frames * fl]));
    let samples = g.input("audio_samples", f32s(&[1]));
    let iota = g.param("frontend.iota", f32s(&[crate::export::IOTA]));

    // Per frame: x − mean, over rms, scaled, asinh.
    let x = g.reshape_(audio, vec![frames as i64, fl as i64]);
    let n = k(&mut g, fl as f32);
    let sum = g.reduce(
        x,
        rlx_ir::op::ReduceOp::Sum,
        vec![1],
        true,
        f32s(&[frames, 1]),
    );
    let mean = bin(&mut g, BinaryOp::Div, sum, n);
    let x = bin(&mut g, BinaryOp::Sub, x, mean);
    let sq = bin(&mut g, BinaryOp::Mul, x, x);
    let var = g.reduce(
        sq,
        rlx_ir::op::ReduceOp::Sum,
        vec![1],
        true,
        f32s(&[frames, 1]),
    );
    let var = bin(&mut g, BinaryOp::Div, var, n);
    let eps = k(&mut g, CMVN_EPS);
    let var = bin(&mut g, BinaryOp::Add, var, eps);
    let rms = act(&mut g, Activation::Sqrt, var);
    let scale = k(&mut g, log_k.exp());
    let z = bin(&mut g, BinaryOp::Mul, scale, x);
    let z = bin(&mut g, BinaryOp::Div, z, rms);
    // asinh z = sign(z) · ln(|z| + √(z² + 1)).
    let a = act(&mut g, Activation::Abs, z);
    let z2 = bin(&mut g, BinaryOp::Mul, a, a);
    let one = k(&mut g, 1.0);
    let z2 = bin(&mut g, BinaryOp::Add, z2, one);
    let r = act(&mut g, Activation::Sqrt, z2);
    let s = bin(&mut g, BinaryOp::Add, a, r);
    let l = act(&mut g, Activation::Log, s);
    let sign = act(&mut g, Activation::Sign, z);
    let y = bin(&mut g, BinaryOp::Mul, sign, l);
    let y = g.reshape(y, vec![1, frames as i64, fl as i64], f32s(&[1, frames, fl]));
    g.node_mut(y).name = Some("frames".into());

    // Validity: the frames wholly within the audio, then each conv's.
    let flk = k(&mut g, fl as f32);
    let v0 = bin(&mut g, BinaryOp::Div, samples, flk);
    let v0 = act(&mut g, Activation::Floor, v0);
    let f_iota = g.narrow_(iota, 0, 0, frames);
    let m0 = flag(&mut g, CmpOp::Lt, f_iota, v0);
    let (m1, v1) = conv_valid(&mut g, iota, t1, v0);
    let (m2, _) = conv_valid(&mut g, iota, t2, v1);
    let m0 = g.reshape(m0, vec![1, frames as i64, 1], f32s(&[1, frames, 1]));
    let m1o = g.reshape(m1, vec![1, 1, t1 as i64], f32s(&[1, 1, t1]));
    let m2o = g.reshape(m2, vec![1, 1, t2 as i64], f32s(&[1, 1, t2]));
    g.node_mut(m0).name = Some("frame_mask".into());
    g.node_mut(m1o).name = Some("conv1_mask".into());
    g.node_mut(m2o).name = Some("conv2_mask".into());

    // Each window's bias: 0 where query q sees key k (k ≤ q within `left`,
    // k > q within `right`) and k is live, MASKED elsewhere.
    let pos = g.narrow_(iota, 0, 0, t2);
    let q = g.reshape(pos, vec![t2 as i64, 1], f32s(&[t2, 1]));
    let kk = g.reshape(pos, vec![1, t2 as i64], f32s(&[1, t2]));
    let back = bin(&mut g, BinaryOp::Sub, q, kk);
    let ahead = bin(&mut g, BinaryOp::Sub, kk, q);
    let zero = k(&mut g, 0.0);
    let past = flag(&mut g, CmpOp::Ge, back, zero);
    let future = flag(&mut g, CmpOp::Gt, ahead, zero);
    let live = g.reshape(m2, vec![1, t2 as i64], f32s(&[1, t2]));
    let mut outs = vec![y, m0, m1o, m2o];
    for (left, right) in windows(cfg) {
        let lk = k(&mut g, left as f32);
        let rk = k(&mut g, right as f32);
        let near_back = flag(&mut g, CmpOp::Lt, back, lk);
        let near_ahead = flag(&mut g, CmpOp::Lt, ahead, rk);
        let a = bin(&mut g, BinaryOp::Mul, past, near_back);
        let b = bin(&mut g, BinaryOp::Mul, future, near_ahead);
        let seen = bin(&mut g, BinaryOp::Add, a, b);
        let seen = bin(&mut g, BinaryOp::Mul, seen, live);
        let off = bin(&mut g, BinaryOp::Sub, seen, one);
        let big = k(&mut g, -MASKED);
        let bias = bin(&mut g, BinaryOp::Mul, off, big);
        let bias = g.reshape(
            bias,
            vec![1, 1, t2 as i64, t2 as i64],
            f32s(&[1, 1, t2, t2]),
        );
        let bias = g.add_node(
            Op::Expand {
                target_shape: vec![1, heads as i64, t2 as i64, t2 as i64],
            },
            vec![bias],
            f32s(&[1, heads, t2, t2]),
        );
        g.node_mut(bias).name = Some(window_input((left, right)));
        outs.push(bias);
    }
    g.set_outputs(outs);
    g
}

/// The encoder's windows, each once, as the runner builds them.
fn windows(cfg: &StreamingConfig) -> Vec<(usize, usize)> {
    let mut w = cfg.encoder.sliding_windows.clone();
    w.sort_unstable();
    w.dedup();
    w
}

/// The frontend's index table length: the longest frame count it takes.
pub(crate) const IOTA: usize = 1 << 16;

/// One decoder function at `cap` tokens and `enc` encoder positions:
/// `tokens` `[1, cap]`, `last` `[1]` and `encoder_states` `[1, enc, d]` →
/// `logits` `[1, vocab]` at `last`.
fn decoder(
    cfg: &StreamingConfig,
    weights: &rlx_core::weight_map::WeightMap,
    pfx: &MoonshineWeightPrefix,
    cap: usize,
    enc: usize,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    let dec = &cfg.decoder;
    let (d, vocab) = (dec.hidden_size, dec.vocab_size);
    let mut src = CloningWeightSource(weights);
    let built = rlx_moonshine::flow::build_decoder_hidden_built(dec, &mut src, pfx, 1, cap, enc)?;
    let (inner, params) = graph_from_built(built)?;
    let mut g = Graph::new(format!("decode_{cap}"));
    let tokens = g.input("tokens", Shape::new(&[1, cap], DType::I32));
    let last = g.input("last", Shape::new(&[1], DType::I32));
    let states = g.input("encoder_states", f32s(&[1, enc, d]));
    let table = g.param(pfx.dec_embed_tokens(), f32s(&[vocab, d]));
    let flat = g.reshape(tokens, vec![cap as i64], Shape::new(&[cap], DType::I32));
    let rows = g.gather(table, flat, 0, f32s(&[cap, d]));
    let embeds = g.reshape(rows, vec![1, cap as i64, d as i64], f32s(&[1, cap, d]));
    let spliced = g
        .splice(
            &inner,
            &rlx_ir::Splice::new()
                .bind("decoder_inputs_embeds", embeds)
                .bind("encoder_hidden", states),
        )
        .map_err(|e| anyhow!("splicing the decoder: {e}"))?;
    let hidden = g.reshape(
        spliced.outputs[0],
        vec![cap as i64, d as i64],
        f32s(&[cap, d]),
    );
    let row = g.gather(hidden, last, 0, f32s(&[1, d]));
    let head_name = pfx
        .proj_out
        .clone()
        .unwrap_or_else(|| pfx.dec_embed_tokens());
    let head = g.param(head_name, f32s(&[vocab, d]));
    let head = g.transpose_(head, vec![1, 0]);
    let logits = g.mm(row, head);
    g.node_mut(logits).name = Some("logits".into());
    g.set_outputs(vec![logits]);
    Ok((g, params))
}

/// Export the Moonshine v2 streaming checkpoint in `dir` (`config.json`,
/// `model.safetensors`, `tokenizer.json`) as the package `out`.
pub fn export_rlxp(dir: &Path, out: &Path, opts: &ExportOptions) -> Result<ExportReport> {
    let ck = Checkpoint::open(dir)?;
    let cfg = StreamingConfig::from_json(&std::fs::read_to_string(dir.join("config.json"))?)?;
    let (_, weights) = rlx_core::weights::open_map(dir.join("model.safetensors"))?;
    let pfx = MoonshineWeightPrefix::detect(&weights);
    let log_k = weights
        .get("model.encoder.embedder.comp.log_k")
        .map(|(v, _)| v[0])
        .ok_or_else(|| anyhow!("no model.encoder.embedder.comp.log_k"))?;

    let front =
        symbolic_in(GROUPS, (5, 7, 4), |n| Ok(frontend(&cfg, log_k, n))).context("the frontend")?;
    let encode = |n: usize| -> Result<(Graph, HashMap<String, Vec<f32>>)> {
        let mut src = CloningWeightSource(&weights);
        let built = build_encoder(&cfg, &mut src, &windows(&cfg), n * GROUP)?;
        let (mut g, p) = graph_from_built(built)?;
        g.name = "encode".into();
        Ok((g, p))
    };
    let enc_g = symbolic_in(GROUPS, (5, 7, 4), |n| Ok(encode(n)?.0)).context("the encoder")?;
    let (at4, mut params) = encode(4)?;
    // Added to the encoder's states before their projection: its width is
    // the encoder's.
    let (pos, pos_shape) = weights
        .get("model.decoder.pos_emb.weight")
        .ok_or_else(|| anyhow!("no model.decoder.pos_emb.weight"))?;
    let width = *pos_shape.last().unwrap_or(&cfg.encoder.hidden_size);
    let pos = pos.to_vec();
    let enc_g = broadcast_params(
        &enc_g,
        &at4,
        &mut params,
        &[("model.decoder.pos_emb.weight", &pos, width)],
    )?;
    params.insert(
        "frontend.iota".into(),
        (0..IOTA).map(|i| i as f32).collect(),
    );

    let longest = opts
        .seqs
        .iter()
        .copied()
        .max()
        .unwrap_or(LONGEST)
        .min(cfg.decoder.max_position_embeddings);
    let mut functions = Vec::new();
    let mut decoders = Vec::new();
    for cap in (DECODER_BUCKET..=longest).step_by(DECODER_BUCKET) {
        let g = symbolic_in(ENCODED, (5, 7, 4), |t| {
            Ok(decoder(&cfg, &weights, &pfx, cap, t)?.0)
        })?;
        let (_, p) = decoder(&cfg, &weights, &pfx, cap, 4)?;
        params.extend(p);
        functions.push(g.name.clone());
        decoders.push(g);
    }
    for name in [pfx.dec_embed_tokens()]
        .into_iter()
        .chain(pfx.proj_out.clone())
    {
        let (v, _) = weights.get(&name).ok_or_else(|| anyhow!("no {name}"))?;
        params.insert(name, v.to_vec());
    }

    let fe_outputs: Vec<String> = ["frames", "frame_mask", "conv1_mask", "conv2_mask"]
        .iter()
        .map(|s| s.to_string())
        .chain(windows(&cfg).into_iter().map(window_input))
        .collect();
    let fe_out: Vec<&str> = fe_outputs.iter().map(String::as_str).collect();
    let mut enc_stage = Stage::new(enc_g).outputs(&["encoder_hidden", "decoder_states"]);
    for name in &fe_outputs {
        enc_stage = enc_stage.input(name, name);
    }
    let stages = vec![
        Stage::new(front)
            .input("audio", "audio")
            .input("audio_samples", "audio_samples")
            .outputs(&fe_out),
        enc_stage,
    ];
    let mut p = Pipeline::new("moonshine-streaming", "speech-recognition", stages);
    p.dims = vec![("groups".into(), GROUPS), ("encoded".into(), ENCODED)];
    p.functions = decoders;
    p.exported_by = "rlx-moonshine-streaming".into();
    let mut pkg = p.package(params, Some(&ck), out, opts)?;
    // The pipeline entry stands for its stages; the package is a recognizer.
    let pipeline = pkg.model["entries"]
        .as_object_mut()
        .and_then(|e| e.remove("pipeline"))
        .ok_or_else(|| anyhow!("the pipeline wrote no entry"))?;
    let tokenizer = pkg.tokenizer(&ck, None)?;
    let dec = &cfg.decoder;
    pkg.entry(
        "transcribe",
        json!({
            "sample_rate": crate::SAMPLE_RATE,
            "audio_multiple": GROUP * cfg.encoder.frame_len,
            "stages": pipeline["stages"],
            "decoder": {
                "prefill": {
                    "functions": functions,
                    "inputs": {
                        "tokens": "tokens",
                        "last": "last",
                        "encoder_states": {"context": "decoder_states"},
                    },
                    "state": [],
                    "logits": 0,
                },
                "vocab_size": dec.vocab_size,
                // No step: each token, the whole sequence through a function.
                "inputs": {},
                "tokenizer": tokenizer,
                "stop_ids": [dec.eos_token_id],
            },
            "tokenizer": tokenizer,
            "prompt_ids": [dec.decoder_start_token_id],
            "stop_ids": [dec.eos_token_id],
            // Moonshine's own runtime: 6.5 tokens a second of audio, and 8.
            "token_budget": {"per_second": 6.5, "extra": 8},
        }),
    );
    pkg.model["task"] = json!("speech-recognition");
    pkg.write(out, opts)
}

/// Register this crate's exporter ([`rlx_core::export::register_exporter`]).
pub fn register() {
    use rlx_core::export::{Exporter, register_exporter};
    register_exporter(
        Exporter::new("moonshine-streaming", "speech-recognition", export_rlxp)
            .claims(&["moonshine_streaming"]),
    );
}
