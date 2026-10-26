// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0

//! Export MetaVoice-1B as an `.rlxp` package rlx runs alone: text and a
//! voice (a 16 kHz reference clip, or the bundled Bria's) in, 24 kHz speech
//! out, through the one exporter ([`rlx_core::export`]).
//!
//! Its graphs, as [`crate::MetaVoice`] computes: `speaker` (one 160-frame
//! partial of the reference — Hann-windowed 400-sample frames, their power
//! spectra as DFT matmuls through the Slaney filterbank, the 3-layer LSTM,
//! the projection and its ReLU); the first-stage GPT as `fs_prefill` (the
//! prompt, symbolic in its length, both classifier-free-guidance rows: the
//! speaker's projection added to row 0's embeddings) and `fs_step` (one
//! token after a cache, symbolic in the cache, each layer's keys and values
//! grown by a row); `second_stage` (the non-causal fine-book model over its
//! 1024 positions, its six heads' logits); EnCodec's `lookup`, `pre`,
//! `lstm` and `post` ([`rlx_encodec::export::encodec_functions`]). The first
//! stage's weights are its checkpoint's, referenced; the rest held. The
//! program (`program/`) tokenizes, embeds the voice, samples the first stage
//! a token at a time (its caches kept in slots), splits the coarse books,
//! runs the second stage and decodes.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use rlx_core::export::graph::{symbolic_in, symbolic_in_lifted_many};
use rlx_core::export::{Checkpoint, ExportOptions, ExportReport, Package, program};
use rlx_core::weight_map::WeightMap;
use rlx_ir::op::MaskKind;
use rlx_ir::{DType, Graph, GraphExt, Op, Shape};
use serde_json::json;

use crate::config::{FirstStageArgs, SAMPLE_RATE, SecondStageArgs};
use crate::first_stage::EOS_AUDIO;
use crate::native::{DEFAULT_ENCODEC_PATH, load_first_args, load_second_args};
use crate::second_stage::{PAD, TEXT_OFFSET};

/// The symbols: EnCodec's frames and codebooks, the prompt's length and the
/// step's cache.
const FRAMES: u32 = 0;
const CODEBOOKS: u32 = 1;
const PROMPT: u32 = 2;
const CAP: u32 = 3;

/// The speaker encoder's framing (`speaker.rs`).
const SPK_N_FFT: usize = 400;
const SPK_HOP: usize = 160;
const SPK_MELS: usize = 40;
const SPK_HIDDEN: usize = 256;
const SPK_LAYERS: usize = 3;
const SPK_PARTIAL: usize = 160;

fn f32s(d: &[usize]) -> Shape {
    Shape::new(d, DType::F32)
}

/// `x @ wᵀ` for a `[out, in]` weight.
fn linear(g: &mut Graph, x: rlx_ir::NodeId, name: &str, out: usize, inp: usize) -> rlx_ir::NodeId {
    let w = g.param(name, f32s(&[out, inp]));
    let wt = g.transpose_(w, vec![1, 0]);
    g.mm(x, wt)
}

/// A step's cache read: the token's position `pos` `[1]` (f32) and the
/// keys it may see, `mask` `[2, cap]`.
struct StepAt {
    pos: rlx_ir::NodeId,
    mask: rlx_ir::NodeId,
}

/// The first stage's blocks over `x` `[2, t, c]`; each layer's keys and
/// values as caches of `cap` rows: a prompt's (`step` none) its own rows
/// written over zeros from row 0, a step's (`step`) its row written into
/// the cache inputs `k{i}` / `v{i}` at `pos`. The final norm's output, and
/// the caches (`k{i}`, `v{i}`; a step's `k{i}.next`, `v{i}.next`).
fn first_blocks(
    g: &mut Graph,
    args: &FirstStageArgs,
    hidden: usize,
    mut x: rlx_ir::NodeId,
    t: usize,
    cap: usize,
    step: Option<&StepAt>,
) -> (rlx_ir::NodeId, Vec<rlx_ir::NodeId>) {
    let c = args.n_embd;
    let hd = c / args.n_head;
    let eps = args.rmsnorm_eps;
    let zero = g.param("fs.zero", f32s(&[c]));
    let cache = f32s(&[2, cap, c]);
    // The prompt's rows go in at row 0, each over zeros of its own (the
    // write is in place).
    let origin = g.add_node(
        Op::Constant {
            data: 0f32.to_le_bytes().to_vec(),
        },
        vec![],
        f32s(&[1]),
    );
    let nought = g.add_node(
        Op::Constant {
            data: 0f32.to_le_bytes().to_vec(),
        },
        vec![],
        f32s(&[1, 1, 1]),
    );
    let mut caches = Vec::new();
    for i in 0..args.n_layer {
        let p = format!("transformer.h.{i}");
        let ln1 = g.param(format!("{p}.ln_1.weight"), f32s(&[c]));
        let n1 = g.rms_norm(x, ln1, zero, eps);
        let qkv = linear(g, n1, &format!("{p}.attn.c_attn.weight"), 3 * c, c);
        let q = g.narrow_(qkv, 2, 0, c);
        let k = g.narrow_(qkv, 2, c, c);
        let v = g.narrow_(qkv, 2, 2 * c, c);
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
                            target_shape: vec![2, cap as i64, c as i64],
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
        let a = match step {
            Some(s) => g.attention(q, kv[0], kv[1], s.mask, args.n_head, hd, f32s(&[2, t, c])),
            None => g.attention_kind(q, k, v, args.n_head, hd, MaskKind::Causal, f32s(&[2, t, c])),
        };
        let a = linear(g, a, &format!("{p}.attn.c_proj.weight"), c, c);
        x = g.add(x, a);
        let ln2 = g.param(format!("{p}.ln_2.weight"), f32s(&[c]));
        let n2 = g.rms_norm(x, ln2, zero, eps);
        let gate = linear(g, n2, &format!("{p}.mlp.swiglu.w1.weight"), hidden, c);
        let gate = g.silu(gate);
        let up = linear(g, n2, &format!("{p}.mlp.swiglu.w3.weight"), hidden, c);
        let h = g.mul(gate, up);
        let m = linear(g, h, &format!("{p}.mlp.c_proj.weight"), c, hidden);
        x = g.add(x, m);
    }
    let lnf = g.param("transformer.ln_f.weight", f32s(&[c]));
    (g.rms_norm(x, lnf, zero, eps), caches)
}

/// The tokens' embeddings `[t, c]` as both guidance rows `[2, t, c]`: the
/// speaker's projection added to row 0's.
fn first_embed(
    g: &mut Graph,
    args: &FirstStageArgs,
    emb: rlx_ir::NodeId,
    t: usize,
) -> rlx_ir::NodeId {
    let c = args.n_embd;
    let s = args.speaker_emb_size;
    let spk = g.input("spk", f32s(&[1, s]));
    let spk = linear(g, spk, "speaker_cond_pos.weight", c, s);
    let cond = g.add(emb, spk);
    let cond = g.reshape_(cond, vec![1, t as i64, c as i64]);
    let plain = g.reshape_(emb, vec![1, t as i64, c as i64]);
    g.concat_(vec![cond, plain], 0)
}

/// Both rows' logits `[2, vocab]` of the final norm's row `row` of `t`.
fn first_logits(
    g: &mut Graph,
    args: &FirstStageArgs,
    xf: rlx_ir::NodeId,
    row: usize,
) -> rlx_ir::NodeId {
    let (c, v) = (args.n_embd, args.vocab_sizes[0]);
    let last = g.narrow_(xf, 1, row, 1);
    let last = g.reshape_(last, vec![2, c as i64]);
    let logits = linear(g, last, "lm_heads.0.weight", v, c);
    g.node_mut(logits).name = Some("logits".into());
    logits
}

/// `fs_prefill`: `ids` `[t]` i32 at positions `0..t` and the speaker `spk`
/// `[1, 256]` → the last row's `logits` `[2, vocab]` and each layer's
/// `k{i}`, `v{i}` `[2, cap, c]` (the prompt's rows first, then zeros).
fn first_prefill(args: &FirstStageArgs, hidden: usize, t: usize, cap: usize) -> Graph {
    let (c, vocab) = (args.n_embd, args.vocab_sizes[0]);
    let mut g = Graph::new("fs_prefill");
    let ids = g.input("ids", Shape::new(&[t], DType::I32));
    let wte = g.param("transformer.wtes.0.weight", f32s(&[vocab, c]));
    let wpe = g.param("transformer.wpe.weight", f32s(&[args.block_size, c]));
    let tok = g.gather_(wte, ids, 0);
    let pos = g.narrow_(wpe, 0, 0, t);
    let emb = g.add(tok, pos);
    let x = first_embed(&mut g, args, emb, t);
    let (xf, caches) = first_blocks(&mut g, args, hidden, x, t, cap, None);
    let logits = first_logits(&mut g, args, xf, t - 1);
    let mut outs = vec![logits];
    outs.extend(caches);
    g.set_outputs(outs);
    g
}

/// `fs_step`: the token `id` `[1]` i32 at `pos` `[1]` (f32), the keys it
/// sees `mask` `[2, cap]` (1 to `pos`), each layer's caches `k{i}`, `v{i}`
/// `[2, cap, c]` → its `logits` `[2, vocab]` and the caches with its row
/// written at `pos` (`k{i}.next`, `v{i}.next`).
fn first_step(args: &FirstStageArgs, hidden: usize, cap: usize) -> Graph {
    let (c, vocab) = (args.n_embd, args.vocab_sizes[0]);
    let mut g = Graph::new("fs_step");
    let id = g.input("id", Shape::new(&[1], DType::I32));
    let at = g.input("pos", f32s(&[1]));
    let mask = g.input("mask", f32s(&[2, cap]));
    let wte = g.param("transformer.wtes.0.weight", f32s(&[vocab, c]));
    let wpe = g.param("transformer.wpe.weight", f32s(&[args.block_size, c]));
    let tok = g.gather_(wte, id, 0);
    let pos = g.gather_(wpe, at, 0);
    let emb = g.add(tok, pos);
    let x = first_embed(&mut g, args, emb, 1);
    let step = StepAt { pos: at, mask };
    let (xf, caches) = first_blocks(&mut g, args, hidden, x, 1, cap, Some(&step));
    let logits = first_logits(&mut g, args, xf, 0);
    let mut outs = vec![logits];
    outs.extend(caches);
    g.set_outputs(outs);
    g
}

/// `second_stage`: the two hierarchies `h0`, `h1` `[block]` i32 and the
/// speaker `spk` `[1, 256]` → each head's `logits` `[heads, block, 1025]`.
/// Its params are prefixed `ss.` (the two checkpoints share names).
fn second_stage(args: &SecondStageArgs, spk_size: usize, heads: &[usize]) -> Graph {
    let (c, t) = (args.n_embd, args.block_size);
    let (v0, v1) = (args.vocab_sizes[0], args.vocab_sizes[1]);
    let hd = c / args.n_head;
    let eps = 1e-5;
    let mut g = Graph::new("second_stage");
    let h0 = g.input("h0", Shape::new(&[t], DType::I32));
    let h1 = g.input("h1", Shape::new(&[t], DType::I32));
    let spk = g.input("spk", f32s(&[1, spk_size]));
    let wte0 = g.param("ss.transformer.wtes.0.weight", f32s(&[v0, c]));
    let wte1 = g.param("ss.transformer.wtes.1.weight", f32s(&[v1, c]));
    let wpe = g.param("ss.transformer.wpe.weight", f32s(&[t, c]));
    let e0 = g.gather_(wte0, h0, 0);
    let e1 = g.gather_(wte1, h1, 0);
    let x = g.add(e0, e1);
    let x = g.add(x, wpe);
    let s = linear(&mut g, spk, "ss.speaker_cond_pos.weight", c, spk_size);
    let x = g.add(x, s);
    let mut x = g.reshape_(x, vec![1, t as i64, c as i64]);
    let zero = g.param("ss.zero", f32s(&[c]));
    for i in 0..args.n_layer {
        let p = format!("ss.transformer.h.{i}");
        let ln1 = g.param(format!("{p}.ln_1.weight"), f32s(&[c]));
        let n1 = g.ln(x, ln1, zero, eps);
        let qkv = linear(&mut g, n1, &format!("{p}.attn.c_attn.weight"), 3 * c, c);
        let q = g.narrow_(qkv, 2, 0, c);
        let k = g.narrow_(qkv, 2, c, c);
        let v = g.narrow_(qkv, 2, 2 * c, c);
        let a = g.attention_kind(q, k, v, args.n_head, hd, MaskKind::None, f32s(&[1, t, c]));
        let a = linear(&mut g, a, &format!("{p}.attn.c_proj.weight"), c, c);
        x = g.add(x, a);
        let ln2 = g.param(format!("{p}.ln_2.weight"), f32s(&[c]));
        let n2 = g.ln(x, ln2, zero, eps);
        let h = linear(&mut g, n2, &format!("{p}.mlp.c_fc.weight"), 4 * c, c);
        let h = g.gelu(h);
        let m = linear(&mut g, h, &format!("{p}.mlp.c_proj.weight"), c, 4 * c);
        x = g.add(x, m);
    }
    let lnf = g.param("ss.transformer.ln_f.weight", f32s(&[c]));
    let xf = g.ln(x, lnf, zero, eps);
    let xf = g.reshape_(xf, vec![t as i64, c as i64]);
    let mut rows = Vec::new();
    for (i, &vs) in heads.iter().enumerate() {
        let l = linear(&mut g, xf, &format!("ss.lm_heads.{i}.weight"), vs, c);
        rows.push(g.reshape_(l, vec![1, t as i64, vs as i64]));
    }
    let logits = g.concat_(rows, 0);
    g.node_mut(logits).name = Some("logits".into());
    g.set_outputs(vec![logits]);
    g
}

fn hz_to_mel(f: f64) -> f64 {
    const F_SP: f64 = 200.0 / 3.0;
    const MIN_LOG_HZ: f64 = 1000.0;
    const LOGSTEP: f64 = 0.068_751_777;
    if f < MIN_LOG_HZ {
        f / F_SP
    } else {
        MIN_LOG_HZ / F_SP + (f / MIN_LOG_HZ).ln() / LOGSTEP
    }
}

fn mel_to_hz(m: f64) -> f64 {
    const F_SP: f64 = 200.0 / 3.0;
    const MIN_LOG_HZ: f64 = 1000.0;
    const LOGSTEP: f64 = 0.068_751_777;
    if m < MIN_LOG_HZ / F_SP {
        F_SP * m
    } else {
        MIN_LOG_HZ * (LOGSTEP * (m - MIN_LOG_HZ / F_SP)).exp()
    }
}

/// The speaker encoder's Slaney filterbank, `[bins, mels]` (its own,
/// transposed).
fn speaker_filterbank() -> Vec<f32> {
    let bins = SPK_N_FFT / 2 + 1;
    let f_max = 8000.0;
    let freqs: Vec<f64> = (0..bins)
        .map(|k| k as f64 * f_max / (bins - 1) as f64)
        .collect();
    let (m_min, m_max) = (hz_to_mel(0.0), hz_to_mel(f_max));
    let pts: Vec<f64> = (0..SPK_MELS + 2)
        .map(|i| mel_to_hz(m_min + (m_max - m_min) * i as f64 / (SPK_MELS + 1) as f64))
        .collect();
    let mut fb = vec![0f32; bins * SPK_MELS];
    for m in 0..SPK_MELS {
        let (lo, ctr, hi) = (pts[m], pts[m + 1], pts[m + 2]);
        let enorm = 2.0 / (hi - lo);
        for (k, &f) in freqs.iter().enumerate() {
            let down = (f - lo) / (ctr - lo);
            let up = (hi - f) / (hi - ctr);
            fb[k * SPK_MELS + m] = down.min(up).max(0.0) as f32 * enorm as f32;
        }
    }
    fb
}

/// `speaker`: one partial's `frames` `[160, 400]` (of the reflect-padded
/// reference; zeros past its last frame) → `emb` `[1, 256]`, the
/// projection's ReLU (the program L2-normalizes and averages).
fn speaker_graph() -> Graph {
    let bins = SPK_N_FFT / 2 + 1;
    let mut g = Graph::new("speaker");
    let frames = g.input("frames", f32s(&[SPK_PARTIAL, SPK_N_FFT]));
    let window = g.param("speaker.window", f32s(&[1, SPK_N_FFT]));
    let x = g.mul(frames, window);
    let cos = g.param("speaker.cos", f32s(&[SPK_N_FFT, bins]));
    let sin = g.param("speaker.sin", f32s(&[SPK_N_FFT, bins]));
    let re = g.mm(x, cos);
    let im = g.mm(x, sin);
    let re2 = g.mul(re, re);
    let im2 = g.mul(im, im);
    let power = g.add(re2, im2);
    let fb = g.param("speaker.filterbank", f32s(&[bins, SPK_MELS]));
    let mel = g.mm(power, fb);
    let mel = g.reshape_(mel, vec![1, SPK_PARTIAL as i64, SPK_MELS as i64]);
    let w_ih = g.param(
        "speaker.lstm.w_ih",
        f32s(&[4 * SPK_HIDDEN * (SPK_MELS + (SPK_LAYERS - 1) * SPK_HIDDEN)]),
    );
    let w_hh = g.param(
        "speaker.lstm.w_hh",
        f32s(&[SPK_LAYERS * 4 * SPK_HIDDEN, SPK_HIDDEN]),
    );
    let bias = g.param("speaker.lstm.bias", f32s(&[SPK_LAYERS * 4 * SPK_HIDDEN]));
    let y = g.lstm(
        mel,
        w_ih,
        w_hh,
        bias,
        SPK_HIDDEN,
        SPK_LAYERS,
        false,
        f32s(&[1, SPK_PARTIAL, SPK_HIDDEN]),
    );
    let h = g.narrow_(y, 1, SPK_PARTIAL - 1, 1);
    let h = g.reshape_(h, vec![1, SPK_HIDDEN as i64]);
    let h = linear(&mut g, h, "speaker.linear.weight", SPK_HIDDEN, SPK_HIDDEN);
    let b = g.param("speaker.linear.bias", f32s(&[1, SPK_HIDDEN]));
    let h = g.add(h, b);
    let emb = g.relu(h);
    g.node_mut(emb).name = Some("emb".into());
    g.set_outputs(vec![emb]);
    g
}

/// The speaker graph's params from `speaker_encoder.safetensors`.
fn speaker_params(w: &WeightMap) -> Result<HashMap<String, Vec<f32>>> {
    let bins = SPK_N_FFT / 2 + 1;
    let get = |k: &str| -> Result<Vec<f32>> {
        Ok(w.get(k).with_context(|| format!("no {k}"))?.0.to_vec())
    };
    let (mut ih, mut hh, mut b) = (Vec::new(), Vec::new(), Vec::new());
    for l in 0..SPK_LAYERS {
        ih.extend(get(&format!("lstm.weight_ih_l{l}"))?);
        hh.extend(get(&format!("lstm.weight_hh_l{l}"))?);
        let (bi, bh) = (
            get(&format!("lstm.bias_ih_l{l}"))?,
            get(&format!("lstm.bias_hh_l{l}"))?,
        );
        b.extend(bi.iter().zip(&bh).map(|(x, y)| x + y));
    }
    let (mut c, mut s) = (vec![0f32; SPK_N_FFT * bins], vec![0f32; SPK_N_FFT * bins]);
    for n in 0..SPK_N_FFT {
        for k in 0..bins {
            let a = -2.0 * std::f64::consts::PI * ((k * n) % SPK_N_FFT) as f64 / SPK_N_FFT as f64;
            c[n * bins + k] = a.cos() as f32;
            s[n * bins + k] = a.sin() as f32;
        }
    }
    Ok(HashMap::from([
        ("speaker.lstm.w_ih".to_string(), ih),
        ("speaker.lstm.w_hh".to_string(), hh),
        ("speaker.lstm.bias".to_string(), b),
        ("speaker.linear.weight".to_string(), get("linear.weight")?),
        ("speaker.linear.bias".to_string(), get("linear.bias")?),
        (
            "speaker.window".to_string(),
            (0..SPK_N_FFT)
                .map(|i| {
                    0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / SPK_N_FFT as f32).cos()
                })
                .collect(),
        ),
        ("speaker.cos".to_string(), c),
        ("speaker.sin".to_string(), s),
        ("speaker.filterbank".to_string(), speaker_filterbank()),
    ]))
}

/// 16 kHz mono PCM of a WAV, as the speaker encoder reads it.
fn read_pcm(p: &Path) -> Result<Vec<f32>> {
    let mut r = hound::WavReader::open(p).with_context(|| format!("open {}", p.display()))?;
    let spec = r.spec();
    anyhow::ensure!(
        spec.sample_rate == 16_000 && spec.channels == 1,
        "{}: not 16 kHz mono",
        p.display()
    );
    Ok(match spec.sample_format {
        hound::SampleFormat::Int => {
            let max = (1i64 << (spec.bits_per_sample - 1)) as f32;
            r.samples::<i32>()
                .map(|s| s.map(|v| v as f32 / max))
                .collect::<Result<Vec<_>, _>>()?
        }
        hound::SampleFormat::Float => r.samples::<f32>().collect::<Result<Vec<_>, _>>()?,
    })
}

/// Export the MetaVoice directory `dir` (`first_stage.safetensors`,
/// `second_stage.safetensors`, `speaker_encoder.safetensors`, their args,
/// `tokenizer_metavoice.json`, `bria_16k.wav`; EnCodec from
/// `RLX_ENCODEC_PATH`, else [`DEFAULT_ENCODEC_PATH`]) as the package `out`.
pub fn export_rlxp(dir: &Path, out: &Path, opts: &ExportOptions) -> Result<ExportReport> {
    let ck = Checkpoint::open(dir)?;
    let first = load_first_args(dir)?;
    let second = load_second_args(dir)?;
    let heads = if second.target_vocab_sizes.is_empty() {
        vec![1025; 6]
    } else {
        second.target_vocab_sizes.clone()
    };
    let mut pkg =
        Package::for_checkpoint(&ck, opts, "metavoice", "text-to-speech", "rlx-metavoice");

    // The first stage, its weights referenced.
    let st = dir.join("first_stage.safetensors");
    let fw = WeightMap::from_file(st.to_str().context("weights path")?)?;
    let hidden = fw
        .get("transformer.h.0.mlp.swiglu.w1.weight")
        .context("no swiglu w1")?
        .1[0];
    // The prompt symbolic in its length and the caches' rows; the step in
    // the rows.
    let (prefill, lifted) = symbolic_in_lifted_many(
        &[(PROMPT, vec![5, 7, 3]), (CAP, vec![37, 53, 20])],
        &[first.block_size, first.block_size],
        |n| Ok(first_prefill(&first, hidden, n[0], n[1])),
    )?;
    anyhow::ensure!(lifted.is_empty(), "the prompt graph lifted tables");
    let step = symbolic_in(CAP, (37, 53, 20), |n| Ok(first_step(&first, hidden, n)))?;
    pkg.function(prefill);
    pkg.function(step);
    let mut params: HashMap<String, Vec<f32>> = fw
        .keys()
        .map(|k| (k.to_string(), fw.get(k).expect("listed").0.to_vec()))
        .collect();
    params.insert("fs.zero".into(), vec![0.0; first.n_embd]);

    // The second stage, the speaker encoder and EnCodec, held.
    let mut held: HashMap<String, Vec<f32>> = HashMap::new();
    pkg.function(second_stage(&second, first.speaker_emb_size, &heads));
    let sw = WeightMap::from_file(
        dir.join("second_stage.safetensors")
            .to_str()
            .context("weights path")?,
    )?;
    for k in sw.keys() {
        held.insert(format!("ss.{k}"), sw.get(k).expect("listed").0.to_vec());
    }
    held.insert("ss.zero".into(), vec![0.0; second.n_embd]);
    pkg.function(speaker_graph());
    let spk = WeightMap::from_file(
        dir.join("speaker_encoder.safetensors")
            .to_str()
            .context("weights path")?,
    )?;
    held.extend(speaker_params(&spk)?);
    let encodec = rlx_ir::env::var("RLX_ENCODEC_PATH")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../..")
                .join(DEFAULT_ENCODEC_PATH)
        });
    let eck =
        Checkpoint::open(&encodec).with_context(|| format!("EnCodec at {}", encodec.display()))?;
    let (graphs, encodec_params, encodec_cfg) =
        rlx_encodec::export::encodec_functions(&eck, FRAMES, CODEBOOKS)?;
    for mut g in graphs {
        g.name = format!("encodec_{}", g.name);
        pkg.function(g);
    }
    held.extend(encodec_params);
    pkg.dims = vec![
        ("frames".into(), FRAMES),
        ("codebooks".into(), CODEBOOKS),
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
    pkg.reference_map(&params, &fw, "safetensors", &st, out, false)?;
    drop(params);

    // The tokenizer and the bundled voice, read by the program.
    pkg.sidecar(
        "tokenizer_metavoice.json",
        "application/json",
        std::fs::read(dir.join("tokenizer_metavoice.json"))?,
    );
    let bria = read_pcm(&dir.join("bria_16k.wav"))?;
    pkg.sidecar(
        "voice/bria.f32",
        "application/octet-stream",
        bria.iter().flat_map(|x| x.to_le_bytes()).collect(),
    );

    let module = program::build_for(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("program/Cargo.toml"),
        "wasm32-wasip1",
    )
    .context("build MetaVoice's program")?;
    program::entry(
        &mut pkg,
        "speak",
        module,
        &["synthesize", "run"],
        "text-to-speech",
        json!({
            "tokenizer": "/package/tokenizer_metavoice.json",
            "voice": "/package/voice/bria.f32",
            "layers": first.n_layer,
            "block_size": first.block_size,
            "second_block_size": second.block_size,
            "eos": EOS_AUDIO,
            "pad": PAD,
            "text_offset": TEXT_OFFSET,
            "max_new_tokens": 864,
            "guidance_scale": 3.0,
            "temperature": 1.0,
            "top_p": 0.95,
            "seed": 1337,
            "greedy": true,
            "sample_rate": SAMPLE_RATE,
            "codebooks": 8,
            "spk_n_fft": SPK_N_FFT,
            "spk_hop": SPK_HOP,
            "spk_partial": SPK_PARTIAL,
            "hop": encodec_cfg.hop_length(),
        }),
    );
    pkg.metadata("sample_rate", json!(SAMPLE_RATE));
    pkg.write(out, opts)
}

/// Register this crate's exporter ([`rlx_core::export::register_exporter`]).
pub fn register() {
    use rlx_core::export::{Exporter, register_exporter};
    register_exporter(Exporter::new("metavoice", "text-to-speech", export_rlxp));
}
