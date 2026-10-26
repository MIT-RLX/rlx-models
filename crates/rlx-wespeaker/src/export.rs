// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only

//! Export WeSpeaker as an `.rlxp` package rlx runs alone: a `pipeline` from
//! 16 kHz audio to its L2-normalized 256-d speaker embedding, every step a
//! graph — the Kaldi fbank of [`crate::log_mel_fbank`] (scaled to 16-bit,
//! the mean removed, pre-emphasis, the Povey window, the power spectrum
//! through its filters, `ln(max(e, 1e-10))`) at the network's
//! [`FIXED_FRAMES`] (later frames dropped, missing ones zero, as
//! [`crate::WeSpeaker`] fits them), the network (imported once from the
//! model's `graphs/wespeaker.rlxp`), and the normalization.
//! `Model.run({audio: f32 [samples]})` returns `embedding`, `f32 [256]`.

use std::path::Path;

use anyhow::Result;
use rlx_core::export::graph::{Pipeline, Stage};
pub use rlx_core::export::{ExportOptions, ExportReport};
use rlx_ir::op::{BinaryOp, CmpOp, ReduceOp};
use rlx_ir::{DType, Graph, GraphExt, NodeId, Op, Shape};

use crate::{EMBED_DIM, FIXED_FRAMES};

const SHIFT: usize = 160;
const LENGTH: usize = 400;
const N_FFT: usize = 512;
const N_MELS: usize = 80;
/// The audio symbol.
const SAMPLES: u32 = 0;

fn f32s(d: &[usize]) -> Shape {
    Shape::new(d, DType::F32)
}

fn constant(g: &mut Graph, values: &[f32]) -> NodeId {
    let data = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    g.add_node(Op::Constant { data }, vec![], f32s(&[values.len()]))
}

fn fbank_at(n: usize) -> Graph {
    let mut g = Graph::new("fbank");
    let audio = g.input("audio", f32s(&[n]));
    let k = constant(&mut g, &[32768.0]);
    let x = g.mul(audio, k);
    let mean = g.reduce(x, ReduceOp::Mean, vec![0], true, f32s(&[1]));
    let x = g.sub(x, mean);
    let head = g.narrow_(x, 0, 0, 1);
    let cur = g.narrow_(x, 0, 1, n - 1);
    let prev = g.narrow_(x, 0, 0, n - 1);
    let k = constant(&mut g, &[0.97]);
    let kp = g.mul(prev, k);
    let d = g.sub(cur, kp);
    let x = g.concat_(vec![head, d], 0);
    // The samples, counted: `sum(x·0 + 1)`, and the frames they make.
    let zero = constant(&mut g, &[0.0]);
    let one = constant(&mut g, &[1.0]);
    let ones = g.mul(x, zero);
    let ones = g.add(ones, one);
    let count = g.reduce(ones, ReduceOp::Sum, vec![0], true, f32s(&[1]));
    let len = constant(&mut g, &[LENGTH as f32]);
    let shift = constant(&mut g, &[SHIFT as f32]);
    let over = g.sub(count, len);
    let over = g.div(over, shift);
    let over = g.add_node(
        Op::Activation(rlx_ir::op::Activation::Floor),
        vec![over],
        f32s(&[1]),
    );
    let frames = g.add(over, one);
    // Rows `f · 160 + j` (`j < 512`; the window is zero past 400), indices
    // past the audio clamped (those rows are masked below).
    let fi: Vec<f32> = (0..FIXED_FRAMES).map(|f| f as f32).collect();
    let fi = constant(&mut g, &fi);
    let fi = g.reshape_(fi, vec![FIXED_FRAMES as i64, 1]);
    let start = g.mul(fi, shift);
    let ji: Vec<f32> = (0..N_FFT).map(|j| j as f32).collect();
    let ji = constant(&mut g, &ji);
    let ji = g.reshape_(ji, vec![1, N_FFT as i64]);
    let at = g.add(start, ji);
    let last = g.sub(count, one);
    let at = g.binary(BinaryOp::Min, at, last, f32s(&[FIXED_FRAMES, N_FFT]));
    let at = g.add_node(
        Op::Cast { to: DType::I32 },
        vec![at],
        Shape::new(&[FIXED_FRAMES, N_FFT], DType::I32),
    );
    let windows = g.gather(x, at, 0, f32s(&[FIXED_FRAMES, N_FFT]));
    let n_freq = N_FFT / 2 + 1;
    let bc = g.param("stft.cos", f32s(&[N_FFT, n_freq]));
    let bs = g.param("stft.sin", f32s(&[N_FFT, n_freq]));
    let re = g.mm(windows, bc);
    let im = g.mm(windows, bs);
    let re2 = g.mul(re, re);
    let im2 = g.mul(im, im);
    let power = g.add(re2, im2);
    let fb = g.param("mel.filters", f32s(&[n_freq, N_MELS]));
    let mel = g.mm(power, fb);
    let floor = constant(&mut g, &[1e-10]);
    let mel = g.binary(BinaryOp::Max, mel, floor, f32s(&[FIXED_FRAMES, N_MELS]));
    let mel = g.log(mel);
    // Frames the audio does not make are zero.
    let frames = g.reshape_(frames, vec![1, 1]);
    let keep = g.add_node(
        Op::Compare(CmpOp::Lt),
        vec![fi, frames],
        Shape::new(&[FIXED_FRAMES, 1], DType::Bool),
    );
    let keep = g.add_node(
        Op::Cast { to: DType::F32 },
        vec![keep],
        f32s(&[FIXED_FRAMES, 1]),
    );
    let mel = g.mul(mel, keep);
    let out = g.reshape_(mel, vec![1, FIXED_FRAMES as i64, N_MELS as i64]);
    g.node_mut(out).name = Some("fbank".into());
    g.set_outputs(vec![out]);
    g
}

/// `audio` `[samples]` (16 kHz, at least 25 ms) → `fbank` `[1, 148, 80]`.
fn fbank() -> Result<Graph> {
    let (a, b) = (fbank_at(24_037), fbank_at(31_013));
    let g = rlx_ir::dynamic::generalize(&a, &b, &[(SAMPLES, 24_037, 31_013)])
        .map_err(|e| anyhow::anyhow!("the fbank is not symbolic in the audio: {e}"))?;
    Ok(g)
}

/// The network's output → `embedding`: its first 256 values, L2-normalized.
fn normalize(width: usize) -> Graph {
    let mut g = Graph::new("normalize");
    let e = g.input("output", f32s(&[width]));
    let e = g.narrow_(e, 0, 0, EMBED_DIM);
    let e2 = g.mul(e, e);
    let ss = g.reduce(e2, ReduceOp::Sum, vec![0], true, f32s(&[1]));
    let norm = g.sqrt(ss);
    // As `l2_normalize`: unchanged below a norm of 1e-8.
    let tiny = constant(&mut g, &[1e-8]);
    let big = g.add_node(
        Op::Compare(CmpOp::Gt),
        vec![norm, tiny],
        Shape::new(&[1], DType::Bool),
    );
    let big = g.add_node(Op::Cast { to: DType::F32 }, vec![big], f32s(&[1]));
    let one = constant(&mut g, &[1.0]);
    let small = g.sub(one, big);
    let by = g.mul(norm, big);
    let by = g.add(by, small);
    let out = g.div(e, by);
    g.node_mut(out).name = Some("embedding".into());
    g.set_outputs(vec![out]);
    g
}

/// Export the WeSpeaker model in `dir` (`graphs/wespeaker.rlxp`) as the
/// package `out`.
pub fn export_rlxp(dir: &Path, out: &Path, opts: &ExportOptions) -> Result<ExportReport> {
    let (mut network, mut params) = rlx_tiny_tts::model::import_rlx_graph(
        &dir.join("graphs/wespeaker.rlxp"),
        "wespeaker",
        FIXED_FRAMES,
        &[],
    )?;
    network.name = "network".into();
    let input = network
        .nodes()
        .iter()
        .find_map(|n| match &n.op {
            Op::Input { name } => Some(name.clone()),
            _ => None,
        })
        .unwrap_or_else(|| "fbank".into());
    let width = network
        .node(network.outputs[0])
        .shape
        .num_elements()
        .unwrap_or(EMBED_DIM);
    let mut window = crate::fbank::povey_window();
    window.resize(N_FFT, 0.0);
    params.extend(rlx_core::export::audio::stft_bases(N_FFT, &window));
    let filters = crate::fbank::mel_filterbank();
    let n_freq = N_FFT / 2 + 1;
    let mut fb = vec![0f32; n_freq * N_MELS];
    for (m, f) in filters.iter().enumerate() {
        for (k, &w) in f.iter().enumerate() {
            fb[k * N_MELS + m] = w;
        }
    }
    params.insert("mel.filters".into(), fb);
    let stages = vec![
        Stage::new(fbank()?).outputs(&["fbank"]),
        Stage::new(network)
            .input(&input, "fbank")
            .outputs(&["output"]),
        Stage::new(normalize(width)).outputs(&["embedding"]),
    ];
    let mut p = Pipeline::new("wespeaker", "audio-classification", stages);
    p.outputs = vec!["embedding".into()];
    p.dims = vec![("samples".into(), SAMPLES)];
    p.metadata = vec![("sample_rate".into(), serde_json::json!(16_000))];
    p.exported_by = "rlx-wespeaker".into();
    p.write(params, None, out, opts)
}

/// Register this crate's exporter ([`rlx_core::export::register_exporter`]),
/// taken by name (`--family wespeaker`).
pub fn register() {
    use rlx_core::export::{Exporter, register_exporter};
    register_exporter(Exporter::new(
        "wespeaker",
        "audio-classification",
        export_rlxp,
    ));
}
