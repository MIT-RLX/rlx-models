// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only

//! Export Silero VAD or Earshot as an `.rlxp` package rlx runs alone: one
//! frame of the network as an rlx graph (`step`: the frame and the
//! recurrent state in; the speech probability and the next state out —
//! [`crate::silero::forward_frame`] / Earshot's `Detector::predict_f32`
//! op for op), and a program answering `run` with what
//! `speech_segments_silero` / `speech_segments_earshot` do around it: the
//! frames, the context and state carried between them, the segments.
//! `Model.run({audio: f32 [samples]})` returns `probs` and `segments`.

use std::collections::HashMap;
use std::path::Path;

use anyhow::Result;
pub use rlx_core::export::{ExportOptions, ExportReport};
use rlx_core::export::{Package, program};
use rlx_ir::{DType, Graph, GraphExt, NodeId, Op, Shape};
use serde_json::json;

use crate::silero::{CONTEXT_16K, FRAME_SAMPLES_16K, SileroWeights};

const HIDDEN: usize = 128;
const STFT_K: usize = 128;
const STFT_STRIDE: usize = 64;
const STFT_PAD_RIGHT: usize = 32;
const BINS: usize = 65;

fn f32s(d: &[usize]) -> Shape {
    Shape::new(d, DType::F32)
}

/// `x` `[1, c, 1, t]` through a 1-D convolution (`w` `[o, c, k]`), its bias
/// added, ReLU'd.
fn conv(
    g: &mut Graph,
    x: NodeId,
    name: &str,
    (o, c, k): (usize, usize, usize),
    (stride, pad): (usize, usize),
    bias: bool,
) -> NodeId {
    let w = g.param(format!("{name}.weight"), f32s(&[o, c, 1, k]));
    let y = g.conv2d(x, w, [1, k], [1, stride], [0, pad], [1, 1], 1);
    if !bias {
        return y;
    }
    let b = g.param(format!("{name}.bias"), f32s(&[o]));
    let b = g.reshape_(b, vec![1, o as i64, 1, 1]);
    let y = g.add(y, b);
    g.relu(y)
}

/// One frame of Silero (16 kHz): `frame` `[576]`, `h` / `c` `[128]` →
/// `prob` `[1]`, `h`, `c`.
fn step() -> Graph {
    let t_in = CONTEXT_16K + FRAME_SAMPLES_16K;
    let mut g = Graph::new("step");
    let frame = g.input("frame", f32s(&[t_in]));
    let h = g.input("h", f32s(&[HIDDEN]));
    let c = g.input("c", f32s(&[HIDDEN]));
    // Reflect 32 samples on the right, the edge excluded: x[t−2], x[t−3], ….
    let idx: Vec<f32> = (0..STFT_PAD_RIGHT).map(|i| (t_in - 2 - i) as f32).collect();
    let data = idx.iter().flat_map(|v| v.to_le_bytes()).collect();
    let idx = g.add_node(Op::Constant { data }, vec![], f32s(&[STFT_PAD_RIGHT]));
    let idx = g.add_node(
        Op::Cast { to: DType::I32 },
        vec![idx],
        Shape::new(&[STFT_PAD_RIGHT], DType::I32),
    );
    let tail = g.gather(frame, idx, 0, f32s(&[STFT_PAD_RIGHT]));
    let x = g.concat_(vec![frame, tail], 0);
    let t = t_in + STFT_PAD_RIGHT;
    let x = g.reshape_(x, vec![1, 1, 1, t as i64]);
    let spec = conv(
        &mut g,
        x,
        "stft_conv",
        (2 * BINS, 1, STFT_K),
        (STFT_STRIDE, 0),
        false,
    );
    let ts = (t - STFT_K) / STFT_STRIDE + 1;
    let re = g.narrow_(spec, 1, 0, BINS);
    let im = g.narrow_(spec, 1, BINS, BINS);
    let re2 = g.mul(re, re);
    let im2 = g.mul(im, im);
    let p = g.add(re2, im2);
    let mag = g.sqrt(p);
    let x = conv(&mut g, mag, "conv1", (128, BINS, 3), (1, 1), true);
    let x = conv(&mut g, x, "conv2", (64, 128, 3), (2, 1), true);
    let x = conv(&mut g, x, "conv3", (64, 64, 3), (2, 1), true);
    let x = conv(&mut g, x, "conv4", (128, 64, 3), (1, 1), true);
    // The last time step.
    let mut tc = ts;
    for (k, s, pd) in [(3, 1, 1), (3, 2, 1), (3, 2, 1), (3, 1, 1)] {
        tc = (tc + 2 * pd - k) / s + 1;
    }
    let x = g.narrow_(x, 3, tc - 1, 1);
    let x = g.reshape_(x, vec![HIDDEN as i64, 1]);
    // The LSTM cell, gates in the weights' order: i, f, o, g.
    let wih = g.param("lstm_cell.weight_ih", f32s(&[4 * HIDDEN, HIDDEN]));
    let whh = g.param("lstm_cell.weight_hh", f32s(&[4 * HIDDEN, HIDDEN]));
    let bih = g.param("lstm_cell.bias_ih", f32s(&[4 * HIDDEN]));
    let bhh = g.param("lstm_cell.bias_hh", f32s(&[4 * HIDDEN]));
    let hc = g.reshape_(h, vec![HIDDEN as i64, 1]);
    let gi = g.mm(wih, x);
    let gi = g.reshape_(gi, vec![(4 * HIDDEN) as i64]);
    let gi = g.add(gi, bih);
    let gh = g.mm(whh, hc);
    let gh = g.reshape_(gh, vec![(4 * HIDDEN) as i64]);
    let gates = g.add(gi, gh);
    let gates = g.add(gates, bhh);
    let part = |g: &mut Graph, i: usize| g.narrow_(gates, 0, i * HIDDEN, HIDDEN);
    let ig = part(&mut g, 0);
    let ig = g.sigmoid(ig);
    let fg = part(&mut g, 1);
    let fg = g.sigmoid(fg);
    let og = part(&mut g, 2);
    let og = g.sigmoid(og);
    let cg = part(&mut g, 3);
    let cg = g.tanh(cg);
    let fc = g.mul(fg, c);
    let icg = g.mul(ig, cg);
    let c2 = g.add(fc, icg);
    let tc2 = g.tanh(c2);
    let h2 = g.mul(og, tc2);
    let r = g.relu(h2);
    let r = g.reshape_(r, vec![1, HIDDEN as i64]);
    let fw = g.param("final_conv.weight", f32s(&[HIDDEN, 1]));
    let fb = g.param("final_conv.bias", f32s(&[1]));
    let logit = g.mm(r, fw);
    let logit = g.reshape_(logit, vec![1]);
    let logit = g.add(logit, fb);
    let prob = g.sigmoid(logit);
    g.node_mut(prob).name = Some("prob".into());
    g.node_mut(h2).name = Some("h".into());
    g.node_mut(c2).name = Some("c".into());
    g.set_outputs(vec![prob, h2, c2]);
    g
}

/// Export Silero VAD (16 kHz): the embedded weights, or the
/// `silero_vad_16k.safetensors` at `path` when it is a file.
pub fn export_rlxp(path: &Path, out: &Path, opts: &ExportOptions) -> Result<ExportReport> {
    let w = if path.is_file() {
        SileroWeights::load(path)?
    } else {
        SileroWeights::embedded()
    };
    let params: HashMap<String, Vec<f32>> = [
        ("stft_conv.weight", w.stft_conv),
        ("conv1.weight", w.conv1_w),
        ("conv1.bias", w.conv1_b),
        ("conv2.weight", w.conv2_w),
        ("conv2.bias", w.conv2_b),
        ("conv3.weight", w.conv3_w),
        ("conv3.bias", w.conv3_b),
        ("conv4.weight", w.conv4_w),
        ("conv4.bias", w.conv4_b),
        ("lstm_cell.weight_ih", w.lstm_w_ih),
        ("lstm_cell.weight_hh", w.lstm_w_hh),
        ("lstm_cell.bias_ih", w.lstm_b_ih),
        ("lstm_cell.bias_hh", w.lstm_b_hh),
        ("final_conv.weight", w.final_w),
        ("final_conv.bias", w.final_b),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v))
    .collect();
    let mut pkg = Package::new(
        "silero-vad",
        &opts.version,
        "silero-vad",
        "voice-activity-detection",
        "rlx-vad",
    );
    pkg.function(step());
    let dims = pkg.param_dims();
    let mut names: Vec<&String> = params.keys().collect();
    names.sort();
    for n in names {
        let shape = dims
            .get(n)
            .cloned()
            .unwrap_or_else(|| vec![params[n].len()]);
        pkg.hold_f32(n.clone(), shape, &params[n]);
    }
    let module = program::build(&Path::new(env!("CARGO_MANIFEST_DIR")).join("program/Cargo.toml"))?;
    program::entry(
        &mut pkg,
        "vad",
        module,
        &["run"],
        "voice-activity-detection",
        json!({
            "frame": FRAME_SAMPLES_16K,
            "context": CONTEXT_16K,
            "states": [{"name": "h", "size": HIDDEN}, {"name": "c", "size": HIDDEN}],
            "algorithm": "silero",
            "sample_rate": 16_000,
        }),
    );
    pkg.metadata("sample_rate", json!(16_000));
    pkg.write(out, opts)
}

fn hold(pkg: &mut Package, params: &HashMap<String, Vec<f32>>) {
    let dims = pkg.param_dims();
    let mut names: Vec<&String> = params.keys().collect();
    names.sort();
    for n in names {
        let shape = dims
            .get(n)
            .cloned()
            .unwrap_or_else(|| vec![params[n].len()]);
        pkg.hold_f32(n.clone(), shape, &params[n]);
    }
}

fn constant(g: &mut Graph, values: &[f32]) -> NodeId {
    let data = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    g.add_node(Op::Constant { data }, vec![], f32s(&[values.len()]))
}

/// The Earshot frame and its state: the ring of pre-emphasized samples, the
/// last raw sample, three frames of features, the two MinGRUs' state.
const RING: usize = 768;
const EARSHOT_FRAME: usize = 256;
const MELS: usize = 40;
const FEATS: usize = MELS * 3;
const RNN: usize = 128;
const EARSHOT_BINS: usize = 513;

/// One Earshot frame: `frame` `[256]`, `ring` `[768]`, `prev` `[1]`,
/// `feats` `[120]`, `state` `[128]` → `prob` `[1]` and their next values.
fn earshot_step() -> Graph {
    let mut g = Graph::new("step");
    let frame = g.input("frame", f32s(&[EARSHOT_FRAME]));
    let ring = g.input("ring", f32s(&[RING]));
    let prev = g.input("prev", f32s(&[1]));
    let feats = g.input("feats", f32s(&[FEATS]));
    let state = g.input("state", f32s(&[RNN]));
    // Pre-emphasis on the 16-bit-scaled samples, from the last frame's last.
    let k = constant(&mut g, &[32768.0]);
    let s = g.mul(frame, k);
    let before = g.narrow_(s, 0, 0, EARSHOT_FRAME - 1);
    let before = g.concat_(vec![prev, before], 0);
    let k = constant(&mut g, &[0.97]);
    let kb = g.mul(before, k);
    let emph = g.sub(s, kb);
    let kept = g.narrow_(ring, 0, EARSHOT_FRAME, RING - EARSHOT_FRAME);
    let ring2 = g.concat_(vec![kept, emph], 0);
    let prev2 = g.narrow_(s, 0, EARSHOT_FRAME - 1, 1);
    // Windowed (in the bases) 1024-point power spectrum, the mel bands.
    let x = g.reshape_(ring2, vec![1, RING as i64]);
    let bc = g.param("stft.cos", f32s(&[RING, EARSHOT_BINS]));
    let bs = g.param("stft.sin", f32s(&[RING, EARSHOT_BINS]));
    let re = g.mm(x, bc);
    let im = g.mm(x, bs);
    let re2 = g.mul(re, re);
    let im2 = g.mul(im, im);
    let p = g.add(re2, im2);
    let k = constant(&mut g, &[1.0 / (32768.0f32 * 32768.0)]);
    let p = g.mul(p, k);
    let fb = g.param("mel.filters", f32s(&[EARSHOT_BINS, MELS]));
    let mel = g.mm(p, fb);
    let k = constant(&mut g, &[1e-20]);
    let mel = g.add(mel, k);
    let cur = g.log(mel);
    let cur = g.reshape_(cur, vec![MELS as i64]);
    // RMS normalization, the norm weight.
    let c2 = g.mul(cur, cur);
    let ms = g.reduce(c2, rlx_ir::op::ReduceOp::Mean, vec![0], true, f32s(&[1]));
    let rms = g.sqrt(ms);
    let one = constant(&mut g, &[1.0]);
    let irms = g.div(one, rms);
    let nw = g.param("norm.weight", f32s(&[MELS]));
    let cur = g.mul(nw, cur);
    let cur = g.mul(cur, irms);
    let old = g.narrow_(feats, 0, MELS, FEATS - MELS);
    let feats2 = g.concat_(vec![old, cur], 0);
    // Layer 1: a 3×3 kernel over the [3, 40] features, an affine per
    // channel, max-pool 3/2 (from 0: a ReLU too).
    let f = g.reshape_(feats2, vec![1, 1, 3, MELS as i64]);
    let k1 = g.param("layer1.kernel", f32s(&[1, 1, 3, 3]));
    let row = g.conv2d(f, k1, [3, 3], [1, 1], [0, 0], [1, 1], 1);
    let w1 = g.param("layer1.weight", f32s(&[1, 16, 1, 1]));
    let b1 = g.param("layer1.bias", f32s(&[1, 16, 1, 1]));
    let x = g.mul(row, w1);
    let x = g.add(x, b1);
    let x = g.pool2d(x, rlx_ir::op::ReduceOp::Max, [1, 3], [1, 2], [0, 0]);
    let x = g.relu(x);
    // Layers 2 and 3: depthwise 3/2 (padded by 1), then 16 × 16, bias, ReLU.
    let layer = |g: &mut Graph, x: NodeId, n: usize| {
        let k = g.param(format!("layer{n}.kernel"), f32s(&[16, 1, 1, 3]));
        let d = g.conv2d(x, k, [1, 3], [1, 2], [0, 1], [1, 1], 16);
        let w = g.param(format!("layer{n}.weight"), f32s(&[16, 16, 1, 1]));
        let y = g.conv2d(d, w, [1, 1], [1, 1], [0, 0], [1, 1], 1);
        let b = g.param(format!("layer{n}.bias"), f32s(&[1, 16, 1, 1]));
        let y = g.add(y, b);
        g.relu(y)
    };
    let x = layer(&mut g, x, 2);
    let x = layer(&mut g, x, 3);
    // [1, 16, 1, 5] → [5, 16] → 80 (position-major).
    let x = g.reshape_(x, vec![16, 5]);
    let x = g.transpose_(x, vec![1, 0]);
    let x = g.reshape_(x, vec![80, 1]);
    let mingru = |g: &mut Graph, x: NodeId, h: NodeId, name: &str, inp: usize| {
        let w = g.param(name.to_string(), f32s(&[RNN, inp]));
        let o = g.mm(w, x);
        let o = g.reshape_(o, vec![RNN as i64]);
        let v = g.narrow_(o, 0, 0, 64);
        let gt = g.narrow_(o, 0, 64, 64);
        let q = constant(g, &[0.25]);
        let gt = g.mul(gt, q);
        let gt = g.clamp_(gt, 0.0, 1.0);
        let one = constant(g, &[1.0]);
        let inv = g.sub(one, gt);
        let a = g.mul(inv, h);
        let b = g.mul(gt, v);
        g.add(a, b)
    };
    let h1 = g.narrow_(state, 0, 0, 64);
    let h2 = g.narrow_(state, 0, 64, 64);
    let n1 = mingru(&mut g, x, h1, "rnn1.weight", 80);
    let x2 = g.reshape_(n1, vec![64, 1]);
    let n2 = mingru(&mut g, x2, h2, "rnn2.weight", 64);
    let state2 = g.concat_(vec![n1, n2], 0);
    let ow = g.param("output.weight", f32s(&[1, RNN]));
    let st = g.reshape_(state2, vec![RNN as i64, 1]);
    let logit = g.mm(ow, st);
    let logit = g.reshape_(logit, vec![1]);
    let prob = g.sigmoid(logit);
    g.node_mut(prob).name = Some("prob".into());
    g.set_outputs(vec![prob, ring2, prev2, feats2, state2]);
    g
}

/// Export Earshot (its embedded weights) as the package `out`.
pub fn export_earshot(out: &Path, opts: &ExportOptions) -> Result<ExportReport> {
    use crate::earshot::{filters, weights};
    let w = weights::weights();
    let mut params: HashMap<String, Vec<f32>> = [
        ("norm.weight", w.norm_weight),
        ("layer1.kernel", w.layer1_kernel),
        ("layer1.weight", w.layer1_weight),
        ("layer1.bias", w.layer1_bias),
        ("layer2.kernel", w.layer2_kernel),
        ("layer2.weight", w.layer2_weight),
        ("layer2.bias", w.layer2_bias),
        ("layer3.kernel", w.layer3_kernel),
        ("layer3.weight", w.layer3_weight),
        ("layer3.bias", w.layer3_bias),
        ("rnn1.weight", w.rnn1_weight),
        ("rnn2.weight", w.rnn2_weight),
        ("output.weight", w.output_weight),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v.to_vec()))
    .collect();
    // A 1024-point DFT of the windowed 768 samples (the rest zero).
    let (mut c, mut sn) = (
        vec![0f32; RING * EARSHOT_BINS],
        vec![0f32; RING * EARSHOT_BINS],
    );
    for n in 0..RING {
        let win = f64::from(filters::HANN_WINDOW[n]);
        for k in 0..EARSHOT_BINS {
            let ang = 2.0 * std::f64::consts::PI * ((k * n) % 1024) as f64 / 1024.0;
            c[n * EARSHOT_BINS + k] = (win * ang.cos()) as f32;
            sn[n * EARSHOT_BINS + k] = (-win * ang.sin()) as f32;
        }
    }
    params.insert("stft.cos".into(), c);
    params.insert("stft.sin".into(), sn);
    let mut fb = vec![0f32; EARSHOT_BINS * MELS];
    for (m, (start, coeffs)) in filters::MEL_COEFFS.iter().enumerate() {
        for (o, &v) in coeffs.iter().enumerate() {
            fb[(start + o) * MELS + m] = v;
        }
    }
    params.insert("mel.filters".into(), fb);
    let mut pkg = Package::new(
        "earshot",
        &opts.version,
        "earshot",
        "voice-activity-detection",
        "rlx-vad",
    );
    pkg.function(earshot_step());
    hold(&mut pkg, &params);
    let module = program::build(&Path::new(env!("CARGO_MANIFEST_DIR")).join("program/Cargo.toml"))?;
    program::entry(
        &mut pkg,
        "vad",
        module,
        &["run"],
        "voice-activity-detection",
        json!({
            "frame": EARSHOT_FRAME,
            "context": 0,
            "states": [
                {"name": "ring", "size": RING},
                {"name": "prev", "size": 1},
                {"name": "feats", "size": FEATS},
                {"name": "state", "size": RNN},
            ],
            "algorithm": "earshot",
            "sample_rate": 16_000,
        }),
    );
    pkg.metadata("sample_rate", json!(16_000));
    pkg.write(out, opts)
}

/// Register this crate's exporters ([`rlx_core::export::register_exporter`]),
/// taken by name (`--family silero-vad`, `--family earshot`).
pub fn register() {
    use rlx_core::export::{Exporter, register_exporter};
    register_exporter(Exporter::new(
        "silero-vad",
        "voice-activity-detection",
        export_rlxp,
    ));
    register_exporter(Exporter::new(
        "earshot",
        "voice-activity-detection",
        |_, out, opts| export_earshot(out, opts),
    ));
}
