// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Several wake phrases in one network, trained on an RLX device with the optimizer on the
//! device too.
//!
//! The model is [`crate::cnn::WakeCnnWeights`] with `K` outputs (the last layer's width), one
//! sigmoid per phrase: a clip of phrase `c` is a positive for output `c` and a negative for the
//! others, everything else a negative for all. One network instead of one per phrase: one
//! forward per hop at run time, one weight file, and the phrases learn from each other's clips.
//!
//! Faster than [`super::graph_cnn`] where it was slow:
//! - the AdamW update is part of the compiled step ([`rlx_runtime::train::ResidentTrainer`]):
//!   weights and moments stay in device buffers, only the loss comes back;
//! - the next batch is cut on every core while the device runs the current one;
//! - mel features come from a cache on disk ([`load_mels_cached`]): repeat runs skip decoding
//!   and FFTs.
//!
//! Sampling is [`super::graph_cnn`]'s: positives' windows end at the phrase's end (jittered),
//! negatives' anywhere in the clip, plus partial-phrase windows of positives (negatives) and,
//! from `mine_after`, the negative windows the network currently scores highest on any output.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use rlx_ir::{DType, Graph, GraphExt, NodeId, Shape};
use rlx_runtime::train::{AdamSpec, ResidentTrainer, TrainableParam};
use rlx_runtime::{Device, Session};

use crate::cnn::{WakeCnnConfig, WakeCnnWeights};
use crate::train::dataset::clip_mel_frames;

/// A clip's mel frames (frame-major, `n_mels` each) and its phrase, if any.
#[derive(Debug, Clone)]
pub struct MultiClip {
    pub mel: Vec<f32>,
    pub class: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct MultiTrainConfig {
    pub epochs: usize,
    pub batch: usize,
    pub lr: f32,
    pub weight_decay: f32,
    /// Window, mel frames (10 ms each).
    pub window_frames: usize,
    /// Partial-phrase windows (negatives) per positive per epoch, and how far before the
    /// phrase's end they stop (frames).
    pub partial_ratio: f32,
    pub partial_gap: usize,
    /// Background frames kept after a clip, so windows can end past the phrase.
    pub tail_frames: usize,
    pub mine_every: usize,
    pub mine_after: usize,
    pub mine_stride: usize,
    pub mine_threshold: f32,
    pub mine_ratio: f32,
    /// Each training window louder or softer by up to this many dB (uniform), as if said
    /// nearer or further: a constant on every log-mel bin. 0 = off.
    pub gain_db: f32,
    /// Each training window, with this probability, gets a stretch of one of the noise clips
    /// ([`train_multi_with_noise`]) added to it in the power domain, at a signal-to-noise ratio
    /// (dB, against the window's mean power) drawn from `noise_snr_db` for positives and from
    /// 10 dB lower for negatives (noise may drown them). 0 = off.
    pub noise_p: f32,
    pub noise_snr_db: (f32, f32),
    /// SpecAugment: with this probability a window loses a band of up to 4 mel bins and a
    /// stretch of up to 10 frames (each set to the window's mean). 0 = off.
    pub mask_p: f32,
    /// Positives' windows end up to this many frames after the clip (into the background
    /// kept after it, `tail_frames`), not only at its end: the network learns to keep firing
    /// for the hops after the phrase, which a detection's `min_hops` needs. 0 = at the end.
    pub pos_tail: usize,
    /// Batch normalization after every convolution while training (batch statistics),
    /// calibrated on training windows and folded into the convolutions for scoring and export.
    pub bn: bool,
    /// With a PCEN front end (`cfg.pcen` of the network), each window's PCEN starts this many
    /// frames earlier, so its smoother has settled as on a stream.
    pub pcen_warm: usize,
    /// With teachers ([`train_multi_taught`]): each taught output's target is
    /// `(1 − w)·label + w·teacher score` (knowledge distillation).
    pub teacher_weight: f32,
    pub log_every: usize,
}

/// A trained network whose scores become part of the targets for some outputs (its outputs of
/// the same index). It reads the student's batch: the same input, or one half of a dual one
/// (its log-mel or PCEN half, as its own front end says).
pub struct Teacher {
    pub weights: WakeCnnWeights,
    pub outputs: Vec<usize>,
    /// The window it was trained on (frames): it reads the last this many of the student's
    /// (a causal student reads more). `None`: the student's.
    pub window_frames: Option<usize>,
}

impl Default for MultiTrainConfig {
    fn default() -> Self {
        Self {
            epochs: 100,
            batch: 1024,
            lr: 4e-3,
            weight_decay: 1e-4,
            window_frames: 120,
            partial_ratio: 1.0,
            partial_gap: 50,
            tail_frames: 60,
            mine_every: 10,
            mine_after: 20,
            mine_stride: 8,
            mine_threshold: 0.2,
            mine_ratio: 0.5,
            gain_db: 12.0,
            noise_p: 0.0,
            noise_snr_db: (0.0, 25.0),
            mask_p: 0.0,
            pos_tail: 0,
            bn: false,
            pcen_warm: 100,
            teacher_weight: 0.5,
            log_every: 5,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct MultiReport {
    pub initial_loss: f32,
    pub final_loss: f32,
    /// The optimizer ran on the device (not the host-chained fallback).
    pub resident: bool,
    pub secs_per_epoch: f64,
    /// Per phrase: training positives detected at 0.5, of how many.
    pub recall: Vec<(usize, usize)>,
    /// Training negatives (end of clip) firing any output at 0.5, of how many.
    pub negatives_firing: (usize, usize),
}

/// A positive's window ends within this many frames of the clip's end.
const POS_JITTER: usize = 3;

fn sh(dims: &[usize]) -> Shape {
    Shape::new(dims, DType::F32)
}

/// Batch normalization's epsilon (training graph and folding).
const BN_EPS: f32 = 1e-5;

/// Which form of the network a graph builds. `Infer`: the engine's (conv biases, no batch
/// normalization). `Train`: batch normalization (batch statistics) after every convolution,
/// whose weights then carry no bias; [`fold_bn`] turns its parameters into `Infer`'s.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Form {
    Infer,
    Train,
}

/// A built network: logits `[B, K]`, its parameters `(name, node)`, and (`Train`) each
/// batch-normalized convolution's output before normalization, in layer order.
struct Net {
    z: NodeId,
    params: Vec<(String, NodeId)>,
    bn_in: Vec<NodeId>,
}

/// The batch-normalized layers' names, in order: the three stem convolutions, then the
/// residual blocks'.
fn bn_layers(c: &WakeCnnConfig) -> Vec<String> {
    (1..=3)
        .map(|l| format!("conv{l}"))
        .chain((0..c.res_blocks).map(|i| format!("res{i}")))
        .collect()
}

/// The CNN over `x [B, T, n_mels]` (frames as stored: a batch is plain copies; the transpose
/// to channel-major is the device's). The 1-D convolutions are `conv2d` over `[B, C, 1, T]`
/// with `1 × k` kernels, in the engine's weight layout. (Rewriting them as GEMMs over shifted
/// views was slower on Metal at these shapes: tall, skinny products plus the slicing.)
fn body(g: &mut Graph, c: &WakeCnnConfig, n_out: usize, b: usize, t: usize, form: Form) -> Net {
    body_on(g, c, n_out, b, t, form, (c.in_ch(), 0), t)
}

/// [`body`] reading channels `[offset, offset + c.in_ch())` of an input `x [B, T, total]`: a
/// teacher scoring the student's batch (log-mel or PCEN half of a dual row).
fn body_on(
    g: &mut Graph,
    c: &WakeCnnConfig,
    n_out: usize,
    b: usize,
    t: usize,
    form: Form,
    (total, offset): (usize, usize),
    t_in: usize,
) -> Net {
    let (k, pad) = (c.k, c.pad());
    // The input may be longer than the window (a teacher reading a causal student's rows):
    // its last `t` frames.
    let x = g.input("x", sh(&[b, t_in, total]));
    let x = if t_in > t { g.narrow_(x, 1, t_in - t, t) } else { x };
    let x = if total == c.in_ch() {
        x
    } else {
        g.narrow_(x, 2, offset, c.in_ch())
    };
    let x = g.transpose_(x, vec![0, 2, 1]);
    let x = g.reshape_(x, vec![b as i64, c.in_ch() as i64, 1, t as i64]);
    let mut params: Vec<(String, NodeId)> = Vec::new();
    let mut bn_in = Vec::new();
    let mut param = |g: &mut Graph, name: String, dims: &[usize]| -> NodeId {
        let n = g.param(&name, sh(dims));
        params.push((name, n));
        n
    };
    // A convolution's output, then its bias (`Infer`) or batch normalization (`Train`).
    let mut affine = |g: &mut Graph,
                      param: &mut dyn FnMut(&mut Graph, String, &[usize]) -> NodeId,
                      y: NodeId,
                      name: &str,
                      ch: usize|
     -> NodeId {
        match form {
            Form::Infer => {
                let bias = param(g, format!("{name}_b"), &[1, ch, 1, 1]);
                g.add(y, bias)
            }
            Form::Train => {
                bn_in.push(y);
                let gamma = param(g, format!("{name}_g"), &[ch]);
                let beta = param(g, format!("{name}_beta"), &[ch]);
                g.batch_norm_training(y, gamma, beta, BN_EPS)
            }
        }
    };
    let mut h = x;
    for (l, (cin, cout, stride)) in [(c.in_ch(), c.c1, 1), (c.c1, c.c2, 2), (c.c2, c.c3, 2)]
        .into_iter()
        .enumerate()
    {
        let name = format!("conv{}", l + 1);
        let w = param(g, format!("{name}_w"), &[cout, cin, 1, k]);
        let y = g.conv2d(h, w, [1, k], [1, stride], [0, pad], [1, 1], 1);
        let y = affine(g, &mut param, y, &name, cout);
        h = g.relu(y);
    }
    let t1 = t + 2 * pad - (k - 1);
    let t2 = (t1 + 2 * pad - k) / 2 + 1;
    let mut t3 = (t2 + 2 * pad - k) / 2 + 1;
    // Residual blocks: h = relu(h + affine(conv_dilated(h))), dilation 2^i, same length.
    for i in 0..c.res_blocks {
        let d = rlx_wakeword_core::WakeCnnConfig::dilation(i);
        let name = format!("res{i}");
        let w = param(g, format!("{name}_w"), &[c.c3, c.c3, 1, k]);
        let y = g.conv2d(h, w, [1, k], [1, 1], [0, pad * d], [1, d], 1);
        // Causal (unpadded): the output is `(k − 1)·d` shorter; its residual is the input's
        // last positions (each output's newest tap).
        let span = (k - 1) * d;
        let skip = if c.causal {
            t3 -= span;
            g.narrow_(h, 3, span, t3)
        } else {
            h
        };
        let y = affine(g, &mut param, y, &name, c.c3);
        let y = g.add(skip, y);
        h = g.relu(y);
    }
    // Each of `pool_chunks` consecutive stretches averaged, in order ([B, c3 · chunks],
    // chunk-major: the engine's `pool_chunks`), then the attention-pooled vector.
    let chunks = c.pool_chunks.max(1);
    let mut parts: Vec<NodeId> = if chunks == 1 {
        vec![g.mean(h, vec![2, 3], false)]
    } else {
        (0..chunks)
            .map(|i| {
                let (lo, hi) = (
                    i * t3 / chunks,
                    ((i + 1) * t3 / chunks).max(i * t3 / chunks + 1).min(t3),
                );
                let part = g.narrow_(h, 3, lo, hi - lo);
                g.mean(part, vec![2, 3], false)
            })
            .collect()
    };
    if c.attn {
        let w = param(g, "attn_w".into(), &[1, c.c3, 1, 1]);
        let bias = param(g, "attn_b".into(), &[1, 1, 1, 1]);
        let u = g.conv2d(h, w, [1, 1], [1, 1], [0, 0], [1, 1], 1);
        let u = g.add(u, bias);
        let u = g.tanh(u);
        let bound = g.constant(f64::from(rlx_wakeword_core::cnn::ATTN_BOUND), DType::F32);
        let s = g.mul(u, bound);
        let e = g.exp(s);
        let den = g.sum(e, vec![3], true);
        let a = g.div(e, den);
        let ya = g.mul(h, a);
        parts.push(g.sum(ya, vec![2, 3], false));
    }
    let pooled = if parts.len() == 1 {
        parts[0]
    } else {
        g.concat_(parts, 1)
    };
    let fc1_w = param(g, "fc1_w".into(), &[c.pooled_len(), c.hidden]);
    let fc1_b = param(g, "fc1_b".into(), &[1, c.hidden]);
    let fc2_w = param(g, "fc2_w".into(), &[c.hidden, n_out]);
    let fc2_b = param(g, "fc2_b".into(), &[1, n_out]);
    let h = g.mm(pooled, fc1_w);
    let h = g.add(h, fc1_b);
    let h = g.relu(h);
    let z = g.mm(h, fc2_w);
    let z = g.add(z, fc2_b);
    Net { z, params, bn_in }
}

fn build_probs(c: &WakeCnnConfig, n_out: usize, b: usize, t: usize) -> Graph {
    let mut g = Graph::new("wake_multi_probs");
    let net = body(&mut g, c, n_out, b, t, Form::Infer);
    let p = g.sigmoid(net.z);
    g.set_outputs(vec![p]);
    g
}

/// Mean BCE-with-logits over `[B, K]` against `y [B, K]`, and the parameters.
fn build_loss(
    c: &WakeCnnConfig,
    n_out: usize,
    b: usize,
    t: usize,
    form: Form,
) -> (Graph, Vec<(String, NodeId)>) {
    let mut g = Graph::new("wake_multi_loss");
    let net = body(&mut g, c, n_out, b, t, form);
    let z = net.z;
    let y = g.input("y", sh(&[b, n_out]));
    // max(z, 0) − z·y + ln(1 + e^(−|z|)).
    let pos = g.relu(z);
    let zy = g.mul(z, y);
    let a = g.abs(z);
    let na = g.neg(a);
    let e = g.exp(na);
    let one = g.constant(1.0, DType::F32);
    let e1 = g.add(e, one);
    let soft = g.log(e1);
    let l = g.sub(pos, zy);
    let l = g.add(l, soft);
    let loss = g.mean(l, vec![0, 1], false);
    g.set_outputs(vec![loss]);
    (g, net.params)
}

/// The batch-normalized layers' input statistics over a batch (`Train` form): per layer the
/// channel means of `x` and of `x²`, in [`bn_layers`] order.
fn build_bn_stats(c: &WakeCnnConfig, n_out: usize, b: usize, t: usize) -> Graph {
    let mut g = Graph::new("wake_multi_bn_stats");
    let net = body(&mut g, c, n_out, b, t, Form::Train);
    let mut outs = Vec::new();
    for x in net.bn_in {
        let sq = g.mul(x, x);
        outs.push(g.mean(x, vec![0, 2, 3], false));
        outs.push(g.mean(sq, vec![0, 2, 3], false));
    }
    g.set_outputs(outs);
    g
}

fn transpose(m: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut t = vec![0.0; m.len()];
    for r in 0..rows {
        for c in 0..cols {
            t[c * rows + r] = m[r * cols + c];
        }
    }
    t
}

/// The engine's weights as `Infer`-form graph parameters.
fn to_graph(w: &WakeCnnWeights) -> HashMap<String, Vec<f32>> {
    let c = &w.cfg;
    let k = w.fc2_b.len();
    let mut m: HashMap<String, Vec<f32>> = HashMap::new();
    for (l, (cw, cb)) in [
        (&w.conv1_w, &w.conv1_b),
        (&w.conv2_w, &w.conv2_b),
        (&w.conv3_w, &w.conv3_b),
    ]
    .into_iter()
    .enumerate()
    {
        m.insert(format!("conv{}_w", l + 1), cw.clone());
        m.insert(format!("conv{}_b", l + 1), cb.clone());
    }
    for (i, (rw, rb)) in w.res_w.iter().zip(&w.res_b).enumerate() {
        m.insert(format!("res{i}_w"), rw.clone());
        m.insert(format!("res{i}_b"), rb.clone());
    }
    if c.attn {
        m.insert("attn_w".into(), w.attn_w.clone());
        m.insert("attn_b".into(), w.attn_b.clone());
    }
    m.insert(
        "fc1_w".into(),
        transpose(&w.fc1_w, c.hidden, c.pooled_len()),
    );
    m.insert("fc1_b".into(), w.fc1_b.clone());
    m.insert("fc2_w".into(), transpose(&w.fc2_w, k, c.hidden));
    m.insert("fc2_b".into(), w.fc2_b.clone());
    m
}

/// `Train`-form starting parameters: the engine's weights without the batch-normalized
/// layers' biases, each with γ = 1, β = 0.
fn to_train_graph(w: &WakeCnnWeights) -> HashMap<String, Vec<f32>> {
    let mut m = to_graph(w);
    for name in bn_layers(&w.cfg) {
        let ch = m[&format!("{name}_b")].len();
        m.remove(&format!("{name}_b"));
        m.insert(format!("{name}_g"), vec![1.0; ch]);
        m.insert(format!("{name}_beta"), vec![0.0; ch]);
    }
    m
}

/// Fold batch normalization into the convolutions: `Train`-form parameters and each layer's
/// input statistics (`(mean, mean of squares)` per channel, [`bn_layers`] order) → `Infer`.
/// `w' = w · γ/σ`, `b' = β − μ · γ/σ` with `σ = √(var + ε)`.
fn fold_bn(
    c: &WakeCnnConfig,
    p: &HashMap<String, Vec<f32>>,
    stats: &[(Vec<f32>, Vec<f32>)],
) -> HashMap<String, Vec<f32>> {
    let mut m = p.clone();
    for (name, (mu, ex2)) in bn_layers(c).iter().zip(stats) {
        let (gamma, beta) = (
            m.remove(&format!("{name}_g")).unwrap(),
            m.remove(&format!("{name}_beta")).unwrap(),
        );
        let w = m.get_mut(&format!("{name}_w")).unwrap();
        let per = w.len() / gamma.len();
        let mut bias = vec![0.0f32; gamma.len()];
        for o in 0..gamma.len() {
            let var = (ex2[o] - mu[o] * mu[o]).max(0.0);
            let scale = gamma[o] / (var + BN_EPS).sqrt();
            w[o * per..(o + 1) * per]
                .iter_mut()
                .for_each(|v| *v *= scale);
            bias[o] = beta[o] - mu[o] * scale;
        }
        m.insert(format!("{name}_b"), bias);
    }
    m
}

/// `Infer`-form parameters into the engine's weights.
fn from_graph(w: &mut WakeCnnWeights, p: &HashMap<String, Vec<f32>>) {
    let (pooled, hidden, k) = (w.cfg.pooled_len(), w.cfg.hidden, w.fc2_b.len());
    w.conv1_w.clone_from(&p["conv1_w"]);
    w.conv1_b.clone_from(&p["conv1_b"]);
    w.conv2_w.clone_from(&p["conv2_w"]);
    w.conv2_b.clone_from(&p["conv2_b"]);
    w.conv3_w.clone_from(&p["conv3_w"]);
    w.conv3_b.clone_from(&p["conv3_b"]);
    w.res_w = (0..w.cfg.res_blocks)
        .map(|i| p[&format!("res{i}_w")].clone())
        .collect();
    w.res_b = (0..w.cfg.res_blocks)
        .map(|i| p[&format!("res{i}_b")].clone())
        .collect();
    if w.cfg.attn {
        w.attn_w.clone_from(&p["attn_w"]);
        w.attn_b.clone_from(&p["attn_b"]);
    }
    w.fc1_w = transpose(&p["fc1_w"], pooled, hidden);
    w.fc1_b.clone_from(&p["fc1_b"]);
    w.fc2_w = transpose(&p["fc2_w"], hidden, k);
    w.fc2_b.clone_from(&p["fc2_b"]);
}

/// `he_init` with `n_out` outputs.
pub fn init_multi(cfg: WakeCnnConfig, n_out: usize, seed: u64) -> WakeCnnWeights {
    let mut w = WakeCnnWeights::he_init(cfg, seed);
    let h = w.cfg.hidden;
    let base = WakeCnnWeights::he_init(w.cfg.clone(), seed ^ 0x5151).fc1_w;
    let a = (6.0 / h as f32).sqrt();
    w.fc2_w = (0..n_out * h)
        .map(|i| base[i % base.len()].signum() * a * ((i * 2_654_435_761 % 1000) as f32 / 1000.0))
        .collect();
    w.fc2_b = vec![0.0; n_out];
    w
}

/// The `t` frames ending at frame `end` (exclusive) of a frame-major clip, frame-major into
/// `dst` (zero frames in front if the clip is short): one contiguous copy.
fn window_into(mel: &[f32], n_mels: usize, end: usize, t: usize, dst: &mut [f32]) {
    let end = end.min(mel.len() / n_mels);
    let take = end.min(t);
    let gap = (t - take) * n_mels;
    dst[..gap].fill(0.0);
    dst[gap..].copy_from_slice(&mel[(end - take) * n_mels..end * n_mels]);
}

/// One feature step per dB: features are `ln(power) / 10 + 2`.
const FEATURE_PER_DB: f32 = std::f32::consts::LN_10 / 100.0;

/// How training windows are varied ([`fill_batch`]); [`Aug::NONE`] for scoring.
#[derive(Clone, Copy)]
struct Aug<'a> {
    gain_db: f32,
    noise: &'a [Vec<f32>],
    noise_p: f32,
    noise_snr_db: (f32, f32),
    mask_p: f32,
    /// The network's PCEN front end, over `warm` more frames before the window. Not an
    /// augmentation: scoring sets it too.
    pcen: bool,
    warm: usize,
    /// Rows of log-mel then PCEN per frame (`2·n_mels` wide); implies `pcen`.
    dual: bool,
}

impl Aug<'_> {
    const NONE: Aug<'static> = Aug {
        gain_db: 0.0,
        noise: &[],
        noise_p: 0.0,
        noise_snr_db: (0.0, 0.0),
        mask_p: 0.0,
        pcen: false,
        warm: 0,
        dual: false,
    };

    /// A row's frame width.
    fn width(&self, n_mels: usize) -> usize {
        n_mels * (1 + usize::from(self.dual))
    }
}

/// A feature's power (features are `ln(power) / 10 + 2`), as a natural log.
#[inline]
fn ln_power(v: f32) -> f32 {
    (v - 2.0) * 10.0
}

/// Add `t` frames of `noise` (from frame `start`, wrapping) to the window `row` in the power
/// domain, `snr_db` below the window's mean power.
fn add_noise(row: &mut [f32], noise: &[f32], n_mels: usize, start: usize, snr_db: f32) {
    let nf = noise.len() / n_mels;
    let t = row.len() / n_mels;
    let frame = |j: usize| &noise[((start + j) % nf) * n_mels..][..n_mels];
    let pw = row.iter().map(|&v| ln_power(v).exp()).sum::<f32>() / row.len() as f32;
    let pn = (0..t)
        .flat_map(frame)
        .map(|&v| ln_power(v).exp())
        .sum::<f32>()
        / row.len() as f32;
    if !(pw > 0.0 && pn > 0.0) {
        return;
    }
    // ln of the gain putting the noise `snr_db` under the window.
    let lg = (pw / pn).ln() - snr_db * std::f32::consts::LN_10 / 10.0;
    for (j, dst) in row.chunks_exact_mut(n_mels).enumerate() {
        for (d, &nv) in dst.iter_mut().zip(frame(j)) {
            let (a, b) = (ln_power(*d), ln_power(nv) + lg);
            let (hi, lo) = if a > b { (a, b) } else { (b, a) };
            *d = (hi + (lo - hi).exp().ln_1p()) / 10.0 + 2.0;
        }
    }
}

/// `(clip, window end, class)` per batch row into `x [B, width·t]` (`width`: `n_mels`, or twice
/// with `dual`) and `y [B, K]`, on every core;
/// each window varied as `aug` says (`seed`: this batch's draws).
#[allow(clippy::too_many_arguments)]
fn fill_batch(
    items: &[(usize, usize, Option<usize>)],
    mels: &[Vec<f32>],
    n_mels: usize,
    t: usize,
    n_out: usize,
    x: &mut [f32],
    y: &mut [f32],
    aug: &Aug,
    seed: u64,
) {
    let per = items.len().div_ceil(8).max(16);
    std::thread::scope(|s| {
        for (part, ((it, xc), yc)) in items
            .chunks(per)
            .zip(x.chunks_mut(per * aug.width(n_mels) * t))
            .zip(y.chunks_mut(per * n_out))
            .enumerate()
        {
            s.spawn(move || {
                let mut r = seed ^ (part as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
                // Uniform in [0, 1).
                let mut u = move || {
                    r = r
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    (r >> 40) as f32 / (1u64 << 24) as f32
                };
                // The window and the frames before it that its PCEN starts from.
                let mut span = vec![0.0f32; (t + aug.warm) * n_mels];
                let mut level = Vec::new();
                let mut norm = Vec::new();
                for (j, &(i, e, c)) in it.iter().enumerate() {
                    window_into(&mels[i], n_mels, e, t + aug.warm, &mut span);
                    let row = &mut span[..];
                    if !aug.noise.is_empty() && u() < aug.noise_p {
                        let nz =
                            &aug.noise[(u() * aug.noise.len() as f32) as usize % aug.noise.len()];
                        let nf = nz.len() / n_mels;
                        if nf > 0 {
                            let (hi, lo) = (
                                aug.noise_snr_db.1,
                                aug.noise_snr_db.0 - if c.is_some() { 0.0 } else { 10.0 },
                            );
                            let start = (u() * nf as f32) as usize % nf;
                            add_noise(row, nz, n_mels, start, lo + u() * (hi - lo));
                        }
                    }
                    if aug.gain_db > 0.0 {
                        let shift = (u() * 2.0 - 1.0) * aug.gain_db * FEATURE_PER_DB;
                        row.iter_mut().for_each(|v| *v += shift);
                    }
                    let wd = aug.width(n_mels);
                    let row = &mut xc[j * wd * t..(j + 1) * wd * t];
                    if aug.dual {
                        // Log-mel, then its PCEN (over the frames before the window too).
                        norm.clear();
                        norm.extend_from_slice(&span);
                        level.clear();
                        rlx_wakeword_core::pcen_inplace(&mut norm, n_mels, &mut level);
                        let at = aug.warm * n_mels;
                        for (f, fr) in row.chunks_exact_mut(wd).enumerate() {
                            fr[..n_mels].copy_from_slice(&span[at + f * n_mels..at + (f + 1) * n_mels]);
                            fr[n_mels..].copy_from_slice(&norm[at + f * n_mels..at + (f + 1) * n_mels]);
                        }
                    } else {
                        if aug.pcen {
                            level.clear();
                            rlx_wakeword_core::pcen_inplace(&mut span, n_mels, &mut level);
                        }
                        row.copy_from_slice(&span[aug.warm * n_mels..]);
                    }
                    if aug.mask_p > 0.0 && u() < aug.mask_p {
                        // The same band and stretch in each half (log-mel, PCEN), each set to its
                        // own mean.
                        let halves = wd / n_mels;
                        let means: Vec<f32> = (0..halves)
                            .map(|h| {
                                row.chunks_exact(wd)
                                    .map(|fr| fr[h * n_mels..(h + 1) * n_mels].iter().sum::<f32>())
                                    .sum::<f32>()
                                    / (t * n_mels) as f32
                            })
                            .collect();
                        let (bw, tw) = (1 + (u() * 4.0) as usize, 1 + (u() * 10.0) as usize);
                        let b0 = (u() * (n_mels - bw.min(n_mels)) as f32) as usize;
                        let t0 = (u() * (t - tw.min(t)) as f32) as usize;
                        for (f, fr) in row.chunks_exact_mut(wd).enumerate() {
                            for (h, &m) in means.iter().enumerate() {
                                let half = &mut fr[h * n_mels..(h + 1) * n_mels];
                                if (t0..t0 + tw).contains(&f) {
                                    half.fill(m);
                                } else {
                                    half[b0..(b0 + bw).min(n_mels)].fill(m);
                                }
                            }
                        }
                    }
                    let row = &mut yc[j * n_out..(j + 1) * n_out];
                    row.fill(0.0);
                    if let Some(c) = c {
                        row[c] = 1.0;
                    }
                }
            });
        }
    });
}

/// Mel features of WAV files, from `cache` when there (keyed by path, size, modification time
/// and the mel front end), else decoded, computed (on every core) and stored there.
pub fn load_mels_cached(paths: &[PathBuf], cache: Option<&Path>) -> Result<Vec<Vec<f32>>> {
    if let Some(c) = cache {
        std::fs::create_dir_all(c).with_context(|| format!("create {}", c.display()))?;
    }
    let key = |p: &Path| -> Option<PathBuf> {
        let c = cache?;
        let real = std::fs::canonicalize(p).ok()?;
        let md = std::fs::metadata(&real).ok()?;
        let mut h = std::collections::hash_map::DefaultHasher::new();
        (real, md.len(), md.modified().ok(), "mel-v1-16k-32").hash(&mut h);
        Some(c.join(format!("{:016x}.mel", h.finish())))
    };
    let one = |p: &PathBuf| -> Result<Vec<f32>> {
        let k = key(p);
        if let Some(k) = &k
            && let Ok(b) = std::fs::read(k)
        {
            return Ok(b
                .as_chunks::<4>()
                .0
                .iter()
                .map(|x| f32::from_le_bytes(*x))
                .collect());
        }
        let (sr, pcm) =
            crate::load_wav_mono_f32(p).with_context(|| format!("read {}", p.display()))?;
        let pcm = crate::resample_linear(&pcm, sr, crate::SAMPLE_RATE_16K);
        let mel = clip_mel_frames(&pcm);
        if let Some(k) = &k {
            let bytes: Vec<u8> = mel.iter().flat_map(|v| v.to_le_bytes()).collect();
            let tmp = k.with_extension("tmp");
            if std::fs::write(&tmp, bytes).is_ok() {
                let _ = std::fs::rename(&tmp, k);
            }
        }
        Ok(mel)
    };
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let per = paths.len().div_ceil(threads).max(1);
    std::thread::scope(|s| {
        let parts: Vec<_> = paths
            .chunks(per)
            .map(|part| s.spawn(move || part.iter().map(one).collect::<Result<Vec<_>>>()))
            .collect();
        let mut out = Vec::with_capacity(paths.len());
        for h in parts {
            out.extend(h.join().expect("mel thread")?);
        }
        Ok(out)
    })
}

/// Train `w` (`K` = `w.fc2_b.len()` outputs) on `clips` on `device`.
pub fn train_multi(
    w: &mut WakeCnnWeights,
    clips: &[MultiClip],
    cfg: &MultiTrainConfig,
    device: Device,
) -> Result<MultiReport> {
    train_multi_with_noise(w, clips, &[], cfg, device)
}

/// [`train_multi`], with `noise` (mel frames of noise recordings, frame-major) mixed into
/// training windows as `cfg.noise_p` / `cfg.noise_snr_db` say.
pub fn train_multi_with_noise(
    w: &mut WakeCnnWeights,
    clips: &[MultiClip],
    noise: &[Vec<f32>],
    cfg: &MultiTrainConfig,
    device: Device,
) -> Result<MultiReport> {
    train_multi_taught(w, clips, noise, &[], cfg, device)
}

/// [`train_multi_with_noise`], with `teachers` scoring every batch on the device: their scores
/// blended into the taught outputs' targets (`cfg.teacher_weight`).
pub fn train_multi_taught(
    w: &mut WakeCnnWeights,
    clips: &[MultiClip],
    noise: &[Vec<f32>],
    teachers: &[Teacher],
    cfg: &MultiTrainConfig,
    device: Device,
) -> Result<MultiReport> {
    let n_mels = w.cfg.n_mels;
    let n_out = w.fc2_b.len();
    let t = cfg.window_frames.max(8);
    let t0 = Instant::now();
    // What a window reads: itself and, with PCEN, the frames its smoother starts from.
    let warm = if w.cfg.pcen || w.cfg.dual {
        cfg.pcen_warm
    } else {
        0
    };
    let in_ch = w.cfg.in_ch();
    let t_in = t + warm;
    // Each clip between `t_in` frames of its own background (its opening, repeated if short)
    // and `tail_frames` more: windows can end from a window before its speech to well after it.
    let lead = t_in;
    let mels: Vec<Vec<f32>> = clips
        .iter()
        .map(|c| {
            let m = &c.mel;
            let opening = &m[..(cfg.tail_frames.max(lead) * n_mels).min(m.len())];
            let mut ext = Vec::with_capacity(m.len() + (lead + cfg.tail_frames) * n_mels);
            while ext.len() < lead * n_mels && !opening.is_empty() {
                let need = lead * n_mels - ext.len();
                ext.extend_from_slice(&opening[..need.min(opening.len())]);
            }
            ext.extend_from_slice(m);
            ext.extend_from_slice(&opening[..(cfg.tail_frames * n_mels).min(opening.len())]);
            ext
        })
        .collect();
    let frames: Vec<usize> = clips.iter().map(|c| c.mel.len() / n_mels + lead).collect();
    let ext: Vec<usize> = mels.iter().map(|m| m.len() / n_mels).collect();
    let n = clips.len();
    let positives: Vec<usize> = (0..n)
        .filter(|&i| clips[i].class.is_some() && frames[i] >= t_in + cfg.partial_gap)
        .collect();
    let negatives: Vec<usize> = (0..n).filter(|&i| clips[i].class.is_none()).collect();
    let n_partial = if positives.is_empty() {
        0
    } else {
        (positives.len() as f32 * cfg.partial_ratio) as usize
    };
    // Balance the phrases: each phrase's positives repeated up to the largest phrase's count.
    let per_class: Vec<usize> = (0..n_out)
        .map(|c| (0..n).filter(|&i| clips[i].class == Some(c)).count())
        .collect();
    let largest = per_class.iter().copied().max().unwrap_or(0);
    let extra: Vec<usize> = (0..n)
        .filter_map(|i| clips[i].class.map(|c| (i, c)))
        .flat_map(|(i, c)| std::iter::repeat_n(i, largest / per_class[c].max(1) - 1))
        .collect();
    let n = n + extra.len();
    let n_clips = clips.len();
    let b = cfg.batch.max(1);

    let form = if cfg.bn { Form::Train } else { Form::Infer };
    let (fwd, nodes) = build_loss(&w.cfg, n_out, b, t, form);
    let wrt: Vec<TrainableParam> = nodes
        .iter()
        .map(|(name, node)| TrainableParam {
            name: name.clone(),
            node: *node,
        })
        .collect();
    let start = if cfg.bn {
        to_train_graph(w)
    } else {
        to_graph(w)
    };
    let spec = AdamSpec {
        lr: cfg.lr,
        weight_decay: cfg.weight_decay,
        ..AdamSpec::default()
    };
    let mut trainer = ResidentTrainer::new(&fwd, &wrt, &start, &spec, device)
        .map_err(|e| anyhow::anyhow!("fused step: {e}"))?;
    let mut probe = Session::new(device).compile(build_probs(&w.cfg, n_out, b, t));
    let mut bn_stats = cfg
        .bn
        .then(|| Session::new(device).compile(build_bn_stats(&w.cfg, n_out, b, t)));
    let resident = trainer.is_resident();
    if cfg.log_every > 0 {
        eprintln!(
            "[rlx-wake multi/{device:?}] {n} clips, {} outputs, window {t}{}: setup {:.1?}, optimizer {}",
            n_out,
            [
                (w.cfg.res_blocks > 0).then(|| format!(", {} residual blocks", w.cfg.res_blocks)),
                w.cfg.attn.then(|| ", attention pooling".to_string()),
                w.cfg.pcen.then(|| format!(", PCEN ({warm} frames before)")),
                w.cfg.dual.then(|| format!(", log-mel + PCEN input ({warm} frames before)")),
                (!teachers.is_empty()).then(|| format!(", {} teachers", teachers.len())),
                cfg.bn.then(|| ", batch norm".to_string()),
            ]
            .into_iter()
            .flatten()
            .collect::<String>(),
            t0.elapsed(),
            if resident {
                "on the device"
            } else {
                "host-chained (no device handles)"
            }
        );
    }

    let mut rng = 0x2545_F491_4F6C_DD1Du64;
    let mut next = move || {
        rng = rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (rng >> 33) as usize
    };
    let aug = Aug {
        gain_db: cfg.gain_db,
        noise,
        noise_p: cfg.noise_p,
        noise_snr_db: cfg.noise_snr_db,
        mask_p: cfg.mask_p,
        pcen: w.cfg.pcen,
        warm,
        dual: w.cfg.dual,
    };
    // The front end alone, for scoring.
    let plain = Aug {
        pcen: w.cfg.pcen,
        warm,
        dual: w.cfg.dual,
        ..Aug::NONE
    };
    // The teachers, compiled on the device, each reading its part of the student's rows.
    let mut taught: Vec<(rlx_runtime::CompiledGraph, &Teacher, usize)> = Vec::new();
    for tch in teachers {
        let tc = &tch.weights.cfg;
        let kt = tch.weights.fc2_b.len();
        let offset = match (w.cfg.dual, tc.dual, tc.pcen) {
            (_, true, _) if w.cfg.dual => 0,
            (true, false, pcen) => usize::from(pcen) * n_mels,
            (false, false, pcen) if pcen == w.cfg.pcen => 0,
            _ => anyhow::bail!("a teacher must read the student's input (or one half of a dual one)"),
        };
        if tc.n_mels != n_mels || tch.outputs.iter().any(|&k| k >= kt.min(n_out)) {
            anyhow::bail!("a teacher's mel bands or outputs do not match the student's");
        }
        let mut g = Graph::new("wake_multi_teacher");
        let tw = tch.window_frames.unwrap_or(t).min(t);
        let net = body_on(&mut g, tc, kt, b, tw, Form::Infer, (in_ch, offset), t);
        let p = g.sigmoid(net.z);
        g.set_outputs(vec![p]);
        let mut probe = Session::new(device).compile(g);
        for (name, v) in to_graph(&tch.weights) {
            probe.set_param(&name, &v);
        }
        taught.push((probe, tch, kt));
    }
    // A training item (an index into clips and their balancing copies): its window's end.
    let base_item =
        |item: usize, next: &mut dyn FnMut() -> usize| -> (usize, usize, Option<usize>) {
            let item = if item < n_clips {
                item
            } else {
                extra[item - n_clips]
            };
            let f = frames[item];
            match clips[item].class {
                Some(c) => {
                    let back = next() % (POS_JITTER + 1).min(f.max(1));
                    let fwd = next() % (cfg.pos_tail.min(ext[item] - f) + 1);
                    // Before the end or after it, half and half when there is a tail.
                    let e = if cfg.pos_tail > 0 && next() % 2 == 0 {
                        f + fwd
                    } else {
                        f - back
                    };
                    (item, e, Some(c))
                }
                None if ext[item] > t_in => (item, t_in + next() % (ext[item] - t_in + 1), None),
                None => (item, f, None),
            }
        };
    // Batch normalization's statistics on `batches` batches of training windows (augmented as
    // in training), per layer `(mean, mean of squares)`.
    let n_bn = bn_layers(&w.cfg).len();
    let mut calibrate = |params: &HashMap<String, Vec<f32>>,
                         batches: usize,
                         next: &mut dyn FnMut() -> usize|
     -> Vec<(Vec<f32>, Vec<f32>)> {
        let g = bn_stats.as_mut().expect("batch norm");
        for (name, p) in params {
            g.set_param(name, p);
        }
        let mut acc: Vec<(Vec<f32>, Vec<f32>)> = Vec::new();
        let (mut px, mut py) = (vec![0.0f32; b * in_ch * t], vec![0.0f32; b * n_out]);
        for _ in 0..batches {
            let items: Vec<(usize, usize, Option<usize>)> =
                (0..b).map(|_| base_item(next() % n, next)).collect();
            fill_batch(
                &items,
                &mels,
                n_mels,
                t,
                n_out,
                &mut px,
                &mut py,
                &aug,
                next() as u64,
            );
            let o = g.run(&[("x", &px)]);
            if acc.is_empty() {
                acc = (0..n_bn)
                    .map(|l| (vec![0.0; o[2 * l].len()], vec![0.0; o[2 * l].len()]))
                    .collect();
            }
            for (l, (m, q)) in acc.iter_mut().enumerate() {
                m.iter_mut()
                    .zip(&o[2 * l])
                    .for_each(|(a, v)| *a += v / batches as f32);
                q.iter_mut()
                    .zip(&o[2 * l + 1])
                    .for_each(|(a, v)| *a += v / batches as f32);
            }
        }
        acc
    };
    let wcfg = w.cfg.clone();
    // The engine's form of the current parameters (batch normalization calibrated and folded).
    let mut infer_params =
        |params: HashMap<String, Vec<f32>>, batches: usize, next: &mut dyn FnMut() -> usize| {
            if cfg.bn {
                let stats = calibrate(&params, batches, next);
                fold_bn(&wcfg, &params, &stats)
            } else {
                params
            }
        };
    // Score windows `(clip, end)` with the current weights: probabilities `[len, K]`.
    let score = |probe: &mut rlx_runtime::CompiledGraph,
                 params: &HashMap<String, Vec<f32>>,
                 wins: &[(usize, usize)]|
     -> Vec<f32> {
        for (name, p) in params {
            probe.set_param(name, p);
        }
        let mut out = Vec::with_capacity(wins.len() * n_out);
        let mut px = vec![0.0f32; b * in_ch * t];
        for chunk in wins.chunks(b) {
            let items: Vec<(usize, usize, Option<usize>)> =
                chunk.iter().map(|&(i, e)| (i, e, None)).collect();
            let mut dummy = vec![0.0f32; b * n_out];
            px.fill(0.0);
            fill_batch(
                &items, &mels, n_mels, t, n_out, &mut px, &mut dummy, &plain, 0,
            );
            let o = probe.run(&[("x", &px)]);
            out.extend_from_slice(&o[0][..chunk.len() * n_out]);
        }
        out
    };

    let mut hard: Vec<(usize, usize)> = Vec::new();
    let (mut xb, mut yb) = (vec![0.0f32; b * in_ch * t], vec![0.0f32; b * n_out]);
    let (mut xn, mut yn) = (xb.clone(), yb.clone());
    let (mut initial, mut last) = (0.0f32, 0.0f32);
    let t_train = Instant::now();
    for epoch in 0..cfg.epochs {
        if cfg.mine_every > 0
            && epoch >= cfg.mine_after
            && (epoch - cfg.mine_after).is_multiple_of(cfg.mine_every)
        {
            let tm = Instant::now();
            let wins: Vec<(usize, usize)> = negatives
                .iter()
                .flat_map(|&i| {
                    (t_in..=ext[i].max(t_in))
                        .step_by(cfg.mine_stride.max(1))
                        .map(move |e| (i, e))
                })
                .collect();
            let now = infer_params(trainer.params(), 16, &mut next);
            let probs = score(&mut probe, &now, &wins);
            // Each window counts against the output it fires most; each output keeps its own
            // worst ones, up to `mine_ratio` × its (balanced) positives.
            let mut by_out: Vec<Vec<(f32, usize, usize)>> = vec![Vec::new(); n_out];
            for (j, &(i, e)) in wins.iter().enumerate() {
                let row = &probs[j * n_out..(j + 1) * n_out];
                let (k, m) = row
                    .iter()
                    .copied()
                    .enumerate()
                    .fold((0, 0.0f32), |a, (k, v)| if v > a.1 { (k, v) } else { a });
                if m > cfg.mine_threshold {
                    by_out[k].push((m, i, e));
                }
            }
            let cap = (largest.max(1) as f32 * cfg.mine_ratio) as usize;
            let found: usize = by_out.iter().map(Vec::len).sum();
            hard = by_out
                .into_iter()
                .flat_map(|mut v| {
                    v.sort_by(|a, b| b.0.total_cmp(&a.0));
                    v.into_iter().take(cap).map(|(_, i, e)| (i, e))
                })
                .collect();
            if cfg.log_every > 0 {
                eprintln!(
                    "[rlx-wake multi] epoch={epoch} mined {found} of {} negative windows above {}: training on {} ({:.1?})",
                    wins.len(),
                    cfg.mine_threshold,
                    hard.len(),
                    tm.elapsed()
                );
            }
        }
        let total = n + n_partial + hard.len();
        let mut order: Vec<usize> = (0..total).collect();
        for i in (1..total).rev() {
            order.swap(i, next() % (i + 1));
        }
        let steps = total.div_ceil(b);
        let pick =
            |s: usize, next: &mut dyn FnMut() -> usize| -> Vec<(usize, usize, Option<usize>)> {
                (0..b)
                    .map(|j| {
                        let item = order[(s * b + j) % total];
                        if item < n {
                            base_item(item, next)
                        } else if item < n + n_partial {
                            let i = positives[next() % positives.len()];
                            (
                                i,
                                t_in + next() % (frames[i] - cfg.partial_gap - t_in + 1),
                                None,
                            )
                        } else {
                            let (i, e) = hard[item - n - n_partial];
                            (i, e, None)
                        }
                    })
                    .collect()
            };
        let first = pick(0, &mut next);
        fill_batch(
            &first,
            &mels,
            n_mels,
            t,
            n_out,
            &mut xb,
            &mut yb,
            &aug,
            next() as u64,
        );
        let mut sum = 0.0f32;
        let (mut in_step, mut waited) = (0.0f64, 0.0f64);
        for s in 0..steps {
            // Cut the next batch while the device runs this one.
            let upcoming = (s + 1 < steps).then(|| (pick(s + 1, &mut next), next() as u64));
            // Distillation: the teachers' scores into the taught outputs' targets.
            for (probe, tch, kt) in taught.iter_mut() {
                let o = probe.run(&[("x", &xb)]);
                for r in 0..b {
                    for &k in &tch.outputs {
                        let y = &mut yb[r * n_out + k];
                        *y = (1.0 - cfg.teacher_weight) * *y + cfg.teacher_weight * o[0][r * *kt + k];
                    }
                }
            }
            let ts = Instant::now();
            let loss = std::thread::scope(|sc| {
                if let Some((items, seed)) = &upcoming {
                    let (xn, yn, mels, aug) = (&mut xn, &mut yn, &mels, &aug);
                    sc.spawn(move || fill_batch(items, mels, n_mels, t, n_out, xn, yn, aug, *seed));
                }
                let r = trainer.step(&[("x", &xb), ("y", &yb)]);
                in_step += ts.elapsed().as_secs_f64();
                r
            });
            waited += ts.elapsed().as_secs_f64();
            let l = loss
                .first()
                .and_then(|v| v.first())
                .copied()
                .unwrap_or(f32::NAN);
            sum += l;
            std::mem::swap(&mut xb, &mut xn);
            std::mem::swap(&mut yb, &mut yn);
        }
        let mean = sum / steps.max(1) as f32;
        if epoch == 0 {
            initial = mean;
        }
        last = mean;
        if cfg.log_every > 0 && (epoch % cfg.log_every == 0 || epoch + 1 == cfg.epochs) {
            eprintln!(
                "[rlx-wake multi/{device:?}] epoch={epoch} loss={mean:.4} ({:.2}s/epoch; this one {steps} steps, {:.1} ms device step, {:.1} ms waiting for the next batch)",
                t_train.elapsed().as_secs_f64() / (epoch + 1) as f64,
                in_step * 1e3 / steps.max(1) as f64,
                (waited - in_step).max(0.0) * 1e3 / steps.max(1) as f64
            );
        }
    }
    let secs_per_epoch = t_train.elapsed().as_secs_f64() / cfg.epochs.max(1) as f64;
    let params = infer_params(trainer.params(), 64, &mut next);
    from_graph(w, &params);

    // Training-set check on the device: positives at their end, negatives at theirs.
    let wins: Vec<(usize, usize)> = (0..n_clips).map(|i| (i, frames[i])).collect();
    let probs = score(&mut probe, &params, &wins);
    let mut recall = vec![(0usize, 0usize); n_out];
    let mut neg = (0usize, 0usize);
    for i in 0..n_clips {
        let p = &probs[i * n_out..(i + 1) * n_out];
        match clips[i].class {
            Some(c) => {
                recall[c].1 += 1;
                recall[c].0 += usize::from(p[c] >= 0.5);
            }
            None => {
                neg.1 += 1;
                neg.0 += usize::from(p.iter().any(|&v| v >= 0.5));
            }
        }
    }
    Ok(MultiReport {
        initial_loss: initial,
        final_loss: last,
        resident,
        secs_per_epoch,
        recall,
        negatives_firing: neg,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn to_core(w: &WakeCnnWeights) -> rlx_wakeword_core::WakeCnnWeights {
        let c = &w.cfg;
        rlx_wakeword_core::WakeCnnWeights::from_parts(
            rlx_wakeword_core::WakeCnnConfig {
                n_mels: c.n_mels,
                c1: c.c1,
                c2: c.c2,
                c3: c.c3,
                k: c.k,
                hidden: c.hidden,
                pool_chunks: c.pool_chunks,
                res_blocks: c.res_blocks,
                attn: c.attn,
                pcen: c.pcen,
                dual: c.dual,
                causal: c.causal,
            },
            w.conv1_w.clone(),
            w.conv1_b.clone(),
            w.conv2_w.clone(),
            w.conv2_b.clone(),
            w.conv3_w.clone(),
            w.conv3_b.clone(),
            w.fc1_w.clone(),
            w.fc1_b.clone(),
            w.fc2_w.clone(),
            w.fc2_b.clone(),
        )
        .with_extras(
            w.res_w.clone(),
            w.res_b.clone(),
            w.attn_w.clone(),
            w.attn_b.clone(),
        )
    }

    fn test_frames(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (s >> 8) as f32 / (1u32 << 24) as f32 * 2.0 + 0.5
            })
            .collect()
    }

    /// The training graph scores a window exactly as the streaming engine on the robot does
    /// (same weights after the layout conversions, K outputs): the plain network, with
    /// residual blocks and attention pooling, and with PCEN (applied to the window as
    /// `fill_batch` does, by the engine on its own).
    #[test]
    fn graph_matches_the_engine() {
        let base = WakeCnnConfig {
            pool_chunks: 4,
            ..WakeCnnConfig::lite()
        };
        let variants = [
            base.clone(),
            WakeCnnConfig {
                res_blocks: 3,
                attn: true,
                ..base.clone()
            },
            WakeCnnConfig {
                res_blocks: 2,
                attn: true,
                pcen: true,
                ..base.clone()
            },
            WakeCnnConfig {
                res_blocks: 2,
                attn: true,
                dual: true,
                ..base.clone()
            },
            WakeCnnConfig {
                res_blocks: 2,
                attn: true,
                dual: true,
                causal: true,
                ..base.clone()
            },
        ];
        for cfg in variants {
            let mut w = init_multi(cfg.clone(), 3, 11);
            w.res_b
                .iter_mut()
                .flatten()
                .enumerate()
                .for_each(|(i, v)| *v = (i % 7) as f32 * 0.01);
            let (b, t) = (2usize, 80usize);
            let n_mels = cfg.n_mels;
            let frames = test_frames(t * n_mels, 3);
            let mut x = frames.clone();
            if cfg.pcen || cfg.dual {
                rlx_wakeword_core::pcen_inplace(&mut x, n_mels, &mut Vec::new());
            }
            if cfg.dual {
                // Each frame: log-mel, then PCEN (fill_batch's layout).
                x = frames
                    .chunks_exact(n_mels)
                    .zip(x.chunks_exact(n_mels))
                    .flat_map(|(r, p)| r.iter().chain(p).copied())
                    .collect();
            }
            let x = [x.clone(), x].concat();
            let mut probe = Session::new(Device::Cpu).compile(build_probs(&cfg, 3, b, t));
            for (name, v) in to_graph(&w) {
                probe.set_param(&name, &v);
            }
            let got = probe.run(&[("x", &x)]).remove(0);
            let mut engine = rlx_wakeword_core::WakeCnn::new(to_core(&w)).with_window_frames(t);
            let want = engine.push_mel_frames_all(&frames).unwrap().to_vec();
            for (j, (g, e)) in got[..3].iter().zip(&want).enumerate() {
                assert!(
                    (g - e).abs() < 1e-4,
                    "{cfg:?} output {j}: graph {g}, engine {e}"
                );
            }
            assert_eq!(&got[..3], &got[3..6], "both batch rows are the same window");
            // And the export layout round-trips.
            let mut back = w.clone();
            from_graph(&mut back, &to_graph(&w));
            assert_eq!(back.conv2_w, w.conv2_w);
            assert_eq!(back.res_w, w.res_w);
            assert_eq!(back.attn_w, w.attn_w);
            assert_eq!(back.fc1_w, w.fc1_w);
            assert_eq!(back.fc2_w, w.fc2_w);
        }
    }

    /// Batch normalization folded into the convolutions scores a batch as the training form
    /// does with that batch's own statistics.
    #[test]
    fn folded_batch_norm_matches_training_form() {
        let cfg = WakeCnnConfig {
            pool_chunks: 2,
            res_blocks: 2,
            attn: true,
            ..WakeCnnConfig::lite()
        };
        let w = init_multi(cfg.clone(), 2, 5);
        let mut p = to_train_graph(&w);
        for (i, name) in bn_layers(&cfg).iter().enumerate() {
            p.get_mut(&format!("{name}_g"))
                .unwrap()
                .iter_mut()
                .for_each(|v| *v = 1.0 + 0.1 * i as f32);
            p.get_mut(&format!("{name}_beta"))
                .unwrap()
                .iter_mut()
                .for_each(|v| *v = 0.05 * i as f32);
        }
        let (b, t) = (4usize, 40usize);
        let x = test_frames(b * t * cfg.n_mels, 9);
        let mut g = Graph::new("train_form_probs");
        let net = body(&mut g, &cfg, 2, b, t, Form::Train);
        let pr = g.sigmoid(net.z);
        g.set_outputs(vec![pr]);
        let mut train_form = Session::new(Device::Cpu).compile(g);
        let mut stats_g = Session::new(Device::Cpu).compile(build_bn_stats(&cfg, 2, b, t));
        for (name, v) in &p {
            train_form.set_param(name, v);
            stats_g.set_param(name, v);
        }
        let want = train_form.run(&[("x", &x)]).remove(0);
        let o = stats_g.run(&[("x", &x)]);
        let stats: Vec<(Vec<f32>, Vec<f32>)> = (0..o.len() / 2)
            .map(|l| (o[2 * l].clone(), o[2 * l + 1].clone()))
            .collect();
        let folded = fold_bn(&cfg, &p, &stats);
        let mut infer = Session::new(Device::Cpu).compile(build_probs(&cfg, 2, b, t));
        for (name, v) in &folded {
            infer.set_param(name, v);
        }
        let got = infer.run(&[("x", &x)]).remove(0);
        for (i, (g, e)) in got.iter().zip(&want).enumerate() {
            assert!((g - e).abs() < 1e-3, "{i}: folded {g}, training form {e}");
        }
    }

    /// Noise is added in the power domain at the asked ratio: equal powers (0 dB) double the
    /// window's power (+3 dB); 30 dB under it barely moves it.
    #[test]
    fn noise_mixes_at_the_asked_ratio() {
        let n_mels = 4;
        let feat = |db: f32| db * FEATURE_PER_DB + 2.0;
        let power =
            |row: &[f32]| row.iter().map(|&v| ln_power(v).exp()).sum::<f32>() / row.len() as f32;
        let window: Vec<f32> = vec![feat(-20.0); 3 * n_mels];
        let noise: Vec<f32> = (0..5 * n_mels)
            .map(|i| feat(-50.0 + (i % 3) as f32))
            .collect();
        for (snr, want_db) in [(0.0f32, 3.01f32), (30.0, 0.004)] {
            let mut row = window.clone();
            add_noise(&mut row, &noise, n_mels, 2, snr);
            let got_db = 10.0 * (power(&row) / power(&window)).log10();
            assert!(
                (got_db - want_db).abs() < 0.01,
                "snr {snr}: +{got_db} dB, want +{want_db}"
            );
        }
    }

    /// Two "phrases" (tones in different mel bands) and noise.
    fn toy_clips() -> (Vec<MultiClip>, WakeCnnConfig) {
        let cfg = WakeCnnConfig::lite();
        let n_mels = cfg.n_mels;
        let mut clips = Vec::new();
        let mut s = 7u32;
        let mut noise = move || {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (s >> 8) as f32 / (1u32 << 24) as f32
        };
        for i in 0..120 {
            let class = match i % 3 {
                0 => Some(0),
                1 => Some(1),
                _ => None,
            };
            let frames = 80;
            let mut mel = vec![0.0f32; frames * n_mels];
            for f in 0..frames {
                for m in 0..n_mels {
                    mel[f * n_mels + m] = noise() * 0.5;
                }
                if f >= 50
                    && let Some(c) = class
                {
                    let band = if c == 0 { 4 } else { 20 };
                    for m in band..band + 4 {
                        mel[f * n_mels + m] += 3.0;
                    }
                }
            }
            clips.push(MultiClip { mel, class });
        }
        (clips, cfg)
    }

    /// Two "phrases" (tones in different mel bands) and noise: the network learns to tell
    /// them apart, each output firing for its own phrase only.
    #[test]
    fn learns_two_phrases_on_cpu() {
        let (clips, cfg) = toy_clips();
        // The plain network, and with residual blocks, attention pooling, PCEN and batch
        // normalization (calibrated and folded: the report scores the engine's form).
        let wide = WakeCnnConfig {
            res_blocks: 2,
            attn: true,
            pcen: true,
            ..cfg.clone()
        };
        for (cfg, bn) in [(cfg, false), (wide, true)] {
            let mut w = init_multi(cfg, 2, 3);
            let tc = MultiTrainConfig {
                epochs: 40,
                batch: 32,
                window_frames: 40,
                partial_ratio: 0.0,
                mine_every: 0,
                log_every: 0,
                lr: 3e-3,
                bn,
                pcen_warm: 20,
                ..Default::default()
            };
            let r = train_multi(&mut w, &clips, &tc, Device::Cpu).unwrap();
            assert!(r.final_loss < r.initial_loss * 0.5, "bn {bn}: {r:?}");
            for (found, total) in &r.recall {
                assert!(*found as f32 >= 0.9 * *total as f32, "bn {bn}: {r:?}");
            }
            assert!(
                r.negatives_firing.0 as f32 <= 0.1 * r.negatives_firing.1 as f32,
                "bn {bn}: {r:?}"
            );
        }
    }

    /// A dual-input student trained with a teacher (a plain network trained first, scoring the
    /// log-mel half of the student's rows) learns the task.
    #[test]
    fn learns_from_a_teacher() {
        let (clips, cfg) = toy_clips();
        let tc = MultiTrainConfig {
            epochs: 30,
            batch: 32,
            window_frames: 40,
            partial_ratio: 0.0,
            mine_every: 0,
            log_every: 0,
            lr: 3e-3,
            pcen_warm: 20,
            ..Default::default()
        };
        let mut teacher = init_multi(cfg.clone(), 2, 3);
        train_multi(&mut teacher, &clips, &tc, Device::Cpu).unwrap();
        let mut student = init_multi(
            WakeCnnConfig {
                dual: true,
                res_blocks: 1,
                attn: true,
                ..cfg
            },
            2,
            4,
        );
        let t = [Teacher {
            weights: teacher,
            outputs: vec![0, 1],
            window_frames: None,
        }];
        let r = train_multi_taught(&mut student, &clips, &[], &t, &tc, Device::Cpu).unwrap();
        assert!(r.final_loss < r.initial_loss * 0.7, "{r:?}");
        for (found, total) in &r.recall {
            assert!(*found as f32 >= 0.9 * *total as f32, "{r:?}");
        }
    }
}
