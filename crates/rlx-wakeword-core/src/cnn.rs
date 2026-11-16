// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Compact wake CNN (same tensor layout as `rlx-wake`).
//!
//! After [`WakeCnnWeights::ternarize`], forward uses fused ternary conv/GEMV when
//! weight tensors are exact `{−1,0,+1}`.

use alloc::vec;
use alloc::vec::Vec;

use crate::ops::{conv1d_nchw, gemv_bias, global_mean_pool_chw, relu, sigmoid};
use crate::ternary::{
    TernaryOpts, TernaryStats, conv1d_ternary, gemv_bias_ternary, is_ternary_f32, pack_trits,
    ternarize_inplace,
};

#[derive(Debug, Clone)]
pub struct WakeCnnConfig {
    pub n_mels: usize,
    pub c1: usize,
    pub c2: usize,
    pub c3: usize,
    pub k: usize,
    pub hidden: usize,
    /// The last layer is averaged over this many consecutive stretches of the window, kept in
    /// order (1: over the whole window). With phrases that share a word ("Hey/Bye/Hi Peachy"),
    /// a single average loses what came before it; a few stretches keep the order.
    pub pool_chunks: usize,
}

impl WakeCnnConfig {
    pub fn lite() -> Self {
        Self {
            n_mels: 32,
            c1: 16,
            c2: 32,
            c3: 32,
            k: 3,
            hidden: 64,
            pool_chunks: 1,
        }
    }

    pub fn full() -> Self {
        Self {
            n_mels: 32,
            c1: 64,
            c2: 128,
            c3: 128,
            k: 3,
            hidden: 256,
            pool_chunks: 1,
        }
    }
}

#[derive(Clone)]
pub struct WakeCnnWeights {
    pub cfg: WakeCnnConfig,
    pub conv1_w: Vec<f32>,
    pub conv1_b: Vec<f32>,
    pub conv2_w: Vec<f32>,
    pub conv2_b: Vec<f32>,
    pub conv3_w: Vec<f32>,
    pub conv3_b: Vec<f32>,
    pub fc1_w: Vec<f32>,
    pub fc1_b: Vec<f32>,
    pub fc2_w: Vec<f32>,
    pub fc2_b: Vec<f32>,
}

impl WakeCnnWeights {
    /// Deterministic synthetic weights for CI / benches (low scores on silence).
    pub fn stub(cfg: WakeCnnConfig) -> Self {
        let mut rng = 0xC0FFEEu64;
        let mut next = || {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((rng >> 33) as f32 / (1u64 << 31) as f32) * 0.02 - 0.01
        };
        let fill =
            |n: usize, f: &mut dyn FnMut() -> f32| -> Vec<f32> { (0..n).map(|_| f()).collect() };
        let k = cfg.k;
        Self {
            conv1_w: fill(cfg.c1 * cfg.n_mels * k, &mut next),
            conv1_b: vec![0.0; cfg.c1],
            conv2_w: fill(cfg.c2 * cfg.c1 * k, &mut next),
            conv2_b: vec![0.0; cfg.c2],
            conv3_w: fill(cfg.c3 * cfg.c2 * k, &mut next),
            conv3_b: vec![0.0; cfg.c3],
            fc1_w: fill(cfg.hidden * cfg.c3 * cfg.pool_chunks.max(1), &mut next),
            fc1_b: vec![0.0; cfg.hidden],
            fc2_w: fill(cfg.hidden, &mut next),
            fc2_b: vec![-2.0],
            cfg,
        }
    }

    /// Weights to start training from: He (Kaiming) uniform, `±sqrt(6 / fan_in)` per layer,
    /// zero biases. [`Self::stub`]'s ±0.01 everywhere shrinks the signal ~100x per layer, so
    /// after five layers the output no longer depends on the input and SGD only learns the
    /// final bias: every trained phrase answered one class (or 0.5) to everything.
    pub fn he_init(cfg: WakeCnnConfig, seed: u64) -> Self {
        let mut rng = seed ^ 0x9E37_79B9_7F4A_7C15;
        let mut unit = move || {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (rng >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
        };
        let mut fill = |n: usize, fan_in: usize| -> Vec<f32> {
            let a = sqrt(6.0 / fan_in.max(1) as f32);
            (0..n).map(|_| unit() * a).collect()
        };
        let k = cfg.k;
        Self {
            conv1_w: fill(cfg.c1 * cfg.n_mels * k, cfg.n_mels * k),
            conv1_b: vec![0.0; cfg.c1],
            conv2_w: fill(cfg.c2 * cfg.c1 * k, cfg.c1 * k),
            conv2_b: vec![0.0; cfg.c2],
            conv3_w: fill(cfg.c3 * cfg.c2 * k, cfg.c2 * k),
            conv3_b: vec![0.0; cfg.c3],
            fc1_w: fill(
                cfg.hidden * cfg.c3 * cfg.pool_chunks.max(1),
                cfg.c3 * cfg.pool_chunks.max(1),
            ),
            fc1_b: vec![0.0; cfg.hidden],
            fc2_w: fill(cfg.hidden, cfg.hidden),
            fc2_b: vec![0.0],
            cfg,
        }
    }

    /// Build from the same field layout as `rlx_wake::WakeCnnWeights`.
    pub fn from_parts(
        cfg: WakeCnnConfig,
        conv1_w: Vec<f32>,
        conv1_b: Vec<f32>,
        conv2_w: Vec<f32>,
        conv2_b: Vec<f32>,
        conv3_w: Vec<f32>,
        conv3_b: Vec<f32>,
        fc1_w: Vec<f32>,
        fc1_b: Vec<f32>,
        fc2_w: Vec<f32>,
        fc2_b: Vec<f32>,
    ) -> Self {
        Self {
            cfg,
            conv1_w,
            conv1_b,
            conv2_w,
            conv2_b,
            conv3_w,
            conv3_b,
            fc1_w,
            fc1_b,
            fc2_w,
            fc2_b,
        }
    }

    /// Convert selected weight tensors to exact `{−1,0,+1}` (biases unchanged).
    ///
    /// Exact ternary MatMul weights are what `rlx-bake` packs as GGUF TQ2_0 +
    /// `DequantMatMul` (add/sub/skip). Inference uses fused ternary kernels.
    pub fn ternarize(&mut self, opts: TernaryOpts) -> TernaryStats {
        let mut stats = TernaryStats::default();
        let mut apply = |w: &mut Vec<f32>| {
            ternarize_inplace(w, opts.keep_frac);
            stats.tensors += 1;
            stats.elems += w.len();
            stats.nonzero += w.iter().filter(|&&v| v != 0.0).count();
            stats.bytes_f32 += w.len() * 4;
            stats.bytes_packed += pack_trits(w).len();
        };
        if opts.conv {
            apply(&mut self.conv1_w);
            apply(&mut self.conv2_w);
            apply(&mut self.conv3_w);
        }
        if opts.fc {
            apply(&mut self.fc1_w);
            apply(&mut self.fc2_w);
        }
        stats
    }

    /// Outputs (phrases) the network scores: the last layer's bias length.
    pub fn n_out(&self) -> usize {
        self.fc2_b.len().max(1)
    }

    pub fn fc_ternary(&self) -> bool {
        is_ternary_f32(&self.fc1_w) && is_ternary_f32(&self.fc2_w)
    }

    pub fn conv_ternary(&self) -> bool {
        is_ternary_f32(&self.conv1_w)
            && is_ternary_f32(&self.conv2_w)
            && is_ternary_f32(&self.conv3_w)
    }
}

/// The CNN over a sliding window of mel frames, scored as frames arrive.
///
/// Streaming: each layer's outputs for the last scored window are kept. When the window
/// slides by a multiple of the network's total stride (4 frames: one 40 ms hop), only the
/// positions that changed are recomputed in each layer: a few at the right (the new frames,
/// and those computed with zero padding past the old window's end) and at the left (those
/// reading the zero padding before the window); the rest are the previous values, shifted.
/// The arithmetic and its order are the full forward's, so scores are bit-identical to
/// recomputing the whole window, at a fraction of the cost (a few frames per layer instead
/// of the window). Anything else (a window not yet full, frames slid in without scoring,
/// an odd shift, ternary weights) recomputes the whole window.
pub struct WakeCnn {
    weights: WakeCnnWeights,
    mel_buf: Vec<f32>,
    window_frames: usize,
    /// Channel-major input window and each layer's (post-ReLU) outputs.
    x: Vec<f32>,
    y1: Vec<f32>,
    y2: Vec<f32>,
    y3: Vec<f32>,
    /// The layer state is the last scored window's, `pending` frames ago.
    stream: bool,
    pending: usize,
    /// The last scores, one per output.
    probs: Vec<f32>,
    /// Scratch for the head.
    pooled: Vec<f32>,
    hidden: Vec<f32>,
    patch: Vec<f32>,
}

/// One time-conv layer: kernel, stride, padding.
const STRIDES: [usize; 3] = [1, 2, 2];

fn conv_out_len(t_in: usize, k: usize, stride: usize, pad: usize) -> usize {
    if t_in + 2 * pad >= k {
        (t_in + 2 * pad - k) / stride + 1
    } else {
        0
    }
}

impl WakeCnn {
    pub fn new(weights: WakeCnnWeights) -> Self {
        let n_out = weights.n_out();
        Self {
            window_frames: 40,
            weights,
            mel_buf: Vec::new(),
            x: Vec::new(),
            y1: Vec::new(),
            y2: Vec::new(),
            y3: Vec::new(),
            stream: false,
            pending: 0,
            probs: vec![0.0; n_out],
            pooled: Vec::new(),
            hidden: Vec::new(),
            patch: Vec::new(),
        }
    }

    pub fn with_window_frames(mut self, frames: usize) -> Self {
        self.window_frames = frames.max(8);
        self.stream = false;
        self
    }

    pub fn weights(&self) -> &WakeCnnWeights {
        &self.weights
    }

    /// How many scores each window gives (one per phrase the network was trained for).
    pub fn n_out(&self) -> usize {
        self.weights.n_out()
    }

    pub fn reset(&mut self) {
        self.mel_buf.clear();
        self.stream = false;
        self.pending = 0;
    }

    fn append(&mut self, frames: &[f32]) {
        let n_mels = self.weights.cfg.n_mels;
        self.mel_buf.extend_from_slice(frames);
        // Compacted at twice the window, so sliding by a hop costs no shifting.
        let max_len = self.window_frames * n_mels;
        if self.mel_buf.len() >= 2 * max_len {
            let drop = self.mel_buf.len() - max_len;
            self.mel_buf.drain(..drop);
        }
    }

    /// Slide `frames` into the window without scoring, so that a gated stretch (the VAD
    /// not yet sure it hears speech) still leaves the window continuous: the phrase onset
    /// is there when scoring resumes.
    pub fn append_mel_frames(&mut self, frames: &[f32]) {
        let n_mels = self.weights.cfg.n_mels;
        if frames.is_empty() || !frames.len().is_multiple_of(n_mels) {
            return;
        }
        self.append(frames);
        self.stream = false;
    }

    /// The scored window: the last `window_frames` frames.
    fn window(&self) -> &[f32] {
        let max_len = self.window_frames * self.weights.cfg.n_mels;
        &self.mel_buf[self.mel_buf.len().saturating_sub(max_len)..]
    }

    /// Slide `frames` in and score: the first output's probability (0 until the window is
    /// full). See [`Self::push_mel_frames_all`].
    pub fn push_mel_frames(&mut self, frames: &[f32]) -> f32 {
        self.push_mel_frames_all(frames).map_or(0.0, |p| p[0])
    }

    /// Slide `frames` (frame-major, `n_mels` each) in and score: every output's probability,
    /// or `None` until the window is full (the model is trained on `window_frames`-long
    /// windows, and a shorter one, the first `context_ms` of a stream, scores out of
    /// distribution).
    pub fn push_mel_frames_all(&mut self, frames: &[f32]) -> Option<&[f32]> {
        let n_mels = self.weights.cfg.n_mels;
        if frames.is_empty() || !frames.len().is_multiple_of(n_mels) {
            return None;
        }
        self.append(frames);
        self.pending += frames.len() / n_mels;
        if self.window().len() < n_mels * self.window_frames {
            self.stream = false;
            return None;
        }
        let ternary = self.weights.conv_ternary();
        if !(self.stream && !ternary && self.forward_stream(self.pending)) {
            self.forward_full();
        }
        self.stream = !ternary;
        self.pending = 0;
        self.head();
        Some(&self.probs)
    }

    /// Layer `l`'s (input channels, output channels, kernel weights, bias).
    fn layer(&self, l: usize) -> (usize, usize, &[f32], &[f32]) {
        let c = &self.weights.cfg;
        let w = &self.weights;
        match l {
            0 => (c.n_mels, c.c1, &w.conv1_w, &w.conv1_b),
            1 => (c.c1, c.c2, &w.conv2_w, &w.conv2_b),
            _ => (c.c2, c.c3, &w.conv3_w, &w.conv3_b),
        }
    }

    /// Output lengths of the three layers for the current window.
    fn lens(&self) -> [usize; 4] {
        let (k, pad) = (self.weights.cfg.k, self.weights.cfg.k / 2);
        let t0 = self.window_frames;
        let t1 = conv_out_len(t0, k, STRIDES[0], pad);
        let t2 = conv_out_len(t1, k, STRIDES[1], pad);
        let t3 = conv_out_len(t2, k, STRIDES[2], pad);
        [t0, t1, t2, t3]
    }

    /// Recompute layer `l`'s outputs `[lo, hi)` from the layer below, then ReLU them.
    fn conv_part(&mut self, l: usize, lo: usize, hi: usize) {
        let (k, pad) = (self.weights.cfg.k, self.weights.cfg.k / 2);
        let t = self.lens();
        let (in_c, out_c, _, _) = self.layer(l);
        let mut out = core::mem::take(match l {
            0 => &mut self.y1,
            1 => &mut self.y2,
            _ => &mut self.y3,
        });
        let mut patch = core::mem::take(&mut self.patch);
        {
            let input: &[f32] = match l {
                0 => &self.x,
                1 => &self.y1,
                _ => &self.y2,
            };
            let (_, _, w, b) = self.layer(l);
            crate::ops::conv1d_positions(
                input,
                in_c,
                t[l],
                w,
                out_c,
                k,
                STRIDES[l],
                pad,
                b,
                &mut out,
                t[l + 1],
                lo,
                hi,
                &mut patch,
            );
        }
        self.patch = patch;
        for c in 0..out_c {
            for v in &mut out[c * t[l + 1] + lo..c * t[l + 1] + hi.min(t[l + 1])] {
                *v = relu(*v);
            }
        }
        match l {
            0 => self.y1 = out,
            1 => self.y2 = out,
            _ => self.y3 = out,
        }
    }

    /// The input window, channel-major, into `self.x` (columns `from..` only).
    fn fill_x(&mut self, from: usize) {
        let n_mels = self.weights.cfg.n_mels;
        let t = self.window_frames;
        let start = self.mel_buf.len() - t * n_mels;
        for ti in from..t {
            for m in 0..n_mels {
                self.x[m * t + ti] = self.mel_buf[start + ti * n_mels + m];
            }
        }
    }

    fn forward_full(&mut self) {
        let cfg = self.weights.cfg.clone();
        let t = self.lens();
        self.x.resize(cfg.n_mels * t[0], 0.0);
        self.fill_x(0);
        self.y1.resize(cfg.c1 * t[1], 0.0);
        self.y2.resize(cfg.c2 * t[2], 0.0);
        self.y3.resize(cfg.c3 * t[3], 0.0);
        if !self.weights.conv_ternary() {
            for l in 0..3 {
                self.conv_part(l, 0, t[l + 1]);
            }
            return;
        }
        // Ternary weights: the fused add/sub kernels, whole window.
        let (k, pad) = (cfg.k, cfg.k / 2);
        for l in 0..3 {
            let (in_c, out_c, w, b) = self.layer(l);
            let (w, b) = (w.to_vec(), b.to_vec());
            let input = match l {
                0 => self.x.clone(),
                1 => self.y1.clone(),
                _ => self.y2.clone(),
            };
            let out = match l {
                0 => &mut self.y1,
                1 => &mut self.y2,
                _ => &mut self.y3,
            };
            if is_ternary_f32(&w) {
                conv1d_ternary(
                    &input,
                    in_c,
                    t[l],
                    &w,
                    out_c,
                    k,
                    STRIDES[l],
                    pad,
                    Some(&b),
                    out,
                );
            } else {
                conv1d_nchw(
                    &input,
                    in_c,
                    t[l],
                    &w,
                    out_c,
                    k,
                    STRIDES[l],
                    pad,
                    Some(&b),
                    out,
                );
            }
            for v in out.iter_mut() {
                *v = relu(*v);
            }
        }
    }

    /// Slide the layer state by `n` input frames and recompute what changed. `false` (state
    /// untouched) if `n` does not slide every layer by whole positions or is a window or more.
    fn forward_stream(&mut self, n: usize) -> bool {
        let cfg = self.weights.cfg.clone();
        let t = self.lens();
        let total: usize = STRIDES.iter().product();
        if n == 0 || !n.is_multiple_of(total) || n >= t[0] || self.x.len() != cfg.n_mels * t[0] {
            return false;
        }
        let (k, pad) = (cfg.k, cfg.k / 2);
        // Input: shift, then the new frames.
        let t0 = t[0];
        for m in 0..cfg.n_mels {
            self.x.copy_within(m * t0 + n..(m + 1) * t0, m * t0);
        }
        self.fill_x(t0 - n);
        // Changed positions of the layer below: `[0, a)` and `[b, t)`; `shift` is its slide.
        let (mut a, mut b, mut shift) = (0usize, t0 - n, n);
        for l in 0..3 {
            let s = STRIDES[l];
            let t_out = t[l + 1];
            let shift_out = shift / s;
            // Left: any output reading a changed input or the zero padding before the window.
            let mut a_out = if a == 0 { 0 } else { (a - 1 + pad) / s + 1 };
            if pad > 0 {
                a_out = a_out.max(1);
            }
            let a_out = a_out.min(t_out);
            // Right: any output reading a changed input (or padding past the window's end).
            let b_out = if b + pad >= k - 1 {
                (b + pad - (k - 1)).div_ceil(s)
            } else {
                0
            }
            .min(t_out);
            let ch = self.layer(l).1;
            let arr = match l {
                0 => &mut self.y1,
                1 => &mut self.y2,
                _ => &mut self.y3,
            };
            for c in 0..ch {
                arr.copy_within(c * t_out + shift_out..(c + 1) * t_out, c * t_out);
            }
            if a_out >= b_out {
                self.conv_part(l, 0, t_out);
            } else {
                self.conv_part(l, 0, a_out);
                self.conv_part(l, b_out, t_out);
            }
            (a, b, shift) = (a_out, b_out, shift_out);
        }
        true
    }

    /// Mean-pool the last layer and run the two dense layers: `self.probs`.
    fn head(&mut self) {
        let cfg = self.weights.cfg.clone();
        let t3 = self.lens()[3];
        let n_out = self.weights.n_out();
        let chunks = cfg.pool_chunks.max(1);
        self.pooled.resize(cfg.c3 * chunks, 0.0);
        if chunks == 1 {
            global_mean_pool_chw(&self.y3[..cfg.c3 * t3], cfg.c3, t3, &mut self.pooled);
        } else {
            pool_chunks(
                &self.y3[..cfg.c3 * t3],
                cfg.c3,
                t3,
                chunks,
                &mut self.pooled,
            );
        }
        let n_in = cfg.c3 * chunks;
        self.hidden.resize(cfg.hidden, 0.0);
        let fc_t = self.weights.fc_ternary();
        let w = &self.weights;
        if fc_t {
            gemv_bias_ternary(
                cfg.hidden,
                n_in,
                &w.fc1_w,
                &self.pooled,
                &w.fc1_b,
                &mut self.hidden,
            );
        } else {
            gemv_bias(
                cfg.hidden,
                n_in,
                &w.fc1_w,
                &self.pooled,
                &w.fc1_b,
                &mut self.hidden,
            );
        }
        for v in &mut self.hidden {
            *v = relu(*v);
        }
        self.probs.resize(n_out, 0.0);
        if fc_t {
            gemv_bias_ternary(
                n_out,
                cfg.hidden,
                &w.fc2_w,
                &self.hidden,
                &w.fc2_b,
                &mut self.probs,
            );
        } else {
            gemv_bias(
                n_out,
                cfg.hidden,
                &w.fc2_w,
                &self.hidden,
                &w.fc2_b,
                &mut self.probs,
            );
        }
        for p in &mut self.probs {
            *p = sigmoid(*p);
        }
    }
}

/// The stretches `[i·t / chunks, (i+1)·t / chunks)` of a channel-major `[c, t]` map, each
/// averaged per channel, chunk-major into `out [chunks · c]`. Shared with training's graph.
pub fn pool_chunks(y: &[f32], c: usize, t: usize, chunks: usize, out: &mut [f32]) {
    for i in 0..chunks {
        let (lo, hi) = (
            i * t / chunks,
            ((i + 1) * t / chunks).max(i * t / chunks + 1).min(t),
        );
        for ch in 0..c {
            let row = &y[ch * t + lo..ch * t + hi];
            out[i * c + ch] = row.iter().sum::<f32>() / row.len().max(1) as f32;
        }
    }
}

/// `x.sqrt()` without std (this crate is `no_std`): Newton's method, for initialisation scales.
fn sqrt(x: f32) -> f32 {
    if x <= 0.0 {
        return 0.0;
    }
    let mut r = if x > 1.0 { x / 2.0 } else { 1.0 };
    for _ in 0..30 {
        r = 0.5 * (r + x / r);
    }
    r
}
