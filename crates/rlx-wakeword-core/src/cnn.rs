// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Compact wake CNN (same tensor layout as `rlx-wake`).
//!
//! After [`WakeCnnWeights::ternarize`], forward uses fused ternary conv/GEMV when
//! weight tensors are exact `{−1,0,+1}`.

use alloc::vec;
use alloc::vec::Vec;

use crate::ops::{conv1d_dilated, conv1d_nchw, gemv_bias, global_mean_pool_chw, relu, sigmoid};
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
    /// order (1: over the whole window). With phrases that share a word ("Hey/Bye/Hi <name>"),
    /// a single average loses what came before it; a few stretches keep the order.
    pub pool_chunks: usize,
    /// Residual blocks after the third layer, on its 40 ms grid: block `i` a `k`-tap
    /// convolution dilated `2^i` (same length), added to its input, then ReLU. Three plain
    /// layers hear 90 ms each; four blocks widen that to about 1.3 s. 0: none.
    pub res_blocks: usize,
    /// Attention pooling: the window's positions weighted by a learned, bounded score
    /// (softmax over time), one more `c3` vector after the stretch averages.
    pub attn: bool,
    /// PCEN front end ([`crate::pcen`]) on the input frames instead of plain log-mel.
    pub pcen: bool,
    /// Both: each frame is its `n_mels` log-mel values then their PCEN (the first layer reads
    /// `2·n_mels` channels; `pcen` is then implied).
    pub dual: bool,
    /// Fully streaming: every convolution causal and unpadded ("valid"), so a position never
    /// changes once computed and each 40 ms hop adds one position per layer past the first two
    /// (the window has no edges to recompute). The first layers shorten the window by their
    /// taps, each residual block by `2·dilation`; the pooled positions are what remains (train
    /// with a window that long plus about 1.2 s).
    pub causal: bool,
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
            res_blocks: 0,
            attn: false,
            pcen: false,
            dual: false,
            causal: false,
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
            res_blocks: 0,
            attn: false,
            pcen: false,
            dual: false,
            causal: false,
        }
    }

    /// The convolutions' zero padding each side: `k / 2`, none when `causal`.
    pub fn pad(&self) -> usize {
        if self.causal { 0 } else { self.k / 2 }
    }

    /// The first layer's input channels: `n_mels`, or twice that with `dual`.
    pub fn in_ch(&self) -> usize {
        self.n_mels * (1 + usize::from(self.dual))
    }

    /// The pooled vector's length (the first dense layer's input).
    pub fn pooled_len(&self) -> usize {
        self.c3 * (self.pool_chunks.max(1) + usize::from(self.attn))
    }

    /// Residual block `i`'s dilation.
    pub fn dilation(i: usize) -> usize {
        1 << i
    }
}

/// Attention scores are `ATTN_BOUND · tanh(·)`: bounded, so the softmax needs no max shift and
/// training and the engine compute the same thing.
pub const ATTN_BOUND: f32 = 8.0;

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
    /// Residual blocks (`cfg.res_blocks`): weights `[c3, c3, k]` and biases `[c3]` each.
    pub res_w: Vec<Vec<f32>>,
    pub res_b: Vec<Vec<f32>>,
    /// Attention pooling (`cfg.attn`): score weights `[c3]` and bias `[1]`.
    pub attn_w: Vec<f32>,
    pub attn_b: Vec<f32>,
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
            conv1_w: fill(cfg.c1 * cfg.in_ch() * k, &mut next),
            conv1_b: vec![0.0; cfg.c1],
            conv2_w: fill(cfg.c2 * cfg.c1 * k, &mut next),
            conv2_b: vec![0.0; cfg.c2],
            conv3_w: fill(cfg.c3 * cfg.c2 * k, &mut next),
            conv3_b: vec![0.0; cfg.c3],
            fc1_w: fill(cfg.hidden * cfg.pooled_len(), &mut next),
            fc1_b: vec![0.0; cfg.hidden],
            fc2_w: fill(cfg.hidden, &mut next),
            fc2_b: vec![-2.0],
            res_w: (0..cfg.res_blocks)
                .map(|_| fill(cfg.c3 * cfg.c3 * k, &mut next))
                .collect(),
            res_b: (0..cfg.res_blocks).map(|_| vec![0.0; cfg.c3]).collect(),
            attn_w: if cfg.attn {
                fill(cfg.c3, &mut next)
            } else {
                Vec::new()
            },
            attn_b: if cfg.attn { vec![0.0] } else { Vec::new() },
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
            conv1_w: fill(cfg.c1 * cfg.in_ch() * k, cfg.in_ch() * k),
            conv1_b: vec![0.0; cfg.c1],
            conv2_w: fill(cfg.c2 * cfg.c1 * k, cfg.c1 * k),
            conv2_b: vec![0.0; cfg.c2],
            conv3_w: fill(cfg.c3 * cfg.c2 * k, cfg.c2 * k),
            conv3_b: vec![0.0; cfg.c3],
            fc1_w: fill(cfg.hidden * cfg.pooled_len(), cfg.pooled_len()),
            fc1_b: vec![0.0; cfg.hidden],
            fc2_w: fill(cfg.hidden, cfg.hidden),
            fc2_b: vec![0.0],
            // Residual branches start small: each block begins close to the identity.
            res_w: (0..cfg.res_blocks)
                .map(|_| {
                    fill(cfg.c3 * cfg.c3 * k, cfg.c3 * k)
                        .into_iter()
                        .map(|v| v * 0.1)
                        .collect()
                })
                .collect(),
            res_b: (0..cfg.res_blocks).map(|_| vec![0.0; cfg.c3]).collect(),
            attn_w: if cfg.attn {
                fill(cfg.c3, cfg.c3)
            } else {
                Vec::new()
            },
            attn_b: if cfg.attn { vec![0.0] } else { Vec::new() },
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
            res_w: Vec::new(),
            res_b: Vec::new(),
            attn_w: Vec::new(),
            attn_b: Vec::new(),
        }
    }

    /// The residual blocks' and attention pooling's tensors (`cfg.res_blocks`, `cfg.attn`), for
    /// weights built with [`Self::from_parts`].
    pub fn with_extras(
        mut self,
        res_w: Vec<Vec<f32>>,
        res_b: Vec<Vec<f32>>,
        attn_w: Vec<f32>,
        attn_b: Vec<f32>,
    ) -> Self {
        self.res_w = res_w;
        self.res_b = res_b;
        self.attn_w = attn_w;
        self.attn_b = attn_b;
        self
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
    opts: EngineOpts,
    /// The residual blocks' padded input, kept between hops (`opts.reuse_pad`).
    pad_buf: Vec<f32>,
    mel_buf: Vec<f32>,
    window_frames: usize,
    /// Channel-major input window and each layer's (post-ReLU) outputs.
    x: Vec<f32>,
    y1: Vec<f32>,
    y2: Vec<f32>,
    y3: Vec<f32>,
    /// Each residual block's output (`cfg.res_blocks > 0`), and scratch for a block's branch.
    yr: Vec<Vec<f32>>,
    branch: Vec<f32>,
    /// After a streamed hop: the third layer's changed positions `[0, a)` and `[b, t)` and how
    /// far it slid, for the residual blocks to recompute only what changed. `None`: all.
    stem_change: Option<(usize, usize, usize)>,
    /// The PCEN smoother (`cfg.pcen`), carried across pushes.
    pcen_m: Vec<f32>,
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

/// How the engine computes (all on by default; off, the plainer forms, to measure each).
#[derive(Debug, Clone, Copy)]
pub struct EngineOpts {
    /// Keep the residual blocks' padded input buffer between calls (else allocate each time).
    pub reuse_pad: bool,
    /// The first layers: four output channels per pass over a position's patch.
    pub stem_blocked: bool,
    /// PCEN and attention with [`crate::fastmath`] (else the library's `exp` / `powf` / `tanh`).
    pub fast_math: bool,
}

impl Default for EngineOpts {
    fn default() -> Self {
        Self {
            reuse_pad: true,
            stem_blocked: true,
            fast_math: true,
        }
    }
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
            opts: EngineOpts::default(),
            pad_buf: Vec::new(),
            mel_buf: Vec::new(),
            x: Vec::new(),
            y1: Vec::new(),
            y2: Vec::new(),
            y3: Vec::new(),
            yr: Vec::new(),
            branch: Vec::new(),
            stem_change: None,
            pcen_m: Vec::new(),
            stream: false,
            pending: 0,
            probs: vec![0.0; n_out],
            pooled: Vec::new(),
            hidden: Vec::new(),
            patch: Vec::new(),
        }
    }

    /// How to compute ([`EngineOpts`]); scores are the same either way except for `fast_math`
    /// (within float rounding).
    pub fn with_opts(mut self, opts: EngineOpts) -> Self {
        self.opts = opts;
        self
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
        self.pcen_m.clear();
        self.stream = false;
        self.pending = 0;
    }

    fn append(&mut self, frames: &[f32]) {
        let cfg = &self.weights.cfg;
        let n_mels = cfg.n_mels;
        let at = self.mel_buf.len();
        if cfg.dual {
            // Each frame stored as its log-mel values then their PCEN.
            let mut p = frames.to_vec();
            if self.opts.fast_math {
                crate::pcen::pcen_inplace(&mut p, n_mels, &mut self.pcen_m);
            } else {
                crate::pcen::pcen_inplace_libm(&mut p, n_mels, &mut self.pcen_m);
            }
            for (raw, norm) in frames.chunks_exact(n_mels).zip(p.chunks_exact(n_mels)) {
                self.mel_buf.extend_from_slice(raw);
                self.mel_buf.extend_from_slice(norm);
            }
        } else {
            self.mel_buf.extend_from_slice(frames);
            if cfg.pcen {
                if self.opts.fast_math {
                    crate::pcen::pcen_inplace(&mut self.mel_buf[at..], n_mels, &mut self.pcen_m);
                } else {
                    crate::pcen::pcen_inplace_libm(&mut self.mel_buf[at..], n_mels, &mut self.pcen_m);
                }
            }
        }
        // Compacted at twice the window, so sliding by a hop costs no shifting.
        let max_len = self.window_frames * cfg.in_ch();
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

    /// Slide `frames` in without scoring now, keeping the streaming state: the next score
    /// updates incrementally over everything queued (scoring every other hop: 8 frames at a
    /// time instead of 4, at no extra cost per score).
    pub fn queue_frames(&mut self, frames: &[f32]) {
        let n_mels = self.weights.cfg.n_mels;
        if frames.is_empty() || !frames.len().is_multiple_of(n_mels) {
            return;
        }
        self.append(frames);
        self.pending += frames.len() / n_mels;
    }

    /// The scored window: the last `window_frames` frames.
    fn window(&self) -> &[f32] {
        let max_len = self.window_frames * self.weights.cfg.in_ch();
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
        if self.window().len() < self.weights.cfg.in_ch() * self.window_frames {
            self.stream = false;
            return None;
        }
        let ternary = self.weights.conv_ternary();
        if !(self.stream && !ternary && self.forward_stream(self.pending)) {
            self.forward_full();
        }
        self.stream = !ternary;
        self.pending = 0;
        self.res_forward();
        self.head();
        Some(&self.probs)
    }

    /// Score the current window now, from scratch (whatever was slid in without scoring):
    /// a verifier fed with [`Self::append_mel_frames`] and asked only when a detector fires.
    /// `None` until the window is full.
    pub fn score_window(&mut self) -> Option<&[f32]> {
        if self.window().len() < self.weights.cfg.in_ch() * self.window_frames {
            return None;
        }
        self.forward_full();
        self.stream = false;
        self.pending = 0;
        self.res_forward();
        self.head();
        Some(&self.probs)
    }

    /// The residual blocks over the third layer's output. After a streamed hop only what
    /// changed: block `i` (dilation `d`) reads positions `p − d`, `p`, `p + d`, so its changed
    /// outputs are the input's changed ones widened by `d` (plus the first `d`, which read the
    /// padding before the window); the rest are the previous outputs, slid. Exact either way.
    fn res_forward(&mut self) {
        let cfg = &self.weights.cfg;
        let change = self.stem_change.take();
        if cfg.res_blocks == 0 {
            return;
        }
        if cfg.causal {
            return self.res_forward_causal(change);
        }
        let (c, k) = (cfg.c3, cfg.k);
        let t = self.lens()[3];
        let mut yr = core::mem::take(&mut self.yr);
        let mut branch = core::mem::take(&mut self.branch);
        branch.resize(c * t, 0.0);
        let reuse = self.opts.reuse_pad;
        let mut pad = core::mem::take(&mut self.pad_buf);
        let fresh = yr.len() != cfg.res_blocks || yr.iter().any(|v| v.len() != c * t);
        if fresh {
            yr = alloc::vec![alloc::vec![0.0; c * t]; cfg.res_blocks];
        }
        // Changed input positions `[0, a)` ∪ `[b, t)` (all: `a = t`), and the slide.
        let (mut a, mut b, shift) = match change {
            Some(ch) if !fresh => ch,
            _ => (t, 0, 0),
        };
        for i in 0..cfg.res_blocks {
            let d = WakeCnnConfig::dilation(i);
            let (a_out, b_out) = ((a + d).min(t), b.saturating_sub(d));
            let ranges: &[(usize, usize)] = if a_out >= b_out {
                &[(0, t), (t, t)]
            } else {
                &[(0, a_out), (b_out, t)]
            };
            let (before, rest) = yr.split_at_mut(i);
            let input: &[f32] = if i == 0 {
                &self.y3[..c * t]
            } else {
                &before[i - 1]
            };
            let out = &mut rest[0];
            if a_out < b_out && shift > 0 {
                for ch in 0..c {
                    out.copy_within(ch * t + shift..(ch + 1) * t, ch * t);
                }
            }
            for &(lo, hi) in ranges {
                conv1d_dilated(
                    input,
                    c,
                    t,
                    &self.weights.res_w[i],
                    c,
                    k,
                    d,
                    &self.weights.res_b[i],
                    &mut branch,
                    lo,
                    hi,
                    &mut pad,
                );
                if !reuse {
                    // The plain form: a fresh buffer each call.
                    pad = Vec::new();
                }
                for ch in 0..c {
                    for p in lo..hi {
                        out[ch * t + p] = relu(input[ch * t + p] + branch[ch * t + p]);
                    }
                }
            }
            (a, b) = (a_out, b_out);
        }
        self.yr = yr;
        self.branch = branch;
        self.pad_buf = pad;
    }

    /// Layer `l`'s (input channels, output channels, kernel weights, bias).
    fn layer(&self, l: usize) -> (usize, usize, &[f32], &[f32]) {
        let c = &self.weights.cfg;
        let w = &self.weights;
        match l {
            0 => (c.in_ch(), c.c1, &w.conv1_w, &w.conv1_b),
            1 => (c.c1, c.c2, &w.conv2_w, &w.conv2_b),
            _ => (c.c2, c.c3, &w.conv3_w, &w.conv3_b),
        }
    }

    /// Output lengths of the three layers for the current window.
    fn lens(&self) -> [usize; 4] {
        let (k, pad) = (self.weights.cfg.k, self.weights.cfg.pad());
        let t0 = self.window_frames;
        let t1 = conv_out_len(t0, k, STRIDES[0], pad);
        let t2 = conv_out_len(t1, k, STRIDES[1], pad);
        let t3 = conv_out_len(t2, k, STRIDES[2], pad);
        [t0, t1, t2, t3]
    }

    /// Recompute layer `l`'s outputs `[lo, hi)` from the layer below, then ReLU them.
    fn conv_part(&mut self, l: usize, lo: usize, hi: usize) {
        let (k, pad) = (self.weights.cfg.k, self.weights.cfg.pad());
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
                self.opts.stem_blocked,
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
        let w = self.weights.cfg.in_ch();
        let t = self.window_frames;
        let start = self.mel_buf.len() - t * w;
        for ti in from..t {
            for m in 0..w {
                self.x[m * t + ti] = self.mel_buf[start + ti * w + m];
            }
        }
    }

    fn forward_full(&mut self) {
        self.stem_change = None;
        let cfg = self.weights.cfg.clone();
        let t = self.lens();
        self.x.resize(cfg.in_ch() * t[0], 0.0);
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
        let (k, pad) = (cfg.k, cfg.pad());
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
        if n == 0 || !n.is_multiple_of(total) || n >= t[0] || self.x.len() != cfg.in_ch() * t[0] {
            return false;
        }
        let (k, pad) = (cfg.k, cfg.pad());
        // Input: shift, then the new frames.
        let t0 = t[0];
        for m in 0..cfg.in_ch() {
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
        self.stem_change = Some((a, b, shift));
        true
    }

    /// The residual blocks' output lengths when `causal` (each `2·dilation` shorter).
    fn causal_lens(&self) -> Vec<usize> {
        let cfg = &self.weights.cfg;
        let mut len = self.lens()[3];
        (0..cfg.res_blocks)
            .map(|i| {
                len = len.saturating_sub(2 * WakeCnnConfig::dilation(i) * (cfg.k / 2));
                len
            })
            .collect()
    }

    /// [`Self::res_forward`] for a `causal` network: block `i`'s output `q` reads its input at
    /// `q`, `q + d`, `q + 2d` (the last one the residual), so after a hop only the newest
    /// outputs (those reading what changed below) are computed; the rest slide.
    fn res_forward_causal(&mut self, change: Option<(usize, usize, usize)>) {
        let cfg = self.weights.cfg.clone();
        let (c, k) = (cfg.c3, cfg.k);
        let t3 = self.lens()[3];
        let lens = self.causal_lens();
        let mut yr = core::mem::take(&mut self.yr);
        let mut branch = core::mem::take(&mut self.branch);
        branch.resize(c * t3, 0.0);
        let mut pad = core::mem::take(&mut self.pad_buf);
        let fresh = yr.len() != cfg.res_blocks || yr.iter().zip(&lens).any(|(v, &l)| v.len() != c * l);
        if fresh {
            yr = lens.iter().map(|&l| alloc::vec![0.0; c * l]).collect();
        }
        // Changed input positions `[b, len)` and the slide; everything when fresh.
        let (mut b, shift, all) = match change {
            Some((_, b, shift)) if !fresh => (b, shift, false),
            _ => (0, 0, true),
        };
        let mut in_len = t3;
        for i in 0..cfg.res_blocks {
            let d = WakeCnnConfig::dilation(i);
            let span = 2 * d * (k / 2);
            let out_len = lens[i];
            let b_out = if all { 0 } else { b.saturating_sub(span) };
            let (before, rest) = yr.split_at_mut(i);
            let input: &[f32] = if i == 0 { &self.y3[..c * t3] } else { &before[i - 1] };
            let out = &mut rest[0];
            if !all && shift > 0 && shift < out_len {
                for ch in 0..c {
                    out.copy_within(ch * out_len + shift..(ch + 1) * out_len, ch * out_len);
                }
            }
            // Output `q` reads input `q + j·d` (taps j = 0..k): one patch per position, then
            // four output channels per pass (NEON); the same per-position arithmetic whether
            // one position or all are computed, so streaming stays exact.
            let w = &self.weights.res_w[i];
            let bias = &self.weights.res_b[i];
            let n = c * k;
            pad.resize(n, 0.0);
            for q in b_out..out_len {
                for ch in 0..c {
                    for j in 0..k {
                        pad[ch * k + j] = input[ch * in_len + q + j * d];
                    }
                }
                let mut o = 0;
                while o + 4 <= c {
                    let r = crate::ops::dot4(
                        [&w[o * n..(o + 1) * n], &w[(o + 1) * n..(o + 2) * n], &w[(o + 2) * n..(o + 3) * n], &w[(o + 3) * n..(o + 4) * n]],
                        &pad[..n],
                    );
                    for (u, v) in r.iter().enumerate() {
                        branch[(o + u) * in_len + q] = bias[o + u] + v;
                    }
                    o += 4;
                }
                for o in o..c {
                    branch[o * in_len + q] = bias[o] + crate::ops::dot(&w[o * n..(o + 1) * n], &pad[..n]);
                }
                for ch in 0..c {
                    out[ch * out_len + q] = relu(input[ch * in_len + q + span] + branch[ch * in_len + q]);
                }
            }
            if !self.opts.reuse_pad {
                pad = Vec::new();
            }
            (b, in_len) = (b_out, out_len);
        }
        self.yr = yr;
        self.branch = branch;
        self.pad_buf = pad;
    }

    /// Mean-pool the last layer and run the two dense layers: `self.probs`.
    fn head(&mut self) {
        let cfg = self.weights.cfg.clone();
        // The pooled positions: the third layer's, or the last residual block's when causal.
        let t3 = if cfg.causal && cfg.res_blocks > 0 {
            *self.causal_lens().last().expect("blocks")
        } else {
            self.lens()[3]
        };
        let n_out = self.weights.n_out();
        let chunks = cfg.pool_chunks.max(1);
        let top: &[f32] = if cfg.res_blocks > 0 {
            &self.yr[cfg.res_blocks - 1]
        } else {
            &self.y3[..cfg.c3 * t3]
        };
        self.pooled.resize(cfg.pooled_len(), 0.0);
        if chunks == 1 {
            global_mean_pool_chw(top, cfg.c3, t3, &mut self.pooled[..cfg.c3]);
        } else {
            pool_chunks(top, cfg.c3, t3, chunks, &mut self.pooled[..cfg.c3 * chunks]);
        }
        if cfg.attn && self.opts.fast_math {
            attention_pool_fast(
                top,
                cfg.c3,
                t3,
                &self.weights.attn_w,
                self.weights.attn_b[0],
                &mut self.pooled[cfg.c3 * chunks..],
            );
        } else if cfg.attn {
            attention_pool(
                top,
                cfg.c3,
                t3,
                &self.weights.attn_w,
                self.weights.attn_b[0],
                &mut self.pooled[cfg.c3 * chunks..],
            );
        }
        let n_in = cfg.pooled_len();
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

/// Attention pooling of a channel-major `[c, t]` map: score per position
/// `ATTN_BOUND · tanh(b + Σ_c w_c·y[c, t])`, softmax over positions, the weighted mean of each
/// channel into `out [c]`. Shared with training's graph.
pub fn attention_pool(y: &[f32], c: usize, t: usize, w: &[f32], b: f32, out: &mut [f32]) {
    let mut a = vec![0.0f32; t];
    for (tt, s) in a.iter_mut().enumerate() {
        let mut u = b;
        for ch in 0..c {
            u += w[ch] * y[ch * t + tt];
        }
        *s = (ATTN_BOUND * u.tanh()).exp();
    }
    let sum: f32 = a.iter().sum();
    for ch in 0..c {
        let row = &y[ch * t..(ch + 1) * t];
        out[ch] = row.iter().zip(&a).map(|(v, p)| v * p).sum::<f32>() / sum;
    }
}

/// [`attention_pool`] with [`crate::fastmath::exp`] (`tanh u = 1 − 2 / (e^{2u} + 1)`) and the
/// scores' sums and products vectorized.
pub fn attention_pool_fast(y: &[f32], c: usize, t: usize, w: &[f32], b: f32, out: &mut [f32]) {
    use crate::fastmath::exp;
    // u[t] = b + Σ_c w_c · y[c, t], channel rows accumulated whole (vectorizes).
    let mut a = vec![b; t];
    for ch in 0..c {
        let row = &y[ch * t..(ch + 1) * t];
        for (s, &v) in a.iter_mut().zip(row) {
            *s = w[ch].mul_add(v, *s);
        }
    }
    for s in a.iter_mut() {
        let th = 1.0 - 2.0 / (exp(2.0 * *s) + 1.0);
        *s = exp(ATTN_BOUND * th);
    }
    let sum: f32 = a.iter().sum();
    for ch in 0..c {
        out[ch] = crate::ops::dot(&y[ch * t..(ch + 1) * t], &a) / sum;
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
