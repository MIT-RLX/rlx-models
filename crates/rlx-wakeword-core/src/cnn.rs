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
            ((rng >> 33) as f32 / u32::MAX as f32) * 0.02 - 0.01
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
            fc1_w: fill(cfg.hidden * cfg.c3, &mut next),
            fc1_b: vec![0.0; cfg.hidden],
            fc2_w: fill(cfg.hidden, &mut next),
            fc2_b: vec![-2.0],
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

    pub fn fc_ternary(&self) -> bool {
        is_ternary_f32(&self.fc1_w) && is_ternary_f32(&self.fc2_w)
    }

    pub fn conv_ternary(&self) -> bool {
        is_ternary_f32(&self.conv1_w)
            && is_ternary_f32(&self.conv2_w)
            && is_ternary_f32(&self.conv3_w)
    }
}

pub struct WakeCnn {
    weights: WakeCnnWeights,
    mel_buf: Vec<f32>,
    window_frames: usize,
}

impl WakeCnn {
    pub fn new(weights: WakeCnnWeights) -> Self {
        Self {
            window_frames: 40,
            weights,
            mel_buf: Vec::new(),
        }
    }

    pub fn with_window_frames(mut self, frames: usize) -> Self {
        self.window_frames = frames.max(8);
        self
    }

    pub fn weights(&self) -> &WakeCnnWeights {
        &self.weights
    }

    pub fn reset(&mut self) {
        self.mel_buf.clear();
    }

    pub fn push_mel_frames(&mut self, frames: &[f32]) -> f32 {
        let n_mels = self.weights.cfg.n_mels;
        if frames.is_empty() || !frames.len().is_multiple_of(n_mels) {
            return 0.0;
        }
        self.mel_buf.extend_from_slice(frames);
        let max_len = self.window_frames * n_mels;
        if self.mel_buf.len() > max_len {
            let drop = self.mel_buf.len() - max_len;
            self.mel_buf.drain(..drop);
        }
        if self.mel_buf.len() < n_mels * 8 {
            return 0.0;
        }
        self.forward_buffer()
    }

    fn forward_buffer(&self) -> f32 {
        let cfg = &self.weights.cfg;
        let n_mels = cfg.n_mels;
        let t = self.mel_buf.len() / n_mels;
        let mut x = vec![0.0f32; n_mels * t];
        for ti in 0..t {
            for m in 0..n_mels {
                x[m * t + ti] = self.mel_buf[ti * n_mels + m];
            }
        }
        let conv_t = self.weights.conv_ternary();
        let fc_t = self.weights.fc_ternary();

        let mut y1 = vec![0.0f32; cfg.c1 * t];
        let t1 = if conv_t && is_ternary_f32(&self.weights.conv1_w) {
            conv1d_ternary(
                &x,
                n_mels,
                t,
                &self.weights.conv1_w,
                cfg.c1,
                cfg.k,
                1,
                cfg.k / 2,
                Some(&self.weights.conv1_b),
                &mut y1,
            )
        } else {
            conv1d_nchw(
                &x,
                n_mels,
                t,
                &self.weights.conv1_w,
                cfg.c1,
                cfg.k,
                1,
                cfg.k / 2,
                Some(&self.weights.conv1_b),
                &mut y1,
            )
        };
        for v in &mut y1[..cfg.c1 * t1] {
            *v = relu(*v);
        }
        let mut y2 = vec![0.0f32; cfg.c2 * t1];
        let t2 = if conv_t && is_ternary_f32(&self.weights.conv2_w) {
            conv1d_ternary(
                &y1[..cfg.c1 * t1],
                cfg.c1,
                t1,
                &self.weights.conv2_w,
                cfg.c2,
                cfg.k,
                2,
                cfg.k / 2,
                Some(&self.weights.conv2_b),
                &mut y2,
            )
        } else {
            conv1d_nchw(
                &y1[..cfg.c1 * t1],
                cfg.c1,
                t1,
                &self.weights.conv2_w,
                cfg.c2,
                cfg.k,
                2,
                cfg.k / 2,
                Some(&self.weights.conv2_b),
                &mut y2,
            )
        };
        for v in &mut y2[..cfg.c2 * t2] {
            *v = relu(*v);
        }
        let mut y3 = vec![0.0f32; cfg.c3 * t2];
        let t3 = if conv_t && is_ternary_f32(&self.weights.conv3_w) {
            conv1d_ternary(
                &y2[..cfg.c2 * t2],
                cfg.c2,
                t2,
                &self.weights.conv3_w,
                cfg.c3,
                cfg.k,
                2,
                cfg.k / 2,
                Some(&self.weights.conv3_b),
                &mut y3,
            )
        } else {
            conv1d_nchw(
                &y2[..cfg.c2 * t2],
                cfg.c2,
                t2,
                &self.weights.conv3_w,
                cfg.c3,
                cfg.k,
                2,
                cfg.k / 2,
                Some(&self.weights.conv3_b),
                &mut y3,
            )
        };
        for v in &mut y3[..cfg.c3 * t3] {
            *v = relu(*v);
        }
        let mut pooled = vec![0.0f32; cfg.c3];
        global_mean_pool_chw(&y3[..cfg.c3 * t3], cfg.c3, t3, &mut pooled);
        let mut h = vec![0.0f32; cfg.hidden];
        if fc_t {
            gemv_bias_ternary(
                cfg.hidden,
                cfg.c3,
                &self.weights.fc1_w,
                &pooled,
                &self.weights.fc1_b,
                &mut h,
            );
        } else {
            gemv_bias(
                cfg.hidden,
                cfg.c3,
                &self.weights.fc1_w,
                &pooled,
                &self.weights.fc1_b,
                &mut h,
            );
        }
        for v in &mut h {
            *v = relu(*v);
        }
        let mut logit = [0.0f32];
        if fc_t {
            gemv_bias_ternary(
                1,
                cfg.hidden,
                &self.weights.fc2_w,
                &h,
                &self.weights.fc2_b,
                &mut logit,
            );
        } else {
            gemv_bias(
                1,
                cfg.hidden,
                &self.weights.fc2_w,
                &h,
                &self.weights.fc2_b,
                &mut logit,
            );
        }
        sigmoid(logit[0])
    }
}
