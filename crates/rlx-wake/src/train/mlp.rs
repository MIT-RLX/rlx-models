// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Two-layer MLP trainer (used for openWakeWord phrase heads).

use crate::ops::{gemv_bias, relu, sigmoid};
use crate::train::dataset::LabeledClip;
use crate::train::report::TrainReport;
use crate::train::sgd::{SgdConfig, bce_dlogit, bce_loss, sgd_step};

#[derive(Debug, Clone)]
pub struct MlpConfig {
    pub in_dim: usize,
    pub hidden: usize,
}

#[derive(Clone)]
pub struct MlpWeights {
    pub cfg: MlpConfig,
    pub fc1_w: Vec<f32>,
    pub fc1_b: Vec<f32>,
    pub fc2_w: Vec<f32>,
    pub fc2_b: Vec<f32>,
}

impl MlpWeights {
    pub fn new(cfg: MlpConfig, seed: u64) -> Self {
        let mut rng = seed;
        let mut next = || {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((rng >> 33) as f32 / u32::MAX as f32) * 0.04 - 0.02
        };
        Self {
            fc1_w: (0..cfg.hidden * cfg.in_dim).map(|_| next()).collect(),
            fc1_b: vec![0.0; cfg.hidden],
            fc2_w: (0..cfg.hidden).map(|_| next()).collect(),
            fc2_b: vec![0.0],
            cfg,
        }
    }

    pub fn forward(&self, x: &[f32]) -> (f32, Vec<f32>, Vec<f32>) {
        let mut h_pre = vec![0.0f32; self.cfg.hidden];
        gemv_bias(
            self.cfg.hidden,
            self.cfg.in_dim,
            &self.fc1_w,
            x,
            &self.fc1_b,
            &mut h_pre,
        );
        let h: Vec<f32> = h_pre.iter().copied().map(relu).collect();
        let mut logit = [0.0f32];
        gemv_bias(
            1,
            self.cfg.hidden,
            &self.fc2_w,
            &h,
            &self.fc2_b,
            &mut logit,
        );
        (sigmoid(logit[0]), h_pre, h)
    }

    pub fn train_step(&mut self, x: &[f32], label: f32, lr: f32, wd: f32) -> f32 {
        let (prob, h_pre, h) = self.forward(x);
        let loss = bce_loss(prob, label);
        let dlogit = bce_dlogit(prob, label);

        // fc2
        let mut dfc2_w = vec![0.0f32; self.cfg.hidden];
        for i in 0..self.cfg.hidden {
            dfc2_w[i] = dlogit * h[i];
        }
        let dfc2_b = dlogit;

        // dh
        let mut dh = vec![0.0f32; self.cfg.hidden];
        for i in 0..self.cfg.hidden {
            dh[i] = dlogit * self.fc2_w[i];
            if h_pre[i] <= 0.0 {
                dh[i] = 0.0;
            }
        }

        // fc1
        let mut dfc1_w = vec![0.0f32; self.cfg.hidden * self.cfg.in_dim];
        let mut dfc1_b = vec![0.0f32; self.cfg.hidden];
        for o in 0..self.cfg.hidden {
            dfc1_b[o] = dh[o];
            for i in 0..self.cfg.in_dim {
                dfc1_w[o * self.cfg.in_dim + i] = dh[o] * x[i];
            }
        }

        sgd_step(&mut self.fc2_w, &dfc2_w, lr, wd);
        sgd_step(&mut self.fc2_b, &[dfc2_b], lr, wd);
        sgd_step(&mut self.fc1_w, &dfc1_w, lr, wd);
        sgd_step(&mut self.fc1_b, &dfc1_b, lr, wd);
        loss
    }
}

/// Train MLP on fixed feature vectors (one vector per clip).
pub fn train_mlp(
    weights: &mut MlpWeights,
    features: &[(Vec<f32>, f32)],
    sgd: &SgdConfig,
    keyword: &str,
) -> TrainReport {
    let mut initial = 0.0f32;
    let mut final_loss = 0.0f32;
    for epoch in 0..sgd.epochs {
        let mut sum = 0.0f32;
        for (x, y) in features {
            sum += weights.train_step(x, *y, sgd.lr, sgd.weight_decay);
        }
        let mean = sum / features.len().max(1) as f32;
        if epoch == 0 {
            initial = mean;
        }
        final_loss = mean;
        if sgd.log_every > 0 && epoch % sgd.log_every == 0 {
            eprintln!("[rlx-wake-train mlp] epoch={epoch} loss={mean:.4}");
        }
    }
    let mut correct = 0usize;
    for (x, y) in features {
        let (p, _, _) = weights.forward(x);
        let pred = if p >= 0.5 { 1.0 } else { 0.0 };
        if (pred - *y).abs() < 0.5 {
            correct += 1;
        }
    }
    TrainReport {
        epochs: sgd.epochs,
        final_loss,
        initial_loss: initial,
        train_acc: correct as f32 / features.len().max(1) as f32,
        keyword: keyword.into(),
    }
}

/// Mean-pool mel frames → feature vector for MLP wake training.
pub fn mel_mean_feature(pcm: &[f32], n_mels: usize) -> Vec<f32> {
    use crate::train::dataset::clip_mel_frames;
    let frames = clip_mel_frames(pcm);
    if frames.is_empty() {
        return vec![0.0; n_mels];
    }
    let n = frames.len() / n_mels;
    let mut out = vec![0.0f32; n_mels];
    for t in 0..n {
        for m in 0..n_mels {
            out[m] += frames[t * n_mels + m];
        }
    }
    let inv = 1.0 / n as f32;
    for v in &mut out {
        *v *= inv;
    }
    out
}

pub fn clips_to_mel_features(clips: &[LabeledClip], n_mels: usize) -> Vec<(Vec<f32>, f32)> {
    clips
        .iter()
        .map(|c| (mel_mean_feature(&c.pcm, n_mels), c.label))
        .collect()
}
