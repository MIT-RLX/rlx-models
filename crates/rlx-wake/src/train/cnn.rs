// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Train `WakeCnn` end-to-end in RLX (conv + FC, SGD on `rlx-cpu` path).

use crate::cnn::{WakeCnnConfig, WakeCnnWeights};
use crate::ops::{conv1d_nchw, gemv_bias, global_mean_pool_chw, relu, sigmoid};
use crate::train::dataset::{LabeledClip, clip_mel_frames};
use crate::train::report::TrainReport;
use crate::train::sgd::{SgdConfig, bce_dlogit, bce_loss, sgd_step};

#[derive(Debug, Clone)]
pub struct CnnTrainConfig {
    pub sgd: SgdConfig,
    /// Keep last N mel frames as CNN input window.
    pub window_frames: usize,
    pub keyword: String,
}

impl Default for CnnTrainConfig {
    fn default() -> Self {
        Self {
            sgd: SgdConfig::default(),
            window_frames: 40,
            keyword: "wake".into(),
        }
    }
}

fn window_mel(frames: &[f32], n_mels: usize, window_frames: usize) -> Vec<f32> {
    if frames.is_empty() || !frames.len().is_multiple_of(n_mels) {
        return vec![0.0; n_mels * window_frames.clamp(1, 8)];
    }
    let t = frames.len() / n_mels;
    let take = t.min(window_frames).max(1);
    let start = t - take;
    frames[start * n_mels..].to_vec()
}

struct FwdCache {
    x: Vec<f32>, // [n_mels, t]
    y1: Vec<f32>,
    t1: usize,
    y2: Vec<f32>,
    t2: usize,
    y3: Vec<f32>,
    t3: usize,
    pooled: Vec<f32>,
    h_pre: Vec<f32>,
    h: Vec<f32>,
    prob: f32,
}

fn forward(w: &WakeCnnWeights, mel_flat: &[f32]) -> FwdCache {
    let cfg = &w.cfg;
    let n_mels = cfg.n_mels;
    let t = mel_flat.len() / n_mels;
    let mut x = vec![0.0f32; n_mels * t];
    for ti in 0..t {
        for m in 0..n_mels {
            x[m * t + ti] = mel_flat[ti * n_mels + m];
        }
    }
    let mut y1 = vec![0.0f32; cfg.c1 * t];
    let t1 = conv1d_nchw(
        &x,
        n_mels,
        t,
        &w.conv1_w,
        cfg.c1,
        cfg.k,
        1,
        cfg.k / 2,
        Some(&w.conv1_b),
        &mut y1,
    );
    for v in &mut y1[..cfg.c1 * t1] {
        *v = relu(*v);
    }
    let mut y2 = vec![0.0f32; cfg.c2 * t1];
    let t2 = conv1d_nchw(
        &y1[..cfg.c1 * t1],
        cfg.c1,
        t1,
        &w.conv2_w,
        cfg.c2,
        cfg.k,
        2,
        cfg.k / 2,
        Some(&w.conv2_b),
        &mut y2,
    );
    for v in &mut y2[..cfg.c2 * t2] {
        *v = relu(*v);
    }
    let mut y3 = vec![0.0f32; cfg.c3 * t2];
    let t3 = conv1d_nchw(
        &y2[..cfg.c2 * t2],
        cfg.c2,
        t2,
        &w.conv3_w,
        cfg.c3,
        cfg.k,
        2,
        cfg.k / 2,
        Some(&w.conv3_b),
        &mut y3,
    );
    for v in &mut y3[..cfg.c3 * t3] {
        *v = relu(*v);
    }
    let mut pooled = vec![0.0f32; cfg.c3];
    global_mean_pool_chw(&y3[..cfg.c3 * t3], cfg.c3, t3, &mut pooled);
    let mut h_pre = vec![0.0f32; cfg.hidden];
    gemv_bias(cfg.hidden, cfg.c3, &w.fc1_w, &pooled, &w.fc1_b, &mut h_pre);
    let h: Vec<f32> = h_pre.iter().copied().map(relu).collect();
    let mut logit = [0.0f32];
    gemv_bias(1, cfg.hidden, &w.fc2_w, &h, &w.fc2_b, &mut logit);
    let prob = sigmoid(logit[0]);
    FwdCache {
        x,
        y1,
        t1,
        y2,
        t2,
        y3,
        t3,
        pooled,
        h_pre,
        h,
        prob,
    }
}

fn conv1d_backward(
    x: &[f32],
    in_ch: usize,
    t_in: usize,
    w: &[f32],
    out_ch: usize,
    k: usize,
    stride: usize,
    pad: usize,
    dy: &[f32],
    t_out: usize,
    dw: &mut [f32],
    db: &mut [f32],
    dx: &mut [f32],
) {
    dw.fill(0.0);
    db.fill(0.0);
    dx.fill(0.0);
    for oc in 0..out_ch {
        for ot in 0..t_out {
            let g = dy[oc * t_out + ot];
            db[oc] += g;
            for ic in 0..in_ch {
                for ki in 0..k {
                    let ti = ot * stride + ki;
                    let ti = ti as isize - pad as isize;
                    if ti < 0 || ti >= t_in as isize {
                        continue;
                    }
                    let x_idx = ic * t_in + ti as usize;
                    let w_idx = oc * (in_ch * k) + ic * k + ki;
                    dw[w_idx] += g * x[x_idx];
                    dx[x_idx] += g * w[w_idx];
                }
            }
        }
    }
}

fn train_step(w: &mut WakeCnnWeights, mel_flat: &[f32], label: f32, lr: f32, wd: f32) -> f32 {
    let cfg = w.cfg.clone();
    let cache = forward(w, mel_flat);
    let loss = bce_loss(cache.prob, label);
    let dlogit = bce_dlogit(cache.prob, label);

    // FC2
    let mut dfc2_w = vec![0.0f32; cfg.hidden];
    for i in 0..cfg.hidden {
        dfc2_w[i] = dlogit * cache.h[i];
    }
    let dfc2_b = [dlogit];

    // dh / relu
    let mut dh = vec![0.0f32; cfg.hidden];
    for i in 0..cfg.hidden {
        dh[i] = dlogit * w.fc2_w[i];
        if cache.h_pre[i] <= 0.0 {
            dh[i] = 0.0;
        }
    }

    // FC1
    let mut dfc1_w = vec![0.0f32; cfg.hidden * cfg.c3];
    let mut dfc1_b = vec![0.0f32; cfg.hidden];
    let mut dpooled = vec![0.0f32; cfg.c3];
    for o in 0..cfg.hidden {
        dfc1_b[o] = dh[o];
        for i in 0..cfg.c3 {
            dfc1_w[o * cfg.c3 + i] = dh[o] * cache.pooled[i];
            dpooled[i] += dh[o] * w.fc1_w[o * cfg.c3 + i];
        }
    }

    // unpool mean → dy3
    let inv = if cache.t3 == 0 {
        0.0
    } else {
        1.0 / cache.t3 as f32
    };
    let mut dy3 = vec![0.0f32; cfg.c3 * cache.t3];
    for c in 0..cfg.c3 {
        for t in 0..cache.t3 {
            // relu gate on y3 (post-relu stored)
            let pre_ok = cache.y3[c * cache.t3 + t] > 0.0;
            dy3[c * cache.t3 + t] = if pre_ok { dpooled[c] * inv } else { 0.0 };
        }
    }

    let mut dconv3_w = vec![0.0f32; w.conv3_w.len()];
    let mut dconv3_b = vec![0.0f32; cfg.c3];
    let mut dy2 = vec![0.0f32; cfg.c2 * cache.t2];
    conv1d_backward(
        &cache.y2[..cfg.c2 * cache.t2],
        cfg.c2,
        cache.t2,
        &w.conv3_w,
        cfg.c3,
        cfg.k,
        2,
        cfg.k / 2,
        &dy3,
        cache.t3,
        &mut dconv3_w,
        &mut dconv3_b,
        &mut dy2,
    );
    for i in 0..cfg.c2 * cache.t2 {
        if cache.y2[i] <= 0.0 {
            dy2[i] = 0.0;
        }
    }

    let mut dconv2_w = vec![0.0f32; w.conv2_w.len()];
    let mut dconv2_b = vec![0.0f32; cfg.c2];
    let mut dy1 = vec![0.0f32; cfg.c1 * cache.t1];
    conv1d_backward(
        &cache.y1[..cfg.c1 * cache.t1],
        cfg.c1,
        cache.t1,
        &w.conv2_w,
        cfg.c2,
        cfg.k,
        2,
        cfg.k / 2,
        &dy2[..cfg.c2 * cache.t2],
        cache.t2,
        &mut dconv2_w,
        &mut dconv2_b,
        &mut dy1,
    );
    for i in 0..cfg.c1 * cache.t1 {
        if cache.y1[i] <= 0.0 {
            dy1[i] = 0.0;
        }
    }

    let mut dconv1_w = vec![0.0f32; w.conv1_w.len()];
    let mut dconv1_b = vec![0.0f32; cfg.c1];
    let t_in = cache.x.len() / cfg.n_mels.max(1);
    let mut dx = vec![0.0f32; cfg.n_mels * t_in];
    conv1d_backward(
        &cache.x,
        cfg.n_mels,
        t_in,
        &w.conv1_w,
        cfg.c1,
        cfg.k,
        1,
        cfg.k / 2,
        &dy1[..cfg.c1 * cache.t1],
        cache.t1,
        &mut dconv1_w,
        &mut dconv1_b,
        &mut dx,
    );

    sgd_step(&mut w.fc2_w, &dfc2_w, lr, wd);
    sgd_step(&mut w.fc2_b, &dfc2_b, lr, wd);
    sgd_step(&mut w.fc1_w, &dfc1_w, lr, wd);
    sgd_step(&mut w.fc1_b, &dfc1_b, lr, wd);
    sgd_step(&mut w.conv3_w, &dconv3_w, lr, wd);
    sgd_step(&mut w.conv3_b, &dconv3_b, lr, wd);
    sgd_step(&mut w.conv2_w, &dconv2_w, lr, wd);
    sgd_step(&mut w.conv2_b, &dconv2_b, lr, wd);
    sgd_step(&mut w.conv1_w, &dconv1_w, lr, wd);
    sgd_step(&mut w.conv1_b, &dconv1_b, lr, wd);
    loss
}

pub fn train_wake_cnn(
    weights: &mut WakeCnnWeights,
    clips: &[LabeledClip],
    cfg: &CnnTrainConfig,
) -> TrainReport {
    let n_mels = weights.cfg.n_mels;
    let feats: Vec<(Vec<f32>, f32)> = clips
        .iter()
        .map(|c| {
            let mel = clip_mel_frames(&c.pcm);
            let win = window_mel(&mel, n_mels, cfg.window_frames);
            (win, c.label)
        })
        .collect();

    let mut initial = 0.0f32;
    let mut final_loss = 0.0f32;
    for epoch in 0..cfg.sgd.epochs {
        let mut sum = 0.0f32;
        for (mel, y) in &feats {
            sum += train_step(weights, mel, *y, cfg.sgd.lr, cfg.sgd.weight_decay);
        }
        let mean = sum / feats.len().max(1) as f32;
        if epoch == 0 {
            initial = mean;
        }
        final_loss = mean;
        if cfg.sgd.log_every > 0 && epoch % cfg.sgd.log_every == 0 {
            eprintln!(
                "[rlx-wake-train cnn] epoch={epoch} loss={mean:.4} keyword={}",
                cfg.keyword
            );
        }
    }

    let mut correct = 0usize;
    for (mel, y) in &feats {
        let p = forward(weights, mel).prob;
        let pred = if p >= 0.5 { 1.0 } else { 0.0 };
        if (pred - *y).abs() < 0.5 {
            correct += 1;
        }
    }
    TrainReport {
        epochs: cfg.sgd.epochs,
        final_loss,
        initial_loss: initial,
        train_acc: correct as f32 / feats.len().max(1) as f32,
        keyword: cfg.keyword.clone(),
    }
}

/// Convenience: fresh lite CNN + train.
pub fn train_new_lite_cnn(
    clips: &[LabeledClip],
    keyword: &str,
    epochs: usize,
) -> (WakeCnnWeights, TrainReport) {
    let mut w = WakeCnnWeights::stub(WakeCnnConfig::lite());
    let cfg = CnnTrainConfig {
        keyword: keyword.into(),
        sgd: SgdConfig {
            epochs,
            lr: 1e-2,
            ..SgdConfig::default()
        },
        ..CnnTrainConfig::default()
    };
    let report = train_wake_cnn(&mut w, clips, &cfg);
    (w, report)
}
