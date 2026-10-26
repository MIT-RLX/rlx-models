// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0

//! pyannote's `XVectorSincNet`: 16 kHz audio → the 512-d x-vector the LM's
//! `spk_proj` was trained on (`speaker/model.safetensors`, MIT).
//!
//! ```text
//! InstanceNorm(wav) ─ SincConv(80, k251, s10) ─ |·| ─ MaxPool3 ─ IN ─ LReLU
//!   ─ Conv(80→60, k5) ─ MaxPool3 ─ IN ─ LReLU ─ Conv(60→60, k5) ─ MaxPool3 ─ IN ─ LReLU
//!   ─ 5 × [Conv(k, dilation) ─ LReLU ─ BatchNorm] ─ mean ‖ std ─ Linear(3000→512)
//! ```
//!
//! The sinc filters are rebuilt from their learned cutoffs on the host
//! (`ParamSincFB`: cosine/sine pairs, Hamming half-window); everything else is
//! one graph over the clip.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, ensure};
use rlx_ir::hir::HirNodeId;
use rlx_ir::op::{Op, ReduceOp};
use rlx_ir::{HirGraphExt, Shape};
use safetensors::{Dtype, SafeTensors};

use crate::s3gen::nn::{Ctx, F32, S3GenWeights, dims};

pub const SAMPLE_RATE: usize = 16_000;
pub const DIM: usize = 512;
const SINC_FILTERS: usize = 80;
const SINC_KERNEL: usize = 251;
const SINC_STRIDE: usize = 10;
const MIN_LOW_HZ: f64 = 50.0;
const MIN_BAND_HZ: f64 = 50.0;
/// `(conv tensor index, dilation)` of the five TDNN layers; kernel sizes
/// (5, 3, 3, 1, 1) come from the weights, the BatchNorm is two entries on.
const TDNN: [(usize, usize); 5] = [(0, 1), (3, 2), (6, 3), (9, 1), (12, 1)];

/// Read `speaker/model.safetensors` into a weight store (f32).
pub fn open_weights(path: &Path) -> Result<S3GenWeights> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let st = SafeTensors::deserialize(&bytes).context("parse speaker safetensors")?;
    let mut t = HashMap::new();
    for name in st.names() {
        let v = st.tensor(name)?;
        if v.dtype() != Dtype::F32 {
            continue;
        }
        let data = v
            .data()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        t.insert(name.to_string(), (data, v.shape().to_vec()));
    }
    Ok(S3GenWeights::from_tensors(t))
}

/// `ParamSincFilterbank.filters()`: `[80, 251]` (40 cosine then 40 sine).
pub fn sinc_filters(w: &S3GenWeights) -> Result<Vec<f32>> {
    let low_ = w.get("sincnet.conv1d.0.filterbank.low_hz_")?;
    let band_ = w.get("sincnet.conv1d.0.filterbank.band_hz_")?;
    let half = SINC_KERNEL / 2;
    let n_pairs = SINC_FILTERS / 2;
    ensure!(
        low_.len() == n_pairs && band_.len() == n_pairs,
        "sinc cutoffs"
    );
    // np.hamming(251)[:125], n_ = 2π · (−125 … −1) / sr.
    let window: Vec<f64> = (0..half)
        .map(|i| 0.54 - 0.46 * (std::f64::consts::TAU * i as f64 / (SINC_KERNEL - 1) as f64).cos())
        .collect();
    let n: Vec<f64> = (0..half)
        .map(|i| std::f64::consts::TAU * (i as f64 - half as f64) / SAMPLE_RATE as f64)
        .collect();
    let mut out = vec![0f32; SINC_FILTERS * SINC_KERNEL];
    for (kind, base) in [(0, 0), (1, n_pairs)] {
        for f in 0..n_pairs {
            let low = MIN_LOW_HZ + low_[f].abs() as f64;
            let high = (low + MIN_BAND_HZ + band_[f].abs() as f64)
                .clamp(MIN_LOW_HZ, SAMPLE_RATE as f64 / 2.0);
            let band = high - low;
            let row = &mut out[(base + f) * SINC_KERNEL..(base + f + 1) * SINC_KERNEL];
            for i in 0..half {
                let (fl, fh) = (low * n[i], high * n[i]);
                let left = if kind == 0 {
                    (fh.sin() - fl.sin()) / (n[i] / 2.0) * window[i]
                } else {
                    (fl.cos() - fh.cos()) / (n[i] / 2.0) * window[i]
                };
                row[i] = (left / (2.0 * band)) as f32;
                row[SINC_KERNEL - 1 - i] =
                    (if kind == 0 { left } else { -left } / (2.0 * band)) as f32;
            }
            // Center tap: `2·band` for the cosine filter, 0 for the sine
            // one, before the shared `/ (2·band)`.
            row[half] = if kind == 0 { 1.0 } else { 0.0 };
        }
    }
    Ok(out)
}

/// `InstanceNorm1d(affine)` over `[1, C, T, 1]` (biased variance, eps 1e-5).
fn instance_norm(ctx: &mut Ctx, w: &S3GenWeights, x: HirNodeId, key: &str) -> Result<HirNodeId> {
    let d = dims(ctx, x);
    let (c, t) = (d[1], d[2]);
    let x2 = ctx.g.reshape_(x, vec![c as i64, t as i64]);
    let ones = ctx.param(&format!("ones:{t}"), &[t], || vec![1.0; t]);
    let zeros = ctx.param(&format!("zeros:{t}"), &[t], || vec![0.0; t]);
    let n = ctx.g.ln(x2, ones, zeros, 1e-5);
    let gk = format!("{key}.weight");
    let bk = format!("{key}.bias");
    let g = ctx.param(&gk, &[c, 1], || w.get(&gk).unwrap().to_vec());
    let b = ctx.param(&bk, &[c, 1], || w.get(&bk).unwrap().to_vec());
    let g = ctx.g.expand_(g, vec![c as i64, t as i64]);
    let b = ctx.g.expand_(b, vec![c as i64, t as i64]);
    let y = ctx.g.mul(n, g);
    let y = ctx.g.add(y, b);
    Ok(ctx.g.reshape_(y, vec![1, c as i64, t as i64, 1]))
}

/// `MaxPool1d(3, stride=3)` (floor) on `[1, C, T, 1]`.
fn max_pool3(ctx: &mut Ctx, x: HirNodeId) -> HirNodeId {
    let d = dims(ctx, x);
    let (c, t) = (d[1], d[2]);
    let to = t / 3;
    let x = ctx.g.narrow_(x, 2, 0, to * 3);
    let r = ctx.g.reshape_(x, vec![c as i64, to as i64, 3]);
    let m = ctx.g.add_node(
        Op::Reduce {
            op: ReduceOp::Max,
            axes: vec![2],
            keep_dim: false,
        },
        vec![r],
        Shape::new(&[c, to], F32),
    );
    ctx.g.reshape_(m, vec![1, c as i64, to as i64, 1])
}

/// Eval-mode `BatchNorm1d` folded to a per-channel affine.
fn batch_norm(ctx: &mut Ctx, w: &S3GenWeights, x: HirNodeId, key: &str) -> Result<HirNodeId> {
    let d = dims(ctx, x);
    let (c, t) = (d[1], d[2]);
    let mean = w.get(&format!("{key}.running_mean"))?;
    let var = w.get(&format!("{key}.running_var"))?;
    let gamma = w.get(&format!("{key}.weight"))?;
    let beta = w.get(&format!("{key}.bias"))?;
    let scale: Vec<f32> = (0..c).map(|i| gamma[i] / (var[i] + 1e-5).sqrt()).collect();
    let shift: Vec<f32> = (0..c).map(|i| beta[i] - mean[i] * scale[i]).collect();
    let s = ctx.param(&format!("{key}^scale"), &[1, c, 1, 1], || scale);
    let b = ctx.param(&format!("{key}^shift"), &[1, c, 1, 1], || shift);
    let full = vec![1, c as i64, t as i64, 1];
    let s = ctx.g.expand_(s, full.clone());
    let b = ctx.g.expand_(b, full);
    let y = ctx.g.mul(x, s);
    Ok(ctx.g.add(y, b))
}

/// Emit the whole model over `wav` (`[1, 1, N, 1]`); returns `[1, 512]`.
pub fn build(ctx: &mut Ctx, w: &S3GenWeights, wav: HirNodeId) -> Result<HirNodeId> {
    let x = instance_norm(ctx, w, wav, "sincnet.wav_norm1d")?;
    let filters = sinc_filters(w)?;
    let fid = ctx.param("sinc_filters", &[SINC_FILTERS, 1, SINC_KERNEL, 1], || {
        filters
    });
    let n = dims(ctx, x)[2];
    ensure!(
        n >= SINC_KERNEL,
        "reference too short for the speaker encoder"
    );
    let t0 = (n - SINC_KERNEL) / SINC_STRIDE + 1;
    let y = ctx.g.add_node(
        Op::Conv {
            kernel_size: vec![SINC_KERNEL, 1],
            stride: vec![SINC_STRIDE, 1],
            padding: vec![0, 0],
            dilation: vec![1, 1],
            groups: 1,
        },
        vec![x, fid],
        Shape::new(&[1, SINC_FILTERS, t0, 1], F32),
    );
    let y = ctx.act(rlx_ir::op::Activation::Abs, y);
    let mut h = y;
    for i in 0..3 {
        if i > 0 {
            h = ctx.conv1d(w, h, &format!("sincnet.conv1d.{i}"), (0, 0), 1, 1)?;
        }
        h = max_pool3(ctx, h);
        h = instance_norm(ctx, w, h, &format!("sincnet.norm1d.{i}"))?;
        h = ctx.leaky_relu(h, 0.01);
    }
    for (idx, dil) in TDNN {
        h = ctx.conv1d(w, h, &format!("tdnns.{idx}"), (0, 0), 1, dil)?;
        h = ctx.leaky_relu(h, 0.01);
        h = batch_norm(ctx, w, h, &format!("tdnns.{}", idx + 2))?;
    }
    // Statistics pooling: mean and unbiased std over time.
    let d = dims(ctx, h);
    let (c, t) = (d[1], d[2]);
    ensure!(t > 1, "reference too short for statistics pooling");
    let h2 = ctx.g.reshape_(h, vec![c as i64, t as i64]);
    let mean = ctx.g.mean(h2, vec![1], true);
    let me = ctx.g.expand_(mean, vec![c as i64, t as i64]);
    let dev = ctx.g.sub(h2, me);
    let sq = ctx.g.mul(dev, dev);
    let ss = ctx.g.sum(sq, vec![1], true);
    let inv = ctx.scalar(1.0 / (t - 1) as f32, &[c, 1]);
    let var = ctx.g.mul(ss, inv);
    let std = ctx.g.sqrt(var);
    let pooled = ctx.g.concat_(vec![mean, std], 0);
    let pooled = ctx.g.reshape_(pooled, vec![1, (2 * c) as i64]);
    ctx.linear(w, pooled, "embedding")
}
