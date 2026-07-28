// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! Native (ort-free) VITS StochasticDurationPredictor for piper.
//!
//! The dp maps the text-encoder hidden `dp_in [192, T]` to per-phoneme integer
//! durations by running a stack of normalizing flows in reverse. Its
//! rational-quadratic-spline coupling flows use boolean-mask indexing
//! (`inputs[inside_interval_mask]`) — a data-dependent `NonZero`/`GatherND`
//! flatten that no static-shape ONNX importer can rank (it keeps the tensor at
//! rank-4 where onnxruntime flattens to rank-2). Rather than force the importer
//! through that, the dp is reimplemented here directly: in scalar Rust the
//! spline's inside/outside split is a plain per-element branch, so the whole
//! blocker disappears.
//!
//! Validated against onnxruntime (`scales=[0,1,0]`, deterministic): the
//! conditioning path is bit-exact (cosine 1.0) and the predicted durations match
//! to ±1 at rare spline bin-boundary phonemes (inherent flow sensitivity), which
//! is perceptually identical. See `scripts/split_piper.py` /
//! `tests/sdp_parity.rs`.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use serde::Deserialize;

const NUM_BINS: usize = 10;
const TAIL: f32 = 5.0;
const FILTER: usize = 192;
const EPS_LN: f32 = 1e-5;
const MIN_BW: f32 = 1e-3;
const MIN_BH: f32 = 1e-3;
const MIN_DER: f32 = 1e-3;
const DILATIONS: [usize; 3] = [1, 3, 9];

#[derive(Deserialize)]
struct WeightMeta {
    offset: usize,
    numel: usize,
}

#[derive(Deserialize)]
struct Manifest {
    weights: HashMap<String, WeightMeta>,
}

/// The dp weight bank, loaded from `dp_weights.f32` + `dp_manifest.json`.
/// Each entry is a flat row-major f32 tensor (shape is implied by how the runner
/// indexes it, so only the data is retained).
pub struct Sdp {
    w: HashMap<String, Vec<f32>>,
}

impl Sdp {
    /// Load the dp weights from a native split bundle directory.
    pub fn load(split_dir: &Path) -> Result<Self> {
        let manifest: Manifest = serde_json::from_slice(
            &std::fs::read(split_dir.join("dp_manifest.json")).context("read dp_manifest.json")?,
        )
        .context("parse dp_manifest.json")?;
        let blob =
            std::fs::read(split_dir.join("dp_weights.f32")).context("read dp_weights.f32")?;
        let flat: Vec<f32> = blob
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        let mut w = HashMap::new();
        for (name, meta) in manifest.weights {
            let end = meta.offset + meta.numel;
            anyhow::ensure!(end <= flat.len(), "dp weight {name} out of range");
            w.insert(name, flat[meta.offset..end].to_vec());
        }
        Ok(Self { w })
    }

    fn t(&self, name: &str) -> &[f32] {
        self.w
            .get(name)
            .unwrap_or_else(|| panic!("dp weight missing: {name}"))
    }

    /// Predict integer durations from the encoder conditioning `dp_in [192, T]`
    /// (row-major). `length_scale` > 1 slows speech; `noise` (len 2·T, row-major
    /// `[2, T]`) is the flow noise (all-zeros = deterministic mean durations).
    pub fn durations(
        &self,
        dp_in: &[f32],
        t: usize,
        length_scale: f32,
        noise: &[f32],
    ) -> Vec<usize> {
        assert_eq!(dp_in.len(), FILTER * t, "dp_in must be [192, T]");
        // Conditioning g = proj(convs(pre(dp_in))) — the flows' DDSConv condition.
        let mut x = conv1d(
            dp_in,
            FILTER,
            t,
            self.t("dp.pre.weight"),
            self.t("dp.pre.bias"),
            1,
        );
        x = self.ddsconv(&x, t, "dp.convs", None);
        let g = conv1d(
            &x,
            FILTER,
            t,
            self.t("dp.proj.weight"),
            self.t("dp.proj.bias"),
            1,
        );

        // Reverse flow order: Flip, CF7, Flip, CF5, Flip, CF3, Flip, EA0.
        let mut z = noise.to_vec(); // [2, T]
        flip2(&mut z, t);
        z = self.convflow_reverse(&z, t, 7, &g);
        flip2(&mut z, t);
        z = self.convflow_reverse(&z, t, 5, &g);
        flip2(&mut z, t);
        z = self.convflow_reverse(&z, t, 3, &g);
        flip2(&mut z, t);
        self.ea0_reverse(&mut z, t);

        // logw = z channel 0 → w = exp(logw)·length_scale → ceil.
        (0..t)
            .map(|i| (z[i].exp() * length_scale).ceil().max(0.0) as usize)
            .collect()
    }

    /// DDSConv: 3 × [depthwise(dil) → LN → GELU → pointwise → LN → GELU] residual.
    /// `g` (conditioning) is added once at the start.
    fn ddsconv(&self, x_in: &[f32], t: usize, prefix: &str, g: Option<&[f32]>) -> Vec<f32> {
        let mut x = x_in.to_vec();
        if let Some(g) = g {
            for (a, b) in x.iter_mut().zip(g) {
                *a += *b;
            }
        }
        for i in 0..3 {
            let sep_w = self.t(&format!("{prefix}.convs_sep.{i}.weight"));
            let sep_b = self.t(&format!("{prefix}.convs_sep.{i}.bias"));
            let mut y = conv1d_depthwise(&x, FILTER, t, sep_w, sep_b, DILATIONS[i]);
            layer_norm_ch(
                &mut y,
                FILTER,
                t,
                self.t(&format!("{prefix}.norms_1.{i}.gamma")),
                self.t(&format!("{prefix}.norms_1.{i}.beta")),
            );
            gelu_(&mut y);
            let p1w = self.t(&format!("{prefix}.convs_1x1.{i}.weight"));
            let p1b = self.t(&format!("{prefix}.convs_1x1.{i}.bias"));
            let mut y2 = conv1d(&y, FILTER, t, p1w, p1b, 1);
            layer_norm_ch(
                &mut y2,
                FILTER,
                t,
                self.t(&format!("{prefix}.norms_2.{i}.gamma")),
                self.t(&format!("{prefix}.norms_2.{i}.beta")),
            );
            gelu_(&mut y2);
            for (a, b) in x.iter_mut().zip(&y2) {
                *a += *b;
            }
        }
        x
    }

    /// ConvFlow reverse: split → spline params from conditioned convs → inverse
    /// rational-quadratic spline on the second channel.
    fn convflow_reverse(&self, z: &[f32], t: usize, idx: usize, g: &[f32]) -> Vec<f32> {
        let pfx = format!("dp.flows.{idx}");
        let x0 = &z[0..t]; // [1, T]
        let x1 = &z[t..2 * t]; // [T]
        // pre: 1 → 192
        let h = conv1d(
            x0,
            1,
            t,
            self.t(&format!("{pfx}.pre.weight")),
            self.t(&format!("{pfx}.pre.bias")),
            1,
        );
        let h = self.ddsconv(&h, t, &format!("{pfx}.convs"), Some(g));
        // proj: 192 → 29
        let proj = conv1d(
            &h,
            FILTER,
            t,
            self.t(&format!("{pfx}.proj.weight")),
            self.t(&format!("{pfx}.proj.bias")),
            1,
        );
        // proj is [29, T]; reshape to per-position [T, 29] and split into bins.
        let inv_sqrt = 1.0 / (FILTER as f32).sqrt();
        let mut x1_out = vec![0.0f32; t];
        for j in 0..t {
            let mut uw = [0.0f32; NUM_BINS];
            let mut uh = [0.0f32; NUM_BINS];
            let mut ud = [0.0f32; NUM_BINS - 1]; // 29 - 20 = 9
            for k in 0..NUM_BINS {
                uw[k] = proj[k * t + j] * inv_sqrt;
                uh[k] = proj[(NUM_BINS + k) * t + j] * inv_sqrt;
            }
            for k in 0..NUM_BINS - 1 {
                ud[k] = proj[(2 * NUM_BINS + k) * t + j];
            }
            x1_out[j] = rqs_inverse_scalar(x1[j], &uw, &uh, &ud);
        }
        // cat([x0, x1_out], channel) → [2, T]
        let mut out = vec![0.0f32; 2 * t];
        out[0..t].copy_from_slice(x0);
        out[t..2 * t].copy_from_slice(&x1_out);
        out
    }

    /// ElementwiseAffine reverse: `(z - m)·exp(-logs)` per channel (logs = 0 when
    /// absent → identity scale).
    fn ea0_reverse(&self, z: &mut [f32], t: usize) {
        let m = self.t("dp.flows.0.m"); // [2,1]
        let logs = self.w.get("dp.flows.0.logs").cloned();
        for c in 0..2 {
            let mc = m[c];
            let sc = logs.as_ref().map(|l| (-l[c]).exp()).unwrap_or(1.0);
            for j in 0..t {
                z[c * t + j] = (z[c * t + j] - mc) * sc;
            }
        }
    }
}

/// Standard (grouped=1) 1-D conv, `same` padding. `x [c_in, t]`,
/// `w [c_out, c_in, k]`, `b [c_out]` → `[c_out, t]`.
fn conv1d(x: &[f32], c_in: usize, t: usize, w: &[f32], b: &[f32], dilation: usize) -> Vec<f32> {
    let k = w.len() / (b.len() * c_in);
    let c_out = b.len();
    let pad = (k - 1) * dilation / 2;
    let mut out = vec![0.0f32; c_out * t];
    for oc in 0..c_out {
        let dst = &mut out[oc * t..oc * t + t];
        for (j, o) in dst.iter_mut().enumerate() {
            let mut acc = b[oc];
            for ic in 0..c_in {
                let wrow = &w[(oc * c_in + ic) * k..(oc * c_in + ic) * k + k];
                let xrow = &x[ic * t..ic * t + t];
                for kk in 0..k {
                    let pos = j as isize + (kk * dilation) as isize - pad as isize;
                    if pos >= 0 && (pos as usize) < t {
                        acc += wrow[kk] * xrow[pos as usize];
                    }
                }
            }
            *o = acc;
        }
    }
    out
}

/// Depthwise 1-D conv (groups == channels), `same` padding. `w [c, 1, k]`.
fn conv1d_depthwise(
    x: &[f32],
    c: usize,
    t: usize,
    w: &[f32],
    b: &[f32],
    dilation: usize,
) -> Vec<f32> {
    let k = w.len() / c;
    let pad = (k - 1) * dilation / 2;
    let mut out = vec![0.0f32; c * t];
    for ch in 0..c {
        let wrow = &w[ch * k..ch * k + k];
        let xrow = &x[ch * t..ch * t + t];
        let dst = &mut out[ch * t..ch * t + t];
        for (j, o) in dst.iter_mut().enumerate() {
            let mut acc = b[ch];
            for kk in 0..k {
                let pos = j as isize + (kk * dilation) as isize - pad as isize;
                if pos >= 0 && (pos as usize) < t {
                    acc += wrow[kk] * xrow[pos as usize];
                }
            }
            *o = acc;
        }
    }
    out
}

/// VITS LayerNorm over the channel axis (per time-step): normalize each column.
fn layer_norm_ch(x: &mut [f32], c: usize, t: usize, gamma: &[f32], beta: &[f32]) {
    for j in 0..t {
        let mut mean = 0.0f32;
        for ch in 0..c {
            mean += x[ch * t + j];
        }
        mean /= c as f32;
        let mut var = 0.0f32;
        for ch in 0..c {
            let d = x[ch * t + j] - mean;
            var += d * d;
        }
        var /= c as f32;
        let inv = 1.0 / (var + EPS_LN).sqrt();
        for ch in 0..c {
            x[ch * t + j] = (x[ch * t + j] - mean) * inv * gamma[ch] + beta[ch];
        }
    }
}

/// erf via Abramowitz-Stegun 7.1.26 (max abs err ~1.5e-7), odd-extended.
fn erf(x: f32) -> f32 {
    let t = 1.0 / (1.0 + 0.3275911 * x.abs());
    let y = 1.0
        - (((((1.061_405_4 * t - 1.453_152_1) * t) + 1.421_413_8) * t - 0.284_496_72) * t
            + 0.254_829_6)
            * t
            * (-x * x).exp();
    y.copysign(x)
}

fn gelu_(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = 0.5 * *v * (1.0 + erf(*v * std::f32::consts::FRAC_1_SQRT_2));
    }
}

/// Reverse the two channels of a `[2, T]` tensor in place.
fn flip2(z: &mut [f32], t: usize) {
    for j in 0..t {
        z.swap(j, t + j);
    }
}

/// Inverse unconstrained rational-quadratic spline for a single scalar input,
/// linear tails, bounds `[-TAIL, TAIL]`. Mirrors neural-spline-flows
/// (Durkan et al.) as used by VITS. Elements outside the interval pass through.
fn rqs_inverse_scalar(
    input: f32,
    uw: &[f32; NUM_BINS],
    uh: &[f32; NUM_BINS],
    ud: &[f32; NUM_BINS - 1],
) -> f32 {
    if !(input >= -TAIL && input <= TAIL) {
        return input; // linear tail = identity
    }
    // widths → cumwidths (padded, scaled to [-TAIL, TAIL]).
    let widths_sm = softmax(uw);
    let mut widths = [0.0f32; NUM_BINS];
    for k in 0..NUM_BINS {
        widths[k] = MIN_BW + (1.0 - MIN_BW * NUM_BINS as f32) * widths_sm[k];
    }
    let mut cumwidths = [0.0f32; NUM_BINS + 1];
    let mut acc = 0.0f32;
    for k in 0..NUM_BINS {
        acc += widths[k];
        cumwidths[k + 1] = acc;
    }
    for c in cumwidths.iter_mut() {
        *c = 2.0 * TAIL * *c - TAIL;
    }
    cumwidths[0] = -TAIL;
    cumwidths[NUM_BINS] = TAIL;
    for k in 0..NUM_BINS {
        widths[k] = cumwidths[k + 1] - cumwidths[k];
    }

    // heights → cumheights.
    let heights_sm = softmax(uh);
    let mut heights = [0.0f32; NUM_BINS];
    for k in 0..NUM_BINS {
        heights[k] = MIN_BH + (1.0 - MIN_BH * NUM_BINS as f32) * heights_sm[k];
    }
    let mut cumheights = [0.0f32; NUM_BINS + 1];
    let mut acc = 0.0f32;
    for k in 0..NUM_BINS {
        acc += heights[k];
        cumheights[k + 1] = acc;
    }
    for c in cumheights.iter_mut() {
        *c = 2.0 * TAIL * *c - TAIL;
    }
    cumheights[0] = -TAIL;
    cumheights[NUM_BINS] = TAIL;
    for k in 0..NUM_BINS {
        heights[k] = cumheights[k + 1] - cumheights[k];
    }

    // derivatives = min_derivative + softplus(pad(ud, (1,1)) with endpoints=const).
    let constant = ((1.0f32 - MIN_DER).exp() - 1.0).ln();
    let mut derivatives = [0.0f32; NUM_BINS + 1];
    for k in 0..=NUM_BINS {
        let raw = if k == 0 || k == NUM_BINS {
            constant
        } else {
            ud[k - 1]
        };
        derivatives[k] = MIN_DER + softplus(raw);
    }

    // searchsorted over cumheights (inverse): last boundary nudged by eps.
    let mut bl = cumheights;
    bl[NUM_BINS] += 1e-6;
    let mut count = 0usize;
    for &b in bl.iter() {
        if input >= b {
            count += 1;
        }
    }
    let bin = count.saturating_sub(1).min(NUM_BINS - 1);

    let input_cumwidths = cumwidths[bin];
    let input_bin_widths = widths[bin];
    let input_cumheights = cumheights[bin];
    let input_heights = heights[bin];
    let input_delta = heights[bin] / widths[bin];
    let input_derivatives = derivatives[bin];
    let input_derivatives_plus_one = derivatives[bin + 1];

    let dy = input - input_cumheights;
    let common = input_derivatives + input_derivatives_plus_one - 2.0 * input_delta;
    let a = dy * common + input_heights * (input_delta - input_derivatives);
    let b = input_heights * input_derivatives - dy * common;
    let c = -input_delta * dy;
    let disc = b * b - 4.0 * a * c;
    let root = (2.0 * c) / (-b - disc.sqrt());
    root * input_bin_widths + input_cumwidths
}

fn softmax(x: &[f32; NUM_BINS]) -> [f32; NUM_BINS] {
    let mx = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut e = [0.0f32; NUM_BINS];
    let mut s = 0.0f32;
    for k in 0..NUM_BINS {
        e[k] = (x[k] - mx).exp();
        s += e[k];
    }
    for k in 0..NUM_BINS {
        e[k] /= s;
    }
    e
}

fn softplus(x: f32) -> f32 {
    // log(1 + exp(x)), numerically stable.
    if x > 20.0 { x } else { (1.0 + x.exp()).ln() }
}
