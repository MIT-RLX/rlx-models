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

//! The three student stages: duration → acoustic latent → waveform.
//!
//! This is the host-eager reference, a direct transcription of upstream's
//! pure-numpy forward passes (`pypkg/sanotts/models.py`), which in turn mirror
//! the fp32 C runtime. Only the subgraphs the shipped voice packages actually
//! use are implemented (`duration_conv` / `token_context` / `piperlite`, no
//! output adapter); anything else errors rather than guessing a tensor layout.

use anyhow::{Result, bail};

use crate::config::{AcousticConfig, DecoderConfig, DurationConfig};
use crate::ops::{
    Mat, conv_transpose1d, conv1d_1x1, conv1d_same, leaky_relu_, linspace01, residual_conv_block,
    tanh_,
};
use crate::voicepack::TensorStore;

/// Fallback id for phonemes outside a component's trained vocab. Schwa is a
/// neutral, always-in-vocab choice — the same fallback the ESP32/WASM ports use.
const SCHWA_FALLBACK_ID: i64 = 59;

/// Residual-bank dilations per branch. (Kernel sizes come from the weights.)
const BANK_DIL1: [usize; 3] = [1, 2, 3];
const BANK_DIL2: [usize; 3] = [2, 6, 12];

/// Remap ids outside `[0, vocab_size)` onto the schwa fallback.
///
/// The frontend's codepoint table is shared and larger than any single
/// component's trained vocab, so this legitimately fires on rare phonemes.
pub fn clamp_ids_to_vocab(ids: &[i64], vocab_size: usize) -> Vec<usize> {
    let fallback = if (SCHWA_FALLBACK_ID as usize) < vocab_size {
        SCHWA_FALLBACK_ID as usize
    } else {
        0
    };
    ids.iter()
        .map(|&i| {
            if i < 0 || i as usize >= vocab_size {
                fallback
            } else {
                i as usize
            }
        })
        .collect()
}

/// numpy's `round`: half-to-even, unlike Rust's half-away-from-zero `f32::round`.
/// A single off-by-one here shifts every later frame, so it is worth matching.
#[inline]
fn round_half_even(v: f32) -> f32 {
    let r = v.round();
    if (v - v.trunc()).abs() == 0.5 && r % 2.0 != 0.0 {
        r - v.signum()
    } else {
        r
    }
}

/// Gather embedding rows into a channel-major `[hidden, n]` matrix.
fn embed_rows(embed: &[f32], hidden: usize, ids: &[usize], n: usize) -> Mat {
    let mut out = Mat::zeros(hidden, n);
    for (t, &id) in ids.iter().enumerate() {
        for h in 0..hidden {
            out.data[h * n + t] = embed[id * hidden + h];
        }
    }
    out
}

fn check_embed(ts: &TensorStore, vocab: usize, hidden: usize, what: &str) -> Result<()> {
    let shape = ts.shape("embedding.weight")?;
    if shape != [vocab, hidden] {
        bail!("{what} embedding shape {shape:?} != config ({vocab}, {hidden})");
    }
    Ok(())
}

fn proj_1x1(x: &Mat, ts: &TensorStore, prefix: &str) -> Result<Mat> {
    let (w, s) = ts.get(&format!("{prefix}.weight"))?;
    let b = ts.data(&format!("{prefix}.bias"))?;
    // Stored as [out, in, 1] (Conv1d k=1) or [out, in] (Linear).
    Ok(conv1d_1x1(x, w, b, s[0]))
}

// ---------------------------------------------------------------------------
// Duration student
// ---------------------------------------------------------------------------

/// Per-token frame counts for `ids`, already scaled by `length_scale`.
pub fn duration_forward(
    ts: &TensorStore,
    cfg: &DurationConfig,
    ids: &[i64],
    length_scale: f32,
) -> Result<Vec<usize>> {
    if cfg.architecture != "duration_conv" {
        bail!("unsupported duration architecture: {:?}", cfg.architecture);
    }
    if ids.is_empty() {
        bail!("duration_forward: empty id sequence");
    }
    check_embed(ts, cfg.vocab_size, cfg.hidden, "duration")?;
    let clamped = clamp_ids_to_vocab(ids, cfg.vocab_size);
    let n = clamped.len();

    let token_x = embed_rows(ts.data("embedding.weight")?, cfg.hidden, &clamped, n);

    // [position, length hint, validity] — the validity row is all ones because
    // this path never batches, so there is nothing to mask.
    let mut features = Mat::zeros(3, n);
    features.row_mut(0).copy_from_slice(&linspace01(n));
    let length_hint = ((n as f64).ln_1p() / (cfg.max_tokens as f64).ln_1p()) as f32;
    features.row_mut(1).fill(length_hint);
    features.row_mut(2).fill(1.0);

    let mut x = proj_1x1(&token_x.vstack(&features), ts, "input_proj")?;
    for i in 0..cfg.depth {
        x = residual_conv_block(&x, ts, &format!("blocks.{i}"))?;
    }
    let log_duration = proj_1x1(&x, ts, "output")?;

    let max_duration = cfg.max_duration as f32;
    Ok(log_duration
        .row(0)
        .iter()
        .map(|&v| {
            let d = v.exp().max(1.0);
            let d = round_half_even(d * length_scale).clamp(1.0, max_duration);
            d as usize
        })
        .collect())
}

// ---------------------------------------------------------------------------
// Acoustic student
// ---------------------------------------------------------------------------

/// Frame-rate input features for the acoustic model's second half: the
/// duration-expanded token context stacked with three positional rows.
///
/// Split out because the expansion is data-dependent (it needs the predicted
/// durations), so it is the natural host/graph boundary.
pub fn acoustic_frame_input(
    ts: &TensorStore,
    cfg: &AcousticConfig,
    ids: &[i64],
    durations: &[usize],
) -> Result<Mat> {
    if cfg.architecture != "token_context" {
        bail!("unsupported acoustic architecture: {:?}", cfg.architecture);
    }
    let adapters: Vec<&str> = ts
        .names()
        .into_iter()
        .filter(|n| n.contains("adapter"))
        .collect();
    if !adapters.is_empty() {
        bail!(
            "acoustic checkpoint has an output adapter ({adapters:?}); only the \
             adapter-free token_context path is implemented"
        );
    }
    if durations.len() != ids.len() {
        bail!("acoustic: id/duration length mismatch");
    }
    if durations.iter().any(|&d| d < 1) {
        bail!("acoustic: non-positive duration");
    }
    check_embed(ts, cfg.vocab_size, cfg.hidden, "acoustic")?;

    let clamped = clamp_ids_to_vocab(ids, cfg.vocab_size);
    let n = clamped.len();
    let frames: usize = durations.iter().sum();

    // -- token stage --
    let token_x = embed_rows(ts.data("embedding.weight")?, cfg.hidden, &clamped, n);
    let mut token_features = Mat::zeros(2, n);
    token_features.row_mut(0).copy_from_slice(&linspace01(n));
    let max_d = durations.iter().copied().max().unwrap_or(1).max(1) as f64;
    let ln_max = max_d.ln_1p();
    for (t, &d) in durations.iter().enumerate() {
        token_features.data[n + t] = ((d as f64).ln_1p() / ln_max) as f32;
    }

    let mut token_x = proj_1x1(&token_x.vstack(&token_features), ts, "token_input_proj")?;
    for i in 0..cfg.token_depth {
        token_x = residual_conv_block(&token_x, ts, &format!("token_blocks.{i}"))?;
    }

    // -- expand token context to frames --
    let mut expanded = Mat::zeros(cfg.hidden, frames);
    for h in 0..cfg.hidden {
        let src = token_x.row(h);
        let dst = &mut expanded.data[h * frames..(h + 1) * frames];
        let mut pos = 0;
        for (t, &d) in durations.iter().enumerate() {
            dst[pos..pos + d].fill(src[t]);
            pos += d;
        }
    }

    let mut frame_features = Mat::zeros(3, frames);
    frame_features
        .row_mut(0)
        .copy_from_slice(&linspace01(frames));
    let token_count = (n.saturating_sub(1)).max(1) as f32;
    let mut pos = 0;
    for (t, &d) in durations.iter().enumerate() {
        let tp = t as f32 / token_count;
        for j in 0..d {
            frame_features.data[frames + pos + j] = tp;
            frame_features.data[2 * frames + pos + j] = if d == 1 {
                0.0
            } else {
                (j as f64 / (d - 1) as f64) as f32
            };
        }
        pos += d;
    }

    Ok(expanded.vstack(&frame_features))
}

/// The acoustic model's frame half: `[hidden + 3, frames]` → `[out_channels, frames]`.
pub fn acoustic_frame_forward(ts: &TensorStore, cfg: &AcousticConfig, input: &Mat) -> Result<Mat> {
    let mut x = proj_1x1(input, ts, "frame_input_proj")?;
    for i in 0..cfg.depth {
        x = residual_conv_block(&x, ts, &format!("frame_blocks.{i}"))?;
    }
    let latent = proj_1x1(&x, ts, "output")?;
    if latent.rows != cfg.out_channels {
        bail!(
            "acoustic: output has {} channels, config says {}",
            latent.rows,
            cfg.out_channels
        );
    }
    Ok(latent)
}

/// Full acoustic pass: ids + durations → latent `[out_channels, frames]`.
pub fn acoustic_forward(
    ts: &TensorStore,
    cfg: &AcousticConfig,
    ids: &[i64],
    durations: &[usize],
) -> Result<Mat> {
    let input = acoustic_frame_input(ts, cfg, ids, durations)?;
    acoustic_frame_forward(ts, cfg, &input)
}

// ---------------------------------------------------------------------------
// Decoder student
// ---------------------------------------------------------------------------

/// Upsampling stage geometry: (kernel, stride, padding, up name, bank prefix).
pub(crate) const STAGES: [(usize, usize, usize, &str, &str); 3] = [
    (16, 8, 4, "up0", "res0.0"),
    (16, 8, 4, "up1", "res1.0"),
    (8, 4, 2, "up2", "res2.0"),
];

/// Reject decoder configs whose subgraph this crate does not implement.
pub(crate) fn check_decoder(cfg: &DecoderConfig) -> Result<()> {
    if cfg.variant != "piperlite" {
        bail!("unsupported decoder variant: {:?}", cfg.variant);
    }
    if cfg.activation != "leaky_relu" {
        bail!("only activation='leaky_relu' decoders are implemented");
    }
    if cfg.pre_tanh_repair_channels > 0 {
        bail!("pre_tanh_repair is not implemented (no shipped voice uses it)");
    }
    if cfg.res_layers != 1 {
        bail!("only res_layers=1 is implemented, got {}", cfg.res_layers);
    }
    if cfg.channels.len() < 4 {
        bail!(
            "decoder config needs 4 channel counts, got {:?}",
            cfg.channels
        );
    }
    Ok(())
}

fn residual_bank(x: &Mat, ts: &TensorStore, prefix: &str, branches: &[usize]) -> Result<Mat> {
    if branches.is_empty() {
        bail!("{prefix}: empty residual-bank branch list");
    }
    let mut acc = Mat::zeros(x.rows, x.cols);
    for &branch in branches {
        if branch >= BANK_DIL1.len() {
            bail!("{prefix}: branch index {branch} out of range");
        }
        let mut t = x.clone();
        leaky_relu_(&mut t.data, 0.1);
        let (w1, s1) = ts.get(&format!("{prefix}.blocks.{branch}.conv1.weight"))?;
        let b1 = ts.data(&format!("{prefix}.blocks.{branch}.conv1.bias"))?;
        let mut y1 = conv1d_same(&t, w1, b1, s1[0], s1[2], BANK_DIL1[branch]);
        for (y, xv) in y1.data.iter_mut().zip(&x.data) {
            *y += xv;
        }

        let mut t2 = y1.clone();
        leaky_relu_(&mut t2.data, 0.1);
        let (w2, s2) = ts.get(&format!("{prefix}.blocks.{branch}.conv2.weight"))?;
        let b2 = ts.data(&format!("{prefix}.blocks.{branch}.conv2.bias"))?;
        let y2 = conv1d_same(&t2, w2, b2, s2[0], s2[2], BANK_DIL2[branch]);

        for ((a, u), y) in acc.data.iter_mut().zip(&y2.data).zip(&y1.data) {
            *a += u + y;
        }
    }
    let inv = 1.0 / branches.len() as f32;
    for a in &mut acc.data {
        *a *= inv;
    }
    Ok(acc)
}

fn apply_post_filter(audio: &mut [f32], ts: &TensorStore, cfg: &DecoderConfig) -> Result<()> {
    if cfg.post_filter_channels == 0 {
        return Ok(());
    }
    let n = audio.len();
    let x = Mat::from_vec(audio.to_vec(), 1, n);
    let (wi, si) = ts.get("post_filter.in_conv.weight")?;
    let bi = ts.data("post_filter.in_conv.bias")?;
    let mut r = conv1d_same(&x, wi, bi, si[0], si[2], 1);
    for layer in 0..cfg.post_filter_layers {
        let scale = ts.scalar(&format!("post_filter.units.{layer}.scale"))?;
        let mut t = r.clone();
        leaky_relu_(&mut t.data, 0.1);
        let (w1, s1) = ts.get(&format!("post_filter.units.{layer}.conv1.weight"))?;
        let b1 = ts.data(&format!("post_filter.units.{layer}.conv1.bias"))?;
        let mut u = conv1d_same(&t, w1, b1, s1[0], s1[2], 1 + layer);
        leaky_relu_(&mut u.data, 0.1);
        let (w2, s2) = ts.get(&format!("post_filter.units.{layer}.conv2.weight"))?;
        let b2 = ts.data(&format!("post_filter.units.{layer}.conv2.bias"))?;
        let u2 = conv1d_same(&u, w2, b2, s2[0], s2[2], 1);
        for (rv, uv) in r.data.iter_mut().zip(&u2.data) {
            *rv += scale * uv;
        }
    }
    let (wo, so) = ts.get("post_filter.out_conv.weight")?;
    let bo = ts.data("post_filter.out_conv.bias")?;
    let out = conv1d_same(&r, wo, bo, so[0], so[2], 1);
    for (a, o) in audio.iter_mut().zip(out.row(0)) {
        *a = (*a + cfg.post_filter_scale * o).tanh();
    }
    Ok(())
}

/// Latent `[in_channels, frames]` → mono waveform `[frames * 256]`.
pub fn decoder_forward(ts: &TensorStore, cfg: &DecoderConfig, latent: &Mat) -> Result<Vec<f32>> {
    check_decoder(cfg)?;

    let (wp, sp) = ts.get("pre.weight")?;
    let bp = ts.data("pre.bias")?;
    let mut x = conv1d_same(latent, wp, bp, sp[0], sp[2], 1);

    for (stage, &(k, stride, pad, up_name, bank_prefix)) in STAGES.iter().enumerate() {
        let out_c = cfg.channels[stage + 1];
        leaky_relu_(&mut x.data, 0.1);
        let (wu, su) = ts.get(&format!("{up_name}.weight"))?;
        let bu = ts.data(&format!("{up_name}.bias"))?;
        debug_assert_eq!(su[2], k);
        x = conv_transpose1d(&x, wu, bu, su[1], su[2], stride, pad);
        if x.rows != out_c {
            bail!(
                "{up_name}: produced {} channels, config says {out_c}",
                x.rows
            );
        }
        x = residual_bank(&x, ts, bank_prefix, &cfg.stage_branches(stage))?;
        let _ = k;
    }

    leaky_relu_(&mut x.data, 0.01);
    let (wo, so) = ts.get("post.weight")?;
    let bo = ts.data("post.bias")?;
    let out = conv1d_same(&x, wo, bo, so[0], so[2], 1);
    let mut audio = out.row(0).to_vec();
    tanh_(&mut audio);
    apply_post_filter(&mut audio, ts, cfg)?;
    Ok(audio)
}
