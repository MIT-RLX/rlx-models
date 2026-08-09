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

//! ECAPA-TDNN forward (matches `Qwen3TTSSpeakerEncoder`).
//!
//! Layout convention: feature tensors are `[channels, T]` (C, T) — the same as
//! PyTorch's NCL after squeezing the batch dim. All Conv1d ops use
//! `padding="same"` with `padding_mode="reflect"`.

use crate::speaker_encoder::config::SpeakerEncoderConfig;
use anyhow::{Context, Result, bail, ensure};
use ndarray::{Array2, ArrayView2, s};
use std::collections::HashMap;

type Tensor = Vec<f32>;

/// One Conv1d (weight `[out, in, k]`, bias `[out]`).
#[derive(Debug, Clone)]
pub struct Conv1d {
    pub weight: Tensor,
    pub bias: Tensor,
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel: usize,
    pub dilation: usize,
}

impl Conv1d {
    /// Forward with reflect-pad to keep `T` constant ("same" padding).
    ///
    /// Implemented as `k` sub-GEMMs:
    ///   `out[c_out, t]  =  bias[c_out]  +  Σ_kk  W[:, :, kk] · pad[:, kk*d : kk*d + t]`
    /// where each kk step is a single `[c_out × c_in] @ [c_in × t]` matmul.
    /// Beats the triple-loop reference on long sequences (≈10-50×).
    pub fn forward(&self, x: ArrayView2<f32>) -> Array2<f32> {
        let (c_in, t) = x.dim();
        debug_assert_eq!(c_in, self.in_channels);
        let k = self.kernel;
        let d = self.dilation;
        let c_out = self.out_channels;
        let total_pad = (k - 1) * d;
        let pad_left = total_pad / 2;
        let pad_right = total_pad - pad_left;
        let padded_t = t + total_pad;

        // Per-channel reflect padding into a contiguous `[c_in, padded_t]` row-major buffer.
        let mut padded = vec![0f32; c_in * padded_t];
        for ci in 0..c_in {
            let row = x.row(ci);
            let base = ci * padded_t;
            for i in 0..pad_left {
                let j = (pad_left - i).min(t.saturating_sub(1));
                padded[base + i] = row[j];
            }
            for i in 0..t {
                padded[base + pad_left + i] = row[i];
            }
            for i in 0..pad_right {
                let j = t.saturating_sub(2 + i);
                padded[base + pad_left + t + i] = row[j];
            }
        }

        // out_flat is row-major [c_out, t] aligned with the GEMM output layout.
        let mut out_flat = vec![0f32; c_out * t];
        // Initialize with bias broadcast over T.
        for co in 0..c_out {
            let b = self.bias[co];
            let off = co * t;
            for ti in 0..t {
                out_flat[off + ti] = b;
            }
        }

        // Workspace for the kk-th column slice of pad: shape [c_in × t].
        // pad is row-major [c_in, padded_t]; pad[:, kk*d : kk*d + t] is a strided
        // view but each row is contiguous, so we just copy `t` floats per channel.
        // For k > 1 we reuse the buffer across iterations.
        let mut slice = vec![0f32; c_in * t];
        for kk in 0..k {
            let col_off = kk * d;
            for ci in 0..c_in {
                let src = ci * padded_t + col_off;
                let dst = ci * t;
                slice[dst..dst + t].copy_from_slice(&padded[src..src + t]);
            }
            // w_kk: take column kk from W[c_out, c_in, k] → [c_out, c_in] contiguous.
            let mut w_kk = vec![0f32; c_out * c_in];
            for co in 0..c_out {
                let row = co * c_in * k;
                for ci in 0..c_in {
                    w_kk[co * c_in + ci] = self.weight[row + ci * k + kk];
                }
            }
            // out_flat += w_kk @ slice  with shapes [c_out × c_in] @ [c_in × t].
            rlx_cpu::blas::sgemm_accumulate(&w_kk, &slice, &mut out_flat, c_out, c_in, t);
        }

        let mut out = Array2::<f32>::zeros((c_out, t));
        for co in 0..c_out {
            let off = co * t;
            for ti in 0..t {
                out[[co, ti]] = out_flat[off + ti];
            }
        }
        out
    }
}

/// `TimeDelayNetBlock` = Conv1d + ReLU.
#[derive(Debug, Clone)]
pub struct Tdnn {
    pub conv: Conv1d,
}

impl Tdnn {
    pub fn forward(&self, x: ArrayView2<f32>) -> Array2<f32> {
        let mut y = self.conv.forward(x);
        for v in y.iter_mut() {
            if *v < 0.0 {
                *v = 0.0;
            }
        }
        y
    }
}

/// `Res2NetBlock`: chunks `C` into `scale` groups, applies a chain of TDNNs
/// (`scale-1` of them) with cumulative sums between groups, concatenates.
#[derive(Debug, Clone)]
pub struct Res2Net {
    pub blocks: Vec<Tdnn>,
    pub scale: usize,
    pub channels: usize,
}

impl Res2Net {
    pub fn forward(&self, x: ArrayView2<f32>) -> Array2<f32> {
        let (c, t) = x.dim();
        debug_assert_eq!(c, self.channels);
        debug_assert!(c % self.scale == 0);
        let group = c / self.scale;
        let mut out = Array2::<f32>::zeros((c, t));
        let mut prev: Option<Array2<f32>> = None;
        for i in 0..self.scale {
            let chunk = x.slice(s![i * group..(i + 1) * group, ..]).to_owned();
            let part = if i == 0 {
                chunk
            } else if i == 1 {
                self.blocks[0].forward(chunk.view())
            } else {
                let prev_ref = prev.as_ref().unwrap();
                let mut sum = chunk;
                sum += prev_ref;
                self.blocks[i - 1].forward(sum.view())
            };
            out.slice_mut(s![i * group..(i + 1) * group, ..])
                .assign(&part);
            prev = Some(part);
        }
        out
    }
}

/// `SqueezeExcitationBlock` = (mean over T) → Conv1d → ReLU → Conv1d → Sigmoid → broadcast multiply.
#[derive(Debug, Clone)]
pub struct SeBlock {
    pub conv1: Conv1d,
    pub conv2: Conv1d,
}

impl SeBlock {
    pub fn forward(&self, x: ArrayView2<f32>) -> Array2<f32> {
        let (c, t) = x.dim();
        // Mean over time → [C, 1].
        let mut mean = Array2::<f32>::zeros((c, 1));
        for ci in 0..c {
            let mut acc = 0f32;
            for ti in 0..t {
                acc += x[[ci, ti]];
            }
            mean[[ci, 0]] = acc / t as f32;
        }
        let mut h = self.conv1.forward(mean.view());
        for v in h.iter_mut() {
            if *v < 0.0 {
                *v = 0.0;
            }
        }
        let mut g = self.conv2.forward(h.view());
        for v in g.iter_mut() {
            *v = 1.0 / (1.0 + (-*v).exp());
        }
        // Broadcast g [C, 1] over x [C, T].
        let mut out = Array2::<f32>::zeros((c, t));
        for ci in 0..c {
            let gv = g[[ci, 0]];
            for ti in 0..t {
                out[[ci, ti]] = x[[ci, ti]] * gv;
            }
        }
        out
    }
}

/// `SqueezeExcitationRes2NetBlock` = tdnn1 → Res2Net → tdnn2 → SE + residual.
#[derive(Debug, Clone)]
pub struct SeRes2NetBlock {
    pub tdnn1: Tdnn,
    pub res2: Res2Net,
    pub tdnn2: Tdnn,
    pub se: SeBlock,
}

impl SeRes2NetBlock {
    pub fn forward(&self, x: ArrayView2<f32>) -> Array2<f32> {
        let residual = x.to_owned();
        let h = self.tdnn1.forward(x);
        let h = self.res2.forward(h.view());
        let h = self.tdnn2.forward(h.view());
        let mut h = self.se.forward(h.view());
        h += &residual;
        h
    }
}

/// Attentive Statistical Pooling with global mask = 1, returning `[C*2, 1]`.
#[derive(Debug, Clone)]
pub struct AttStatPool {
    pub tdnn: Tdnn,
    pub conv: Conv1d,
    pub channels: usize,
}

impl AttStatPool {
    pub fn forward(&self, x: ArrayView2<f32>) -> Array2<f32> {
        let (c, t) = x.dim();
        debug_assert_eq!(c, self.channels);
        // mean / std over T with uniform mask = 1/T.
        let inv_t = 1.0 / t as f32;
        let mut mean = vec![0f32; c];
        for ci in 0..c {
            let mut acc = 0f32;
            for ti in 0..t {
                acc += x[[ci, ti]];
            }
            mean[ci] = acc * inv_t;
        }
        let mut var = vec![0f32; c];
        for ci in 0..c {
            let mu = mean[ci];
            let mut acc = 0f32;
            for ti in 0..t {
                let d = x[[ci, ti]] - mu;
                acc += d * d;
            }
            var[ci] = (acc * inv_t).max(1e-12);
        }
        let std: Vec<f32> = var.iter().map(|v| v.sqrt()).collect();

        // attention input: [x | broadcast(mean) | broadcast(std)] over T → [3C, T].
        let mut att_in = Array2::<f32>::zeros((3 * c, t));
        for ci in 0..c {
            for ti in 0..t {
                att_in[[ci, ti]] = x[[ci, ti]];
            }
            for ti in 0..t {
                att_in[[c + ci, ti]] = mean[ci];
            }
            for ti in 0..t {
                att_in[[2 * c + ci, ti]] = std[ci];
            }
        }
        // tdnn (Conv1d + ReLU) → tanh → conv → softmax over T.
        let h = self.tdnn.forward(att_in.view());
        let mut h2 = Array2::<f32>::zeros(h.dim());
        for ((c0, c1), v) in h.indexed_iter() {
            h2[[c0, c1]] = v.tanh();
        }
        let mut logits = self.conv.forward(h2.view());
        // softmax over T per channel.
        for ci in 0..c {
            let mut maxv = f32::NEG_INFINITY;
            for ti in 0..t {
                let v = logits[[ci, ti]];
                if v > maxv {
                    maxv = v;
                }
            }
            let mut sum = 0f32;
            for ti in 0..t {
                let v = (logits[[ci, ti]] - maxv).exp();
                logits[[ci, ti]] = v;
                sum += v;
            }
            let inv = 1.0 / sum.max(1e-12);
            for ti in 0..t {
                logits[[ci, ti]] *= inv;
            }
        }
        // Weighted mean and std.
        let mut w_mean = vec![0f32; c];
        for ci in 0..c {
            let mut acc = 0f32;
            for ti in 0..t {
                acc += logits[[ci, ti]] * x[[ci, ti]];
            }
            w_mean[ci] = acc;
        }
        let mut w_std = vec![0f32; c];
        for ci in 0..c {
            let mu = w_mean[ci];
            let mut acc = 0f32;
            for ti in 0..t {
                let d = x[[ci, ti]] - mu;
                acc += logits[[ci, ti]] * d * d;
            }
            w_std[ci] = acc.max(1e-12).sqrt();
        }
        // Concat → [2C, 1].
        let mut out = Array2::<f32>::zeros((2 * c, 1));
        for ci in 0..c {
            out[[ci, 0]] = w_mean[ci];
            out[[c + ci, 0]] = w_std[ci];
        }
        out
    }
}

/// Full speaker encoder.
#[derive(Debug, Clone)]
pub struct SpeakerEncoder {
    pub initial: Tdnn,
    pub blocks: Vec<SeRes2NetBlock>,
    pub mfa: Tdnn,
    pub asp: AttStatPool,
    pub fc: Conv1d,
    pub cfg: SpeakerEncoderConfig,
}

impl SpeakerEncoder {
    /// `mel` is `[mel_dim, T]` (no batch dim). Returns x-vector `[enc_dim]`.
    pub fn forward(&self, mel: ArrayView2<f32>) -> Vec<f32> {
        let mut hidden = self.initial.forward(mel);
        let mut outs: Vec<Array2<f32>> = Vec::with_capacity(self.blocks.len() + 1);
        outs.push(hidden.clone());
        for b in &self.blocks {
            hidden = b.forward(hidden.view());
            outs.push(hidden.clone());
        }
        // Concat outputs of blocks[1..].
        let (_, t) = hidden.dim();
        let cat_c: usize = outs[1..].iter().map(|h| h.dim().0).sum();
        let mut cat = Array2::<f32>::zeros((cat_c, t));
        let mut off = 0;
        for h in &outs[1..] {
            let c = h.dim().0;
            cat.slice_mut(s![off..off + c, ..]).assign(h);
            off += c;
        }
        let mfa = self.mfa.forward(cat.view());
        let asp = self.asp.forward(mfa.view());
        let fc = self.fc.forward(asp.view());
        // [enc_dim, 1] → [enc_dim].
        let (out_c, _) = fc.dim();
        let mut x = vec![0f32; out_c];
        for i in 0..out_c {
            x[i] = fc[[i, 0]];
        }
        x
    }
}

// -----------------------------------------------------------------------------
// Weight loader

fn take_conv(
    raw: &mut HashMap<String, (Vec<f32>, Vec<usize>)>,
    prefix: &str,
    in_channels: usize,
    out_channels: usize,
    kernel: usize,
    dilation: usize,
) -> Result<Conv1d> {
    let w_key = format!("{prefix}.weight");
    let b_key = format!("{prefix}.bias");
    let (w, w_shape) = raw
        .remove(&w_key)
        .with_context(|| format!("missing tensor {w_key}"))?;
    let (b, b_shape) = raw
        .remove(&b_key)
        .with_context(|| format!("missing tensor {b_key}"))?;
    ensure!(
        w_shape == vec![out_channels, in_channels, kernel],
        "{w_key} shape {:?} != [{out_channels}, {in_channels}, {kernel}]",
        w_shape
    );
    ensure!(
        b_shape == vec![out_channels],
        "{b_key} shape {:?} != [{out_channels}]",
        b_shape
    );
    Ok(Conv1d {
        weight: w,
        bias: b,
        in_channels,
        out_channels,
        kernel,
        dilation,
    })
}

fn build_tdnn(
    raw: &mut HashMap<String, (Vec<f32>, Vec<usize>)>,
    prefix: &str,
    in_c: usize,
    out_c: usize,
    k: usize,
    d: usize,
) -> Result<Tdnn> {
    Ok(Tdnn {
        conv: take_conv(raw, &format!("{prefix}.conv"), in_c, out_c, k, d)?,
    })
}

fn build_se_res2net(
    raw: &mut HashMap<String, (Vec<f32>, Vec<usize>)>,
    prefix: &str,
    in_c: usize,
    out_c: usize,
    k: usize,
    d: usize,
    scale: usize,
    se_channels: usize,
) -> Result<SeRes2NetBlock> {
    ensure!(
        in_c == out_c,
        "SE-Res2Net assumes in==out for residual (got {in_c} vs {out_c})"
    );
    let tdnn1 = build_tdnn(raw, &format!("{prefix}.tdnn1"), in_c, out_c, 1, 1)?;
    let group = out_c / scale;
    let mut sub_blocks = Vec::with_capacity(scale - 1);
    for j in 0..(scale - 1) {
        sub_blocks.push(build_tdnn(
            raw,
            &format!("{prefix}.res2net_block.blocks.{j}"),
            group,
            group,
            k,
            d,
        )?);
    }
    let res2 = Res2Net {
        blocks: sub_blocks,
        scale,
        channels: out_c,
    };
    let tdnn2 = build_tdnn(raw, &format!("{prefix}.tdnn2"), out_c, out_c, 1, 1)?;
    let se = SeBlock {
        conv1: take_conv(
            raw,
            &format!("{prefix}.se_block.conv1"),
            out_c,
            se_channels,
            1,
            1,
        )?,
        conv2: take_conv(
            raw,
            &format!("{prefix}.se_block.conv2"),
            se_channels,
            out_c,
            1,
            1,
        )?,
    };
    Ok(SeRes2NetBlock {
        tdnn1,
        res2,
        tdnn2,
        se,
    })
}

pub fn build_speaker_encoder(
    cfg: &SpeakerEncoderConfig,
    mut raw: HashMap<String, (Vec<f32>, Vec<usize>)>,
) -> Result<SpeakerEncoder> {
    let prefix = "speaker_encoder";
    let ec = &cfg.enc_channels;
    let ek = &cfg.enc_kernel_sizes;
    let ed = &cfg.enc_dilations;
    ensure!(
        ec.len() == ek.len() && ec.len() == ed.len() && ec.len() >= 3,
        "enc_channels/kernel_sizes/dilations must align (got {} / {} / {})",
        ec.len(),
        ek.len(),
        ed.len()
    );
    let initial = build_tdnn(
        &mut raw,
        &format!("{prefix}.blocks.0"),
        cfg.mel_dim,
        ec[0],
        ek[0],
        ed[0],
    )?;
    let mut blocks = Vec::with_capacity(ec.len() - 2);
    for i in 1..(ec.len() - 1) {
        let block = build_se_res2net(
            &mut raw,
            &format!("{prefix}.blocks.{i}"),
            ec[i - 1],
            ec[i],
            ek[i],
            ed[i],
            cfg.enc_res2net_scale,
            cfg.enc_se_channels,
        )?;
        blocks.push(block);
    }
    let mfa_in = blocks.iter().fold(0, |acc, _| acc + ec[1]); // each block out is ec[i] but they're all ec[1..len-1]
    // Actually mfa input = sum of block[1..end] outputs (HF: hidden_states_list[1:])
    // With enc_channels = [mel_in, c1, c2, c3, c_last], blocks emit c1, c2, c3 → concat = c1+c2+c3
    let mfa_in_sum: usize = ec[1..ec.len() - 1].iter().sum();
    ensure!(
        mfa_in_sum == ec[ec.len() - 1],
        "expected mfa input {} to equal enc_channels last {}",
        mfa_in_sum,
        ec[ec.len() - 1]
    );
    let _ = mfa_in; // silence unused warning of placeholder
    let mfa = build_tdnn(
        &mut raw,
        &format!("{prefix}.mfa"),
        ec[ec.len() - 1],
        ec[ec.len() - 1],
        ek[ek.len() - 1],
        ed[ed.len() - 1],
    )?;
    let asp = AttStatPool {
        tdnn: build_tdnn(
            &mut raw,
            &format!("{prefix}.asp.tdnn"),
            ec[ec.len() - 1] * 3,
            cfg.enc_attention_channels,
            1,
            1,
        )?,
        conv: take_conv(
            &mut raw,
            &format!("{prefix}.asp.conv"),
            cfg.enc_attention_channels,
            ec[ec.len() - 1],
            1,
            1,
        )?,
        channels: ec[ec.len() - 1],
    };
    let fc = take_conv(
        &mut raw,
        &format!("{prefix}.fc"),
        ec[ec.len() - 1] * 2,
        cfg.enc_dim,
        1,
        1,
    )?;
    if !raw.is_empty() {
        let leftover: Vec<&String> = raw.keys().take(5).collect();
        bail!(
            "{} unused speaker_encoder tensors (first: {:?})",
            raw.len(),
            leftover
        );
    }
    Ok(SpeakerEncoder {
        initial,
        blocks,
        mfa,
        asp,
        fc,
        cfg: cfg.clone(),
    })
}
