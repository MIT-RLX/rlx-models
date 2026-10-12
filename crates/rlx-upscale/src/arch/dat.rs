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

//! **DAT** — Dual Aggregation Transformer.
//!
//! DAT alternates two block types and fuses each with a parallel convolution:
//!
//! * **DSTB** (even blocks) — *spatial* aggregation. Channels are split in half
//!   and attended in **rectangular** windows of opposite orientation: `[h, w]`
//!   for the first half, `[w, h]` for the second. Long thin windows see far
//!   along one axis cheaply, and the two halves together cover both.
//! * **DCTB** (odd blocks) — *channel* aggregation. Attention is transposed:
//!   `q·kᵀ` is taken over the **channel** axis with L2-normalized operands and
//!   a learned per-head temperature, so the attention matrix is `head_dim ×
//!   head_dim` and independent of the tile area.
//!
//! Both wrap an **Adaptive Interaction Module**: a depthwise conv branch and
//! the attention branch each gate the other, one by a channel map and one by a
//! spatial map.
//!
//! # What is resolved on the host
//!
//! DAT has no relative-position *table*; it has a small MLP
//! (`DynamicPosBias`) that maps a displacement `(dy, dx)` to a per-head bias.
//! Its inputs (`rpe_biases`) are a fixed buffer and its weights are frozen at
//! inference, so `dynamic_pos_bias` evaluates the whole MLP on the host and
//! injects the result as a constant — the graph never sees a `Linear` there.
//! `BatchNorm2d` is likewise folded to a per-channel scale/shift pair.
//!
//! # Window padding
//!
//! The reference pads the feature map up to `max(split_size)` inside every
//! attention. [`crate::config::ModelConfig::size_multiple`] already forces the
//! tile to a multiple of both extents, so that padding is always zero-width
//! here and is asserted rather than emitted.

use anyhow::{Context, Result, ensure};
use rlx_core::weight_map::WeightMap;
use rlx_ir::HirGraphExt;
use rlx_ir::hir::HirNodeId;

use crate::config::{DatParams, ModelConfig, ResiConnection, Upsampler};
use crate::nn::Builder;

/// Geometry of one rectangular-window attention branch.
#[derive(Clone, Copy)]
struct Rect {
    h: usize,
    w: usize,
    /// Window extent down and across.
    sh: usize,
    sw: usize,
    gh: usize,
    gw: usize,
    n: usize,
    nw: usize,
}

impl Rect {
    fn new(h: usize, w: usize, sh: usize, sw: usize) -> Result<Self> {
        ensure!(
            h.is_multiple_of(sh) && w.is_multiple_of(sw),
            "a {h}×{w} feature map does not partition into {sh}×{sw} windows"
        );
        Ok(Self {
            h,
            w,
            sh,
            sw,
            gh: h / sh,
            gw: w / sw,
            n: sh * sw,
            nw: (h / sh) * (w / sw),
        })
    }
}

/// `img2windows`: `[1, C, H, W]` → `[nW, sh·sw, C]`.
fn img2windows(b: &mut Builder, x: HirNodeId, r: Rect, c: usize) -> HirNodeId {
    // Rank 5, batch axis elided — CoreML's MIL caps rank at 5.
    let five = b.g().reshape_(
        x,
        vec![c as i64, r.gh as i64, r.sh as i64, r.gw as i64, r.sw as i64],
    );
    // [C, gh, sh, gw, sw] → [gh, gw, sh, sw, C]
    let perm = b.g().transpose_(five, vec![1, 3, 2, 4, 0]);
    b.g()
        .reshape_(perm, vec![r.nw as i64, r.n as i64, c as i64])
}

/// `windows2img`: `[nW, sh·sw, C]` → `[1, H·W, C]` (token order, i.e. the
/// reference's `[B, H, W, C]` flattened).
fn windows2img(b: &mut Builder, x: HirNodeId, r: Rect, c: usize) -> HirNodeId {
    let five = b.g().reshape_(
        x,
        vec![r.gh as i64, r.gw as i64, r.sh as i64, r.sw as i64, c as i64],
    );
    // [gh, gw, sh, sw, C] → [gh, sh, gw, sw, C]
    let perm = b.g().transpose_(five, vec![0, 2, 1, 3, 4]);
    b.g().reshape_(perm, vec![1, (r.h * r.w) as i64, c as i64])
}

/// `torch.roll` over the spatial axes of a `[1, H·W, C]` token tensor.
fn roll(
    b: &mut Builder,
    x: HirNodeId,
    h: usize,
    w: usize,
    c: usize,
    sy: isize,
    sx: isize,
) -> HirNodeId {
    if sy == 0 && sx == 0 {
        return x;
    }
    let four = b.g().reshape_(x, vec![1, h as i64, w as i64, c as i64]);
    let shift = |b: &mut Builder, t: HirNodeId, axis: usize, extent: usize, s: isize| {
        let s = s.rem_euclid(extent as isize) as usize;
        if s == 0 {
            return t;
        }
        let tail = b.g().narrow_(t, axis, extent - s, s);
        let head = b.g().narrow_(t, axis, 0, extent - s);
        b.g().concat_(vec![tail, head], axis)
    };
    let t = shift(b, four, 1, h, sy);
    let t = shift(b, t, 2, w, sx);
    b.g().reshape_(t, vec![1, (h * w) as i64, c as i64])
}

/// Relative-position index for a rectangular window.
fn rect_rpe_index(sh: usize, sw: usize) -> Vec<usize> {
    let n = sh * sw;
    let span = 2 * sw - 1;
    let mut idx = vec![0usize; n * n];
    for i in 0..n {
        let (ih, iw) = (i / sw, i % sw);
        for j in 0..n {
            let (jh, jw) = (j / sw, j % sw);
            let dh = ih + sh - 1 - jh;
            let dw = iw + sw - 1 - jw;
            idx[i * n + j] = dh * span + dw;
        }
    }
    idx
}

/// The displacement grid `DynamicPosBias` is evaluated on:
/// `[(2sh−1)·(2sw−1), 2]`, row-major over `(dy, dx)`.
fn rpe_biases(sh: usize, sw: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity((2 * sh - 1) * (2 * sw - 1) * 2);
    for dy in 0..(2 * sh - 1) {
        for dx in 0..(2 * sw - 1) {
            out.push(dy as f32 + 1.0 - sh as f32);
            out.push(dx as f32 + 1.0 - sw as f32);
        }
    }
    out
}

/// Row-major `[rows, in] × [out, in]ᵀ + [out]` on the host.
fn host_linear(
    x: &[f32],
    rows: usize,
    in_f: usize,
    w: &[f32],
    bias: &[f32],
    out_f: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * out_f];
    for r in 0..rows {
        for o in 0..out_f {
            let mut acc = bias[o];
            for i in 0..in_f {
                acc += x[r * in_f + i] * w[o * in_f + i];
            }
            out[r * out_f + o] = acc;
        }
    }
    out
}

/// Row-wise `LayerNorm` on the host.
fn host_layer_norm(x: &mut [f32], rows: usize, n: usize, gamma: &[f32], beta: &[f32]) {
    for r in 0..rows {
        let row = &mut x[r * n..(r + 1) * n];
        let mean = row.iter().sum::<f32>() / n as f32;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n as f32;
        let inv = 1.0 / (var + 1e-5).sqrt();
        for (i, v) in row.iter_mut().enumerate() {
            *v = (*v - mean) * inv * gamma[i] + beta[i];
        }
    }
}

/// Evaluate `DynamicPosBias` on the host and return the `[1, heads, N, N]`
/// additive bias.
///
/// The MLP is `pos_proj → pos1 → pos2 → pos3`, each `posN` being
/// `Linear(ReLU(LayerNorm(·)))`. Its input is a fixed displacement grid and its
/// weights are frozen, so the whole thing is a compile-time constant — running
/// it in the graph would recompute the same `(2sh−1)(2sw−1) × heads` table on
/// every tile.
#[allow(clippy::too_many_arguments)]
fn dynamic_pos_bias(
    b: &mut Builder,
    wm: &mut WeightMap,
    prefix: &str,
    sh: usize,
    sw: usize,
    heads: usize,
    pos_dim: usize,
    tag: &str,
) -> Result<HirNodeId> {
    let rows = (2 * sh - 1) * (2 * sw - 1);
    let biases = rpe_biases(sh, sw);

    let w0 = b.take_host(wm, &format!("{prefix}.pos_proj.weight"), &[pos_dim, 2])?;
    let b0 = b.take_host(wm, &format!("{prefix}.pos_proj.bias"), &[pos_dim])?;
    let mut h = host_linear(&biases, rows, 2, &w0, &b0, pos_dim);

    for (stage, out_f) in [(1usize, pos_dim), (2, pos_dim), (3, heads)] {
        let g = b.take_host(wm, &format!("{prefix}.pos{stage}.0.weight"), &[pos_dim])?;
        let be = b.take_host(wm, &format!("{prefix}.pos{stage}.0.bias"), &[pos_dim])?;
        host_layer_norm(&mut h, rows, pos_dim, &g, &be);
        for v in h.iter_mut() {
            *v = v.max(0.0);
        }
        let w = b.take_host(
            wm,
            &format!("{prefix}.pos{stage}.2.weight"),
            &[out_f, pos_dim],
        )?;
        let bi = b.take_host(wm, &format!("{prefix}.pos{stage}.2.bias"), &[out_f])?;
        h = host_linear(&h, rows, pos_dim, &w, &bi, out_f);
    }

    let index = rect_rpe_index(sh, sw);
    let n = sh * sw;
    let mut bias = vec![0.0f32; heads * n * n];
    for (pos, &row) in index.iter().enumerate() {
        ensure!(
            row < rows,
            "{prefix}: position index {row} past {rows} rows"
        );
        for head in 0..heads {
            bias[head * n * n + pos] = h[row * heads + head];
        }
    }
    Ok(b.constant(tag, bias, &[1, heads, n, n]))
}

/// The `0` / `−100` shift mask for one rectangular orientation.
fn rect_shift_mask(r: Rect, sy: usize, sx: usize) -> Vec<f32> {
    let region = |p: usize, extent: usize, win: usize, shift: usize| -> usize {
        if p < extent - win {
            0
        } else if p < extent - shift {
            1
        } else {
            2
        }
    };
    let mut ids = vec![0usize; r.h * r.w];
    for y in 0..r.h {
        for x in 0..r.w {
            ids[y * r.w + x] = region(y, r.h, r.sh, sy) * 3 + region(x, r.w, r.sw, sx);
        }
    }
    let mut out = vec![0.0f32; r.nw * r.n * r.n];
    for wi in 0..r.nw {
        let (by, bx) = (wi / r.gw, wi % r.gw);
        let win: Vec<usize> = (0..r.n)
            .map(|t| ids[(by * r.sh + t / r.sw) * r.w + bx * r.sw + t % r.sw])
            .collect();
        for i in 0..r.n {
            for j in 0..r.n {
                if win[i] != win[j] {
                    out[wi * r.n * r.n + i * r.n + j] = -100.0;
                }
            }
        }
    }
    out
}

/// Fold an inference-mode `BatchNorm2d` into one per-channel scale and shift.
fn batch_norm(
    b: &mut Builder,
    wm: &mut WeightMap,
    prefix: &str,
    x: HirNodeId,
    c: usize,
) -> Result<HirNodeId> {
    let gamma = b.take_host(wm, &format!("{prefix}.weight"), &[c])?;
    let beta = b.take_host(wm, &format!("{prefix}.bias"), &[c])?;
    let mean = b.take_host(wm, &format!("{prefix}.running_mean"), &[c])?;
    let var = b.take_host(wm, &format!("{prefix}.running_var"), &[c])?;
    // PyTorch's BatchNorm2d default eps.
    let mut scale = vec![0.0f32; c];
    let mut shift = vec![0.0f32; c];
    for i in 0..c {
        // A negative running variance means a corrupt checkpoint. Saying so is
        // far better than the alternative: `sqrt` returns NaN, every ReLU
        // downstream quietly maps it back to 0, and the image comes out wrong
        // in a way that looks like a bad model rather than a bad load.
        ensure!(
            var[i] >= 0.0,
            "{prefix}: running_var[{i}] is {}, which is not a variance",
            var[i]
        );
        scale[i] = gamma[i] / (var[i] + 1e-5).sqrt();
        shift[i] = beta[i] - mean[i] * scale[i];
    }
    let s = b.constant(&format!("{prefix}.scale"), scale, &[1, c, 1, 1]);
    let t = b.constant(&format!("{prefix}.shift"), shift, &[1, c, 1, 1]);
    let y = b.g().mul(x, s);
    Ok(b.g().add(y, t))
}

/// Depthwise 3×3 → BatchNorm → GELU, on NCHW.
fn dwconv(
    b: &mut Builder,
    wm: &mut WeightMap,
    prefix: &str,
    x: HirNodeId,
    c: usize,
) -> Result<HirNodeId> {
    let h = b.conv(
        wm,
        &format!("{prefix}.0"),
        x,
        c,
        [3, 3],
        [1, 1],
        [1, 1],
        c,
        true,
    )?;
    let h = batch_norm(b, wm, &format!("{prefix}.1"), h, c)?;
    Ok(b.gelu(h))
}

/// `channel_interaction`: global pool → squeeze → BN → GELU → excite.
fn channel_interaction(
    b: &mut Builder,
    wm: &mut WeightMap,
    prefix: &str,
    x: HirNodeId,
    c: usize,
) -> Result<HirNodeId> {
    let mid = c / 8;
    let pooled = b.g().mean(x, vec![2, 3], true);
    let h = b.conv1x1(wm, &format!("{prefix}.1"), pooled, mid)?;
    let h = batch_norm(b, wm, &format!("{prefix}.2"), h, mid)?;
    let h = b.gelu(h);
    b.conv1x1(wm, &format!("{prefix}.4"), h, c)
}

/// `spatial_interaction`: squeeze to one channel per pixel.
fn spatial_interaction(
    b: &mut Builder,
    wm: &mut WeightMap,
    prefix: &str,
    x: HirNodeId,
    c: usize,
) -> Result<HirNodeId> {
    let mid = c / 16;
    let h = b.conv1x1(wm, &format!("{prefix}.0"), x, mid)?;
    let h = batch_norm(b, wm, &format!("{prefix}.1"), h, mid)?;
    let h = b.gelu(h);
    b.conv1x1(wm, &format!("{prefix}.3"), h, 1)
}

fn to_nchw(b: &mut Builder, x: HirNodeId, h: usize, w: usize, c: usize) -> HirNodeId {
    let t = b.g().transpose_(x, vec![0, 2, 1]);
    b.g().reshape_(t, vec![1, c as i64, h as i64, w as i64])
}

fn to_tokens(b: &mut Builder, x: HirNodeId, h: usize, w: usize, c: usize) -> HirNodeId {
    let f = b.g().reshape_(x, vec![1, c as i64, (h * w) as i64]);
    b.g().transpose_(f, vec![0, 2, 1])
}

/// One rectangular-window attention branch over half the channels.
#[allow(clippy::too_many_arguments)]
fn spatial_attention(
    b: &mut Builder,
    wm: &mut WeightMap,
    prefix: &str,
    q: HirNodeId,
    k: HirNodeId,
    v: HirNodeId,
    r: Rect,
    c: usize,
    heads: usize,
    pos_dim: usize,
    mask: Option<HirNodeId>,
) -> Result<HirNodeId> {
    ensure!(
        c.is_multiple_of(heads),
        "{prefix}: {c} channels do not split into {heads} heads"
    );
    let hd = c / heads;
    let im2win = |b: &mut Builder, t: HirNodeId| {
        let nchw = to_nchw(b, t, r.h, r.w, c);
        let win = img2windows(b, nchw, r, c);
        let s = b
            .g()
            .reshape_(win, vec![r.nw as i64, r.n as i64, heads as i64, hd as i64]);
        b.g().transpose_(s, vec![0, 2, 1, 3])
    };
    let qh = im2win(b, q);
    let kh = im2win(b, k);
    let vh = im2win(b, v);

    let scale = b.scalar_like("dat_scale", 1.0 / (hd as f32).sqrt(), qh);
    let qh = b.g().mul(qh, scale);
    let kt = b.g().transpose_(kh, vec![0, 1, 3, 2]);
    let mut attn = b.g().mm(qh, kt);

    let bias = dynamic_pos_bias(
        b,
        wm,
        &format!("{prefix}.pos"),
        r.sh,
        r.sw,
        heads,
        pos_dim,
        "dat_rpe",
    )?;
    attn = b.g().add(attn, bias);

    if let Some(m) = mask {
        let grouped = b.g().reshape_(
            attn,
            vec![1, r.nw as i64, heads as i64, r.n as i64, r.n as i64],
        );
        let masked = b.g().add(grouped, m);
        attn = b.g().reshape_(
            masked,
            vec![r.nw as i64, heads as i64, r.n as i64, r.n as i64],
        );
    }

    let attn = b.g().sm(attn, -1);
    let out = b.g().mm(attn, vh);
    let out = b.g().transpose_(out, vec![0, 2, 1, 3]);
    let out = b.g().reshape_(out, vec![r.nw as i64, r.n as i64, c as i64]);
    Ok(windows2img(b, out, r, c))
}

/// `Adaptive_Spatial_Attention` (the even, DSTB blocks).
#[allow(clippy::too_many_arguments)]
fn adaptive_spatial_attention(
    b: &mut Builder,
    wm: &mut WeightMap,
    prefix: &str,
    x: HirNodeId,
    h: usize,
    w: usize,
    c: usize,
    heads: usize,
    split: [usize; 2],
    qkv_bias: bool,
    shifted: bool,
) -> Result<HirNodeId> {
    let half = c / 2;
    let qkv = b.linear(wm, &format!("{prefix}.qkv"), x, 3 * c, qkv_bias)?;
    let q = b.g().narrow_(qkv, 2, 0, c);
    let k = b.g().narrow_(qkv, 2, c, c);
    let v = b.g().narrow_(qkv, 2, 2 * c, c);

    // Branch 0 keeps the window as given; branch 1 transposes it, which is what
    // lets the two halves cover both a wide and a tall receptive field.
    let r0 = Rect::new(h, w, split[0], split[1])?;
    let r1 = Rect::new(h, w, split[1], split[0])?;
    let (sy, sx) = (split[0] / 2, split[1] / 2);
    // `dim // 4` of the *branch* width, which is `c / 2`; then `// 4` again.
    let pos_dim = (half / 4) / 4;
    let branch_heads = heads / 2;

    let take_half = |b: &mut Builder, t: HirNodeId, lo: bool| {
        if lo {
            b.g().narrow_(t, 2, 0, half)
        } else {
            b.g().narrow_(t, 2, half, half)
        }
    };

    let (x1, x2) = if shifted {
        // Branch 0 shifts by (sy, sx); branch 1 by the swapped (sx, sy),
        // matching its swapped window.
        let sh = |b: &mut Builder, t: HirNodeId, lo: bool, a: isize, c2: isize| {
            let part = take_half(b, t, lo);
            roll(b, part, h, w, half, a, c2)
        };
        let q0 = sh(b, q, true, -(sy as isize), -(sx as isize));
        let k0 = sh(b, k, true, -(sy as isize), -(sx as isize));
        let v0 = sh(b, v, true, -(sy as isize), -(sx as isize));
        let q1 = sh(b, q, false, -(sx as isize), -(sy as isize));
        let k1 = sh(b, k, false, -(sx as isize), -(sy as isize));
        let v1 = sh(b, v, false, -(sx as isize), -(sy as isize));

        let m0 = b.constant(
            "dat_mask0",
            rect_shift_mask(r0, sy, sx),
            &[1, r0.nw, 1, r0.n, r0.n],
        );
        let m1 = b.constant(
            "dat_mask1",
            rect_shift_mask(r1, sx, sy),
            &[1, r1.nw, 1, r1.n, r1.n],
        );
        let a0 = spatial_attention(
            b,
            wm,
            &format!("{prefix}.attns.0"),
            q0,
            k0,
            v0,
            r0,
            half,
            branch_heads,
            pos_dim,
            Some(m0),
        )?;
        let a1 = spatial_attention(
            b,
            wm,
            &format!("{prefix}.attns.1"),
            q1,
            k1,
            v1,
            r1,
            half,
            branch_heads,
            pos_dim,
            Some(m1),
        )?;
        (
            roll(b, a0, h, w, half, sy as isize, sx as isize),
            roll(b, a1, h, w, half, sx as isize, sy as isize),
        )
    } else {
        let a0 = {
            let (q0, k0, v0) = (
                take_half(b, q, true),
                take_half(b, k, true),
                take_half(b, v, true),
            );
            spatial_attention(
                b,
                wm,
                &format!("{prefix}.attns.0"),
                q0,
                k0,
                v0,
                r0,
                half,
                branch_heads,
                pos_dim,
                None,
            )?
        };
        let a1 = {
            let (q1, k1, v1) = (
                take_half(b, q, false),
                take_half(b, k, false),
                take_half(b, v, false),
            );
            spatial_attention(
                b,
                wm,
                &format!("{prefix}.attns.1"),
                q1,
                k1,
                v1,
                r1,
                half,
                branch_heads,
                pos_dim,
                None,
            )?
        };
        (a0, a1)
    };
    let attended = b.g().concat_(vec![x1, x2], 2);

    // Adaptive Interaction: each branch gates the other.
    let v_nchw = to_nchw(b, v, h, w, c);
    let conv_x = dwconv(b, wm, &format!("{prefix}.dwconv"), v_nchw, c)?;
    let ch_map = channel_interaction(b, wm, &format!("{prefix}.channel_interaction"), conv_x, c)?;
    let ch_map = b.g().reshape_(ch_map, vec![1, 1, c as i64]);
    let attn_nchw = to_nchw(b, attended, h, w, c);
    let sp_map = spatial_interaction(
        b,
        wm,
        &format!("{prefix}.spatial_interaction"),
        attn_nchw,
        c,
    )?;

    let ch_gate = b.sigmoid(ch_map);
    let attended = b.g().mul(attended, ch_gate);
    let sp_gate = b.sigmoid(sp_map);
    let conv_x = b.g().mul(sp_gate, conv_x);
    let conv_tokens = to_tokens(b, conv_x, h, w, c);

    let sum = b.g().add(attended, conv_tokens);
    b.linear(wm, &format!("{prefix}.proj"), sum, c, true)
}

/// `Adaptive_Channel_Attention` (the odd, DCTB blocks).
#[allow(clippy::too_many_arguments)]
fn adaptive_channel_attention(
    b: &mut Builder,
    wm: &mut WeightMap,
    prefix: &str,
    x: HirNodeId,
    h: usize,
    w: usize,
    c: usize,
    heads: usize,
    qkv_bias: bool,
) -> Result<HirNodeId> {
    ensure!(
        c.is_multiple_of(heads),
        "{prefix}: {c} channels do not split into {heads} heads"
    );
    let hd = c / heads;
    let n = h * w;
    let qkv = b.linear(wm, &format!("{prefix}.qkv"), x, 3 * c, qkv_bias)?;

    // [1, N, C] → [1, heads, hd, N]: the transposed layout is the point — the
    // attention matrix is hd × hd, so its size does not grow with the tile.
    let to_heads = |b: &mut Builder, t: HirNodeId| {
        let r = b
            .g()
            .reshape_(t, vec![1, n as i64, heads as i64, hd as i64]);
        let p = b.g().transpose_(r, vec![0, 2, 1, 3]);
        b.g().transpose_(p, vec![0, 1, 3, 2])
    };
    let q = b.g().narrow_(qkv, 2, 0, c);
    let k = b.g().narrow_(qkv, 2, c, c);
    let v = b.g().narrow_(qkv, 2, 2 * c, c);
    let q = to_heads(b, q);
    let k = to_heads(b, k);
    let v = to_heads(b, v);

    // `F.normalize(·, dim=-1)`: unit L2 norm along N.
    let l2 = |b: &mut Builder, t: HirNodeId| {
        let sq = b.g().mul(t, t);
        let sum = b.g().sum(sq, vec![3], true);
        let eps = b.scalar_like("dat_l2_eps", 1e-12, sum);
        let safe = b.g().add(sum, eps);
        let norm = b.g().sqrt(safe);
        b.g().div(t, norm)
    };
    let qn = l2(b, q);
    let kn = l2(b, k);

    let kt = b.g().transpose_(kn, vec![0, 1, 3, 2]);
    let attn = b.g().mm(qn, kt);
    let temp = b.take(wm, &format!("{prefix}.temperature"), &[heads, 1, 1])?;
    let temp = b.g().reshape_(temp, vec![1, heads as i64, 1, 1]);
    let attn = b.g().mul(attn, temp);
    let attn = b.g().sm(attn, -1);

    let out = b.g().mm(attn, v);
    // [1, heads, hd, N] → [1, N, heads, hd] → [1, N, C]
    let out = b.g().transpose_(out, vec![0, 3, 1, 2]);
    let attended = b.g().reshape_(out, vec![1, n as i64, c as i64]);

    let v_nchw = b.g().reshape_(v, vec![1, c as i64, h as i64, w as i64]);
    let conv_x = dwconv(b, wm, &format!("{prefix}.dwconv"), v_nchw, c)?;
    let attn_nchw = to_nchw(b, attended, h, w, c);
    let ch_map = channel_interaction(
        b,
        wm,
        &format!("{prefix}.channel_interaction"),
        attn_nchw,
        c,
    )?;
    let sp_map = spatial_interaction(b, wm, &format!("{prefix}.spatial_interaction"), conv_x, c)?;
    let sp_map = b.g().reshape_(sp_map, vec![1, n as i64, 1]);

    let sp_gate = b.sigmoid(sp_map);
    let attended = b.g().mul(attended, sp_gate);
    let ch_gate = b.sigmoid(ch_map);
    let conv_x = b.g().mul(conv_x, ch_gate);
    let conv_tokens = to_tokens(b, conv_x, h, w, c);

    let sum = b.g().add(attended, conv_tokens);
    b.linear(wm, &format!("{prefix}.proj"), sum, c, true)
}

/// `SGFN` — a feed-forward network whose hidden activations are gated by a
/// depthwise convolution of their own second half.
#[allow(clippy::too_many_arguments)]
fn sgfn(
    b: &mut Builder,
    wm: &mut WeightMap,
    prefix: &str,
    x: HirNodeId,
    h: usize,
    w: usize,
    c: usize,
    hidden: usize,
) -> Result<HirNodeId> {
    ensure!(
        hidden.is_multiple_of(2),
        "{prefix}: hidden width {hidden} must be even"
    );
    let half = hidden / 2;
    let t = b.linear(wm, &format!("{prefix}.fc1"), x, hidden, true)?;
    let t = b.gelu(t);

    let x1 = b.g().narrow_(t, 2, 0, half);
    let x2 = b.g().narrow_(t, 2, half, half);
    let x2 = b.layer_norm(wm, &format!("{prefix}.sg.norm"), x2, 1e-5)?;
    let x2 = to_nchw(b, x2, h, w, half);
    let x2 = b.conv(
        wm,
        &format!("{prefix}.sg.conv"),
        x2,
        half,
        [3, 3],
        [1, 1],
        [1, 1],
        half,
        true,
    )?;
    let x2 = to_tokens(b, x2, h, w, half);
    let gated = b.g().mul(x1, x2);
    let _ = c;
    b.linear(wm, &format!("{prefix}.fc2"), gated, c, true)
}

/// Whether the spatial block at `(group, block)` uses shifted windows.
///
/// The schedule has period 4 and is offset by the group's parity, so
/// consecutive groups do not shift in lockstep. Checkpoints register an
/// `attn_mask_0` / `attn_mask_1` buffer for exactly the shifted blocks, which
/// makes this independently checkable against real weights — see the tests.
fn is_shifted(group: usize, block: usize) -> bool {
    (group.is_multiple_of(2) && block > 0 && (block - 2).is_multiple_of(4))
        || (!group.is_multiple_of(2) && block.is_multiple_of(4))
}

/// Build DAT.
pub fn build(
    b: &mut Builder,
    wm: &mut WeightMap,
    cfg: &ModelConfig,
    p: &DatParams,
    x_in: HirNodeId,
) -> Result<HirNodeId> {
    let [_, _, h, w] = b.dims4(x_in);
    let c = p.embed_dim;
    ensure!(
        h % p.split_size[0] == 0
            && h % p.split_size[1] == 0
            && w % p.split_size[0] == 0
            && w % p.split_size[1] == 0,
        "a {h}×{w} tile does not divide by both window extents {:?}",
        p.split_size
    );

    let mean = b.constant("dat_mean", vec![0.4488, 0.4371, 0.4040], &[1, 3, 1, 1]);
    let centered = b.g().sub(x_in, mean);
    let range = b.scalar_like("dat_range", p.img_range, centered);
    let x = b.g().mul(centered, range);

    let shallow = b.conv3x3(wm, "conv_first", x, c)?;
    let mut tokens = to_tokens(b, shallow, h, w, c);
    tokens = b.layer_norm(wm, "before_RG.1", tokens, 1e-5)?;

    let hidden = (c as f32 * p.expansion_factor) as usize;
    for (gi, &depth) in p.depths.iter().enumerate() {
        let group_in = tokens;
        for bi in 0..depth {
            let prefix = format!("layers.{gi}.blocks.{bi}");
            let shortcut = tokens;
            let normed = b.layer_norm(wm, &format!("{prefix}.norm1"), tokens, 1e-5)?;
            let attended = if bi % 2 == 0 {
                let shifted = is_shifted(gi, bi);
                adaptive_spatial_attention(
                    b,
                    wm,
                    &format!("{prefix}.attn"),
                    normed,
                    h,
                    w,
                    c,
                    p.num_heads[gi],
                    p.split_size,
                    p.qkv_bias,
                    shifted,
                )
            } else {
                adaptive_channel_attention(
                    b,
                    wm,
                    &format!("{prefix}.attn"),
                    normed,
                    h,
                    w,
                    c,
                    p.num_heads[gi],
                    p.qkv_bias,
                )
            }
            .with_context(|| format!("DAT group {gi} block {bi}"))?;

            tokens = b.g().add(shortcut, attended);
            let n2 = b.layer_norm(wm, &format!("{prefix}.norm2"), tokens, 1e-5)?;
            let ff = sgfn(b, wm, &format!("{prefix}.ffn"), n2, h, w, c, hidden)?;
            tokens = b.g().add(tokens, ff);
        }

        let nchw = to_nchw(b, tokens, h, w, c);
        let conved = match p.resi_connection {
            ResiConnection::Identity => nchw,
            ResiConnection::Conv1 => b.conv3x3(wm, &format!("layers.{gi}.conv"), nchw, c)?,
            ResiConnection::Conv3 => {
                let t = b.conv3x3(wm, &format!("layers.{gi}.conv.0"), nchw, c / 4)?;
                let t = b.leaky_relu(t, 0.2);
                let t = b.conv1x1(wm, &format!("layers.{gi}.conv.2"), t, c / 4)?;
                let t = b.leaky_relu(t, 0.2);
                b.conv3x3(wm, &format!("layers.{gi}.conv.4"), t, c)?
            }
        };
        tokens = to_tokens(b, conved, h, w, c);
        tokens = b.g().add(group_in, tokens);
    }

    let tokens = b.layer_norm(wm, "norm", tokens, 1e-5)?;
    let deep = to_nchw(b, tokens, h, w, c);
    let body = b.conv3x3(wm, "conv_after_body", deep, c)?;
    let feat = b.g().add(body, shallow);

    let r = cfg.scale;
    let out = match p.upsampler {
        Upsampler::PixelShuffleDirect => {
            let up = b.conv3x3(wm, "upsample.0", feat, cfg.out_ch * r * r)?;
            b.pixel_shuffle(up, r)?
        }
        Upsampler::PixelShuffle => {
            let nf = 64;
            let h0 = b.conv3x3(wm, "conv_before_upsample.0", feat, nf)?;
            let mut h0 = b.leaky_relu(h0, 0.01);
            let stages: Vec<usize> = if r == 3 {
                vec![3]
            } else {
                ensure!(
                    r.is_power_of_two(),
                    "DAT's upsampler supports 2^n and 3, not ×{r}"
                );
                vec![2; r.trailing_zeros() as usize]
            };
            for (i, s) in stages.into_iter().enumerate() {
                h0 = b.conv3x3(wm, &format!("upsample.{}", i * 2), h0, nf * s * s)?;
                h0 = b.pixel_shuffle(h0, s)?;
            }
            b.conv3x3(wm, "conv_last", h0, cfg.out_ch)?
        }
        other => anyhow::bail!("DAT does not use the {other:?} upsampler"),
    };

    let inv = b.scalar_like("dat_inv_range", 1.0 / p.img_range, out);
    let scaled = b.g().mul(out, inv);
    let mean = b.constant("dat_mean_out", vec![0.4488, 0.4371, 0.4040], &[1, 3, 1, 1]);
    Ok(b.g().add(scaled, mean))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Zero displacement must land on the table's centre row, and every index
    /// must address a real row of the `(2sh−1)(2sw−1)` grid.
    #[test]
    fn rect_rpe_index_is_centred_and_in_range() {
        let (sh, sw) = (2usize, 4usize);
        let idx = rect_rpe_index(sh, sw);
        let n = sh * sw;
        let rows = (2 * sh - 1) * (2 * sw - 1);
        let centre = (sh - 1) * (2 * sw - 1) + (sw - 1);
        for i in 0..n {
            assert_eq!(idx[i * n + i], centre);
        }
        assert!(idx.iter().all(|&v| v < rows));
    }

    /// The displacement grid must line up with the index: row `k` of
    /// `rpe_biases` is the displacement that `rect_rpe_index` maps to `k`.
    #[test]
    fn rpe_biases_matches_the_index_layout() {
        let (sh, sw) = (2usize, 4usize);
        let biases = rpe_biases(sh, sw);
        let idx = rect_rpe_index(sh, sw);
        let n = sh * sw;
        for i in 0..n {
            for j in 0..n {
                let row = idx[i * n + j];
                let (dy, dx) = (biases[row * 2], biases[row * 2 + 1]);
                assert_eq!(dy, (i / sw) as f32 - (j / sw) as f32);
                assert_eq!(dx, (i % sw) as f32 - (j % sw) as f32);
            }
        }
    }

    /// A window wholly inside the image spans one region and is unmasked; the
    /// wrapped windows at the far edge are not.
    #[test]
    fn rect_shift_mask_blocks_only_wrapped_windows() {
        let r = Rect::new(8, 8, 2, 4).unwrap();
        let m = rect_shift_mask(r, 1, 2);
        assert_eq!(m.len(), r.nw * r.n * r.n);
        assert!(m[..r.n * r.n].iter().all(|&v| v == 0.0));
        assert!(m[(r.nw - 1) * r.n * r.n..].iter().any(|&v| v == -100.0));
    }

    /// Pinned against a release DAT-2 checkpoint (`4xBHI_dat2_otf_nn`, six
    /// groups of six): it registers `attn_mask_0`/`_1` for precisely the shifted
    /// blocks, and those are `layers.0.blocks.2`, `layers.1.blocks.0` and
    /// `layers.1.blocks.4`. Only even blocks are spatial, so only they can shift.
    #[test]
    fn shift_schedule_matches_a_real_checkpoints_mask_buffers() {
        let shifted: Vec<(usize, usize)> = (0..2)
            .flat_map(|g| (0..6).map(move |b| (g, b)))
            .filter(|&(g, b)| b % 2 == 0 && is_shifted(g, b))
            .collect();
        assert_eq!(shifted, vec![(0, 2), (1, 0), (1, 4)]);
    }

    /// The two branches use transposed windows; both must tile the same map.
    #[test]
    fn both_window_orientations_must_tile() {
        assert!(Rect::new(32, 32, 8, 32).is_ok());
        assert!(Rect::new(32, 32, 32, 8).is_ok());
        assert!(Rect::new(16, 32, 32, 8).is_err());
    }
}
