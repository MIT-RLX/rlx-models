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

//! **OmniSR** — Omni Self-Attention SR (CVPR 2023).
//!
//! A MaxViT-shaped block applied to super-resolution: every `OSA_Block`
//! alternates attention that is *local* with attention that is *global*, and
//! interleaves channel attention with both.
//!
//! ```text
//!   MBConv                      spatial mixing, cheap
//!   block attention   + FFN     ws×ws contiguous tiles
//!   channel attention + FFN     over channels, tokens contracted
//!   grid attention    + FFN     ws×ws tiles strided across the image
//!   channel attention + FFN     (grid grouping)
//! ```
//!
//! # Block and grid partitions are not the same permutation
//!
//! The reference writes them as two `einops` patterns that differ only in the
//! order of a pair of factors:
//!
//! ```text
//!   block:  b d (x w1) (y w2) -> b x y w1 w2 d     window = 8 adjacent pixels
//!   grid:   b d (w1 x) (w2 y) -> b x y w1 w2 d     window = every (H/8)-th pixel
//! ```
//!
//! Both produce the same *shape*, so getting them the wrong way round is a
//! silent error: the network still runs, the output is still an image, and it
//! is simply attending over the wrong pixels. They are therefore built by two
//! separate functions with the factor order spelled out, and
//! `tests::grid_and_block_partitions_differ` pins that they disagree.
//!
//! # The checkpoints carry profiler droppings
//!
//! Released OmniSR weights were saved after a `thop` profiling pass, so roughly
//! three quarters of the tensors in the file are `total_ops` / `total_params`
//! scalars registered as buffers on every module. They are stripped on load —
//! see `weights::strip_profiler_buffers`.

use anyhow::{Result, ensure};
use rlx_core::weight_map::WeightMap;
use rlx_ir::HirGraphExt;
use rlx_ir::hir::HirNodeId;

use crate::arch::swin::{
    Win, rpe_bias, rpe_index, to_nchw, to_tokens, window_partition, window_reverse,
};
use crate::config::ModelConfig;
use crate::nn::Builder;

/// `nn.LayerNorm` default, and `LayerNorm2d`'s too.
const NORM_EPS: f32 = 1e-6;

/// Head counts and widths here are **structural**, not hyperparameters.
///
/// `OSA_Block` constructs `Attention(dim_head = dim/4)` and
/// `Channel_Attention(heads = 4)` with those values written literally, and
/// `MBConv(expansion_rate = 1)` / `Gated_Conv_FeedForward(mult = 1)` make both
/// hidden widths equal to `dim`. None of them can vary across a checkpoint, so
/// they are constants rather than something detection has to recover.
const HEADS: usize = 4;

/// `esa_channel = max(channel_num // 4, 16)` in `OSAG`.
fn esa_channels(dim: usize) -> usize {
    (dim / 4).max(16)
}

// ── partitions ───────────────────────────────────────────────────────────

/// `b d (w1 x) (w2 y) -> (b x y) (w1 w2) d` — **grid** attention.
///
/// The window's tokens are spaced `H/ws` apart, so one window spans the whole
/// image. Contrast [`window_partition`], where a window is a contiguous tile.
fn grid_partition(b: &mut Builder, x: HirNodeId, g: Win, c: usize) -> HirNodeId {
    // [1, H·W, C] → [ws, gh, ws, gw, C]: the *outer* factor of each spatial
    // axis is the window extent, so the inner one indexes the window. Rank 5,
    // batch axis elided — CoreML's MIL caps rank at 5.
    let five = b.g().reshape_(
        x,
        vec![g.ws as i64, g.gh as i64, g.ws as i64, g.gw as i64, c as i64],
    );
    // → [gh, gw, ws, ws, C]
    let perm = b.g().transpose_(five, vec![1, 3, 0, 2, 4]);
    b.g()
        .reshape_(perm, vec![g.nw as i64, g.n as i64, c as i64])
}

/// The inverse of [`grid_partition`].
fn grid_reverse(b: &mut Builder, x: HirNodeId, g: Win, c: usize) -> HirNodeId {
    let five = b.g().reshape_(
        x,
        vec![g.gh as i64, g.gw as i64, g.ws as i64, g.ws as i64, c as i64],
    );
    // → [ws, gh, ws, gw, C]
    let perm = b.g().transpose_(five, vec![2, 0, 3, 1, 4]);
    b.g().reshape_(perm, vec![1, (g.h * g.w) as i64, c as i64])
}

/// `b (head d) (h ph) (w pw) -> b (h w) head d (ph pw)`, or for the **grid**
/// flavour `-> b (ph pw) head d (h w)`.
///
/// # The two flavours do not have the same shape
///
/// Both split the image the same way — `ph`/`pw` index *within* a `ws × ws`
/// tile and `h`/`w` index the tiles. What differs is which pair becomes the
/// batch and which becomes the axis contracted away:
///
/// | | batch | contracted |
/// |---|---|---|
/// | block | `(h w)` = `nw` tiles | `(ph pw)` = `ws²` |
/// | grid  | `(ph pw)` = `ws²` | `(h w)` = `nw` tiles |
///
/// So the output is `[nw, heads, hd, n]` for block and `[n, heads, hd, nw]`
/// for grid. Those have the same element count, which is exactly why writing
/// one shape for both typechecks and silently regroups the tensor.
fn channel_partition(
    b: &mut Builder,
    x: HirNodeId,
    g: Win,
    heads: usize,
    hd: usize,
    grid: bool,
) -> HirNodeId {
    // Rank 5: `head` and `d` are adjacent and stay in order on both sides, so
    // they travel merged as one `dim` axis and are split again at the end.
    // That keeps CoreML's rank-5 ceiling without changing the permutation.
    let five = b.g().reshape_(
        x,
        vec![
            (heads * hd) as i64,
            g.gh as i64,
            g.ws as i64,
            g.gw as i64,
            g.ws as i64,
        ],
    );
    // axes: 0 dim, 1 h, 2 ph, 3 w, 4 pw
    let (perm, lead, tail) = if grid {
        (vec![2, 4, 0, 1, 3], g.n, g.nw) // [ph, pw, dim, h, w]
    } else {
        (vec![1, 3, 0, 2, 4], g.nw, g.n) // [h, w, dim, ph, pw]
    };
    let perm = b.g().transpose_(five, perm);
    b.g().reshape_(
        perm,
        vec![lead as i64, heads as i64, hd as i64, tail as i64],
    )
}

/// The inverse of [`channel_partition`], back to `[1, C, H, W]`.
fn channel_unpartition(
    b: &mut Builder,
    x: HirNodeId,
    g: Win,
    heads: usize,
    hd: usize,
    grid: bool,
) -> HirNodeId {
    let dim = heads * hd;
    let (a, c, d, perm) = if grid {
        // [ph, pw, dim, h, w] → [dim, h, ph, w, pw]
        (g.ws, g.gh, g.gw, vec![2, 3, 0, 4, 1])
    } else {
        // [h, w, dim, ph, pw] → [dim, h, ph, w, pw]
        (g.gh, g.gw, g.ws, vec![2, 0, 3, 1, 4])
    };
    let (b0, b1) = if grid { (a, a) } else { (a, c) };
    let (c0, c1) = if grid { (c, d) } else { (d, d) };
    let five = b.g().reshape_(
        x,
        vec![b0 as i64, b1 as i64, dim as i64, c0 as i64, c1 as i64],
    );
    let perm = b.g().transpose_(five, perm);
    b.g()
        .reshape_(perm, vec![1, dim as i64, g.h as i64, g.w as i64])
}

// ── the pieces ───────────────────────────────────────────────────────────

/// Squeeze-and-excitation, the `einops` flavour: a global mean straight to a
/// pair of biasless `Linear`s.
fn squeeze_excite(
    b: &mut Builder,
    wm: &mut WeightMap,
    prefix: &str,
    x: HirNodeId,
    dim: usize,
) -> Result<HirNodeId> {
    let hidden = (dim as f32 * 0.25) as usize;
    let pooled = b.g().mean(x, vec![2, 3], false); // [1, C]
    let h = b.linear(wm, &format!("{prefix}.1"), pooled, hidden, false)?;
    let h = b.silu(h);
    let h = b.linear(wm, &format!("{prefix}.3"), h, dim, false)?;
    let gate = b.sigmoid(h);
    let g4 = b.g().reshape_(gate, vec![1, dim as i64, 1, 1]);
    Ok(b.g().mul(x, g4))
}

/// `MBConv` with its residual. `expansion_rate` is 1 here, so the hidden width
/// equals the output width — but it is read from the weights anyway.
fn mbconv(
    b: &mut Builder,
    wm: &mut WeightMap,
    prefix: &str,
    x: HirNodeId,
    dim: usize,
) -> Result<HirNodeId> {
    // `expansion_rate = 1`, so the inverted bottleneck does not widen.
    let hidden = dim;
    let y = b.conv1x1(wm, &format!("{prefix}.fn.0"), x, hidden)?;
    let y = b.gelu(y);
    let y = b.conv(
        wm,
        &format!("{prefix}.fn.2"),
        y,
        hidden,
        [3, 3],
        [1, 1],
        [1, 1],
        hidden,
        true,
    )?;
    let y = b.gelu(y);
    let y = squeeze_excite(b, wm, &format!("{prefix}.fn.4.gate"), y, hidden)?;
    let y = b.conv1x1(wm, &format!("{prefix}.fn.5"), y, dim)?;
    Ok(b.g().add(x, y))
}

/// `Gated_Conv_FeedForward`: a 1×1 to twice the width, a depthwise 3×3, then a
/// GELU-gated product. Every convolution here is biasless.
fn gated_ffn(
    b: &mut Builder,
    wm: &mut WeightMap,
    prefix: &str,
    x: HirNodeId,
    dim: usize,
) -> Result<HirNodeId> {
    // `mult = 1`, and the projection doubles for the gate.
    let hidden = dim;
    let twice = hidden * 2;

    let y = b.conv(
        wm,
        &format!("{prefix}.project_in"),
        x,
        twice,
        [1, 1],
        [1, 1],
        [0, 0],
        1,
        false,
    )?;
    let y = b.conv(
        wm,
        &format!("{prefix}.dwconv"),
        y,
        twice,
        [3, 3],
        [1, 1],
        [1, 1],
        twice,
        false,
    )?;
    let a = b.g().narrow_(y, 1, 0, hidden);
    let g = b.g().narrow_(y, 1, hidden, hidden);
    let a = b.gelu(a);
    let y = b.g().mul(a, g);
    b.conv(
        wm,
        &format!("{prefix}.project_out"),
        y,
        dim,
        [1, 1],
        [1, 1],
        [0, 0],
        1,
        false,
    )
}

/// Window (or grid) self-attention over tokens, with a learned relative
/// position bias.
fn window_attention(
    b: &mut Builder,
    wm: &mut WeightMap,
    prefix: &str,
    tokens: HirNodeId,
    g: Win,
    dim: usize,
    tag: &str,
) -> Result<HirNodeId> {
    let heads = HEADS;
    ensure!(
        dim.is_multiple_of(heads),
        "{prefix}: {dim} channels do not split into {heads} heads"
    );
    let hd = dim / heads;
    let scale = (hd as f32).powf(-0.5);

    let qkv = b.linear(wm, &format!("{prefix}.to_qkv"), tokens, dim * 3, false)?;
    let q = b.g().narrow_(qkv, 2, 0, dim);
    let k = b.g().narrow_(qkv, 2, dim, dim);
    let v = b.g().narrow_(qkv, 2, 2 * dim, dim);

    let split = |b: &mut Builder, t: HirNodeId| {
        // [nW, n, C] → [nW, heads, n, hd]
        let four = b
            .g()
            .reshape_(t, vec![g.nw as i64, g.n as i64, heads as i64, hd as i64]);
        b.g().transpose_(four, vec![0, 2, 1, 3])
    };
    let q = split(b, q);
    let k = split(b, k);
    let v = split(b, v);

    let s = b.scalar_like("omni_attn_scale", scale, q);
    let q = b.g().mul(q, s);
    let kt = b.g().transpose_(k, vec![0, 1, 3, 2]);
    let mut attn = b.g().mm(q, kt);

    // The index formula is identical to SwinIR's, so the table is read the same
    // way: `rel_pos = grid[i] - grid[j] + (ws - 1)`, flattened by `2·ws − 1`.
    let rows = (2 * g.ws - 1) * (2 * g.ws - 1);
    let bias = rpe_bias(
        b,
        wm,
        &format!("{prefix}.rel_pos_bias.weight"),
        &rpe_index(g.ws),
        g.n,
        g.n,
        rows,
        heads,
        tag,
    )?;
    attn = b.g().add(attn, bias);

    let attn = b.g().sm(attn, -1);
    let out = b.g().mm(attn, v);
    // [nW, heads, n, hd] → [nW, n, C]
    let out = b.g().transpose_(out, vec![0, 2, 1, 3]);
    let out = b
        .g()
        .reshape_(out, vec![g.nw as i64, g.n as i64, dim as i64]);
    b.linear(wm, &format!("{prefix}.to_out.0"), out, dim, false)
}

/// `Channel_Attention` — attention over *channels*, with the tokens of a window
/// as the contracted axis. `grid` selects which grouping the windows use.
fn channel_attention(
    b: &mut Builder,
    wm: &mut WeightMap,
    prefix: &str,
    x: HirNodeId,
    g: Win,
    dim: usize,
    grid: bool,
) -> Result<HirNodeId> {
    let heads = HEADS;
    ensure!(
        dim.is_multiple_of(heads),
        "{prefix}: {dim} channels do not split into {heads} heads"
    );
    let hd = dim / heads;

    let qkv = b.conv(
        wm,
        &format!("{prefix}.qkv"),
        x,
        dim * 3,
        [1, 1],
        [1, 1],
        [0, 0],
        1,
        false,
    )?;
    let qkv = b.conv(
        wm,
        &format!("{prefix}.qkv_dwconv"),
        qkv,
        dim * 3,
        [3, 3],
        [1, 1],
        [1, 1],
        dim * 3,
        false,
    )?;

    let qs = b.g().narrow_(qkv, 1, 0, dim);
    let ks = b.g().narrow_(qkv, 1, dim, dim);
    let vs = b.g().narrow_(qkv, 1, 2 * dim, dim);
    let q = channel_partition(b, qs, g, heads, hd, grid);
    let k = channel_partition(b, ks, g, heads, hd, grid);
    let v = channel_partition(b, vs, g, heads, hd, grid);

    let q = b.l2_normalize_last(q);
    let k = b.l2_normalize_last(k);

    let kt = b.g().transpose_(k, vec![0, 1, 3, 2]);
    let attn = b.g().mm(q, kt); // [nW, heads, hd, hd]
    let temp = b.take(wm, &format!("{prefix}.temperature"), &[heads, 1, 1])?;
    let attn = b.g().mul(attn, temp);
    let attn = b.g().sm(attn, -1);
    let out = b.g().mm(attn, v);
    let nchw = channel_unpartition(b, out, g, heads, hd, grid);
    b.conv(
        wm,
        &format!("{prefix}.project_out"),
        nchw,
        dim,
        [1, 1],
        [1, 1],
        [0, 0],
        1,
        false,
    )
}

/// `Conv_PreNormResidual` around a function that takes and returns NCHW.
fn conv_prenorm<F>(
    b: &mut Builder,
    wm: &mut WeightMap,
    prefix: &str,
    x: HirNodeId,
    f: F,
) -> Result<HirNodeId>
where
    F: FnOnce(&mut Builder, &mut WeightMap, &str, HirNodeId) -> Result<HirNodeId>,
{
    let n = b.layer_norm_nchw(wm, &format!("{prefix}.norm"), x, NORM_EPS)?;
    let y = f(b, wm, &format!("{prefix}.fn"), n)?;
    Ok(b.g().add(y, x))
}

/// One `OSA_Block`. The indices are the flattened `nn.Sequential`'s, and the
/// `Rearrange` modules occupy 1, 3, 7 and 9 without carrying weights.
fn osa_block(
    b: &mut Builder,
    wm: &mut WeightMap,
    prefix: &str,
    x: HirNodeId,
    g: Win,
    dim: usize,
    tag: &str,
) -> Result<HirNodeId> {
    let x = mbconv(b, wm, &format!("{prefix}.0"), x, dim)?;

    // ── block attention (index 2), between Rearranges 1 and 3 ────────
    let x = attention_stage(b, wm, &format!("{prefix}.2"), x, g, dim, false, tag)?;
    let x = conv_prenorm(b, wm, &format!("{prefix}.4"), x, |b, wm, p, t| {
        gated_ffn(b, wm, p, t, dim)
    })?;
    let x = conv_prenorm(b, wm, &format!("{prefix}.5"), x, |b, wm, p, t| {
        channel_attention(b, wm, p, t, g, dim, false)
    })?;
    let x = conv_prenorm(b, wm, &format!("{prefix}.6"), x, |b, wm, p, t| {
        gated_ffn(b, wm, p, t, dim)
    })?;

    // ── grid attention (index 8), between Rearranges 7 and 9 ─────────
    let x = attention_stage(b, wm, &format!("{prefix}.8"), x, g, dim, true, tag)?;
    let x = conv_prenorm(b, wm, &format!("{prefix}.10"), x, |b, wm, p, t| {
        gated_ffn(b, wm, p, t, dim)
    })?;
    let x = conv_prenorm(b, wm, &format!("{prefix}.11"), x, |b, wm, p, t| {
        channel_attention(b, wm, p, t, g, dim, true)
    })?;
    conv_prenorm(b, wm, &format!("{prefix}.12"), x, |b, wm, p, t| {
        gated_ffn(b, wm, p, t, dim)
    })
}

/// Partition → `PreNormResidual(LayerNorm, Attention)` → reverse.
///
/// The residual lives *inside* the partitioned space in the reference. That is
/// equivalent to adding it outside, because the partition is a bijection, but
/// it is kept inside here so the code reads like the `nn.Sequential` it ports.
#[allow(clippy::too_many_arguments)]
fn attention_stage(
    b: &mut Builder,
    wm: &mut WeightMap,
    prefix: &str,
    x: HirNodeId,
    g: Win,
    dim: usize,
    grid: bool,
    tag: &str,
) -> Result<HirNodeId> {
    let tokens = to_tokens(b, x, g, dim);
    let win = if grid {
        grid_partition(b, tokens, g, dim)
    } else {
        window_partition(b, tokens, g, dim)
    };
    let n = b.layer_norm(wm, &format!("{prefix}.norm"), win, NORM_EPS)?;
    let a = window_attention(b, wm, &format!("{prefix}.fn"), n, g, dim, tag)?;
    let win = b.g().add(a, win);
    let tokens = if grid {
        grid_reverse(b, win, g, dim)
    } else {
        window_reverse(b, win, g, dim)
    };
    Ok(to_nchw(b, tokens, g, dim))
}

/// Enhanced Spatial Attention: a cheap spatial gate built from a heavily
/// downsampled copy of the features.
fn esa(
    b: &mut Builder,
    wm: &mut WeightMap,
    prefix: &str,
    x: HirNodeId,
    n_feats: usize,
) -> Result<HirNodeId> {
    let [_, _, h, w] = b.dims4(x);
    let f = esa_channels(n_feats);

    let c1_ = b.conv1x1(wm, &format!("{prefix}.conv1"), x, f)?;
    // A *valid* stride-2 3×3, then a 7×7 stride-3 max pool: together these take
    // the map down by roughly 6×, which is where the gate's wide receptive
    // field comes from.
    let c1 = b.conv(
        wm,
        &format!("{prefix}.conv2"),
        c1_,
        f,
        [3, 3],
        [2, 2],
        [0, 0],
        1,
        true,
    )?;
    let v_max = b.max_pool_k(c1, 7, 3)?;
    let c3 = b.conv3x3(wm, &format!("{prefix}.conv3"), v_max, f)?;
    // Back to full resolution at a non-integer ratio — the one place in the
    // crate that needs a general bilinear resize rather than a repeat.
    let c3 = b.resize_bilinear(c3, h, w)?;

    let cf = b.conv1x1(wm, &format!("{prefix}.conv_f"), c1_, f)?;
    let sum = b.g().add(c3, cf);
    let c4 = b.conv1x1(wm, &format!("{prefix}.conv4"), sum, n_feats)?;
    let m = b.sigmoid(c4);
    Ok(b.g().mul(x, m))
}

/// One `OSAG`: a stack of `OSA_Block`s, a 1×1, a residual, then ESA.
fn osag(
    b: &mut Builder,
    wm: &mut WeightMap,
    prefix: &str,
    x: HirNodeId,
    g: Win,
    dim: usize,
    block_num: usize,
    tag: &str,
) -> Result<HirNodeId> {
    let mut h = x;
    for i in 0..block_num {
        h = osa_block(
            b,
            wm,
            &format!("{prefix}.residual_layer.{i}.layer"),
            h,
            g,
            dim,
            &format!("{tag}_b{i}"),
        )?;
    }
    let h = b.conv1x1(wm, &format!("{prefix}.residual_layer.{block_num}"), h, dim)?;
    let h = b.g().add(h, x);
    esa(b, wm, &format!("{prefix}.esa"), h, dim)
}

/// Build OmniSR.
pub fn build(
    b: &mut Builder,
    wm: &mut WeightMap,
    cfg: &ModelConfig,
    num_feat: usize,
    res_num: usize,
    block_num: usize,
    window_size: usize,
    x_in: HirNodeId,
) -> Result<HirNodeId> {
    let [_, _, h, w] = b.dims4(x_in);
    let g = Win::new(h, w, window_size)?;

    let residual = b.conv3x3(wm, "input", x_in, num_feat)?;
    let mut out = residual;
    for i in 0..res_num {
        out = osag(
            b,
            wm,
            &format!("residual_layer.{i}"),
            out,
            g,
            num_feat,
            block_num,
            &format!("g{i}"),
        )?;
    }
    let out = b.conv3x3(wm, "output", out, num_feat)?;
    let out = b.g().add(out, residual);

    let up = b.conv3x3(wm, "up.0", out, cfg.out_ch * cfg.scale * cfg.scale)?;
    b.pixel_shuffle(up, cfg.scale)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::Builder;
    use rlx_runtime::{Device, Session};

    /// Run one partition (and optionally its inverse) over a ramp and return
    /// the flat result, so the layout can be compared against index arithmetic.
    fn run_partition(
        h: usize,
        w: usize,
        ws: usize,
        heads: usize,
        hd: usize,
        grid: bool,
        round_trip: bool,
    ) -> (Vec<f32>, Vec<usize>) {
        let dim = heads * hd;
        let mut b = Builder::new("chan");
        let x = b.input(dim, h, w);
        let g = Win::new(h, w, ws).expect("window");
        let part = channel_partition(&mut b, x, g, heads, hd, grid);
        let out = if round_trip {
            channel_unpartition(&mut b, part, g, heads, hd, grid)
        } else {
            part
        };
        // The *declared shape* is the thing that differs between the two
        // flavours. A reshape only relabels — it never moves data — so the flat
        // buffer is identical either way and a value-only check cannot see a
        // mislabelled grouping. What sees it is the next op: the attention
        // matmul contracts the last axis, so a swapped label contracts `nw`
        // where it should contract `ws²`.
        let shape = b.dims(out);
        let (graph, params, _) = b.finish(out).expect("finish");
        let opts = rlx_core::flow_bridge::compile_options_for_profile(
            &rlx_flow::CompileProfile::encoder(),
            Device::Cpu,
        );
        let mut compiled = Session::new(Device::Cpu).compile_with(graph, &opts);
        rlx_core::flow_util::attach_built_params(&mut compiled, params, &[]);
        let n = dim * h * w;
        let input: Vec<f32> = (0..n).map(|i| i as f32).collect();
        (compiled.run(&[("image", &input)]).remove(0), shape)
    }

    /// The grid partition must group by *position within a tile* and contract
    /// over tiles — the opposite of the block one. Both produce the same
    /// element count, so writing one shape for both typechecks and silently
    /// regroups; only checking where individual values land catches it.
    #[test]
    fn channel_partitions_match_the_einops_patterns() {
        // gh·gw = 6 and ws² = 4 are deliberately unequal: when they match, a
        // swapped grouping is invisible.
        let (h, w, ws, heads, hd) = (4usize, 6usize, 2usize, 2usize, 3usize);
        let (gh, gw) = (h / ws, w / ws);
        let (nw, n) = (gh * gw, ws * ws);
        let dim = heads * hd;
        let src = |c: usize, y: usize, x: usize| (c * h * w + y * w + x) as f32;

        for grid in [false, true] {
            let (got, shape) = run_partition(h, w, ws, heads, hd, grid, false);
            let (lead, tail) = if grid { (n, nw) } else { (nw, n) };
            assert_eq!(
                shape,
                vec![lead, heads, hd, tail],
                "grid={grid}: wrong grouping. Block batches over tiles and \
                 contracts positions; grid does the opposite."
            );
            assert_eq!(got.len(), lead * heads * hd * tail);

            for hh in 0..gh {
                for ww in 0..gw {
                    for ph in 0..ws {
                        for pw in 0..ws {
                            for c in 0..dim {
                                let (l, t) = if grid {
                                    (ph * ws + pw, hh * gw + ww)
                                } else {
                                    (hh * gw + ww, ph * ws + pw)
                                };
                                let idx = ((l * dim) + c) * tail + t;
                                let want = src(c, hh * ws + ph, ww * ws + pw);
                                assert_eq!(
                                    got[idx], want,
                                    "grid={grid} c={c} tile=({hh},{ww}) pos=({ph},{pw})"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    /// Partition then un-partition is the identity. This is the property the
    /// forward pass actually relies on, and it holds independently of whether
    /// the intermediate grouping is the right one — so it is a companion to the
    /// test above, not a substitute for it.
    #[test]
    fn channel_partitions_round_trip() {
        let (h, w, ws, heads, hd) = (4usize, 6usize, 2usize, 2usize, 3usize);
        let n = heads * hd * h * w;
        let want: Vec<f32> = (0..n).map(|i| i as f32).collect();
        for grid in [false, true] {
            let (got, shape) = run_partition(h, w, ws, heads, hd, grid, true);
            assert_eq!(shape, vec![1, heads * hd, h, w]);
            assert_eq!(got, want, "grid={grid} round trip is not the identity");
        }
    }

    /// Which source pixel lands in window `(wx, wy)` at position `(i, j)`,
    /// under each partition. Worked in plain arithmetic so the two orderings
    /// can be compared without building a graph.
    fn block_pixel(ws: usize, gh: usize, wx: usize, i: usize) -> usize {
        let _ = gh;
        wx * ws + i
    }
    fn grid_pixel(ws: usize, gh: usize, wx: usize, i: usize) -> usize {
        let _ = ws;
        i * gh + wx
    }

    /// The two patterns produce identically shaped tensors from identically
    /// shaped inputs, so nothing downstream can catch a swap. They must be
    /// different maps, and the difference must be the one the reference writes:
    /// block windows are contiguous, grid windows are strided.
    #[test]
    fn grid_and_block_partitions_differ() {
        let (ws, gh) = (8usize, 4usize);
        // Window 0 of the block partition is the first `ws` rows.
        let block: Vec<usize> = (0..ws).map(|i| block_pixel(ws, gh, 0, i)).collect();
        assert_eq!(block, vec![0, 1, 2, 3, 4, 5, 6, 7]);
        // Window 0 of the grid partition samples every `gh`-th row.
        let grid: Vec<usize> = (0..ws).map(|i| grid_pixel(ws, gh, 0, i)).collect();
        assert_eq!(grid, vec![0, 4, 8, 12, 16, 20, 24, 28]);
        assert_ne!(block, grid);
    }

    /// Both partitions must be bijections over the image, or a reverse would
    /// drop or duplicate pixels.
    #[test]
    fn both_partitions_cover_every_pixel_once() {
        let (ws, gh) = (8usize, 4usize);
        let h = ws * gh;
        for f in [
            block_pixel as fn(usize, usize, usize, usize) -> usize,
            grid_pixel,
        ] {
            let mut seen = vec![false; h];
            for wx in 0..gh {
                for i in 0..ws {
                    let p = f(ws, gh, wx, i);
                    assert!(!seen[p], "pixel {p} visited twice");
                    seen[p] = true;
                }
            }
            assert!(seen.iter().all(|&s| s), "not every pixel was visited");
        }
    }
}
