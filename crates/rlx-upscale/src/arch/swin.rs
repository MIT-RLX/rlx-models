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

//! The Swin lineage: **SwinIR**, **HAT** and **DRCT**.
//!
//! All three are the same backbone — shifted-window attention over a
//! convolutional stem, in residual groups each closed by a convolution — and
//! differ only in what goes inside a group:
//!
//! | | group body | extra per block |
//! |---|---|---|
//! | SwinIR | `depth` Swin blocks | — |
//! | HAT | `depth` blocks + one overlapping-cross-attention block | a channel-attention conv branch |
//! | DRCT | five densely-connected blocks | 1×1 `adjust` after each |
//!
//! # Everything static is precomputed on the host
//!
//! Two tensors dominate the naive implementation and neither depends on the
//! input: the relative-position bias (a *gather* of a learned table through a
//! fixed index) and the shifted-window attention mask (a `0`/`−100` pattern
//! fixed by the tile geometry). At inference the table is frozen, so the gather
//! is performed once at build time in `rpe_bias` and injected as a constant.
//! That removes a `Gather` from the hot path of every block and keeps the mask
//! at `[1, nW, 1, N, N]` — broadcast over heads — rather than materializing the
//! `[B·nW, nH, N, N]` bias a fused-attention op would demand. For a 192×192
//! tile with 6 heads and window 16 that is the difference between 590 KB and
//! 340 MB.
//!
//! # Attention is written out, not fused
//!
//! `Op::Attention`'s `MaskKind::Bias` wants a full `[batch, heads, q, k]`
//! tensor, which is exactly the tensor the point above is avoiding. The
//! explicit `q·kᵀ → +bias → softmax → ·v` form broadcasts the two small
//! constants instead, and is what the reference computes anyway.
//!
//! # A note on CPU cost
//!
//! Window partition/reverse swaps two *middle* axes, so it misses rlx-cpu's
//! last-two-axis fast path and falls to the
//! general N-D index walk. There are two per block. This is the main reason
//! tier-2 models want a GPU backend; the tile default is sized accordingly.

use anyhow::{Context, Result, ensure};
use rlx_core::weight_map::WeightMap;
use rlx_ir::HirGraphExt;
use rlx_ir::hir::HirNodeId;
use rlx_ir::op::PadMode;

use crate::config::{Arch, ModelConfig, ResiConnection, SwinParams, Upsampler};
use crate::nn::Builder;

/// Geometry of one windowed feature map.
/// `cpb_mlp` is `Linear(2, 512) → ReLU → Linear(512, nH)`; the 512 is written
/// literally in the reference and cannot vary.
const CPB_HIDDEN: usize = 512;

#[derive(Clone, Copy)]
pub(crate) struct Win {
    pub(crate) h: usize,
    pub(crate) w: usize,
    pub(crate) ws: usize,
    /// Windows down and across.
    pub(crate) gh: usize,
    pub(crate) gw: usize,
    /// Tokens per window (`ws²`).
    pub(crate) n: usize,
    /// Windows per image (`gh · gw`).
    pub(crate) nw: usize,
}

impl Win {
    pub(crate) fn new(h: usize, w: usize, ws: usize) -> Result<Self> {
        ensure!(
            h.is_multiple_of(ws) && w.is_multiple_of(ws),
            "a {h}×{w} feature map does not partition into {ws}×{ws} windows"
        );
        Ok(Self {
            h,
            w,
            ws,
            gh: h / ws,
            gw: w / ws,
            n: ws * ws,
            nw: (h / ws) * (w / ws),
        })
    }
}

/// `[1, H·W, C]` → `[nW, ws², C]`.
///
/// Written at **rank 5**, without the leading unit batch axis the natural form
/// would carry. CoreML's MIL rejects anything above rank 5 and window
/// partitioning is the single most frequent shape op in the Swin family, so a
/// rank-6 spelling costs the whole backend for no gain.
pub(crate) fn window_partition(b: &mut Builder, x: HirNodeId, g: Win, c: usize) -> HirNodeId {
    let five = b.g().reshape_(
        x,
        vec![g.gh as i64, g.ws as i64, g.gw as i64, g.ws as i64, c as i64],
    );
    let perm = b.g().transpose_(five, vec![0, 2, 1, 3, 4]);
    b.g()
        .reshape_(perm, vec![g.nw as i64, g.n as i64, c as i64])
}

/// `[nW, ws², C]` → `[1, H·W, C]`. The permutation is its own inverse.
pub(crate) fn window_reverse(b: &mut Builder, x: HirNodeId, g: Win, c: usize) -> HirNodeId {
    let five = b.g().reshape_(
        x,
        vec![g.gh as i64, g.gw as i64, g.ws as i64, g.ws as i64, c as i64],
    );
    let perm = b.g().transpose_(five, vec![0, 2, 1, 3, 4]);
    b.g().reshape_(perm, vec![1, (g.h * g.w) as i64, c as i64])
}

/// `torch.roll(x, shifts, dims=(1, 2))` on a `[1, H, W, C]` view, composed from
/// two narrow/concat pairs so no `Roll` op is required.
pub(crate) fn roll(b: &mut Builder, x: HirNodeId, g: Win, c: usize, shift: isize) -> HirNodeId {
    if shift == 0 {
        return x;
    }
    let four = b.g().reshape_(x, vec![1, g.h as i64, g.w as i64, c as i64]);
    let sh = shift.rem_euclid(g.h as isize) as usize;
    let sw = shift.rem_euclid(g.w as isize) as usize;

    // roll(+s) moves element i to i+s, i.e. the last s rows come first.
    let rolled_h = if sh == 0 {
        four
    } else {
        let tail = b.g().narrow_(four, 1, g.h - sh, sh);
        let head = b.g().narrow_(four, 1, 0, g.h - sh);
        b.g().concat_(vec![tail, head], 1)
    };
    let rolled = if sw == 0 {
        rolled_h
    } else {
        let tail = b.g().narrow_(rolled_h, 2, g.w - sw, sw);
        let head = b.g().narrow_(rolled_h, 2, 0, g.w - sw);
        b.g().concat_(vec![tail, head], 2)
    };
    b.g()
        .reshape_(rolled, vec![1, (g.h * g.w) as i64, c as i64])
}

/// Relative-position index for a square window: `index[i, j]` selects the bias
/// table row for the displacement between tokens `i` and `j`.
///
/// Always lands in `[0, (2W−1)²)`.
pub(crate) fn rpe_index(ws: usize) -> Vec<isize> {
    let n = ws * ws;
    let mut idx = vec![0isize; n * n];
    for i in 0..n {
        let (ih, iw) = (i / ws, i % ws);
        for j in 0..n {
            let (jh, jw) = (j / ws, j % ws);
            let dh = ih as isize - jh as isize + ws as isize - 1;
            let dw = iw as isize - jw as isize + ws as isize - 1;
            idx[i * n + j] = dh * (2 * ws as isize - 1) + dw;
        }
    }
    idx
}

/// Relative-position index for overlapping cross attention: queries come from a
/// `ws × ws` window, keys from the `ows × ows` window around it.
///
/// # This index goes negative, on purpose
///
/// Upstream HAT shifts by `ws − ows + 1` where centring the displacement would
/// need `(ows − ws)/2 + ws − 1`. With the released ×4 configuration
/// (`ws = 16`, `overlap_ratio = 0.5` ⇒ `ows = 24`) the per-axis displacement
/// lands in `[−22, 16]` and the flattened index in `[−880, 640]` against a
/// 1521-row table. PyTorch indexes negatives from the end, so the model *was
/// trained* reading rows `1521 − 880 …` for those displacements.
///
/// Replicating the wrap is therefore required for the released weights to mean
/// anything; "fixing" the centring would silently permute the learned bias.
/// The signed value is returned here and wrapped in [`rpe_bias`].
fn oca_rpe_index(ws: usize, ows: usize) -> Vec<isize> {
    let (nq, nk) = (ws * ws, ows * ows);
    let span = (ws + ows - 1) as isize;
    let mut idx = vec![0isize; nq * nk];
    for i in 0..nq {
        let (ih, iw) = (i / ws, i % ws);
        for j in 0..nk {
            let (jh, jw) = (j / ows, j % ows);
            let dh = jh as isize - ih as isize + ws as isize - ows as isize + 1;
            let dw = jw as isize - iw as isize + ws as isize - ows as isize + 1;
            idx[i * nk + j] = dh * span + dw;
        }
    }
    idx
}

/// Take a `[rows, heads]` bias table and an index, and emit the `[1, nH, Q, K]`
/// additive bias the attention needs — resolved on the host, since the table is
/// frozen at inference.
pub(crate) fn rpe_bias(
    b: &mut Builder,
    wm: &mut WeightMap,
    key: &str,
    index: &[isize],
    q: usize,
    k: usize,
    rows: usize,
    heads: usize,
    tag: &str,
) -> Result<HirNodeId> {
    ensure!(
        index.len() == q * k,
        "{key}: index has {} entries, attention needs {q}×{k}",
        index.len()
    );
    // Rows are fixed by the window geometry, so the expected table size is
    // known before the table is read — which is what lets a mismatched
    // table be reported as one.
    let table = b
        .take_host(wm, key, &[rows, heads])
        .with_context(|| format!("reading relative position bias table {key}"))?;
    let mut bias = vec![0.0f32; heads * q * k];
    for (pos, &signed) in index.iter().enumerate() {
        // Python-style wrapping. Only OCA ever produces a negative index, and
        // there it is load-bearing — see `oca_rpe_index`.
        let row = signed.rem_euclid(rows as isize) as usize;
        ensure!(
            row < rows,
            "{key}: relative position index {signed} does not reduce into the \
             table\'s {rows} rows"
        );
        for h in 0..heads {
            bias[h * q * k + pos] = table[row * heads + h];
        }
    }
    Ok(b.constant(tag, bias, &[1, heads, q, k]))
}

/// The `0` / `−100` mask that stops a shifted window from attending across the
/// wrap-around seam. Shape `[1, nW, 1, N, N]` — broadcast over heads.
pub(crate) fn shift_mask(g: Win, shift: usize) -> Vec<f32> {
    // Region id per pixel, exactly the reference's three-slice labelling.
    let region = |p: usize, extent: usize| -> usize {
        if p < extent - g.ws {
            0
        } else if p < extent - shift {
            1
        } else {
            2
        }
    };
    let mut ids = vec![0usize; g.h * g.w];
    for y in 0..g.h {
        for x in 0..g.w {
            ids[y * g.w + x] = region(y, g.h) * 3 + region(x, g.w);
        }
    }

    let mut out = vec![0.0f32; g.nw * g.n * g.n];
    for wi in 0..g.nw {
        let (by, bx) = (wi / g.gw, wi % g.gw);
        let win: Vec<usize> = (0..g.n)
            .map(|t| {
                let (ty, tx) = (t / g.ws, t % g.ws);
                ids[(by * g.ws + ty) * g.w + bx * g.ws + tx]
            })
            .collect();
        for i in 0..g.n {
            for j in 0..g.n {
                // The reference builds `mask[i] − mask[j]` and fills any
                // non-zero difference with −100.
                if win[i] != win[j] {
                    out[wi * g.n * g.n + i * g.n + j] = -100.0;
                }
            }
        }
    }
    out
}

/// `WindowAttention`: `[nW, N, C]` in, `[nW, N, C]` out.
#[allow(clippy::too_many_arguments)]
fn window_attention(
    b: &mut Builder,
    wm: &mut WeightMap,
    prefix: &str,
    x: HirNodeId,
    g: Win,
    c: usize,
    heads: usize,
    qkv_bias: bool,
    mask: Option<HirNodeId>,
) -> Result<HirNodeId> {
    ensure!(
        c.is_multiple_of(heads),
        "{prefix}: {c} channels do not split into {heads} heads"
    );
    let hd = c / heads;
    let qkv = b.linear(wm, &format!("{prefix}.qkv"), x, 3 * c, qkv_bias)?;

    // `nn.Linear(dim, 3·dim)` stacks q, k and v as contiguous row blocks, so a
    // narrow on the feature axis is exactly the reference's reshape+permute.
    let head_split = |b: &mut Builder, t: HirNodeId| {
        let r = b
            .g()
            .reshape_(t, vec![g.nw as i64, g.n as i64, heads as i64, hd as i64]);
        b.g().transpose_(r, vec![0, 2, 1, 3])
    };
    let q = b.g().narrow_(qkv, 2, 0, c);
    let k = b.g().narrow_(qkv, 2, c, c);
    let v = b.g().narrow_(qkv, 2, 2 * c, c);
    let q = head_split(b, q);
    let k = head_split(b, k);
    let v = head_split(b, v);

    let scale = b.scalar_like("attn_scale", 1.0 / (hd as f32).sqrt(), q);
    let q = b.g().mul(q, scale);
    let kt = b.g().transpose_(k, vec![0, 1, 3, 2]);
    let mut attn = b.g().mm(q, kt);

    let index = rpe_index(g.ws);
    let bias = rpe_bias(
        b,
        wm,
        &format!("{prefix}.relative_position_bias_table"),
        &index,
        g.n,
        g.n,
        (2 * g.ws - 1) * (2 * g.ws - 1),
        heads,
        "rpe",
    )?;
    attn = b.g().add(attn, bias);

    if let Some(m) = mask {
        let grouped = b.g().reshape_(
            attn,
            vec![1, g.nw as i64, heads as i64, g.n as i64, g.n as i64],
        );
        let masked = b.g().add(grouped, m);
        attn = b.g().reshape_(
            masked,
            vec![g.nw as i64, heads as i64, g.n as i64, g.n as i64],
        );
    }

    let attn = b.g().sm(attn, -1);
    let out = b.g().mm(attn, v);
    let out = b.g().transpose_(out, vec![0, 2, 1, 3]);
    let out = b.g().reshape_(out, vec![g.nw as i64, g.n as i64, c as i64]);
    b.linear(wm, &format!("{prefix}.proj"), out, c, true)
}

fn mlp(
    b: &mut Builder,
    wm: &mut WeightMap,
    prefix: &str,
    x: HirNodeId,
    hidden: usize,
    c: usize,
) -> Result<HirNodeId> {
    let h = b.linear(wm, &format!("{prefix}.fc1"), x, hidden, true)?;
    let h = b.gelu(h);
    b.linear(wm, &format!("{prefix}.fc2"), h, c, true)
}

/// HAT's `CAB` — a small conv bottleneck closed by channel attention. Runs in
/// NCHW alongside the token-form attention branch.
fn conv_attention_block(
    b: &mut Builder,
    wm: &mut WeightMap,
    prefix: &str,
    x_nchw: HirNodeId,
    c: usize,
    compress_ratio: usize,
    squeeze_factor: usize,
) -> Result<HirNodeId> {
    let mid = c / compress_ratio;
    let h = b.conv3x3(wm, &format!("{prefix}.cab.0"), x_nchw, mid)?;
    let h = b.gelu(h);
    let h = b.conv3x3(wm, &format!("{prefix}.cab.2"), h, c)?;

    let pooled = b.g().mean(h, vec![2, 3], true);
    let sq = b.conv1x1(
        wm,
        &format!("{prefix}.cab.3.attention.1"),
        pooled,
        c / squeeze_factor,
    )?;
    let sq = b.relu(sq);
    let ex = b.conv1x1(wm, &format!("{prefix}.cab.3.attention.3"), sq, c)?;
    let gate = b.sigmoid(ex);
    Ok(b.g().mul(h, gate))
}

/// `[1, L, C]` token form → `[1, C, H, W]`.
pub(crate) fn to_nchw(b: &mut Builder, x: HirNodeId, g: Win, c: usize) -> HirNodeId {
    debug_assert_eq!(
        b.dims(x).len(),
        3,
        "to_nchw wants tokens, got {:?}",
        b.dims(x)
    );
    let t = b.g().transpose_(x, vec![0, 2, 1]);
    b.g().reshape_(t, vec![1, c as i64, g.h as i64, g.w as i64])
}

/// `[1, C, H, W]` → `[1, L, C]`.
pub(crate) fn to_tokens(b: &mut Builder, x: HirNodeId, g: Win, c: usize) -> HirNodeId {
    let f = b.g().reshape_(x, vec![1, c as i64, (g.h * g.w) as i64]);
    b.g().transpose_(f, vec![0, 2, 1])
}

/// One Swin transformer block, optionally with HAT's conv branch.
#[allow(clippy::too_many_arguments)]
/// `PatchEmbed.proj`, a 1x1 convolution that only Swin V2 checkpoints carry.
fn patch_proj(
    b: &mut Builder,
    wm: &mut WeightMap,
    prefix: &str,
    x: HirNodeId,
    c: usize,
    v2: bool,
) -> Result<HirNodeId> {
    if !v2 {
        return Ok(x);
    }
    b.conv1x1(wm, &format!("{prefix}.proj"), x, c)
}

fn swin_block(
    b: &mut Builder,
    wm: &mut WeightMap,
    prefix: &str,
    x: HirNodeId,
    g: Win,
    c: usize,
    heads: usize,
    shift: usize,
    mlp_hidden: usize,
    qkv_bias: bool,
    hat: Option<(usize, usize, f32)>,
    v2: bool,
) -> Result<HirNodeId> {
    let shortcut = x;
    // Swin **V2** moves both normalizations *after* their sublayer — the
    // "res-post-norm" of the V2 paper. The pre-norm form still runs and still
    // produces an image, so this branch is the whole difference between a
    // correct Swin2SR and a plausible wrong one.
    let normed = if v2 {
        x
    } else {
        b.layer_norm(wm, &format!("{prefix}.norm1"), x, 1e-5)?
    };

    // HAT's conv branch reads the *normalized* tokens, before the shift.
    let conv_branch = match hat {
        Some((compress, squeeze, _)) => {
            let nchw = to_nchw(b, normed, g, c);
            let cab = conv_attention_block(
                b,
                wm,
                &format!("{prefix}.conv_block"),
                nchw,
                c,
                compress,
                squeeze,
            )?;
            Some(to_tokens(b, cab, g, c))
        }
        None => None,
    };

    let shifted = roll(b, normed, g, c, -(shift as isize));
    let mask = if shift > 0 {
        let m = shift_mask(g, shift);
        Some(b.constant("shift_mask", m, &[1, g.nw, 1, g.n, g.n]))
    } else {
        None
    };

    let windows = window_partition(b, shifted, g, c);
    let attended = if v2 {
        window_attention_v2(b, wm, &format!("{prefix}.attn"), windows, g, c, heads, mask)?
    } else {
        window_attention(
            b,
            wm,
            &format!("{prefix}.attn"),
            windows,
            g,
            c,
            heads,
            qkv_bias,
            mask,
        )?
    };
    let merged = window_reverse(b, attended, g, c);
    let unshifted = roll(b, merged, g, c, shift as isize);
    let unshifted = if v2 {
        b.layer_norm(wm, &format!("{prefix}.norm1"), unshifted, 1e-5)?
    } else {
        unshifted
    };

    let mut h = b.g().add(shortcut, unshifted);
    if let (Some(cb), Some((_, _, conv_scale))) = (conv_branch, hat) {
        let s = b.scalar_like("conv_scale", conv_scale, cb);
        let scaled = b.g().mul(cb, s);
        h = b.g().add(h, scaled);
    }

    let m = if v2 {
        let m = mlp(b, wm, &format!("{prefix}.mlp"), h, mlp_hidden, c)?;
        b.layer_norm(wm, &format!("{prefix}.norm2"), m, 1e-5)?
    } else {
        let n2 = b.layer_norm(wm, &format!("{prefix}.norm2"), h, 1e-5)?;
        mlp(b, wm, &format!("{prefix}.mlp"), n2, mlp_hidden, c)?
    };
    Ok(b.g().add(h, m))
}

/// Swin **V2** window attention: cosine similarity, a learned per-head
/// temperature, and a position bias produced by a small MLP.
///
/// # Everything constant is folded on the host
///
/// The bias is `16·sigmoid(cpb_mlp(relative_coords_table))`, and both the MLP's
/// weights and the coordinate table are frozen at inference — so the whole
/// `[nH, n, n]` bias is computed once here rather than running a 512-wide MLP
/// over a constant on every tile. The temperature is folded the same way. The
/// resulting graph has exactly the shape of the V1 one; only the constants and
/// the `q·k` normalization differ.
///
/// The coordinate table is *read* rather than recomputed: it is derived from
/// `pretrained_window_size`, which the weights do not otherwise record.
#[allow(clippy::too_many_arguments)]
fn window_attention_v2(
    b: &mut Builder,
    wm: &mut WeightMap,
    prefix: &str,
    x: HirNodeId,
    g: Win,
    c: usize,
    heads: usize,
    mask: Option<HirNodeId>,
) -> Result<HirNodeId> {
    ensure!(
        c.is_multiple_of(heads),
        "{prefix}: {c} channels do not split into {heads} heads"
    );
    let hd = c / heads;

    // `qkv` is biasless; q and v carry their own bias parameters and k gets
    // none, which is how the reference keeps k's scale free.
    let qkv = b.linear(wm, &format!("{prefix}.qkv"), x, c * 3, false)?;
    let qkv = if b.has(wm, &format!("{prefix}.q_bias")) {
        let q_bias = b.take_host(wm, &format!("{prefix}.q_bias"), &[c])?;
        let v_bias = b.take_host(wm, &format!("{prefix}.v_bias"), &[c])?;
        let mut bias = q_bias;
        bias.extend(std::iter::repeat_n(0.0, c));
        bias.extend(v_bias);
        let node = b.constant(&format!("{prefix}_qkv_bias"), bias, &[1, 1, c * 3]);
        b.g().add(qkv, node)
    } else {
        qkv
    };

    let split = |b: &mut Builder, t: HirNodeId| {
        let four = b
            .g()
            .reshape_(t, vec![g.nw as i64, g.n as i64, heads as i64, hd as i64]);
        b.g().transpose_(four, vec![0, 2, 1, 3])
    };
    let qs = b.g().narrow_(qkv, 2, 0, c);
    let ks = b.g().narrow_(qkv, 2, c, c);
    let vs = b.g().narrow_(qkv, 2, 2 * c, c);
    let q = split(b, qs);
    let k = split(b, ks);
    let v = split(b, vs);

    // Cosine attention: no 1/sqrt(d) scale at all — the temperature replaces it.
    let q = b.l2_normalize_last(q);
    let k = b.l2_normalize_last(k);
    let kt = b.g().transpose_(k, vec![0, 1, 3, 2]);
    let mut attn = b.g().mm(q, kt);

    let scale = v2_logit_scale(b, wm, prefix, heads)?;
    attn = b.g().mul(attn, scale);

    let bias = v2_position_bias(b, wm, prefix, g, heads)?;
    attn = b.g().add(attn, bias);

    if let Some(m) = mask {
        let grouped = b.g().reshape_(
            attn,
            vec![1, g.nw as i64, heads as i64, g.n as i64, g.n as i64],
        );
        let masked = b.g().add(grouped, m);
        attn = b.g().reshape_(
            masked,
            vec![g.nw as i64, heads as i64, g.n as i64, g.n as i64],
        );
    }

    let attn = b.g().sm(attn, -1);
    let out = b.g().mm(attn, v);
    let out = b.g().transpose_(out, vec![0, 2, 1, 3]);
    let out = b.g().reshape_(out, vec![g.nw as i64, g.n as i64, c as i64]);
    b.linear(wm, &format!("{prefix}.proj"), out, c, true)
}

/// `clamp(logit_scale, max = ln 100).exp()`, folded to a constant.
///
/// The clamp is at `log(1/0.01)`, which caps the temperature at 100 — without
/// it a learned scale can saturate the softmax. It is applied here in f64 and
/// baked in.
fn v2_logit_scale(
    b: &mut Builder,
    wm: &mut WeightMap,
    prefix: &str,
    heads: usize,
) -> Result<HirNodeId> {
    let raw = b.take_host(wm, &format!("{prefix}.logit_scale"), &[heads, 1, 1])?;
    let cap = 100f64.ln();
    let folded: Vec<f32> = raw
        .iter()
        .map(|&v| ((v as f64).min(cap)).exp() as f32)
        .collect();
    Ok(b.constant(&format!("{prefix}_logit_scale"), folded, &[1, heads, 1, 1]))
}

/// `16·sigmoid(cpb_mlp(relative_coords_table))[index]`, folded to a constant.
fn v2_position_bias(
    b: &mut Builder,
    wm: &mut WeightMap,
    prefix: &str,
    g: Win,
    heads: usize,
) -> Result<HirNodeId> {
    let span = 2 * g.ws - 1;
    let rows = span * span;
    // `[1, 2·Wh−1, 2·Ww−1, 2]` — read, not recomputed, because its scaling
    // depends on `pretrained_window_size`, which is nowhere else in the file.
    let table = b.take_host(
        wm,
        &format!("{prefix}.relative_coords_table"),
        &[1, span, span, 2],
    )?;
    let w0 = b.take_host(wm, &format!("{prefix}.cpb_mlp.0.weight"), &[CPB_HIDDEN, 2])?;
    let b0 = b.take_host(wm, &format!("{prefix}.cpb_mlp.0.bias"), &[CPB_HIDDEN])?;
    let w2 = b.take_host(
        wm,
        &format!("{prefix}.cpb_mlp.2.weight"),
        &[heads, CPB_HIDDEN],
    )?;

    // Linear(2 → 512) → ReLU → Linear(512 → nH, bias=False).
    let mut flat = vec![0.0f32; rows * heads];
    let mut hidden = vec![0.0f32; CPB_HIDDEN];
    for r in 0..rows {
        let (x0, x1) = (table[r * 2], table[r * 2 + 1]);
        for (h, hv) in hidden.iter_mut().enumerate() {
            *hv = (w0[h * 2] * x0 + w0[h * 2 + 1] * x1 + b0[h]).max(0.0);
        }
        for head in 0..heads {
            let row = &w2[head * CPB_HIDDEN..(head + 1) * CPB_HIDDEN];
            flat[r * heads + head] = row.iter().zip(&hidden).map(|(a, c)| a * c).sum();
        }
    }

    // Index into `[nH, n, n]`, then the `16·sigmoid` squash.
    let index = rpe_index(g.ws);
    let mut bias = vec![0.0f32; heads * g.n * g.n];
    for (pos, &signed) in index.iter().enumerate() {
        let r = signed.rem_euclid(rows as isize) as usize;
        for head in 0..heads {
            let v = flat[r * heads + head];
            bias[head * g.n * g.n + pos] = 16.0 / (1.0 + (-v).exp());
        }
    }
    Ok(b.constant(&format!("{prefix}_cpb_bias"), bias, &[1, heads, g.n, g.n]))
}

/// HAT's overlapping-cross-attention block.
///
/// Queries come from the plain `ws × ws` partition; keys and values come from
/// an `ows × ows` window centred on it, gathered by an `nn.Unfold`. That unfold
/// is expressed here as a single `Gather` through a precomputed index over the
/// zero-padded feature map — `nW` separate slices would otherwise add hundreds
/// of nodes per group.
#[allow(clippy::too_many_arguments)]
fn overlap_attention(
    b: &mut Builder,
    wm: &mut WeightMap,
    prefix: &str,
    x: HirNodeId,
    g: Win,
    c: usize,
    heads: usize,
    overlap_ratio: f32,
    mlp_hidden: usize,
    qkv_bias: bool,
) -> Result<HirNodeId> {
    let ows = g.ws + (g.ws as f32 * overlap_ratio) as usize;
    let pad = (ows - g.ws) / 2;
    let hd = c / heads;
    let nk = ows * ows;

    let shortcut = x;
    let normed = b.layer_norm(wm, &format!("{prefix}.norm1"), x, 1e-5)?;
    let qkv = b.linear(wm, &format!("{prefix}.qkv"), normed, 3 * c, qkv_bias)?;

    let q = b.g().narrow_(qkv, 2, 0, c);
    let q_windows = window_partition(b, q, g, c);
    let q = b.g().reshape_(
        q_windows,
        vec![g.nw as i64, g.n as i64, heads as i64, hd as i64],
    );
    let q = b.g().transpose_(q, vec![0, 2, 1, 3]);

    // k and v travel together through the unfold, as in the reference.
    let kv = b.g().narrow_(qkv, 2, c, 2 * c);
    let kv_nchw = {
        let t = b.g().transpose_(kv, vec![0, 2, 1]);
        b.g()
            .reshape_(t, vec![1, (2 * c) as i64, g.h as i64, g.w as i64])
    };
    let padded = b.g().pad_(
        kv_nchw,
        vec![[0, 0], [0, 0], [pad, pad], [pad, pad]],
        PadMode::Constant(0.0),
    );
    let (ph, pw) = (g.h + 2 * pad, g.w + 2 * pad);
    // Transpose to spatial-major so the unfold becomes a gather along axis 0 —
    // the only axis every backend is guaranteed to implement — and the
    // transpose itself is a last-two-axis swap, which takes the fast path.
    let flat = b
        .g()
        .reshape_(padded, vec![(2 * c) as i64, (ph * pw) as i64]);
    let spatial = b.g().transpose_(flat, vec![1, 0]);

    // For each window, the flat offsets of its ows × ows neighbourhood.
    let mut idx = Vec::with_capacity(g.nw * nk);
    for wi in 0..g.nw {
        let (by, bx) = (wi / g.gw, wi % g.gw);
        for ky in 0..ows {
            for kx in 0..ows {
                idx.push(((by * g.ws + ky) * pw + bx * g.ws + kx) as f32);
            }
        }
    }
    let idx = b.constant("oca_unfold", idx, &[g.nw * nk]);
    let gathered = b.g().gather_(spatial, idx, 0);
    // [nW·nk, 2C] → [nW, nk, 2C]; k occupies the first C, v the rest.
    let gathered = b
        .g()
        .reshape_(gathered, vec![g.nw as i64, nk as i64, (2 * c) as i64]);
    let k = b.g().narrow_(gathered, 2, 0, c);
    let v = b.g().narrow_(gathered, 2, c, c);
    let to_heads = |b: &mut Builder, t: HirNodeId| {
        let r = b
            .g()
            .reshape_(t, vec![g.nw as i64, nk as i64, heads as i64, hd as i64]);
        b.g().transpose_(r, vec![0, 2, 1, 3])
    };
    let k = to_heads(b, k);
    let v = to_heads(b, v);

    let scale = b.scalar_like("oca_scale", 1.0 / (hd as f32).sqrt(), q);
    let q = b.g().mul(q, scale);
    let kt = b.g().transpose_(k, vec![0, 1, 3, 2]);
    let attn = b.g().mm(q, kt);

    let index = oca_rpe_index(g.ws, ows);
    let bias = rpe_bias(
        b,
        wm,
        &format!("{prefix}.relative_position_bias_table"),
        &index,
        g.n,
        nk,
        (g.ws + ows - 1) * (g.ws + ows - 1),
        heads,
        "oca_rpe",
    )?;
    let attn = b.g().add(attn, bias);
    let attn = b.g().sm(attn, -1);

    let out = b.g().mm(attn, v);
    let out = b.g().transpose_(out, vec![0, 2, 1, 3]);
    let out = b.g().reshape_(out, vec![g.nw as i64, g.n as i64, c as i64]);
    let merged = window_reverse(b, out, g, c);
    let projected = b.linear(wm, &format!("{prefix}.proj"), merged, c, true)?;
    let h = b.g().add(projected, shortcut);

    let n2 = b.layer_norm(wm, &format!("{prefix}.norm2"), h, 1e-5)?;
    let m = mlp(b, wm, &format!("{prefix}.mlp"), n2, mlp_hidden, c)?;
    Ok(b.g().add(h, m))
}

/// The convolution that closes a residual group.
fn resi_conv(
    b: &mut Builder,
    wm: &mut WeightMap,
    prefix: &str,
    x: HirNodeId,
    c: usize,
    kind: ResiConnection,
) -> Result<HirNodeId> {
    match kind {
        ResiConnection::Identity => Ok(x),
        ResiConnection::Conv1 => b.conv3x3(wm, prefix, x, c),
        ResiConnection::Conv3 => {
            let h = b.conv3x3(wm, &format!("{prefix}.0"), x, c / 4)?;
            let h = b.leaky_relu(h, 0.2);
            let h = b.conv1x1(wm, &format!("{prefix}.2"), h, c / 4)?;
            let h = b.leaky_relu(h, 0.2);
            b.conv3x3(wm, &format!("{prefix}.4"), h, c)
        }
    }
}

/// Build SwinIR / HAT / DRCT.
pub fn build(
    b: &mut Builder,
    wm: &mut WeightMap,
    cfg: &ModelConfig,
    p: &SwinParams,
    x_in: HirNodeId,
) -> Result<HirNodeId> {
    let [_, _, h, w] = b.dims4(x_in);
    let g = Win::new(h, w, p.window_size)?;
    let c = p.embed_dim;

    let mean = b.constant("swin_mean", vec![0.4488, 0.4371, 0.4040], &[1, 3, 1, 1]);
    let centered = b.g().sub(x_in, mean);
    let range = b.scalar_like("swin_range", p.img_range, centered);
    let x = b.g().mul(centered, range);

    let shallow = b.conv3x3(wm, "conv_first", x, c)?;

    // ── deep features ────────────────────────────────────────────────
    // Swin2SR's `PatchEmbed` carries a real 1x1 convolution, applied before the
    // flatten; SwinIR's only flattens. It is easy to miss because the module is
    // named the same and sits in the same place, and dropping it costs one
    // projection per group with no shape error to show for it.
    let embedded = patch_proj(b, wm, "patch_embed", shallow, c, p.v2)?;
    let mut tokens = to_tokens(b, embedded, g, c);
    if p.patch_norm {
        tokens = b.layer_norm(wm, "patch_embed.norm", tokens, 1e-5)?;
    }

    let mlp_hidden = (c as f32 * p.mlp_ratio) as usize;
    for (gi, &depth) in p.depths.iter().enumerate() {
        let group_in = tokens;
        tokens = match cfg.arch {
            Arch::Drct => drct_group(b, wm, gi, tokens, g, c, p)?,
            _ => {
                let mut t = tokens;
                for d in 0..depth {
                    t = swin_block(
                        b,
                        wm,
                        &format!("layers.{gi}.residual_group.blocks.{d}"),
                        t,
                        g,
                        c,
                        p.num_heads[gi],
                        if d % 2 == 0 { 0 } else { p.window_size / 2 },
                        mlp_hidden,
                        p.qkv_bias,
                        p.hat
                            .as_ref()
                            .map(|hp| (hp.compress_ratio, hp.squeeze_factor, hp.conv_scale)),
                        p.v2,
                    )
                    .with_context(|| format!("group {gi} block {d}"))?;
                }
                if let Some(hp) = &p.hat {
                    t = overlap_attention(
                        b,
                        wm,
                        &format!("layers.{gi}.residual_group.overlap_attn"),
                        t,
                        g,
                        c,
                        p.num_heads[gi],
                        hp.overlap_ratio,
                        mlp_hidden,
                        p.qkv_bias,
                    )
                    .with_context(|| format!("group {gi} overlap attention"))?;
                }
                t
            }
        };

        // RSTB closes with a convolution in NCHW and adds the group's input.
        if p.resi_connection != ResiConnection::Identity {
            let nchw = to_nchw(b, tokens, g, c);
            let conved = resi_conv(
                b,
                wm,
                &format!("layers.{gi}.conv"),
                nchw,
                c,
                p.resi_connection,
            )?;
            // The RSTB's own `patch_embed` projection, after its convolution.
            let projected =
                patch_proj(b, wm, &format!("layers.{gi}.patch_embed"), conved, c, p.v2)?;
            tokens = to_tokens(b, projected, g, c);
        }
        tokens = b.g().add(tokens, group_in);
    }

    let tokens = b.layer_norm(wm, "norm", tokens, 1e-5)?;
    let deep = to_nchw(b, tokens, g, c);
    let body = b.conv3x3(wm, "conv_after_body", deep, c)?;
    let feat = b.g().add(body, shallow);

    // ── upsampling tail ──────────────────────────────────────────────
    let r = cfg.scale;
    let out = match p.upsampler {
        Upsampler::PixelShuffleDirect => {
            let up = b.conv3x3(wm, "upsample.0", feat, cfg.out_ch * r * r)?;
            b.pixel_shuffle(up, r)?
        }
        Upsampler::PixelShuffle => {
            // `conv_before_upsample` is Conv → LeakyReLU at PyTorch's default
            // 0.01 slope (the reference constructs it with no argument).
            let nf = p.num_feat;
            let h = b.conv3x3(wm, "conv_before_upsample.0", feat, nf)?;
            let mut h = b.leaky_relu(h, 0.01);
            // A chain of ×2 stages, or a single ×3. The convolutions sit at
            // even indices with the pixel shuffles between them.
            for (i, s) in upsample_stages(r)?.into_iter().enumerate() {
                h = b.conv3x3(wm, &format!("upsample.{}", i * 2), h, nf * s * s)?;
                h = b.pixel_shuffle(h, s)?;
            }
            b.conv3x3(wm, "conv_last", h, cfg.out_ch)?
        }
        Upsampler::NearestConv => {
            let nf = p.num_feat;
            ensure!(
                r.is_power_of_two() && (2..=8).contains(&r),
                "the nearest+conv tail only supports ×2, ×4 and ×8, not ×{r}"
            );
            let h = b.conv3x3(wm, "conv_before_upsample.0", feat, nf)?;
            let mut h = b.leaky_relu(h, 0.01);
            for stage in 1..=(r.trailing_zeros() as usize) {
                let up = b.nearest_upsample(h, 2);
                let conv = b.conv3x3(wm, &format!("conv_up{stage}"), up, nf)?;
                h = b.leaky_relu(conv, 0.2);
            }
            let hr = b.conv3x3(wm, "conv_hr", h, nf)?;
            let hr = b.leaky_relu(hr, 0.2);
            b.conv3x3(wm, "conv_last", hr, cfg.out_ch)?
        }
        // Denoising / JPEG-artifact reduction. The network predicts a
        // *correction*: one convolution back to image width, added to the
        // (normalized) input rather than replacing it.
        Upsampler::Residual => {
            ensure!(
                r == 1,
                "the residual tail is a ×1 restoration path, not ×{r}"
            );
            let correction = b.conv3x3(wm, "conv_last", feat, cfg.out_ch)?;
            b.g().add(x, correction)
        }
        Upsampler::DySample => {
            anyhow::bail!("DySample is not a Swin-family upsampler")
        }
    };

    // Denormalize. Unlike SPAN, these models *do* undo the input shift.
    let inv = b.scalar_like("swin_inv_range", 1.0 / p.img_range, out);
    let scaled = b.g().mul(out, inv);
    let mean = b.constant("swin_mean_out", vec![0.4488, 0.4371, 0.4040], &[1, 3, 1, 1]);
    Ok(b.g().add(scaled, mean))
}

/// Decompose a scale factor into the pixel-shuffle stages the reference builds:
/// a chain of ×2 for powers of two, or a single ×3.
fn upsample_stages(scale: usize) -> Result<Vec<usize>> {
    if scale == 3 {
        return Ok(vec![3]);
    }
    ensure!(
        scale.is_power_of_two(),
        "the classical upsampler supports 2^n and 3, not ×{scale}"
    );
    Ok(vec![2; scale.trailing_zeros() as usize])
}

/// DRCT's `RDG`: five blocks whose inputs grow by `gc` channels each, with a
/// 1×1 `adjust` projecting every output back down to the growth width.
fn drct_group(
    b: &mut Builder,
    wm: &mut WeightMap,
    gi: usize,
    x: HirNodeId,
    g: Win,
    c: usize,
    p: &SwinParams,
) -> Result<HirNodeId> {
    let gc = p
        .drct_gc
        .context("DRCT config is missing its growth-channel count")?;
    let mut feats = vec![x];

    for i in 1..=5usize {
        let width = c + (i - 1) * gc;
        let cat = if feats.len() == 1 {
            feats[0]
        } else {
            b.g().concat_(feats.clone(), 2)
        };
        // Heads shrink so they keep dividing the grown width. The reference
        // computes `num_heads − (width % num_heads)` for every block whose
        // input has been widened by the dense connections.
        let heads = drct_block_heads(p.num_heads[gi], width);
        // DRCT's last two blocks pin mlp_ratio to 1; the rest scale with
        // the block's own grown width, not the group's base width.
        let hidden = if i >= 4 {
            width
        } else {
            (width as f32 * p.mlp_ratio) as usize
        };
        let blk = swin_block(
            b,
            wm,
            &format!("layers.{gi}.swin{i}"),
            cat,
            g,
            width,
            heads,
            if i % 2 == 0 { p.window_size / 2 } else { 0 },
            hidden,
            p.qkv_bias,
            None,
            // DRCT is a V1 architecture; there is no V2 variant of it.
            false,
        )
        .with_context(|| format!("DRCT group {gi} swin{i}"))?;

        let nchw = to_nchw(b, blk, g, width);
        let out_c = if i == 5 { c } else { gc };
        let adj = b.conv1x1(wm, &format!("layers.{gi}.adjust{i}"), nchw, out_c)?;
        let adj = to_tokens(b, adj, g, out_c);
        // Every adjust but the last is followed by a leaky ReLU.
        let adj = if i == 5 { adj } else { b.leaky_relu(adj, 0.2) };
        feats.push(adj);
    }

    let last = *feats.last().expect("five blocks always push");
    let scale = b.scalar_like("drct_residual", 0.2, last);
    let scaled = b.g().mul(last, scale);
    Ok(b.g().add(scaled, x))
}

/// DRCT's per-block head count: `num_heads − (width % num_heads)`. The first
/// block of a group is unwidened, so the remainder is zero and it keeps the
/// group's own head count.
fn drct_block_heads(group_heads: usize, width: usize) -> usize {
    let r = width % group_heads;
    if r == 0 { group_heads } else { group_heads - r }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `index[i, j]` must be symmetric under swapping i and j only through the
    /// centre: displacement (0,0) always lands on the middle row.
    #[test]
    fn rpe_index_centres_zero_displacement() {
        let ws = 4;
        let idx = rpe_index(ws);
        let n = ws * ws;
        let centre = ((2 * ws - 1) * (ws - 1) + (ws - 1)) as isize;
        for i in 0..n {
            assert_eq!(idx[i * n + i], centre, "self-attention must be centred");
        }
        assert!(
            idx.iter()
                .all(|&v| v >= 0 && v < ((2 * ws - 1) * (2 * ws - 1)) as isize)
        );
    }

    /// Opposite displacements sit symmetrically about the centre row.
    #[test]
    fn rpe_index_is_antisymmetric_about_the_centre() {
        let ws = 3;
        let n = ws * ws;
        let idx = rpe_index(ws);
        let rows = ((2 * ws - 1) * (2 * ws - 1)) as isize;
        for i in 0..n {
            for j in 0..n {
                assert_eq!(idx[i * n + j] + idx[j * n + i], rows - 1);
            }
        }
    }

    /// Pins upstream HAT\'s off-centre shift, negative indices and all. If this
    /// ever "looks wrong" and gets centred, the released weights silently read
    /// the wrong bias rows.
    #[test]
    fn oca_rpe_index_reproduces_upstreams_negative_wrap() {
        let (ws, ows) = (16usize, 24usize);
        let idx = oca_rpe_index(ws, ows);
        assert_eq!(idx.len(), ws * ws * ows * ows);
        let rows = ((ws + ows - 1) * (ws + ows - 1)) as isize;

        let lo = *idx.iter().min().unwrap();
        let hi = *idx.iter().max().unwrap();
        assert_eq!((lo, hi), (-880, 640), "upstream shift changed");
        // Every index — wrapped or not — must reduce into a real table row, and
        // the span must not be wide enough to alias two displacements together.
        assert!(idx.iter().all(|&v| v.rem_euclid(rows) < rows));
        assert!(hi - lo < rows, "displacement span {} aliases", hi - lo);
    }

    /// A window fully inside the image spans one region, so nothing is masked;
    /// the wrapped windows at the far edge mix regions and must be.
    #[test]
    fn shift_mask_only_blocks_the_wrapped_windows() {
        let g = Win::new(8, 8, 4).unwrap();
        let m = shift_mask(g, 2);
        assert_eq!(m.len(), g.nw * g.n * g.n);
        // Window 0 is the top-left interior window: all one region.
        assert!(m[..g.n * g.n].iter().all(|&v| v == 0.0));
        // The last window straddles the wrap in both axes.
        let last = &m[(g.nw - 1) * g.n * g.n..];
        assert!(last.iter().any(|&v| v == -100.0));
        // Masked entries are only ever 0 or −100.
        assert!(m.iter().all(|&v| v == 0.0 || v == -100.0));
    }

    #[test]
    fn a_feature_map_that_does_not_tile_is_rejected() {
        assert!(Win::new(10, 8, 4).is_err());
        assert!(Win::new(8, 8, 4).is_ok());
    }
}
