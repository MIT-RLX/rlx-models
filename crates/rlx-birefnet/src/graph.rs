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

//! The BiRefNet graph (`ZhengPeng7/BiRefNet*`, `birefnet.py`), one HIR module:
//!
//! - **Encoder**: Swin-v1 (`bb.*`, Microsoft key layout) run on the image and
//!   on a half-size copy; per-stage features concatenated (`mul_scl_ipt =
//!   cat`), the three shallower ones also resized onto the deepest
//!   (`cxt`), then a squeeze block.
//! - **Decoder**: four `BasicDecBlk`s (conv → deformable ASPP → conv), each fed
//!   the image as block space-to-depth "patches" (`dec_ipt_split`), lateral 1×1
//!   skips, and a gradient-attention gate (`out_ref`); a 1×1 head gives logits.
//!
//! Deformable convolution (torchvision `deform_conv2d`, DCNv2 with a
//! `2·sigmoid` modulator) is built from `grid_sample2d`: all `K = k²` taps'
//! sampling positions (`base + learned offset`) form one grid, one bilinear
//! sample gathers them, the modulator scales them, and a single matmul with the
//! kernel (`[O, C·K]`) contracts taps and channels.

use anyhow::{Context, Result, bail, ensure};
use rlx_flow::{BuiltModel, Emit, MapWeights, ModelFlow};
use rlx_ir::hir::{GridMode, GridPad, HirMut};
use rlx_ir::op::{Activation, MaskKind, Op, PadMode};
use rlx_ir::{DType, HirGraphExt, HirNodeId, Shape};
use std::collections::HashMap;
use std::sync::Arc;

use crate::weights::Weights;

const LN_EPS: f32 = 1e-5;

/// A feature map `[1, c, h, w]`.
#[derive(Clone, Copy, Debug)]
pub struct T {
    pub id: HirNodeId,
    pub c: usize,
    pub h: usize,
    pub w: usize,
}

/// Intermediate outputs a graph can expose (parity tests).
pub const TAPS: &[&str] = &[
    "bb1", "bb2", "bb3", "bb4", "enc1", "enc2", "enc3", "enc4", "sq4", "logits",
];

/// Decoder intermediates, for backend bisection (no reference dump needed).
pub const DEBUG_TAPS: &[&str] = &["d4", "g4", "d3", "g3", "d2", "g2", "d1", "up1"];

struct G<'e, 'a> {
    emit: &'e mut Emit<'a>,
    w: Arc<Weights>,
    n: usize,
    /// Swin window size of the stage being built (7 for Swin-T/S/B, 12 for Swin-L),
    /// read from its relative-position table: `(2·ws − 1)²` rows.
    ws: usize,
}

fn f(dims: &[usize]) -> Shape {
    Shape::new(dims, DType::F32)
}

fn i64s(v: &[usize]) -> Vec<i64> {
    v.iter().map(|&d| d as i64).collect()
}

impl G<'_, '_> {
    fn g(&mut self) -> HirMut<'_> {
        HirMut::new(self.emit.hir())
    }

    fn param(&mut self, data: Vec<f32>, dims: &[usize]) -> HirNodeId {
        self.n += 1;
        let name = format!("brn.p{}", self.n);
        self.emit.synth_param(&name, data, f(dims))
    }

    fn scalar_like(&mut self, v: f32, dims: &[usize]) -> HirNodeId {
        let s = self.param(vec![v], &vec![1; dims.len()]);
        self.g().expand_(s, i64s(dims))
    }

    /// `x @ Wᵀ (+ b)` over the last axis.
    fn linear(&mut self, x: HirNodeId, prefix: &str, bias: bool) -> Result<HirNodeId> {
        let (w, s) = self.w.get(&format!("{prefix}.weight"))?.clone();
        let (o, i) = (s[0], s[1]);
        let wt = rlx_core::weight_map::transpose_2d(&w, o, i);
        let wt = self.param(wt, &[i, o]);
        let b = if bias {
            Some(self.w.get(&format!("{prefix}.bias"))?.0.clone())
        } else {
            None
        };
        let b = b.map(|b| self.param(b, &[o]));
        let mut g = self.g();
        let y = g.mm(x, wt);
        Ok(match b {
            Some(b) => g.add(y, b),
            None => y,
        })
    }

    fn ln(&mut self, x: HirNodeId, prefix: &str) -> Result<HirNodeId> {
        let gw = self.w.get(&format!("{prefix}.weight"))?.0.clone();
        let gb = self.w.get(&format!("{prefix}.bias"))?.0.clone();
        let n = gw.len();
        let (a, b) = (self.param(gw, &[n]), self.param(gb, &[n]));
        Ok(self.g().ln(x, a, b, LN_EPS))
    }

    /// Conv with "same" padding (`k/2`) (+ folded BN) (+ ReLU).
    fn conv(&mut self, x: T, conv: &str, bn: Option<&str>, stride: usize, relu: bool) -> Result<T> {
        self.conv_pad(x, conv, bn, stride, None, relu)
    }

    /// [`Self::conv`] with explicit padding (`None` = `k/2`).
    fn conv_pad(
        &mut self,
        x: T,
        conv: &str,
        bn: Option<&str>,
        stride: usize,
        pad: Option<usize>,
        relu: bool,
    ) -> Result<T> {
        let (w, s, b) = self.w.conv(conv, bn)?;
        let (o, ig, kh, kw) = (s[0], s[1], s[2], s[3]);
        ensure!(
            x.c.is_multiple_of(ig),
            "{conv}: input {} channels, kernel expects {ig}/group",
            x.c
        );
        let groups = x.c / ig;
        let (ph, pw) = pad.map_or((kh / 2, kw / 2), |p| (p, p));
        let (oh, ow) = (
            (x.h + 2 * ph - kh) / stride + 1,
            (x.w + 2 * pw - kw) / stride + 1,
        );
        let wid = self.param(w, &s);
        let bid = self.param(b, &[1, o, 1, 1]);
        let mut g = self.g();
        let y = g.conv2d(
            x.id,
            wid,
            [kh, kw],
            [stride, stride],
            [ph, pw],
            groups,
            f(&[1, o, oh, ow]),
        );
        let bb = g.expand_(bid, vec![1, o as i64, oh as i64, ow as i64]);
        let y = g.add(y, bb);
        let y = if relu { g.relu(y) } else { y };
        Ok(T {
            id: y,
            c: o,
            h: oh,
            w: ow,
        })
    }

    fn resize(&mut self, x: T, h: usize, w: usize) -> T {
        if (x.h, x.w) == (h, w) {
            return x;
        }
        let id = if (x.h, x.w) == (1, 1) {
            self.g()
                .expand_(x.id, vec![1, x.c as i64, h as i64, w as i64])
        } else {
            self.g().resize_bilinear2d(x.id, h, w, true)
        };
        T { id, h, w, ..x }
    }

    fn cat(&mut self, xs: &[T]) -> T {
        let c = xs.iter().map(|t| t.c).sum();
        let id = self.g().concat_(xs.iter().map(|t| t.id).collect(), 1);
        T { id, c, ..xs[0] }
    }

    // ── Swin ───────────────────────────────────────────────────────────────

    /// `[1, H·W, C]` tokens → `[1, C, H, W]`.
    fn tokens_to_map(&mut self, x: HirNodeId, c: usize, h: usize, w: usize) -> T {
        let mut g = self.g();
        let t = g.transpose_(x, vec![0, 2, 1]);
        let id = g.reshape_(t, vec![1, c as i64, h as i64, w as i64]);
        T { id, c, h, w }
    }

    fn roll(&mut self, x: HirNodeId, axis: usize, n: usize, shift: isize) -> HirNodeId {
        // torch.roll(x, shift): out[i] = x[(i - shift) mod n].
        let s = shift.rem_euclid(n as isize) as usize;
        if s == 0 {
            return x;
        }
        let mut g = self.g();
        let tail = g.narrow_(x, axis, n - s, s);
        let head = g.narrow_(x, axis, 0, n - s);
        g.concat_(vec![tail, head], axis)
    }

    /// Relative-position bias `[heads, N, N]` (+ shifted-window mask) as `[nW, heads, N, N]`.
    #[allow(non_snake_case)]
    fn window_bias(
        &mut self,
        prefix: &str,
        heads: usize,
        hp: usize,
        wp: usize,
        shift: usize,
    ) -> Result<HirNodeId> {
        let WS = self.ws;
        let table = &self
            .w
            .get(&format!("{prefix}.attn.relative_position_bias_table"))?
            .0;
        let n = WS * WS;
        let mut rel = vec![0.0f32; heads * n * n];
        for i in 0..n {
            for j in 0..n {
                let (yi, xi, yj, xj) = (i / WS, i % WS, j / WS, j % WS);
                let idx = (yi + WS - 1 - yj) * (2 * WS - 1) + (xi + WS - 1 - xj);
                for h in 0..heads {
                    rel[(h * n + i) * n + j] = table[idx * heads + h];
                }
            }
        }
        let (nh, nw) = (hp / WS, wp / WS);
        let nwin = nh * nw;
        let mut out = vec![0.0f32; nwin * heads * n * n];
        // Region labels of the padded map (BasicLayer.forward's img_mask).
        let region = |v: usize, len: usize| -> usize {
            if v < len - WS {
                0
            } else if v < len - shift {
                1
            } else {
                2
            }
        };
        for wy in 0..nh {
            for wx in 0..nw {
                let win = wy * nw + wx;
                for i in 0..n {
                    for j in 0..n {
                        let mask = if shift > 0 {
                            let lab = |t: usize| {
                                let (y, x) = (wy * WS + t / WS, wx * WS + t % WS);
                                region(y, hp) * 3 + region(x, wp)
                            };
                            if lab(i) != lab(j) { -100.0 } else { 0.0 }
                        } else {
                            0.0
                        };
                        for h in 0..heads {
                            out[((win * heads + h) * n + i) * n + j] =
                                rel[(h * n + i) * n + j] + mask;
                        }
                    }
                }
            }
        }
        Ok(self.param(out, &[nwin, heads, n, n]))
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(non_snake_case)]
    fn swin_block(
        &mut self,
        x: HirNodeId,
        prefix: &str,
        c: usize,
        heads: usize,
        h: usize,
        w: usize,
        shift: usize,
    ) -> Result<HirNodeId> {
        let WS = self.ws;
        let (hp, wp) = (h.div_ceil(WS) * WS, w.div_ceil(WS) * WS);
        let (nh, nw) = (hp / WS, wp / WS);
        let n = WS * WS;
        let bias = self.window_bias(prefix, heads, hp, wp, shift)?;
        let y = self.ln(x, &format!("{prefix}.norm1"))?;
        let y = {
            let mut g = self.g();
            let y = g.reshape_(y, vec![1, h as i64, w as i64, c as i64]);
            if (hp, wp) != (h, w) {
                g.pad_(
                    y,
                    vec![[0, 0], [0, hp - h], [0, wp - w], [0, 0]],
                    PadMode::Constant(0.0),
                )
            } else {
                y
            }
        };
        let y = self.roll(y, 1, hp, -(shift as isize));
        let y = self.roll(y, 2, wp, -(shift as isize));
        let win = {
            let mut g = self.g();
            let t = g.reshape_(y, i64s(&[nh, WS, nw, WS, c]));
            let t = g.transpose_(t, vec![0, 2, 1, 3, 4]);
            g.reshape_(t, i64s(&[nh * nw, n, c]))
        };
        let qkv = self.linear(win, &format!("{prefix}.attn.qkv"), true)?;
        let a = {
            let mut g = self.g();
            let (q, k, v) = (
                g.narrow_(qkv, 2, 0, c),
                g.narrow_(qkv, 2, c, c),
                g.narrow_(qkv, 2, 2 * c, c),
            );
            g.add_node(
                Op::Attention {
                    num_heads: heads,
                    head_dim: c / heads,
                    v_head_dim: None,
                    mask_kind: MaskKind::Bias,
                    score_scale: None,
                    attn_logit_softcap: None,
                },
                vec![q, k, v, bias],
                f(&[nh * nw, n, c]),
            )
        };
        let a = self.linear(a, &format!("{prefix}.attn.proj"), true)?;
        let y = {
            let mut g = self.g();
            let t = g.reshape_(a, i64s(&[nh, nw, WS, WS, c]));
            let t = g.transpose_(t, vec![0, 2, 1, 3, 4]);
            g.reshape_(t, i64s(&[1, hp, wp, c]))
        };
        let y = self.roll(y, 1, hp, shift as isize);
        let y = self.roll(y, 2, wp, shift as isize);
        let x = {
            let mut g = self.g();
            let y = if (hp, wp) != (h, w) {
                let t = g.narrow_(y, 1, 0, h);
                g.narrow_(t, 2, 0, w)
            } else {
                y
            };
            let y = g.reshape_(y, i64s(&[1, h * w, c]));
            g.add(x, y)
        };
        let m = self.ln(x, &format!("{prefix}.norm2"))?;
        let m = self.linear(m, &format!("{prefix}.mlp.fc1"), true)?;
        let m = self.g().gelu(m);
        let m = self.linear(m, &format!("{prefix}.mlp.fc2"), true)?;
        Ok(self.g().add(x, m))
    }

    fn patch_merge(
        &mut self,
        x: HirNodeId,
        prefix: &str,
        c: usize,
        h: usize,
        w: usize,
    ) -> Result<(HirNodeId, usize, usize)> {
        let (h2, w2) = (h.div_ceil(2), w.div_ceil(2));
        let parts = {
            let mut g = self.g();
            let t = g.reshape_(x, i64s(&[1, h, w, c]));
            let t = if h % 2 == 1 || w % 2 == 1 {
                g.pad_(
                    t,
                    vec![[0, 0], [0, h % 2], [0, w % 2], [0, 0]],
                    PadMode::Constant(0.0),
                )
            } else {
                t
            };
            let t = g.reshape_(t, i64s(&[h2, 2, w2, 2, c]));
            let mut parts = Vec::new();
            for (dy, dx) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                let p = g.narrow_(t, 1, dy, 1);
                let p = g.narrow_(p, 3, dx, 1);
                parts.push(g.reshape_(p, i64s(&[1, h2 * w2, c])));
            }
            g.concat_(parts, 2)
        };
        let y = self.ln(parts, &format!("{prefix}.norm"))?;
        let y = self.linear(y, &format!("{prefix}.reduction"), false)?;
        Ok((y, h2, w2))
    }

    /// Swin backbone: the four `norm{i}`-ed stage outputs.
    fn swin(&mut self, img: T) -> Result<Vec<T>> {
        let pe = self.conv_pad(img, "bb.patch_embed.proj", None, 4, Some(0), false)?;
        let (mut h, mut w, mut c) = (pe.h, pe.w, pe.c);
        let tok = {
            let mut g = self.g();
            let t = g.reshape_(pe.id, i64s(&[1, c, h * w]));
            g.transpose_(t, vec![0, 2, 1])
        };
        let mut x = if self.w.has("bb.patch_embed.norm.weight") {
            self.ln(tok, "bb.patch_embed.norm")?
        } else {
            tok
        };
        let mut outs = Vec::new();
        for s in 0.. {
            if !self.w.has(&format!("bb.layers.{s}.blocks.0.norm1.weight")) {
                break;
            }
            let table = self
                .w
                .get(&format!(
                    "bb.layers.{s}.blocks.0.attn.relative_position_bias_table"
                ))?
                .1
                .clone();
            let heads = table[1];
            self.ws = ((table[0] as f64).sqrt() as usize).div_ceil(2);
            ensure!(
                (2 * self.ws - 1).pow(2) == table[0],
                "bad relative-position table {table:?}"
            );
            for b in 0.. {
                let p = format!("bb.layers.{s}.blocks.{b}");
                if !self.w.has(&format!("{p}.norm1.weight")) {
                    break;
                }
                let shift = if b % 2 == 1 { self.ws / 2 } else { 0 };
                x = self.swin_block(x, &p, c, heads, h, w, shift)?;
            }
            let o = self.ln(x, &format!("bb.norm{s}"))?;
            outs.push(self.tokens_to_map(o, c, h, w));
            if self
                .w
                .has(&format!("bb.layers.{s}.downsample.reduction.weight"))
            {
                let (nx, nh, nw) =
                    self.patch_merge(x, &format!("bb.layers.{s}.downsample"), c, h, w)?;
                x = nx;
                h = nh;
                w = nw;
                c *= 2;
            }
        }
        Ok(outs)
    }

    // ── Decoder ────────────────────────────────────────────────────────────

    /// torchvision `deform_conv2d` with DCNv2 modulation, stride 1, + folded BN + ReLU.
    fn deform(&mut self, x: T, prefix: &str, bn: &str) -> Result<T> {
        let (wk, ws, b) = self.w.conv(&format!("{prefix}.regular_conv"), Some(bn))?;
        let (o, cin, kh, kw) = (ws[0], ws[1], ws[2], ws[3]);
        ensure!(cin == x.c, "{prefix}: {} input channels, kernel {cin}", x.c);
        let k = kh * kw;
        let (h, w) = (x.h, x.w);
        let (ph, pw) = (kh / 2, kw / 2);
        let off = self.conv(x, &format!("{prefix}.offset_conv"), None, 1, false)?;
        let m = self.conv(x, &format!("{prefix}.modulator_conv"), None, 1, false)?;
        // Base sampling positions per tap, in pixels: (y - ph + i, x - pw + j).
        let mut by = vec![0.0f32; k * h * w];
        let mut bx = vec![0.0f32; k * h * w];
        for t in 0..k {
            let (i, j) = (t / kw, t % kw);
            for y in 0..h {
                for xx in 0..w {
                    by[(t * h + y) * w + xx] = y as f32 - ph as f32 + i as f32;
                    bx[(t * h + y) * w + xx] = xx as f32 - pw as f32 + j as f32;
                }
            }
        }
        let by = self.param(by, &[1, k, h, w]);
        let bx = self.param(bx, &[1, k, h, w]);
        // align_corners=True normalization: g = 2p/(n-1) - 1.
        let sy = self.scalar_like(2.0 / (h.max(2) - 1) as f32, &[1, k, h, w]);
        let sx = self.scalar_like(2.0 / (w.max(2) - 1) as f32, &[1, k, h, w]);
        let one = self.scalar_like(1.0, &[1, k, h, w]);
        let two = self.scalar_like(2.0, &[1, 1, k, h * w]);
        let wk = self.param(wk, &[o, cin * k]);
        let bid = self.param(b, &[1, o, 1]);
        let mut g = self.g();
        let off5 = g.reshape_(off.id, i64s(&[1, k, 2, h, w]));
        let dy = g.narrow_(off5, 2, 0, 1);
        let dy = g.reshape_(dy, i64s(&[1, k, h, w]));
        let dx = g.narrow_(off5, 2, 1, 1);
        let dx = g.reshape_(dx, i64s(&[1, k, h, w]));
        let py = g.add(by, dy);
        let px = g.add(bx, dx);
        let gy = g.mul(py, sy);
        let gy = g.sub(gy, one);
        let gx = g.mul(px, sx);
        let gx = g.sub(gx, one);
        let gx = g.reshape_(gx, i64s(&[1, k * h, w, 1]));
        let gy = g.reshape_(gy, i64s(&[1, k * h, w, 1]));
        let grid = g.concat_(vec![gx, gy], 3);
        let s = g.grid_sample2d(x.id, grid, GridMode::Bilinear, GridPad::Zeros, true);
        let s = g.reshape_(s, i64s(&[1, cin, k, h * w]));
        let mm = g.activation(Activation::Sigmoid, m.id, f(&[1, k, h, w]));
        let mm = g.reshape_(mm, i64s(&[1, 1, k, h * w]));
        let mm = g.mul(mm, two);
        let mm = g.expand_(mm, i64s(&[1, cin, k, h * w]));
        let s = g.mul(s, mm);
        let cols = g.reshape_(s, i64s(&[cin * k, h * w]));
        let y = g.mm(wk, cols);
        let y = g.reshape_(y, i64s(&[1, o, h * w]));
        let bb = g.expand_(bid, i64s(&[1, o, h * w]));
        let y = g.add(y, bb);
        let y = g.relu(y);
        let y = g.reshape_(y, i64s(&[1, o, h, w]));
        Ok(T { id: y, c: o, h, w })
    }

    fn aspp_deformable(&mut self, x: T, p: &str) -> Result<T> {
        let mut branches = vec![self.deform(
            x,
            &format!("{p}.aspp1.atrous_conv"),
            &format!("{p}.aspp1.bn"),
        )?];
        for i in 0.. {
            let q = format!("{p}.aspp_deforms.{i}");
            if !self.w.has(&format!("{q}.atrous_conv.regular_conv.weight")) {
                break;
            }
            branches.push(self.deform(x, &format!("{q}.atrous_conv"), &format!("{q}.bn"))?);
        }
        let pooled = T {
            id: self.g().mean(x.id, vec![2, 3], true),
            h: 1,
            w: 1,
            ..x
        };
        let gp = self.conv(
            pooled,
            &format!("{p}.global_avg_pool.1"),
            Some(&format!("{p}.global_avg_pool.2")),
            1,
            true,
        )?;
        branches.push(self.resize(gp, x.h, x.w));
        let cat = self.cat(&branches);
        self.conv(
            cat,
            &format!("{p}.conv1"),
            Some(&format!("{p}.bn1")),
            1,
            true,
        )
    }

    fn dec_block(&mut self, x: T, p: &str) -> Result<T> {
        let y = self.conv(
            x,
            &format!("{p}.conv_in"),
            Some(&format!("{p}.bn_in")),
            1,
            true,
        )?;
        let y = if self.w.has(&format!(
            "{p}.dec_att.aspp1.atrous_conv.regular_conv.weight"
        )) {
            self.aspp_deformable(y, &format!("{p}.dec_att"))?
        } else if self.w.has(&format!("{p}.dec_att.conv1.weight")) {
            bail!("{p}: plain ASPP decoder attention is not implemented (only ASPPDeformable)")
        } else {
            y
        };
        self.conv(
            y,
            &format!("{p}.conv_out"),
            Some(&format!("{p}.bn_out")),
            1,
            false,
        )
    }

    /// `image2patches(x, patch_ref)` + `SimpleConvs`: the image as block
    /// space-to-depth `c (G h) (G w) → (c G G) h w` at `ref`'s size.
    fn ipt(&mut self, img: T, r: T, blk: &str) -> Result<T> {
        let gsz = img.h / r.h;
        ensure!(
            img.h == gsz * r.h && img.w == gsz * r.w,
            "{blk}: image {}x{} not a multiple of {}x{}",
            img.h,
            img.w,
            r.h,
            r.w
        );
        let p = {
            let mut g = self.g();
            let t = g.reshape_(img.id, i64s(&[3, gsz, r.h, gsz, r.w]));
            let t = g.transpose_(t, vec![0, 1, 3, 2, 4]);
            g.reshape_(t, i64s(&[1, 3 * gsz * gsz, r.h, r.w]))
        };
        let p = T {
            id: p,
            c: 3 * gsz * gsz,
            h: r.h,
            w: r.w,
        };
        let p = self.conv(p, &format!("{blk}.conv1"), None, 1, false)?;
        self.conv(p, &format!("{blk}.conv_out"), None, 1, false)
    }

    fn gdt(&mut self, p: T, lvl: usize) -> Result<T> {
        let pre = format!("decoder.gdt_convs_{lvl}");
        let feat = self.conv(p, &format!("{pre}.0"), Some(&format!("{pre}.1")), 1, true)?;
        let a = self.conv(
            feat,
            &format!("decoder.gdt_convs_attn_{lvl}.0"),
            None,
            1,
            false,
        )?;
        let mut g = self.g();
        let a = g.activation(Activation::Sigmoid, a.id, f(&[1, 1, p.h, p.w]));
        let a = g.expand_(a, i64s(&[1, p.c, p.h, p.w]));
        Ok(T {
            id: g.mul(p.id, a),
            ..p
        })
    }
}

/// Build the graph for a `side × side` input; outputs the requested `taps`
/// (see [`TAPS`]) flattened and concatenated in order.
pub fn build(weights: Arc<Weights>, side: usize, taps: &[&str]) -> Result<BuiltModel> {
    ensure!(
        side.is_multiple_of(32),
        "BiRefNet input side must be a multiple of 32, got {side}"
    );
    for t in taps {
        ensure!(
            TAPS.contains(t) || DEBUG_TAPS.contains(t),
            "unknown tap {t}"
        );
    }
    let taps: Vec<String> = taps.iter().map(|s| s.to_string()).collect();
    ModelFlow::new("birefnet")
        .input("image", f(&[1, 3, side, side]))
        .plugin_named("birefnet", move |emit, _| {
            let x = emit.flow_input("image")?.hir_id();
            let mut g = G {
                emit,
                w: weights.clone(),
                n: 0,
                ws: 7,
            };
            let img = T {
                id: x,
                c: 3,
                h: side,
                w: side,
            };
            let mut tap: HashMap<&str, T> = HashMap::new();

            // Encoder: full + half resolution, concatenated per stage.
            let full = g.swin(img)?;
            for (i, t) in full.iter().enumerate() {
                tap.insert(TAPS[i], *t);
            }
            let half_img = g.resize(img, side / 2, side / 2);
            let half = g.swin(half_img)?;
            let mut enc: Vec<T> = Vec::new();
            for (a, b) in full.iter().zip(&half) {
                let up = g.resize(*b, a.h, a.w);
                enc.push(g.cat(&[*a, up]));
            }
            // Context: shallower stages resized onto the deepest.
            let x4 = enc[3];
            let mut ctx: Vec<T> = enc[..3].iter().map(|t| g.resize(*t, x4.h, x4.w)).collect();
            ctx.push(x4);
            let x4 = g.cat(&ctx);
            let enc4 = x4;
            for (i, t) in enc[..3].iter().enumerate() {
                tap.insert(TAPS[4 + i], *t);
            }
            tap.insert("enc4", enc4);
            let x4 = if g.w.has("squeeze_module.0.conv_in.weight") {
                g.dec_block(x4, "squeeze_module.0")?
            } else {
                x4
            };
            tap.insert("sq4", x4);
            let (x1, x2, x3) = (enc[0], enc[1], enc[2]);

            // Decoder.
            let ip = g.ipt(img, x4, "decoder.ipt_blk5")?;
            let c4 = g.cat(&[x4, ip]);
            let p4 = g.dec_block(c4, "decoder.decoder_block4")?;
            tap.insert("d4", p4);
            let p4 = g.gdt(p4, 4)?;
            tap.insert("g4", p4);
            let up = g.resize(p4, x3.h, x3.w);
            let lat = g.conv(x3, "decoder.lateral_block4.conv", None, 1, false)?;
            let p3 = T {
                id: g.g().add(up.id, lat.id),
                ..lat
            };
            let ip = g.ipt(img, p3, "decoder.ipt_blk4")?;
            let c3 = g.cat(&[p3, ip]);
            let p3 = g.dec_block(c3, "decoder.decoder_block3")?;
            tap.insert("d3", p3);
            let p3 = g.gdt(p3, 3)?;
            tap.insert("g3", p3);
            let up = g.resize(p3, x2.h, x2.w);
            let lat = g.conv(x2, "decoder.lateral_block3.conv", None, 1, false)?;
            let p2 = T {
                id: g.g().add(up.id, lat.id),
                ..lat
            };
            let ip = g.ipt(img, p2, "decoder.ipt_blk3")?;
            let c2 = g.cat(&[p2, ip]);
            let p2 = g.dec_block(c2, "decoder.decoder_block2")?;
            tap.insert("d2", p2);
            let p2 = g.gdt(p2, 2)?;
            tap.insert("g2", p2);
            let up = g.resize(p2, x1.h, x1.w);
            let lat = g.conv(x1, "decoder.lateral_block2.conv", None, 1, false)?;
            let p1 = T {
                id: g.g().add(up.id, lat.id),
                ..lat
            };
            let ip = g.ipt(img, p1, "decoder.ipt_blk2")?;
            let c1 = g.cat(&[p1, ip]);
            let p1 = g.dec_block(c1, "decoder.decoder_block1")?;
            tap.insert("d1", p1);
            let p1 = g.resize(p1, side, side);
            tap.insert("up1", p1);
            let ip = g.ipt(img, p1, "decoder.ipt_blk1")?;
            let c0 = g.cat(&[p1, ip]);
            let logits = g.conv(c0, "decoder.conv_out1.0", None, 1, false)?;
            tap.insert("logits", logits);

            let mut flat = Vec::new();
            let mut total = 0;
            for t in &taps {
                let v = *tap.get(t.as_str()).context("tap not produced")?;
                let n = v.c * v.h * v.w;
                total += n;
                flat.push(g.g().reshape_(v.id, vec![n as i64]));
            }
            let out = if flat.len() == 1 {
                flat[0]
            } else {
                g.g().concat_(flat, 0)
            };
            Ok(Some(g.emit.wrap(out, f(&[total]))))
        })
        .output("out")
        .build(&mut MapWeights::default())
}
