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

//! The graph-building layer every architecture is written against.
//!
//! [`Builder`] owns an [`HirModule`] plus the host-side parameter blobs, and
//! exposes the handful of primitives these networks are made of: convolutions,
//! linears, the activations PyTorch uses, pixel shuffle, and nearest upsample.
//! Weight names are the checkpoint's own, so a block's code reads like the
//! module it ports.
//!
//! # Activations rlx does not have natively
//!
//! `Activation` covers GELU, SiLU, Mish, Sigmoid and friends, but not the two
//! leaky families these nets lean on. Both are composed rather than added as
//! ops, since the identity is exact and costs two extra elementwise passes:
//!
//! ```text
//!   leaky_relu(x, s) = relu(x) − s·relu(−x)
//!   prelu(x, a)      = relu(x) − a ⊙ relu(−x)      a broadcast as [1, C, 1, 1]
//! ```
//!
//! # Pixel shuffle
//!
//! [`Builder::pixel_shuffle`] is the reshape/permute/reshape PyTorch uses,
//! written at rank 5 (see `rank5`). The permutation is not a last-two-axis
//! swap, so on CPU it falls to the
//! general N-D index walk — which is why it is emitted once per forward (at the
//! very end, on the smallest tensor in the graph) and never inside a block.
//! [`Builder::nearest_upsample`] deliberately avoids it: an `Expand` over two
//! inserted unit axes produces pixel repetition with no permutation at all.

use anyhow::{Context, Result, ensure};
use rlx_core::vision_ops_ir::{conv2d_bias, conv2d_bias_groups, conv2d_no_bias, nchw_shape};
use rlx_core::weight_map::WeightMap;
use rlx_ir::hir::{HirModule, HirMut, HirNodeId};
use rlx_ir::op::{Activation, BinaryOp, Op, ReduceOp};
use rlx_ir::{DType, HirGraphExt, Shape};
use std::collections::HashMap;

/// NCHW extents of a rank-4 node.
pub type Dims4 = [usize; 4];

/// Graph under construction: HIR plus the host blobs its params refer to.
pub struct Builder {
    hir: HirModule,
    params: HashMap<String, Vec<f32>>,
    consts: usize,
    /// When set, a weight the checkpoint does not have is synthesized from this
    /// seed instead of failing. See [`Builder::with_autofill`].
    autofill: Option<u64>,
    synthesized: Vec<String>,
}

impl Builder {
    pub fn new(name: &str) -> Self {
        Self {
            hir: HirModule::new(name),
            params: HashMap::new(),
            consts: 0,
            autofill: None,
            synthesized: Vec::new(),
        }
    }

    /// Synthesize any weight the checkpoint is missing, from `seed`.
    ///
    /// This exists so a whole architecture can be built and *run* without the
    /// real (often multi-hundred-megabyte) checkpoint. It does not test
    /// numerics — nothing can, without the trained weights — but it does
    /// exercise every shape, reshape and permutation in the graph, which is
    /// where porting bugs actually live. Weights are drawn from a small
    /// deterministic LCG so a run is reproducible.
    ///
    /// Never enable this on an inference path: it would turn a mis-detected
    /// architecture from a hard error into a plausible-looking wrong image.
    pub fn with_autofill(name: &str, seed: u64) -> Self {
        let mut b = Self::new(name);
        b.autofill = Some(seed.max(1));
        b
    }

    /// Names synthesized by [`Self::with_autofill`], in request order.
    pub fn synthesized(&self) -> &[String] {
        &self.synthesized
    }

    /// Deterministic small values — centred on zero so a deep stack neither
    /// saturates nor collapses, which would make a shape test pass vacuously
    /// on all-zero activations.
    fn synth(&mut self, key: &str, want: &[usize]) -> Vec<f32> {
        let n: usize = want.iter().product();
        let mut s = self
            .autofill
            .expect("synth is only reachable with autofill on")
            .wrapping_add(
                key.bytes()
                    .fold(0u64, |a, b| a.wrapping_mul(31).wrapping_add(b as u64)),
            );
        self.synthesized.push(key.to_string());
        // A running variance is a sum of squares; a signed fill would send
        // `sqrt(var + eps)` to NaN and turn a shape test into a mystery.
        let nonneg = key.ends_with("running_var");
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let u = (s >> 33) as f32 / (1u64 << 31) as f32;
                if nonneg {
                    u * 0.1 + 0.05
                } else {
                    (u - 0.5) * 0.1
                }
            })
            .collect()
    }

    pub fn g(&mut self) -> HirMut<'_> {
        HirMut::new(&mut self.hir)
    }

    /// Declare the network input as NCHW `[1, c, h, w]`.
    pub fn input(&mut self, c: usize, h: usize, w: usize) -> HirNodeId {
        let shape = nchw_shape(1, c, h, w, DType::F32);
        self.g().input("image", shape)
    }

    pub fn finish(
        self,
        out: HirNodeId,
    ) -> Result<(rlx_ir::Graph, HashMap<String, Vec<f32>>, usize)> {
        let mut hir = self.hir;
        HirMut::new(&mut hir).set_outputs(vec![out]);
        let nodes = hir.nodes().len();
        let (graph, params) = rlx_core::flow_util::graph_from_hir(hir, self.params)?;
        Ok((graph, params, nodes))
    }

    /// NCHW extents of `x`. Panics if `x` is not rank 4 — every caller here
    /// works in NCHW and a rank slip is a porting bug, not a runtime condition.
    pub fn dims4(&mut self, x: HirNodeId) -> Dims4 {
        let s = self.g().shape(x).clone();
        assert_eq!(s.rank(), 4, "expected NCHW, got rank {}", s.rank());
        [
            s.dim(0).unwrap_static(),
            s.dim(1).unwrap_static(),
            s.dim(2).unwrap_static(),
            s.dim(3).unwrap_static(),
        ]
    }

    /// Extents of `x` at any rank.
    pub fn dims(&mut self, x: HirNodeId) -> Vec<usize> {
        let s = self.g().shape(x).clone();
        (0..s.rank()).map(|i| s.dim(i).unwrap_static()).collect()
    }

    // ── parameters ──────────────────────────────────────────────────────

    /// Take `key` from the checkpoint, asserting its shape.
    pub fn take(&mut self, wm: &mut WeightMap, key: &str, want: &[usize]) -> Result<HirNodeId> {
        if self.autofill.is_some() && !wm.has(key) {
            let data = self.synth(key, want);
            return Ok(self.raw_param(key, data, want));
        }
        let (data, shape) = wm
            .take(key)
            .with_context(|| format!("missing weight {key}"))?;
        ensure!(
            shape == want,
            "weight {key} has shape {shape:?}, the graph needs {want:?}"
        );
        Ok(self.raw_param(key, data, want))
    }

    /// Take `key` and reinterpret it under a different — same-element-count —
    /// shape. Used where PyTorch stores a logically N-D buffer flattened.
    pub fn take_as(&mut self, wm: &mut WeightMap, key: &str, want: &[usize]) -> Result<HirNodeId> {
        if self.autofill.is_some() && !wm.has(key) {
            let data = self.synth(key, want);
            return Ok(self.raw_param(key, data, want));
        }
        let (data, shape) = wm
            .take(key)
            .with_context(|| format!("missing weight {key}"))?;
        let have: usize = shape.iter().product();
        let need: usize = want.iter().product();
        ensure!(
            have == need,
            "weight {key} has {have} elements ({shape:?}), the graph needs {need} ({want:?})"
        );
        Ok(self.raw_param(key, data, want))
    }

    /// Read `key`'s data to the host rather than into the graph.
    ///
    /// For parameters that are *resolved at build time* — an index table
    /// gathered through a frozen bias table, say — rather than executed.
    pub fn take_host(&mut self, wm: &mut WeightMap, key: &str, want: &[usize]) -> Result<Vec<f32>> {
        if self.autofill.is_some() && !wm.has(key) {
            return Ok(self.synth(key, want));
        }
        let (data, shape) = wm
            .take(key)
            .with_context(|| format!("missing weight {key}"))?;
        ensure!(
            shape == want,
            "weight {key} has shape {shape:?}, the graph needs {want:?}"
        );
        Ok(data)
    }

    /// A host-computed tensor injected as a parameter.
    pub fn constant(&mut self, tag: &str, data: Vec<f32>, shape: &[usize]) -> HirNodeId {
        self.consts += 1;
        let name = format!("const.{tag}.{}", self.consts);
        self.raw_param(&name, data, shape)
    }

    /// A scalar broadcastable over NCHW.
    ///
    /// Prefer [`Self::scalar_like`] unless the operand is known to be rank 4:
    /// broadcasting a rank-4 `[1, 1, 1, 1]` against a rank-3 tensor *promotes*
    /// the result to rank 4, which every downstream `reshape` happily absorbs
    /// and every downstream `transpose` then rejects — a long way from the
    /// multiply that caused it.
    pub fn splat(&mut self, tag: &str, v: f32) -> HirNodeId {
        self.constant(tag, vec![v], &[1, 1, 1, 1])
    }

    /// A scalar shaped `[1; rank(x)]`, so combining it with `x` cannot change
    /// `x`'s rank.
    pub fn scalar_like(&mut self, tag: &str, v: f32, x: HirNodeId) -> HirNodeId {
        let rank = self.g().shape(x).rank();
        self.constant(tag, vec![v], &vec![1; rank])
    }

    fn raw_param(&mut self, name: &str, data: Vec<f32>, shape: &[usize]) -> HirNodeId {
        let id = self.g().param(name, Shape::new(shape, DType::F32));
        self.params.insert(name.to_string(), data);
        id
    }

    pub fn has(&self, wm: &WeightMap, key: &str) -> bool {
        wm.has(key)
    }

    // ── layers ──────────────────────────────────────────────────────────

    /// `nn.Conv2d` named `{prefix}.weight` / `{prefix}.bias`, "same"-style
    /// padding taken from the caller. `bias` selects whether a bias is read.
    #[allow(clippy::too_many_arguments)]
    pub fn conv(
        &mut self,
        wm: &mut WeightMap,
        prefix: &str,
        x: HirNodeId,
        out_c: usize,
        k: [usize; 2],
        stride: [usize; 2],
        pad: [usize; 2],
        groups: usize,
        bias: bool,
    ) -> Result<HirNodeId> {
        let [n, in_c, h, w] = self.dims4(x);
        ensure!(
            in_c % groups == 0 && out_c.is_multiple_of(groups),
            "{prefix}: {in_c}→{out_c} channels do not split into {groups} groups"
        );
        let wshape = [out_c, in_c / groups, k[0], k[1]];
        let weight = self.take(wm, &format!("{prefix}.weight"), &wshape)?;
        let bias = if bias {
            Some(self.take(wm, &format!("{prefix}.bias"), &[out_c])?)
        } else {
            None
        };
        let out_h = (h + 2 * pad[0] - k[0]) / stride[0] + 1;
        let out_w = (w + 2 * pad[1] - k[1]) / stride[1] + 1;

        let mut g = HirMut::new(&mut self.hir);
        let y = match bias {
            Some(b) if groups == 1 => conv2d_bias(
                &mut g, x, weight, b, n, out_c, k[0], k[1], stride, pad, out_h, out_w,
            ),
            Some(b) => conv2d_bias_groups(
                &mut g, x, weight, b, n, out_c, k[0], k[1], stride, pad, groups, out_h, out_w,
            ),
            None if groups == 1 => conv2d_no_bias(
                &mut g, x, weight, n, out_c, k[0], k[1], stride, pad, out_h, out_w,
            ),
            None => {
                let out_shape = nchw_shape(n, out_c, out_h, out_w, DType::F32);
                g.conv2d(x, weight, k, stride, pad, groups, out_shape)
            }
        };
        Ok(y)
    }

    /// `nn.ConvTranspose2d` with a bias, as Real-CUGAN uses it.
    ///
    /// The weight layout is **transposed relative to a forward convolution**:
    /// PyTorch stores `[C_in, C_out / groups, kH, kW]`, so the checkpoint's
    /// first axis is the *input* width. Declaring it the other way round loads
    /// a real checkpoint without complaint whenever the two happen to match —
    /// which, in a U-Net whose upsamplers are all 64→64, is every time.
    pub fn conv_transpose(
        &mut self,
        wm: &mut WeightMap,
        prefix: &str,
        x: HirNodeId,
        out_c: usize,
        k: [usize; 2],
        stride: [usize; 2],
        pad: [usize; 2],
    ) -> Result<HirNodeId> {
        let [n, in_c, h, w] = self.dims4(x);
        let weight = self.take(wm, &format!("{prefix}.weight"), &[in_c, out_c, k[0], k[1]])?;
        let bias = self.take(wm, &format!("{prefix}.bias"), &[out_c])?;

        let out_h = (h - 1) * stride[0] + k[0] - 2 * pad[0];
        let out_w = (w - 1) * stride[1] + k[1] - 2 * pad[1];
        let shape = nchw_shape(n, out_c, out_h, out_w, DType::F32);

        let y = self
            .g()
            .conv_transpose2d(x, weight, k, stride, pad, [1, 1], [0, 0], 1, shape);
        // Bias is a per-channel add, broadcast over the spatial axes.
        let b4 = self.g().reshape_(bias, vec![1, out_c as i64, 1, 1]);
        Ok(self.g().add(y, b4))
    }

    /// The overwhelmingly common case: 3×3, stride 1, pad 1, biased.
    pub fn conv3x3(
        &mut self,
        wm: &mut WeightMap,
        prefix: &str,
        x: HirNodeId,
        out_c: usize,
    ) -> Result<HirNodeId> {
        self.conv(wm, prefix, x, out_c, [3, 3], [1, 1], [1, 1], 1, true)
    }

    /// 1×1, stride 1, no padding, biased.
    pub fn conv1x1(
        &mut self,
        wm: &mut WeightMap,
        prefix: &str,
        x: HirNodeId,
        out_c: usize,
    ) -> Result<HirNodeId> {
        self.conv(wm, prefix, x, out_c, [1, 1], [1, 1], [0, 0], 1, true)
    }

    /// `nn.Linear` over the last axis of a rank-3 `[B, N, in]` tensor.
    ///
    /// PyTorch stores `weight` as `[out, in]` for `x @ Wᵀ`; it is transposed on
    /// load so the graph can use a plain `MatMul`.
    pub fn linear(
        &mut self,
        wm: &mut WeightMap,
        prefix: &str,
        x: HirNodeId,
        out_f: usize,
        bias: bool,
    ) -> Result<HirNodeId> {
        let d = self.dims(x);
        let in_f = *d.last().expect("linear input must have a last axis");
        let key = format!("{prefix}.weight");
        if self.autofill.is_some() && !wm.has(&key) {
            let data = self.synth(&key, &[in_f, out_f]);
            let w = self.raw_param(&key, data, &[in_f, out_f]);
            let mut y = self.g().mm(x, w);
            if bias {
                let b = self.take(wm, &format!("{prefix}.bias"), &[out_f])?;
                y = self.g().add(y, b);
            }
            return Ok(y);
        }
        let (data, shape) = wm
            .take_transposed(&key)
            .with_context(|| format!("missing weight {key}"))?;
        ensure!(
            shape == [in_f, out_f],
            "weight {key} is {shape:?} after transpose, the graph needs [{in_f}, {out_f}]"
        );
        let w = self.raw_param(&key, data, &[in_f, out_f]);
        let mut y = self.g().mm(x, w);
        if bias {
            let b = self.take(wm, &format!("{prefix}.bias"), &[out_f])?;
            y = self.g().add(y, b);
        }
        Ok(y)
    }

    /// `nn.LayerNorm` over the last axis of a rank-3 tensor.
    pub fn layer_norm(
        &mut self,
        wm: &mut WeightMap,
        prefix: &str,
        x: HirNodeId,
        eps: f32,
    ) -> Result<HirNodeId> {
        let d = self.dims(x);
        let n = *d.last().expect("layer norm input must have a last axis");
        let gamma = self.take(wm, &format!("{prefix}.weight"), &[n])?;
        let beta = self.take(wm, &format!("{prefix}.bias"), &[n])?;
        Ok(self.g().ln(x, gamma, beta, eps))
    }

    /// `nn.GroupNorm` over NCHW.
    pub fn group_norm(
        &mut self,
        wm: &mut WeightMap,
        prefix: &str,
        x: HirNodeId,
        groups: usize,
        eps: f32,
    ) -> Result<HirNodeId> {
        let [_, c, _, _] = self.dims4(x);
        let gamma = self.take(wm, &format!("{prefix}.weight"), &[c])?;
        let beta = self.take(wm, &format!("{prefix}.bias"), &[c])?;
        Ok(self.g().group_norm(x, gamma, beta, groups, eps))
    }

    /// The channel-first `LayerNorm` RealPLKSR defines by hand: normalize over
    /// C at each pixel, then scale/shift per channel.
    pub fn layer_norm_nchw(
        &mut self,
        wm: &mut WeightMap,
        prefix: &str,
        x: HirNodeId,
        eps: f32,
    ) -> Result<HirNodeId> {
        let [_, c, _, _] = self.dims4(x);
        let gamma = self.take_as(wm, &format!("{prefix}.weight"), &[1, c, 1, 1])?;
        let beta = self.take_as(wm, &format!("{prefix}.bias"), &[1, c, 1, 1])?;
        // One group spanning all channels is exactly a per-pixel LayerNorm over
        // C — but GroupNorm normalizes over (C, H, W) per group, so it is *not*
        // interchangeable here. Compose the moments explicitly.
        let mean = self.g().mean(x, vec![1], true);
        let centered = self.g().sub(x, mean);
        let sq = self.g().mul(centered, centered);
        let var = self.g().mean(sq, vec![1], true);
        let eps_c = self.splat("ln_eps", eps);
        let var_eps = self.g().add(var, eps_c);
        let std = self.g().sqrt(var_eps);
        let normed = self.g().div(centered, std);
        let scaled = self.g().mul(normed, gamma);
        Ok(self.g().add(scaled, beta))
    }

    // ── activations ─────────────────────────────────────────────────────

    pub fn act(&mut self, a: Activation, x: HirNodeId) -> HirNodeId {
        let shape = self.g().shape(x).clone();
        self.g().activation(a, x, shape)
    }

    pub fn relu(&mut self, x: HirNodeId) -> HirNodeId {
        self.act(Activation::Relu, x)
    }

    pub fn sigmoid(&mut self, x: HirNodeId) -> HirNodeId {
        self.act(Activation::Sigmoid, x)
    }

    pub fn gelu(&mut self, x: HirNodeId) -> HirNodeId {
        self.act(Activation::Gelu, x)
    }

    pub fn silu(&mut self, x: HirNodeId) -> HirNodeId {
        self.act(Activation::Silu, x)
    }

    pub fn mish(&mut self, x: HirNodeId) -> HirNodeId {
        self.act(Activation::Mish, x)
    }

    /// `relu(x) − slope·relu(−x)`, exact for any slope.
    pub fn leaky_relu(&mut self, x: HirNodeId, slope: f32) -> HirNodeId {
        let pos = self.relu(x);
        let negated = self.g().neg(x);
        let neg = self.relu(negated);
        let s = self.scalar_like("lrelu", slope, x);
        let scaled = self.g().mul(neg, s);
        self.g().sub(pos, scaled)
    }

    /// `relu(x) − a ⊙ relu(−x)` with a per-channel slope read from the
    /// checkpoint. A single-element `a` is the shared-slope PReLU.
    pub fn prelu(&mut self, wm: &mut WeightMap, prefix: &str, x: HirNodeId) -> Result<HirNodeId> {
        let [_, c, _, _] = self.dims4(x);
        let key = format!("{prefix}.weight");
        if self.autofill.is_some() && !wm.has(&key) {
            let data = self.synth(&key, &[c]);
            let a = self.raw_param(&key, data, &[1, c, 1, 1]);
            let pos = self.relu(x);
            let negated = self.g().neg(x);
            let neg = self.relu(negated);
            let scaled = self.g().mul(neg, a);
            return Ok(self.g().sub(pos, scaled));
        }
        let (data, shape) = wm
            .take(&key)
            .with_context(|| format!("missing weight {key}"))?;
        let n: usize = shape.iter().product();
        ensure!(
            n == c || n == 1,
            "PReLU {key} has {n} slopes, expected 1 or {c}"
        );
        let a = self.raw_param(&key, data, &[1, n, 1, 1]);
        let pos = self.relu(x);
        let negated = self.g().neg(x);
        let neg = self.relu(negated);
        let scaled = self.g().mul(neg, a);
        Ok(self.g().sub(pos, scaled))
    }

    // ── resampling ──────────────────────────────────────────────────────

    /// `nn.PixelShuffle(r)`: `[N, C·r², H, W]` → `[N, C, H·r, W·r]`, with
    /// `out[n, c, h·r+i, w·r+j] = in[n, c·r² + i·r + j, h, w]`.
    pub fn pixel_shuffle(&mut self, x: HirNodeId, r: usize) -> Result<HirNodeId> {
        let [n, c_in, h, w] = self.dims4(x);
        ensure!(
            c_in % (r * r) == 0,
            "pixel shuffle ×{r} needs a multiple of {} channels, got {c_in}",
            r * r
        );
        if r == 1 {
            return Ok(x);
        }
        let c = c_in / (r * r);
        rank5(n, "pixel shuffle")?;
        // The batch axis is elided: `[C, r, r, H, W]`, not `[1, C, r, r, H, W]`.
        // CoreML's MIL rejects any tensor above rank 5, and a pixel shuffle is
        // the most common place in this crate to exceed it — see `rank5`.
        let five = self
            .g()
            .reshape_(x, vec![c as i64, r as i64, r as i64, h as i64, w as i64]);
        // [C, r, r, H, W] → [C, H, r, W, r]
        let perm = self.g().transpose_(five, vec![0, 3, 1, 4, 2]);
        Ok(self
            .g()
            .reshape_(perm, vec![1, c as i64, (h * r) as i64, (w * r) as i64]))
    }

    /// `nn.PixelUnshuffle(r)`: `[N, C, H, W]` → `[N, C·r², H/r, W/r]`, the exact
    /// inverse of [`Self::pixel_shuffle`].
    ///
    /// Real-ESRGAN's ×1 and ×2 RRDB variants feed the network an unshuffled
    /// input so the trunk always runs at ×4, then divide the scale back out.
    pub fn pixel_unshuffle(&mut self, x: HirNodeId, r: usize) -> Result<HirNodeId> {
        if r == 1 {
            return Ok(x);
        }
        let [n, c, h, w] = self.dims4(x);
        ensure!(
            h % r == 0 && w % r == 0,
            "pixel unshuffle ÷{r} needs a multiple of {r} in both extents, got {h}×{w}"
        );
        rank5(n, "pixel unshuffle")?;
        let five = self.g().reshape_(
            x,
            vec![c as i64, (h / r) as i64, r as i64, (w / r) as i64, r as i64],
        );
        // [C, H', r, W', r] → [C, r, r, H', W'], the inverse permutation of the
        // one pixel shuffle applies.
        let perm = self.g().transpose_(five, vec![0, 2, 4, 1, 3]);
        Ok(self.g().reshape_(
            perm,
            vec![1, (c * r * r) as i64, (h / r) as i64, (w / r) as i64],
        ))
    }

    /// `F.interpolate(x, size, mode="bilinear", align_corners=False)` for an
    /// arbitrary output size.
    ///
    /// Built as two matrix multiplies rather than a sampling kernel. A
    /// separable bilinear resize *is* a pair of linear maps — one per axis —
    /// and for the sizes that occur here (9 → 64) the matrices are a few
    /// kilobytes. That buys exactness and portability: no `grid_sample`, no
    /// per-backend padding-mode convention to get wrong at the border, and the
    /// coefficients are computed on the host where they can be tested.
    ///
    /// The source index follows PyTorch's `align_corners=False` rule,
    /// `src = (dst + 0.5)·scale − 0.5` clamped up at zero, which is *not* the
    /// same as `dst·scale` and differs visibly at the edges.
    pub fn resize_bilinear(
        &mut self,
        x: HirNodeId,
        out_h: usize,
        out_w: usize,
    ) -> Result<HirNodeId> {
        let [n, c, h, w] = self.dims4(x);
        if (h, w) == (out_h, out_w) {
            return Ok(x);
        }
        ensure!(
            h > 0 && w > 0 && out_h > 0 && out_w > 0,
            "bilinear resize {h}×{w} → {out_h}×{out_w}: every extent must be positive"
        );

        // Rows: [out_h, h] · [.., h, w] → [.., out_h, w]
        let ah = self.constant(
            &format!("bilin_h_{h}_{out_h}"),
            bilinear_weights(h, out_h),
            &[out_h, h],
        );
        let y = self.g().mm(ah, x);
        let y = self
            .g()
            .reshape_(y, vec![n as i64, c as i64, out_h as i64, w as i64]);

        // Columns: [.., out_h, w] · [w, out_w] → [.., out_h, out_w]
        let aw = self.constant(
            &format!("bilin_w_{w}_{out_w}"),
            transpose_2d(&bilinear_weights(w, out_w), out_w, w),
            &[w, out_w],
        );
        let z = self.g().mm(y, aw);
        Ok(self
            .g()
            .reshape_(z, vec![n as i64, c as i64, out_h as i64, out_w as i64]))
    }

    /// `F.adaptive_max_pool2d(x, (h / r, w / r))` for an integer ratio.
    ///
    /// Adaptive pooling only reduces to a plain strided window when the input
    /// divides evenly; otherwise PyTorch uses ragged windows that no fixed
    /// kernel reproduces. Callers that pool by `2^i` therefore have to keep the
    /// tile a multiple of the deepest level, which is exactly what
    /// [`ModelConfig::size_multiple`](crate::config::ModelConfig::size_multiple)
    /// exists to declare — so this rejects the ragged case rather than silently
    /// pooling a slightly different region than the reference.
    pub fn max_pool(&mut self, x: HirNodeId, r: usize) -> Result<HirNodeId> {
        if r == 1 {
            return Ok(x);
        }
        let [_, _, h, w] = self.dims4(x);
        ensure!(
            h % r == 0 && w % r == 0,
            "adaptive max pool by {r}: {h}×{w} does not divide evenly, so the \
             reference's ragged windows cannot be reproduced by a strided kernel"
        );
        self.max_pool_k(x, r, r)
    }

    /// `F.max_pool2d(x, kernel_size=k, stride=s)`, no padding.
    ///
    /// Unlike [`Builder::max_pool`] this is the plain strided kernel, where a
    /// partial trailing window is *discarded* rather than being made ragged —
    /// which is what `max_pool2d` does and `adaptive_max_pool2d` does not.
    pub fn max_pool_k(&mut self, x: HirNodeId, k: usize, stride: usize) -> Result<HirNodeId> {
        let [n, c, h, w] = self.dims4(x);
        ensure!(
            h >= k && w >= k,
            "max pool with a {k}×{k} kernel does not fit a {h}×{w} feature map"
        );
        let shape = nchw_shape(n, c, (h - k) / stride + 1, (w - k) / stride + 1, DType::F32);
        Ok(self.g().0.mir(
            Op::Pool {
                kind: ReduceOp::Max,
                kernel_size: vec![k, k],
                stride: vec![stride, stride],
                padding: vec![0, 0],
            },
            vec![x],
            shape,
        ))
    }

    /// `F.normalize(x, p=2, dim=-1)` — scale each row to unit L2 norm.
    ///
    /// The reference's `eps` is 1e-12 and it *clamps the norm* rather than
    /// adding to it, so a genuinely zero row divides by 1e-12 instead of
    /// becoming 1/sqrt(eps). Adding the epsilon inside the square root would be
    /// the natural-looking version and would give a different answer.
    pub fn l2_normalize_last(&mut self, x: HirNodeId) -> HirNodeId {
        let rank = self.dims(x).len();
        let sq = self.g().mul(x, x);
        let sum = self.g().sum(sq, vec![rank - 1], true);
        let norm = self.g().sqrt(sum);
        let eps = self.splat("l2_eps", 1e-12);
        let shape = self.g().shape(norm).clone();
        let clamped = self
            .g()
            .0
            .mir(Op::Binary(BinaryOp::Max), vec![norm, eps], shape);
        self.g().div(x, clamped)
    }

    /// Nearest-neighbour upsample by an integer factor, with no permutation:
    /// insert unit axes after H and W, `Expand` them to `r`, and collapse.
    pub fn nearest_upsample(&mut self, x: HirNodeId, r: usize) -> HirNodeId {
        if r == 1 {
            return x;
        }
        let [n, c, h, w] = self.dims4(x);
        debug_assert_eq!(n, 1, "nearest upsample assumes a single tile");
        let five = self
            .g()
            .reshape_(x, vec![c as i64, h as i64, 1, w as i64, 1]);
        let expanded = self
            .g()
            .expand_(five, vec![c as i64, h as i64, r as i64, w as i64, r as i64]);
        self.g()
            .reshape_(expanded, vec![1, c as i64, (h * r) as i64, (w * r) as i64])
    }

    /// `torch.repeat_interleave(x, r, dim=1)` on NCHW — each channel repeated
    /// `r` times in place. Composed from `Expand`, so no permutation.
    pub fn repeat_interleave_channels(&mut self, x: HirNodeId, r: usize) -> HirNodeId {
        if r == 1 {
            return x;
        }
        let [n, c, h, w] = self.dims4(x);
        let five = self
            .g()
            .reshape_(x, vec![n as i64, c as i64, 1, h as i64, w as i64]);
        let expanded = self
            .g()
            .expand_(five, vec![n as i64, c as i64, r as i64, h as i64, w as i64]);
        self.g()
            .reshape_(expanded, vec![n as i64, (c * r) as i64, h as i64, w as i64])
    }
}

/// Refuse a batch dimension where the rank-5 form needs it elided.
///
/// CoreML's MIL rejects any tensor above **rank 5**, and the natural way to
/// write a pixel shuffle or a window partition is rank 6 (`[N, C, r, r, H, W]`).
/// Every graph this crate builds runs one tile at a time, so `N` is always 1
/// and dropping it costs nothing — but that has to be checked rather than
/// assumed, because the failure if it were ever false is a silently wrong
/// reshape, not an error.
fn rank5(n: usize, what: &str) -> Result<()> {
    ensure!(
        n == 1,
        "{what}: the rank-5 form elides the batch axis, so it needs N = 1, got {n}"
    );
    Ok(())
}

/// Row `o` of a bilinear resampling matrix from `n_in` to `n_out` samples.
///
/// PyTorch's `align_corners=False` source index, including its asymmetric
/// clamp: negatives are pulled to zero but the top is handled by reusing the
/// last input sample rather than by clamping the coordinate.
fn bilinear_weights(n_in: usize, n_out: usize) -> Vec<f32> {
    let scale = n_in as f64 / n_out as f64;
    let mut m = vec![0.0f32; n_out * n_in];
    for o in 0..n_out {
        let src = ((o as f64 + 0.5) * scale - 0.5).max(0.0);
        let i0 = src.floor() as usize;
        let i0 = i0.min(n_in - 1);
        let i1 = (i0 + 1).min(n_in - 1);
        let frac = (src - i0 as f64) as f32;
        m[o * n_in + i0] += 1.0 - frac;
        m[o * n_in + i1] += frac;
    }
    m
}

/// Transpose a row-major `[rows, cols]` matrix.
fn transpose_2d(m: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut t = vec![0.0f32; m.len()];
    for r in 0..rows {
        for c in 0..cols {
            t[c * rows + r] = m[r * cols + c];
        }
    }
    t
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Bilinear resampling weights must sum to one per output sample, or the
    /// resize changes the image's mean — a gain error that looks like a
    /// brightness bug far from its cause.
    #[test]
    fn bilinear_weights_are_a_partition_of_unity() {
        for (a, b) in [(9usize, 64usize), (4, 8), (8, 4), (3, 3), (1, 5), (31, 7)] {
            for row in bilinear_weights(a, b).chunks(a) {
                let sum: f32 = row.iter().sum();
                assert!((sum - 1.0).abs() < 1e-6, "{a}->{b}: row sums to {sum}");
            }
        }
    }

    /// `align_corners=False` replicates at the border rather than extrapolating,
    /// and its interior offset is half a source pixel — both differ from the
    /// naive `dst * scale` mapping, and both are visible in a 4→8 upsample.
    #[test]
    fn bilinear_weights_match_align_corners_false() {
        let m = bilinear_weights(4, 8);
        assert_eq!(&m[0..4], &[1.0, 0.0, 0.0, 0.0], "first output replicates");
        assert_eq!(&m[28..32], &[0.0, 0.0, 0.0, 1.0], "last output replicates");
        // src = (1 + 0.5) * 0.5 - 0.5 = 0.25
        assert_eq!(&m[4..8], &[0.75, 0.25, 0.0, 0.0]);
    }

    #[test]
    fn transpose_2d_round_trips() {
        let m = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = transpose_2d(&m, 2, 3);
        assert_eq!(t, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        assert_eq!(transpose_2d(&t, 3, 2), m);
    }
    use rlx_runtime::{Device, Session};

    /// Run a single-input graph on CPU and return the flat output.
    fn run(b: Builder, out: HirNodeId, input: &[f32]) -> Vec<f32> {
        let (graph, params, _) = b.finish(out).unwrap();
        let opts = rlx_core::flow_bridge::compile_options_for_profile(
            &rlx_flow::CompileProfile::encoder(),
            Device::Cpu,
        );
        let mut compiled = Session::new(Device::Cpu).compile_with(graph, &opts);
        rlx_core::flow_util::attach_built_params(&mut compiled, params, &[]);
        compiled.run(&[("image", input)]).remove(0)
    }

    /// `out[c, h*r+i, w*r+j] = in[c*r² + i*r + j, h, w]` — the exact PyTorch
    /// index identity. A transposed permutation would still produce the right
    /// *shape*, which is why this pins values, not dimensions.
    #[test]
    fn pixel_shuffle_matches_torch_indexing() {
        let (c, r, h, w) = (2usize, 2usize, 2usize, 3usize);
        let mut b = Builder::new("ps");
        let x = b.input(c * r * r, h, w);
        let y = b.pixel_shuffle(x, r).unwrap();
        let input: Vec<f32> = (0..(c * r * r * h * w)).map(|i| i as f32).collect();
        let got = run(b, y, &input);

        let mut want = vec![0.0f32; c * h * r * w * r];
        for cc in 0..c {
            for i in 0..r {
                for j in 0..r {
                    for hh in 0..h {
                        for ww in 0..w {
                            let src = ((cc * r * r + i * r + j) * h + hh) * w + ww;
                            let dst = ((cc * h * r + hh * r + i) * w * r) + ww * r + j;
                            want[dst] = input[src];
                        }
                    }
                }
            }
        }
        assert_eq!(got, want);
    }

    /// Feeding a nearest-upsampled image through pixel shuffle is how every
    /// PLKSR-family net forms its residual base, so the two must agree.
    #[test]
    fn repeat_interleave_then_pixel_shuffle_is_nearest_upsample() {
        let (c, r, h, w) = (2usize, 2usize, 2usize, 3usize);
        let input: Vec<f32> = (0..(c * h * w)).map(|i| (i * 7 % 13) as f32).collect();

        let mut b = Builder::new("via_ps");
        let x = b.input(c, h, w);
        let rep = b.repeat_interleave_channels(x, r * r);
        let y = b.pixel_shuffle(rep, r).unwrap();
        let via_ps = run(b, y, &input);

        let mut b = Builder::new("direct");
        let x = b.input(c, h, w);
        let y = b.nearest_upsample(x, r);
        let direct = run(b, y, &input);

        assert_eq!(via_ps, direct);
        // Spot-check the actual upsampling: every 2×2 block is one source pixel.
        assert_eq!(direct[0], input[0]);
        assert_eq!(direct[1], input[0]);
        assert_eq!(direct[w * r], input[0]);
    }

    /// Unshuffle then shuffle must be the identity — Real-ESRGAN's ×1/×2
    /// variants unshuffle the input and the trunk shuffles it back, so an
    /// inverse that is merely the right *shape* would scramble the image.
    #[test]
    fn pixel_unshuffle_inverts_pixel_shuffle() {
        let (c, r, h, w) = (2usize, 2usize, 4usize, 6usize);
        let input: Vec<f32> = (0..(c * h * w)).map(|i| i as f32).collect();
        let mut b = Builder::new("roundtrip");
        let x = b.input(c, h, w);
        let down = b.pixel_unshuffle(x, r).unwrap();
        assert_eq!(b.dims4(down), [1, c * r * r, h / r, w / r]);
        let up = b.pixel_shuffle(down, r).unwrap();
        assert_eq!(run(b, up, &input), input);
    }

    #[test]
    fn leaky_relu_matches_the_reference_formula() {
        let mut b = Builder::new("lrelu");
        let x = b.input(1, 1, 5);
        let y = b.leaky_relu(x, 0.1);
        let input = vec![-2.0, -0.5, 0.0, 0.5, 2.0];
        let got = run(b, y, &input);
        let want: Vec<f32> = input
            .iter()
            .map(|&v| if v > 0.0 { v } else { 0.1 * v })
            .collect();
        for (g, w) in got.iter().zip(&want) {
            assert!((g - w).abs() < 1e-6, "{got:?} != {want:?}");
        }
    }

    /// `scale = 1` restoration models are legal; both resamplers must be exact
    /// identities there rather than emitting a degenerate reshape.
    #[test]
    fn scale_one_resampling_is_the_identity() {
        let mut b = Builder::new("id");
        let x = b.input(3, 2, 2);
        assert_eq!(b.pixel_shuffle(x, 1).unwrap(), x);
        assert_eq!(b.nearest_upsample(x, 1), x);
        assert_eq!(b.repeat_interleave_channels(x, 1), x);
    }
}
