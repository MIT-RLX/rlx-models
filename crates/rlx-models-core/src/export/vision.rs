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

//! What a vision transformer's runner does on the host before its encoder,
//! as graphs a package runs ([`super::graph::Stage`]s):
//!
//! - [`resize_normalize`]: an RGB image of any size → normalized NCHW
//!   pixels, bilinear with the corners aligned, as the DINO / UNI runners
//!   resize;
//! - [`patch_tokens`]: pixels → the encoder's input rows — each patch
//!   through the patch projection (the checkpoint's convolution weight, read
//!   as stored), the class and register tokens before them, the position
//!   embeddings added;
//! - [`first_row`]: the class token's row of the encoder's output.

use anyhow::{Result, bail};
use rlx_ir::op::BinaryOp;
use rlx_ir::{DType, Graph, GraphExt, NodeId, Op, Shape};

use crate::image_preprocess::{Filter, ImagePreprocessor, ResizeMode};

fn f32s(d: &[usize]) -> Shape {
    Shape::new(d, DType::F32)
}

fn constant(g: &mut Graph, values: &[f32]) -> NodeId {
    let data = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    g.add_node(Op::Constant { data }, vec![], f32s(&[values.len()]))
}

/// The image symbols: its height and width.
pub const HEIGHT: u32 = 0;
pub const WIDTH: u32 = 1;

/// The resize at one image size: `image` `[h, w, 3]` (u8 values as f32) and
/// `dims` (its shape as numbers) → `pixels` `[1, 3, size, size]`.
fn resize_normalize_at(h: usize, w: usize, size: usize, mean: [f32; 3], std: [f32; 3]) -> Graph {
    let mut g = Graph::new("preprocess");
    let image = g.input("image", f32s(&[h, w, 3]));
    let dims = g.input("dims", f32s(&[3]));
    let one = constant(&mut g, &[1.0]);
    let span = constant(&mut g, &[(size.max(1) as f32 - 1.0).max(1.0)]);
    let iota: Vec<f32> = (0..size).map(|i| i as f32).collect();
    let iota = constant(&mut g, &iota);
    // Per axis: the source coordinate of each output pixel, its two
    // neighbours (the second clamped to the image), and the weight between.
    let axis = |g: &mut Graph, at: usize| {
        let n = g.narrow_(dims, 0, at, 1);
        let last = g.sub(n, one);
        let scale = g.div(last, span);
        let pos = g.mul(iota, scale);
        let lo = g.add_node(
            Op::Activation(rlx_ir::op::Activation::Floor),
            vec![pos],
            f32s(&[size]),
        );
        let frac = g.sub(pos, lo);
        let next = g.add(lo, one);
        let hi = g.binary(BinaryOp::Min, next, last, f32s(&[size]));
        let i0 = g.add_node(
            Op::Cast { to: DType::I32 },
            vec![lo],
            Shape::new(&[size], DType::I32),
        );
        let i1 = g.add_node(
            Op::Cast { to: DType::I32 },
            vec![hi],
            Shape::new(&[size], DType::I32),
        );
        (i0, i1, frac)
    };
    let (y0, y1, dy) = axis(&mut g, 0);
    let (x0, x1, dx) = axis(&mut g, 1);
    // Along x for every row, then along y: `p00·(1−dx) + p01·dx`, then
    // `top·(1−dy) + bot·dy`, as the runners compute each pixel.
    let lerp = |g: &mut Graph, a: NodeId, b: NodeId, t: NodeId, shape: &[usize]| {
        let inv = g.sub(one, t);
        let wa = g.binary(BinaryOp::Mul, a, inv, f32s(shape));
        let wb = g.binary(BinaryOp::Mul, b, t, f32s(shape));
        g.binary(BinaryOp::Add, wa, wb, f32s(shape))
    };
    let c0 = g.gather(image, x0, 1, f32s(&[h, size, 3]));
    let c1 = g.gather(image, x1, 1, f32s(&[h, size, 3]));
    let dxb = g.reshape(dx, vec![1, size as i64, 1], f32s(&[1, size, 1]));
    let rows = lerp(&mut g, c0, c1, dxb, &[h, size, 3]);
    let r0 = g.gather(rows, y0, 0, f32s(&[size, size, 3]));
    let r1 = g.gather(rows, y1, 0, f32s(&[size, size, 3]));
    let dyb = g.reshape(dy, vec![size as i64, 1, 1], f32s(&[size, 1, 1]));
    let v = lerp(&mut g, r0, r1, dyb, &[size, size, 3]);
    let k255 = constant(&mut g, &[255.0]);
    let v = g.binary(BinaryOp::Div, v, k255, f32s(&[size, size, 3]));
    let mean = constant(&mut g, &mean);
    let std = constant(&mut g, &std);
    let v = g.binary(BinaryOp::Sub, v, mean, f32s(&[size, size, 3]));
    let v = g.binary(BinaryOp::Div, v, std, f32s(&[size, size, 3]));
    let chw = g.transpose_(v, vec![2, 0, 1]);
    let pixels = g.reshape(
        chw,
        vec![1, 3, size as i64, size as i64],
        f32s(&[1, 3, size, size]),
    );
    g.node_mut(pixels).name = Some("pixels".into());
    g.set_outputs(vec![pixels]);
    g
}

/// An RGB image (`image` `[height, width, 3]`, 0–255, any size; `dims` its
/// shape) → `pixels` `[1, 3, size, size]`: resized bilinearly with the
/// corners aligned, then `(v / 255 − mean) / std` per channel. Symbolic in
/// the image's height and width ([`HEIGHT`], [`WIDTH`]); feed `dims` from
/// `"image#shape"`.
pub fn resize_normalize(size: usize, mean: [f32; 3], std: [f32; 3]) -> Result<Graph> {
    let a = resize_normalize_at(301, 307, size, mean, std);
    let b = resize_normalize_at(311, 313, size, mean, std);
    let g = rlx_ir::dynamic::generalize(&a, &b, &[(HEIGHT, 301, 311), (WIDTH, 307, 313)])
        .map_err(|e| anyhow::anyhow!("the resize is not symbolic in the image size: {e}"))?;
    let c = resize_normalize_at(200, 120, size, mean, std);
    let mut bind = rlx_ir::DimBinding::new();
    bind.set(HEIGHT, 200);
    bind.set(WIDTH, 120);
    if format!("{}", rlx_ir::dynamic::specialize(&g, &bind)) != format!("{c}") {
        bail!("the resize generalized from two sizes is not the one built at a third");
    }
    Ok(g)
}

/// The longest image side [`clip_resize_crop`] takes: its index table,
/// [`CLIP_IOTA`] (`0, 1, …`), is this long.
pub const MAX_IMAGE_SIDE: usize = 8192;
/// The param [`clip_resize_crop`] reads its indices from; hold
/// [`clip_iota`] under it.
pub const CLIP_IOTA: &str = "preprocess.iota";

/// The values of [`CLIP_IOTA`].
pub fn clip_iota() -> Vec<f32> {
    (0..MAX_IMAGE_SIDE).map(|i| i as f32).collect()
}

fn floor(g: &mut Graph, x: NodeId) -> NodeId {
    let shape = g.node(x).shape.clone();
    g.add_node(
        Op::Activation(rlx_ir::op::Activation::Floor),
        vec![x],
        shape,
    )
}

/// `a < b` (`lt`) or `a ≤ b` as 0 / 1.
fn flag(g: &mut Graph, op: rlx_ir::op::CmpOp, a: NodeId, b: NodeId) -> NodeId {
    let shape = g
        .node(a)
        .shape
        .broadcast_with(&g.node(b).shape)
        .expect("broadcastable");
    let dims = shape.dims().to_vec();
    let to = g.node(a).shape.dtype();
    let bools = Shape::from_dims(&dims, DType::Bool);
    let c = g.add_node(Op::Compare(op), vec![a, b], bools);
    g.add_node(Op::Cast { to }, vec![c], Shape::from_dims(&dims, to))
}

/// `⌊n / d⌋` of integers below `2^24`, exact: the quotient's floor, moved
/// by one where the division rounded across an integer.
fn div_floor(g: &mut Graph, n: NodeId, d: NodeId) -> NodeId {
    let q = g.div(n, d);
    let q = floor(g, q);
    let back = g.mul(q, d);
    let r = g.sub(n, back);
    let zero = constant(g, &[0.0]);
    let under = flag(g, rlx_ir::op::CmpOp::Lt, r, zero);
    let over = flag(g, rlx_ir::op::CmpOp::Ge, r, d);
    let q = g.sub(q, under);
    g.add(q, over)
}

/// Pillow's cubic kernel (`a = −0.5`) at `t ≥ 0`.
fn bicubic_kernel(g: &mut Graph, t: NodeId) -> NodeId {
    let one = constant(g, &[1.0]);
    let two = constant(g, &[2.0]);
    // a = −0.5: ((a + 2)·t − (a + 3))·t·t + 1 below 1, then
    // (((t − 5)·t + 8)·t − 4)·a below 2.
    let c = |g: &mut Graph, v: f32| constant(g, &[v]);
    let k = c(g, 1.5);
    let near = g.mul(t, k);
    let k = c(g, 2.5);
    let near = g.sub(near, k);
    let near = g.mul(near, t);
    let near = g.mul(near, t);
    let near = g.add(near, one);
    let k = c(g, 5.0);
    let far = g.sub(t, k);
    let far = g.mul(far, t);
    let k = c(g, 8.0);
    let far = g.add(far, k);
    let far = g.mul(far, t);
    let k = c(g, 4.0);
    let far = g.sub(far, k);
    let k = c(g, -0.5);
    let far = g.mul(far, k);
    let in1 = flag(g, rlx_ir::op::CmpOp::Lt, t, one);
    let in2 = flag(g, rlx_ir::op::CmpOp::Lt, t, two);
    let ring = g.sub(in2, in1);
    let near = g.mul(near, in1);
    let far = g.mul(far, ring);
    g.add(near, far)
}

/// Pillow's antialiased coefficients (bicubic or bilinear) for `n` samples (`n_value`
/// `[1, 1]` holding it) resized to `d` (`d_value`), of the outputs
/// `offset, …, offset + out − 1` (a center crop): `[out, n]`, each
/// normalized and in Pillow's 22-bit fixed point (rounded half away from
/// zero).
///
/// A tap's distance from its output's center, `(2i + 1)·d − (2x + 1)·n`
/// over `2·d`, is an integer numerator that cancels: it is formed exactly
/// from 9-bit limbs (sides up to 8192, resized sides below 32768), so the
/// kernel's argument rounds once, in one division. The kernel and the
/// normalization are f32; Pillow's are f64, so a coefficient can differ
/// by one in `2^22`.
#[allow(clippy::too_many_arguments)]
fn bicubic_matrix(
    g: &mut Graph,
    iota: NodeId,
    n: usize,
    n_value: NodeId,
    d_value: NodeId,
    offset: NodeId,
    out: usize,
    filter: Filter,
) -> NodeId {
    let one = constant(g, &[1.0]);
    let two = constant(g, &[2.0]);
    let limb = constant(g, &[512.0]);
    let split = |g: &mut Graph, v: NodeId| {
        let hi = g.div(v, limb);
        let hi = floor(g, hi);
        let back = g.mul(hi, limb);
        let lo = g.sub(v, back);
        (hi, lo)
    };
    // (2x + 1) for the outputs kept, [out, 1].
    let x = g.narrow_(iota, 0, 0, out);
    let x = g.reshape_(x, vec![out as i64, 1]);
    let x = g.add(x, offset);
    let x = g.mul(x, two);
    let x = g.add(x, one);
    // (2i + 1) for every sample, [1, n].
    let i = g.narrow_(iota, 0, 0, n);
    let i = g.reshape_(i, vec![1, n as i64]);
    let i = g.mul(i, two);
    let i = g.add(i, one);
    let (i1, i0) = split(g, i);
    let (x1, x0) = split(g, x);
    // 512·(i1·d − x1·n) + (i0·d − x0·n).
    let a = g.mul(i1, d_value);
    let b = g.mul(x1, n_value);
    let hi = g.sub(a, b);
    let a = g.mul(i0, d_value);
    let b = g.mul(x0, n_value);
    let lo = g.sub(a, b);
    let hi = g.mul(hi, limb);
    let num = g.add(hi, lo);
    // Over 2·d, then by min(d / n, 1): over 2·max(n, d).
    let wide = g.binary(BinaryOp::Max, n_value, d_value, f32s(&[1, 1]));
    let wide = g.mul(wide, two);
    let c = |g: &mut Graph, v: f32| constant(g, &[v]);
    let w = match filter {
        Filter::Bicubic => {
            let t = g.div(num, wide);
            let t = g.abs(t);
            bicubic_kernel(g, t)
        }
        // The triangle in integers, `max(0, 2·max(n, d) − |num|)` (exact),
        // so a coefficient rounds once, in the division that normalizes it.
        Filter::Bilinear => {
            let d = g.abs(num);
            let w = g.sub(wide, d);
            g.relu(w)
        }
    };
    let total = g.reduce(w, rlx_ir::op::ReduceOp::Sum, vec![1], true, f32s(&[out, 1]));
    let w = g.div(w, total);
    let fixed = c(g, (1u32 << 22) as f32);
    let w = g.mul(w, fixed);
    let half = c(g, 0.5);
    let mag = g.abs(w);
    let mag = g.add(mag, half);
    let mag = floor(g, mag);
    let shape = g.node(w).shape.clone();
    let sign = g.add_node(Op::Activation(rlx_ir::op::Activation::Sign), vec![w], shape);
    g.mul(mag, sign)
}

/// `x` `[rows, n]` (8-bit values) through Pillow's pass with the fixed-point
/// coefficients `k` `[out, n]`: `clamp((2^21 + x·kᵀ) >> 22, 0, 255)`,
/// exact — `k` split `hi·2^7 + lo` so every sum is an integer below `2^24`.
fn pillow_pass(g: &mut Graph, x: NodeId, k: NodeId) -> NodeId {
    let step = constant(g, &[128.0]);
    let hi = g.div(k, step);
    let hi = floor(g, hi);
    let lo = g.mul(hi, step);
    let lo = g.sub(k, lo);
    let hi_t = g.transpose_(hi, vec![1, 0]);
    let lo_t = g.transpose_(lo, vec![1, 0]);
    let a = g.mm(x, hi_t);
    let b = g.mm(x, lo_t);
    // ⌊(a·2^7 + b + 2^21) / 2^22⌋ = ⌊(a + ⌊(b + 2^21) / 2^7⌋) / 2^15⌋.
    let round = constant(g, &[(1u32 << 21) as f32]);
    let b = g.add(b, round);
    let b = g.div(b, step);
    let b = floor(g, b);
    let sum = g.add(a, b);
    let down = constant(g, &[(1u32 << 15) as f32]);
    let sum = g.div(sum, down);
    let sum = floor(g, sum);
    g.clamp_(sum, 0.0, 255.0)
}

/// An [`ImagePreprocessor`] at one image size: `image` `[h, w, 3]` and
/// `dims` → `pixels` `[1, 3, size, size]`.
fn pillow_preprocess_at(h: usize, w: usize, pre: &ImagePreprocessor) -> Graph {
    let size = pre.size;
    let mut g = Graph::new("preprocess");
    let image = g.input("image", f32s(&[h, w, 3]));
    let dims = g.input("dims", f32s(&[3]));
    let iota = g.param(CLIP_IOTA, f32s(&[MAX_IMAGE_SIDE]));
    let side = |g: &mut Graph, axis: usize| {
        let v = g.narrow_(dims, 0, axis, 1);
        g.reshape_(v, vec![1, 1])
    };
    let (hv, wv) = (side(&mut g, 0), side(&mut g, 1));
    let s = constant(&mut g, &[size as f32]);
    let two = constant(&mut g, &[2.0]);
    let one = constant(&mut g, &[1.0]);
    let (dw, dh, cx, cy) = match pre.resize_mode {
        ResizeMode::Exact => {
            let zero = constant(&mut g, &[0.0]);
            (s, s, zero, zero)
        }
        ResizeMode::Shortest | ResizeMode::Longest => {
            // The chosen side to `size`, the other `round(other · size /
            // chosen)` (half up): `⌊(2·other·size + chosen) / (2·chosen)⌋`.
            let scaled = |g: &mut Graph, a: NodeId, b: NodeId| {
                let n = g.mul(a, s);
                let n = g.mul(n, two);
                let n = g.add(n, b);
                let d = g.mul(b, two);
                div_floor(g, n, d)
            };
            let rw = scaled(&mut g, wv, hv);
            let rh = scaled(&mut g, hv, wv);
            // Shortest: width to `size` when it is the shorter (w ≤ h);
            // longest: when it is the longer (h ≤ w).
            let tall = match pre.resize_mode {
                ResizeMode::Shortest => flag(&mut g, rlx_ir::op::CmpOp::Le, wv, hv),
                _ => flag(&mut g, rlx_ir::op::CmpOp::Le, hv, wv),
            };
            let wide = g.sub(one, tall);
            let pick = |g: &mut Graph, a: NodeId, b: NodeId| {
                let a = g.mul(tall, a);
                let b = g.mul(wide, b);
                g.add(a, b)
            };
            let dw = pick(&mut g, s, rw);
            let dh = pick(&mut g, rh, s);
            // The center crop's corner: ⌊(resized − size) / 2⌋.
            let corner = |g: &mut Graph, d: NodeId| {
                let c = g.sub(d, s);
                let c = g.div(c, two);
                floor(g, c)
            };
            let (cx, cy) = (corner(&mut g, dw), corner(&mut g, dh));
            (dw, dh, cx, cy)
        }
    };
    let kx = bicubic_matrix(&mut g, iota, w, wv, dw, cx, size, pre.filter);
    let ky = bicubic_matrix(&mut g, iota, h, hv, dh, cy, size, pre.filter);
    // [h, w, 3] → rows of [3·h, w] → [3·h, size]: Pillow's horizontal pass.
    let x = g.transpose_(image, vec![2, 0, 1]);
    let x = g.reshape_(x, vec![(3 * h) as i64, w as i64]);
    let x = pillow_pass(&mut g, x, kx);
    // [3, h, size] → rows of [3·size, h] → [3·size, size]: the vertical one.
    let x = g.reshape_(x, vec![3, h as i64, size as i64]);
    let x = g.transpose_(x, vec![0, 2, 1]);
    let x = g.reshape_(x, vec![(3 * size) as i64, h as i64]);
    let x = pillow_pass(&mut g, x, ky);
    // [3, x, y] → [3, y, x], then `(v / 255 − mean) / std`.
    let x = g.reshape_(x, vec![3, size as i64, size as i64]);
    let x = g.transpose_(x, vec![0, 2, 1]);
    let k255 = constant(&mut g, &[255.0]);
    let x = g.div(x, k255);
    let m = constant(&mut g, &pre.mean);
    let m = g.reshape_(m, vec![3, 1, 1]);
    let sd = constant(&mut g, &pre.std);
    let sd = g.reshape_(sd, vec![3, 1, 1]);
    let x = g.sub(x, m);
    let x = g.div(x, sd);
    let out = g.reshape_(x, vec![1, 3, size as i64, size as i64]);
    g.node_mut(out).name = Some("pixels".into());
    g.set_outputs(vec![out]);
    g
}

/// An RGB image (`image` `[height, width, 3]`, 0–255, any size up to
/// [`MAX_IMAGE_SIDE`]; `dims` its shape) → `pixels` `[1, 3, size, size]`,
/// as `pre` preprocesses it ([`ImagePreprocessor::from_rgb`]): resized with
/// Pillow's antialiased bicubic or bilinear (its fixed-point passes, its
/// 8-bit store between them) — to `size × size`, or the shortest / longest
/// side to `size` and the center cropped — then `(v / 255 − mean) / std`
/// per channel. Symbolic in [`HEIGHT`] and [`WIDTH`]; its indices are the
/// param [`CLIP_IOTA`] (hold [`clip_iota`]). Run it on the CPU (a GPU's
/// fast division moves a coefficient).
pub fn pillow_preprocess(pre: &ImagePreprocessor) -> Result<Graph> {
    if pre.resize_mode != ResizeMode::Exact && !pre.center_crop {
        bail!("a resize that keeps the aspect ratio without a crop is not square");
    }
    if pre.resize_mode == ResizeMode::Longest {
        // The runner's crop of a side shorter than `size` repeats its edge.
        bail!("a longest-side resize is not exported");
    }
    let a = pillow_preprocess_at(301, 307, pre);
    let b = pillow_preprocess_at(311, 313, pre);
    let g = rlx_ir::dynamic::generalize(&a, &b, &[(HEIGHT, 301, 311), (WIDTH, 307, 313)])
        .map_err(|e| anyhow::anyhow!("the resize is not symbolic in the image size: {e}"))?;
    let c = pillow_preprocess_at(200, 120, pre);
    let mut bind = rlx_ir::DimBinding::new();
    bind.set(HEIGHT, 200);
    bind.set(WIDTH, 120);
    if format!("{}", rlx_ir::dynamic::specialize(&g, &bind)) != format!("{c}") {
        bail!("the resize generalized from two sizes is not the one built at a third");
    }
    Ok(g)
}

/// An RGB image (`image` `[height, width, 3]`, 0–255, any size up to
/// [`MAX_IMAGE_SIDE`]; `dims` its shape) → `pixels` `[1, 3, size, size]`,
/// as OpenCLIP preprocesses: the shortest side resized to `size` with
/// Pillow's antialiased bicubic, the center `size × size` cropped,
/// `(v / 255 − mean) / std` per channel ([`pillow_preprocess`]).
pub fn clip_resize_crop(size: usize, mean: [f32; 3], std: [f32; 3]) -> Result<Graph> {
    pillow_preprocess(&ImagePreprocessor {
        size,
        mean,
        std,
        filter: Filter::Bicubic,
        resize_mode: ResizeMode::Shortest,
        center_crop: true,
    })
}

/// A filter of the `image` crate's `imageops::resize`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageFilter {
    /// `FilterType::Triangle`.
    Triangle,
    /// `FilterType::CatmullRom`.
    CatmullRom,
}

/// `|x|` through `filter`'s kernel, in f32 as the `image` crate evaluates
/// it.
fn image_kernel(g: &mut Graph, x: NodeId, filter: ImageFilter) -> NodeId {
    let c = |g: &mut Graph, v: f32| constant(g, &[v]);
    let a = g.abs(x);
    let one = c(g, 1.0);
    let in1 = flag(g, rlx_ir::op::CmpOp::Lt, a, one);
    match filter {
        ImageFilter::Triangle => {
            let w = g.sub(one, a);
            g.mul(w, in1)
        }
        ImageFilter::CatmullRom => {
            // `bc_cubic_spline(x, 0, 0.5)`: 9a³ − 15a² + 6 below 1,
            // −3a³ + 15a² − 24a + 12 below 2, over 6.
            let two = c(g, 2.0);
            let in2 = flag(g, rlx_ir::op::CmpOp::Lt, a, two);
            let ring = g.sub(in2, in1);
            let a2 = g.mul(a, a);
            let a3 = g.mul(a2, a);
            let term = |g: &mut Graph, v: NodeId, k: f32| {
                let k = c(g, k);
                g.mul(v, k)
            };
            let n3 = term(g, a3, 9.0);
            let n2 = term(g, a2, -15.0);
            let near = g.add(n3, n2);
            let k = c(g, 6.0);
            let near = g.add(near, k);
            let f3 = term(g, a3, -3.0);
            let f2 = term(g, a2, 15.0);
            let far = g.add(f3, f2);
            let f1 = term(g, a, -24.0);
            let far = g.add(far, f1);
            let k = c(g, 12.0);
            let far = g.add(far, k);
            let near = g.mul(near, in1);
            let far = g.mul(far, ring);
            let w = g.add(near, far);
            let six = c(g, 6.0);
            g.div(w, six)
        }
    }
}

/// The `image` crate's resample weights from `n` samples (`n_value`,
/// `[1, 1]`) to `out` (`out_value`): `[out, n]`, output `o` centred at
/// `(o + 0.5)·n/out − 0.5`, the kernel widened by `max(n/out, 1)`, each row
/// normalized — every step in f32, in the crate's order.
fn image_matrix(
    g: &mut Graph,
    iota: NodeId,
    n: usize,
    n_value: NodeId,
    out: usize,
    out_value: NodeId,
    filter: ImageFilter,
) -> NodeId {
    let one = constant(g, &[1.0]);
    let half = constant(g, &[0.5]);
    let ratio = g.div(n_value, out_value);
    let sratio = g.binary(BinaryOp::Max, ratio, one, f32s(&[1, 1]));
    let o = g.narrow_(iota, 0, 0, out);
    let o = g.reshape_(o, vec![out as i64, 1]);
    let o = g.add(o, half);
    let centre = g.mul(o, ratio);
    let centre = g.sub(centre, half);
    let i = g.narrow_(iota, 0, 0, n);
    let i = g.reshape_(i, vec![1, n as i64]);
    let d = g.sub(i, centre);
    let t = g.div(d, sratio);
    let w = image_kernel(g, t, filter);
    let total = g.reduce(w, rlx_ir::op::ReduceOp::Sum, vec![1], true, f32s(&[out, 1]));
    g.div(w, total)
}

/// `image` `[h, w, 3]` (u8 values) → `[oh, ow, 3]` as `imageops::resize`
/// makes it: the vertical pass in f32, then the horizontal one clamped to
/// 0–255 and rounded (half away from zero).
fn image_resize_core(
    g: &mut Graph,
    image: NodeId,
    (h, w): (usize, usize),
    (hv, wv): (NodeId, NodeId),
    (oh, ow): (usize, usize),
    (ohv, owv): (NodeId, NodeId),
    filter: ImageFilter,
) -> NodeId {
    let iota = g.param(CLIP_IOTA, f32s(&[MAX_IMAGE_SIDE]));
    let ky = image_matrix(g, iota, h, hv, oh, ohv, filter);
    let kx = image_matrix(g, iota, w, wv, ow, owv, filter);
    // [h, w·3] → [oh, w·3].
    let x = g.reshape_(image, vec![h as i64, (w * 3) as i64]);
    let x = g.mm(ky, x);
    // [oh, w, 3] → [oh, 3, w] → [oh·3, w] · kxᵀ → [oh·3, ow].
    let x = g.reshape_(x, vec![oh as i64, w as i64, 3]);
    let x = g.transpose_(x, vec![0, 2, 1]);
    let x = g.reshape_(x, vec![(oh * 3) as i64, w as i64]);
    let kxt = g.transpose_(kx, vec![1, 0]);
    let x = g.mm(x, kxt);
    let x = g.clamp_(x, 0.0, 255.0);
    let half = constant(g, &[0.5]);
    let x = g.add(x, half);
    let x = floor(g, x);
    // → [oh, ow, 3].
    let x = g.reshape_(x, vec![oh as i64, 3, ow as i64]);
    g.transpose_(x, vec![0, 2, 1])
}

fn image_resize_normalize_at(
    h: usize,
    w: usize,
    size: usize,
    filter: ImageFilter,
    mean: [f32; 3],
    std: [f32; 3],
) -> Graph {
    let mut g = Graph::new("preprocess");
    let image = g.input("image", f32s(&[h, w, 3]));
    let dims = g.input("dims", f32s(&[3]));
    let side = |g: &mut Graph, axis: usize| {
        let v = g.narrow_(dims, 0, axis, 1);
        g.reshape_(v, vec![1, 1])
    };
    let (hv, wv) = (side(&mut g, 0), side(&mut g, 1));
    let s = constant(&mut g, &[size as f32]);
    let s = g.reshape_(s, vec![1, 1]);
    let x = image_resize_core(
        &mut g,
        image,
        (h, w),
        (hv, wv),
        (size, size),
        (s, s),
        filter,
    );
    // [size, size, 3] → [3, size, size], `(v / 255 − mean) / std`.
    let x = g.transpose_(x, vec![2, 0, 1]);
    let k255 = constant(&mut g, &[255.0]);
    let x = g.div(x, k255);
    let m = constant(&mut g, &mean);
    let m = g.reshape_(m, vec![3, 1, 1]);
    let sd = constant(&mut g, &std);
    let sd = g.reshape_(sd, vec![3, 1, 1]);
    let x = g.sub(x, m);
    let x = g.div(x, sd);
    let out = g.reshape_(x, vec![1, 3, size as i64, size as i64]);
    g.node_mut(out).name = Some("pixels".into());
    g.set_outputs(vec![out]);
    g
}

/// Check a graph `build` makes at an image size generalizes in
/// [`HEIGHT`] / [`WIDTH`]: from two sizes, checked at a third.
fn symbolic_in_image(build: impl Fn(usize, usize) -> Graph) -> Result<Graph> {
    let a = build(301, 307);
    let b = build(311, 313);
    let g = rlx_ir::dynamic::generalize(&a, &b, &[(HEIGHT, 301, 311), (WIDTH, 307, 313)])
        .map_err(|e| anyhow::anyhow!("the resize is not symbolic in the image size: {e}"))?;
    let c = build(200, 120);
    let mut bind = rlx_ir::DimBinding::new();
    bind.set(HEIGHT, 200);
    bind.set(WIDTH, 120);
    if format!("{}", rlx_ir::dynamic::specialize(&g, &bind)) != format!("{c}") {
        bail!("the resize generalized from two sizes is not the one built at a third");
    }
    Ok(g)
}

/// An RGB image (`image` `[height, width, 3]`, 0–255; `dims` its shape) →
/// `pixels` `[1, 3, size, size]`: `image::imageops::resize` to `size ×
/// size` with `filter` (its f32 weights, its vertical pass first, its
/// rounding to 8 bits), then `(v / 255 − mean) / std`. Symbolic in
/// [`HEIGHT`] and [`WIDTH`]; its indices are the param [`CLIP_IOTA`].
pub fn image_resize_normalize(
    size: usize,
    filter: ImageFilter,
    mean: [f32; 3],
    std: [f32; 3],
) -> Result<Graph> {
    symbolic_in_image(|h, w| image_resize_normalize_at(h, w, size, filter, mean, std))
}

fn image_resize_mask_at(h: usize, w: usize, side: usize, filter: ImageFilter) -> Graph {
    let mut g = Graph::new("mask_to_image");
    let mask = g.input("mask", f32s(&[side * side]));
    // The image itself, unread: its shape binds the output's.
    g.input("image", f32s(&[h, w, 3]));
    let dims = g.input("dims", f32s(&[3]));
    let at = |g: &mut Graph, axis: usize| {
        let v = g.narrow_(dims, 0, axis, 1);
        g.reshape_(v, vec![1, 1])
    };
    let (hv, wv) = (at(&mut g, 0), at(&mut g, 1));
    let s = constant(&mut g, &[side as f32]);
    let s = g.reshape_(s, vec![1, 1]);
    let iota = g.param(CLIP_IOTA, f32s(&[MAX_IMAGE_SIDE]));
    let ky = image_matrix(&mut g, iota, side, s, h, hv, filter);
    let kx = image_matrix(&mut g, iota, side, s, w, wv, filter);
    let x = g.reshape_(mask, vec![side as i64, side as i64]);
    let x = g.mm(ky, x);
    let kxt = g.transpose_(kx, vec![1, 0]);
    let x = g.mm(x, kxt);
    // An f32 image's subpixels are clamped to 0–1, and not rounded.
    let x = g.clamp_(x, 0.0, 1.0);
    g.node_mut(x).name = Some("alpha".into());
    g.set_outputs(vec![x]);
    g
}

/// A `side × side` mask in 0–1 (`mask`, `[side²]`) → `alpha` `[height,
/// width]`, as `imageops::resize` takes a `Luma<f32>` image back to the
/// input's size (`dims`, its `[height, width, 3]`): f32 throughout, clamped
/// to 0–1. Feed the image itself as `image` too: its shape binds the
/// output's. Symbolic in [`HEIGHT`] and [`WIDTH`]; its indices are the param
/// [`CLIP_IOTA`].
pub fn image_resize_mask(side: usize, filter: ImageFilter) -> Result<Graph> {
    symbolic_in_image(|h, w| image_resize_mask_at(h, w, side, filter))
}

/// Where a vision transformer adds its position embeddings.
#[derive(Debug, Clone)]
pub enum Positions {
    /// None (the encoder rotates its queries and keys).
    None,
    /// One per row, class and register tokens included: the param, `[1,
    /// rows, width]`.
    AllRows(String),
    /// One per patch: the param, `[1, patches, width]`.
    Patches(String),
}

/// A vision transformer's patch stem, by its checkpoint's tensor names.
#[derive(Debug, Clone)]
pub struct PatchStem {
    pub image_size: usize,
    pub patch_size: usize,
    pub channels: usize,
    /// The token width.
    pub width: usize,
    /// The patch projection, a convolution `[width, channels, patch,
    /// patch]`, and its bias `[width]`.
    pub weight: String,
    pub bias: String,
    /// The class token, `[1, 1, width]`.
    pub class_token: String,
    /// The register tokens, `[1, n, width]`, and how many.
    pub registers: Option<(String, usize)>,
    pub positions: Positions,
}

impl PatchStem {
    /// Patches along a side.
    pub fn side(&self) -> usize {
        self.image_size / self.patch_size
    }
    /// The encoder's rows: class, registers, patches.
    pub fn rows(&self) -> usize {
        1 + self.registers.as_ref().map_or(0, |(_, n)| *n) + self.side() * self.side()
    }
}

/// `pixels` `[1, channels, size, size]` → `hidden` `[1, rows, width]`, the
/// rows `[class, registers…, patches…]` (patch `(y, x)` at `y · side + x`),
/// as the DINO / UNI runners assemble them.
pub fn patch_tokens(stem: &PatchStem) -> Graph {
    let (c, s, p, e) = (stem.channels, stem.image_size, stem.patch_size, stem.width);
    let side = stem.side();
    let n = side * side;
    let pd = c * p * p;
    let mut g = Graph::new("patch_tokens");
    let pixels = g.input("pixels", f32s(&[1, c, s, s]));
    let weight = g.param(stem.weight.clone(), f32s(&[e, c, p, p]));
    let bias = g.param(stem.bias.clone(), f32s(&[e]));
    // [c, py, ry, px, rx] → [py, px, c, ry, rx]: one row per patch, in the
    // convolution weight's [c, ry, rx] order.
    let x = g.reshape_(
        pixels,
        vec![c as i64, side as i64, p as i64, side as i64, p as i64],
    );
    let x = g.transpose_(x, vec![1, 3, 0, 2, 4]);
    let x = g.reshape_(x, vec![n as i64, pd as i64]);
    let w = g.reshape_(weight, vec![e as i64, pd as i64]);
    let w = g.transpose_(w, vec![1, 0]);
    let patches = g.mm(x, w);
    let mut patches = g.add(patches, bias);
    if let Positions::Patches(name) = &stem.positions {
        let pos = g.param(name.clone(), f32s(&[1, n, e]));
        let pos = g.reshape_(pos, vec![n as i64, e as i64]);
        patches = g.add(patches, pos);
    }
    let class = g.param(stem.class_token.clone(), f32s(&[1, 1, e]));
    let mut rows = vec![g.reshape_(class, vec![1, e as i64])];
    if let Some((name, k)) = &stem.registers {
        let reg = g.param(name.clone(), f32s(&[1, *k, e]));
        rows.push(g.reshape_(reg, vec![*k as i64, e as i64]));
    }
    rows.push(patches);
    let mut hidden = g.concat_(rows, 0);
    if let Positions::AllRows(name) = &stem.positions {
        let r = stem.rows();
        let pos = g.param(name.clone(), f32s(&[1, r, e]));
        let pos = g.reshape_(pos, vec![r as i64, e as i64]);
        hidden = g.add(hidden, pos);
    }
    let hidden = g.reshape_(hidden, vec![1, stem.rows() as i64, e as i64]);
    g.node_mut(hidden).name = Some("hidden".into());
    g.set_outputs(vec![hidden]);
    g
}

/// `tokens` `[1, rows, width]` → `embedding` `[1, width]`: the class
/// token's row.
pub fn first_row(rows: usize, width: usize) -> Graph {
    let mut g = Graph::new("class_row");
    let t = g.input("tokens", f32s(&[1, rows, width]));
    let r = g.narrow_(t, 1, 0, 1);
    let r = g.reshape_(r, vec![1, width as i64]);
    g.node_mut(r).name = Some("embedding".into());
    g.set_outputs(vec![r]);
    g
}

/// The pipeline input an RGB image is, for an entry's `inputs`.
pub fn image_contract() -> serde_json::Value {
    serde_json::json!({"name": "image", "kind": "image", "dtype": "u8",
        "shape": [null, null, 3], "color": "rgb", "layout": "hwc", "range": [0, 255]})
}

/// A vision transformer as a package: an RGB image in; the encoder's
/// output (its tokens, or a head's logits) and, from tokens, the class
/// embedding out.
pub struct Vit {
    pub family: String,
    pub exported_by: String,
    pub stem: PatchStem,
    /// The encoder, its input `[1, rows, width]` named `encoder_input`, its
    /// first output what it returns, and its params.
    pub encoder: Graph,
    pub encoder_input: String,
    pub params: std::collections::HashMap<String, Vec<f32>>,
    /// The encoder's output: `"tokens"` (rows; the class embedding is also
    /// returned) or `"logits"`.
    pub output: String,
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

/// Export `vit`, its weights from `ck`, as the package `out`.
pub fn export_vit(
    ck: &super::Checkpoint,
    mut vit: Vit,
    out: &std::path::Path,
    opts: &super::ExportOptions,
) -> Result<super::ExportReport> {
    use super::graph::{Pipeline, Stage};
    let stem = vit.stem.clone();
    // The stem reads the checkpoint's tensors as stored.
    let mut loader = ck.loader()?;
    let mut names = vec![
        stem.weight.clone(),
        stem.bias.clone(),
        stem.class_token.clone(),
    ];
    if let Some((r, _)) = &stem.registers {
        names.push(r.clone());
    }
    match &stem.positions {
        Positions::AllRows(p) | Positions::Patches(p) => names.push(p.clone()),
        Positions::None => {}
    }
    for n in names {
        let (v, _) = loader.take(&n)?;
        vit.params.insert(n, v);
    }
    drop(loader);
    vit.encoder.name = "encoder".into();
    let mut stages = vec![
        Stage::new(resize_normalize(stem.image_size, vit.mean, vit.std)?)
            .input("image", "image")
            .input("dims", "image#shape")
            .outputs(&["pixels"])
            .on("cpu"),
        Stage::new(patch_tokens(&stem)).outputs(&["hidden"]),
        Stage::new(vit.encoder)
            .input(&vit.encoder_input, "hidden")
            .outputs(&[vit.output.as_str()]),
    ];
    let mut outputs = vec![vit.output.clone()];
    if vit.output == "tokens" {
        stages.push(Stage::new(first_row(stem.rows(), stem.width)).outputs(&["embedding"]));
        outputs.push("embedding".into());
    }
    let mut p = Pipeline::new(&vit.family, "image-feature-extraction", stages);
    p.inputs = vec![image_contract()];
    p.outputs = outputs;
    p.dims = vec![("height".into(), HEIGHT), ("width".into(), WIDTH)];
    p.metadata = vec![
        ("image_size".into(), serde_json::json!(stem.image_size)),
        ("mean".into(), serde_json::json!(vit.mean)),
        ("std".into(), serde_json::json!(vit.std)),
    ];
    p.exported_by = vit.exported_by;
    p.write(vit.params, Some(ck), out, opts)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The resize as the DINO / UNI runners write it.
    fn reference(
        rgb: &[u8],
        h: usize,
        w: usize,
        size: usize,
        m: [f32; 3],
        s: [f32; 3],
    ) -> Vec<f32> {
        let mut out = vec![0f32; 3 * size * size];
        let sx = (w as f32 - 1.0) / (size.max(1) as f32 - 1.0).max(1.0);
        let sy = (h as f32 - 1.0) / (size.max(1) as f32 - 1.0).max(1.0);
        for y in 0..size {
            let fy = y as f32 * sy;
            let y0 = fy.floor() as usize;
            let y1 = (y0 + 1).min(h - 1);
            let dy = fy - y0 as f32;
            for x in 0..size {
                let fx = x as f32 * sx;
                let x0 = fx.floor() as usize;
                let x1 = (x0 + 1).min(w - 1);
                let dx = fx - x0 as f32;
                for c in 0..3 {
                    let p = |yy: usize, xx: usize| rgb[(yy * w + xx) * 3 + c] as f32;
                    let top = p(y0, x0) * (1.0 - dx) + p(y0, x1) * dx;
                    let bot = p(y1, x0) * (1.0 - dx) + p(y1, x1) * dx;
                    let v = (top * (1.0 - dy) + bot * dy) / 255.0;
                    out[c * size * size + y * size + x] = (v - m[c]) / s[c];
                }
            }
        }
        out
    }

    /// The CLIP preprocessing graph against the runners' (Pillow's
    /// bicubic, in f32): within one 8-bit level, on few pixels — the two
    /// round coefficients and sums differently.
    #[test]
    fn the_clip_preprocessing_is_the_runners() {
        let (m, s) = ([0.481, 0.458, 0.408], [0.269, 0.261, 0.276]);
        let size = 32;
        let g = clip_resize_crop(size, m, s).unwrap();
        let reference = crate::image_preprocess::ImagePreprocessor::clip(size, m, s);
        let level = 1.0 / 255.0 / 0.26;
        for (h, w) in [(50, 37), (32, 32), (19, 64), (32, 90), (400, 41), (24, 24)] {
            let rgb: Vec<u8> = (0..h * w * 3)
                .map(|i| ((i * 37 + i / 7 + (i / (3 * w)) * 11) % 251) as u8)
                .collect();
            let mut bind = rlx_ir::DimBinding::new();
            bind.set(HEIGHT, h);
            bind.set(WIDTH, w);
            let at = rlx_ir::dynamic::specialize(&g, &bind);
            let mut c = rlx_runtime::Session::new(rlx_runtime::Device::Cpu).compile(at);
            c.set_param(CLIP_IOTA, &clip_iota());
            c.finalize_params();
            let image: Vec<f32> = rgb.iter().map(|&v| v as f32).collect();
            let dims = [h as f32, w as f32, 3.0];
            let got = c.run(&[("image", &image), ("dims", &dims)]).remove(0);
            let want = reference.from_rgb(&rgb, w, h);
            assert_eq!(got.len(), want.len());
            let off: Vec<f32> = got
                .iter()
                .zip(&want)
                .map(|(x, y)| (x - y).abs())
                .filter(|d| *d > 1e-5)
                .collect();
            let worst = off.iter().fold(0f32, |a, &d| a.max(d));
            eprintln!("{h}x{w}: {} of {} off, by {worst}", off.len(), got.len());
            assert!(worst < 1.01 * level, "{h}x{w}: off by {worst}");
            assert!(off.len() * 100 <= got.len(), "{h}x{w}: {} off", off.len());
        }
    }

    #[test]
    fn every_pillow_preprocessing_is_the_runners() {
        use crate::image_preprocess::{Filter, ImagePreprocessor, ResizeMode};
        let size = 24;
        for (filter, mode, crop) in [
            (Filter::Bicubic, ResizeMode::Exact, false),
            (Filter::Bilinear, ResizeMode::Exact, false),
            (Filter::Bilinear, ResizeMode::Shortest, true),
        ] {
            let pre = ImagePreprocessor {
                size,
                mean: [0.5; 3],
                std: [0.5; 3],
                filter,
                resize_mode: mode,
                center_crop: crop,
            };
            let g = pillow_preprocess(&pre).unwrap();
            let level = 2.0 / 255.0;
            for (h, w) in [(50, 37), (24, 24), (19, 64), (13, 9), (400, 41)] {
                let rgb: Vec<u8> = (0..h * w * 3)
                    .map(|i| ((i * 37 + i / 7 + (i / (3 * w)) * 11) % 251) as u8)
                    .collect();
                let mut bind = rlx_ir::DimBinding::new();
                bind.set(HEIGHT, h);
                bind.set(WIDTH, w);
                let at = rlx_ir::dynamic::specialize(&g, &bind);
                let mut c = rlx_runtime::Session::new(rlx_runtime::Device::Cpu).compile(at);
                c.set_param(CLIP_IOTA, &clip_iota());
                c.finalize_params();
                let image: Vec<f32> = rgb.iter().map(|&v| v as f32).collect();
                let dims = [h as f32, w as f32, 3.0];
                let got = c.run(&[("image", &image), ("dims", &dims)]).remove(0);
                let want = pre.from_rgb(&rgb, w, h);
                assert_eq!(got.len(), want.len());
                let off = got
                    .iter()
                    .zip(&want)
                    .filter(|(x, y)| (*x - *y).abs() > 1e-5)
                    .count();
                let worst = got
                    .iter()
                    .zip(&want)
                    .fold(0f32, |a, (x, y)| a.max((x - y).abs()));
                eprintln!(
                    "{filter:?} {mode:?} {h}x{w}: {off} of {} off, by {worst}",
                    got.len()
                );
                assert!(
                    worst < 1.01 * level,
                    "{filter:?} {mode:?} {h}x{w}: off by {worst}"
                );
                // Bicubic matches exactly. The triangle is taken in exact
                // integers, Pillow's in f64 from an inexact center: at a
                // ratio with a small denominator (64 → 24 is 8/3) their
                // fixed-point ties round apart, a level in a few percent.
                let allowed = match filter {
                    Filter::Bicubic => 0,
                    Filter::Bilinear => got.len() * 3 / 100,
                };
                assert!(off <= allowed, "{filter:?} {mode:?} {h}x{w}: {off} off");
            }
        }
    }

    #[test]
    fn the_image_crate_resize_is_the_image_crates() {
        use image::imageops::FilterType;
        let (m, sd) = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]);
        let size = 32;
        for (filter, ft) in [
            (ImageFilter::Triangle, FilterType::Triangle),
            (ImageFilter::CatmullRom, FilterType::CatmullRom),
        ] {
            let g = image_resize_normalize(size, filter, m, sd).unwrap();
            let back = image_resize_mask(size, filter).unwrap();
            for (h, w) in [(50, 37), (32, 32), (19, 64), (13, 9), (400, 41)] {
                let rgb: Vec<u8> = (0..h * w * 3)
                    .map(|i| ((i * 37 + i / 7 + (i / (3 * w)) * 11) % 251) as u8)
                    .collect();
                let run = |g: &Graph, inputs: &[(&str, &[f32])]| {
                    let mut bind = rlx_ir::DimBinding::new();
                    bind.set(HEIGHT, h);
                    bind.set(WIDTH, w);
                    let at = rlx_ir::dynamic::specialize(g, &bind);
                    let mut c = rlx_runtime::Session::new(rlx_runtime::Device::Cpu).compile(at);
                    c.set_param(CLIP_IOTA, &clip_iota());
                    c.finalize_params();
                    c.run(inputs).remove(0)
                };
                let image: Vec<f32> = rgb.iter().map(|&v| v as f32).collect();
                let dims = [h as f32, w as f32, 3.0];
                let got = run(&g, &[("image", &image), ("dims", &dims)]);
                let img = image::RgbImage::from_raw(w as u32, h as u32, rgb.clone()).unwrap();
                let r = image::imageops::resize(&img, size as u32, size as u32, ft);
                let n = size * size;
                let mut want = vec![0f32; 3 * n];
                for (i, p) in r.pixels().enumerate() {
                    for c in 0..3 {
                        want[c * n + i] = (p.0[c] as f32 / 255.0 - m[c]) / sd[c];
                    }
                }
                let level = 1.0 / 255.0 / 0.224;
                let off = got
                    .iter()
                    .zip(&want)
                    .filter(|(a, b)| (*a - *b).abs() > 1e-5)
                    .count();
                let worst = got
                    .iter()
                    .zip(&want)
                    .fold(0f32, |a, (x, y)| a.max((x - y).abs()));
                eprintln!("{filter:?} {h}x{w}: {off} of {} off, by {worst}", got.len());
                assert!(worst < 1.01 * level, "{filter:?} {h}x{w}: off by {worst}");
                assert!(off * 1000 <= got.len(), "{filter:?} {h}x{w}: {off} off");

                // A mask back to the image's size.
                let mask: Vec<f32> = (0..n).map(|i| ((i * 7919) % 1000) as f32 / 999.0).collect();
                let got = run(
                    &back,
                    &[("mask", &mask), ("image", &image), ("dims", &dims)],
                );
                let gm = image::ImageBuffer::<image::Luma<f32>, Vec<f32>>::from_raw(
                    size as u32,
                    size as u32,
                    mask,
                )
                .unwrap();
                let want = image::imageops::resize(&gm, w as u32, h as u32, ft).into_raw();
                let worst = got
                    .iter()
                    .zip(&want)
                    .fold(0f32, |a, (x, y)| a.max((x - y).abs()));
                eprintln!("{filter:?} mask {h}x{w}: off by {worst}");
                assert!(worst < 1e-5, "{filter:?} mask {h}x{w}: off by {worst}");
            }
        }
    }

    #[test]
    fn the_resize_graph_is_the_runners_resize() {
        let (m, s) = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]);
        let size = 28;
        let g = resize_normalize(size, m, s).unwrap();
        for (h, w) in [(50, 37), (28, 28), (19, 64)] {
            let rgb: Vec<u8> = (0..h * w * 3)
                .map(|i| ((i * 37 + i / 7) % 251) as u8)
                .collect();
            let mut bind = rlx_ir::DimBinding::new();
            bind.set(HEIGHT, h);
            bind.set(WIDTH, w);
            let at = rlx_ir::dynamic::specialize(&g, &bind);
            let mut c = rlx_runtime::Session::new(rlx_runtime::Device::Cpu).compile(at);
            let image: Vec<f32> = rgb.iter().map(|&v| v as f32).collect();
            let dims = [h as f32, w as f32, 3.0];
            let got = c.run(&[("image", &image), ("dims", &dims)]).remove(0);
            let want = reference(&rgb, h, w, size, m, s);
            let worst = got
                .iter()
                .zip(&want)
                .fold(0f32, |a, (x, y)| a.max((x - y).abs()));
            assert!(worst < 1e-5, "{h}x{w}: off by {worst}");
        }
    }
}
