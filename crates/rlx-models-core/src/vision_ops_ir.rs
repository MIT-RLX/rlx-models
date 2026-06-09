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

//! Shared HIR builders for NCHW vision ops (`Conv`, `ConvTranspose2d`,
//! `LayerNorm2d`, bias broadcast). Used by SAM / SAM2 / SAM3.

use rlx_ir::hir::{HirMut, HirNodeId};
use rlx_ir::op::Op;
use rlx_ir::{DType, HirGraphExt, Shape};

pub fn nchw_shape(batch: usize, c: usize, h: usize, w: usize, dt: DType) -> Shape {
    Shape::new(&[batch, c, h, w], dt)
}

/// `[B, H·W, C]` (BHWC row-major) → `[B, C, H, W]` NCHW.
pub fn bhwc_to_nchw(
    g: &mut HirMut<'_>,
    x: HirNodeId,
    batch: usize,
    h: usize,
    w: usize,
    c: usize,
) -> HirNodeId {
    let x4 = g.reshape_(x, vec![batch as i64, h as i64, w as i64, c as i64]);
    g.transpose_(x4, vec![0, 3, 1, 2])
}

/// Broadcast `[C]` bias onto NCHW activations and add.
pub fn add_bias_nchw(
    g: &mut HirMut<'_>,
    y: HirNodeId,
    bias: HirNodeId,
    batch: usize,
    out_c: usize,
    h: usize,
    w: usize,
) -> HirNodeId {
    let out_shape = g.shape(y).clone();
    let bias4 = g.reshape_(bias, vec![1, out_c as i64, 1, 1]);
    let expanded = g.add_node(
        Op::Expand {
            target_shape: vec![batch as i64, out_c as i64, h as i64, w as i64],
        },
        vec![bias4],
        out_shape.clone(),
    );
    g.add(y, expanded)
}

/// `Conv2d` + bias. Weight layout `[C_out, C_in/g, kH, kW]`.
pub fn conv2d_bias(
    g: &mut HirMut<'_>,
    x: HirNodeId,
    weight: HirNodeId,
    bias: HirNodeId,
    batch: usize,
    out_c: usize,
    kh: usize,
    kw: usize,
    stride: [usize; 2],
    pad: [usize; 2],
    out_h: usize,
    out_w: usize,
) -> HirNodeId {
    let dt = g.shape(x).dtype();
    let out_shape = nchw_shape(batch, out_c, out_h, out_w, dt);
    let y = g.conv2d(x, weight, [kh, kw], stride, pad, 1, out_shape);
    add_bias_nchw(g, y, bias, batch, out_c, out_h, out_w)
}

/// `Conv2d` + bias with explicit `groups` (depthwise when `groups == out_c`).
pub fn conv2d_bias_groups(
    g: &mut HirMut<'_>,
    x: HirNodeId,
    weight: HirNodeId,
    bias: HirNodeId,
    batch: usize,
    out_c: usize,
    kh: usize,
    kw: usize,
    stride: [usize; 2],
    pad: [usize; 2],
    groups: usize,
    out_h: usize,
    out_w: usize,
) -> HirNodeId {
    let dt = g.shape(x).dtype();
    let out_shape = nchw_shape(batch, out_c, out_h, out_w, dt);
    let y = g.conv2d(x, weight, [kh, kw], stride, pad, groups, out_shape);
    add_bias_nchw(g, y, bias, batch, out_c, out_h, out_w)
}

/// `Conv2d` without bias (SAM encoder neck 3×3).
pub fn conv2d_no_bias(
    g: &mut HirMut<'_>,
    x: HirNodeId,
    weight: HirNodeId,
    batch: usize,
    out_c: usize,
    kh: usize,
    kw: usize,
    stride: [usize; 2],
    pad: [usize; 2],
    out_h: usize,
    out_w: usize,
) -> HirNodeId {
    let dt = g.shape(x).dtype();
    let out_shape = nchw_shape(batch, out_c, out_h, out_w, dt);
    g.conv2d(x, weight, [kh, kw], stride, pad, 1, out_shape)
}

/// `ConvTranspose2d` k=2 s=2 pad=0 + bias. Weight `[C_in, C_out, 2, 2]`.
pub fn conv_transpose2d_stride2_k2_bias(
    g: &mut HirMut<'_>,
    x: HirNodeId,
    weight: HirNodeId,
    bias: HirNodeId,
    batch: usize,
    out_c: usize,
    h: usize,
    w: usize,
) -> HirNodeId {
    let dt = g.shape(x).dtype();
    let out_h = h * 2;
    let out_w = w * 2;
    let out_shape = nchw_shape(batch, out_c, out_h, out_w, dt);
    let y = g.conv_transpose2d(
        x,
        weight,
        [2, 2],
        [2, 2],
        [0, 0],
        [1, 1],
        [0, 0],
        1,
        out_shape,
    );
    add_bias_nchw(g, y, bias, batch, out_c, out_h, out_w)
}

/// `MaxPool2d` 2×2 stride 2, padding 0.
pub fn max_pool2d_2x2(
    g: &mut HirMut<'_>,
    x: HirNodeId,
    batch: usize,
    c: usize,
    h: usize,
    w: usize,
) -> HirNodeId {
    use rlx_ir::op::{Op, ReduceOp};
    let dt = g.shape(x).dtype();
    let out_h = h / 2;
    let out_w = w / 2;
    let out_shape = nchw_shape(batch, c, out_h, out_w, dt);
    g.add_node(
        Op::Pool {
            kind: ReduceOp::Max,
            kernel_size: vec![2, 2],
            stride: vec![2, 2],
            padding: vec![0, 0],
        },
        vec![x],
        out_shape,
    )
}

/// `AvgPool2d` with explicit kernel and stride (NCHW).
pub fn avg_pool2d(
    g: &mut HirMut<'_>,
    x: HirNodeId,
    kernel: [usize; 2],
    stride: [usize; 2],
    batch: usize,
    c: usize,
    h: usize,
    w: usize,
) -> HirNodeId {
    use rlx_ir::op::{Op, ReduceOp};
    let dt = g.shape(x).dtype();
    let out_h = (h.saturating_sub(kernel[0])) / stride[0] + 1;
    let out_w = (w.saturating_sub(kernel[1])) / stride[1] + 1;
    let out_shape = nchw_shape(batch, c, out_h, out_w, dt);
    g.add_node(
        Op::Pool {
            kind: ReduceOp::Mean,
            kernel_size: kernel.to_vec(),
            stride: stride.to_vec(),
            padding: vec![0, 0],
        },
        vec![x],
        out_shape,
    )
}

/// `ConvTranspose2d` k=3 s=2 (ocrs detection upsample) + bias; trims 1px from H/W if needed.
pub fn conv_transpose2d_k3s2_bias_trim(
    g: &mut HirMut<'_>,
    x: HirNodeId,
    weight: HirNodeId,
    bias: HirNodeId,
    batch: usize,
    out_c: usize,
    in_h: usize,
    in_w: usize,
    target_h: usize,
    target_w: usize,
) -> HirNodeId {
    let dt = g.shape(x).dtype();
    let raw_h = in_h * 2 + 1;
    let raw_w = in_w * 2 + 1;
    let out_shape = nchw_shape(batch, out_c, raw_h, raw_w, dt);
    let y = g.conv_transpose2d(
        x,
        weight,
        [3, 3],
        [2, 2],
        [0, 0],
        [1, 1],
        [0, 0],
        1,
        out_shape,
    );
    let mut y = add_bias_nchw(g, y, bias, batch, out_c, raw_h, raw_w);
    if raw_h > target_h {
        y = g.narrow_(y, 2, 0, target_h);
    }
    if raw_w > target_w {
        y = g.narrow_(y, 3, 0, target_w);
    }
    y
}

/// Sigmoid activation on NCHW tensor.
pub fn sigmoid_nchw(g: &mut HirMut<'_>, x: HirNodeId) -> HirNodeId {
    use rlx_ir::op::Activation;
    let s = g.shape(x).clone();
    g.activation(Activation::Sigmoid, x, s)
}

/// SAM/candle `LayerNorm2d` on NCHW.
pub fn layer_norm2d_nchw(
    g: &mut HirMut<'_>,
    x: HirNodeId,
    gamma: HirNodeId,
    beta: HirNodeId,
    eps: f32,
) -> HirNodeId {
    g.layer_norm2d(x, gamma, beta, eps)
}
