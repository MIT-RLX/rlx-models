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

//! The `cr_tr_latincyrillic_v3` recognizer as a native rlx-ir graph.
//!
//! CRNN + CTC: VGG conv front (11× conv, 5 pools; H32→1, W→seq) → 2× bidirectional
//! LSTM(128) with the two directions **summed** → FC(439) logits.
//! Weight layout/gate conventions (planar 8-bit LUT convs, uint8 per-channel FC,
//! LSTM col-layout with gate order `i,f,o,g`) are validated against a numeric fixture.

use crate::graph::OcrGraphBuilder;
use anyhow::{Result, bail};
use rlx_core::vision_ops_ir::{conv2d_bias, max_pool2d_2x2, nchw_shape};
use rlx_core::weight_map::WeightMap;
use rlx_ir::hir::{HirMut, HirNodeId};
use rlx_ir::op::{Op, ReduceOp};
use rlx_ir::{DType, HirGraphExt, Shape};

pub const REC_HEIGHT: usize = 32;
pub const NUM_CLASSES: usize = 439;
pub const HIDDEN: usize = 128;
const FEAT: usize = 192;

/// Build the recognition graph for a fixed input width (must be a multiple of 4).
/// Input `"image"` = `[batch, 1, 32, width]`; output = `[batch, seq, 439]` raw logits.
pub fn build_recognition_graph(
    wm: &mut WeightMap,
    batch: usize,
    width: usize,
) -> Result<(rlx_ir::Graph, std::collections::HashMap<String, Vec<f32>>)> {
    if !width.is_multiple_of(4) {
        bail!("recognition width {width} must be a multiple of 4");
    }
    let mut b = OcrGraphBuilder::new("ocr2_recognition");
    let (mut h, mut w) = (REC_HEIGHT, width);
    let mut x = b
        .m()
        .input("image", Shape::new(&[batch, 1, h, w], DType::F32));

    // Stage 1: conv0(1→24) conv1(24→24) → pool 2×2/2
    x = conv_relu(&mut b, wm, x, 0, batch, 24, h, w)?;
    x = conv_relu(&mut b, wm, x, 1, batch, 24, h, w)?;
    x = max_pool2d_2x2(&mut b.m(), x, batch, 24, h, w);
    h /= 2;
    w /= 2;

    // Stage 2: conv2(24→48) conv3(48→48) → pool 2×2/2
    x = conv_relu(&mut b, wm, x, 2, batch, 48, h, w)?;
    x = conv_relu(&mut b, wm, x, 3, batch, 48, h, w)?;
    x = max_pool2d_2x2(&mut b.m(), x, batch, 48, h, w);
    h /= 2;
    w /= 2;

    // Stage 3: conv4(48→96) conv5(96→96) conv6(96→96) → pool k[2,2] s[2,1] pad[0,1] (H/2, W+1)
    x = conv_relu(&mut b, wm, x, 4, batch, 96, h, w)?;
    x = conv_relu(&mut b, wm, x, 5, batch, 96, h, w)?;
    x = conv_relu(&mut b, wm, x, 6, batch, 96, h, w)?;
    let (oh, ow) = ((h - 2) / 2 + 1, w + 1);
    x = pool_max(&mut b.m(), x, batch, 96, oh, ow, [2, 2], [2, 1], [0, 1]);
    h = oh;
    w = ow;

    // Stage 4: conv7(96→192) conv8 conv9 → pool k[2,2] s[2,1] pad0 (H/2, W-1)
    x = conv_relu(&mut b, wm, x, 7, batch, 192, h, w)?;
    x = conv_relu(&mut b, wm, x, 8, batch, 192, h, w)?;
    x = conv_relu(&mut b, wm, x, 9, batch, 192, h, w)?;
    let (oh, ow) = ((h - 2) / 2 + 1, (w - 2) + 1);
    x = pool_max(&mut b.m(), x, batch, 192, oh, ow, [2, 2], [2, 1], [0, 0]);
    h = oh;
    w = ow;

    // Stage 5: conv10(192→192, 2×2 asymmetric pad b1/r1, relu) → pool k[2,1] s[2,1] (H/2, W)
    x = conv10_asym(&mut b, wm, x, batch, h, w)?;
    let (oh, ow) = ((h - 2) / 2 + 1, w);
    x = pool_max(&mut b.m(), x, batch, 192, oh, ow, [2, 1], [2, 1], [0, 0]);
    h = oh;
    w = ow;
    debug_assert_eq!(h, 1, "conv front must collapse height to 1");
    let seq = w;

    // [batch, 192, 1, seq] → [batch, seq, 192]
    let x = b
        .m()
        .reshape_(x, vec![batch as i64, FEAT as i64, seq as i64]);
    let x = b.m().transpose_(x, vec![0, 2, 1]);

    // 2× bidirectional LSTM, directions summed.
    let x = lstm_sum(&mut b, wm, x, 0, batch, seq, HIDDEN)?;
    let x = lstm_sum(&mut b, wm, x, 1, batch, seq, HIDDEN)?;

    // FC 128→439 (+bias)
    let fc_w = b.load_param(wm, "fc.weight")?; // [128, 439]
    let fc_b = b.load_param(wm, "fc.bias")?; // [439]
    let logits = b.m().mm(x, fc_w); // [batch, seq, 439]
    let bias3 = b.m().reshape_(fc_b, vec![1, 1, NUM_CLASSES as i64]);
    let logits = b.m().add(logits, bias3);
    b.m().set_outputs(vec![logits]);
    b.finish()
}

fn conv_relu(
    b: &mut OcrGraphBuilder,
    wm: &mut WeightMap,
    x: HirNodeId,
    idx: usize,
    batch: usize,
    out_c: usize,
    h: usize,
    w: usize,
) -> Result<HirNodeId> {
    let weight = b.load_param(wm, &format!("conv.{idx}.weight"))?;
    let bias = b.load_param(wm, &format!("conv.{idx}.bias"))?;
    let y = conv2d_bias(
        &mut b.m(),
        x,
        weight,
        bias,
        batch,
        out_c,
        3,
        3,
        [1, 1],
        [1, 1],
        h,
        w,
    );
    Ok(b.m().relu(y))
}

/// Final 2×2 conv with asymmetric padding (bottom 1, right 1). Explicit
/// zero-pad then a valid conv, so output spatial dims are unchanged (`h`,`w`).
fn conv10_asym(
    b: &mut OcrGraphBuilder,
    wm: &mut WeightMap,
    x: HirNodeId,
    batch: usize,
    h: usize,
    w: usize,
) -> Result<HirNodeId> {
    let zrow = b.zeros("conv10.padrow", &[batch, FEAT, 1, w]);
    let x = b.m().concat_(vec![x, zrow], 2); // [batch,192,h+1,w]
    let zcol = b.zeros("conv10.padcol", &[batch, FEAT, h + 1, 1]);
    let x = b.m().concat_(vec![x, zcol], 3); // [batch,192,h+1,w+1]
    let weight = b.load_param(wm, "conv.10.weight")?;
    let bias = b.load_param(wm, "conv.10.bias")?;
    let y = conv2d_bias(
        &mut b.m(),
        x,
        weight,
        bias,
        batch,
        FEAT,
        2,
        2,
        [1, 1],
        [0, 0],
        h,
        w,
    );
    Ok(b.m().relu(y))
}

fn pool_max(
    g: &mut HirMut<'_>,
    x: HirNodeId,
    batch: usize,
    c: usize,
    out_h: usize,
    out_w: usize,
    kernel: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
) -> HirNodeId {
    let dt = g.shape(x).dtype();
    let out_shape = nchw_shape(batch, c, out_h, out_w, dt);
    g.add_node(
        Op::Pool {
            kind: ReduceOp::Max,
            kernel_size: kernel.to_vec(),
            stride: stride.to_vec(),
            padding: padding.to_vec(),
        },
        vec![x],
        out_shape,
    )
}

/// One bidirectional LSTM layer; sum the forward/backward hidden halves (directions
/// are merged by addition, keeping the feature dim at `hidden`).
fn lstm_sum(
    b: &mut OcrGraphBuilder,
    wm: &mut WeightMap,
    x: HirNodeId,
    idx: usize,
    batch: usize,
    seq: usize,
    hidden: usize,
) -> Result<HirNodeId> {
    let w_ih = b.load_param(wm, &format!("lstm.{idx}.w_ih"))?; // [2*4H, in]
    let w_hh = b.load_param(wm, &format!("lstm.{idx}.w_hh"))?; // [2*4H, H]
    let bias = b.load_param(wm, &format!("lstm.{idx}.bias"))?; // [2*4H]
    let out_shape = Shape::new(&[batch, seq, 2 * hidden], DType::F32);
    let y = b
        .m()
        .0
        .lstm(x, w_ih, w_hh, bias, hidden, 1, true, out_shape); // [b,seq,2H]
    let fwd = b.m().narrow_(y, 2, 0, hidden);
    let rev = b.m().narrow_(y, 2, hidden, hidden);
    Ok(b.m().add(fwd, rev))
}
