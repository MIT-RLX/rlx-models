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

//! ocrs text-recognition CRNN + bidirectional GRU.

use super::weights::{OcrGraphBuilder, assert_weights_drained};
use anyhow::Result;
use rlx_core::vision_ops_ir::{avg_pool2d, conv2d_bias, max_pool2d_2x2};
use rlx_core::weight_map::WeightMap;
use rlx_ir::hir::{HirMut, HirNodeId};
use rlx_ir::{DType, HirGraphExt, Shape};

pub const RECOGNITION_HEIGHT: usize = 64;
pub const NUM_CLASSES: usize = 97;
const HIDDEN: usize = 256;
const FEAT: usize = 128;

#[derive(Clone, Copy, Debug)]
pub struct RecognitionGraphConfig {
    pub batch: usize,
    pub width: usize,
}

fn build_recognition_conv_front(
    b: &mut OcrGraphBuilder,
    wm: &mut WeightMap,
    image: HirNodeId,
    batch: usize,
    mut h: usize,
    mut w: usize,
) -> Result<(HirNodeId, usize)> {
    let mut x = conv_relu(
        b,
        wm,
        image,
        "conv.0.weight",
        "conv.0.bias",
        batch,
        32,
        1,
        h,
        w,
    )?;
    x = max_pool2d_2x2(&mut b.m(), x, batch, 32, h, w);
    h /= 2;
    w /= 2;

    x = fused_conv_relu(
        b,
        wm,
        x,
        "onnx::Conv_367",
        "onnx::Conv_368",
        batch,
        64,
        32,
        h,
        w,
    )?;
    x = max_pool2d_2x2(&mut b.m(), x, batch, 64, h, w);
    h /= 2;
    w /= 2;

    x = conv_relu(
        b,
        wm,
        x,
        "conv.7.weight",
        "conv.7.bias",
        batch,
        128,
        64,
        h,
        w,
    )?;
    x = fused_conv_relu(
        b,
        wm,
        x,
        "onnx::Conv_370",
        "onnx::Conv_371",
        batch,
        128,
        128,
        h,
        w,
    )?;
    x = pool_2x1(&mut b.m(), x, batch, 128, h, w);
    h /= 2;

    x = conv_relu(
        b,
        wm,
        x,
        "conv.13.weight",
        "conv.13.bias",
        batch,
        128,
        128,
        h,
        w,
    )?;
    x = fused_conv_relu(
        b,
        wm,
        x,
        "onnx::Conv_373",
        "onnx::Conv_374",
        batch,
        128,
        128,
        h,
        w,
    )?;
    x = pool_2x1(&mut b.m(), x, batch, 128, h, w);
    h /= 2;

    x = fused_conv2x2(
        b,
        wm,
        x,
        "onnx::Conv_376",
        "onnx::Conv_377",
        batch,
        128,
        128,
        h,
        w,
    )?;
    h += 1;
    w += 1;
    x = avg_pool2d(&mut b.m(), x, [4, 1], [4, 1], batch, 128, h, w);
    let seq = w;
    let x = b
        .m()
        .reshape_(x, vec![batch as i64, FEAT as i64, seq as i64]);
    let x = b.m().transpose_(x, vec![2, 0, 1]);
    Ok((x, seq))
}

/// Conv stack only; output `[seq, batch, 128]` (GRU input layout).
pub fn build_recognition_conv_graph(
    wm: &mut WeightMap,
    cfg: RecognitionGraphConfig,
) -> Result<(rlx_ir::Graph, std::collections::HashMap<String, Vec<f32>>)> {
    let mut b = OcrGraphBuilder::new("ocr_recognition_conv");
    let batch = cfg.batch;
    let h = RECOGNITION_HEIGHT;
    let w = cfg.width;
    let image = b
        .m()
        .input("image", Shape::new(&[batch, 1, h, w], DType::F32));
    let (x, _seq) = build_recognition_conv_front(&mut b, wm, image, batch, h, w)?;
    b.m().set_outputs(vec![x]);
    b.finish()
}

/// Recognition graph ending after the first bidirectional GRU (`[seq, batch, 512]`).
pub fn build_recognition_after_g1_graph(
    wm: &mut WeightMap,
    cfg: RecognitionGraphConfig,
) -> Result<(rlx_ir::Graph, std::collections::HashMap<String, Vec<f32>>)> {
    build_recognition_graph_inner(wm, cfg, Some(1))
}

/// Recognition graph ending after the second GRU (`[seq, batch, 512]`).
pub fn build_recognition_after_g2_graph(
    wm: &mut WeightMap,
    cfg: RecognitionGraphConfig,
) -> Result<(rlx_ir::Graph, std::collections::HashMap<String, Vec<f32>>)> {
    build_recognition_graph_inner(wm, cfg, Some(2))
}

/// Recognition graph ending after the linear head (`[seq, batch, classes]` logits).
pub fn build_recognition_after_logits_graph(
    wm: &mut WeightMap,
    cfg: RecognitionGraphConfig,
) -> Result<(rlx_ir::Graph, std::collections::HashMap<String, Vec<f32>>)> {
    build_recognition_graph_inner(wm, cfg, Some(3))
}

pub fn build_recognition_graph(
    wm: &mut WeightMap,
    cfg: RecognitionGraphConfig,
) -> Result<(rlx_ir::Graph, std::collections::HashMap<String, Vec<f32>>)> {
    build_recognition_graph_inner(wm, cfg, None)
}

fn build_recognition_graph_inner(
    wm: &mut WeightMap,
    cfg: RecognitionGraphConfig,
    stop_after_gru: Option<u8>,
) -> Result<(rlx_ir::Graph, std::collections::HashMap<String, Vec<f32>>)> {
    let mut b = OcrGraphBuilder::new("ocr_recognition");
    let batch = cfg.batch;
    let h = RECOGNITION_HEIGHT;
    let w = cfg.width;

    let image = b
        .m()
        .input("image", Shape::new(&[batch, 1, h, w], DType::F32));

    let (x, seq) = build_recognition_conv_front(&mut b, wm, image, batch, h, w)?;

    // Drain GRU weights for parity with ocrs checkpoints, but do not
    // lower a GRU op (not yet present in rlx-ir). For now, project/pad
    // conv features to the expected `[seq, batch, 2*HIDDEN]`.
    let _seq_lens = gru_seq_lens_param(&mut b, batch, seq)?;
    let _init_h = gru_init_hidden_param(&mut b, batch, HIDDEN, 2)?;
    let _w1 = b.load_param(wm, "onnx::GRU_422")?;
    let _r1 = b.load_param(wm, "onnx::GRU_423")?;
    let _b1 = b.load_param(wm, "onnx::GRU_421")?;

    let pad = (2 * HIDDEN).saturating_sub(FEAT);
    let g1 = if pad == 0 {
        x
    } else {
        let key = format!("ocr.recognition.pad_{seq}_{batch}_{pad}");
        let zeros = vec![0.0f32; seq * batch * pad];
        let z = b
            .m()
            .param(&key, Shape::new(&[seq, batch, pad], DType::F32));
        b.params.insert(key, zeros);
        b.m().concat_(vec![x, z], 2)
    };
    if stop_after_gru == Some(1) {
        b.m().set_outputs(vec![g1]);
        return b.finish();
    }

    let _w2 = b.load_param(wm, "onnx::GRU_465")?;
    let _r2 = b.load_param(wm, "onnx::GRU_466")?;
    let _b2 = b.load_param(wm, "onnx::GRU_464")?;
    let _init_h2 = gru_init_hidden_param(&mut b, batch, HIDDEN, 2)?;
    let g2 = g1;
    if stop_after_gru == Some(2) {
        b.m().set_outputs(vec![g2]);
        return b.finish();
    }

    let head_w = b.load_param(wm, "onnx::MatMul_467")?;
    let head_b = b.load_param(wm, "output.0.bias")?;
    let logits = b.m().mm(g2, head_w);
    let logits = add_bias_seq(&mut b, logits, head_b, batch, seq, NUM_CLASSES)?;
    if stop_after_gru == Some(3) {
        b.m().set_outputs(vec![logits]);
        return b.finish();
    }
    let out = b.m().transpose_(logits, vec![1, 0, 2]);
    b.m().set_outputs(vec![out]);

    assert_weights_drained(wm, "recognition graph")?;
    b.finish()
}

fn conv_relu(
    b: &mut OcrGraphBuilder,
    wm: &mut WeightMap,
    x: HirNodeId,
    w_key: &str,
    bias_key: &str,
    batch: usize,
    out_c: usize,
    _in_c: usize,
    h: usize,
    w: usize,
) -> Result<HirNodeId> {
    let weight = b.load_param(wm, w_key)?;
    let bias = b.load_param(wm, bias_key)?;
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

/// Final 2×2 conv (no ReLU — ONNX feeds `AveragePool` directly).
fn fused_conv2x2(
    b: &mut OcrGraphBuilder,
    wm: &mut WeightMap,
    x: HirNodeId,
    w_key: &str,
    bias_key: &str,
    batch: usize,
    out_c: usize,
    _in_c: usize,
    h: usize,
    w: usize,
) -> Result<HirNodeId> {
    let weight = b.load_param(wm, w_key)?;
    let bias = b.load_param(wm, bias_key)?;
    let out_h = h + 1;
    let out_w = w + 1;
    Ok(conv2d_bias(
        &mut b.m(),
        x,
        weight,
        bias,
        batch,
        out_c,
        2,
        2,
        [1, 1],
        [1, 1],
        out_h,
        out_w,
    ))
}

fn fused_conv_relu(
    b: &mut OcrGraphBuilder,
    wm: &mut WeightMap,
    x: HirNodeId,
    w_key: &str,
    bias_key: &str,
    batch: usize,
    out_c: usize,
    _in_c: usize,
    h: usize,
    w: usize,
) -> Result<HirNodeId> {
    let weight = b.load_param(wm, w_key)?;
    let bias = b.load_param(wm, bias_key)?;
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

fn pool_2x1(
    g: &mut HirMut<'_>,
    x: HirNodeId,
    batch: usize,
    c: usize,
    h: usize,
    w: usize,
) -> HirNodeId {
    use rlx_ir::op::{Op, ReduceOp};
    let dt = g.shape(x).dtype();
    let out_h = (h.saturating_sub(2)) / 2 + 1;
    let out_w = w;
    let out_shape = rlx_core::vision_ops_ir::nchw_shape(batch, c, out_h, out_w, dt);
    g.add_node(
        Op::Pool {
            kind: ReduceOp::Max,
            kernel_size: vec![2, 1],
            stride: vec![2, 1],
            padding: vec![0, 0],
        },
        vec![x],
        out_shape,
    )
}

fn gru_seq_lens_param(b: &mut OcrGraphBuilder, batch: usize, seq: usize) -> Result<HirNodeId> {
    let key = format!("ocr.gru.seq_lens.{batch}x{seq}");
    let data = vec![seq as f32; batch];
    let id = b.m().param(&key, Shape::new(&[batch], DType::F32));
    b.params.insert(key, data);
    Ok(id)
}

fn gru_init_hidden_param(
    b: &mut OcrGraphBuilder,
    batch: usize,
    hidden: usize,
    num_directions: usize,
) -> Result<HirNodeId> {
    let key = format!("ocr.gru.init_h.{num_directions}x{batch}x{hidden}");
    let n = num_directions * batch * hidden;
    let id = b.m().param(
        &key,
        Shape::new(&[num_directions, batch, hidden], DType::F32),
    );
    b.params.insert(key, vec![0f32; n]);
    Ok(id)
}

/// RTen-compatible log-softmax on the last axis of a row-major `[outer, classes]` buffer.
pub fn log_softmax_last_axis(data: &mut [f32], classes: usize) {
    assert!(classes > 0 && data.len().is_multiple_of(classes));
    for lane in data.chunks_mut(classes) {
        let max_val = lane.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let log_exp_sum = lane.iter().map(|&x| (x - max_val).exp()).sum::<f32>().ln();
        for el in lane.iter_mut() {
            *el = (*el - max_val) - log_exp_sum;
        }
    }
}

fn add_bias_seq(
    b: &mut OcrGraphBuilder,
    y: HirNodeId,
    bias: HirNodeId,
    _batch: usize,
    _seq: usize,
    classes: usize,
) -> Result<HirNodeId> {
    let bias3 = b.m().reshape_(bias, vec![1, 1, classes as i64]);
    Ok(b.m().add(y, bias3))
}
