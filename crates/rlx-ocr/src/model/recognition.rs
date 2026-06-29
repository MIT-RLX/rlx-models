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

/// Where `build_recognition_graph_inner` stops emitting — the bisect/stage
/// tests build prefixes of the full recognition graph.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Stage {
    AfterG1,
    AfterG2,
    AfterLogits,
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

    x = conv_relu(
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
    x = conv_relu(
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
    x = conv_relu(
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
    build_recognition_graph_inner(wm, cfg, Some(Stage::AfterG1))
}

/// Recognition graph ending after the second GRU (`[seq, batch, 512]`).
pub fn build_recognition_after_g2_graph(
    wm: &mut WeightMap,
    cfg: RecognitionGraphConfig,
) -> Result<(rlx_ir::Graph, std::collections::HashMap<String, Vec<f32>>)> {
    build_recognition_graph_inner(wm, cfg, Some(Stage::AfterG2))
}

/// Recognition graph ending after the linear head (`[seq, batch, classes]` logits).
pub fn build_recognition_after_logits_graph(
    wm: &mut WeightMap,
    cfg: RecognitionGraphConfig,
) -> Result<(rlx_ir::Graph, std::collections::HashMap<String, Vec<f32>>)> {
    build_recognition_graph_inner(wm, cfg, Some(Stage::AfterLogits))
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
    stop: Option<Stage>,
) -> Result<(rlx_ir::Graph, std::collections::HashMap<String, Vec<f32>>)> {
    let mut b = OcrGraphBuilder::new("ocr_recognition");
    let batch = cfg.batch;
    let h = RECOGNITION_HEIGHT;
    let w = cfg.width;

    let image = b
        .m()
        .input("image", Shape::new(&[batch, 1, h, w], DType::F32));

    let (x, seq) = build_recognition_conv_front(&mut b, wm, image, batch, h, w)?;

    // Two stacked bidirectional GRUs on the native rlx `Op::Gru`. The conv front
    // is seq-first `[seq, batch, FEAT]`; rlx GRU is batch-first, so transpose
    // around it. ocrs ships ONNX-layout GRU weights (gate order z,r,h) which
    // `gru_layer` repacks to rlx/PyTorch layout.
    let xb = b.m().transpose_(x, vec![1, 0, 2]); // [batch, seq, FEAT]
    let g1b = gru_layer(
        &mut b,
        wm,
        xb,
        "onnx::GRU_422",
        "onnx::GRU_423",
        "onnx::GRU_421",
        batch,
        seq,
        FEAT,
        HIDDEN,
    )?; // [batch, seq, 2*HIDDEN]
    if stop == Some(Stage::AfterG1) {
        let g1 = b.m().transpose_(g1b, vec![1, 0, 2]); // [seq, batch, 2*HIDDEN]
        b.m().set_outputs(vec![g1]);
        return b.finish();
    }

    let g2b = gru_layer(
        &mut b,
        wm,
        g1b,
        "onnx::GRU_465",
        "onnx::GRU_466",
        "onnx::GRU_464",
        batch,
        seq,
        2 * HIDDEN,
        HIDDEN,
    )?; // [batch, seq, 2*HIDDEN]
    let g2 = b.m().transpose_(g2b, vec![1, 0, 2]); // [seq, batch, 2*HIDDEN]
    if stop == Some(Stage::AfterG2) {
        b.m().set_outputs(vec![g2]);
        return b.finish();
    }

    let head_w = b.load_param(wm, "onnx::MatMul_467")?;
    let head_b = b.load_param(wm, "output.0.bias")?;
    let logits = b.m().mm(g2, head_w);
    let logits = add_bias_seq(&mut b, logits, head_b, batch, seq, NUM_CLASSES)?;
    if stop == Some(Stage::AfterLogits) {
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

/// Insert a constant f32 param `[len]` and return its node.
fn gru_param(b: &mut OcrGraphBuilder, key: String, len: usize, data: Vec<f32>) -> HirNodeId {
    debug_assert_eq!(data.len(), len);
    let id = b.m().param(&key, Shape::new(&[len], DType::F32));
    b.params.insert(key, data);
    id
}

/// One bidirectional GRU layer via native `Op::Gru`. Loads ocrs ONNX-layout
/// weights — `W` `[2,3h,in]`, `R` `[2,3h,h]`, `B` `[2,6h]`, gate order **z,r,h** —
/// and repacks them to rlx's PyTorch layout (per-direction contiguous, gate order
/// **r,z,n**, separate `b_ih`/`b_hh`). `x` is `[batch, seq, in]`; output
/// `[batch, seq, 2*hidden]` (forward hidden then backward hidden).
#[allow(clippy::too_many_arguments)]
fn gru_layer(
    b: &mut OcrGraphBuilder,
    wm: &mut WeightMap,
    x: HirNodeId,
    w_key: &str,
    r_key: &str,
    b_key: &str,
    batch: usize,
    seq: usize,
    in_size: usize,
    hidden: usize,
) -> Result<HirNodeId> {
    use anyhow::Context;
    const NUM_DIR: usize = 2;
    // ONNX gate order is [z, r, h]; rlx wants [r, z, n]: rlx gate i ← onnx MAP[i].
    const MAP: [usize; 3] = [1, 0, 2];
    let g3 = 3 * hidden;

    let (w_data, _) = wm
        .take(w_key)
        .with_context(|| format!("missing weight {w_key}"))?;
    let (r_data, _) = wm
        .take(r_key)
        .with_context(|| format!("missing weight {r_key}"))?;
    let (b_data, _) = wm
        .take(b_key)
        .with_context(|| format!("missing weight {b_key}"))?;

    let mut w_ih = vec![0f32; NUM_DIR * g3 * in_size];
    let mut w_hh = vec![0f32; NUM_DIR * g3 * hidden];
    let mut b_ih = vec![0f32; NUM_DIR * g3];
    let mut b_hh = vec![0f32; NUM_DIR * g3];
    for d in 0..NUM_DIR {
        for rg in 0..3 {
            let og = MAP[rg];
            let wblk = hidden * in_size;
            let (ws, wd) = ((d * 3 + og) * wblk, (d * 3 + rg) * wblk);
            w_ih[wd..wd + wblk].copy_from_slice(&w_data[ws..ws + wblk]);
            let rblk = hidden * hidden;
            let (rs, rd) = ((d * 3 + og) * rblk, (d * 3 + rg) * rblk);
            w_hh[rd..rd + rblk].copy_from_slice(&r_data[rs..rs + rblk]);
            // B per direction = [Wb(3h) | Rb(3h)], each gate `[hidden]`.
            let bd = (d * 3 + rg) * hidden;
            let wb = d * 6 * hidden + og * hidden;
            let rb = d * 6 * hidden + g3 + og * hidden;
            b_ih[bd..bd + hidden].copy_from_slice(&b_data[wb..wb + hidden]);
            b_hh[bd..bd + hidden].copy_from_slice(&b_data[rb..rb + hidden]);
        }
    }

    let wih = gru_param(b, format!("{w_key}.rlx_wih"), w_ih.len(), w_ih);
    let whh = gru_param(b, format!("{r_key}.rlx_whh"), w_hh.len(), w_hh);
    let bih = gru_param(b, format!("{b_key}.rlx_bih"), b_ih.len(), b_ih);
    let bhh = gru_param(b, format!("{b_key}.rlx_bhh"), b_hh.len(), b_hh);

    let shape = Shape::new(&[batch, seq, NUM_DIR * hidden], DType::F32);
    // `gru` lives on `HirModule` (the public `.0` of `HirMut`).
    Ok(b.m().0.gru(x, wih, whh, bih, bhh, hidden, 1, true, shape))
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
