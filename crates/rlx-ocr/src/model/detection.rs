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

//! ocrs text-detection U-Net ([`ocrs_models::DetectionModel`](https://github.com/robertknight/ocrs-models)).

use super::weights::{
    DET_DW_KEYS, DET_ONNX_PW, OcrGraphBuilder, assert_weights_drained, detection_input_hw,
};
use anyhow::Result;
use rlx_core::vision_ops_ir::{
    conv_transpose2d_k3s2_bias_trim, conv2d_bias, conv2d_bias_groups, max_pool2d_2x2, sigmoid_nchw,
};
use rlx_core::weight_map::WeightMap;
use rlx_ir::hir::HirNodeId;
use rlx_ir::{DType, HirGraphExt, Shape};

/// Fixed compile-time input size for the HF detection checkpoint (override via `OCR_DETECTION_HW`).
#[allow(dead_code)]
pub const DEFAULT_DETECTION_INPUT_HW: (usize, usize) = (800, 600);

#[derive(Clone, Copy, Debug)]
pub struct DetectionGraphConfig {
    pub batch: usize,
    pub height: usize,
    pub width: usize,
}

impl Default for DetectionGraphConfig {
    fn default() -> Self {
        let (height, width) = detection_input_hw();
        Self {
            batch: 1,
            height,
            width,
        }
    }
}

/// Channel widths at each U-Net level (matches `ocrs_models`).
const DEPTH_SCALE: [usize; 7] = [8, 16, 32, 32, 64, 128, 256];

pub fn build_detection_graph(
    wm: &mut WeightMap,
    cfg: DetectionGraphConfig,
) -> Result<(rlx_ir::Graph, std::collections::HashMap<String, Vec<f32>>)> {
    build_detection_graph_to_stage(wm, cfg, None)
}

/// Build the detection graph, optionally stopping early and emitting the
/// intermediate feature map. Stage numbering: `0` = after in_conv, `1..=6` =
/// after each encoder down-level, `7..=12` = after each decoder up-level,
/// `None` = full (logits → sigmoid mask). Used by the Metal-vs-CPU bisect.
pub fn build_detection_graph_to_stage(
    wm: &mut WeightMap,
    cfg: DetectionGraphConfig,
    stop: Option<u8>,
) -> Result<(rlx_ir::Graph, std::collections::HashMap<String, Vec<f32>>)> {
    let mut b = OcrGraphBuilder::new("ocr_detection");
    let f = DType::F32;
    let batch = cfg.batch;
    let mut h = cfg.height;
    let mut w = cfg.width;

    let image = b.m().input("image", Shape::new(&[batch, 1, h, w], f));

    // Sub-op bisection of the in_conv (stages 50..=53: after dw0/pw0/dw1/pw1).
    if matches!(stop, Some(50..=53)) {
        let s = stop.unwrap();
        let c0 = DEPTH_SCALE[0];
        let mut x = depthwise_conv(&mut b, wm, image, DET_DW_KEYS[0], 1, batch, h, w)?;
        if s == 50 {
            b.m().set_outputs(vec![x]);
            return b.finish();
        }
        let (pw0_w, pw0_b) = DET_ONNX_PW[0];
        x = pointwise_relu(&mut b, wm, x, pw0_w, pw0_b, 1, c0, batch, h, w)?;
        if s == 51 {
            b.m().set_outputs(vec![x]);
            return b.finish();
        }
        x = depthwise_conv(&mut b, wm, x, DET_DW_KEYS[1], c0, batch, h, w)?;
        if s == 52 {
            b.m().set_outputs(vec![x]);
            return b.finish();
        }
        let (pw1_w, pw1_b) = DET_ONNX_PW[1];
        x = pointwise_relu(&mut b, wm, x, pw1_w, pw1_b, c0, c0, batch, h, w)?;
        b.m().set_outputs(vec![x]);
        return b.finish();
    }

    let mut block = 0usize;
    let mut x = double_conv(
        &mut b,
        wm,
        image,
        &mut block,
        1,
        DEPTH_SCALE[0],
        batch,
        h,
        w,
    )?;
    if stop == Some(0) {
        b.m().set_outputs(vec![x]);
        return b.finish();
    }
    let in_conv_skip = (x, h, w);

    let mut x_down: Vec<(HirNodeId, usize, usize)> = Vec::new();
    for level in 0..DEPTH_SCALE.len() - 1 {
        let in_c = DEPTH_SCALE[level];
        let out_c = DEPTH_SCALE[level + 1];
        x = double_conv(&mut b, wm, x, &mut block, in_c, out_c, batch, h, w)?;
        x = max_pool2d_2x2(&mut b.m(), x, batch, out_c, h, w);
        h /= 2;
        w /= 2;
        x_down.push((x, h, w));
        if stop == Some(1 + level as u8) {
            b.m().set_outputs(vec![x]);
            return b.finish();
        }
    }

    let mut x_up = x;
    let mut up_h = h;
    let mut up_w = w;
    let mut up_stage = 7u8;
    for up_idx in (0..DEPTH_SCALE.len() - 1).rev() {
        let out_c = DEPTH_SCALE[up_idx];
        let cross_c = DEPTH_SCALE[up_idx];
        let (skip, skip_h, skip_w) = if up_idx == 0 {
            (in_conv_skip.0, in_conv_skip.1, in_conv_skip.2)
        } else {
            let (skip_node, sh, sw) = x_down[up_idx - 1];
            (skip_node, sh, sw)
        };

        let up_w_key = format!("up.{up_idx}.up.weight");
        let up_b_key = format!("up.{up_idx}.up.bias");
        let up_weight = b.load_param(wm, &up_w_key)?;
        let up_bias = b.load_param(wm, &up_b_key)?;
        let upscaled = conv_transpose2d_k3s2_bias_trim(
            &mut b.m(),
            x_up,
            up_weight,
            up_bias,
            batch,
            out_c,
            up_h,
            up_w,
            skip_h,
            skip_w,
        );
        up_h = skip_h;
        up_w = skip_w;

        let cat = b.m().concat_(vec![upscaled, skip], 1);
        x_up = double_conv(
            &mut b,
            wm,
            cat,
            &mut block,
            out_c + cross_c,
            out_c,
            batch,
            up_h,
            up_w,
        )?;
        if stop == Some(up_stage) {
            b.m().set_outputs(vec![x_up]);
            return b.finish();
        }
        up_stage += 1;
    }

    let out_w = b.load_param(wm, "out_conv.0.weight")?;
    let out_b = b.load_param(wm, "out_conv.0.bias")?;
    let logits = conv2d_bias(
        &mut b.m(),
        x_up,
        out_w,
        out_b,
        batch,
        1,
        1,
        1,
        [1, 1],
        [0, 0],
        up_h,
        up_w,
    );
    if stop == Some(13) {
        b.m().set_outputs(vec![logits]);
        return b.finish();
    }
    let mask = sigmoid_nchw(&mut b.m(), logits);
    b.m().set_outputs(vec![mask]);

    if stop.is_none() {
        assert_weights_drained(wm, "detection graph")?;
    }
    b.finish()
}

fn double_conv(
    b: &mut OcrGraphBuilder,
    wm: &mut WeightMap,
    mut x: HirNodeId,
    block: &mut usize,
    in_c: usize,
    out_c: usize,
    batch: usize,
    h: usize,
    w: usize,
) -> Result<HirNodeId> {
    let (pw0_w, pw0_b) = DET_ONNX_PW[*block];
    // `DepthwiseConv`: dw 3×3 → pw 1×1 (+ fused BN in onnx keys) → ReLU (once).
    x = depthwise_conv(b, wm, x, DET_DW_KEYS[*block], in_c, batch, h, w)?;
    x = pointwise_relu(b, wm, x, pw0_w, pw0_b, in_c, out_c, batch, h, w)?;
    *block += 1;
    let (pw1_w, pw1_b) = DET_ONNX_PW[*block];
    x = depthwise_conv(b, wm, x, DET_DW_KEYS[*block], out_c, batch, h, w)?;
    x = pointwise_relu(b, wm, x, pw1_w, pw1_b, out_c, out_c, batch, h, w)?;
    *block += 1;
    Ok(x)
}

fn depthwise_conv(
    b: &mut OcrGraphBuilder,
    wm: &mut WeightMap,
    x: HirNodeId,
    dw_key: &str,
    channels: usize,
    batch: usize,
    h: usize,
    w: usize,
) -> Result<HirNodeId> {
    let weight = b.load_param(wm, dw_key)?;
    let bias = b.zero_bias(channels)?;
    Ok(conv2d_bias_groups(
        &mut b.m(),
        x,
        weight,
        bias,
        batch,
        channels,
        3,
        3,
        [1, 1],
        [1, 1],
        channels,
        h,
        w,
    ))
}

fn pointwise_relu(
    b: &mut OcrGraphBuilder,
    wm: &mut WeightMap,
    x: HirNodeId,
    w_key: &str,
    b_key: &str,
    _in_c: usize,
    out_c: usize,
    batch: usize,
    h: usize,
    w: usize,
) -> Result<HirNodeId> {
    let weight = b.load_param(wm, w_key)?;
    let bias = b.load_param(wm, b_key)?;
    let y = conv2d_bias(
        &mut b.m(),
        x,
        weight,
        bias,
        batch,
        out_c,
        1,
        1,
        [1, 1],
        [0, 0],
        h,
        w,
    );
    Ok(b.m().relu(y))
}
