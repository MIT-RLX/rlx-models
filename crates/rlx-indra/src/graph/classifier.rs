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

//! Classifiers: the line script router (`script.onnx`) and the page
//! orientation model (`orient.onnx`).

use super::{Builder, Weights};
use anyhow::{Result, bail};
use rlx_ir::hir::HirNodeId;
use rlx_ir::op::{Activation, ReduceOp};
use rlx_ir::{Graph, HirGraphExt};
use std::collections::HashMap;

/// Input columns per script-classifier column after its five 2× poolings.
pub const SCRIPT_STRIDE: usize = 32;

/// Script router: `x [B, 1, 32, W]`, `mask [B, 1, W/32]` and
/// `inv_count [B, 1]` (the masked column mean) → class probabilities `[B, G]`.
pub fn build_script(
    w: &Weights,
    batch: usize,
    width: usize,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    let mut b = Builder::new("indra_script", w);
    let x = b.input("x", &[batch, 1, super::rec::BAND, width]);
    let cols = width / SCRIPT_STRIDE;
    let mask = b.input("mask", &[batch, 1, cols]);
    let inv_count = b.input("inv_count", &[batch, 1]);
    let mut h = b.pool(x, ReduceOp::Mean, 2);
    let convs: Vec<usize> = (0..32)
        .filter(|i| w.has(&format!("m.body.{i}.weight")))
        .collect();
    if convs.is_empty() {
        bail!("script classifier has no conv layers");
    }
    for i in convs {
        h = b.conv_act(h, &format!("m.body.{i}"), 1, Some(Activation::Relu))?;
        h = b.pool(h, ReduceOp::Max, 2);
    }
    let [_, c, rows, cols2] = b.dims(h)[..] else {
        unreachable!()
    };
    debug_assert_eq!(cols2, cols);
    let h = if rows == 1 {
        b.m().reshape_(h, vec![batch as i64, c as i64, cols as i64])
    } else {
        b.m().mean(h, vec![2], false)
    };
    let h = b.mul_b(h, mask);
    let h = b.m().sum(h, vec![2], false); // [B, C]
    let h = b.mul_b(h, inv_count);
    let logits = b.gemm_t(h, "m.head")?;
    let p = b.m().sm(logits, -1);
    b.finish(vec![p])
}

/// Orientation: `x [B, 1, S, S]` page ink → class probabilities `[B, 5]`
/// (quarter turns 0–3 counter-clockwise, 4 = vertical columns).
pub fn build_orient(
    w: &Weights,
    batch: usize,
    side: usize,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    let mut b = Builder::new("indra_orient", w);
    let x = b.input("x", &[batch, 1, side, side]);
    let h = mobile_body(&mut b, x, "m.body")?;
    let h = b.m().mean(h, vec![2, 3], false);
    let logits = b.gemm_t(h, "m.head")?;
    let p = b.m().sm(logits, -1);
    b.finish(vec![p])
}

/// A MobileNet body: `{p}.{i}.0` a stride-2 stem conv, `{p}.{i}.0.0 / .1.0` a
/// stride-2 depthwise + pointwise downsample, `{p}.{i}.body.{0,1,2}.0` an
/// inverted residual (expand, depthwise, project). HardSwish after every conv
/// but a block's projection. Shared with the detector's stages.
pub fn mobile_body(b: &mut Builder<'_>, mut x: HirNodeId, p: &str) -> Result<HirNodeId> {
    let hs = Some(Activation::HardSwish);
    let mut i = 0;
    loop {
        let unit = format!("{p}.{i}");
        if b.w.has(&format!("{unit}.0.weight")) {
            x = b.conv_act(x, &format!("{unit}.0"), 2, hs)?;
        } else if b.w.has(&format!("{unit}.0.0.weight")) {
            x = downsample(b, x, &unit)?;
        } else if b.w.has(&format!("{unit}.body.0.0.weight")) {
            x = inverted_residual(b, x, &unit)?;
        } else {
            break;
        }
        i += 1;
    }
    if i == 0 {
        bail!("{p}: no layers");
    }
    Ok(x)
}

pub fn downsample(b: &mut Builder<'_>, x: HirNodeId, unit: &str) -> Result<HirNodeId> {
    let hs = Some(Activation::HardSwish);
    let x = b.conv_act(x, &format!("{unit}.0.0"), 2, hs)?;
    b.conv_act(x, &format!("{unit}.1.0"), 1, hs)
}

pub fn inverted_residual(b: &mut Builder<'_>, x: HirNodeId, unit: &str) -> Result<HirNodeId> {
    let hs = Some(Activation::HardSwish);
    let h = b.conv_act(x, &format!("{unit}.body.0.0"), 1, hs)?;
    let h = b.conv_act(h, &format!("{unit}.body.1.0"), 1, hs)?;
    let h = b.conv_act(h, &format!("{unit}.body.2.0"), 1, None)?;
    Ok(b.m().add(x, h))
}

/// `mask [B, 1, C]` and `inv_count [B, 1]` for the script router: a line
/// covers `max(width / 32, 1)` of its `cols` columns.
pub fn script_masks(widths: &[usize], cols: usize) -> (Vec<f32>, Vec<f32>) {
    let mut mask = vec![0.0; widths.len() * cols];
    let mut inv = Vec::with_capacity(widths.len());
    for (i, &w) in widths.iter().enumerate() {
        let len = (w / SCRIPT_STRIDE).max(1).min(cols);
        mask[i * cols..i * cols + len].fill(1.0);
        inv.push(1.0 / len.max(1) as f32);
    }
    (mask, inv)
}
