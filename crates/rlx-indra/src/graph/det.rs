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

//! Line detector (`det.onnx`): a MobileNet backbone in four stages, a
//! top-down FPN to 1/4 scale, the stem's features folded in by
//! space-to-depth, and a head that pixel-shuffles to 1/2 scale.
//!
//! `x [1, 1, H, W]` (ink, `H, W % 32 == 0`) → `[1, C, H/2, W/2]`: channel 0
//! the line-core logit, 1–2 the distances to the line's top and bottom
//! (softplus × 8, input pixels); a fourth channel, when the model has one,
//! carries the picture map through the same softplus × 8.

use super::classifier::{downsample, inverted_residual};
use super::{Builder, Weights};
use anyhow::{Result, bail};
use rlx_ir::hir::HirNodeId;
use rlx_ir::op::Activation;
use rlx_ir::{Graph, HirGraphExt};
use std::collections::HashMap;

/// Input sides must divide by this.
pub const INPUT_MULTIPLE: usize = 32;
/// The output is at 1/OUT_STRIDE of the input.
pub const OUT_STRIDE: usize = 2;
/// Scale of the two distance channels (`softplus · 8`).
const DIST_SCALE: f32 = 8.0;

pub fn build(
    w: &Weights,
    height: usize,
    width: usize,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    if !height.is_multiple_of(INPUT_MULTIPLE) || !width.is_multiple_of(INPUT_MULTIPLE) {
        bail!("detector input {height}×{width} is not a multiple of {INPUT_MULTIPLE}");
    }
    let hs = Some(Activation::HardSwish);
    let mut b = Builder::new("indra_det", w);
    let x = b.input("x", &[1, 1, height, width]);
    let stem = b.conv_act(x, "stem.0", 2, hs)?;

    // Backbone: stage s, unit 0 downsamples, the rest are inverted residuals.
    let mut feats = Vec::new();
    let mut h = stem;
    for s in 0.. {
        if !w.has(&format!("stages.{s}.0.0.0.weight")) {
            break;
        }
        h = downsample(&mut b, h, &format!("stages.{s}.0"))?;
        for j in 1.. {
            let unit = format!("stages.{s}.{j}");
            if !w.has(&format!("{unit}.body.0.0.weight")) {
                break;
            }
            h = inverted_residual(&mut b, h, &unit)?;
        }
        feats.push(h);
    }
    if feats.is_empty() {
        bail!("detector has no stages");
    }

    // Top-down FPN: lateral 1×1 per stage, nearest 2× upsampling, sum.
    let mut p: Option<HirNodeId> = None;
    for (s, f) in feats.iter().enumerate().rev() {
        let lat = b.conv_act(*f, &format!("lat.{s}.0"), 1, hs)?;
        p = Some(match p {
            None => lat,
            Some(up) => {
                let up = b.m().resize_nearest_2x(up);
                b.m().add(lat, up)
            }
        });
    }
    let p = b.conv_act(p.unwrap(), "smooth.0", 1, hs)?;

    // Stem lateral at 1/2 scale, folded to 1/4 (channel = c·4 + dy·2 + dx).
    let sl = b.conv_act(stem, "stem_lat.0", 1, hs)?;
    let [_, c, h2, w2] = b.dims(sl)[..] else {
        unreachable!()
    };
    let (ci, hi, wi) = (c as i64, (h2 / 2) as i64, (w2 / 2) as i64);
    let sl = b.m().reshape_(sl, vec![1, ci, hi, 2, wi, 2]);
    let sl = b.m().transpose_(sl, vec![0, 1, 3, 5, 2, 4]);
    let sl = b.m().reshape_(sl, vec![1, ci * 4, hi, wi]);
    let h = b.m().concat_(vec![p, sl], 1);

    let h = b.conv_act(h, "head.0.0", 1, hs)?;
    let h = b.conv_act(h, "head.1.0", 1, hs)?;
    let h = b.conv_act(h, "head.2", 1, None)?; // [1, 4·C, H/4, W/4]

    // DepthToSpace (CRD, block 2): out[c, 2y+i, 2x+j] = in[4c + 2i + j, y, x].
    let [_, c4, h4, w4] = b.dims(h)[..] else {
        unreachable!()
    };
    let (co, hi, wi) = ((c4 / 4) as i64, h4 as i64, w4 as i64);
    let h = b.m().reshape_(h, vec![1, co, 2, 2, hi, wi]);
    let h = b.m().transpose_(h, vec![0, 1, 4, 2, 5, 3]);
    let h = b.m().reshape_(h, vec![1, co, hi * 2, wi * 2]);

    // Channel 0 stays a logit; the export applies softplus · 8 to every other.
    let co = co as usize;
    let core = b.m().narrow_(h, 1, 0, 1);
    let rest = b.m().narrow_(h, 1, 1, co - 1);
    let rest = b.act(rest, Activation::Softplus);
    let rest = b.scale(rest, DIST_SCALE, "det.dist_scale");
    let out = b.m().concat_(vec![core, rest], 1);
    b.finish(vec![out])
}
