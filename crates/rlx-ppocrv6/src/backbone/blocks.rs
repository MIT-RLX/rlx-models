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

//! Fused LCNetV4 block helpers for hand-written / parity graphs.
//!
//! Production inference uses offline-decomposed HIR in [`crate::native`].
//! These builders document the MetaFormer token/channel mixer layout
//! (DW 3×3 + optional SE + expand/GELU/compress).

use anyhow::Result;
use rlx_core::vision_ops_ir::{conv2d_bias, conv2d_bias_groups};
use rlx_ir::HirGraphExt;
use rlx_ir::hir::{HirMut, HirNodeId};

/// Fused GELU activation.
pub fn gelu(g: &mut HirMut<'_>, x: HirNodeId) -> HirNodeId {
    g.gelu(x)
}

/// Depthwise 3×3 + bias (fused reparam form).
pub fn dw_conv3(
    g: &mut HirMut<'_>,
    x: HirNodeId,
    weight: HirNodeId,
    bias: HirNodeId,
    batch: usize,
    channels: usize,
    h: usize,
    w: usize,
) -> Result<HirNodeId> {
    Ok(conv2d_bias_groups(
        g,
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

/// Pointwise 1×1 + bias.
pub fn pw_conv(
    g: &mut HirMut<'_>,
    x: HirNodeId,
    weight: HirNodeId,
    bias: HirNodeId,
    batch: usize,
    in_c: usize,
    out_c: usize,
    h: usize,
    w: usize,
) -> Result<HirNodeId> {
    let _ = in_c;
    Ok(conv2d_bias(
        g,
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
    ))
}
