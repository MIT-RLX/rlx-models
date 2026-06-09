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

//! SAM2 FPN neck IR: lateral 1×1 convs + top-down nearest ×2 fusion.

use super::fpn_neck::FpnNeckWeights;
use anyhow::Result;
use rlx_core::vision_ops_ir::{bhwc_to_nchw, conv2d_bias};
use rlx_flow::CompileProfile;
use rlx_ir::hir::{HirModule, HirMut, HirNodeId};
use rlx_ir::{DType, Graph, HirGraphExt, Shape};
use rlx_runtime::{CompiledGraph, Device};
use std::collections::HashMap;

/// Per-backbone-stage lateral 1×1 (`dim_in` → `d_model`).
pub struct Sam2FpnLateralCompiled {
    graph: CompiledGraph,
    pub in_c: usize,
    pub out_c: usize,
    pub h: usize,
    pub w: usize,
}

impl Sam2FpnLateralCompiled {
    pub fn compile(
        in_c: usize,
        out_c: usize,
        h: usize,
        w: usize,
        weight: &[f32],
        bias: &[f32],
        device: Device,
    ) -> Result<Self> {
        Self::compile_with_profile(
            in_c,
            out_c,
            h,
            w,
            weight,
            bias,
            device,
            &CompileProfile::sam_encoder(),
        )
    }

    pub fn compile_with_profile(
        in_c: usize,
        out_c: usize,
        h: usize,
        w: usize,
        weight: &[f32],
        bias: &[f32],
        device: Device,
        profile: &CompileProfile,
    ) -> Result<Self> {
        let (graph, params) = build_lateral_graph(in_c, out_c, h, w, weight, bias)?;
        let mut compiled =
            rlx_core::flow_bridge::compile_graph_with_profile(device, graph, profile)?;
        for (name, data) in &params {
            compiled.set_param(name, data);
        }
        Ok(Self {
            graph: compiled,
            in_c,
            out_c,
            h,
            w,
        })
    }

    /// Encoder stage output, BHWC flat `[1, H, W, in_c]`.
    pub fn run(&mut self, stage_bhwc: &[f32]) -> Result<Vec<f32>> {
        let expected = self.in_c * self.h * self.w;
        anyhow::ensure!(
            stage_bhwc.len() == expected,
            "FPN lateral input len {} ≠ {expected}",
            stage_bhwc.len()
        );
        let outs = self.graph.run(&[("stage", stage_bhwc)]);
        Ok(outs.into_iter().next().expect("fpn lateral output"))
    }
}

/// Top-down fusion: `lat + ResizeNearest2x(prev)` at 2× resolution.
pub struct Sam2FpnTopDownCompiled {
    graph: CompiledGraph,
    pub channels: usize,
    pub prev_h: usize,
    pub prev_w: usize,
    pub out_h: usize,
    pub out_w: usize,
}

impl Sam2FpnTopDownCompiled {
    pub fn compile(channels: usize, prev_h: usize, prev_w: usize, device: Device) -> Result<Self> {
        Self::compile_with_profile(
            channels,
            prev_h,
            prev_w,
            device,
            &CompileProfile::sam_encoder(),
        )
    }

    pub fn compile_with_profile(
        channels: usize,
        prev_h: usize,
        prev_w: usize,
        device: Device,
        profile: &CompileProfile,
    ) -> Result<Self> {
        let out_h = prev_h * 2;
        let out_w = prev_w * 2;
        let (graph, _params) = build_top_down_graph(channels, prev_h, prev_w)?;
        let compiled = rlx_core::flow_bridge::compile_graph_with_profile(device, graph, profile)?;
        Ok(Self {
            graph: compiled,
            channels,
            prev_h,
            prev_w,
            out_h,
            out_w,
        })
    }

    /// `lat` / `prev`: NCHW flat `[C, H, W]` at output / previous resolution.
    pub fn run(&mut self, lat: &[f32], prev: &[f32]) -> Result<Vec<f32>> {
        let lat_n = self.channels * self.out_h * self.out_w;
        let prev_n = self.channels * self.prev_h * self.prev_w;
        anyhow::ensure!(
            lat.len() == lat_n,
            "FPN fuse lat len {} ≠ {lat_n}",
            lat.len()
        );
        anyhow::ensure!(
            prev.len() == prev_n,
            "FPN fuse prev len {} ≠ {prev_n}",
            prev.len()
        );
        let outs = self.graph.run(&[("lat", lat), ("prev", prev)]);
        Ok(outs.into_iter().next().expect("fpn top_down output"))
    }
}

/// One compiled lateral per encoder stage (index = stage 0 finest … n-1 coarsest).
pub struct Sam2FpnNeckIr {
    pub laterals: Vec<Sam2FpnLateralCompiled>,
    /// `fuses[stage_idx]` when that stage receives top-down fusion.
    pub fuses: Vec<Option<Sam2FpnTopDownCompiled>>,
    /// Per-stage sinusoidal PE `[d_model, h, w]` NCHW flat (index = stage).
    pub pos: Vec<Vec<f32>>,
}

pub fn compile_fpn_neck_ir(
    neck: &FpnNeckWeights,
    stage_hw: &[(usize, usize)],
    stage_dims: &[usize],
    device: Device,
    profile: &CompileProfile,
) -> Result<Sam2FpnNeckIr> {
    let n = stage_hw.len();
    anyhow::ensure!(
        stage_dims.len() == n && neck.conv_w.len() == n,
        "FPN compile: stage count mismatch"
    );
    let mut laterals = Vec::with_capacity(n);
    let mut pos = Vec::with_capacity(n);
    for stage_idx in 0..n {
        let (h, w) = stage_hw[stage_idx];
        pos.push(super::fpn_neck::sinusoidal_pos_2d(neck.d_model, h, w));
        let conv_idx = n - 1 - stage_idx;
        let (h, w) = stage_hw[stage_idx];
        let in_c = stage_dims[stage_idx];
        laterals.push(Sam2FpnLateralCompiled::compile_with_profile(
            in_c,
            neck.d_model,
            h,
            w,
            &neck.conv_w[conv_idx],
            &neck.conv_b[conv_idx],
            device,
            profile,
        )?);
    }
    let mut fuses: Vec<Option<Sam2FpnTopDownCompiled>> = (0..n).map(|_| None).collect();
    for &stage_idx in &neck.fpn_top_down_levels {
        anyhow::ensure!(
            stage_idx < n,
            "fpn_top_down_levels index {stage_idx} out of range"
        );
        let (h, w) = stage_hw[stage_idx];
        anyhow::ensure!(
            h % 2 == 0 && w % 2 == 0,
            "FPN top-down at stage {stage_idx} needs even h,w, got {h}×{w}"
        );
        fuses[stage_idx] = Some(Sam2FpnTopDownCompiled::compile_with_profile(
            neck.d_model,
            h / 2,
            w / 2,
            device,
            profile,
        )?);
    }
    Ok(Sam2FpnNeckIr {
        laterals,
        fuses,
        pos,
    })
}

fn build_top_down_graph(
    channels: usize,
    prev_h: usize,
    prev_w: usize,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    let f = DType::F32;
    let out_h = prev_h * 2;
    let out_w = prev_w * 2;
    let mut hir = HirModule::new("sam2_fpn_top_down");
    let mut g = HirMut::new(&mut hir);

    let lat = g.input("lat", Shape::new(&[1, channels, out_h, out_w], f));
    let prev = g.input("prev", Shape::new(&[1, channels, prev_h, prev_w], f));
    let up = g.resize_nearest_2x(prev);
    let out = g.add(lat, up);

    hir.set_outputs(vec![out]);
    let graph = Graph::from_hir(hir).map_err(|e| anyhow::anyhow!("{e}"))?;
    Ok((graph, HashMap::new()))
}

fn build_lateral_graph(
    in_c: usize,
    out_c: usize,
    h: usize,
    w: usize,
    weight: &[f32],
    bias: &[f32],
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    let f = DType::F32;
    let mut hir = HirModule::new("sam2_fpn_lateral");
    let mut params = HashMap::new();
    let mut g = HirMut::new(&mut hir);

    let stage = g.input("stage", Shape::new(&[1, h, w, in_c], f));
    let x = bhwc_to_nchw(&mut g, stage, 1, h, w, in_c);
    let wt = param(&mut g, &mut params, "w", weight, &[out_c, in_c, 1, 1]);
    let bt = param(&mut g, &mut params, "b", bias, &[out_c]);
    let y = conv2d_bias(&mut g, x, wt, bt, 1, out_c, 1, 1, [1, 1], [0, 0], h, w);

    hir.set_outputs(vec![y]);
    let graph = Graph::from_hir(hir).map_err(|e| anyhow::anyhow!("{e}"))?;
    Ok((graph, params))
}

fn param(
    g: &mut HirMut<'_>,
    params: &mut HashMap<String, Vec<f32>>,
    name: &str,
    data: &[f32],
    shape: &[usize],
) -> HirNodeId {
    let id = g.param(name, Shape::new(shape, DType::F32));
    params.insert(name.to_string(), data.to_vec());
    id
}
