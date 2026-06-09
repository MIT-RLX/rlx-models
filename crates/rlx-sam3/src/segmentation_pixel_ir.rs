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

//! SAM3 pixel-decoder fusion steps + instance/semantic 1×1 heads (IR).

use crate::gguf_ir::{linear_gguf_bias, packed_linear_for_key};
use anyhow::Result;
use rlx_core::vision_ops_ir::{conv2d_bias, nchw_shape};
use rlx_flow::{CompileProfile, GgufPackedParams};
use rlx_ir::hir::{HirModule, HirMut, HirNodeId};
use rlx_ir::{DType, Graph, HirGraphExt, Shape};
use rlx_runtime::{CompiledGraph, Device};
use std::collections::HashMap;

const D_MODEL: usize = 256;
const GN_GROUPS: usize = 8;

type ConvGraphParts = (
    Graph,
    HashMap<String, Vec<f32>>,
    Vec<(String, Vec<u8>, DType)>,
);

/// One pixel-decoder layer: upsample `prev` 2×, add `curr`, conv3×3, GN, ReLU.
pub struct Sam3PixelDecoderStepCompiled {
    graph: CompiledGraph,
    pub out_h: usize,
    pub out_w: usize,
}

impl Sam3PixelDecoderStepCompiled {
    pub fn compile(
        prev_h: usize,
        prev_w: usize,
        out_h: usize,
        out_w: usize,
        conv_w: &[f32],
        conv_b: &[f32],
        gn_w: &[f32],
        gn_b: &[f32],
        device: Device,
    ) -> Result<Self> {
        Self::compile_with_profile(
            prev_h,
            prev_w,
            out_h,
            out_w,
            conv_w,
            conv_b,
            gn_w,
            gn_b,
            device,
            &CompileProfile::sam3(),
        )
    }

    pub fn compile_with_profile(
        prev_h: usize,
        prev_w: usize,
        out_h: usize,
        out_w: usize,
        conv_w: &[f32],
        conv_b: &[f32],
        gn_w: &[f32],
        gn_b: &[f32],
        device: Device,
        profile: &CompileProfile,
    ) -> Result<Self> {
        anyhow::ensure!(
            out_h == prev_h * 2 && out_w == prev_w * 2,
            "pixel_decoder step expects 2× upsample {prev_h}×{prev_w} → {out_h}×{out_w}"
        );
        let (graph, params) =
            build_pixel_step_graph(prev_h, prev_w, out_h, out_w, conv_w, conv_b, gn_w, gn_b)?;
        let mut compiled =
            rlx_core::flow_bridge::compile_graph_with_profile(device, graph, profile)?;
        for (name, data) in &params {
            compiled.set_param(name, data);
        }
        Ok(Self {
            graph: compiled,
            out_h,
            out_w,
        })
    }

    pub fn run(&mut self, prev: &[f32], curr: &[f32]) -> Result<Vec<f32>> {
        let n = D_MODEL * self.out_h * self.out_w;
        anyhow::ensure!(prev.len() == n / 4 && curr.len() == n);
        let outs = self.graph.run(&[("prev", prev), ("curr", curr)]);
        Ok(outs.into_iter().next().expect("pixel_decoder step output"))
    }
}

pub struct Sam3Conv1x1Compiled {
    graph: CompiledGraph,
    pub out_c: usize,
    pub h: usize,
    pub w: usize,
}

impl Sam3Conv1x1Compiled {
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
            &CompileProfile::sam3(),
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
        let (graph, params, typed) = build_conv1x1_graph(in_c, out_c, h, w, weight, bias)?;
        let mut compiled =
            rlx_core::flow_bridge::compile_graph_with_profile(device, graph, profile)?;
        for (name, data) in &params {
            compiled.set_param(name, data);
        }
        rlx_core::flow_util::attach_built_params(&mut compiled, params, &typed);
        Ok(Self {
            graph: compiled,
            out_c,
            h,
            w,
        })
    }

    /// F32 conv when materialized, or `DequantMatMul` when `gguf_key` is set.
    pub fn compile_with_gguf(
        in_c: usize,
        out_c: usize,
        h: usize,
        w: usize,
        weight: &[f32],
        bias: &[f32],
        gguf_key: Option<&str>,
        gguf_packed: &GgufPackedParams,
        device: Device,
        profile: &CompileProfile,
    ) -> Result<Self> {
        let (graph, params, typed) = if let (Some(key), Some(p)) = (
            gguf_key,
            gguf_key.and_then(|k| packed_linear_for_key(Some(gguf_packed), k)),
        ) {
            build_conv1x1_graph_gguf(in_c, out_c, h, w, p, bias, key)?
        } else {
            anyhow::ensure!(
                !weight.is_empty(),
                "conv1x1: missing F32 weights and GGUF key"
            );
            build_conv1x1_graph(in_c, out_c, h, w, weight, bias)?
        };
        let mut compiled =
            rlx_core::flow_bridge::compile_graph_with_profile(device, graph, profile)?;
        for (name, data) in &params {
            compiled.set_param(name, data);
        }
        rlx_core::flow_util::attach_built_params(&mut compiled, params, &typed);
        Ok(Self {
            graph: compiled,
            out_c,
            h,
            w,
        })
    }

    pub fn run(&mut self, x: &[f32]) -> Result<Vec<f32>> {
        anyhow::ensure!(x.len() == D_MODEL * self.h * self.w);
        let outs = self.graph.run(&[("x", x)]);
        Ok(outs.into_iter().next().expect("conv1x1 output"))
    }
}

fn build_pixel_step_graph(
    prev_h: usize,
    prev_w: usize,
    out_h: usize,
    out_w: usize,
    conv_w: &[f32],
    conv_b: &[f32],
    gn_w: &[f32],
    gn_b: &[f32],
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    let f = DType::F32;
    let mut hir = HirModule::new("sam3_pixel_decoder_step");
    let mut params = HashMap::new();
    let mut g = HirMut::new(&mut hir);

    let prev = g.input("prev", Shape::new(&[1, D_MODEL, prev_h, prev_w], f));
    let curr = g.input("curr", Shape::new(&[1, D_MODEL, out_h, out_w], f));

    let up = g.resize_nearest_2x(prev);
    let combined = g.add(curr, up);

    let cw = param_f32(
        &mut g,
        &mut params,
        "conv_w",
        conv_w,
        &[D_MODEL, D_MODEL, 3, 3],
    );
    let cb = param_f32(&mut g, &mut params, "conv_b", conv_b, &[D_MODEL]);
    let mut y = conv2d_bias(
        &mut g,
        combined,
        cw,
        cb,
        1,
        D_MODEL,
        3,
        3,
        [1, 1],
        [1, 1],
        out_h,
        out_w,
    );

    let gnw = param_f32(&mut g, &mut params, "gn_w", gn_w, &[D_MODEL]);
    let gnb = param_f32(&mut g, &mut params, "gn_b", gn_b, &[D_MODEL]);
    y = g.group_norm(y, gnw, gnb, GN_GROUPS, 1e-5);
    let out = g.relu(y);

    hir.set_outputs(vec![out]);
    let graph = Graph::from_hir(hir).map_err(|e| anyhow::anyhow!("{e}"))?;
    Ok((graph, params))
}

fn build_conv1x1_graph(
    in_c: usize,
    out_c: usize,
    h: usize,
    w: usize,
    weight: &[f32],
    bias: &[f32],
) -> Result<ConvGraphParts> {
    let f = DType::F32;
    let mut hir = HirModule::new("sam3_conv1x1");
    let mut params = HashMap::new();
    let mut g = HirMut::new(&mut hir);

    let x = g.input("x", nchw_shape(1, in_c, h, w, f));
    let wt = param_f32(&mut g, &mut params, "w", weight, &[out_c, in_c, 1, 1]);
    let bt = param_f32(&mut g, &mut params, "b", bias, &[out_c]);
    let y = conv2d_bias(&mut g, x, wt, bt, 1, out_c, 1, 1, [1, 1], [0, 0], h, w);

    hir.set_outputs(vec![y]);
    let graph = Graph::from_hir(hir).map_err(|e| anyhow::anyhow!("{e}"))?;
    Ok((graph, params, Vec::new()))
}

fn build_conv1x1_graph_gguf(
    in_c: usize,
    out_c: usize,
    h: usize,
    w: usize,
    p: &rlx_flow::GgufPackedLinear,
    bias: &[f32],
    gguf_key: &str,
) -> Result<ConvGraphParts> {
    let f = DType::F32;
    let mut hir = HirModule::new("sam3_conv1x1_gguf");
    let mut params = HashMap::new();
    let mut typed = Vec::new();
    let mut gguf_cache = HashMap::new();
    let mut g = HirMut::new(&mut hir);

    let x = g.input("x", nchw_shape(1, in_c, h, w, f));
    let spatial = (h * w) as i64;
    let flat = g.reshape_(x, vec![1, spatial, in_c as i64]);
    let stem = gguf_key.strip_suffix(".weight").unwrap_or(gguf_key);
    let y_flat = linear_gguf_bias(
        &mut g,
        &mut params,
        &mut typed,
        &mut gguf_cache,
        stem,
        p,
        flat,
        bias,
        in_c,
        out_c,
    )?;
    let y = g.reshape_(y_flat, vec![1, out_c as i64, h as i64, w as i64]);

    hir.set_outputs(vec![y]);
    let graph = Graph::from_hir(hir).map_err(|e| anyhow::anyhow!("{e}"))?;
    Ok((graph, params, typed))
}

fn param_f32(
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

/// Compile pixel-decoder steps for SAM3 base (neck scales 4× / 2× on 72×72 trunk).
pub fn compile_pixel_decoder_steps(
    pixel_conv_w: &[Vec<f32>],
    pixel_conv_b: &[Vec<f32>],
    pixel_gn_w: &[Vec<f32>],
    pixel_gn_b: &[Vec<f32>],
    trunk_grid: usize,
    device: Device,
    profile: &CompileProfile,
) -> Result<Vec<Sam3PixelDecoderStepCompiled>> {
    // After scalp=1: FPN levels are 288×288, 144×144, 72×72 (fine→coarse indices 0,1,2).
    // pop finest 72; fuse 144 then 288.
    let g0 = trunk_grid;
    let g1 = trunk_grid * 2;
    let g2 = trunk_grid * 4;
    let steps = [(g0, g0, g1, g1, 0usize), (g1, g1, g2, g2, 1usize)];
    steps
        .iter()
        .map(|(ph, pw, oh, ow, i)| {
            Sam3PixelDecoderStepCompiled::compile_with_profile(
                *ph,
                *pw,
                *oh,
                *ow,
                &pixel_conv_w[*i],
                &pixel_conv_b[*i],
                &pixel_gn_w[*i],
                &pixel_gn_b[*i],
                device,
                profile,
            )
        })
        .collect()
}
