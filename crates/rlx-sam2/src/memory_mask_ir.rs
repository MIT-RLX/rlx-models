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

//! SAM2 memory-encoder IR subgraphs (`MaskDownSampler`, prefix fuse, `Fuser`).

use super::memory_encoder::{Sam2CXBlockWeights, Sam2FuserWeights, Sam2MaskDownSamplerWeights};
use anyhow::Result;
use rlx_core::vision_ops_ir::{conv2d_bias, conv2d_bias_groups, layer_norm2d_nchw, nchw_shape};
use rlx_flow::CompileProfile;
use rlx_ir::hir::{HirModule, HirMut, HirNodeId};
use rlx_ir::op::Op;
use rlx_ir::{DType, Graph, HirGraphExt, Shape};
use rlx_runtime::{CompiledGraph, Device};
use std::collections::HashMap;

const LN_EPS: f32 = 1e-6;

pub struct Sam2MemoryMaskDownCompiled {
    graph: CompiledGraph,
    pub embed_dim: usize,
    pub in_h: usize,
    pub in_w: usize,
    pub out_h: usize,
    pub out_w: usize,
}

/// Fused mask down + `pix_feat_proj` + add → fuser input.
pub struct Sam2MemoryPrefixCompiled {
    graph: CompiledGraph,
    pub in_dim: usize,
    pub mask_in_h: usize,
    pub mask_in_w: usize,
    pub feat_h: usize,
    pub feat_w: usize,
}

impl Sam2MemoryPrefixCompiled {
    pub fn compile(
        mask_ds: &Sam2MaskDownSamplerWeights,
        in_dim: usize,
        mask_in_h: usize,
        mask_in_w: usize,
        feat_h: usize,
        feat_w: usize,
        pix_w: &[f32],
        pix_b: &[f32],
        device: Device,
    ) -> Result<Self> {
        Self::compile_with_profile(
            mask_ds,
            in_dim,
            mask_in_h,
            mask_in_w,
            feat_h,
            feat_w,
            pix_w,
            pix_b,
            device,
            &CompileProfile::sam_encoder(),
        )
    }

    pub fn compile_with_profile(
        mask_ds: &Sam2MaskDownSamplerWeights,
        in_dim: usize,
        mask_in_h: usize,
        mask_in_w: usize,
        feat_h: usize,
        feat_w: usize,
        pix_w: &[f32],
        pix_b: &[f32],
        device: Device,
        profile: &CompileProfile,
    ) -> Result<Self> {
        let (graph, params) = build_prefix_graph(
            mask_ds, in_dim, mask_in_h, mask_in_w, feat_h, feat_w, pix_w, pix_b,
        )?;
        let mut compiled =
            rlx_core::flow_bridge::compile_graph_with_profile(device, graph, profile)?;
        for (name, data) in &params {
            compiled.set_param(name, data);
        }
        Ok(Self {
            graph: compiled,
            in_dim,
            mask_in_h,
            mask_in_w,
            feat_h,
            feat_w,
        })
    }

    /// Post-sigmoid mask `[1, H, W]` flat + `pix_feat` NCHW `[in_dim, feat_h, feat_w]`.
    pub fn run(&mut self, mask: &[f32], pix_feat: &[f32]) -> Result<Vec<f32>> {
        anyhow::ensure!(
            mask.len() == self.mask_in_h * self.mask_in_w,
            "prefix mask len {} ≠ {}",
            mask.len(),
            self.mask_in_h * self.mask_in_w
        );
        anyhow::ensure!(
            pix_feat.len() == self.in_dim * self.feat_h * self.feat_w,
            "prefix pix_feat len {} ≠ {}",
            pix_feat.len(),
            self.in_dim * self.feat_h * self.feat_w
        );
        let outs = self.graph.run(&[("mask", mask), ("pix", pix_feat)]);
        Ok(outs.into_iter().next().expect("memory prefix output"))
    }
}

impl Sam2MemoryMaskDownCompiled {
    pub fn compile(
        w: &Sam2MaskDownSamplerWeights,
        in_h: usize,
        in_w: usize,
        device: Device,
    ) -> Result<Self> {
        Self::compile_with_profile(w, in_h, in_w, device, &CompileProfile::sam_encoder())
    }

    pub fn compile_with_profile(
        w: &Sam2MaskDownSamplerWeights,
        in_h: usize,
        in_w: usize,
        device: Device,
        profile: &CompileProfile,
    ) -> Result<Self> {
        let (graph, params, out_h, out_w) = build_mask_downsampler_graph(w, in_h, in_w)?;
        let mut compiled =
            rlx_core::flow_bridge::compile_graph_with_profile(device, graph, profile)?;
        for (name, data) in &params {
            compiled.set_param(name, data);
        }
        Ok(Self {
            graph: compiled,
            embed_dim: w.embed_dim,
            in_h,
            in_w,
            out_h,
            out_w,
        })
    }

    /// Flat `[1, H, W]` mask (post-sigmoid).
    pub fn run(&mut self, mask: &[f32]) -> Result<Vec<f32>> {
        let expected = self.in_h * self.in_w;
        anyhow::ensure!(
            mask.len() == expected,
            "mask len {} ≠ {expected} (1×{}×{})",
            mask.len(),
            self.in_h,
            self.in_w
        );
        let outs = self.graph.run(&[("mask", mask)]);
        Ok(outs.into_iter().next().expect("memory mask_down output"))
    }
}

#[allow(clippy::type_complexity)]
fn build_mask_downsampler_graph(
    w: &Sam2MaskDownSamplerWeights,
    in_h: usize,
    in_w: usize,
) -> Result<(Graph, HashMap<String, Vec<f32>>, usize, usize)> {
    let f = DType::F32;
    let mut hir = HirModule::new("sam2_memory_mask_down");
    let mut params = HashMap::new();
    let mut g = HirMut::new(&mut hir);

    let x = g.input("mask", Shape::new(&[1, 1, in_h, in_w], f));
    let (out, out_h, out_w) = append_mask_downsampler(&mut g, &mut params, x, w, in_h, in_w, "")?;

    hir.set_outputs(vec![out]);
    let graph = Graph::from_hir(hir).map_err(|e| anyhow::anyhow!("{e}"))?;
    Ok((graph, params, out_h, out_w))
}

fn build_prefix_graph(
    mask_ds: &Sam2MaskDownSamplerWeights,
    in_dim: usize,
    mask_in_h: usize,
    mask_in_w: usize,
    feat_h: usize,
    feat_w: usize,
    pix_w: &[f32],
    pix_b: &[f32],
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    let f = DType::F32;
    let mut hir = HirModule::new("sam2_memory_prefix");
    let mut params = HashMap::new();
    let mut g = HirMut::new(&mut hir);

    let mask = g.input("mask", Shape::new(&[1, 1, mask_in_h, mask_in_w], f));
    let (m_down, down_h, down_w) = append_mask_downsampler(
        &mut g,
        &mut params,
        mask,
        mask_ds,
        mask_in_h,
        mask_in_w,
        "md_",
    )?;
    anyhow::ensure!(
        down_h == feat_h && down_w == feat_w,
        "mask down {down_h}×{down_w} ≠ pix {feat_h}×{feat_w}"
    );

    let pix = g.input("pix", nchw_shape(1, in_dim, feat_h, feat_w, f));
    let pp_w = param(&mut g, &mut params, "pp_w", pix_w, &[in_dim, in_dim, 1, 1]);
    let pp_b = param(&mut g, &mut params, "pp_b", pix_b, &[in_dim]);
    let pix_y = conv2d_bias(
        &mut g,
        pix,
        pp_w,
        pp_b,
        1,
        in_dim,
        1,
        1,
        [1, 1],
        [0, 0],
        feat_h,
        feat_w,
    );
    let out = g.add(pix_y, m_down);

    hir.set_outputs(vec![out]);
    let graph = Graph::from_hir(hir).map_err(|e| anyhow::anyhow!("{e}"))?;
    Ok((graph, params))
}

/// Append MaskDownSampler ops; `pfx` prefixes parameter names.
fn append_mask_downsampler(
    g: &mut HirMut<'_>,
    params: &mut HashMap<String, Vec<f32>>,
    mut x: HirNodeId,
    w: &Sam2MaskDownSamplerWeights,
    in_h: usize,
    in_w: usize,
    pfx: &str,
) -> Result<(HirNodeId, usize, usize)> {
    let mut cur_h = in_h;
    let mut cur_w = in_w;

    for (li, level) in w.levels.iter().enumerate() {
        let out_h = (cur_h + 2 * w.padding - w.kernel) / w.stride + 1;
        let out_w = (cur_w + 2 * w.padding - w.kernel) / w.stride + 1;
        let k = w.kernel;
        let cw = param(
            g,
            params,
            &format!("{pfx}conv{li}_w"),
            &level.conv_w,
            &[level.out_c, level.in_c, k, k],
        );
        let cb = param(
            g,
            params,
            &format!("{pfx}conv{li}_b"),
            &level.conv_b,
            &[level.out_c],
        );
        x = conv2d_bias(
            g,
            x,
            cw,
            cb,
            1,
            level.out_c,
            k,
            k,
            [w.stride, w.stride],
            [w.padding, w.padding],
            out_h,
            out_w,
        );
        let ln_g = param(
            g,
            params,
            &format!("{pfx}ln{li}_g"),
            &level.ln_g,
            &[level.out_c],
        );
        let ln_b = param(
            g,
            params,
            &format!("{pfx}ln{li}_b"),
            &level.ln_b,
            &[level.out_c],
        );
        x = layer_norm2d_nchw(g, x, ln_g, ln_b, LN_EPS);
        x = g.gelu(x);
        cur_h = out_h;
        cur_w = out_w;
    }

    let last_c = w.levels.last().map(|l| l.out_c).unwrap_or(1);
    let fw = param(
        g,
        params,
        &format!("{pfx}final_w"),
        &w.final_conv_w,
        &[w.embed_dim, last_c, 1, 1],
    );
    let fb = param(
        g,
        params,
        &format!("{pfx}final_b"),
        &w.final_conv_b,
        &[w.embed_dim],
    );
    let out = conv2d_bias(
        g,
        x,
        fw,
        fb,
        1,
        w.embed_dim,
        1,
        1,
        [1, 1],
        [0, 0],
        cur_h,
        cur_w,
    );
    Ok((out, cur_h, cur_w))
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

/// Compiled 1×1 conv on NCHW `[C, H, W]` flat.
pub struct Sam2MemoryConv1x1Compiled {
    graph: CompiledGraph,
    in_c: usize,
    pub out_c: usize,
    pub h: usize,
    pub w: usize,
}

impl Sam2MemoryConv1x1Compiled {
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
        let (graph, params) = build_conv1x1_graph(in_c, out_c, h, w, weight, bias)?;
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

    pub fn run(&mut self, x: &[f32]) -> Result<Vec<f32>> {
        let expected = self.in_c * self.h * self.w;
        anyhow::ensure!(
            x.len() == expected,
            "conv1x1 input len {} ≠ {} ({}×{}×{})",
            x.len(),
            expected,
            self.in_c,
            self.h,
            self.w
        );
        let outs = self.graph.run(&[("x", x)]);
        Ok(outs.into_iter().next().expect("conv1x1 output"))
    }
}

fn build_conv1x1_graph(
    in_c: usize,
    out_c: usize,
    h: usize,
    w: usize,
    weight: &[f32],
    bias: &[f32],
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    let f = DType::F32;
    let mut hir = HirModule::new("sam2_conv1x1");
    let mut params = HashMap::new();
    let mut g = HirMut::new(&mut hir);

    let x = g.input("x", nchw_shape(1, in_c, h, w, f));
    let wt = param(&mut g, &mut params, "w", weight, &[out_c, in_c, 1, 1]);
    let bt = param(&mut g, &mut params, "b", bias, &[out_c]);
    let y = conv2d_bias(&mut g, x, wt, bt, 1, out_c, 1, 1, [1, 1], [0, 0], h, w);

    hir.set_outputs(vec![y]);
    let graph = Graph::from_hir(hir).map_err(|e| anyhow::anyhow!("{e}"))?;
    Ok((graph, params))
}

/// ConvNeXt-style `Fuser` (optional input 1×1 + `CXBlock` stack).
pub struct Sam2MemoryFuserCompiled {
    graph: CompiledGraph,
    pub dim: usize,
    pub h: usize,
    pub w: usize,
}

impl Sam2MemoryFuserCompiled {
    pub fn compile(w: &Sam2FuserWeights, h: usize, ww: usize, device: Device) -> Result<Self> {
        Self::compile_with_profile(w, h, ww, device, &CompileProfile::sam_encoder())
    }

    pub fn compile_with_profile(
        w: &Sam2FuserWeights,
        h: usize,
        ww: usize,
        device: Device,
        profile: &CompileProfile,
    ) -> Result<Self> {
        let (graph, params) = build_fuser_graph(w, h, ww)?;
        let mut compiled =
            rlx_core::flow_bridge::compile_graph_with_profile(device, graph, profile)?;
        for (name, data) in &params {
            compiled.set_param(name, data);
        }
        Ok(Self {
            graph: compiled,
            dim: w.dim,
            h,
            w: ww,
        })
    }

    pub fn run(&mut self, x: &[f32]) -> Result<Vec<f32>> {
        let expected = self.dim * self.h * self.w;
        anyhow::ensure!(
            x.len() == expected,
            "fuser input len {} ≠ {expected}",
            x.len()
        );
        let outs = self.graph.run(&[("x", x)]);
        Ok(outs.into_iter().next().expect("fuser output"))
    }
}

fn build_fuser_graph(
    w: &Sam2FuserWeights,
    h: usize,
    ww: usize,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    let f = DType::F32;
    let dim = w.dim;
    let mut hir = HirModule::new("sam2_memory_fuser");
    let mut params = HashMap::new();
    let mut g = HirMut::new(&mut hir);

    let mut x = g.input("x", nchw_shape(1, dim, h, ww, f));

    if let (Some(pw), Some(pb)) = (&w.input_proj_w, &w.input_proj_b) {
        let wt = param(&mut g, &mut params, "input_proj_w", pw, &[dim, dim, 1, 1]);
        let bt = param(&mut g, &mut params, "input_proj_b", pb, &[dim]);
        x = conv2d_bias(&mut g, x, wt, bt, 1, dim, 1, 1, [1, 1], [0, 0], h, ww);
    }

    for (li, layer) in w.layers.iter().enumerate() {
        x = cx_block_hir(&mut g, &mut params, x, layer, li, h, ww)?;
    }

    hir.set_outputs(vec![x]);
    let graph = Graph::from_hir(hir).map_err(|e| anyhow::anyhow!("{e}"))?;
    Ok((graph, params))
}

fn cx_block_hir(
    g: &mut HirMut<'_>,
    params: &mut HashMap<String, Vec<f32>>,
    x: HirNodeId,
    w: &Sam2CXBlockWeights,
    li: usize,
    h: usize,
    ww: usize,
) -> Result<HirNodeId> {
    let dim = w.dim;
    let k = w.kernel;
    let p = w.padding;
    let residual = x;

    let dw_w = param(
        g,
        params,
        &format!("l{li}_dw_w"),
        &w.dw_conv_w,
        &[dim, 1, k, k],
    );
    let dw_b = param(g, params, &format!("l{li}_dw_b"), &w.dw_conv_b, &[dim]);
    let mut y = conv2d_bias_groups(g, x, dw_w, dw_b, 1, dim, k, k, [1, 1], [p, p], dim, h, ww);

    let ln_g = param(g, params, &format!("l{li}_ln_g"), &w.ln_g, &[dim]);
    let ln_b = param(g, params, &format!("l{li}_ln_b"), &w.ln_b, &[dim]);
    y = layer_norm2d_nchw(g, y, ln_g, ln_b, LN_EPS);

    let pw1_w = param(
        g,
        params,
        &format!("l{li}_pw1_w"),
        &w.pw1_w,
        &[4 * dim, dim, 1, 1],
    );
    let pw1_b = param(g, params, &format!("l{li}_pw1_b"), &w.pw1_b, &[4 * dim]);
    y = conv2d_bias(g, y, pw1_w, pw1_b, 1, 4 * dim, 1, 1, [1, 1], [0, 0], h, ww);
    y = g.gelu(y);

    let pw2_w = param(
        g,
        params,
        &format!("l{li}_pw2_w"),
        &w.pw2_w,
        &[dim, 4 * dim, 1, 1],
    );
    let pw2_b = param(g, params, &format!("l{li}_pw2_b"), &w.pw2_b, &[dim]);
    y = conv2d_bias(g, y, pw2_w, pw2_b, 1, dim, 1, 1, [1, 1], [0, 0], h, ww);

    if let Some(gamma) = &w.gamma {
        let gparam = param(g, params, &format!("l{li}_gamma"), gamma, &[dim]);
        let out_shape = g.shape(y).clone();
        let g4 = g.reshape_(gparam, vec![1, dim as i64, 1, 1]);
        let scaled = g.add_node(
            Op::Expand {
                target_shape: vec![1, dim as i64, h as i64, ww as i64],
            },
            vec![g4],
            out_shape.clone(),
        );
        y = g.mul(y, scaled);
    }

    Ok(g.add(residual, y))
}
