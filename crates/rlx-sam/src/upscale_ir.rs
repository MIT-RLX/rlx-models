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

//! SAM v1 mask-decoder upscaling subgraph (ConvTranspose2d + LN2d + GELU).

use super::config::SAM_EMBED_HW;
use super::mask_decoder::MaskDecoderWeights;
use anyhow::Result;
use rlx_core::vision_ops_ir::{conv_transpose2d_stride2_k2_bias, layer_norm2d_nchw};
use rlx_flow::CompileProfile;
use rlx_ir::hir::{HirModule, HirMut, HirNodeId};
use rlx_ir::{DType, Graph, HirGraphExt, Shape};
use rlx_runtime::{CompiledGraph, Device};
use std::collections::HashMap;

/// Compiled upscale stack: `src_nchw` → `up2` after two transposed convs.
pub struct SamMaskUpscaleCompiled {
    graph: CompiledGraph,
    e: usize,
    hw: usize,
}

impl SamMaskUpscaleCompiled {
    pub fn compile(w: &MaskDecoderWeights, device: Device) -> Result<Self> {
        Self::compile_with_profile(w, device, &CompileProfile::sam_encoder())
    }

    pub fn compile_with_profile(
        w: &MaskDecoderWeights,
        device: Device,
        profile: &CompileProfile,
    ) -> Result<Self> {
        let (graph, params) = build_mask_upscale_graph(w)?;
        let mut compiled =
            rlx_core::flow_bridge::compile_graph_with_profile(device, graph, profile)?;
        for (name, data) in &params {
            compiled.set_param(name, data);
        }
        Ok(Self {
            graph: compiled,
            e: w.transformer_dim,
            hw: SAM_EMBED_HW,
        })
    }

    /// `src_nchw` is `[E, hw, hw]` NCHW (same layout as mask decoder).
    /// Returns `up2` `[E/8, 4·hw, 4·hw]`.
    pub fn run(&mut self, src_nchw: &[f32]) -> Result<Vec<f32>> {
        let e = self.e;
        let hw = self.hw;
        anyhow::ensure!(
            src_nchw.len() == e * hw * hw,
            "src_nchw len {} ≠ E·hw·hw",
            src_nchw.len()
        );
        let outs = self.graph.run(&[("src", src_nchw)]);
        Ok(outs.into_iter().next().expect("upscale output"))
    }
}

pub fn build_mask_upscale_hir(
    w: &MaskDecoderWeights,
) -> Result<(HirModule, HashMap<String, Vec<f32>>)> {
    let e = w.transformer_dim;
    let hw = SAM_EMBED_HW;
    let q4 = e / 4;
    let q8 = e / 8;
    let eps = 1e-6f32;
    let f = DType::F32;

    let mut hir = HirModule::new("sam_mask_upscale");
    let mut params = HashMap::new();
    let mut g = HirMut::new(&mut hir);

    let src = g.input("src", Shape::new(&[1, e, hw, hw], f));

    let up1_w = param(
        &mut g,
        &mut params,
        "upscale_conv1_w",
        w.upscale_conv1_w.clone(),
        &[e, q4, 2, 2],
    );
    let up1_b = param(
        &mut g,
        &mut params,
        "upscale_conv1_b",
        w.upscale_conv1_b.clone(),
        &[q4],
    );
    let mut up1 = conv_transpose2d_stride2_k2_bias(&mut g, src, up1_w, up1_b, 1, q4, hw, hw);

    let ln_g = param(
        &mut g,
        &mut params,
        "upscale_ln_g",
        w.upscale_ln_g.clone(),
        &[q4],
    );
    let ln_b = param(
        &mut g,
        &mut params,
        "upscale_ln_b",
        w.upscale_ln_b.clone(),
        &[q4],
    );
    up1 = layer_norm2d_nchw(&mut g, up1, ln_g, ln_b, eps);
    up1 = g.gelu(up1);

    let h1 = hw * 2;
    let up2_w = param(
        &mut g,
        &mut params,
        "upscale_conv2_w",
        w.upscale_conv2_w.clone(),
        &[q4, q8, 2, 2],
    );
    let up2_b = param(
        &mut g,
        &mut params,
        "upscale_conv2_b",
        w.upscale_conv2_b.clone(),
        &[q8],
    );
    let up2 = conv_transpose2d_stride2_k2_bias(&mut g, up1, up2_w, up2_b, 1, q8, h1, h1);
    let up2 = g.gelu(up2);

    hir.set_outputs(vec![up2]);
    Ok((hir, params))
}

pub fn build_mask_upscale_graph(
    w: &MaskDecoderWeights,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    let (hir, params) = build_mask_upscale_hir(w)?;
    Graph::from_hir(hir)
        .map_err(|e| anyhow::anyhow!("{e}"))
        .map(|g| (g, params))
}

fn param(
    g: &mut HirMut<'_>,
    params: &mut HashMap<String, Vec<f32>>,
    name: &str,
    data: Vec<f32>,
    shape: &[usize],
) -> HirNodeId {
    let id = g.param(name, Shape::new(shape, DType::F32));
    params.insert(name.to_string(), data);
    id
}
