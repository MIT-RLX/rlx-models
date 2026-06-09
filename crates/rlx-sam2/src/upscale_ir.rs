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

//! SAM2 mask-decoder upscaling (ConvTranspose2d + LN2d + optional high-res 1×1 fuse).

use super::mask_decoder::Sam2MaskDecoderWeights;
use anyhow::Result;
use rlx_core::vision_ops_ir::{conv_transpose2d_stride2_k2_bias, conv2d_bias, layer_norm2d_nchw};
use rlx_flow::CompileProfile;
use rlx_ir::hir::{HirModule, HirMut, HirNodeId};
use rlx_ir::{DType, Graph, HirGraphExt, Shape};
use rlx_runtime::{CompiledGraph, Device};
use std::collections::HashMap;

pub struct Sam2MaskUpscaleCompiled {
    graph: CompiledGraph,
    e: usize,
    use_high_res: bool,
}

impl Sam2MaskUpscaleCompiled {
    pub fn compile(w: &Sam2MaskDecoderWeights, grid: usize, device: Device) -> Result<Self> {
        Self::compile_with_profile(w, grid, device, &CompileProfile::sam_encoder())
    }

    pub fn compile_with_profile(
        w: &Sam2MaskDecoderWeights,
        grid: usize,
        device: Device,
        profile: &CompileProfile,
    ) -> Result<Self> {
        let (graph, params) = build_mask_upscale_graph(w, grid)?;
        let mut compiled =
            rlx_core::flow_bridge::compile_graph_with_profile(device, graph, profile)?;
        for (name, data) in &params {
            compiled.set_param(name, data);
        }
        Ok(Self {
            graph: compiled,
            e: w.transformer_dim,
            use_high_res: w.use_high_res_features,
        })
    }

    /// `src_nchw` `[E, g, g]`. When `use_high_res`, pass `feat_s1` `[E, 2g, 2g]`
    /// and `feat_s0` `[E, 4g, 4g]`; otherwise pass empty slices.
    pub fn run(
        &mut self,
        src_nchw: &[f32],
        feat_s1: &[f32],
        feat_s0: &[f32],
        grid: usize,
    ) -> Result<Vec<f32>> {
        let e = self.e;
        let g = grid;
        anyhow::ensure!(src_nchw.len() == e * g * g);
        let mut inputs = vec![("src", src_nchw)];
        let s1_buf;
        let s0_buf;
        if self.use_high_res {
            let h1 = g * 2;
            let h2 = g * 4;
            anyhow::ensure!(feat_s1.len() == e * h1 * h1 && feat_s0.len() == e * h2 * h2);
            s1_buf = feat_s1;
            s0_buf = feat_s0;
            inputs.push(("feat_s1", s1_buf));
            inputs.push(("feat_s0", s0_buf));
        }
        let outs = self
            .graph
            .run(&inputs.iter().map(|(n, d)| (*n, *d)).collect::<Vec<_>>());
        Ok(outs.into_iter().next().expect("sam2 upscale output"))
    }
}

pub fn build_mask_upscale_graph(
    w: &Sam2MaskDecoderWeights,
    grid: usize,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    let e = w.transformer_dim;
    let g = grid;
    let q4 = e / 4;
    let q8 = e / 8;
    let eps = 1e-6f32;
    let f = DType::F32;

    let mut hir = HirModule::new("sam2_mask_upscale");
    let mut params = HashMap::new();
    let mut hg = HirMut::new(&mut hir);

    let src = hg.input("src", Shape::new(&[1, e, g, g], f));

    let up1_w = p(
        &mut hg,
        &mut params,
        "upscale_conv1_w",
        w.upscale_conv1_w.clone(),
        &[e, q4, 2, 2],
    );
    let up1_b = p(
        &mut hg,
        &mut params,
        "upscale_conv1_b",
        w.upscale_conv1_b.clone(),
        &[q4],
    );
    let mut up1 = conv_transpose2d_stride2_k2_bias(&mut hg, src, up1_w, up1_b, 1, q4, g, g);

    if w.use_high_res_features {
        let h1 = g * 2;
        let feat_s1 = hg.input("feat_s1", Shape::new(&[1, e, h1, h1], f));
        let s1_w = p(
            &mut hg,
            &mut params,
            "conv_s1_w",
            w.conv_s1_w.clone().unwrap(),
            &[q4, e, 1, 1],
        );
        let s1_b = p(
            &mut hg,
            &mut params,
            "conv_s1_b",
            w.conv_s1_b.clone().unwrap(),
            &[q4],
        );
        let s1_proj = conv2d_bias(
            &mut hg,
            feat_s1,
            s1_w,
            s1_b,
            1,
            q4,
            1,
            1,
            [1, 1],
            [0, 0],
            h1,
            h1,
        );
        up1 = hg.add(up1, s1_proj);
    }

    let ln_g = p(
        &mut hg,
        &mut params,
        "upscale_ln_g",
        w.upscale_ln_g.clone(),
        &[q4],
    );
    let ln_b = p(
        &mut hg,
        &mut params,
        "upscale_ln_b",
        w.upscale_ln_b.clone(),
        &[q4],
    );
    up1 = layer_norm2d_nchw(&mut hg, up1, ln_g, ln_b, eps);
    up1 = hg.gelu(up1);

    let h1 = g * 2;
    let up2_w = p(
        &mut hg,
        &mut params,
        "upscale_conv2_w",
        w.upscale_conv2_w.clone(),
        &[q4, q8, 2, 2],
    );
    let up2_b = p(
        &mut hg,
        &mut params,
        "upscale_conv2_b",
        w.upscale_conv2_b.clone(),
        &[q8],
    );
    let mut up2 = conv_transpose2d_stride2_k2_bias(&mut hg, up1, up2_w, up2_b, 1, q8, h1, h1);

    if w.use_high_res_features {
        let h2 = g * 4;
        let feat_s0 = hg.input("feat_s0", Shape::new(&[1, e, h2, h2], f));
        let s0_w = p(
            &mut hg,
            &mut params,
            "conv_s0_w",
            w.conv_s0_w.clone().unwrap(),
            &[q8, e, 1, 1],
        );
        let s0_b = p(
            &mut hg,
            &mut params,
            "conv_s0_b",
            w.conv_s0_b.clone().unwrap(),
            &[q8],
        );
        let s0_proj = conv2d_bias(
            &mut hg,
            feat_s0,
            s0_w,
            s0_b,
            1,
            q8,
            1,
            1,
            [1, 1],
            [0, 0],
            h2,
            h2,
        );
        up2 = hg.add(up2, s0_proj);
    }

    let up2 = hg.gelu(up2);
    hir.set_outputs(vec![up2]);
    Graph::from_hir(hir)
        .map_err(|e| anyhow::anyhow!("{e}"))
        .map(|g| (g, params))
}

fn p(
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
