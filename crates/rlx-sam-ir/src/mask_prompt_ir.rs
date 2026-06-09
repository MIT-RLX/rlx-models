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

//! Shared prompt-encoder mask downscale stack (Conv2d → LN2d → GELU ×2 + 1×1).

use anyhow::Result;
use rlx_core::vision_ops_ir::{conv2d_bias, layer_norm2d_nchw};
use rlx_flow::CompileProfile;
use rlx_ir::hir::{HirModule, HirMut, HirNodeId};
use rlx_ir::{DType, Graph, HirGraphExt, Shape};
use rlx_runtime::{CompiledGraph, Device};
use std::collections::HashMap;

/// Weights for the three-stage mask downscaling stack (SAM1 + SAM2).
pub struct MaskDownscaleWeights<'a> {
    pub mask_in_chans: usize,
    pub embed_dim: usize,
    pub mask_conv1_w: &'a [f32],
    pub mask_conv1_b: &'a [f32],
    pub mask_ln1_g: &'a [f32],
    pub mask_ln1_b: &'a [f32],
    pub mask_conv2_w: &'a [f32],
    pub mask_conv2_b: &'a [f32],
    pub mask_ln2_g: &'a [f32],
    pub mask_ln2_b: &'a [f32],
    pub mask_conv3_w: &'a [f32],
    pub mask_conv3_b: &'a [f32],
}

pub struct PromptMaskCompiled {
    graph: CompiledGraph,
    #[allow(dead_code)]
    embed_dim: usize,
}

impl PromptMaskCompiled {
    pub fn compile(
        w: MaskDownscaleWeights<'_>,
        in_h: usize,
        in_w: usize,
        device: Device,
    ) -> Result<Self> {
        Self::compile_with_profile(w, in_h, in_w, device, &CompileProfile::sam_encoder())
    }

    pub fn compile_with_profile(
        w: MaskDownscaleWeights<'_>,
        in_h: usize,
        in_w: usize,
        device: Device,
        profile: &CompileProfile,
    ) -> Result<Self> {
        let embed_dim = w.embed_dim;
        let (graph, params) = build_prompt_mask_graph(w, in_h, in_w)?;
        let mut compiled =
            rlx_core::flow_bridge::compile_graph_with_profile(device, graph, profile)?;
        for (name, data) in &params {
            compiled.set_param(name, data);
        }
        Ok(Self {
            graph: compiled,
            embed_dim,
        })
    }

    /// Flat NCHW mask `[1, 1, H, W]` (H·W elements).
    pub fn run(&mut self, mask: &[f32], in_h: usize, in_w: usize) -> Result<Vec<f32>> {
        let expected = in_h * in_w;
        anyhow::ensure!(
            mask.len() == expected,
            "mask len {} ≠ {expected} (1×{in_h}×{in_w})",
            mask.len()
        );
        let outs = self.graph.run(&[("mask", mask)]);
        Ok(outs.into_iter().next().expect("prompt mask output"))
    }
}

pub fn build_prompt_mask_graph(
    w: MaskDownscaleWeights<'_>,
    in_h: usize,
    in_w: usize,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    let q = w.mask_in_chans / 4;
    let m = w.mask_in_chans;
    let e = w.embed_dim;
    let f = DType::F32;
    let eps = 1e-6f32;

    let mut hir = HirModule::new("mask_prompt_downscale");
    let mut params = HashMap::new();
    let mut g = HirMut::new(&mut hir);

    let mask = g.input("mask", Shape::new(&[1, 1, in_h, in_w], f));

    let w0 = param(
        &mut g,
        &mut params,
        "mask_conv1_w",
        w.mask_conv1_w,
        &[q, 1, 2, 2],
    );
    let b0 = param(&mut g, &mut params, "mask_conv1_b", w.mask_conv1_b, &[q]);
    let h1 = in_h / 2;
    let w1 = in_w / 2;
    let mut x = conv2d_bias(&mut g, mask, w0, b0, 1, q, 2, 2, [2, 2], [0, 0], h1, w1);

    let ln1_g = param(&mut g, &mut params, "mask_ln1_g", w.mask_ln1_g, &[q]);
    let ln1_b = param(&mut g, &mut params, "mask_ln1_b", w.mask_ln1_b, &[q]);
    x = layer_norm2d_nchw(&mut g, x, ln1_g, ln1_b, eps);
    x = g.gelu(x);

    let w1w = param(
        &mut g,
        &mut params,
        "mask_conv2_w",
        w.mask_conv2_w,
        &[m, q, 2, 2],
    );
    let b1 = param(&mut g, &mut params, "mask_conv2_b", w.mask_conv2_b, &[m]);
    let h2 = h1 / 2;
    let w2 = w1 / 2;
    x = conv2d_bias(&mut g, x, w1w, b1, 1, m, 2, 2, [2, 2], [0, 0], h2, w2);

    let ln2_g = param(&mut g, &mut params, "mask_ln2_g", w.mask_ln2_g, &[m]);
    let ln2_b = param(&mut g, &mut params, "mask_ln2_b", w.mask_ln2_b, &[m]);
    x = layer_norm2d_nchw(&mut g, x, ln2_g, ln2_b, eps);
    x = g.gelu(x);

    let w2w = param(
        &mut g,
        &mut params,
        "mask_conv3_w",
        w.mask_conv3_w,
        &[e, m, 1, 1],
    );
    let b2 = param(&mut g, &mut params, "mask_conv3_b", w.mask_conv3_b, &[e]);
    let out = conv2d_bias(&mut g, x, w2w, b2, 1, e, 1, 1, [1, 1], [0, 0], h2, w2);

    hir.set_outputs(vec![out]);
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
