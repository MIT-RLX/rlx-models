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

//! SAM3 detection-neck branch via IR (`ConvTranspose2d`, `Conv`, `Pool`).

use super::neck::Sam3NeckBranch;
use anyhow::Result;
use rlx_core::vision_ops_ir::{conv_transpose2d_stride2_k2_bias, conv2d_bias};
use rlx_flow::CompileProfile;
use rlx_ir::hir::{HirModule, HirMut, HirNodeId};
use rlx_ir::op::ReduceOp;
use rlx_ir::{DType, Graph, HirGraphExt, Op, Shape};
use rlx_runtime::{CompiledGraph, Device};
use std::collections::HashMap;

type BranchGraphParts = (Graph, HashMap<String, Vec<f32>>, usize, usize);

pub struct Sam3NeckBranchCompiled {
    graph: CompiledGraph,
    pub out_h: usize,
    pub out_w: usize,
}

impl Sam3NeckBranchCompiled {
    pub fn compile(
        branch: &Sam3NeckBranch,
        in_c: usize,
        h: usize,
        w: usize,
        device: Device,
    ) -> Result<Self> {
        Self::compile_with_profile(branch, in_c, h, w, device, &CompileProfile::sam3())
    }

    pub fn compile_with_profile(
        branch: &Sam3NeckBranch,
        in_c: usize,
        h: usize,
        w: usize,
        device: Device,
        profile: &CompileProfile,
    ) -> Result<Self> {
        let (graph, params, out_h, out_w) = build_branch_graph(branch, in_c, h, w)?;
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

    pub fn run(&mut self, x_nchw: &[f32], in_c: usize, h: usize, w: usize) -> Result<Vec<f32>> {
        anyhow::ensure!(x_nchw.len() == in_c * h * w);
        let outs = self.graph.run(&[("x", x_nchw)]);
        Ok(outs.into_iter().next().expect("sam3 neck branch output"))
    }
}

pub fn build_branch_graph(
    branch: &Sam3NeckBranch,
    in_c: usize,
    mut h: usize,
    mut w: usize,
) -> Result<BranchGraphParts> {
    let det = super::config::SAM3_DET_DIM;
    let f = DType::F32;
    let mut hir = HirModule::new("sam3_neck_branch");
    let mut params = HashMap::new();
    let mut g = HirMut::new(&mut hir);

    let x = g.input("x", Shape::new(&[1, in_c, h, w], f));
    let mut cur = x;

    if (branch.scale - 4.0).abs() < 1e-6 {
        let dw0 = branch.dconv0_w.as_ref().unwrap();
        let db0 = branch.dconv0_b.as_ref().unwrap();
        let w0 = p(
            &mut g,
            &mut params,
            "dconv0_w",
            dw0.clone(),
            &[in_c, 512, 2, 2],
        );
        let b0 = p(&mut g, &mut params, "dconv0_b", db0.clone(), &[512]);
        cur = conv_transpose2d_stride2_k2_bias(&mut g, cur, w0, b0, 1, 512, h, w);
        h *= 2;
        w *= 2;
        cur = g.gelu(cur);
        let dw1 = branch.dconv1_w.as_ref().unwrap();
        let db1 = branch.dconv1_b.as_ref().unwrap();
        let w1 = p(
            &mut g,
            &mut params,
            "dconv1_w",
            dw1.clone(),
            &[512, 256, 2, 2],
        );
        let b1 = p(&mut g, &mut params, "dconv1_b", db1.clone(), &[256]);
        cur = conv_transpose2d_stride2_k2_bias(&mut g, cur, w1, b1, 1, 256, h, w);
        h *= 2;
        w *= 2;
    } else if (branch.scale - 2.0).abs() < 1e-6 {
        let dw = branch.dconv0_w.as_ref().unwrap();
        let db = branch.dconv0_b.as_ref().unwrap();
        let wt = p(
            &mut g,
            &mut params,
            "dconv_w",
            dw.clone(),
            &[in_c, 512, 2, 2],
        );
        let bt = p(&mut g, &mut params, "dconv_b", db.clone(), &[512]);
        cur = conv_transpose2d_stride2_k2_bias(&mut g, cur, wt, bt, 1, 512, h, w);
        h *= 2;
        w *= 2;
    } else if (branch.scale - 0.5).abs() < 1e-6 {
        let out_h = h / 2;
        let out_w = w / 2;
        let pool_shape = Shape::new(&[1, in_c, out_h, out_w], f);
        cur = g.add_node(
            Op::Pool {
                kernel_size: vec![2, 2],
                stride: vec![2, 2],
                padding: vec![0, 0],
                kind: ReduceOp::Max,
            },
            vec![cur],
            pool_shape,
        );
        h = out_h;
        w = out_w;
    }

    let c1_w = p(
        &mut g,
        &mut params,
        "c1x1_w",
        branch.c1x1_w.clone(),
        &[det, branch.c1x1_in, 1, 1],
    );
    let c1_b = p(&mut g, &mut params, "c1x1_b", branch.c1x1_b.clone(), &[det]);
    cur = conv2d_bias(&mut g, cur, c1_w, c1_b, 1, det, 1, 1, [1, 1], [0, 0], h, w);

    let c3_w = p(
        &mut g,
        &mut params,
        "c3x3_w",
        branch.c3x3_w.clone(),
        &[det, det, 3, 3],
    );
    let c3_b = p(&mut g, &mut params, "c3x3_b", branch.c3x3_b.clone(), &[det]);
    cur = conv2d_bias(&mut g, cur, c3_w, c3_b, 1, det, 3, 3, [1, 1], [1, 1], h, w);

    hir.set_outputs(vec![cur]);
    let graph = Graph::from_hir(hir).map_err(|e| anyhow::anyhow!("{e}"))?;
    Ok((graph, params, h, w))
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
