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

//! ReLU MLP stacks (`Linear` + ReLU × (n-1) + optional final sigmoid on host).

use anyhow::Result;
use rlx_flow::CompileProfile;
use rlx_ir::hir::{HirModule, HirMut, HirNodeId};
use rlx_ir::op::Op;
use rlx_ir::{DType, Graph, HirGraphExt, Shape};
use rlx_runtime::{CompiledGraph, Device};
use std::collections::HashMap;

type MlpGraphParts = (Graph, HashMap<String, Vec<f32>>, usize, usize);

/// One `Linear` + optional ReLU (all but last layer).
#[derive(Clone)]
pub struct MlpLayerSpec {
    pub w: Vec<f32>,
    pub b: Vec<f32>,
    pub in_d: usize,
    pub out_d: usize,
}

/// ReLU MLP matching SAM mask-decoder `mlp_forward` (PyTorch weight layout `[out, in]`).
pub struct MlpReluCompiled {
    graph: CompiledGraph,
    rows: usize,
    in_d: usize,
    #[allow(dead_code)]
    out_d: usize,
    apply_host_sigmoid: bool,
}

impl MlpReluCompiled {
    pub fn compile(
        layers: &[MlpLayerSpec],
        sigmoid_output: bool,
        rows: usize,
        device: Device,
    ) -> Result<Self> {
        Self::compile_with_profile(
            layers,
            sigmoid_output,
            rows,
            device,
            &CompileProfile::sam_encoder(),
        )
    }

    pub fn compile_with_profile(
        layers: &[MlpLayerSpec],
        sigmoid_output: bool,
        rows: usize,
        device: Device,
        profile: &CompileProfile,
    ) -> Result<Self> {
        let (graph, params, in_d, out_d) = build_mlp_graph(layers, rows)?;
        let mut compiled =
            rlx_core::flow_bridge::compile_graph_with_profile(device, graph, profile)?;
        for (name, data) in &params {
            compiled.set_param(name, data);
        }
        Ok(Self {
            graph: compiled,
            rows,
            in_d,
            out_d,
            apply_host_sigmoid: sigmoid_output,
        })
    }

    pub fn compiled_rows(&self) -> usize {
        self.rows
    }

    /// Input row-major `[rows, in_d]` flat; `rows` must match compile-time batch.
    pub fn run(&mut self, x: &[f32], rows: usize) -> Result<Vec<f32>> {
        anyhow::ensure!(
            rows == self.rows,
            "mlp rows {rows} ≠ compiled rows {}",
            self.rows
        );
        anyhow::ensure!(
            x.len() == rows * self.in_d,
            "mlp input len {} ≠ rows·in_d ({}·{})",
            x.len(),
            rows,
            self.in_d
        );
        let outs = self.graph.run(&[("x", x)]);
        let mut y = outs.into_iter().next().expect("mlp output");
        if self.apply_host_sigmoid {
            for v in y.iter_mut() {
                *v = 1.0 / (1.0 + (-*v).exp());
            }
        }
        Ok(y)
    }
}

fn build_mlp_graph(layers: &[MlpLayerSpec], rows: usize) -> Result<MlpGraphParts> {
    anyhow::ensure!(!layers.is_empty(), "mlp needs at least one layer");
    let in_d = layers[0].in_d;
    let out_d = layers.last().unwrap().out_d;
    let f = DType::F32;
    let mut hir = HirModule::new("mlp_relu");
    let mut params = HashMap::new();
    let mut g = HirMut::new(&mut hir);

    let mut x = g.input("x", Shape::new(&[rows, in_d], f));

    let n = layers.len();
    for (i, layer) in layers.iter().enumerate() {
        let w_id = param_linear(
            &mut g,
            &mut params,
            &format!("w{i}"),
            &layer.w,
            layer.in_d,
            layer.out_d,
        );
        let b_id = param(
            &mut g,
            &mut params,
            &format!("b{i}"),
            &layer.b,
            &[layer.out_d],
        );
        x = linear_layer(&mut g, x, w_id, b_id, rows, layer.in_d, layer.out_d);
        if i + 1 < n {
            x = g.relu(x);
        }
    }

    hir.set_outputs(vec![x]);
    let graph = Graph::from_hir(hir).map_err(|e| anyhow::anyhow!("{e}"))?;
    Ok((graph, params, in_d, out_d))
}

fn linear_layer(
    g: &mut HirMut<'_>,
    x: HirNodeId,
    w: HirNodeId,
    b: HirNodeId,
    rows: usize,
    _in_d: usize,
    out_d: usize,
) -> HirNodeId {
    let y = g.mm(x, w);
    add_bias_rows(g, y, b, rows, out_d)
}

fn add_bias_rows(
    g: &mut HirMut<'_>,
    y: HirNodeId,
    bias: HirNodeId,
    rows: usize,
    out_d: usize,
) -> HirNodeId {
    let out_shape = g.shape(y).clone();
    let b2 = g.reshape_(bias, vec![1, out_d as i64]);
    let expanded = g.add_node(
        Op::Expand {
            target_shape: vec![rows as i64, out_d as i64],
        },
        vec![b2],
        out_shape.clone(),
    );
    g.add(y, expanded)
}

fn param_linear(
    g: &mut HirMut<'_>,
    params: &mut HashMap<String, Vec<f32>>,
    name: &str,
    w_out_in: &[f32],
    in_d: usize,
    out_d: usize,
) -> HirNodeId {
    let mut w_t = vec![0f32; in_d * out_d];
    for o in 0..out_d {
        for k in 0..in_d {
            w_t[k * out_d + o] = w_out_in[o * in_d + k];
        }
    }
    param(g, params, name, &w_t, &[in_d, out_d])
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
