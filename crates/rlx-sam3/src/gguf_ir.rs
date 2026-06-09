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

//! Shared HIR helpers for SAM3 packed GGUF (`Op::DequantMatMul`).

use crate::packed_gguf::packed_linear;
use anyhow::{Result, ensure};
use rlx_flow::GgufPackedLinear;
use rlx_flow::GgufPackedParams;
use rlx_ir::hir::{HirGraphExt, HirMut, HirNodeId};
use rlx_ir::{DType, Op, Shape};
use std::collections::HashMap;

pub fn gguf_weight_param(
    g: &mut HirMut<'_>,
    typed: &mut Vec<(String, Vec<u8>, DType)>,
    cache: &mut HashMap<String, HirNodeId>,
    ir_name: &str,
    p: &GgufPackedLinear,
) -> HirNodeId {
    if let Some(&id) = cache.get(ir_name) {
        return id;
    }
    let id = g.param(ir_name, Shape::new(&[p.w_q.len()], DType::U8));
    typed.push((ir_name.to_string(), p.w_q.clone(), DType::U8));
    cache.insert(ir_name.to_string(), id);
    id
}

pub fn linear_gguf_matmul(
    g: &mut HirMut<'_>,
    typed: &mut Vec<(String, Vec<u8>, DType)>,
    cache: &mut HashMap<String, HirNodeId>,
    ir_stem: &str,
    p: &GgufPackedLinear,
    input: HirNodeId,
    in_dim: usize,
    out_dim: usize,
) -> Result<HirNodeId> {
    ensure!(
        p.in_dim == in_dim && p.out_dim == out_dim,
        "packed linear {ir_stem}: shape {}x{} vs {in_dim}x{out_dim}",
        p.in_dim,
        p.out_dim
    );
    let w_name = format!("{ir_stem}.w");
    let w_id = gguf_weight_param(g, typed, cache, &w_name, p);
    let cur = g.shape(input);
    let mut dims: Vec<usize> = cur.dims().iter().map(|d| d.unwrap_static()).collect();
    *dims.last_mut().unwrap() = out_dim;
    let out_shape = Shape::new(&dims, DType::F32);
    Ok(g.add_node(
        Op::DequantMatMul { scheme: p.scheme },
        vec![input, w_id],
        out_shape,
    ))
}

pub fn add_f32_bias(
    g: &mut HirMut<'_>,
    params: &mut HashMap<String, Vec<f32>>,
    name: &str,
    input: HirNodeId,
    bias: &[f32],
) -> HirNodeId {
    if bias.iter().all(|&v| v == 0.0) {
        return input;
    }
    let out_dim = bias.len();
    let b_id = add_param_f32(g, params, name, bias, &[out_dim]);
    g.add(input, b_id)
}

pub fn add_param_f32(
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

pub fn linear_gguf_bias(
    g: &mut HirMut<'_>,
    params: &mut HashMap<String, Vec<f32>>,
    typed: &mut Vec<(String, Vec<u8>, DType)>,
    cache: &mut HashMap<String, HirNodeId>,
    ir_stem: &str,
    p: &GgufPackedLinear,
    input: HirNodeId,
    bias: &[f32],
    in_dim: usize,
    out_dim: usize,
) -> Result<HirNodeId> {
    let y = linear_gguf_matmul(g, typed, cache, ir_stem, p, input, in_dim, out_dim)?;
    Ok(add_f32_bias(g, params, &format!("{ir_stem}.b"), y, bias))
}

/// Lookup packed GGUF linear for a checkpoint `*.weight` key.
pub fn packed_linear_for_key<'a>(
    gguf_packed: Option<&'a GgufPackedParams>,
    key: &str,
) -> Option<&'a GgufPackedLinear> {
    gguf_packed.and_then(|m| packed_linear(m, key))
}
