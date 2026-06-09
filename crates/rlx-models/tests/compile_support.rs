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

//! Shared tier-1 compile helpers for integration tests.
//!
//! Prefer [`compile_qwen35_prefill`] / [`compile_qwen35_decode`] (and the
//! SAM/Qwen3/encoder variants) over [`compile_legacy`], which uses default
//! [`CompileOptions`] only. Production runners build options via
//! [`rlx_models::flow_bridge::compile_options_from_profile`].

#![allow(dead_code)]

use rlx_flow::CompileProfile;
use rlx_ir::Graph;
use rlx_runtime::{CompileOptions, CompiledGraph, Device, Session};
use std::collections::HashMap;

pub fn attach_params(compiled: &mut CompiledGraph, params: &HashMap<String, Vec<f32>>) {
    for (name, data) in params {
        compiled.set_param(name, data.as_slice());
    }
}

pub fn compile_legacy(
    device: Device,
    graph: Graph,
    params: HashMap<String, Vec<f32>>,
) -> CompiledGraph {
    let mut compiled = Session::new(device).compile_with(graph, &CompileOptions::new());
    attach_params(&mut compiled, &params);
    compiled
}

pub fn compile_with_options(
    device: Device,
    graph: Graph,
    params: HashMap<String, Vec<f32>>,
    opts: &CompileOptions,
) -> CompiledGraph {
    let mut compiled = Session::new(device).compile_with(graph, opts);
    attach_params(&mut compiled, &params);
    compiled
}

pub fn compile_with_profile(
    device: Device,
    graph: Graph,
    params: HashMap<String, Vec<f32>>,
    profile: &CompileProfile,
) -> CompiledGraph {
    let mut compiled = rlx_models::flow_bridge::compile_graph_with_profile(device, graph, profile)
        .expect("compile_graph_with_profile");
    attach_params(&mut compiled, &params);
    compiled
}

pub fn compile_sam(
    device: Device,
    graph: Graph,
    params: HashMap<String, Vec<f32>>,
) -> CompiledGraph {
    compile_with_profile(device, graph, params, &CompileProfile::sam_encoder())
}

pub fn compile_encoder(
    device: Device,
    graph: Graph,
    params: HashMap<String, Vec<f32>>,
) -> CompiledGraph {
    compile_with_profile(device, graph, params, &CompileProfile::encoder())
}

pub fn compile_qwen35_prefill(
    device: Device,
    graph: Graph,
    params: HashMap<String, Vec<f32>>,
) -> CompiledGraph {
    compile_with_profile(device, graph, params, &CompileProfile::qwen35_prefill())
}

pub fn compile_qwen35_decode(
    device: Device,
    graph: Graph,
    params: HashMap<String, Vec<f32>>,
) -> CompiledGraph {
    compile_with_profile(device, graph, params, &CompileProfile::qwen35_decode())
}

pub fn compile_qwen3_prefill(
    device: Device,
    graph: Graph,
    params: HashMap<String, Vec<f32>>,
) -> CompiledGraph {
    compile_with_profile(device, graph, params, &CompileProfile::qwen3_prefill())
}

pub fn compile_llama32_prefill(
    device: Device,
    graph: Graph,
    params: HashMap<String, Vec<f32>>,
) -> CompiledGraph {
    compile_with_profile(device, graph, params, &CompileProfile::llama32_prefill())
}

pub fn compile_llama32_decode(
    device: Device,
    graph: Graph,
    params: HashMap<String, Vec<f32>>,
) -> CompiledGraph {
    compile_with_profile(device, graph, params, &CompileProfile::llama32_decode())
}

pub fn compile_llada2(
    device: Device,
    graph: Graph,
    params: HashMap<String, Vec<f32>>,
) -> CompiledGraph {
    compile_with_profile(device, graph, params, &CompileProfile::llada2_diffusion())
}
