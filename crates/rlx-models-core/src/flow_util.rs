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

//! Shared helpers for tier-0 model flow migration.

use anyhow::Result;
use rlx_flow::{BuiltModel, CompileProfile, WeightSource};
use rlx_ir::{Graph, HirModule};
use rlx_runtime::compile_cache::{BucketedCompileCache, CompileCache};
use rlx_runtime::{CompileOptions, CompiledGraph, Device, Session};

use crate::weight_map::WeightMap;

/// Adapt in-memory [`WeightMap`] to [`WeightSource`].
pub struct WeightMapSource<'a>(pub &'a mut WeightMap);

impl WeightSource for WeightMapSource<'_> {
    fn take(&mut self, key: &str, transpose: bool) -> Result<(Vec<f32>, Vec<usize>)> {
        let (data, shape) = self.0.take(key)?;
        if !transpose {
            return Ok((data, shape));
        }
        if shape.len() != 2 {
            anyhow::bail!("transpose requires rank-2 weight: {key}");
        }
        let rows = shape[0];
        let cols = shape[1];
        let mut out = vec![0f32; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                out[c * rows + r] = data[r * cols + c];
            }
        }
        Ok((out, vec![cols, rows]))
    }

    fn has(&self, key: &str) -> bool {
        self.0.has(key)
    }
}

pub fn built_from_hir(
    hir: HirModule,
    params: std::collections::HashMap<String, Vec<f32>>,
) -> Result<BuiltModel> {
    BuiltModel::from_hir(hir, params)
}

pub fn built_from_graph(
    graph: Graph,
    params: std::collections::HashMap<String, Vec<f32>>,
) -> Result<BuiltModel> {
    BuiltModel::from_graph(graph, params)
}

pub fn built_from_hir_with_profile(
    hir: HirModule,
    params: std::collections::HashMap<String, Vec<f32>>,
    profile: CompileProfile,
) -> Result<BuiltModel> {
    let mut built = BuiltModel::from_hir(hir, params)?;
    built.profile = profile;
    Ok(built)
}

/// Build a flow and return `(Graph, params)` — preferred compile entry point.
pub fn graph_from_built(
    built: BuiltModel,
) -> Result<(Graph, std::collections::HashMap<String, Vec<f32>>)> {
    built.into_graph_parts()
}

/// Lower an existing HIR module through [`BuiltModel`] (utility for HIR-first builders).
pub fn graph_from_hir(
    hir: HirModule,
    params: std::collections::HashMap<String, Vec<f32>>,
) -> Result<(Graph, std::collections::HashMap<String, Vec<f32>>)> {
    graph_from_built(built_from_hir(hir, params)?)
}

/// Build via flow and lower to MIR graph + params.
pub fn build_graph<F>(
    mut build: F,
    weights: &mut WeightMap,
) -> Result<(Graph, std::collections::HashMap<String, Vec<f32>>)>
where
    F: FnMut(&mut WeightMapSource<'_>) -> Result<BuiltModel>,
{
    let built = build(&mut WeightMapSource(weights))?;
    graph_from_built(built)
}

/// Compile helper — build graph + params from a flow, then compile with a configured session.
pub fn compile_from_flow<F>(
    mut build: F,
    weights: &mut WeightMap,
    configure: impl FnOnce(Session) -> Session,
) -> Result<CompiledGraph>
where
    F: FnMut(&mut WeightMapSource<'_>) -> Result<BuiltModel>,
{
    let built = build(&mut WeightMapSource(weights))?;
    let profile = built.profile().clone();
    let typed = built.typed_params.clone();
    let (graph, params) = built.into_graph_parts()?;
    let options = crate::flow_bridge::compile_options_for_profile(&profile, Device::Cpu);
    let session = configure(Session::new(Device::Cpu));
    let mut compiled = session.compile_with(graph, &options);
    attach_built_params(&mut compiled, params, &typed);
    Ok(compiled)
}

/// Attach f32 and typed (U8 packed GGUF) params after compile.
pub fn attach_built_params(
    compiled: &mut CompiledGraph,
    params: std::collections::HashMap<String, Vec<f32>>,
    typed_params: &[(String, Vec<u8>, rlx_ir::DType)],
) {
    for (name, data) in params {
        compiled.set_param(&name, &data);
    }
    for (name, data, dtype) in typed_params {
        compiled.set_param_typed(name, data, *dtype);
    }
}

/// Compile a [`BuiltModel`] on the given device using its embedded profile.
pub fn compile_built(built: BuiltModel, device: Device) -> Result<CompiledGraph> {
    let profile = built.profile().clone();
    let typed = built.typed_params.clone();
    let (graph, params) = built.into_graph_parts()?;
    let options = crate::flow_bridge::compile_options_for_profile(&profile, device);
    let mut compiled = Session::new(device).compile_with(graph, &options);
    attach_built_params(&mut compiled, params, &typed);
    Ok(compiled)
}

/// Compile a [`BuiltModel`] on CPU with default options (embedding quick-check tests).
pub fn compile_built_cpu(built: BuiltModel) -> Result<CompiledGraph> {
    compile_built(built, Device::Cpu)
}

/// Unprofiled compile + params (layer probes; matches historical `Session::compile`).
pub fn compile_graph_legacy_with_params(
    device: Device,
    graph: Graph,
    params: std::collections::HashMap<String, Vec<f32>>,
) -> Result<CompiledGraph> {
    let mut compiled = crate::flow_bridge::compile_graph_legacy(device, graph)?;
    for (name, data) in params {
        compiled.set_param(&name, data.as_slice());
    }
    Ok(compiled)
}

/// Llama 3.2 prefill + params.
pub fn compile_graph_gemma_prefill_with_params(
    device: Device,
    graph: Graph,
    params: std::collections::HashMap<String, Vec<f32>>,
) -> Result<CompiledGraph> {
    compile_graph_profile(device, graph, params, &CompileProfile::gemma_prefill())
}

pub fn compile_graph_gemma_decode_with_params(
    device: Device,
    graph: Graph,
    params: std::collections::HashMap<String, Vec<f32>>,
) -> Result<CompiledGraph> {
    compile_graph_profile(device, graph, params, &CompileProfile::gemma_decode())
}

pub fn compile_graph_llama32_prefill_with_params(
    device: Device,
    graph: Graph,
    params: std::collections::HashMap<String, Vec<f32>>,
) -> Result<CompiledGraph> {
    compile_graph_profile(device, graph, params, &CompileProfile::llama32_prefill())
}

/// Llama 3.2 decode + params.
pub fn compile_graph_llama32_decode_with_params(
    device: Device,
    graph: Graph,
    params: std::collections::HashMap<String, Vec<f32>>,
) -> Result<CompiledGraph> {
    compile_graph_profile(device, graph, params, &CompileProfile::llama32_decode())
}

/// Legacy default compile options (plumbing tests with hand-built graphs).
pub fn compile_graph_default_with_params(
    device: Device,
    graph: Graph,
    params: std::collections::HashMap<String, Vec<f32>>,
) -> Result<CompiledGraph> {
    compile_graph_profile(device, graph, params, &CompileProfile::default())
}

/// Lower a graph with a tier-1 profile and attach params (tests / examples).
pub fn compile_graph_profile(
    device: Device,
    graph: Graph,
    params: std::collections::HashMap<String, Vec<f32>>,
    profile: &CompileProfile,
) -> Result<CompiledGraph> {
    let mut compiled = crate::flow_bridge::compile_graph_with_profile(device, graph, profile)?;
    for (name, data) in params {
        compiled.set_param(&name, data.as_slice());
    }
    Ok(compiled)
}

/// [`CompileProfile::encoder`] + params.
pub fn compile_graph_encoder_with_params(
    device: Device,
    graph: Graph,
    params: std::collections::HashMap<String, Vec<f32>>,
) -> Result<CompiledGraph> {
    compile_graph_profile(device, graph, params, &CompileProfile::encoder())
}

/// [`CompileProfile::sam_encoder`] + params.
pub fn compile_graph_sam_with_params(
    device: Device,
    graph: Graph,
    params: std::collections::HashMap<String, Vec<f32>>,
) -> Result<CompiledGraph> {
    compile_graph_profile(device, graph, params, &CompileProfile::sam_encoder())
}

/// [`CompileProfile::qwen3_prefill`] + params.
pub fn compile_graph_qwen3_prefill_with_params(
    device: Device,
    graph: Graph,
    params: std::collections::HashMap<String, Vec<f32>>,
) -> Result<CompiledGraph> {
    compile_graph_profile(device, graph, params, &CompileProfile::qwen3_prefill())
}

/// [`CompileProfile::qwen35_prefill`] + params.
pub fn compile_graph_qwen35_prefill_with_params(
    device: Device,
    graph: Graph,
    params: std::collections::HashMap<String, Vec<f32>>,
) -> Result<CompiledGraph> {
    compile_graph_profile(device, graph, params, &CompileProfile::qwen35_prefill())
}

/// [`CompileProfile::qwen35_decode`] + params.
pub fn compile_graph_qwen35_decode_with_params(
    device: Device,
    graph: Graph,
    params: std::collections::HashMap<String, Vec<f32>>,
) -> Result<CompiledGraph> {
    compile_graph_profile(device, graph, params, &CompileProfile::qwen35_decode())
}

/// Tier-1 profile + params (including graphs that export KV side outputs).
pub fn compile_graph_with_kv_export_params(
    device: Device,
    graph: Graph,
    params: std::collections::HashMap<String, Vec<f32>>,
    profile: &CompileProfile,
) -> Result<CompiledGraph> {
    use rlx_runtime::Session;
    let mut compiled = Session::new(device).compile_with(
        graph,
        &crate::flow_bridge::compile_options_for_profile(profile, device),
    );
    for (name, data) in params {
        compiled.set_param(&name, data.as_slice());
    }
    Ok(compiled)
}

/// Insert a [`BuiltModel`] into an LRU [`CompileCache`] (compile + params on first `key`).
#[allow(clippy::needless_lifetimes)]
pub fn compile_cache_ensure_built<'a>(
    cache: &'a mut CompileCache,
    key: u64,
    built: BuiltModel,
) -> Result<&'a mut CompiledGraph> {
    compile_cache_ensure_built_with_options(cache, key, built, &CompileOptions::new())
}

/// Like [`compile_cache_ensure_built`] but uses tier-1 [`CompileOptions`].
#[allow(clippy::needless_lifetimes)]
pub fn compile_cache_ensure_built_with_options<'a>(
    cache: &'a mut CompileCache,
    key: u64,
    built: BuiltModel,
    options: &CompileOptions,
) -> Result<&'a mut CompiledGraph> {
    if !cache.contains(key) {
        let (graph, params) = graph_from_built(built)?;
        let compiled = cache.get_or_compile_with_options(key, || graph, options);
        attach_built_params(compiled, params, &[]);
    }
    Ok(cache.get_or_compile_with_options(
        key,
        || panic!("compile_cache_ensure_built_with_options: missing entry for key {key}"),
        options,
    ))
}

/// Compile a decode bucket once and attach params (see [`BucketedCompileCache::ensure_graph_with_params`]).
pub fn bucket_cache_ensure_built<'a, F>(
    cache: &'a mut BucketedCompileCache,
    key: u64,
    build: F,
    options: &CompileOptions,
) -> Option<(u64, &'a mut CompiledGraph)>
where
    F: FnOnce(u64) -> Result<BuiltModel>,
{
    cache.ensure_graph_with_params(
        key,
        |upper| {
            let built = build(upper).expect("bucket_cache_ensure_built build failed");
            graph_from_built(built).expect("bucket_cache_ensure_built lower failed")
        },
        options,
    )
}
