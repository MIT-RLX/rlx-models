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

//! FLUX.2 HIR graph builder (compile minimal + incremental lowering).
//!
//! Model builders target [`HirModule`] and lower via [`HirModule::lower_to_mir`].
//! Minimal graph only — see [`super::hir_builder`] for the full transformer HIR.

use super::config::Flux2Config;
use super::weights::{Flux2Weights, LinearWeights};
use anyhow::Result;
use rlx_ir::{DType, FusionPolicy, Graph, HirModule, Shape};
use std::collections::HashMap;

/// Param tensors keyed by name for [`rlx_runtime::CompiledGraph::set_param`].
pub type Flux2GraphParams = HashMap<String, Vec<f32>>;

/// Build a compile-minimal HIR module: `x_embedder(hidden)` → `proj_out`.
///
/// Inputs: `hidden` `[batch, img_seq, in_channels]`.
/// Output: `[batch, img_seq, proj_out_dim]`.
pub fn build_flux2_minimal_hir(
    cfg: &Flux2Config,
    weights: &Flux2Weights,
    batch: usize,
    img_seq: usize,
) -> Result<(HirModule, Flux2GraphParams)> {
    let mut hir = HirModule::new("flux2_minimal").with_fusion_policy(FusionPolicy::Direct);
    let mut params = Flux2GraphParams::new();
    let f = DType::F32;

    let hidden_in = hir.input("hidden", Shape::new(&[batch, img_seq, cfg.in_channels], f));
    let embedded = linear_hir(
        &mut hir,
        &mut params,
        hidden_in,
        &weights.x_embedder,
        "x_embedder",
        Shape::new(&[batch, img_seq, weights.x_embedder.out_dim], f),
    )?;
    let out = linear_hir(
        &mut hir,
        &mut params,
        embedded,
        &weights.proj_out,
        "proj_out",
        Shape::new(&[batch, img_seq, cfg.proj_out_dim()], f),
    )?;
    hir.outputs = vec![out];
    Ok((hir, params))
}

/// Lower minimal HIR to legacy [`Graph`] (MIR inner) for `Session::compile`.
pub fn build_flux2_minimal_graph(
    cfg: &Flux2Config,
    weights: &Flux2Weights,
    batch: usize,
    img_seq: usize,
) -> Result<(Graph, Flux2GraphParams)> {
    rlx_core::flow_util::graph_from_built(crate::flow::build_flux2_minimal_built(
        cfg, weights, batch, img_seq,
    )?)
}

/// Compile minimal HIR on CPU (HIR → MIR → LIR).
pub fn compile_flux2_minimal(
    cfg: &Flux2Config,
    weights: &Flux2Weights,
    batch: usize,
    img_seq: usize,
) -> Result<(rlx_runtime::CompiledGraph, Flux2GraphParams)> {
    use rlx_runtime::Device;

    let (hir, params) = build_flux2_minimal_hir(cfg, weights, batch, img_seq)?;
    let profile = crate::compile_util::flux2_compile_profile();
    let mut compiled = rlx_core::flow_bridge::compile_hir_with_profile(Device::Cpu, hir, &profile)?;
    for (name, data) in &params {
        compiled.set_param(name, data);
    }
    Ok((compiled, params))
}

pub(crate) fn linear_hir(
    hir: &mut HirModule,
    params: &mut Flux2GraphParams,
    x: rlx_ir::HirNodeId,
    lw: &LinearWeights,
    name: &str,
    out_shape: Shape,
) -> Result<rlx_ir::HirNodeId> {
    let w = hir.param(
        format!("{name}.weight"),
        Shape::new(&[lw.in_dim, lw.out_dim], DType::F32),
    );
    params.insert(format!("{name}.weight"), lw.w_t.clone());
    let bias = if lw.bias.iter().all(|&v| v == 0.0) {
        None
    } else {
        let b = hir.param(
            format!("{name}.bias"),
            Shape::new(&[lw.out_dim], DType::F32),
        );
        params.insert(format!("{name}.bias"), lw.bias.clone());
        Some(b)
    };
    Ok(hir.linear(x, w, bias, None, out_shape))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Flux2Config, extract_flux2_weights, prepare_weight_map, synthetic_weights};

    #[test]
    fn minimal_hir_lowers_to_mir() {
        let cfg = Flux2Config::tiny();
        let wm = synthetic_weights(&cfg);
        let w = extract_flux2_weights(prepare_weight_map(wm), &cfg).unwrap();
        let (hir, _) = build_flux2_minimal_hir(&cfg, &w, 1, 4).unwrap();
        assert_eq!(hir.outputs.len(), 1);
        let mir = hir.lower_to_mir().expect("lower");
        assert_eq!(mir.outputs().len(), 1);
    }

    #[test]
    fn minimal_compiles_on_cpu() {
        let cfg = Flux2Config::tiny();
        let wm = synthetic_weights(&cfg);
        let w = extract_flux2_weights(prepare_weight_map(wm), &cfg).unwrap();
        let (mut compiled, _) = compile_flux2_minimal(&cfg, &w, 1, 4).unwrap();
        let hidden = vec![0.0f32; cfg.in_channels * 4];
        let out = compiled.run(&[("hidden", hidden.as_slice())]);
        assert_eq!(out[0].len(), 4 * cfg.proj_out_dim());
    }
}
