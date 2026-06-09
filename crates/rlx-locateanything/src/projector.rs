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

//! MLP projector (`mlp1`) — MoonViT merged patches → Qwen hidden size.

use crate::config::LocateAnythingConfig;
use anyhow::Result;
use rlx_core::flow_util::WeightMapSource;
use rlx_core::weight_map::WeightMap;
use rlx_flow::{BuiltModel, CompileProfile, ModelFlow};
use rlx_ir::hir::HirMut;
use rlx_ir::{DType, HirGraphExt, Shape};

pub struct ProjectorBuilt {
    pub model: BuiltModel,
    pub in_dim: usize,
    pub out_dim: usize,
}

pub fn build_projector_built(
    cfg: &LocateAnythingConfig,
    weights: &mut WeightMap,
    batch: usize,
    seq: usize,
) -> Result<ProjectorBuilt> {
    let in_dim = cfg.projector_input_dim();
    let out_dim = cfg.text_config.hidden_size;
    let f = DType::F32;
    let eps = 1e-5f32;

    let flow = ModelFlow::new("locateanything_projector")
        .with_profile(CompileProfile::encoder())
        .input("vision", Shape::new(&[batch, seq, in_dim], f))
        .plugin_named("locateanything.mlp1", move |emit, hidden| {
            let v = hidden.ok_or_else(|| anyhow::anyhow!("projector needs vision"))?;
            let ln_w = emit.load_param("mlp1.0.weight", false)?;
            let ln_b = emit.load_param("mlp1.0.bias", false)?;
            let fc1_w = emit.load_param("mlp1.1.weight", true)?;
            let fc1_b = emit.load_param("mlp1.1.bias", false)?;
            let fc2_w = emit.load_param("mlp1.3.weight", true)?;
            let fc2_b = emit.load_param("mlp1.3.bias", false)?;
            let mut gb = HirMut::new(emit.hir());
            let normed = gb.ln(v.hir_id(), ln_w, ln_b, eps);
            let fc1_mm = gb.mm(normed, fc1_w);
            let fc1 = gb.add(fc1_mm, fc1_b);
            let act = gb.gelu(fc1);
            let fc2_mm = gb.mm(act, fc2_w);
            let out = gb.add(fc2_mm, fc2_b);
            Ok(Some(
                emit.wrap(out, Shape::new(&[batch, seq, out_dim], DType::F32)),
            ))
        })
        .output("lm_embeds");

    Ok(ProjectorBuilt {
        model: flow.build_with(&mut WeightMapSource(weights), None)?,
        in_dim,
        out_dim,
    })
}
