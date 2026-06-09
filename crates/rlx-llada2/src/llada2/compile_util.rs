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

// RLX — tier-1 compile helpers for LLaDA2.

use anyhow::Result;
use rlx_core::flow_bridge::compile_options_from_profile;
use rlx_flow::{BuiltModel, CompileProfile};
use rlx_ir::logical_kernel::KernelDispatchConfig;
use rlx_runtime::{CompiledGraph, Device, Session};

pub fn llada2_profile() -> CompileProfile {
    CompileProfile::llada2_diffusion()
}

pub fn compile_llada2_built(built: BuiltModel, device: Device) -> Result<CompiledGraph> {
    let profile = built.profile().clone();
    let (graph, params) = built.into_graph_parts()?;
    let opts = compile_options_from_profile(&profile, device, KernelDispatchConfig::default());
    let mut compiled = Session::new(device).compile_with(graph, &opts);
    for (name, data) in params {
        compiled.set_param(&name, &data);
    }
    Ok(compiled)
}
