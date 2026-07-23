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

//! Shared graph compilation for the detector and recognizer stages.
//!
//! Both build an rlx-ir graph + host params and compile it with the `encoder`
//! profile; this centralises the compile-and-attach dance so the runtimes only
//! deal with caching + running.

use rlx_core::flow_bridge::compile_options_for_profile;
use rlx_core::flow_util::attach_built_params;
use rlx_flow::CompileProfile;
use rlx_runtime::{CompiledGraph, Device, Session};
use std::collections::HashMap;

/// Compile `graph` for the `encoder` profile on `device` and attach `params`.
/// `skip_fusion` disables the conv+bias+act fusion pass (`OCR2_NO_FUSION`).
pub fn compile_encoder(
    graph: rlx_ir::Graph,
    params: HashMap<String, Vec<f32>>,
    device: Device,
    skip_fusion: bool,
) -> CompiledGraph {
    let mut profile = CompileProfile::encoder();
    if skip_fusion {
        profile.fusion.skip = true;
    }
    let opts = compile_options_for_profile(&profile, device);
    let mut compiled = Session::new(device).compile_with(graph, &opts);
    attach_built_params(&mut compiled, params, &[]);
    compiled
}
