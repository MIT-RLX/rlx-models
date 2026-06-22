// RLX — versatile ML compiler + runtime. GPLv3.
//! Gemma 4 E2B mobile QAT helpers — compile profile for packed QAT prefill.

use std::collections::HashMap;

use anyhow::Result;
use rlx_core::flow_util::compile_graph_profile;
use rlx_flow::CompileProfile;
use rlx_ir::Graph;
use rlx_runtime::{CompiledGraph, Device};

/// Pick the execution device for E2B QAT inference (pass-through).
pub fn resolve_e2b_device(requested: Device) -> Device {
    requested
}

/// Optional Metal prefill env overrides. Disabled by default for throughput;
/// set `RLX_GEMMA4_E2B_METAL_PRECISE=1` to force MPSGraph off + precise F32.
pub fn with_e2b_metal_guard<R, F>(device: Device, f: F) -> R
where
    F: FnOnce() -> R,
{
    if device != Device::Metal || !rlx_ir::env::flag("RLX_GEMMA4_E2B_METAL_PRECISE") {
        return f();
    }
    let save_mps = rlx_ir::env::var("RLX_DISABLE_MPSGRAPH");
    let save_precise = rlx_ir::env::var("RLX_METAL_PRECISE");
    let save_fusion = rlx_ir::env::var("RLX_METAL_NO_FUSION");
    rlx_ir::env::set("RLX_DISABLE_MPSGRAPH", "1");
    rlx_ir::env::set("RLX_METAL_PRECISE", "1");
    rlx_ir::env::set("RLX_METAL_NO_FUSION", "1");
    let out = f();
    match save_mps {
        Some(v) => rlx_ir::env::set("RLX_DISABLE_MPSGRAPH", v),
        None => rlx_ir::env::unset("RLX_DISABLE_MPSGRAPH"),
    }
    match save_precise {
        Some(v) => rlx_ir::env::set("RLX_METAL_PRECISE", v),
        None => rlx_ir::env::unset("RLX_METAL_PRECISE"),
    }
    match save_fusion {
        Some(v) => rlx_ir::env::set("RLX_METAL_NO_FUSION", v),
        None => rlx_ir::env::unset("RLX_METAL_NO_FUSION"),
    }
    out
}

/// Compile E2B prefill with fusion disabled (packed QAT graphs) and optional Metal guard.
pub fn compile_e2b_prefill(
    device: Device,
    graph: Graph,
    params: HashMap<String, Vec<f32>>,
) -> Result<CompiledGraph> {
    let exec = resolve_e2b_device(device);
    let mut profile = CompileProfile::gemma_prefill();
    profile.fusion.skip = true;
    with_e2b_metal_guard(exec, || {
        compile_graph_profile(exec, graph, params, &profile)
    })
}
