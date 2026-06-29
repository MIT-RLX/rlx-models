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

//! Compile RLX training backward graphs on CPU / GPU backends.

use anyhow::{Result, bail};
use rlx_ir::Graph;
use rlx_runtime::{CompileOptions, CompiledGraph, Device, Precision, Session};
use std::cell::Cell;
use std::collections::HashMap;
use std::panic::{AssertUnwindSafe, catch_unwind};

thread_local! {
    static METAL_GUARD_DEPTH: Cell<usize> = const { Cell::new(0) };
}

/// Env knob (also set by the global `--precision` CLI flag) selecting the float
/// dtype used when compiling FFT graphs. Default `f32`; `f16` lets the butterfly
/// / mask / denoiser graph run on fp16 silicon (Apple Neural Engine via CoreML,
/// Metal/MLX half), at the cost of accuracy — train a butterfly for the target
/// precision to recover it.
pub const PRECISION_ENV: &str = "RLX_FFT_PRECISION";

/// Parse a precision string (`f32`/`f16`/`bf16` and common aliases).
pub fn parse_precision(s: &str) -> Result<Precision> {
    match s.trim().to_ascii_lowercase().as_str() {
        "" | "f32" | "fp32" | "float32" | "single" => Ok(Precision::F32),
        "f16" | "fp16" | "float16" | "half" => Ok(Precision::F16),
        "bf16" | "bfloat16" | "brain" => Ok(Precision::BF16),
        other => bail!("unknown precision '{other}' (try: f32, f16, bf16)"),
    }
}

/// Set the compile precision for subsequent `try_compile_graph*` calls.
pub fn set_compile_precision(p: Precision) {
    rlx_ir::env::set(PRECISION_ENV, p.to_string());
}

/// Precision applied to FFT compiles, resolved from [`PRECISION_ENV`] (default F32).
pub fn compile_precision() -> Precision {
    rlx_ir::env::var(PRECISION_ENV)
        .as_deref()
        .and_then(|s| parse_precision(s).ok())
        .unwrap_or(Precision::F32)
}

/// Metal: disable MPSGraph for graphs with many narrow/concat ops (butterfly FFT).
/// Matches whisper / qwen3-tts / packed GGUF compile guards.
pub fn metal_compile_guard<R, F>(device: Device, f: F) -> R
where
    F: FnOnce() -> R,
{
    if device == Device::Metal {
        METAL_GUARD_DEPTH.with(|depth| {
            if depth.get() == 0 {
                rlx_ir::env::set("RLX_DISABLE_MPSGRAPH", "1");
            }
            depth.set(depth.get() + 1);
        });
        let out = f();
        METAL_GUARD_DEPTH.with(|depth| {
            let next = depth.get().saturating_sub(1);
            depth.set(next);
            if next == 0 {
                rlx_ir::env::unset("RLX_DISABLE_MPSGRAPH");
            }
        });
        out
    } else {
        f()
    }
}

pub fn compile_train_backward(
    device: Device,
    backward_graph: Graph,
    label: &str,
) -> Result<(Device, CompiledGraph)> {
    match try_compile(device, backward_graph.clone()) {
        Ok(c) => Ok((device, c)),
        Err(e) if device != Device::Cpu => {
            eprintln!("[{label}] {device:?} backward failed ({e}) — CPU fallback");
            Ok((Device::Cpu, try_compile(Device::Cpu, backward_graph)?))
        }
        Err(e) => Err(e),
    }
}

/// Compile without aborting the process if the backend panics during lowering.
pub fn try_compile_graph(device: Device, graph: Graph) -> Result<CompiledGraph> {
    try_compile_graph_with_params(device, graph, None)
}

/// Compile with fixed param bindings baked into constants before fusion/DCE.
pub fn try_compile_graph_with_params(
    device: Device,
    graph: Graph,
    param_bindings: Option<HashMap<String, Vec<f32>>>,
) -> Result<CompiledGraph> {
    catch_unwind(AssertUnwindSafe(|| {
        metal_compile_guard(device, || {
            let session = Session::new(device);
            let mut opts = CompileOptions::new().precision(compile_precision());
            if let Some(bindings) = param_bindings {
                opts = opts.param_bindings(bindings);
            }
            session.compile_with(graph, &opts)
        })
    }))
    .map_err(|_| anyhow::anyhow!("compile on {device:?} failed (see log above)"))
}

fn try_compile(device: Device, graph: Graph) -> Result<CompiledGraph> {
    try_compile_graph(device, graph)
}

/// Try `device`, then CPU when compilation fails on a non-CPU backend.
pub fn compile_graph_with_cpu_fallback(
    device: Device,
    graph: Graph,
    label: &str,
) -> Result<(Device, CompiledGraph)> {
    compile_graph_with_cpu_fallback_params(device, graph, label, None)
}

pub fn compile_graph_with_cpu_fallback_params(
    device: Device,
    graph: Graph,
    label: &str,
    param_bindings: Option<HashMap<String, Vec<f32>>>,
) -> Result<(Device, CompiledGraph)> {
    match try_compile_graph_with_params(device, graph.clone(), param_bindings.clone()) {
        Ok(c) => Ok((device, c)),
        Err(e) if device != Device::Cpu => {
            eprintln!("[{label}] {device:?} compile failed ({e}) — CPU fallback");
            Ok((
                Device::Cpu,
                try_compile_graph_with_params(Device::Cpu, graph, param_bindings)?,
            ))
        }
        Err(e) => Err(e),
    }
}
