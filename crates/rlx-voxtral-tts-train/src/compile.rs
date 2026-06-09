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

//! Session compile with per-graph device selection (GPU forward + CPU backward when needed).

use anyhow::{Result, bail};
use rlx_ir::Graph;
use rlx_runtime::{
    CompiledGraph, Device, Session, compile_output_cap, device_has_compile_output_cap,
};
use std::panic::{AssertUnwindSafe, catch_unwind};

use crate::backward_prep::prepare_backward_for_device;

pub struct TrainSession {
    pub forward_device: Device,
    pub backward_device: Device,
    pub forward: CompiledGraph,
    pub backward: CompiledGraph,
}

/// When set, do not auto-fallback to CPU backward for output-cap graphs (bench / debug).
pub fn native_backward_from_env() -> bool {
    for key in [
        "RLX_VOXTRAL_TTS_TRAIN_NATIVE_BACKWARD",
        "RLX_VOXTRAL_TTS_TRAIN_MLX_NATIVE_BACKWARD",
    ] {
        if std::env::var(key)
            .ok()
            .is_some_and(|v| matches!(v.as_str(), "1" | "true" | "yes"))
        {
            return true;
        }
    }
    false
}

/// Legacy alias.
pub fn mlx_native_backward_from_env() -> bool {
    native_backward_from_env()
}

/// When set, skip GPU backward lowering and force CPU backward (hybrid).
pub fn backward_cpu_only_from_env() -> bool {
    std::env::var("RLX_VOXTRAL_TTS_TRAIN_BACKWARD_CPU")
        .ok()
        .is_some_and(|v| matches!(v.as_str(), "1" | "true" | "yes"))
}

fn backward_exceeds_output_cap(graph: &Graph) -> bool {
    graph.outputs.len() > compile_output_cap()
}

fn output_cap_cpu_fallback_reason(device: Device, graph: &Graph) -> Option<&'static str> {
    if device == Device::Cpu {
        return None;
    }
    if backward_cpu_only_from_env() {
        return Some("RLX_VOXTRAL_TTS_TRAIN_BACKWARD_CPU");
    }
    if !native_backward_from_env()
        && device_has_compile_output_cap(device)
        && backward_exceeds_output_cap(graph)
    {
        return Some("RLX_COMPILE_OUTPUT_CAP");
    }
    None
}

pub fn compile_train_session(
    device: Device,
    forward_graph: Graph,
    backward_graph: Graph,
    label: &str,
) -> Result<TrainSession> {
    compile_train_session_opts(device, forward_graph, backward_graph, label, false)
}

pub fn compile_train_session_opts(
    device: Device,
    forward_graph: Graph,
    backward_graph: Graph,
    label: &str,
    force_cpu_backward: bool,
) -> Result<TrainSession> {
    let (forward_device, forward) = match try_compile_on(
        device,
        forward_graph.clone(),
        label,
        "forward",
        false,
    ) {
        Ok(v) => v,
        Err(reason) if device != Device::Cpu => {
            eprintln!(
                "[{label}] {device:?} forward unsupported ({reason}) — falling back to CPU forward."
            );
            try_compile_on(Device::Cpu, forward_graph, label, "forward", false)?
        }
        Err(reason) => bail!("{label} forward compile failed on CPU: {reason}"),
    };

    let (backward_device, backward) =
        compile_backward_on_device(device, backward_graph, label, force_cpu_backward)?;

    if forward_device != backward_device {
        eprintln!("[{label}] hybrid: forward={forward_device:?} backward={backward_device:?}");
    }

    Ok(TrainSession {
        forward_device,
        backward_device,
        forward,
        backward,
    })
}

/// Compile only the backward graph (encoder training runs loss+grad in one pass).
pub fn compile_train_backward(
    device: Device,
    backward_graph: Graph,
    label: &str,
) -> Result<(Device, CompiledGraph)> {
    compile_train_backward_opts(device, backward_graph, label, false)
}

pub fn compile_train_backward_opts(
    device: Device,
    backward_graph: Graph,
    label: &str,
    force_cpu_backward: bool,
) -> Result<(Device, CompiledGraph)> {
    compile_backward_on_device(device, backward_graph, label, force_cpu_backward)
}

fn compile_backward_on_device(
    device: Device,
    backward_graph: Graph,
    label: &str,
    force_cpu_backward: bool,
) -> Result<(Device, CompiledGraph)> {
    if force_cpu_backward {
        eprintln!("[{label}] multi-layer LoRA graph — backward on CPU.");
        return try_compile_on(Device::Cpu, backward_graph, label, "backward", false);
    }
    if let Some(reason) = output_cap_cpu_fallback_reason(device, &backward_graph) {
        match reason {
            "RLX_COMPILE_OUTPUT_CAP" => {
                let cap = compile_output_cap();
                eprintln!("[{label}] {device:?} compile output cap ({cap}) — backward on CPU.");
            }
            "RLX_VOXTRAL_TTS_TRAIN_BACKWARD_CPU" => {
                eprintln!("[{label}] RLX_VOXTRAL_TTS_TRAIN_BACKWARD_CPU=1 — backward on CPU.");
            }
            _ => {}
        }
        return try_compile_on(Device::Cpu, backward_graph, label, "backward", false);
    }

    match try_compile_on(device, backward_graph.clone(), label, "backward", true) {
        Ok(v) => Ok(v),
        Err(reason) if device != Device::Cpu => {
            eprintln!(
                "[{label}] {device:?} backward unsupported ({reason}) — running backward on CPU."
            );
            try_compile_on(Device::Cpu, backward_graph, label, "backward", false)
        }
        Err(reason) => bail!("{label} backward compile failed on CPU: {reason}"),
    }
}

fn try_compile_on(
    device: Device,
    graph: Graph,
    _label: &str,
    which: &str,
    prep_backward: bool,
) -> Result<(Device, CompiledGraph)> {
    let graph = if prep_backward && which == "backward" {
        match catch_unwind(AssertUnwindSafe(|| {
            prepare_backward_for_device(graph, device)
        })) {
            Ok(g) => g,
            Err(_) => {
                bail!("backward prep failed for {device:?} (see panic above)")
            }
        }
    } else {
        graph
    };
    let session = Session::new(device);
    let compiled = try_compile(&session, graph)?;
    Ok((device, compiled))
}

fn try_compile(session: &Session, graph: Graph) -> Result<CompiledGraph> {
    catch_unwind(AssertUnwindSafe(|| session.compile(graph))).map_err(|_| {
        anyhow::anyhow!("backend missing op support (see rlx-opt compile error above)")
    })
}
