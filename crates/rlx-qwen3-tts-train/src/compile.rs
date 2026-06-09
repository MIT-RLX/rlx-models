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

//! Compile training backward on MLX / Metal / CUDA (fused lowering when available).

use anyhow::Result;
use rlx_ir::Graph;
use rlx_runtime::{CompiledGraph, Device, Session};
use std::panic::{AssertUnwindSafe, catch_unwind};

use crate::backward_prep::prepare_backward_for_device;

pub fn backward_cpu_only_from_env() -> bool {
    std::env::var("RLX_QWEN3_TTS_TRAIN_BACKWARD_CPU")
        .ok()
        .is_some_and(|v| matches!(v.as_str(), "1" | "true" | "yes"))
}

pub fn compile_train_backward(
    device: Device,
    backward_graph: Graph,
    label: &str,
) -> Result<(Device, CompiledGraph)> {
    compile_train_backward_opts(device, backward_graph, label, backward_cpu_only_from_env())
}

pub fn compile_train_backward_opts(
    device: Device,
    backward_graph: Graph,
    label: &str,
    force_cpu_backward: bool,
) -> Result<(Device, CompiledGraph)> {
    if force_cpu_backward {
        eprintln!("[{label}] RLX_QWEN3_TTS_TRAIN_BACKWARD_CPU=1 — backward on CPU");
        return Ok((Device::Cpu, try_compile(Device::Cpu, backward_graph, true)?));
    }
    match try_compile(device, backward_graph.clone(), true) {
        Ok(c) => Ok((device, c)),
        Err(e) if device != Device::Cpu => {
            eprintln!("[{label}] {device:?} backward failed ({e}) — CPU fallback");
            Ok((Device::Cpu, try_compile(Device::Cpu, backward_graph, true)?))
        }
        Err(e) => Err(e),
    }
}

fn try_compile(device: Device, graph: Graph, prep: bool) -> Result<CompiledGraph> {
    let graph = if prep {
        prepare_backward_for_device(graph, device)
    } else {
        graph
    };
    catch_unwind(AssertUnwindSafe(|| Session::new(device).compile(graph)))
        .map_err(|_| anyhow::anyhow!("compile on {device:?} failed (see log above)"))
}
