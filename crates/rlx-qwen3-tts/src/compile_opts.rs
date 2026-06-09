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

//! Per-device compile options for talker / code-predictor Qwen3 graphs.

use rlx_core::flow_bridge::compile_options_from_profile;
use rlx_flow::{CompileProfile, FusionTargetKind};
use rlx_ir::logical_kernel::KernelDispatchConfig;
use rlx_runtime::{CompileOptions, Device};
use std::cell::Cell;

thread_local! {
    static METAL_GUARD_DEPTH: Cell<usize> = const { Cell::new(0) };
}

/// Metal: disable MPSGraph for stable Qwen3 lowering (matches voxtral-tts / packed GGUF guard).
/// Nested calls only set/unset the env var once (hot synthesis loop calls this per decode step).
pub fn metal_mpsgraph_enabled() -> bool {
    if std::env::var("RLX_QWEN3_TTS_METAL_MPSGRAPH")
        .ok()
        .as_deref()
        != Some("1")
    {
        return false;
    }
    // Bucketed decode + native Metal compile still hits MPSGraph SDPA matmul bugs.
    if talker_metal_native_compile(Device::Metal)
        && std::env::var("RLX_QWEN3_TTS_METAL_MPSGRAPH_FORCE")
            .ok()
            .as_deref()
            != Some("1")
    {
        return false;
    }
    true
}

/// Metal graph **execution** must not run under `RLX_DISABLE_MPSGRAPH` when compile
/// paths set it via [`ensure_metal_lowering_env`] / [`metal_compile_guard`]. Leaving
/// the flag active during run zeros/NaNs bucketed talker decode graphs.
pub fn metal_mpsgraph_run_guard<R, F>(device: Device, f: F) -> R
where
    F: FnOnce() -> R,
{
    if device != Device::Metal {
        return f();
    }
    let had_disable = rlx_ir::env::var("RLX_DISABLE_MPSGRAPH").is_some();
    if had_disable {
        rlx_ir::env::unset("RLX_DISABLE_MPSGRAPH");
    }
    let out = f();
    if had_disable {
        rlx_ir::env::set("RLX_DISABLE_MPSGRAPH", "1");
    }
    out
}

pub fn metal_compile_guard<R, F>(device: Device, f: F) -> R
where
    F: FnOnce() -> R,
{
    if device == Device::Metal && !metal_mpsgraph_enabled() {
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

fn fusion_skip_for_device(device: Device) -> bool {
    if std::env::var("RLX_QWEN3_TTS_FUSION_SKIP").ok().as_deref() == Some("1") {
        return true;
    }
    // MLX bucketed decode mis-binds fused `inputs_embeds`; portable GPU stacks lack fused lowerings.
    matches!(device, Device::Mlx | Device::Gpu | Device::Vulkan)
}

fn fusion_target_for_device(device: Device) -> FusionTargetKind {
    match device {
        Device::Metal => FusionTargetKind::Metal,
        Device::Mlx => FusionTargetKind::Mlx,
        Device::Cuda => FusionTargetKind::Cuda,
        Device::Rocm => FusionTargetKind::Rocm,
        _ => FusionTargetKind::Auto,
    }
}

/// CP compile profile (same fusion target as talker; override via `RLX_QWEN3_TTS_CP_METAL_UNFUSE=1`).
pub fn tune_cp_qwen3_profile(profile: &mut CompileProfile, device: Device) {
    tune_qwen3_profile(profile, device);
    if device == Device::Metal
        && std::env::var("RLX_QWEN3_TTS_CP_METAL_UNFUSE")
            .ok()
            .as_deref()
            == Some("1")
    {
        profile.fusion.skip = true;
        profile.backend.metal.unfuse_regions = true;
        profile.passes.constant_folding = false;
    }
}

fn metal_fusion_skip() -> bool {
    std::env::var("RLX_QWEN3_TTS_METAL_FUSION_SKIP")
        .ok()
        .as_deref()
        == Some("1")
}

/// Native Metal talker compile caches (prefill + decode). Default on GPU sessions; opt out with `METAL_COMPILED=0`.
pub fn talker_metal_native_compile(device: Device) -> bool {
    if device != Device::Metal {
        return false;
    }
    match std::env::var("RLX_QWEN3_TTS_METAL_COMPILED")
        .ok()
        .as_deref()
    {
        Some("1") => true,
        Some("0") => false,
        _ => {
            crate::gpu_pipeline::gpu_session_enabled(device)
                && crate::gpu_pipeline::metal_compiled_default()
        }
    }
}

/// Native Metal talker decode via HIR (experimental; still diverges today — decode stays on CPU).
pub fn talker_decode_use_hir_compile(device: Device) -> bool {
    talker_metal_native_compile(device)
        && std::env::var("RLX_QWEN3_TTS_METAL_DECODE_HIR")
            .ok()
            .as_deref()
            == Some("1")
}

/// Keep MPSGraph off for stable Metal bucketed decode (set once per synthesis session).
pub fn ensure_metal_lowering_env(device: Device) {
    if device == Device::Metal && !metal_mpsgraph_enabled() {
        rlx_ir::env::set("RLX_DISABLE_MPSGRAPH", "1");
    }
}

/// Device for talker compile caches (prefill + bucketed decode). Metal sessions default to CPU graphs.
pub fn talker_compile_device(session_device: Device) -> Device {
    if session_device == Device::Metal && !talker_metal_native_compile(session_device) {
        Device::Cpu
    } else {
        session_device
    }
}

/// Bucketed decode compile/run device. GPU sessions default to native Metal decode.
/// Force CPU graphs: `RLX_QWEN3_TTS_METAL_DECODE_CPU=1`.
pub fn talker_decode_compile_device(session_device: Device) -> Device {
    if session_device == Device::Metal {
        if std::env::var("RLX_QWEN3_TTS_METAL_DECODE_CPU")
            .ok()
            .as_deref()
            == Some("1")
        {
            return Device::Cpu;
        }
        if std::env::var("RLX_QWEN3_TTS_METAL_DECODE_NATIVE")
            .ok()
            .as_deref()
            == Some("1")
        {
            return session_device;
        }
        if crate::gpu_pipeline::gpu_session_enabled(session_device)
            && crate::gpu_pipeline::metal_decode_native_default()
        {
            return session_device;
        }
        Device::Cpu
    } else {
        talker_compile_device(session_device)
    }
}

/// Fusion/compile profile target for talker **prefill** (may differ from decode on Metal hybrid).
pub fn talker_prefill_profile_device(session_device: Device) -> Device {
    if talker_metal_cpu_prefill(session_device) {
        Device::Cpu
    } else {
        talker_compile_device(session_device)
    }
}

/// Code-predictor compile device. GPU sessions compile CP on the session device.
/// Force CPU CP graphs: `RLX_QWEN3_TTS_CP_CPU_COMPILE=1`.
pub fn cp_compile_device(session_device: Device) -> Device {
    if session_device == Device::Metal {
        if std::env::var("RLX_QWEN3_TTS_CP_CPU_COMPILE")
            .ok()
            .as_deref()
            == Some("1")
        {
            return Device::Cpu;
        }
        if std::env::var("RLX_QWEN3_TTS_CP_METAL").ok().as_deref() == Some("1")
            || (crate::gpu_pipeline::gpu_session_enabled(session_device)
                && crate::gpu_pipeline::cp_compiled_default())
        {
            return session_device;
        }
        return Device::Cpu;
    }
    talker_compile_device(session_device)
}

/// Metal hybrid: CPU prefill cache + Metal decode cache (experimental; decode still diverges).
/// Opt out with `RLX_QWEN3_TTS_METAL_CPU_PREFILL=0`.
pub fn talker_metal_cpu_prefill(device: Device) -> bool {
    if device != Device::Metal || !talker_metal_native_compile(device) {
        return false;
    }
    std::env::var("RLX_QWEN3_TTS_METAL_CPU_PREFILL")
        .ok()
        .as_deref()
        != Some("0")
}

/// Tier-1 Qwen3 profile tuned for the execution device (fused Metal/CUDA decode by default).
pub fn tune_qwen3_profile(profile: &mut CompileProfile, device: Device) {
    if fusion_skip_for_device(device) {
        profile.fusion.skip = true;
        return;
    }
    if device == Device::Metal && metal_fusion_skip() {
        profile.fusion.skip = true;
        profile.fusion.policy = rlx_flow::FusionPolicyKind::Direct;
        return;
    }
    profile.fusion.skip = false;
    profile.fusion.target = fusion_target_for_device(device);
}

/// Tier-1 profile + backend fusion target.
pub fn talker_compile_options(profile: &CompileProfile, device: Device) -> CompileOptions {
    let mut profile = profile.clone();
    tune_qwen3_profile(&mut profile, device);
    compile_options_from_profile(&profile, device, KernelDispatchConfig::default())
}

/// Decode compile options for bucketed talker graphs.
pub fn talker_decode_compile_options(profile: &CompileProfile, device: Device) -> CompileOptions {
    talker_compile_options(profile, device)
}
