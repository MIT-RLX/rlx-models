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

//! Per-backend compile options / Metal MPSGraph + packed-GGUF guards for
//! Unlimited-OCR LM.

use crate::expert_pack::PackedLmWeights;
use crate::lm_precision::ResolvedLmPrecision;
use rlx_core::flow_bridge::{compile_options_from_profile, packed_gguf_compile_guard};
use rlx_flow::CompileProfile;
use rlx_ir::logical_kernel::KernelDispatchConfig;
use rlx_runtime::{CompileOptions, Device};

/// Disable fused layers on backends that do not lower residual+RMSNorm fusion yet.
pub fn lm_decode_compile_options(device: Device) -> CompileOptions {
    let mut profile = CompileProfile::llama32_decode();
    if matches!(device, Device::Gpu | Device::Vulkan | Device::Mlx) {
        profile.fusion.skip = true;
    }
    compile_options_from_profile(&profile, device, KernelDispatchConfig::default())
}

pub fn lm_prefill_compile_options(device: Device) -> CompileOptions {
    let mut profile = CompileProfile::llama32_prefill();
    if matches!(device, Device::Gpu | Device::Vulkan) {
        profile.fusion.skip = true;
    }
    compile_options_from_profile(&profile, device, KernelDispatchConfig::default())
}

/// Wrap LM graph compile/run for F32/F16 IR (Metal MPSGraph attention issues).
pub fn metal_lm_compile_guard<R, F>(device: Device, f: F) -> R
where
    F: FnOnce() -> R,
{
    if device == Device::Metal {
        rlx_ir::env::set("RLX_DISABLE_MPSGRAPH", "1");
        let out = f();
        rlx_ir::env::unset("RLX_DISABLE_MPSGRAPH");
        out
    } else {
        f()
    }
}

/// Compile/run guard for Unlimited-OCR LM graphs.
///
/// - Packed Q8/Q4 IR → [`packed_gguf_compile_guard`] (Metal MPSGraph off + MLX lazy).
/// - Otherwise → [`metal_lm_compile_guard`].
pub fn lm_runtime_guard<R, F>(device: Device, packed_quants_in_ir: bool, f: F) -> R
where
    F: FnOnce() -> R,
{
    if packed_quants_in_ir {
        packed_gguf_compile_guard(device, f)
    } else {
        metal_lm_compile_guard(device, f)
    }
}

/// Pack-aware compile/run guard.
///
/// On Metal + Q4_0 soft-pack, disables fused grouped GEMV unless the user
/// already set `RLX_METAL_GROUPED_GEMV_DISABLE`. crates.io `rlx` 0.2.14 still
/// ships an interleaved-nibble `q4_0_mv_f32`; the non-fused path stays correct.
/// Local RLX with the split-nibble GEMV fix can set
/// `RLX_METAL_GROUPED_GEMV_DISABLE=0` to keep the fast path.
pub fn lm_runtime_guard_for_pack<R, F>(device: Device, pack: &PackedLmWeights, f: F) -> R
where
    F: FnOnce() -> R,
{
    let packed = pack.keeps_quants_in_ir();
    let need_q4_grouped_off = device == Device::Metal
        && packed
        && pack.resolved_precision == ResolvedLmPrecision::Q4_0
        && std::env::var_os("RLX_METAL_GROUPED_GEMV_DISABLE").is_none();
    if need_q4_grouped_off {
        rlx_ir::env::set("RLX_METAL_GROUPED_GEMV_DISABLE", "1");
        let out = lm_runtime_guard(device, packed, f);
        rlx_ir::env::unset("RLX_METAL_GROUPED_GEMV_DISABLE");
        out
    } else {
        lm_runtime_guard(device, packed, f)
    }
}
