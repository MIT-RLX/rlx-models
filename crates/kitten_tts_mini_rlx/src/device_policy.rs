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

//! KittenTTS device policy: graph placement, discrete NVIDIA memory caps, QMatMul.
//!
//! ## Discrete NVIDIA (`Device::Gpu` / `Device::Vulkan`, non-macOS)
//!
//! Act arenas must stay **unsharded**. Sharded Vulkan no longer SIGSEGVs, but NSF
//! still collapses (peak ≪ 0.1). Validated on RTX 3080 Ti:
//!
//! | | wgpu (`Gpu`) | Vulkan |
//! |---|---|---|
//! | frames/token | import default (CPU wave); 8 when on-device | 8 |
//! | wave cap | 32 k only if `GPU_WAVE=wgpu`; else Vulkan 80 k via `GPU_WAVE=1` | 80 k |
//! | wave default | **Cuda** when available; else **CPU**; `GPU_WAVE=1` → Vulkan | on-device |
//!
//! Call [`prepare`] / [`resolve_device`] at engine load. Discrete `Gpu` upgrades
//! to `Cuda` when the CUDA runtime is present (`KITTEN_RLX_FORCE_WGPU=1` to keep
//! wgpu). Override caps with `KITTEN_RLX_WGPU_WAVEFORM_CAP` /
//! `KITTEN_RLX_VULKAN_WAVEFORM_CAP` (`0`/`off` disables). Force on-device wave with
//! `KITTEN_RLX_GPU_WAVE=1` (routes to Vulkan when not on Cuda).

use rlx_runtime::Device;
use std::sync::atomic::{AtomicBool, Ordering};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Mel-time frames/token on discrete Gpu/Vulkan (import default is 64).
pub const DISCRETE_MAX_FRAMES_PER_TOKEN: usize = 8;

/// Unsharded long-IPA wave cap on NVIDIA wgpu (~2 GiB storage bind).
pub const WGPU_WAVEFORM_CAP: usize = 32_000;

/// Unsharded long-IPA wave cap on native Vulkan (~4 GiB `maxStorageBufferRange`).
/// 80 k ≈ 3.9 GiB on RTX 3080 Ti; 88 k snaps to 2×4 GiB and quality drops.
pub const VULKAN_WAVEFORM_CAP: usize = 80_000;

/// Legacy aliases (prefer [`WGPU_WAVEFORM_CAP`] / [`DISCRETE_MAX_FRAMES_PER_TOKEN`]).
pub const DISCRETE_WGPU_WAVEFORM_CAP: usize = WGPU_WAVEFORM_CAP;
pub const DISCRETE_VULKAN_WAVEFORM_CAP: usize = VULKAN_WAVEFORM_CAP;
pub const DISCRETE_WGPU_MAX_FRAMES_PER_TOKEN: usize = DISCRETE_MAX_FRAMES_PER_TOKEN;

const WAVEFORM_CAP_FLOOR: usize = 24_000;
const SHARD_STAGE_MIB: &str = "64";

static DISCRETE_DEFAULTS_APPLIED: AtomicBool = AtomicBool::new(false);

// ---------------------------------------------------------------------------
// Env helpers
// ---------------------------------------------------------------------------

fn env_truthy(key: &str) -> bool {
    std::env::var(key)
        .is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("yes"))
}

fn env_falsy(key: &str) -> bool {
    std::env::var(key)
        .is_ok_and(|v| v == "0" || v.eq_ignore_ascii_case("false") || v.eq_ignore_ascii_case("no"))
}

fn env_disabled(key: &str) -> bool {
    std::env::var(key)
        .is_ok_and(|v| v == "0" || v.eq_ignore_ascii_case("false") || v.eq_ignore_ascii_case("off"))
}

fn env_usize(key: &str) -> Option<usize> {
    std::env::var(key).ok()?.parse().ok()
}

fn set_env_default(key: &str, value: &str) -> bool {
    if std::env::var(key).is_err() {
        crate::set_env_var(key, value);
        true
    } else {
        false
    }
}

// ---------------------------------------------------------------------------
// Device classification
// ---------------------------------------------------------------------------

/// Non-macOS `Device::Gpu` (discrete NVIDIA / DX12 wgpu).
#[inline]
pub fn is_discrete_wgpu(device: Device) -> bool {
    #[cfg(not(target_os = "macos"))]
    {
        device == Device::Gpu
    }
    #[cfg(target_os = "macos")]
    {
        let _ = device;
        false
    }
}

/// Non-macOS `Device::Vulkan`.
#[inline]
pub fn is_discrete_vulkan(device: Device) -> bool {
    #[cfg(not(target_os = "macos"))]
    {
        device == Device::Vulkan
    }
    #[cfg(target_os = "macos")]
    {
        let _ = device;
        false
    }
}

/// Discrete NVIDIA path that needs wave/frame caps (`Gpu` or `Vulkan`, non-macOS).
#[inline]
pub fn is_discrete_nvidia(device: Device) -> bool {
    is_discrete_wgpu(device) || is_discrete_vulkan(device)
}

fn device_label(device: Device) -> &'static str {
    match device {
        Device::Vulkan => "Device::Vulkan",
        Device::Gpu => "Device::Gpu",
        _ => "device",
    }
}

// ---------------------------------------------------------------------------
// Graph placement
// ---------------------------------------------------------------------------

/// Map a requested device to the fastest correct backend for Kitten.
///
/// On discrete NVIDIA, `Device::Gpu` (wgpu) is much slower and weaker than
/// CUDA for long IPA. When the CUDA runtime is available in this build,
/// upgrade `Gpu` → `Cuda` unless `KITTEN_RLX_FORCE_WGPU=1` /
/// `KITTEN_RLX_NO_CUDA_UPGRADE=1`.
pub fn resolve_device(requested: Device) -> Device {
    if env_truthy("KITTEN_RLX_FORCE_WGPU") || env_truthy("KITTEN_RLX_NO_CUDA_UPGRADE") {
        return requested;
    }
    if is_discrete_wgpu(requested) && rlx_runtime::is_available(Device::Cuda) {
        eprintln!(
            "[kittentts] Device::Gpu → Cuda (native NVIDIA; set KITTEN_RLX_FORCE_WGPU=1 to keep wgpu)"
        );
        return Device::Cuda;
    }
    requested
}

/// Device for the duration-refine graph.
///
/// Default: CPU on discrete wgpu / ANE (integer duration drifts on those
/// f32-uniform arenas). Vulkan, Metal, MLX, and CUDA keep on-device duration
/// (Whisper / peak-validated on NVIDIA for Vulkan). When discrete `Gpu` wave is
/// routed to Vulkan (`KITTEN_RLX_GPU_WAVE=1`), duration follows so both graphs
/// share one backend. Force GPU with `KITTEN_RLX_CPU_DURATION=0` or
/// `RLX_KITTEN_GPU_DURATION=1`; force CPU with `KITTEN_RLX_CPU_DURATION=1`.
pub fn duration_device(target: Device) -> Device {
    if target == Device::Cpu {
        return target;
    }
    if env_falsy("KITTEN_RLX_CPU_DURATION") || env_truthy("RLX_KITTEN_GPU_DURATION") {
        return target;
    }
    if env_truthy("KITTEN_RLX_CPU_DURATION") {
        return Device::Cpu;
    }
    match target {
        Device::Ane => Device::Cpu,
        // Discrete wgpu alone drifts duration; when wave is already on Vulkan
        // (GPU_WAVE=1), keep duration there too (same as `--device vulkan`).
        Device::Gpu if is_discrete_wgpu(target) => {
            if wave_device(target) == Device::Vulkan {
                Device::Vulkan
            } else {
                Device::Cpu
            }
        }
        Device::Gpu => Device::Cpu,
        _ => target,
    }
}

/// Device for the waveform / vocoder graph.
///
/// Default: on-device except ANE and discrete wgpu. Discrete `Device::Gpu` pins
/// wave to CPU — the wgpu path hosts Binary/Expand/Conv (SPIR-V collapses NSF)
/// and is ~17 s/hello; Vulkan keeps on-device wave (~1.4 s/hello on RTX 3080 Ti).
/// Force on-device Gpu wave with `KITTEN_RLX_GPU_WAVE=1` — on discrete NVIDIA
/// that **routes to [`Device::Vulkan`]** (same GPU, working kernels). Opt into
/// the slow wgpu Gpu wave with `KITTEN_RLX_GPU_WAVE=wgpu`. Force CPU with
/// `KITTEN_RLX_CPU_WAVE=1`.
pub fn wave_device(target: Device) -> Device {
    if target == Device::Cpu {
        return target;
    }
    if env_truthy("KITTEN_RLX_CPU_WAVE") {
        return Device::Cpu;
    }
    // Explicit wgpu Gpu wave (slow / diagnostic).
    if std::env::var("KITTEN_RLX_GPU_WAVE")
        .map(|v| v.eq_ignore_ascii_case("wgpu"))
        .unwrap_or(false)
    {
        return target;
    }
    if env_truthy("KITTEN_RLX_GPU_WAVE") || env_falsy("KITTEN_RLX_CPU_WAVE") {
        // Discrete wgpu Device::Gpu: prefer Vulkan for on-device wave.
        if is_discrete_wgpu(target) {
            return Device::Vulkan;
        }
        return target;
    }
    match target {
        Device::Ane => Device::Cpu,
        Device::Gpu if is_discrete_wgpu(target) => Device::Cpu,
        _ => target,
    }
}

/// Rewrite quantized ALBERT matmuls to native f32 GEMM
/// ([`crate::hir_qdq_fuse::rewrite_qmatmul_to_native_f32`]).
///
/// Default on for CPU / CUDA / ROCm / Vulkan / discrete wgpu. Off on Metal /
/// MLX / ANE / macOS Gpu (zeros the wave graph). Override with
/// `KITTEN_RLX_NATIVE_QMATMUL=0|1`.
pub fn native_qmatmul(device: Device) -> bool {
    if let Ok(v) = std::env::var("KITTEN_RLX_NATIVE_QMATMUL") {
        if env_truthy_val(&v) {
            return true;
        }
        if env_falsy_val(&v) {
            return false;
        }
    }
    match device {
        Device::Cpu | Device::Cuda | Device::Rocm | Device::Vulkan => true,
        Device::Gpu => is_discrete_wgpu(device),
        _ => false,
    }
}

fn env_truthy_val(v: &str) -> bool {
    v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("yes")
}

fn env_falsy_val(v: &str) -> bool {
    v == "0" || v.eq_ignore_ascii_case("false") || v.eq_ignore_ascii_case("no")
}

// ---------------------------------------------------------------------------
// Memory policy
// ---------------------------------------------------------------------------

/// Effective alignment frames/token (`KITTEN_RLX_MAX_FRAMES_PER_TOKEN`, else import max).
pub fn max_frames_per_token() -> usize {
    let hard_max = crate::bundle_compile::MAX_FRAMES_PER_TOKEN;
    env_usize("KITTEN_RLX_MAX_FRAMES_PER_TOKEN")
        .map(|n| n.max(1).min(hard_max))
        .unwrap_or(hard_max)
}

fn wave_cap_env_key(device: Device) -> &'static str {
    if is_discrete_vulkan(device) {
        "KITTEN_RLX_VULKAN_WAVEFORM_CAP"
    } else {
        "KITTEN_RLX_WGPU_WAVEFORM_CAP"
    }
}

fn default_wave_cap(device: Device) -> usize {
    if is_discrete_vulkan(device) {
        VULKAN_WAVEFORM_CAP
    } else {
        WGPU_WAVEFORM_CAP
    }
}

/// Clamp waveform compile width for discrete Gpu/Vulkan. No-op elsewhere.
///
/// When wave is pinned to CPU ([`wave_device`]), skip the bind-window cap so
/// long IPA can single-pass on the host vocoder.
pub fn clamp_waveform(device: Device, max_wave: usize) -> usize {
    if !is_discrete_nvidia(device) {
        return max_wave;
    }
    let wave = wave_device(device);
    // Wave on CPU → no storage-bind limit.
    if wave == Device::Cpu {
        return max_wave;
    }
    // Cap follows the *wave* device (Gpu+GPU_WAVE routes to Vulkan → 80 k).
    let key = wave_cap_env_key(wave);
    if env_disabled(key) {
        return max_wave;
    }
    let cap = env_usize(key)
        .map(|n| n.max(WAVEFORM_CAP_FLOOR))
        .unwrap_or_else(|| default_wave_cap(wave));
    max_wave.min(cap)
}

/// Install discrete Gpu/Vulkan/Cuda env defaults once (idempotent).
pub fn apply_defaults(device: Device) {
    let cuda = device == Device::Cuda;
    if !is_discrete_nvidia(device) && !cuda {
        return;
    }
    if DISCRETE_DEFAULTS_APPLIED.swap(true, Ordering::Relaxed) {
        return;
    }

    let mut notes: Vec<String> = Vec::new();
    if is_discrete_wgpu(device) {
        if set_env_default("RLX_WGPU_NO_F16_SHADOW", "1") {
            notes.push("NO_F16_SHADOW=1".into());
        }
        if set_env_default("RLX_WGPU_SHARD_STAGE_MIB", SHARD_STAGE_MIB) {
            notes.push("SHARD_STAGE_MIB=64".into());
        }
    }
    if is_discrete_vulkan(device) && set_env_default("RLX_VULKAN_SHARD_STAGE_MIB", SHARD_STAGE_MIB)
    {
        notes.push("VULKAN_SHARD_STAGE_MIB=64".into());
    }
    // Cuda vocoder: TF32 + force cuDNN for 1×k / grouped fwd convs (HiFi-GAN
    // style). Without FWD_CUDNN the direct kernel burns ~1 s of the wave pass.
    // Opt out with RLX_CUDA_CONV_TF32=0 / RLX_CUDA_CONV_FWD_CUDNN=0 /
    // RLX_CUDA_NO_TF32=1 / RLX_CUDA_NO_CUDNN=1.
    if cuda {
        if set_env_default("RLX_CUDA_CONV_TF32", "1") {
            notes.push("CONV_TF32=1".into());
        }
        if set_env_default("RLX_CUDA_CONV_FWD_CUDNN", "1") {
            notes.push("CONV_FWD_CUDNN=1".into());
        }
        // Philox fills on-device (Ort polar RNG is host-only and ~8 ms/wave).
        // Peak stays ~0.64; set KITTEN_RLX_RNG_BACKEND=ort for ORT-matching noise.
        if set_env_default("KITTEN_RLX_RNG_BACKEND", "philox") {
            notes.push("RNG_BACKEND=philox".into());
        }
    }
    // Mel frames/token=8 keeps discrete on-device wave unsharded. Skip when Gpu
    // wave is pinned to CPU (default) — import frames + full wave buffer give
    // Cuda-class long IPA. Force frames when Vulkan or `KITTEN_RLX_GPU_WAVE=1`.
    let need_frames = is_discrete_vulkan(device)
        || (is_discrete_wgpu(device) && wave_device(device) != Device::Cpu);
    if need_frames
        && set_env_default(
            "KITTEN_RLX_MAX_FRAMES_PER_TOKEN",
            &DISCRETE_MAX_FRAMES_PER_TOKEN.to_string(),
        )
    {
        notes.push(format!(
            "max_frames_per_token={DISCRETE_MAX_FRAMES_PER_TOKEN}"
        ));
    }
    if !notes.is_empty() {
        let label = if cuda {
            "Device::Cuda"
        } else {
            device_label(device)
        };
        eprintln!(
            "[kittentts] {label} defaults ({}); override via env",
            notes.join(", ")
        );
    }
}

/// Apply runtime + discrete defaults, then clamp waveform. Prefer at engine load.
///
/// Resolves [`resolve_device`] first so discrete `Gpu` becomes `Cuda` when available.
pub fn prepare(device: Device, max_waveform_samples: usize) -> (Device, usize) {
    let device = resolve_device(device);
    crate::compile_profile::apply_device_runtime_defaults(device);
    let capped = clamp_waveform(device, max_waveform_samples);
    if capped < max_waveform_samples {
        let key = wave_cap_env_key(wave_device(device));
        eprintln!(
            "[kittentts] {}: max_waveform_samples {max_waveform_samples} → {capped} \
             (storage bind; {key}=0 to disable)",
            device_label(device)
        );
    }
    (device, capped)
}

/// Alias for [`prepare`]; returns only the clamped waveform size (device via
/// [`resolve_device`] is applied by callers that use [`prepare`] directly).
#[inline]
pub fn prepare_device(device: Device, max_waveform_samples: usize) -> usize {
    prepare(device, max_waveform_samples).1
}

// ---------------------------------------------------------------------------
// Compatibility aliases
// ---------------------------------------------------------------------------

/// Alias for [`clamp_waveform`].
#[inline]
pub fn clamp_waveform_for_device(device: Device, max_wave: usize) -> usize {
    clamp_waveform(device, max_wave)
}

/// Alias for [`apply_defaults`].
#[inline]
pub fn apply_discrete_wgpu_defaults(device: Device) {
    apply_defaults(device)
}

/// Alias for [`duration_device`].
#[inline]
pub fn parity_duration_device(target: Device) -> Device {
    duration_device(target)
}

/// Alias for [`wave_device`].
#[inline]
pub fn parity_wave_device(target: Device) -> Device {
    wave_device(target)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wave_cap_defaults_and_override() {
        assert_eq!(clamp_waveform(Device::Cpu, 48_000), 48_000);

        #[cfg(not(target_os = "macos"))]
        {
            // Default Gpu wave is CPU → no bind clamp.
            if std::env::var("KITTEN_RLX_GPU_WAVE").is_err()
                && std::env::var("KITTEN_RLX_CPU_WAVE").is_err()
            {
                assert_eq!(clamp_waveform(Device::Gpu, 48_000), 48_000);
            }
            // Forced on-device Gpu wave routes to Vulkan → 80 k cap.
            if !env_disabled("KITTEN_RLX_VULKAN_WAVEFORM_CAP")
                && env_usize("KITTEN_RLX_VULKAN_WAVEFORM_CAP").is_none()
            {
                crate::set_env_var("KITTEN_RLX_GPU_WAVE", "1");
                assert_eq!(wave_device(Device::Gpu), Device::Vulkan);
                assert_eq!(clamp_waveform(Device::Gpu, 160_000), VULKAN_WAVEFORM_CAP);
                assert_eq!(clamp_waveform(Device::Gpu, 16_000), 16_000);
                std::env::remove_var("KITTEN_RLX_GPU_WAVE");
            }
            // Explicit wgpu Gpu wave keeps the 32 k wgpu cap.
            if !env_disabled("KITTEN_RLX_WGPU_WAVEFORM_CAP")
                && env_usize("KITTEN_RLX_WGPU_WAVEFORM_CAP").is_none()
            {
                crate::set_env_var("KITTEN_RLX_GPU_WAVE", "wgpu");
                assert_eq!(wave_device(Device::Gpu), Device::Gpu);
                assert_eq!(clamp_waveform(Device::Gpu, 48_000), WGPU_WAVEFORM_CAP);
                std::env::remove_var("KITTEN_RLX_GPU_WAVE");
            }
            if !env_disabled("KITTEN_RLX_VULKAN_WAVEFORM_CAP")
                && env_usize("KITTEN_RLX_VULKAN_WAVEFORM_CAP").is_none()
            {
                assert_eq!(clamp_waveform(Device::Vulkan, 160_000), VULKAN_WAVEFORM_CAP);
                assert_eq!(clamp_waveform(Device::Vulkan, 32_000), 32_000);
            }
        }
    }

    #[test]
    fn frames_respect_hard_max() {
        assert!(max_frames_per_token() <= crate::bundle_compile::MAX_FRAMES_PER_TOKEN);
        const {
            assert!(DISCRETE_MAX_FRAMES_PER_TOKEN <= crate::bundle_compile::MAX_FRAMES_PER_TOKEN);
        }
    }

    #[test]
    fn placement_pins() {
        assert_eq!(duration_device(Device::Cpu), Device::Cpu);
        assert_eq!(wave_device(Device::Cpu), Device::Cpu);
        assert_eq!(wave_device(Device::Ane), Device::Cpu);
        assert_eq!(duration_device(Device::Cuda), Device::Cuda);
        assert_eq!(wave_device(Device::Cuda), Device::Cuda);
        #[cfg(not(target_os = "macos"))]
        {
            if std::env::var("KITTEN_RLX_CPU_DURATION").is_err()
                && std::env::var("RLX_KITTEN_GPU_DURATION").is_err()
            {
                assert_eq!(duration_device(Device::Vulkan), Device::Vulkan);
                assert_eq!(duration_device(Device::Gpu), Device::Cpu);
                crate::set_env_var("KITTEN_RLX_GPU_WAVE", "1");
                assert_eq!(duration_device(Device::Gpu), Device::Vulkan);
                std::env::remove_var("KITTEN_RLX_GPU_WAVE");
            }
            if std::env::var("KITTEN_RLX_CPU_WAVE").is_err()
                && std::env::var("KITTEN_RLX_GPU_WAVE").is_err()
            {
                assert_eq!(wave_device(Device::Vulkan), Device::Vulkan);
                assert_eq!(wave_device(Device::Gpu), Device::Cpu);
            }
        }
    }

    #[test]
    fn native_qmatmul_cpu_cuda() {
        // Do not assert env-overridable defaults when the var may be set in CI.
        if std::env::var("KITTEN_RLX_NATIVE_QMATMUL").is_err() {
            assert!(native_qmatmul(Device::Cpu));
            assert!(native_qmatmul(Device::Cuda));
            assert!(!native_qmatmul(Device::Metal));
            assert!(!native_qmatmul(Device::Ane));
        }
    }
}
