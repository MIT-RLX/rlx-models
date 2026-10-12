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

//! Shared backend policy for RLX model crates.
//!
//! Every model family in this workspace targets the same seven execution
//! backends. Call [`validate_standard_device`] at runner / loader build
//! time; enable matching `rlx-runtime` features on the model crate
//! (`metal`, `mlx`, `cuda`, `rocm`, `gpu`, `vulkan`, or `all-backends`).

use anyhow::{Result, bail};
use rlx_runtime::{Device, memory_estimate};

/// Backends every model crate is expected to support when the matching
/// `rlx-runtime` feature is enabled at build time.
pub const STANDARD_DEVICES: &[Device] = &[
    Device::Cpu,
    Device::Metal,
    Device::Mlx,
    Device::Cuda,
    Device::Rocm,
    Device::Gpu,
    Device::Vulkan,
];

/// CLI / help string for `--device`.
pub const STANDARD_DEVICE_NAMES: &str = "auto|cpu|metal|mps|mlx|cuda|rocm|hip|gpu|wgpu|vulkan";

/// [`STANDARD_DEVICE_NAMES`] plus CoreML / ANE when the `coreml` feature is enabled.
pub const LM_DEVICE_NAMES: &str =
    "auto|cpu|metal|mps|mlx|cuda|rocm|hip|gpu|wgpu|vulkan|coreml|ane|tpu";

/// Preferred causal-LM inference order on this host (excludes ANE — request `coreml` explicitly).
pub const LM_INFERENCE_DEVICE_PRIORITY: &[Device] = &[
    Device::Cuda,
    Device::Rocm,
    Device::Mlx,
    Device::Metal,
    Device::Gpu,
    Device::Vulkan,
    Device::Cpu,
];

/// Best available accelerator for text LM inference (CUDA → ROCm → MLX → Metal → … → CPU).
pub fn pick_lm_device() -> Device {
    let avail = rlx_runtime::available_devices();
    for &d in LM_INFERENCE_DEVICE_PRIORITY {
        if avail.contains(&d) {
            return d;
        }
    }
    Device::Cpu
}

/// Parse `--device auto` (or empty) via [`pick_lm_device`]; otherwise delegate to `FromStr`.
pub fn resolve_lm_device_str(family: &str, s: &str) -> Result<Device> {
    let key = s.trim().to_ascii_lowercase();
    if key == "auto" || key.is_empty() {
        return Ok(pick_lm_device());
    }
    let d = std::str::FromStr::from_str(s.trim()).map_err(|e| anyhow::anyhow!("{e}"))?;
    validate_lm_device(family, d)?;
    Ok(d)
}

/// True when `device` is in [`STANDARD_DEVICES`].
pub fn is_standard_device(device: Device) -> bool {
    STANDARD_DEVICES.contains(&device)
}

/// Causal LM runners: standard backends, plus CoreML (`Device::Ane`) and TPU
/// (`Device::Tpu`).
///
/// Unlike [`validate_lm_device`] this does not consider features — it answers
/// "is this device in the LM device *class*", which is what a `--device` help
/// string or a device menu wants. Use `validate_lm_device` before building.
pub fn is_lm_device(device: Device) -> bool {
    is_standard_device(device) || device == Device::Ane || device == Device::Tpu
}

/// Fail fast on exotic runtime devices (TPU, ANE, OpenGL, …).
pub fn validate_standard_device(family: &str, device: Device) -> Result<()> {
    if is_standard_device(device) {
        Ok(())
    } else {
        bail!(
            "{family}: device {device:?} is not supported \
             (use {STANDARD_DEVICE_NAMES})"
        )
    }
}

/// Like [`validate_standard_device`], but allows the two devices that are real
/// backends yet not "standard": `Device::Ane` with `coreml`, and `Device::Tpu`
/// with `tpu`.
///
/// Both are gated on their feature rather than accepted unconditionally,
/// because a build without the backend compiled in cannot reach the hardware —
/// and saying so here is better than a missing-symbol or an unsupported-op
/// failure later.
pub fn validate_lm_device(family: &str, device: Device) -> Result<()> {
    if device == Device::Ane {
        #[cfg(feature = "coreml")]
        return Ok(());
        #[cfg(not(feature = "coreml"))]
        bail!(
            "{family}: device Ane requires the `coreml` feature \
             (enable `coreml` or `apple-silicon` on this crate)"
        );
    }
    if device == Device::Tpu {
        #[cfg(feature = "tpu")]
        return Ok(());
        #[cfg(not(feature = "tpu"))]
        bail!(
            "{family}: device Tpu requires the `tpu` feature \
             (enable `tpu` on this crate; it forwards to rlx-runtime/tpu, \
             which drives libtpu through PJRT)"
        );
    }
    validate_standard_device(family, device)
}

/// `(free_bytes, total_bytes)` for TIDE MoE VRAM budget sizing.
///
/// Override with `RLX_CUDA_FREE_BYTES` / `RLX_CUDA_TOTAL_BYTES` or
/// `RLX_DEVICE_FREE_BYTES` / `RLX_DEVICE_TOTAL_BYTES`. On Apple Silicon
/// (Metal / MLX), falls back to unified memory when env vars are unset.
pub fn device_memory_for_moe_offload(device: Device) -> Option<(usize, usize)> {
    if let (Ok(free), Ok(total)) = (
        std::env::var("RLX_CUDA_FREE_BYTES"),
        std::env::var("RLX_CUDA_TOTAL_BYTES"),
    ) && let (Ok(f), Ok(t)) = (free.parse(), total.parse())
    {
        return Some((f, t));
    }
    if let (Ok(free), Ok(total)) = (
        std::env::var("RLX_DEVICE_FREE_BYTES"),
        std::env::var("RLX_DEVICE_TOTAL_BYTES"),
    ) && let (Ok(f), Ok(t)) = (free.parse(), total.parse())
    {
        return Some((f, t));
    }
    match device {
        Device::Metal | Device::Mlx | Device::Ane => {
            memory_estimate::available_unified_memory().map(|t| (t, t))
        }
        Device::Cuda | Device::Rocm | Device::Gpu | Device::Vulkan => {
            memory_estimate::available_unified_memory().map(|t| (t, t))
        }
        _ => None,
    }
}

/// SAM v1 also documents `tpu` on `rlx_sam::Sam::from_safetensors_on`.
pub fn validate_sam_device(family: &str, device: Device) -> Result<()> {
    if device == Device::Tpu || is_standard_device(device) {
        Ok(())
    } else {
        bail!(
            "{family}: device {device:?} is not supported \
             (use {STANDARD_DEVICE_NAMES} or tpu)"
        )
    }
}

/// Every device this build can actually reach: CPU, plus each GPU backend whose
/// cargo feature is on *and* whose hardware is present.
///
/// Both halves matter. A feature can be enabled on a machine with no such
/// device, and a device can be present in a build that cannot talk to it — so
/// neither `cfg!` nor [`rlx_runtime::is_available`] alone is the right question.
/// The feature gating has to live here rather than in `rlx-runtime`, because it
/// is *this* crate's features that pull the backends in.
///
/// Cross-backend tests should iterate this rather than hard-code a device: a
/// port that only ever ran on CPU is a port that has not been tested.
pub fn available_devices() -> Vec<Device> {
    compiled_devices()
        .into_iter()
        .filter(|d| *d == Device::Cpu || rlx_runtime::is_available(*d))
        .collect()
}

/// Every device this build *could* reach, whether or not the hardware is here.
///
/// Kept separate from [`available_devices`] so a caller can tell "the feature is
/// off" from "the feature is on but the driver is missing". A cross-backend test
/// that only sees the second list silently passes on CPU alone, which is the
/// worst outcome: the feature flag was set, the suite was green, and nothing was
/// actually checked.
pub fn compiled_devices() -> Vec<Device> {
    #[allow(unused_mut)]
    let mut v = vec![Device::Cpu];
    #[cfg(feature = "metal")]
    v.push(Device::Metal);
    #[cfg(feature = "mlx")]
    v.push(Device::Mlx);
    #[cfg(feature = "gpu")]
    v.push(Device::Gpu);
    #[cfg(feature = "vulkan")]
    v.push(Device::Vulkan);
    #[cfg(feature = "coreml")]
    v.push(Device::Ane);
    #[cfg(feature = "cuda")]
    v.push(Device::Cuda);
    #[cfg(feature = "rocm")]
    v.push(Device::Rocm);
    v.dedup();
    v
}

/// Devices this build has compiled in but cannot reach right now.
pub fn unavailable_compiled_devices() -> Vec<Device> {
    let live = available_devices();
    compiled_devices()
        .into_iter()
        .filter(|d| !live.contains(d))
        .collect()
}

/// Devices a run insists on, from `RLX_REQUIRE_DEVICES` (comma-separated, e.g.
/// `metal,mlx`).
///
/// A cross-backend test can consult this to turn "the GPU was not there, so only
/// CPU ran" from a silent pass into a failure — which is what CI wants, and what
/// a developer checking a specific backend wants.
pub fn required_devices() -> Vec<String> {
    std::env::var("RLX_REQUIRE_DEVICES")
        .ok()
        .map(|s| {
            s.split(',')
                .map(|t| t.trim().to_ascii_lowercase())
                .filter(|t| !t.is_empty())
                .collect()
        })
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standard_set_covers_cli_backends() {
        for dev in STANDARD_DEVICES {
            assert!(is_standard_device(*dev));
        }
        assert!(!is_standard_device(Device::Tpu));
    }

    #[test]
    fn pick_lm_device_is_available() {
        let picked = pick_lm_device();
        assert!(rlx_runtime::is_available(picked));
        assert!(is_lm_device(picked));
    }

    #[test]
    fn resolve_auto_picks_fastest() {
        let d = resolve_lm_device_str("test", "auto").unwrap();
        assert_eq!(d, pick_lm_device());
    }
}

#[cfg(test)]
mod tpu_tests {
    use super::*;

    /// TPU is in the LM device *class* regardless of features.
    ///
    /// `is_lm_device` answers "could an LM ever run here", which is what a
    /// `--device` help string needs; whether *this build* can is
    /// `validate_lm_device`'s question. Conflating the two is how a device
    /// menu ends up hiding a backend the user could enable.
    #[test]
    fn tpu_is_an_lm_device_class_member() {
        assert!(is_lm_device(Device::Tpu));
        assert!(is_lm_device(Device::Ane));
        assert!(!is_standard_device(Device::Tpu), "but not `standard`");
        // Genuinely unsupported devices stay out of both.
        for d in [Device::OpenGl, Device::Hexagon, Device::Xdna] {
            assert!(!is_lm_device(d), "{d:?}");
        }
        assert!(
            LM_DEVICE_NAMES.contains("tpu"),
            "and the help string says so"
        );
    }

    /// Validation follows the feature, and says which one to enable.
    #[test]
    fn validating_tpu_follows_the_feature() {
        let r = validate_lm_device("qwen4exp", Device::Tpu);
        if cfg!(feature = "tpu") {
            assert!(r.is_ok(), "with the feature on: {r:?}");
        } else {
            let msg = format!("{:#}", r.expect_err("no tpu feature"));
            assert!(msg.contains("`tpu` feature"), "{msg}");
            assert!(msg.contains("qwen4exp"), "and names the family: {msg}");
        }
        // The standard devices are unaffected either way.
        assert!(validate_lm_device("qwen4exp", Device::Cpu).is_ok());
        assert!(validate_lm_device("qwen4exp", Device::OpenGl).is_err());
    }

    /// SAM accepted TPU before LMs did, and still does.
    #[test]
    fn sam_still_accepts_tpu_unconditionally() {
        assert!(validate_sam_device("sam", Device::Tpu).is_ok());
    }
}
