// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Device selection for Orpheus synthesis (LM backbone + SNAC decoder backends).

use anyhow::{Result, bail};
use rlx_runtime::{Device, is_available};

/// Resolved execution targets for Orpheus (LM device + SNAC backend).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OrpheusRuntimeDevice {
    /// RLX GGUF backbone device (Metal on Apple Silicon by default).
    pub lm: Device,
    /// When true, SNAC runs on native RLX CoreML ([`Device::Ane`]); LM uses [`Self::lm`].
    pub snac_coreml: bool,
}

impl OrpheusRuntimeDevice {
    pub fn snac_load_options(self) -> crate::decoder::SnacLoadOptions {
        crate::decoder::SnacLoadOptions {
            coreml: self.snac_coreml,
        }
    }
}

fn env_flag(name: &str) -> Option<bool> {
    match std::env::var(name).ok().as_deref() {
        Some("1") | Some("true") | Some("TRUE") => Some(true),
        Some("0") | Some("false") | Some("FALSE") => Some(false),
        _ => None,
    }
}

/// Whether incremental KV LM decode is supported for Orpheus on `device`.
///
/// MLX is opt-in (`ORPHEUS_MLX_KV=1`) — default off until rlx-mlx Orpheus decode is stable.
pub fn lm_kv_decode_supported(device: Device) -> bool {
    match device {
        Device::Mlx => env_flag("ORPHEUS_MLX_KV") == Some(true),
        Device::Metal | Device::Cuda | Device::Rocm | Device::Gpu | Device::Vulkan => true,
        Device::Cpu => env_flag("ORPHEUS_FAST_LM") == Some(true),
        _ => false,
    }
}

/// Best accelerator for Orpheus LM inference.
///
/// Prefers **Metal** on Apple Silicon (on-device prefill + decode).
/// MLX is skipped unless [`lm_kv_decode_supported`] (opt-in via `ORPHEUS_MLX_KV=1`).
pub fn preferred_synth_device() -> Device {
    if is_available(Device::Metal) {
        Device::Metal
    } else if is_available(Device::Mlx) && lm_kv_decode_supported(Device::Mlx) {
        Device::Mlx
    } else if is_available(Device::Cuda) {
        Device::Cuda
    } else if is_available(Device::Rocm) {
        Device::Rocm
    } else if is_available(Device::Gpu) {
        Device::Gpu
    } else if is_available(Device::Vulkan) {
        Device::Vulkan
    } else {
        Device::Cpu
    }
}

/// Device for integration tests; honors `ORPHEUS_SYNTH_DEVICE` then [`preferred_synth_device`].
#[cfg(feature = "llama")]
pub fn synth_device_for_tests() -> Device {
    if let Ok(s) = std::env::var("ORPHEUS_SYNTH_DEVICE") {
        if let Ok(d) = rlx_cli::parse_device(s.trim()) {
            return d;
        }
    }
    preferred_synth_device()
}

#[cfg(feature = "llama")]
pub fn parse_orpheus_device(s: &str) -> Result<OrpheusRuntimeDevice> {
    resolve_orpheus_device(s)
}

#[cfg(feature = "llama")]
pub fn resolve_orpheus_device(s: &str) -> Result<OrpheusRuntimeDevice> {
    let s = s.trim();
    if s.eq_ignore_ascii_case("auto") {
        return Ok(OrpheusRuntimeDevice {
            lm: preferred_synth_device(),
            snac_coreml: false,
        });
    }
    if s.eq_ignore_ascii_case("coreml") {
        let lm = preferred_synth_device();
        if !matches!(lm, Device::Metal | Device::Mlx | Device::Cpu) {
            bail!("coreml SNAC path expects an Apple host with Metal or MLX; got lm={lm:?}");
        }
        return Ok(OrpheusRuntimeDevice {
            lm,
            snac_coreml: true,
        });
    }
    Ok(OrpheusRuntimeDevice {
        lm: rlx_cli::parse_device(s)?,
        snac_coreml: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mlx_kv_off_by_default() {
        assert!(!lm_kv_decode_supported(Device::Mlx));
    }

    #[test]
    fn metal_kv_on_by_default() {
        assert!(lm_kv_decode_supported(Device::Metal));
    }

    #[test]
    fn wgpu_kv_on_by_default() {
        assert!(lm_kv_decode_supported(Device::Gpu));
        assert!(lm_kv_decode_supported(Device::Vulkan));
    }

    #[test]
    fn resolve_coreml_sets_snac_flag() {
        let rt = resolve_orpheus_device("coreml").unwrap();
        assert!(rt.snac_coreml);
    }
}
