//! Device parsing and per-build device availability for Kyutai TTS.

use anyhow::Result;
use rlx_runtime::{Device, is_available};

/// Parse `--device` for Kyutai TTS (cpu / metal / cuda / mlx / auto / …).
pub fn parse_kyutai_tts_device(name: &str) -> Result<Device> {
    rlx_cli::parse_standard_device("kyutai-tts", name)
}

/// Devices to exercise in multi-backend tests (env `RLX_KYUTAI_TTS_DEVICES`).
pub fn test_devices() -> Vec<Device> {
    let raw = std::env::var("RLX_KYUTAI_TTS_DEVICES").unwrap_or_else(|_| "cpu".into());
    if raw.eq_ignore_ascii_case("all") {
        return all_compiled_devices();
    }
    raw.split(',')
        .filter_map(|s| {
            let s = s.trim();
            if s.is_empty() {
                None
            } else {
                parse_kyutai_tts_device(s).ok()
            }
        })
        .collect()
}

fn all_compiled_devices() -> Vec<Device> {
    #[allow(unused_mut)]
    let mut out = vec![Device::Cpu];
    #[cfg(feature = "metal")]
    if is_available(Device::Metal) {
        out.push(Device::Metal);
    }
    #[cfg(feature = "mlx")]
    if is_available(Device::Mlx) {
        out.push(Device::Mlx);
    }
    #[cfg(feature = "cuda")]
    if is_available(Device::Cuda) {
        out.push(Device::Cuda);
    }
    #[cfg(feature = "vulkan")]
    if is_available(Device::Vulkan) {
        out.push(Device::Vulkan);
    }
    #[cfg(feature = "rocm")]
    if is_available(Device::Rocm) {
        out.push(Device::Rocm);
    }
    out
}

/// Skip GPU devices that were requested but are not compiled/available.
pub fn device_ready(device: Device) -> bool {
    device == Device::Cpu || is_available(device)
}
