use anyhow::Result;
use rlx_runtime::{Device, is_available};

/// Resolve codec execution device (may differ from the requested tag).
pub fn resolve_codec_device(requested: Device) -> Device {
    if requested == Device::Cpu {
        return Device::Cpu;
    }
    Device::Cpu
}

pub fn gpu_codec_available(_device: Device) -> bool {
    false
}

pub fn parse_mimi_device(name: &str) -> Result<Device> {
    rlx_cli::parse_standard_device("mimi", name)
}

pub fn test_devices() -> Vec<Device> {
    let raw = rlx_ir::env::var_result("RLX_MIMI_DEVICES").unwrap_or_else(|_| "cpu".into());
    if raw.eq_ignore_ascii_case("all") {
        return all_compiled_devices();
    }
    raw.split(',')
        .filter_map(|s| {
            let s = s.trim();
            if s.is_empty() {
                None
            } else {
                parse_mimi_device(s).ok()
            }
        })
        .collect()
}

fn all_compiled_devices() -> Vec<Device> {
    // `mut` only needed when a GPU backend feature compiles in push sites below.
    #[cfg(any(
        feature = "metal",
        feature = "mlx",
        feature = "cuda",
        feature = "vulkan"
    ))]
    let mut out = vec![Device::Cpu];
    #[cfg(not(any(
        feature = "metal",
        feature = "mlx",
        feature = "cuda",
        feature = "vulkan"
    )))]
    let out = vec![Device::Cpu];
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
    out
}

pub fn device_ready(device: Device) -> bool {
    device == Device::Cpu || is_available(device)
}
