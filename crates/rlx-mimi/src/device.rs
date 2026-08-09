use anyhow::Result;
use rlx_runtime::{Device, is_available};

/// Resolve codec execution device (may differ from the requested tag).
pub fn resolve_codec_device(requested: Device) -> Device {
    if requested == Device::Cpu {
        return Device::Cpu;
    }
    #[cfg(feature = "parity-mimi")]
    {
        if candle_codec_available(requested) {
            return requested;
        }
        // Apple MLX tag → Candle Metal kernels on the same GPU.
        if requested == Device::Mlx && candle_codec_available(Device::Metal) {
            return Device::Metal;
        }
    }
    Device::Cpu
}

#[cfg(feature = "parity-mimi")]
pub fn candle_codec_available(device: Device) -> bool {
    match device {
        Device::Metal => candle::utils::metal_is_available(),
        Device::Cuda => candle::utils::cuda_is_available(),
        Device::Mlx
        | Device::Cpu
        | Device::Gpu
        | Device::Vulkan
        | Device::Rocm
        | Device::Ane
        | Device::Tpu
        | Device::OpenGl
        | Device::DirectX
        | Device::Xdna
        | Device::OneApi
        | Device::Hexagon
        | Device::WebGpu => false,
    }
}

#[cfg(not(feature = "parity-mimi"))]
pub fn candle_codec_available(_device: Device) -> bool {
    false
}

pub fn parse_mimi_device(name: &str) -> Result<Device> {
    rlx_cli::parse_standard_device("mimi", name)
}

pub fn test_devices() -> Vec<Device> {
    let raw = std::env::var("RLX_MIMI_DEVICES").unwrap_or_else(|_| "cpu".into());
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
    if device == Device::Cpu {
        return true;
    }
    #[cfg(feature = "parity-mimi")]
    {
        if candle_codec_available(device) {
            return true;
        }
        if device == Device::Mlx && candle_codec_available(Device::Metal) {
            return true;
        }
    }
    device == Device::Cpu || is_available(device)
}
