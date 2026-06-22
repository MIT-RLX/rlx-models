use anyhow::Result;
use rlx_runtime::{Device, is_available};

/// Map a requested RLX device to the backend used for native inference.
pub fn resolve_codec_device(requested: Device) -> Device {
    if requested == Device::Cpu {
        return Device::Cpu;
    }
    if native_device_ready(requested) {
        return requested;
    }
    if requested == Device::Mlx && native_device_ready(Device::Metal) {
        return Device::Metal;
    }
    if requested == Device::Gpu {
        if native_device_ready(Device::Vulkan) {
            return Device::Vulkan;
        }
        if native_device_ready(Device::Cuda) {
            return Device::Cuda;
        }
    }
    if (requested == Device::Metal || requested == Device::Mlx)
        && native_device_ready(Device::Vulkan)
    {
        return Device::Vulkan;
    }
    Device::Cpu
}

/// Whether an RLX-graph backend (the `Correct` codec = Descript-DAC on rlx) can
/// run on `device`. Unlike [`native_device_ready`] this is about rlx-runtime
/// backends (metal/mlx/wgpu), not Bellard's libnc (cuda/vulkan).
pub fn rlx_device_ready(device: Device) -> bool {
    match device {
        Device::Cpu => true,
        Device::Metal => cfg!(feature = "metal") && is_available(Device::Metal),
        Device::Mlx => cfg!(feature = "mlx") && is_available(Device::Mlx),
        Device::Gpu => cfg!(feature = "gpu") && is_available(Device::Gpu),
        _ => false,
    }
}

/// Device the `Correct` (RLX DAC) codec will actually run on: the requested rlx
/// backend when available, else CPU. No libnc-style cuda/vulkan remapping.
pub fn resolve_rlx_device(requested: Device) -> Device {
    if rlx_device_ready(requested) {
        requested
    } else {
        Device::Cpu
    }
}

pub fn parse_tsac_device(name: &str) -> Result<Device> {
    rlx_cli::parse_standard_device("tsac", name)
}

pub fn native_device_ready(device: Device) -> bool {
    match device {
        Device::Cpu => true,
        Device::Cuda => cfg!(feature = "cuda") && is_available(Device::Cuda),
        Device::Rocm => cfg!(feature = "rocm") && is_available(Device::Rocm),
        Device::Vulkan | Device::Gpu => cfg!(feature = "vulkan") && is_available(Device::Vulkan),
        Device::Metal => cfg!(feature = "metal") && is_available(Device::Metal),
        Device::Mlx => cfg!(feature = "mlx") && is_available(Device::Mlx),
        _ => false,
    }
}

pub fn device_ready(device: Device) -> bool {
    if device == Device::Cpu {
        return true;
    }
    if native_device_ready(device) {
        return true;
    }
    if device == Device::Mlx && native_device_ready(Device::Metal) {
        return true;
    }
    if device == Device::Gpu
        && (native_device_ready(Device::Vulkan) || native_device_ready(Device::Cuda))
    {
        return true;
    }
    external_binary_ready(device)
}

fn external_binary_ready(device: Device) -> bool {
    crate::platform::tsac_binary_supported() && matches!(device, Device::Cpu | Device::Cuda)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_always_resolves() {
        assert_eq!(resolve_codec_device(Device::Cpu), Device::Cpu);
    }
}
