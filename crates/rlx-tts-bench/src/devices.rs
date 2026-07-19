//! Device selection helpers (same filter pattern as per-crate matrices).

use anyhow::{Result, bail};
use rlx_runtime::{Device, is_available};

pub fn parse_device_list(spec: &str) -> Result<Vec<Device>> {
    let s = spec.trim().to_ascii_lowercase();
    if s.is_empty() || s == "auto" {
        return Ok(auto_devices());
    }
    let mut out = Vec::new();
    for part in s.split(',') {
        let d = match part.trim() {
            "cpu" => Device::Cpu,
            "metal" => Device::Metal,
            "mlx" => Device::Mlx,
            "gpu" | "wgpu" => Device::Gpu,
            "cuda" => Device::Cuda,
            "ane" | "coreml" => Device::Ane,
            "vulkan" => Device::Vulkan,
            other => bail!("unknown device '{other}'"),
        };
        out.push(d);
    }
    Ok(out)
}

pub fn auto_devices() -> Vec<Device> {
    let candidates = [
        Device::Cpu,
        Device::Metal,
        Device::Mlx,
        Device::Gpu,
        Device::Cuda,
        Device::Ane,
    ];
    candidates
        .into_iter()
        .filter(|d| *d == Device::Cpu || is_available(*d))
        .collect()
}

pub fn device_label(d: Device) -> &'static str {
    match d {
        Device::Cpu => "cpu",
        Device::Metal => "metal",
        Device::Mlx => "mlx",
        Device::Gpu => "gpu",
        Device::Cuda => "cuda",
        Device::Ane => "ane",
        Device::Vulkan => "vulkan",
        _ => "other",
    }
}

pub fn filter_available(devices: &[Device]) -> Vec<Device> {
    devices
        .iter()
        .copied()
        .filter(|d| *d == Device::Cpu || is_available(*d))
        .collect()
}
