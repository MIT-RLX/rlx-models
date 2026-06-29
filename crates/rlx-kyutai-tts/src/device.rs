//! Device parsing and per-build device availability for Kyutai TTS.

use anyhow::Result;
use rlx_runtime::{Device, is_available};

/// Parse `--device` for Kyutai TTS (cpu / metal / cuda / mlx / auto / …).
pub fn parse_kyutai_tts_device(name: &str) -> Result<Device> {
    rlx_cli::parse_standard_device("kyutai-tts", name)
}

/// Best accelerator for Kyutai TTS on this host (Metal → CUDA → ROCm → wgpu → Vulkan → MLX).
pub fn preferred_kyutai_device() -> Device {
    for dev in [
        Device::Metal,
        Device::Cuda,
        Device::Rocm,
        Device::Gpu,
        Device::Vulkan,
        Device::Mlx,
    ] {
        if is_available(dev) {
            return dev;
        }
    }
    Device::Cpu
}

/// Resolve a CLI / env device string (`auto` picks [`preferred_kyutai_device`]).
pub fn resolve_kyutai_tts_device(name: &str) -> Result<Device> {
    let name = name.trim();
    if name.eq_ignore_ascii_case("auto") {
        let d = preferred_kyutai_device();
        if d != Device::Cpu {
            eprintln!("kyutai-tts: auto → {d:?}");
        }
        return Ok(d);
    }
    let d = parse_kyutai_tts_device(name)?;
    Ok(resolve_lm_device(d))
}

/// Honour availability; fall back to CPU when the requested backend is missing.
pub fn resolve_lm_device(requested: Device) -> Device {
    if requested == Device::Cpu {
        return Device::Cpu;
    }
    if is_available(requested) {
        requested
    } else {
        eprintln!("kyutai-tts: {requested:?} unavailable — using CPU");
        Device::Cpu
    }
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
                resolve_kyutai_tts_device(s).ok()
            }
        })
        .collect()
}

fn all_compiled_devices() -> Vec<Device> {
    let candidates = [
        Device::Metal,
        Device::Mlx,
        Device::Cuda,
        Device::Rocm,
        Device::Gpu,
        Device::Vulkan,
        Device::Ane,
    ];
    let mut out = vec![Device::Cpu];
    for dev in candidates {
        if is_available(dev) {
            out.push(dev);
        }
    }
    out
}

/// True when the requested device can run (CPU always; GPU when compiled in).
pub fn device_ready(device: Device) -> bool {
    device == Device::Cpu || is_available(device)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_resolves_to_cpu_when_no_gpu() {
        // When no GPU backend is linked, preferred is CPU — should not error.
        let d = resolve_kyutai_tts_device("auto").unwrap();
        assert!(device_ready(d));
    }

    #[test]
    fn explicit_cpu_stays_cpu() {
        assert_eq!(resolve_kyutai_tts_device("cpu").unwrap(), Device::Cpu);
    }
}
