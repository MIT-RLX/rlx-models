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

//! Training device selection (`--device auto|cpu|metal|…`).

use anyhow::{Context, Result, ensure};
use rlx_cli::parse_device;
use rlx_core::STANDARD_DEVICE_NAMES;
use rlx_runtime::{Device, is_available};
use std::env;

const FAMILY: &str = "rlx-fft";

/// CLI aliases → names accepted by [`parse_device`].
pub fn normalize_device_alias(name: &str) -> String {
    match name.trim().to_ascii_lowercase().as_str() {
        "" => String::new(),
        "wgu" | "wgpu" => "wgpu".into(),
        "mps" => "metal".into(),
        "hip" => "rocm".into(),
        // CoreML / Apple Neural Engine: `parse_device` accepts `ane`/`neural-engine`
        // (Device::Ane); map the user-facing `coreml` alias onto it too.
        "coreml" | "neural-engine" | "neural_engine" => "ane".into(),
        other => other.into(),
    }
}

/// Stable label stored in bench JSON / HTML (`gpu` → `wgpu`).
pub fn bench_device_label(device: Device) -> String {
    match device {
        Device::Gpu => "wgpu".into(),
        other => format!("{other:?}").to_lowercase(),
    }
}

/// Parse `--device cpu,metal,mlx,wgpu` or `all` / `apple-silicon`.
pub fn parse_bench_device_list(csv: &str) -> Result<Vec<String>> {
    let csv = csv.trim();
    if csv.eq_ignore_ascii_case("all") {
        return Ok(crate::bench_sweep::available_devices()
            .into_iter()
            .map(str::to_string)
            .collect());
    }
    if csv.eq_ignore_ascii_case("apple-silicon") {
        let mut out = vec!["cpu".to_string()];
        for name in ["metal", "mlx", "wgpu", "ane"] {
            if let Ok(dev) = parse_device(name) {
                if is_available(dev) {
                    out.push(bench_device_label(dev));
                }
            }
        }
        out.sort();
        out.dedup();
        ensure!(!out.is_empty(), "no devices selected");
        return Ok(out);
    }

    let mut out = Vec::new();
    for part in csv.split(',') {
        let norm = normalize_device_alias(part);
        if norm.is_empty() {
            continue;
        }
        let dev = parse_device(&norm).with_context(|| {
            format!("parse device {norm} ({STANDARD_DEVICE_NAMES}|all|apple-silicon)")
        })?;
        ensure_backend_ready(dev)?;
        let label = bench_device_label(dev);
        if !out.iter().any(|d| d == &label) {
            out.push(label);
        }
    }
    ensure!(!out.is_empty(), "no devices selected");
    Ok(out)
}

pub fn resolve_train_device(requested: Option<&str>) -> Result<Device> {
    let name = requested
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(normalize_device_alias)
        .or_else(|| {
            env::var("RLX_DEVICE")
                .ok()
                .map(|s| normalize_device_alias(&s))
        })
        .unwrap_or_else(|| "auto".to_string());

    if name.eq_ignore_ascii_case("auto") {
        let device = pick_auto_device();
        ensure_backend_ready(device)?;
        return Ok(device);
    }

    let device = parse_device(&name)
        .with_context(|| format!("parse --device {name} ({STANDARD_DEVICE_NAMES}|auto)"))?;
    ensure_backend_ready(device)?;
    Ok(device)
}

pub fn pick_auto_device() -> Device {
    for device in [
        Device::Cuda,
        Device::Metal,
        Device::Mlx,
        Device::Rocm,
        Device::Gpu,
        Device::Vulkan,
    ] {
        if is_available(device) {
            return device;
        }
    }
    Device::Cpu
}

pub fn ensure_backend_ready(device: Device) -> Result<()> {
    if device == Device::Cpu {
        return Ok(());
    }
    ensure!(
        is_available(device),
        "{FAMILY}: {device:?} is not available — build with the matching feature or pass `--device cpu`"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_wgu_alias() {
        assert_eq!(normalize_device_alias("wgu"), "wgpu");
        assert_eq!(normalize_device_alias("WGPU"), "wgpu");
    }

    #[test]
    fn normalize_coreml_alias() {
        // CoreML / ANE aliases all resolve to the `ane` name parse_device accepts.
        assert_eq!(normalize_device_alias("coreml"), "ane");
        assert_eq!(normalize_device_alias("CoreML"), "ane");
        assert_eq!(normalize_device_alias("neural-engine"), "ane");
        assert_eq!(normalize_device_alias("ane"), "ane");
        assert!(parse_device(&normalize_device_alias("coreml")).is_ok());
    }

    #[test]
    fn bench_device_label_wgpu() {
        assert_eq!(bench_device_label(Device::Gpu), "wgpu");
    }

    #[test]
    fn parse_bench_list_dedupes_gpu_aliases() {
        if !is_available(Device::Gpu) {
            return;
        }
        let list = parse_bench_device_list("cpu,wgpu,wgu").unwrap();
        assert!(list.contains(&"cpu".to_string()));
        assert!(list.contains(&"wgpu".to_string()));
        assert_eq!(list.iter().filter(|d| *d == "wgpu").count(), 1);
    }
}
