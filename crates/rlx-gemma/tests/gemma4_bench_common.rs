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

//! Shared device selection for Gemma 4 integration benches / sweeps.

// Each test binary that includes this module uses only a subset of the helpers.
#![allow(dead_code)]

use anyhow::{Context, Result};
use rlx_gemma::config::{GemmaConfig, gemma_cfg_from_gguf};
use rlx_gguf::GgufFile;
use rlx_runtime::{Device, device_ext::is_available};
use std::path::{Path, PathBuf};

/// Known Gemma 4 12B GGUF filenames (base IT + coder fine-tunes).
pub const GEMMA4_GGUF_CANDIDATES: &[&str] = &[
    "gemma-4-12b-it-Q4_K_M.gguf",
    "gemma4-coding-Q4_K_M.gguf",
    "model.gguf",
];

/// Pick the first GGUF under `dir` from [`GEMMA4_GGUF_CANDIDATES`], or any `*.gguf`.
pub fn resolve_gemma4_gguf(dir: &Path) -> Option<PathBuf> {
    for name in GEMMA4_GGUF_CANDIDATES {
        let p = dir.join(name);
        if p.is_file() {
            return Some(p);
        }
    }
    std::fs::read_dir(dir)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .find(|p| {
            p.extension()
                .and_then(|s| s.to_str())
                .is_some_and(|s| s.eq_ignore_ascii_case("gguf"))
        })
}

/// Load config from sibling `config.json` or embedded GGUF metadata.
pub fn resolve_gemma4_config(dir: &Path, gguf: &Path) -> Result<GemmaConfig> {
    let config_path = dir.join("config.json");
    if config_path.is_file() {
        return GemmaConfig::from_file(&config_path);
    }
    let raw = GgufFile::from_path(gguf).with_context(|| format!("open GGUF {gguf:?}"))?;
    gemma_cfg_from_gguf(&raw)
}

/// Single device from `RLX_GEMMA4_DEVICE` (`mlx` | `metal` | `cpu`), else Metal on macOS.
pub fn bench_device_from_env() -> Device {
    if let Ok(raw) = std::env::var("RLX_GEMMA4_DEVICE") {
        match raw.to_ascii_lowercase().as_str() {
            "mlx"
                if cfg!(all(target_os = "macos", feature = "mlx")) && is_available(Device::Mlx) =>
            {
                return Device::Mlx;
            }
            "metal"
                if cfg!(all(target_os = "macos", feature = "metal"))
                    && is_available(Device::Metal) =>
            {
                return Device::Metal;
            }
            "cpu" => return Device::Cpu,
            _ => {}
        }
    }
    #[cfg(all(target_os = "macos", feature = "metal"))]
    if is_available(Device::Metal) {
        return Device::Metal;
    }
    Device::Cpu
}

/// Devices to run when `RLX_GEMMA4_DEVICE` is unset: Metal then MLX on macOS, else CPU.
#[allow(dead_code)]
pub fn bench_devices_from_env() -> Vec<Device> {
    if std::env::var("RLX_GEMMA4_DEVICE").is_ok() {
        return vec![bench_device_from_env()];
    }
    let mut out = Vec::new();
    #[cfg(all(target_os = "macos", feature = "metal"))]
    if is_available(Device::Metal) {
        out.push(Device::Metal);
    }
    #[cfg(all(target_os = "macos", feature = "mlx"))]
    if is_available(Device::Mlx) {
        out.push(Device::Mlx);
    }
    if out.is_empty() {
        out.push(Device::Cpu);
    }
    out
}
