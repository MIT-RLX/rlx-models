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

use anyhow::Result;
use rlx_core::{validate_sam_device, validate_standard_device};
use rlx_runtime::Device;
use std::str::FromStr;

/// Parse a device alias. Delegates to the upstream `FromStr for Device`
/// impl in `rlx-driver`, which knows every alias (`mps`, `wgpu`, `hip`,
/// …) and returns a uniform error string. Kept as a wrapper that
/// surfaces `anyhow::Result` so existing callers don't change.
pub fn parse_device(s: &str) -> Result<Device> {
    Device::from_str(s).map_err(|e| anyhow::anyhow!("{e}"))
}

/// Parse a CLI device name and enforce the workspace-wide standard backend set.
pub fn parse_standard_device(family: &str, s: &str) -> Result<Device> {
    let d = parse_device(s)?;
    validate_standard_device(family, d)?;
    Ok(d)
}

pub fn parse_llama32_device(s: &str) -> Result<Device> {
    parse_standard_device("llama32", s)
}

pub fn parse_gemma_device(s: &str) -> Result<Device> {
    parse_standard_device("gemma", s)
}

pub fn parse_qwen35_device(s: &str) -> Result<Device> {
    parse_standard_device("qwen35", s)
}

pub fn parse_llada2_device(s: &str) -> Result<Device> {
    parse_standard_device("llada2", s)
}

/// Parse a CLI device name and enforce the SAM backend set (standard + `tpu`).
pub fn parse_sam_device(family: &str, s: &str) -> Result<Device> {
    let d = match s {
        "tpu" => Device::Tpu,
        other => parse_device(other)?,
    };
    validate_sam_device(family, d)?;
    Ok(d)
}
