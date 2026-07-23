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

//! Backend capability checks for PP-OCRv6 graphs.

use anyhow::Result;
use rlx_core::validate_lm_device;
pub use rlx_core::{STANDARD_DEVICE_NAMES, STANDARD_DEVICES};
use rlx_runtime::Device;

pub fn validate_device(device: Device) -> Result<()> {
    // Allow CoreML / ANE when built with `coreml` (same as causal-LM crates).
    validate_lm_device("ppocrv6", device)
}

/// Prefer Neural Engine for CoreML OCR unless the user already set units.
///
/// Upstream defaults fp32 graphs to `CpuAndGpu`. GPU MIL execution corrupts the
/// small-tier SVTR Softmax path (`Hello OCR` → garbage). ANE and CPU match
/// Metal/CPU reference.
pub fn configure_coreml_for_ocr(device: Device) {
    if !matches!(device, Device::Ane) {
        return;
    }
    if std::env::var_os("RLX_COREML_UNITS").is_some() {
        return;
    }
    // Process-wide toggle read by `rlx-coreml` at compile time; set once before
    // Session::compile. Safe here: OCR engine construction is single-threaded.
    unsafe {
        std::env::set_var("RLX_COREML_UNITS", "ane");
    }
}
