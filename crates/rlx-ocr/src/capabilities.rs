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

//! Backend capability checks for OCR graphs.
//!
//! Every standard RLX device (`cpu`, `metal`, `mlx`, `cuda`, `rocm`, `gpu`, `vulkan`) is
//! accepted when the matching `rlx-runtime` feature is enabled at build time.

use anyhow::Result;
use rlx_core::validate_standard_device;
pub use rlx_core::{STANDARD_DEVICE_NAMES, STANDARD_DEVICES};
use rlx_runtime::Device;

/// Validate that `device` is in the workspace standard backend set.
pub fn validate_device(device: Device) -> Result<()> {
    validate_standard_device("ocr", device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_core::STANDARD_DEVICES;

    #[test]
    fn all_standard_backends_allowed() {
        for dev in STANDARD_DEVICES {
            validate_device(*dev).unwrap();
        }
    }
}
