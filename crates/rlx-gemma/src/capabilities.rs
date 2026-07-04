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

//! Backend capability checks for Gemma.2.

use crate::config::GemmaConfig;
use anyhow::Result;
use rlx_core::validate_lm_device;
pub use rlx_core::{LM_DEVICE_NAMES, STANDARD_DEVICE_NAMES, STANDARD_DEVICES, pick_lm_device};
use rlx_runtime::Device;

pub fn validate_device(cfg: &GemmaConfig, device: Device, packed_weights: bool) -> Result<()> {
    let _ = (cfg, packed_weights);
    validate_lm_device("gemma", device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_core::STANDARD_DEVICES;

    fn tiny_cfg() -> GemmaConfig {
        GemmaConfig::tiny_test()
    }

    #[test]
    fn all_standard_backends_allowed() {
        let cfg = tiny_cfg();
        for dev in STANDARD_DEVICES {
            validate_device(&cfg, *dev, false).unwrap();
            validate_device(&cfg, *dev, true).unwrap();
        }
    }

    #[cfg(feature = "coreml")]
    #[test]
    fn coreml_ane_allowed() {
        let cfg = tiny_cfg();
        validate_device(&cfg, Device::Ane, true).unwrap();
    }
}
