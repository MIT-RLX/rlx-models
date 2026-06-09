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

use crate::config::LLaDA2MoeConfig;
use anyhow::Result;
use rlx_core::validate_standard_device;
use rlx_runtime::Device;

/// Supported execution devices (standard RLX backends).
pub fn validate_device(cfg: &LLaDA2MoeConfig, device: Device) -> Result<()> {
    let _ = cfg;
    validate_standard_device("llada2", device)
}

/// VRAM / unified-memory budget hint for MoE offload sizing.
pub fn default_memory_budget_bytes(device: Device) -> Option<usize> {
    rlx_core::device_memory_for_moe_offload(device).map(|(free, _)| free)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_core::STANDARD_DEVICES;

    #[test]
    fn all_standard_backends_supported() {
        let cfg = crate::synth::tiny_cfg();
        for dev in STANDARD_DEVICES {
            validate_device(&cfg, *dev).expect("supported");
        }
    }
}
