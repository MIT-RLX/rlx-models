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

//! Tier-1 compile profiles for SAM v1 / SAM2 / SAM3 loaders.

use std::path::Path;

use rlx_flow::CompileProfile;

use rlx_core::flow_bridge::profile_near_weights as load_profile_near_weights;

/// Colocated with safetensors weights (`sam.rlx.toml`).
pub const SAM_PROFILE_FILE: &str = "sam.rlx.toml";

/// Load `sam.rlx.toml` next to weights, or [`CompileProfile::sam_encoder`].
pub fn sam_profile_near_weights(weights: &Path) -> CompileProfile {
    load_profile_near_weights(weights, SAM_PROFILE_FILE, CompileProfile::sam_encoder())
}

/// SAM3 detector graphs — same defaults as SAM encoder unless overridden on disk.
pub fn sam3_profile_near_weights(weights: &Path) -> CompileProfile {
    load_profile_near_weights(weights, SAM_PROFILE_FILE, CompileProfile::sam3())
}

/// SAM2 checkpoint graphs — loads `sam.rlx.toml` next to weights when present.
pub fn sam2_profile_near_weights(weights: &Path) -> CompileProfile {
    load_profile_near_weights(weights, SAM_PROFILE_FILE, CompileProfile::sam2())
}

pub fn sam2_profile_default() -> CompileProfile {
    CompileProfile::sam2()
}

/// Profile search for tests (no weights directory).
pub fn sam_profile_default() -> CompileProfile {
    CompileProfile::sam_encoder()
}

/// Default SAM3 profile (tests / inline builds).
pub fn sam3_profile_default() -> CompileProfile {
    CompileProfile::sam3()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_flow::FusionPolicyKind;

    #[test]
    fn sam_rlx_toml_loads_for_sam3() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/sam.rlx.toml");
        let p = CompileProfile::from_toml_path(&path).unwrap();
        assert_eq!(p.fusion.policy, FusionPolicyKind::Direct);
        assert!(p.passes.dce);
        assert!(p.backend.cpu.unfuse_regions);
    }
}
