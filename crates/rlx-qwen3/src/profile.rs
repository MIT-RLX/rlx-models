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

//! Tier-1 compile profiles for Qwen3 generators.

use std::path::Path;

use rlx_flow::CompileProfile;

use rlx_core::flow_bridge::profile_near_weights;

/// Colocated with weights (`qwen3.rlx.toml`).
pub const QWEN3_PROFILE_FILE: &str = "qwen3.rlx.toml";

/// Load `qwen3.rlx.toml` next to weights, or use built-in prefill/decode defaults.
pub fn qwen3_profile_near_weights(weights: &Path, decode: bool) -> CompileProfile {
    let default = if decode {
        CompileProfile::qwen3_decode()
    } else {
        CompileProfile::qwen3_prefill()
    };
    profile_near_weights(weights, QWEN3_PROFILE_FILE, default)
}

/// Profile search path for inline / test builds (cwd).
pub fn qwen3_profile_default(decode: bool) -> CompileProfile {
    if decode {
        CompileProfile::qwen3_decode()
    } else {
        CompileProfile::qwen3_prefill()
    }
}
