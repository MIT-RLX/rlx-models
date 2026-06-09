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

//! RLX-linked NeuCodec decode path.
//!
//! Until a compiled RLX graph exists, this module **delegates to the eager ndarray
//! forward pass** so `feature = "rlx"` builds stay bit-identical to the baseline
//! (parity). When the graph lands, swap the body of [`decode`] only.

use anyhow::Result;

use super::eager::{DecoderWeights, decode_forward};

/// Decode speech token IDs using the RLX stack.
///
/// Today: proves `rlx` links and returns the same audio as [`decode_forward`].
pub fn decode(codes: &[i32], weights: &DecoderWeights) -> Result<Vec<f32>> {
    // Touch RLX so the feature actually pulls the runtime (parity with other rlx-models crates).
    let _device = rlx::Device::Cpu;
    Ok(decode_forward(codes, weights))
}
