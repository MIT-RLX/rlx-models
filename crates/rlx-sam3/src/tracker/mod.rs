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

//! SAM3 video tracker — weight extraction scaffolding.
//!
//! Full forward parity for the video memory bank + propagated mask decoder
//! is a separate effort matching the upstream
//! `sam3.model.sam3_video_inference.SAM3VideoPredictor` pipeline. This
//! module currently:
//!
//!   * Walks the tracker.* prefix in the checkpoint and consumes the
//!     weights so they don't appear as "unexpected" during load. Each
//!     tensor is kept in `Sam3TrackerWeights::raw` keyed by short suffix,
//!     ready for the per-module forward implementations to slot in.
//!   * Exposes a `tracker_forward_native` shim that mirrors the previous
//!     stub behaviour (per-frame mask carry-over) so callers don't crash
//!     until the real propagation lands.

use crate::sam3::{Sam3ImagePrediction, Sam3VideoFramePrediction, Sam3VideoState};
use anyhow::Result;
use rlx_core::weight_map::WeightMap;
use std::collections::HashMap;

#[derive(Debug, Clone, Default)]
pub struct Sam3TrackerWeights {
    pub loaded: bool,
    /// All `tracker.*` tensors keyed by the suffix after the `tracker.`
    /// prefix. The per-module forwards (when wired) read from this map.
    pub raw: HashMap<String, (Vec<f32>, Vec<usize>)>,
}

pub fn extract_tracker_weights(weights: &mut WeightMap) -> Result<Sam3TrackerWeights> {
    // Collect all tracker.* keys, then take them out of the map so the
    // sam3 model load reports a clean "unexpected=0" diagnostic.
    let prefixes = ["tracker.", "detector.tracker."];
    let mut owned = HashMap::new();
    let mut to_take: Vec<String> = Vec::new();
    for key in weights.keys() {
        for p in prefixes {
            if let Some(suffix) = key.strip_prefix(p) {
                to_take.push(key.to_string());
                let _ = suffix;
                break;
            }
        }
    }
    for full in to_take {
        let suffix = full
            .strip_prefix("detector.tracker.")
            .or_else(|| full.strip_prefix("tracker."))
            .unwrap()
            .to_string();
        if let Ok(tensor) = weights.take(&full) {
            owned.insert(suffix, tensor);
        }
    }
    Ok(Sam3TrackerWeights {
        loaded: !owned.is_empty(),
        raw: owned,
    })
}

pub fn tracker_forward_native(
    _weights: &Sam3TrackerWeights,
    state: &mut Sam3VideoState,
    image: Sam3ImagePrediction,
) -> Sam3VideoFramePrediction {
    let frame_index = state.frame_index;
    state.frame_index += 1;
    state.last_prediction = Some(image.clone());
    state.memory_tokens.push(image.scores.clone());
    Sam3VideoFramePrediction {
        frame_index,
        image,
        memory_len: state.memory_tokens.len(),
    }
}
