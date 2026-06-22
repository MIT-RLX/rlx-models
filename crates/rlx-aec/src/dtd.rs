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

//! Double-talk detector — pause filter adaptation when near-end dominates.

use crate::metrics::signal_power;

#[derive(Debug, Clone)]
pub struct DoubleTalkDetector {
    pub far_min_power: f32,
    /// When mic power exceeds echo estimate by this factor while far-end is active, pause.
    pub near_end_ratio: f32,
}

impl Default for DoubleTalkDetector {
    fn default() -> Self {
        Self {
            far_min_power: 1e-6,
            near_end_ratio: 8.0,
        }
    }
}

impl DoubleTalkDetector {
    /// Returns true when adaptation should be paused.
    pub fn pause_adaptation(
        &self,
        mic_power: f32,
        far_power: f32,
        echo_estimate_power: f32,
    ) -> bool {
        if far_power < self.far_min_power {
            return true;
        }
        // Only pause once echo path is partially modeled; during startup echo_estimate≈0.
        if echo_estimate_power < far_power * 0.01 {
            return false;
        }
        mic_power > self.near_end_ratio * echo_estimate_power.max(1e-8)
    }

    pub fn frame_power(samples: &[f32]) -> f32 {
        signal_power(samples)
    }
}
