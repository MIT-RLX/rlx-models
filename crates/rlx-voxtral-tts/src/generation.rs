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

//! Generation options (parity with vLLM-Omni sampling).

#[derive(Debug, Clone, Copy)]
pub struct GenerationConfig {
    pub cfg_alpha: f32,
    pub seed: u64,
    pub max_frames: usize,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            cfg_alpha: crate::tokens::DEFAULT_CFG_ALPHA,
            seed: 42,
            max_frames: 2500,
        }
    }
}
