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

//! Earshot-style VAD (mel frontend + tiny CNN + MinGRU), RLX CPU inference.
//!
//! Architecture reference: <https://github.com/pykeio/earshot>

mod detector;
mod fft;
mod filters;
mod predictor;
mod weights;

pub use detector::Detector;

pub const FRAME_SAMPLES: usize = 256;
pub const SAMPLE_RATE: usize = 16_000;
