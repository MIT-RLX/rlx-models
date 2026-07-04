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

//! Distilled FFT students — float and ternary — plus their compiled deploys.
//!
//! Includes the ternary architecture / gate definitions and the banded
//! wide-sparse correction applied after the pruned butterfly.

pub mod band_correct;
pub mod compile;
pub mod fused;
pub mod model;
pub mod ternary_arch;
pub mod ternary_compile;
pub mod ternary_gates;
pub mod ternary_model;
