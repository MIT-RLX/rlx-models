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

//! Learned-FFT network definition and numerics.
//!
//! Butterfly / Stockham graphs, twiddle parameters and their stability
//! projections, the reference `rustfft` oracle, and the model variants
//! (pruned, matryoshka, Q8, fused, domain-adaptive, unitary) used across
//! training, benchmarking, and ablation.

pub mod butterfly;
pub mod config;
pub mod denoise;
pub mod domain;
pub mod fused;
pub mod matryoshka;
pub mod precision_fft;
pub mod pruned;
pub mod q8;
pub mod reference;
pub mod stockham;
pub mod twiddle;
pub mod twiddle_stability;
pub mod unitary;
pub mod variants;
pub mod weights;
