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

//! The three refinable representations the paper instantiates Algorithm 1 with.
//!
//! | Module | Space | Refinement | Error bound `U` |
//! |--------|-------|------------|-----------------|
//! | [`grid`] | `L²` | split every cell `2^D`-ways | within-cell variance (exact) |
//! | [`tree`] | `L^∞` | split every leaf's widest side | Lipschitz × leaf radius |
//! | [`fourier`] | `H²` | split every frequency cell eight ways | Sobolev-weighted residual |
//! | [`voxel`] | `L²(Ω) × L²(Ω → [H^S]³)` | split every voxel eight ways | per-voxel least-squares residual |
//!
//! All three satisfy Lemma 3.9's requirement that repeated refinement drives
//! `U → 0`, which is what makes Algorithm 1's inner loop terminate.

pub mod complex;
pub mod fourier;
pub mod grid;
pub mod harmonics;
pub mod tree;
pub mod voxel;

pub use complex::Complex;
pub use fourier::{FourierGrid, sobolev_symbol};
pub use grid::{CellFit, DyadicGrid, unbiased_error_density};
pub use tree::RegressionTree;
pub use voxel::VoxelGrid;
