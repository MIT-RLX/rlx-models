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

//! The functionals the paper instantiates Algorithm 1 on.
//!
//! | Module | Paper | `H` | `B` | Representation |
//! |--------|-------|-----|-----|----------------|
//! | [`fit`] | Figure 1, A.1.1 | `L²([0,1]²)` | `L²` | dyadic grid |
//! | [`rkhs`] | Section 4.1, A.1.2 | RKHS `H_K` | `L^∞` | decision tree |
//! | [`wave`] | Section 4.2, A.1.3 | `H²(R³)` | `H²` | frequency grid |
//! | [`radiance`] | Section 4.3, A.1.4 | `L²(Ω) × L²(Ω → [H^S]³)` | same | voxel grid + harmonics |

pub mod fit;
pub mod radiance;
pub mod rkhs;
pub mod wave;
