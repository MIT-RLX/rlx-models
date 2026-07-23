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

//! Offline-decomposed native HIR builders (no runtime ONNX import).
//!
//! Each `{tiny,small}_{det,rec}` module exposes `load_weights` + `build_hir`.
//! Regenerate with `scripts/ppocrv6_emit_native.py` after emit/decompose changes.

pub mod small_det;
pub mod small_rec;
pub mod tiny_det;
pub mod tiny_rec;
