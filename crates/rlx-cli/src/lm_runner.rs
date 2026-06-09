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

//! Re-export of the upstream `rlx_runtime::LmRunner` trait.
//!
//! Historically this module defined `LmRunner` itself; the trait moved
//! into `rlx-runtime` so model crates can implement it without taking a
//! dependency on this CLI helper crate (which would otherwise pull in
//! `clap`, `phf`, `minijinja`, etc.). The re-export keeps the existing
//! `rlx_cli::LmRunner` path working for downstream callers.

pub use rlx_runtime::LmRunner;
