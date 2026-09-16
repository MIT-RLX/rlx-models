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

//! Shared helpers for the parity suites.

use rlx_runtime::Device;
use std::str::FromStr;

/// Device the parity tests run on, from `RLX_TADA_TEST_DEVICE` (default `cpu`).
///
/// The fixtures pin each component against upstream torch; pointing them at a
/// different backend turns the same suite into a cross-backend check, which is
/// how a backend that quietly computes something else gets caught at the
/// component that does it rather than at the waveform.
pub fn device() -> Device {
    match std::env::var("RLX_TADA_TEST_DEVICE") {
        Ok(name) if !name.is_empty() => {
            Device::from_str(&name).unwrap_or_else(|e| panic!("bad RLX_TADA_TEST_DEVICE: {e}"))
        }
        _ => Device::Cpu,
    }
}
