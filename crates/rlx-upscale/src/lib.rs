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

//! Single-image super-resolution on RLX.

pub mod arch;
pub mod cli;
pub mod config;
pub mod detect;
pub mod detect_swin;
pub mod device;
#[cfg(feature = "export")]
pub mod export;
pub mod graph;
pub mod nn;
pub mod runner;
pub mod sample;
pub mod tile;
pub mod weights;

pub use config::{Arch, ArchParams, ModelConfig};
pub use device::parse_upscale_device;
pub use runner::{AlphaMode, UpscaleOptions, Upscaler};
pub use weights::Checkpoint;
