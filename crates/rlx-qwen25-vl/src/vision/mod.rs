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

//! Qwen2.5-VL mmproj — config, weights, preprocess, encoder.

mod builder;
mod config;
mod encoder;
mod preprocess;
mod weights;

pub use builder::build_qwen25_vl_vision_built;
pub use config::MmProjConfig;
pub use encoder::{Qwen25VlVisionEncoder, VisionEncodeOutput, load_vision_encoder};
pub use weights::{MmProjWeights, load_vision_weights};

#[cfg(feature = "qwen25-vl-vision")]
pub use preprocess::load_rgb_image;
