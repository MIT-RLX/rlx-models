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

//! Decompose ONNX models into generated RLX Rust crates + external weights.
//!
//! ```no_run
//! use std::path::Path;
//! use rlx_onnx_decompose::{decompose, DecomposeOptions, WeightsFormat};
//!
//! decompose(
//!     Path::new("model.onnx"),
//!     Path::new("out/my_model_rlx"),
//!     &DecomposeOptions {
//!         weights_format: WeightsFormat::Safetensors,
//!         ..Default::default()
//!     },
//! )?;
//! # Ok::<(), anyhow::Error>(())
//! ```

mod emit;
mod plan;
mod weights;

pub use plan::sanitize_crate_name;
pub use plan::{
    DecomposeOptions, DecomposePlan, WeightsFormat, decompose, decompose_bundle, default_rlx_root,
    resolve_rlx_root,
};
