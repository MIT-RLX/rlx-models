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

//! Higher-Order Cell Tracking Transformer (HOCT) for RLX.
//!
//! Port of [royerlab/hoct](https://github.com/royerlab/hoct) / [arXiv:2607.11754](https://arxiv.org/abs/2607.11754):
//! label volumes → regionprops → candidate graph → edge transformer → parental
//! softmax → ILP → CTC / GEFF export.
//!
//! # Quick start
//!
//! ```ignore
//! use rlx_hoct::{HoctRunner, OutputFormat};
//!
//! let runner = HoctRunner::builder()
//!     .weights("general_v0.safetensors")
//!     .build()?;
//! let (sol, nodes) = runner.track_labels(&labels, None)?;
//! rlx_hoct::write_solution("out/", &labels, &nodes, &sol, OutputFormat::Ctc)?;
//! ```
//!
//! # Modules
//!
//! - [`model`] — eager [`HoctModel`] (TorchScript parity reference)
//! - [`device`] — [`HoctDeviceRunner`]: eager body + compiled score head
//! - [`runner`] — end-to-end [`HoctRunner`]
//! - [`features`] / [`graph`] / [`softmax`] / [`ilp`] / [`io`] — pipeline stages
//! - [`flow`] — padded compile-shaped API + edge-head [`ModelFlow`](rlx_flow::ModelFlow)
//!
//! Weights: `just fetch-hoct` or `crates/rlx-hoct/scripts/export_jit_safetensors.py`.

pub mod attn;
pub mod builder;
pub mod cli;
pub mod config;
pub mod dataset;
pub mod device;
pub mod features;
pub mod flow;
pub mod geometry;
pub mod graph;
pub mod ilp;
pub mod io;
pub mod model;
pub mod rope3d;
pub mod runner;
pub mod softmax;
pub mod weights;

pub use builder::build_hoct_eager;
pub use config::{FEATURE_MEAN, FEATURE_STD, GraphConfig, HoctConfig, IlpWeights};
pub use device::{HoctDeviceRunner, device_label, parity_backends};
pub use flow::{HoctCompiled, HoctFlow};
pub use io::{OutputFormat, assign_tracklets, write_ctc, write_geff_minimal, write_solution};
pub use model::{HoctModel, HoctOutput};
pub use runner::{HoctRunner, HoctRunnerBuilder};
pub use weights::load_hoct_weights;
