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

//! Kraken — the HTR engine of eScriptorium and HTR-United — natively in RLX.
//!
//! A Kraken recognition model is a VGSL network (convolutions, max-pools,
//! a reshape, bidirectional LSTMs and a CTC output) serialized as a Core ML
//! `.mlmodel`. The model zoo (Zenodo / HTRMoPo) covers medieval and modern
//! French, Latin, and many other scripts — e.g. CATMuS Medieval
//! (`10.5281/zenodo.12743230`).
//!
//! - [`mlmodel::MlModel`] — the weights and metadata, read off the protobuf.
//! - [`vgsl::Vgsl`] — the network description.
//! - [`graph::build`] — the network as one RLX graph per line size.
//! - [`KrakenRunner`] — line image → text, with [`KrakenRunner::logits`] as
//!   the parity entry point.
//!
//! Only line recognition is covered: segmentation (`blla`) is a separate
//! model, so lines come in pre-cropped.

/// Command-line entry point (`rlx-kraken`).
pub mod cli;
/// Label codec.
pub mod codec;
/// Graph assembly.
pub mod graph;
/// Core ML `.mlmodel` reader.
pub mod mlmodel;
/// Kraken's line transform.
pub mod preprocess;
/// Line image → text.
pub mod runner;
/// VGSL spec parser.
pub mod vgsl;

pub use codec::Codec;
pub use mlmodel::MlModel;
pub use runner::{KrakenRunner, Recognition};
pub use vgsl::Vgsl;
