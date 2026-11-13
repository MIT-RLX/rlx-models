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

//! PyLaia — Teklia's CRNN for handwritten text recognition, natively in RLX.
//!
//! `LaiaCRNN` reads one text line: VGG-style conv blocks (conv, batchnorm,
//! LeakyReLU, max-pool), a column sequencer, a bidirectional LSTM and a
//! linear layer, decoded with greedy CTC. Teklia publishes trained models
//! for many collections on the Hugging Face Hub (`Teklia/pylaia-rimes` is
//! French, `Teklia/pylaia-iam` English, …); a model directory holds
//! `model` (hyperparameters), `weights.ckpt` and `syms.txt`.
//!
//! - [`PylaiaConfig`] — architecture, read from the `model` file.
//! - [`PylaiaWeights`] — the checkpoint with BatchNorm folded into the convs.
//! - [`graph::build`] — the forward as one RLX graph for a given image size.
//! - [`PylaiaRunner`] — image → text, with [`PylaiaRunner::logits`] as the
//!   parity entry point.

/// Command-line entry point (`rlx-pylaia`).
pub mod cli;
/// Architecture hyperparameters.
pub mod config;
/// Graph assembly.
pub mod graph;
/// PyLaia's prediction-time image transform.
pub mod preprocess;
/// Line image → text.
pub mod runner;
/// `syms.txt` symbol table.
pub mod symbols;
/// Checkpoint loading and BatchNorm folding.
pub mod weights;

pub use config::{Activation, ConvBlockConfig, PylaiaConfig, RnnKind, Sequencer};
pub use runner::{PylaiaRunner, Recognition};
pub use symbols::SymbolTable;
pub use weights::{PylaiaWeights, read_checkpoint};
