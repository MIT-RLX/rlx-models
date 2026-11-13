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

//! DAN — the Document Attention Network (Coquenet et al., TPAMI 2023) —
//! natively in RLX: whole-page handwritten text recognition, no line
//! segmentation, with layout tokens marking the page structure.
//!
//! A fully convolutional encoder turns the page into a 256-channel map at
//! 1/32 × 1/8 resolution; a transformer decoder reads it character by
//! character. The authors release checkpoints for RIMES 2009 (French
//! letters) and READ 2016 (German) pages (Zenodo 7244382; Faster DAN:
//! 7568900).
//!
//! - [`model::DanCheckpoint`] — the training checkpoint and its charset.
//! - [`graph`] — the encoder (with every layer's cross-attention K/V) and a
//!   one-token decoder step.
//! - [`DanRunner`] — page → text, with [`DanRunner::encode`] /
//!   [`DanRunner::step`] as the parity entry points.

/// Command-line entry point (`rlx-dan`).
pub mod cli;
/// Encoder and decoder-step graphs.
pub mod graph;
/// Checkpoint, fixed architecture, positional tables.
pub mod model;
/// Page → text.
pub mod runner;

pub use model::DanCheckpoint;
pub use runner::{DEFAULT_MAX_CHARS, DanRunner, LAYOUT_TOKENS, Recognition};
