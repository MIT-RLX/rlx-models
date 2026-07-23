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

//! `rlx-ocr2` — a native RLX OCR pipeline:
//! image → detector (CRAFT-style heatmaps) → line grouping → per-line recognizer
//! (CRNN+CTC) → optional n-gram + lexicon rescoring.
//!
//! Every model stage is a native rlx-ir graph, validated numerically against
//! stored fixtures (see `tests/`). Weight conventions (planar 8-bit LUT convs,
//! uint8 per-channel FC, LSTM col-layout with gate order `i,f,o,g`) were
//! reproduced exactly.

// Internal helpers.
mod compile;
mod env;

// Pipeline stages.
pub mod beam;
pub mod detection;
pub mod graph;
pub mod grouping;
pub mod ngram;
pub mod pipeline;
pub mod preprocess;
pub mod recognition;
pub mod rescore;
pub mod runner;

pub use detection::{Detector, build_detector_graph};
pub use ngram::NgramModel;
pub use pipeline::{Ocr2, OcrLine};
pub use recognition::{HIDDEN, NUM_CLASSES, REC_HEIGHT, build_recognition_graph};
pub use rescore::{Lexicon, Rescorer};
pub use runner::Recognizer;
