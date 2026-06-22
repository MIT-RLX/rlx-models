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

//! Semantic module map for the native Kitten mini graph.
//!
//! The full HIR is built by [`super::build_native_hir`] via [`crate::graph`]
//! (decomposed from ONNX). Submodule files document boundaries for incremental
//! hand-porting (OCR-style) — see [`crate::native::config::ModuleKind`].

use crate::native::config::ModuleKind;

/// Human-readable module index (compile-time documentation).
pub const MODULE_INDEX: &[(ModuleKind, &str)] = &[
    (
        ModuleKind::Bert,
        "kmodel.bert.* — phoneme Albert encoder on input_ids",
    ),
    (
        ModuleKind::TextEncoder,
        "/text_encoder/* — style LSTM banks, duration side paths",
    ),
    (
        ModuleKind::MelDecoder,
        "kmodel.decoder.* — mel conv decoder (encode + decode stacks)",
    ),
    (
        ModuleKind::Predictor,
        "kmodel.predictor.*, /N.*, /F0.* — F0 + duration conv heads",
    ),
    (
        ModuleKind::Duration,
        "/duration_proj/*, /Expand_1, /Where_1 — duration epilogue",
    ),
    (
        ModuleKind::Vocoder,
        "/decoder/generator/* — HiFi-GAN vocoder (see vocoder.rs)",
    ),
];
