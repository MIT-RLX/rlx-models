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

//! Central registry of the crate's `OCR2_*` environment knobs.
//!
//! | var | effect |
//! |-----|--------|
//! | `OCR2_DEVICE` | backend selector for the CLI (cpu/metal/mlx/cuda/gpu/vulkan/coreml) |
//! | `OCR2_REPEAT` | run the CLI pipeline N times in-process (warm-timing aid) |
//! | `OCR2_TIMING` | print per-stage timings |
//! | `OCR2_NO_FUSION` | disable conv+bias+act fusion in the detector |
//! | `OCR2_RESCORE_DEBUG` | print each beam candidate's rec/rescore/total |
//! | `OCR2_LEX_W` | override the lexicon rescoring weight |
//!
//! `OCR2_DEVICE` / `OCR2_REPEAT` are parsed by the CLI (`bin/rlx_ocr2.rs`); the
//! rest are library knobs read through the accessors below.

fn set(key: &str) -> bool {
    std::env::var(key).is_ok()
}

/// Print per-stage timings (`OCR2_TIMING`).
pub fn timing() -> bool {
    set("OCR2_TIMING")
}

/// Disable conv+bias+act fusion when compiling the detector (`OCR2_NO_FUSION`).
pub fn no_fusion() -> bool {
    set("OCR2_NO_FUSION")
}

/// Print each beam candidate's component scores (`OCR2_RESCORE_DEBUG`).
pub fn rescore_debug() -> bool {
    set("OCR2_RESCORE_DEBUG")
}

/// Lexicon rescoring weight, overridable via `OCR2_LEX_W` (falls back to `default`).
pub fn lex_weight(default: f32) -> f32 {
    std::env::var("OCR2_LEX_W")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}
