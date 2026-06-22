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

use std::cell::Cell;

thread_local! {
    /// Per-thread runtime seq for tests/probes; wins over process env when set.
    static RUNTIME_SEQUENCE_OVERRIDE: Cell<Option<usize>> = const { Cell::new(None) };
    /// Active token count for duration alignment kernels (ConcatFromSequence trip_count).
    static RUNTIME_ACTIVE_TOKENS: Cell<Option<usize>> = const { Cell::new(None) };
    /// Active mel frames (2× alignment) for F0/N predictor DynamicQuantizeLinear.
    static RUNTIME_MEL_FRAMES: Cell<Option<usize>> = const { Cell::new(None) };
}

/// Feedback state for the duration fixed-point loop (`Expand_1` / `Where_1`).
pub use rlx_onnx_import::control_flow::DURATION_CARRY;

/// Runtime param holding alignment frame count for `/Gather_5` (replaces stale `/Shape_8` stub).
pub const ALIGNMENT_FRAME_COUNT: &str = "__onnx_runtime__/alignment_frame_count";

/// Decomposed-graph stub for `/Range_2` (filled each infer with `0..frames`).
pub const RANGE_2_STUB: &str = "__stub__//Range_2_output_0";

/// Decomposed-graph stub for `/ConcatFromSequence` alignment buffer.
pub const CONCAT_SEQUENCE_STUB: &str = "__stub__//ConcatFromSequence_output_0";

/// Runtime param holding `[1, seq]` for dynamic `input_ids` reshape.
pub const RUNTIME_INPUT_IDS_SHAPE: &str = "__onnx_runtime__/input_ids_shape";

/// Env var read by `rlx-compile` to restore collapsed sequence axes.
pub const COMPILE_SEQUENCE_LENGTH_ENV: &str = "RLX_ONNX_SEQUENCE_LENGTH";

/// Legacy alias still accepted by model-local scripts and probes.
pub const LEGACY_SEQUENCE_LENGTH_ENV: &str = "KITTEN_SEQUENCE_LENGTH";

/// Env var for an exported RLX ONNX bundle directory.
pub const ONNX_BUNDLE_ENV: &str = "RLX_ONNX_BUNDLE";

/// Legacy bundle env alias.
pub const LEGACY_BUNDLE_ENV: &str = "KITTEN_RLX_BUNDLE";

/// Dynamic dimension bindings used when lowering ONNX.
#[derive(Debug, Clone)]
pub struct GraphOptions {
    pub sequence_length: usize,
    pub max_waveform_samples: usize,
}

impl Default for GraphOptions {
    fn default() -> Self {
        Self {
            sequence_length: 128,
            max_waveform_samples: 48_000,
        }
    }
}

/// Bind active token width for compile-time shape restoration in `rlx-compile`.
pub fn set_compile_sequence_length(seq: usize) {
    let s = seq.to_string();
    crate::set_env_var(COMPILE_SEQUENCE_LENGTH_ENV, &s);
    crate::set_env_var(LEGACY_SEQUENCE_LENGTH_ENV, &s);
}

/// Hint ConcatFromSequence and similar ops of the runtime token count (not compile slots).
pub fn set_runtime_active_tokens(active: usize) {
    let n = active.max(1);
    RUNTIME_ACTIVE_TOKENS.with(|c| c.set(Some(n)));
    rlx_runtime::onnx_active::set_active_token_count(Some(n));
}

/// Active mel length for F0/N stacks (`2 ×` alignment frames).
pub fn set_runtime_mel_frames(mel: usize) {
    RUNTIME_MEL_FRAMES.with(|c| c.set(Some(mel.max(1))));
}

pub fn runtime_mel_frames() -> Option<usize> {
    RUNTIME_MEL_FRAMES.with(|c| c.get())
}

/// Active token width for alignment kernels; falls back to compile env when unset.
pub fn runtime_active_tokens() -> Option<usize> {
    RUNTIME_ACTIVE_TOKENS
        .with(|c| c.get())
        .or_else(compile_sequence_length_from_env)
}

/// Restore compile-sequence env after a scoped test/probe (avoids cross-test leakage).
pub struct CompileSequenceLengthGuard {
    rlx: Option<String>,
    legacy: Option<String>,
}

impl CompileSequenceLengthGuard {
    pub fn set(seq: usize) -> Self {
        let guard = Self {
            rlx: std::env::var(COMPILE_SEQUENCE_LENGTH_ENV).ok(),
            legacy: std::env::var(LEGACY_SEQUENCE_LENGTH_ENV).ok(),
        };
        RUNTIME_SEQUENCE_OVERRIDE.with(|c| c.set(Some(seq)));
        set_compile_sequence_length(seq);
        guard
    }
}

impl Drop for CompileSequenceLengthGuard {
    fn drop(&mut self) {
        RUNTIME_SEQUENCE_OVERRIDE.with(|c| c.set(None));
        RUNTIME_ACTIVE_TOKENS.with(|c| c.set(None));
        RUNTIME_MEL_FRAMES.with(|c| c.set(None));
        match &self.rlx {
            Some(v) => crate::set_env_var(COMPILE_SEQUENCE_LENGTH_ENV, v),
            None => unsafe {
                std::env::remove_var(COMPILE_SEQUENCE_LENGTH_ENV);
            },
        }
        match &self.legacy {
            Some(v) => crate::set_env_var(LEGACY_SEQUENCE_LENGTH_ENV, v),
            None => unsafe {
                std::env::remove_var(LEGACY_SEQUENCE_LENGTH_ENV);
            },
        }
    }
}

/// Active token width from thread override (tests), then compile env.
pub fn compile_sequence_length_from_env() -> Option<usize> {
    if let Some(seq) = RUNTIME_SEQUENCE_OVERRIDE.with(|c| c.get()) {
        return Some(seq);
    }
    std::env::var(COMPILE_SEQUENCE_LENGTH_ENV)
        .or_else(|_| std::env::var(LEGACY_SEQUENCE_LENGTH_ENV))
        .ok()
        .and_then(|s| s.parse().ok())
}
