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
}

/// Feedback state for the duration fixed-point loop (`Expand_1` / `Where_1`).
pub use rlx_onnx_import::control_flow::DURATION_CARRY;

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
