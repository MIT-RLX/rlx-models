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

//! Native Kitten TTS — Rust-encoded graph + safetensors/GGUF weights (no ONNX bundle).

pub mod compile;
pub mod config;
pub mod flow;

pub use compile::{NativeSeqCompileCache, compile_native, compile_native_fresh};
pub use config::{KITTEN_GGUF_ARCH, KITTEN_GGUF_ARCHES, KittenTtsConfig, ModuleKind};
pub use flow::build_native_hir;

use std::path::{Path, PathBuf};

use crate::weights::native_weights_available;

/// Resolve a directory containing `model.safetensors` or `model.gguf`.
pub fn native_weights_dir_near(base: &Path) -> Option<PathBuf> {
    if native_weights_available(base) {
        return Some(base.to_path_buf());
    }
    for candidate in [base.join("weights"), base.to_path_buf()] {
        if native_weights_available(&candidate) {
            return Some(candidate);
        }
    }
    base.parent()
        .map(|p| p.join("weights"))
        .filter(|p| native_weights_available(p))
}
