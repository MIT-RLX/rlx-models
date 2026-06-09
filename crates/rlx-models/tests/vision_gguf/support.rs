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

//! Shared helpers for env-gated vision GGUF integration tests.

#![allow(dead_code)]

use std::path::{Path, PathBuf};

pub fn env_gguf_path(var: &str) -> Option<PathBuf> {
    let p = std::env::var(var).ok().filter(|s| !s.is_empty())?;
    let path = PathBuf::from(p);
    if path.is_file() {
        Some(path)
    } else {
        eprintln!("skip {var}: not a file ({path:?})");
        None
    }
}

pub fn env_path(var: &str) -> Option<PathBuf> {
    let p = std::env::var(var).ok().filter(|s| !s.is_empty())?;
    let path = PathBuf::from(p);
    if path.exists() {
        Some(path)
    } else {
        eprintln!("skip {var}: path missing ({path:?})");
        None
    }
}

/// Heavy compile tests (SAM3 full stack) require `VISION_GGUF_COMPILE=1`.
pub fn compile_gate() -> bool {
    std::env::var("VISION_GGUF_COMPILE").ok().as_deref() == Some("1")
}

pub fn weights_dir_for_gguf(gguf: &Path, dir_var: &str) -> PathBuf {
    if let Some(dir) = env_path(dir_var) {
        return dir;
    }
    gguf.parent()
        .filter(|p| !p.as_os_str().is_empty())
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."))
}

/// Model tree root: explicit `FLUX_MODEL_ROOT` or parent of a GGUF file path.
pub fn flux_model_root(weights_var: &str, gguf_var: &str) -> Option<PathBuf> {
    if let Some(root) = env_path("FLUX_MODEL_ROOT") {
        return Some(root);
    }
    let w = env_gguf_path(weights_var).or_else(|| env_path(gguf_var))?;
    if w.is_dir() {
        Some(w)
    } else {
        w.parent()
            .filter(|p| !p.as_os_str().is_empty())
            .map(Path::to_path_buf)
    }
}
