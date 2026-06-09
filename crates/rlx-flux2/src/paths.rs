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

//! Resolve sibling HF component dirs (`text_encoder/`, `vae/`, `tokenizer/`) from a
//! weights file that may live under nested folders such as `transformer/*.safetensors`.

use std::path::{Path, PathBuf};

fn search_roots(model_path: &Path) -> Vec<PathBuf> {
    let start = if model_path.is_dir() {
        model_path.to_path_buf()
    } else if let Some(p) = model_path.parent() {
        p.to_path_buf()
    } else {
        return Vec::new();
    };
    std::iter::successors(Some(start), |cur| cur.parent().map(|p| p.to_path_buf()))
        .take(4)
        .collect()
}

/// Find `component/` (e.g. `text_encoder`, `vae`) walking up from `model_path`.
pub fn find_component_dir(model_path: &Path, component: &str) -> Option<PathBuf> {
    for root in search_roots(model_path) {
        let candidate = root.join(component);
        if candidate.is_dir() {
            return Some(candidate);
        }
    }
    None
}

/// Find `config.json` next to transformer weights (same dir or parent `transformer/`).
pub fn find_transformer_config(model_path: &Path) -> Option<PathBuf> {
    for root in search_roots(model_path) {
        let candidate = root.join("config.json");
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

/// Resolve transformer `config.json` from explicit override or sibling search.
pub fn resolve_transformer_config(
    model_path: &Path,
    override_path: Option<&Path>,
) -> Option<PathBuf> {
    override_path
        .map(|p| p.to_path_buf())
        .or_else(|| find_transformer_config(model_path))
}

/// Find `tokenizer/tokenizer.json` or `tokenizer.json` walking up from `model_path`.
pub fn find_tokenizer_json(model_path: &Path) -> Option<PathBuf> {
    for root in search_roots(model_path) {
        let nested = root.join("tokenizer/tokenizer.json");
        if nested.is_file() {
            return Some(nested);
        }
        let flat = root.join("tokenizer.json");
        if flat.is_file() {
            return Some(flat);
        }
    }
    None
}
