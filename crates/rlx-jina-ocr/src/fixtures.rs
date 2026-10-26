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

//! Bundled sample assets and env helpers (tests, CLI, examples).

use std::path::{Path, PathBuf};

/// Path relative to the crate root: `fixtures/sample.png`.
pub const SAMPLE_IMAGE_REL: &str = "fixtures/sample.png";

/// Crate root (`CARGO_MANIFEST_DIR`).
pub fn crate_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Bundled sample document image.
pub fn sample_image_path() -> PathBuf {
    crate_root().join(SAMPLE_IMAGE_REL)
}

/// `RLX_JINA_OCR_IMAGE` when set, otherwise [`sample_image_path`].
pub fn probe_image_path() -> PathBuf {
    rlx_ir::env::var_result("RLX_JINA_OCR_IMAGE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| sample_image_path())
}

/// The probe image path when the file exists.
pub fn require_probe_image() -> Option<PathBuf> {
    let path = probe_image_path();
    path.is_file().then_some(path)
}

/// Whether `dir` carries the tokenizer assets the runner needs.
pub fn model_dir_has_tokenizer(dir: &Path) -> bool {
    dir.join("tokenizer.json").is_file()
}

/// Whether `dir` carries safetensors weights (sharded or flat).
///
/// Checked separately from the tokenizer because a Hub cache entry is populated
/// file by file: resolving `config.json` alone creates a snapshot directory, and
/// a test that treated that as "the checkpoint is here" would fail on the
/// missing shards rather than skip.
pub fn model_dir_has_weights(dir: &Path) -> bool {
    if dir.join("model.safetensors.index.json").is_file() || dir.join("model.safetensors").is_file()
    {
        return true;
    }
    std::fs::read_dir(dir).is_ok_and(|entries| {
        entries.flatten().any(|e| {
            e.path()
                .extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("safetensors"))
        })
    })
}

/// Checkpoint dir when config, tokenizer *and* weights are all present.
pub fn require_model_dir() -> Option<PathBuf> {
    crate::hub::default_model_dir()
        .ok()
        .filter(|dir| model_dir_has_tokenizer(dir) && model_dir_has_weights(dir))
}

/// Resolve a model directory string (absolute, cwd-relative, or under workspace).
pub fn resolve_model_dir_path(raw: &str) -> Option<PathBuf> {
    let path = PathBuf::from(raw);
    if path.join("config.json").is_file() {
        return Some(path);
    }
    let rooted = crate_root().join("../../").join(raw);
    if rooted.join("config.json").is_file() {
        return Some(rooted.canonicalize().unwrap_or(rooted));
    }
    None
}

/// Default image for CLI / examples: explicit path or bundled sample.
pub fn resolve_image_path(explicit: Option<impl AsRef<Path>>) -> PathBuf {
    explicit
        .map(|p| p.as_ref().to_path_buf())
        .unwrap_or_else(probe_image_path)
}
