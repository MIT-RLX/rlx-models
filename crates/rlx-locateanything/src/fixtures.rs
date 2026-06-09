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

//! Bundled sample image and env helpers (tests, CLI, examples).

use std::path::{Path, PathBuf};

/// Path relative to the crate root: `fixtures/sample.jpg`.
pub const SAMPLE_IMAGE_REL: &str = "fixtures/sample.jpg";

/// Crate root (`CARGO_MANIFEST_DIR`).
pub fn crate_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Bundled 640×360 JPEG ([yolov5 zidane](https://github.com/ultralytics/yolov5/blob/master/data/images/zidane.jpg)).
pub fn sample_image_path() -> PathBuf {
    crate_root().join(SAMPLE_IMAGE_REL)
}

/// `RLX_LOCATEANYTHING_IMAGE` when set, otherwise [`sample_image_path`].
pub fn probe_image_path() -> PathBuf {
    std::env::var("RLX_LOCATEANYTHING_IMAGE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| sample_image_path())
}

/// Returns the probe image path when the file exists.
pub fn require_probe_image() -> Option<PathBuf> {
    let path = probe_image_path();
    path.is_file().then_some(path)
}

/// `tokenizer.json` or BPE `vocab.json` + `merges.txt`.
pub fn model_dir_has_tokenizer(dir: &Path) -> bool {
    dir.join("tokenizer.json").is_file()
        || (dir.join("vocab.json").is_file() && dir.join("merges.txt").is_file())
}

/// Checkpoint dir when config + tokenizer assets are available ([`crate::hub::default_model_dir`]).
pub fn require_model_dir() -> Option<PathBuf> {
    crate::hub::default_model_dir()
        .ok()
        .filter(|dir| model_dir_has_tokenizer(dir))
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
