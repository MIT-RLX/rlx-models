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

//! Load ocrs-compatible `.rten` checkpoints with mmap + prepacked weights when available.

use anyhow::{Context, Result};
use rten::Model;
use std::path::Path;

/// Load an RTen checkpoint for inference (mmap when enabled, prepacked conv weights).
pub fn load_rten_model(path: impl AsRef<Path>) -> Result<Model> {
    let path = path.as_ref();
    let mut opts = rten::ModelOptions::default();
    opts.prepack_weights(true);

    #[cfg(feature = "rten-mmap")]
    {
        // SAFETY: file is read-only; RTen maps weight blobs without mutating the file.
        unsafe { opts.load_mmap(path) }.with_context(|| format!("mmap-load model {:?}", path))
    }

    #[cfg(not(feature = "rten-mmap"))]
    {
        opts.load_file(path)
            .with_context(|| format!("load model {:?}", path))
    }
}
