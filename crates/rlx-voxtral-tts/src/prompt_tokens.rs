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

//! Load prompt token ids exported by the Docker tools image.

use anyhow::{Context, Result};
use std::path::Path;

pub fn load_prompt_tokens(path: &Path) -> Result<Vec<u32>> {
    let text = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    text.split_whitespace()
        .map(|s| s.parse().context("parse prompt token id"))
        .collect()
}
