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

//! Small helpers for pulling named tensors out of a [`WeightMap`].

use anyhow::{Context, Result};
use rlx_core::weight_map::WeightMap;

/// Clone a tensor's data (PyTorch `[out, in]` / native layout — no transpose).
pub fn get(wm: &WeightMap, key: &str) -> Result<Vec<f32>> {
    let (d, _) = wm
        .get(key)
        .with_context(|| format!("missing weight: {key}"))?;
    Ok(d.to_vec())
}

/// Clone a tensor's data and shape.
pub fn get_with_shape(wm: &WeightMap, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
    let (d, s) = wm
        .get(key)
        .with_context(|| format!("missing weight: {key}"))?;
    Ok((d.to_vec(), s.to_vec()))
}

/// Optional clone — `None` if absent (for optional biases).
#[allow(dead_code)] // used for optional biases in later phases
pub fn get_opt(wm: &WeightMap, key: &str) -> Option<Vec<f32>> {
    wm.get(key).map(|(d, _)| d.to_vec())
}
