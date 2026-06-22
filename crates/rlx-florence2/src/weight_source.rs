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

//! Non-destructive [`WeightSource`] that clones tensors out of a [`WeightMap`].
//!
//! The graphs (vision / encoder / decoder, recompiled per sequence length)
//! all read from one persistent [`WeightMap`], so we cannot consume tensors
//! with the default destructive `take`. This source clones instead, applying
//! the same 2-D transpose semantics as `WeightMap::take_transposed`.

use anyhow::{Result, anyhow};
use rlx_core::weight_map::WeightMap;
use rlx_flow::WeightSource;

pub struct CloningWeightSource<'a>(pub &'a WeightMap);

impl WeightSource for CloningWeightSource<'_> {
    fn take(&mut self, key: &str, transpose: bool) -> Result<(Vec<f32>, Vec<usize>)> {
        let (data, shape) = self
            .0
            .get(key)
            .ok_or_else(|| anyhow!("missing weight `{key}`"))?;
        if transpose {
            if shape.len() != 2 {
                return Err(anyhow!(
                    "transpose requested for non-2D tensor `{key}` (shape {shape:?})"
                ));
            }
            let (r, c) = (shape[0], shape[1]);
            let mut out = vec![0f32; r * c];
            for i in 0..r {
                for j in 0..c {
                    out[j * r + i] = data[i * c + j];
                }
            }
            Ok((out, vec![c, r]))
        } else {
            Ok((data.to_vec(), shape.to_vec()))
        }
    }

    fn has(&self, key: &str) -> bool {
        self.0.has(key)
    }
}
