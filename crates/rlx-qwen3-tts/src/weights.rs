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

//! Weight loader helpers.

use rlx_core::weight_loader::WeightLoader;
use rlx_core::weight_map::WeightMap;
use std::collections::HashMap;

pub struct SnapshotLoader {
    map: HashMap<String, (Vec<f32>, Vec<usize>)>,
}

impl SnapshotLoader {
    pub fn new(map: HashMap<String, (Vec<f32>, Vec<usize>)>) -> Self {
        Self { map }
    }
}

impl WeightLoader for SnapshotLoader {
    fn len(&self) -> usize {
        self.map.len()
    }

    fn take(&mut self, key: &str) -> anyhow::Result<(Vec<f32>, Vec<usize>)> {
        self.map
            .remove(key)
            .ok_or_else(|| anyhow::anyhow!("missing weight {key}"))
    }

    fn take_transposed(&mut self, key: &str) -> anyhow::Result<(Vec<f32>, Vec<usize>)> {
        let (data, shape) = self.take(key)?;
        if shape.len() != 2 {
            anyhow::bail!("transpose requires 2D weight: {key}");
        }
        let (rows, cols) = (shape[0], shape[1]);
        let mut out = vec![0f32; data.len()];
        for r in 0..rows {
            for c in 0..cols {
                out[c * rows + r] = data[r * cols + c];
            }
        }
        Ok((out, vec![cols, rows]))
    }

    fn remaining_keys(&self) -> Vec<String> {
        self.map.keys().cloned().collect()
    }
}

pub fn weight_map_from_cache(
    cache: &HashMap<String, (Vec<f32>, Vec<usize>)>,
) -> anyhow::Result<WeightMap> {
    let mut loader =
        SnapshotLoader::new(cache.iter().map(|(k, v)| (k.clone(), v.clone())).collect());
    WeightMap::from_weight_loader(&mut loader)
}
