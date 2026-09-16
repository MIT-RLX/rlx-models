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

//! Strip / prepend a checkpoint key prefix for nested multimodal layouts
//! (`backbone_llm.model.language_model.*` → `model.language_model.*`).

use anyhow::Result;
use rlx_core::weight_loader::WeightLoader;

/// Look up `key`, then `{prefix}{key}` (FireRedAudio backbone nesting).
pub struct PrefixStripLoader<L: WeightLoader> {
    inner: L,
    prefix: String,
}

impl<L: WeightLoader> PrefixStripLoader<L> {
    pub fn new(inner: L, prefix: impl Into<String>) -> Self {
        Self {
            inner,
            prefix: prefix.into(),
        }
    }
}

impl<L: WeightLoader> WeightLoader for PrefixStripLoader<L> {
    fn format_id(&self) -> &'static str {
        self.inner.format_id()
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn take(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        match self.inner.take(key) {
            Ok(v) => Ok(v),
            Err(first) => {
                let pref = format!("{}{key}", self.prefix);
                self.inner.take(&pref).map_err(|_| first)
            }
        }
    }

    fn take_transposed(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        match self.inner.take_transposed(key) {
            Ok(v) => Ok(v),
            Err(first) => {
                let pref = format!("{}{key}", self.prefix);
                self.inner.take_transposed(&pref).map_err(|_| first)
            }
        }
    }

    fn take_packed(
        &mut self,
        key: &str,
    ) -> Result<Option<rlx_core::weight_map::PackedWeightTensor>> {
        match self.inner.take_packed(key)? {
            Some(v) => Ok(Some(v)),
            None => {
                let pref = format!("{}{key}", self.prefix);
                self.inner.take_packed(&pref)
            }
        }
    }

    fn release_mapped_pages(&self) {
        self.inner.release_mapped_pages();
    }

    fn remaining_keys(&self) -> Vec<String> {
        self.inner
            .remaining_keys()
            .into_iter()
            .map(|k| {
                k.strip_prefix(&self.prefix)
                    .map(str::to_string)
                    .unwrap_or(k)
            })
            .collect()
    }
}
