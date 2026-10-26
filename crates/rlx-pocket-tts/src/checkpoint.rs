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

//! A released `model.safetensors` (`flow_lm.*` + `mimi.*`, BF16 or F32).

use std::path::Path;

use anyhow::{Result, anyhow};

use crate::graph::Tensor;
use crate::weights::WeightFile;

pub struct Checkpoint {
    file: WeightFile,
}

impl Checkpoint {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Ok(Self {
            file: WeightFile::open(path)?,
        })
    }

    pub fn has(&self, name: &str) -> bool {
        self.file.tensor(name).is_ok()
    }

    /// Tensor `name` widened to f32.
    pub fn tensor(&self, name: &str) -> Result<Tensor> {
        let (data, shape) = self.file.get_f32(name)?;
        Ok(Tensor { data, shape })
    }

    /// Like [`Self::tensor`], `None` when absent.
    pub fn opt(&self, name: &str) -> Result<Option<Tensor>> {
        if self.has(name) {
            self.tensor(name).map(Some)
        } else {
            Ok(None)
        }
    }

    /// Tensor `name` as a flat vector, checked against `len`.
    pub fn vec(&self, name: &str, len: usize) -> Result<Vec<f32>> {
        let t = self.tensor(name)?;
        if t.data.len() != len {
            return Err(anyhow!(
                "{name}: {} values (shape {:?}), expected {len}",
                t.data.len(),
                t.shape
            ));
        }
        Ok(t.data)
    }
}
