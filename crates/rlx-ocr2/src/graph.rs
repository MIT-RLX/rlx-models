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

//! Thin HIR graph builder shared by the detector and recognizer stages: wraps an
//! `HirModule`, tracks host params, and loads named weights from a `WeightMap`.

use anyhow::{Context, Result};
use rlx_core::weight_map::WeightMap;
use rlx_ir::hir::{HirMut, HirNodeId};
use rlx_ir::{DType, Shape};
use std::collections::HashMap;

pub struct OcrGraphBuilder {
    pub hir: rlx_ir::hir::HirModule,
    pub params: HashMap<String, Vec<f32>>,
}

impl OcrGraphBuilder {
    pub fn new(name: &str) -> Self {
        Self {
            hir: rlx_ir::hir::HirModule::new(name),
            params: HashMap::new(),
        }
    }

    pub fn m(&mut self) -> HirMut<'_> {
        HirMut::new(&mut self.hir)
    }

    /// Load a named f32 tensor from the weight map as a graph param.
    pub fn load_param(&mut self, wm: &mut WeightMap, key: &str) -> Result<HirNodeId> {
        let (data, shape) = wm
            .take(key)
            .with_context(|| format!("missing weight {key}"))?;
        let id = self.m().param(key, Shape::new(&shape, DType::F32));
        self.params.insert(key.to_string(), data);
        Ok(id)
    }

    /// A constant zero tensor of the given shape (used for explicit asymmetric padding).
    pub fn zeros(&mut self, key: &str, shape: &[usize]) -> HirNodeId {
        self.const_full(key, shape, 0.0)
    }

    /// A constant tensor of the given shape filled with `val`.
    pub fn const_full(&mut self, key: &str, shape: &[usize], val: f32) -> HirNodeId {
        let n: usize = shape.iter().product();
        let id = self.m().param(key, Shape::new(shape, DType::F32));
        self.params.insert(key.to_string(), vec![val; n]);
        id
    }

    pub fn finish(self) -> Result<(rlx_ir::Graph, HashMap<String, Vec<f32>>)> {
        rlx_core::flow_util::graph_from_hir(self.hir, self.params)
    }
}
