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

//! Shared HIR construction helpers for the NLLB encoder / decoder graphs.
//! Architecture-specific `emit_*` methods live in `language.rs` as additional
//! `impl NllbBuilder` blocks.

use crate::config::LN_EPS;
use crate::weights::lang as lk;
use anyhow::Result;
use rlx_flow::WeightSource;
use rlx_ir::hir::{HirGraphExt, HirModule, HirMut, HirNodeId};
use rlx_ir::{DType, Shape};
use std::collections::HashMap;

pub(crate) struct NllbBuilder<'a> {
    pub hir: &'a mut HirModule,
    pub params: &'a mut HashMap<String, Vec<f32>>,
    pub weights: &'a mut dyn WeightSource,
    pub batch: usize,
    pub f: DType,
}

impl<'a> NllbBuilder<'a> {
    pub(crate) fn new(
        hir: &'a mut HirModule,
        params: &'a mut HashMap<String, Vec<f32>>,
        weights: &'a mut dyn WeightSource,
        batch: usize,
    ) -> Self {
        Self {
            hir,
            params,
            weights,
            batch,
            f: DType::F32,
        }
    }

    pub(crate) fn g(&mut self) -> HirMut<'_> {
        HirMut::new(self.hir)
    }

    /// HF M2M100 uses sinusoidal positions + Pre-LN; synthetic BART tests use learned + Post-LN.
    pub(crate) fn m2m100_style(&self) -> bool {
        !self.weights.has(&lk::enc_embed_positions())
    }

    /// Load a checkpoint tensor as a graph parameter. `transpose` flips the
    /// last two dims (HF linears store `[out, in]`; we want `[in, out]`).
    pub(crate) fn load_param(&mut self, key: &str, transpose: bool) -> Result<HirNodeId> {
        let (data, shape) = self.weights.take(key, transpose)?;
        let id = self.hir.param(key, Shape::new(&shape, self.f));
        self.params.insert(key.to_string(), data);
        Ok(id)
    }

    /// `x @ W^T (+ b)` for an HF `nn.Linear` with weight `[out, in]`.
    pub(crate) fn linear(
        &mut self,
        x: HirNodeId,
        w_key: &str,
        b_key: Option<&str>,
    ) -> Result<HirNodeId> {
        let w = self.load_param(w_key, true)?;
        let mut y = self.g().mm(x, w);
        if let Some(bk) = b_key {
            let b = self.load_param(bk, false)?;
            y = self.g().add(y, b);
        }
        Ok(y)
    }

    /// LayerNorm with the standard `1e-5` epsilon.
    pub(crate) fn layer_norm(
        &mut self,
        x: HirNodeId,
        w_key: &str,
        b_key: &str,
    ) -> Result<HirNodeId> {
        let gamma = self.load_param(w_key, false)?;
        let beta = self.load_param(b_key, false)?;
        Ok(self.g().ln(x, gamma, beta, LN_EPS))
    }
}
