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

//! Shared HIR construction helpers for the Florence-2 vision and language
//! graphs. Architecture-specific `emit_*` methods live in `vision.rs` and
//! `language.rs` as additional `impl Florence2Builder` blocks.

use crate::config::LN_EPS;
use anyhow::Result;
use rlx_flow::WeightSource;
use rlx_ir::hir::{HirGraphExt, HirModule, HirMut, HirNodeId};
use rlx_ir::{DType, Shape};
use std::collections::HashMap;

pub(crate) struct Florence2Builder<'a> {
    pub hir: &'a mut HirModule,
    pub params: &'a mut HashMap<String, Vec<f32>>,
    pub weights: &'a mut dyn WeightSource,
    pub batch: usize,
    pub f: DType,
}

impl<'a> Florence2Builder<'a> {
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

    /// Load a checkpoint tensor as a graph parameter. `transpose` flips the
    /// last two dims (HF linears store `[out, in]`; we want `[in, out]`).
    pub(crate) fn load_param(&mut self, key: &str, transpose: bool) -> Result<HirNodeId> {
        let (data, shape) = self.weights.take(key, transpose)?;
        let id = self.hir.param(key, Shape::new(&shape, self.f));
        self.params.insert(key.to_string(), data);
        Ok(id)
    }

    /// Register a host-computed tensor as a graph parameter under `key`.
    pub(crate) fn register_param(
        &mut self,
        key: &str,
        data: Vec<f32>,
        dims: &[usize],
    ) -> Result<HirNodeId> {
        let id = self.hir.param(key, Shape::new(dims, self.f));
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

    /// `[B, C, H, W]` → `[B, H·W, C]`.
    pub(crate) fn nchw_to_bnc(&mut self, x: HirNodeId, c: usize, h: usize, w: usize) -> HirNodeId {
        let b = self.batch as i64;
        let n = (h * w) as i64;
        let flat = self.g().reshape_(x, vec![b, c as i64, n]);
        self.g().transpose_(flat, vec![0, 2, 1])
    }

    /// `[B, H·W, C]` → `[B, C, H, W]`.
    pub(crate) fn bnc_to_nchw(&mut self, x: HirNodeId, h: usize, w: usize, c: usize) -> HirNodeId {
        let batch = self.batch;
        rlx_core::vision_ops_ir::bhwc_to_nchw(&mut self.g(), x, batch, h, w, c)
    }
}

/// Convolution output dimension: `floor((in + 2·pad − k) / stride) + 1`.
pub(crate) fn conv_out(in_dim: usize, k: usize, stride: usize, pad: usize) -> usize {
    (in_dim + 2 * pad - k) / stride + 1
}
