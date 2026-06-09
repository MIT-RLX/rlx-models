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

//! Tier-0 V-JEPA2 flows — encoder, predictor, pooler.

use anyhow::Result;
use rlx_flow::BuiltModel;

use super::config::Vjepa2Config;
use super::predictor::Vjepa2PredictorLayout;
use super::weights::{Vjepa2EncoderWeights, Vjepa2PoolerWeights, Vjepa2PredictorWeights};
use rlx_core::flow_util::built_from_hir;

#[derive(Clone)]
pub struct Vjepa2EncoderFlow<'a> {
    cfg: &'a Vjepa2Config,
    encoder: &'a Vjepa2EncoderWeights,
    batch: usize,
}

impl<'a> Vjepa2EncoderFlow<'a> {
    pub fn new(cfg: &'a Vjepa2Config, encoder: &'a Vjepa2EncoderWeights, batch: usize) -> Self {
        Self {
            cfg,
            encoder,
            batch,
        }
    }

    pub fn build(self) -> Result<Vjepa2EncoderBuilt> {
        let (hir, params, preprocess) =
            super::builder::build_vjepa2_encoder_hir_sized(self.cfg, self.encoder, self.batch)?;
        Ok(Vjepa2EncoderBuilt {
            model: built_from_hir(hir, params)?,
            preprocess,
        })
    }
}

pub struct Vjepa2EncoderBuilt {
    pub model: BuiltModel,
    pub preprocess: super::builder::Vjepa2GraphPreprocess,
}

#[derive(Clone)]
pub struct Vjepa2PredictorFlow<'a> {
    cfg: &'a Vjepa2Config,
    predictor: &'a Vjepa2PredictorWeights,
    layout: &'a Vjepa2PredictorLayout,
    mask_rows: &'a [f32],
    batch: usize,
}

impl<'a> Vjepa2PredictorFlow<'a> {
    pub fn new(
        cfg: &'a Vjepa2Config,
        predictor: &'a Vjepa2PredictorWeights,
        layout: &'a Vjepa2PredictorLayout,
        mask_rows: &'a [f32],
        batch: usize,
    ) -> Self {
        Self {
            cfg,
            predictor,
            layout,
            mask_rows,
            batch,
        }
    }

    pub fn build(self) -> Result<BuiltModel> {
        let (hir, params) = super::builder::build_vjepa2_predictor_hir_sized(
            self.cfg,
            self.predictor,
            self.layout,
            self.mask_rows,
            self.batch,
        )?;
        built_from_hir(hir, params.f32)
    }
}

#[derive(Clone)]
pub struct Vjepa2PoolerFlow<'a> {
    cfg: &'a Vjepa2Config,
    pooler: &'a Vjepa2PoolerWeights,
    batch: usize,
}

impl<'a> Vjepa2PoolerFlow<'a> {
    pub fn new(cfg: &'a Vjepa2Config, pooler: &'a Vjepa2PoolerWeights, batch: usize) -> Self {
        Self { cfg, pooler, batch }
    }

    pub fn build(self) -> Result<BuiltModel> {
        let (hir, params) =
            super::builder::build_vjepa2_pooler_hir_sized(self.cfg, self.pooler, self.batch)?;
        built_from_hir(hir, params.f32)
    }
}
