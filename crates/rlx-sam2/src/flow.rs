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

//! Tier-0 SAM2 Hiera image encoder flow.

use anyhow::Result;
use rlx_flow::BuiltModel;

use super::config::Sam2HieraConfig;
use super::fpn_neck::FpnNeckWeights;
use super::preprocess::Sam2PreprocessWeights;
use rlx_core::flow_util::built_from_hir;
use rlx_core::weight_map::WeightMap;

#[derive(Debug, Clone)]
pub struct Sam2ImageEncoderFlow<'a> {
    cfg: &'a Sam2HieraConfig,
}

impl<'a> Sam2ImageEncoderFlow<'a> {
    pub fn new(cfg: &'a Sam2HieraConfig) -> Self {
        Self { cfg }
    }

    pub fn build(self, weights: &mut WeightMap) -> Result<Sam2ImageEncoderBuilt> {
        let (hir, params, preprocess, neck) =
            super::image_encoder::build_sam2_image_encoder_hir(self.cfg, weights)?;
        Ok(Sam2ImageEncoderBuilt {
            model: built_from_hir(hir, params)?,
            preprocess,
            neck,
        })
    }
}

pub struct Sam2ImageEncoderBuilt {
    pub model: BuiltModel,
    pub preprocess: Sam2PreprocessWeights,
    pub neck: FpnNeckWeights,
}

pub fn build_sam2_image_encoder_built(
    cfg: &Sam2HieraConfig,
    weights: &mut WeightMap,
) -> Result<Sam2ImageEncoderBuilt> {
    Sam2ImageEncoderFlow::new(cfg).build(weights)
}
