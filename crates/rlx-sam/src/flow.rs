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

//! Tier-0 SAM v1 image encoder flow.

use anyhow::Result;
use rlx_flow::BuiltModel;

use super::config::SamEncoderConfig;
use super::preprocess::SamPreprocessWeights;
use rlx_core::flow_util::built_from_hir;
use rlx_core::weight_map::WeightMap;

#[derive(Debug, Clone)]
pub struct SamEncoderFlow<'a> {
    cfg: &'a SamEncoderConfig,
}

impl<'a> SamEncoderFlow<'a> {
    pub fn new(cfg: &'a SamEncoderConfig) -> Self {
        Self { cfg }
    }

    pub fn build(self, weights: &mut WeightMap) -> Result<SamEncoderBuilt> {
        let (hir, params, preprocess) =
            super::image_encoder::build_sam_encoder_hir(self.cfg, weights)?;
        Ok(SamEncoderBuilt {
            model: built_from_hir(hir, params)?,
            preprocess,
        })
    }
}

pub struct SamEncoderBuilt {
    pub model: BuiltModel,
    pub preprocess: SamPreprocessWeights,
}

pub fn build_sam_encoder_built(
    cfg: &SamEncoderConfig,
    weights: &mut WeightMap,
) -> Result<SamEncoderBuilt> {
    SamEncoderFlow::new(cfg).build(weights)
}
