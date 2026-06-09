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

//! Tier-0 Qwen3.5-VL vision tower flow.

use anyhow::Result;
use rlx_flow::{BuiltModel, CompileProfile};

use super::config::MmProjConfig;
use super::weights::MmProjWeights;
use rlx_core::flow_util::built_from_hir_with_profile;

#[derive(Debug, Clone)]
pub struct Qwen35VisionFlow<'a> {
    cfg: &'a MmProjConfig,
    weights: &'a MmProjWeights,
    img_w: usize,
    img_h: usize,
}

impl<'a> Qwen35VisionFlow<'a> {
    pub fn new(
        cfg: &'a MmProjConfig,
        weights: &'a MmProjWeights,
        img_w: usize,
        img_h: usize,
    ) -> Self {
        Self {
            cfg,
            weights,
            img_w,
            img_h,
        }
    }

    pub fn build(self) -> Result<BuiltModel> {
        let (hir, params) = super::builder::build_qwen35_vision_hir(
            self.cfg,
            self.weights,
            self.img_w,
            self.img_h,
        )?;
        built_from_hir_with_profile(hir, params, CompileProfile::encoder())
    }
}

pub fn build_qwen35_vision_built(
    cfg: &MmProjConfig,
    weights: &MmProjWeights,
    img_w: usize,
    img_h: usize,
) -> Result<BuiltModel> {
    Qwen35VisionFlow::new(cfg, weights, img_w, img_h).build()
}
