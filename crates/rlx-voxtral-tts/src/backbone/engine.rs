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

//! Shared backbone interface (eager CPU or RLX-compiled).

use crate::backbone::compiled::CompiledMinistralLm;
use crate::backbone::lm::MinistralLm;
use crate::config::TextConfig;
use crate::load::VoxtralTtsWeightStore;
use crate::load::WeightSnapshot;
use anyhow::Result;
use ndarray::{Array1, Array2, ArrayView2};
use rlx_runtime::Device;

pub enum BackboneLm {
    Eager(MinistralLm),
    Compiled(Box<CompiledMinistralLm>),
}

impl BackboneLm {
    pub fn open(
        store: &VoxtralTtsWeightStore,
        cfg: &TextConfig,
        backbone_tensors: Option<&WeightSnapshot>,
        device: Device,
        eager_lm: bool,
        lora: Option<&crate::lora::LoraBank>,
    ) -> Result<Self> {
        if eager_lm {
            let tensors = backbone_tensors
                .ok_or_else(|| anyhow::anyhow!("eager LM requires backbone weight tensors"))?;
            Ok(Self::Eager(MinistralLm::from_tensors_with_lora(
                tensors, cfg, lora,
            )?))
        } else {
            Ok(Self::Compiled(Box::new(CompiledMinistralLm::open(
                store,
                cfg,
                device,
                backbone_tensors,
                lora,
            )?)))
        }
    }

    pub fn reset_cache(&mut self) {
        match self {
            Self::Eager(lm) => lm.reset_cache(),
            Self::Compiled(lm) => lm.reset_cache(),
        }
    }

    pub fn forward(&mut self, inputs_embeds: ArrayView2<f32>) -> Result<Array2<f32>> {
        match self {
            Self::Eager(lm) => lm.forward(inputs_embeds),
            Self::Compiled(lm) => lm.forward(inputs_embeds),
        }
    }

    pub fn last_hidden(&self, hidden: &Array2<f32>) -> Array1<f32> {
        match self {
            Self::Eager(lm) => lm.last_hidden(hidden),
            Self::Compiled(lm) => lm.last_hidden(hidden),
        }
    }
}
