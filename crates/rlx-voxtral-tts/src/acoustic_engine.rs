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

//! Acoustic head backend — eager CPU reference or RLX-compiled stack.

use crate::acoustic::AcousticTransformer;
use crate::acoustic_compiled::CompiledAcousticStack;
use crate::config::AudioModelArgs;
use crate::load::VoxtralTtsWeightStore;
use anyhow::Result;
use rlx_runtime::Device;
use std::collections::HashMap;

pub enum AcousticHead {
    Eager(AcousticTransformer),
    Compiled {
        outer: AcousticTransformer,
        stack: CompiledAcousticStack,
    },
}

impl AcousticHead {
    pub fn open(
        store: &VoxtralTtsWeightStore,
        prefix: &str,
        tensors: &HashMap<String, (Vec<f32>, Vec<usize>)>,
        audio: &AudioModelArgs,
        device: Device,
        eager: bool,
    ) -> Result<Self> {
        let outer = AcousticTransformer::from_tensors(
            prefix,
            tensors,
            &audio.acoustic_transformer_args,
            audio.n_acoustic_codebook,
            audio.semantic_codebook_size,
        )?;
        if eager {
            return Ok(Self::Eager(outer));
        }
        let stack = CompiledAcousticStack::open(store, &audio.acoustic_transformer_args, device)?;
        Ok(Self::Compiled { outer, stack })
    }

    pub fn predict_frame(
        &mut self,
        llm_hidden: ndarray::ArrayView1<f32>,
        cfg_alpha: f32,
        seed: u64,
        frame_index: usize,
    ) -> Result<Vec<u32>> {
        match self {
            Self::Eager(t) => t.predict_frame(llm_hidden, cfg_alpha, seed, frame_index),
            Self::Compiled { outer, stack } => {
                outer.predict_frame_compiled(llm_hidden, cfg_alpha, seed, frame_index, stack)
            }
        }
    }
}
