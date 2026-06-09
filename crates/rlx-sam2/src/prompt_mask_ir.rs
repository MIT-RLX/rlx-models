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

//! SAM2 prompt-encoder mask downscale (IR).

use super::prompt_encoder::{SAM2_PROMPT_GRID, Sam2PromptEncoderWeights};
use anyhow::Result;
use rlx_flow::CompileProfile;
use rlx_runtime::Device;
use rlx_sam_ir::mask_prompt_ir::{MaskDownscaleWeights, PromptMaskCompiled};

pub struct Sam2PromptMaskCompiled(PromptMaskCompiled);

impl Sam2PromptMaskCompiled {
    pub fn compile(w: &Sam2PromptEncoderWeights, device: Device) -> Result<Self> {
        Self::compile_with_profile(w, device, &CompileProfile::sam_encoder())
    }

    pub fn compile_with_profile(
        w: &Sam2PromptEncoderWeights,
        device: Device,
        profile: &CompileProfile,
    ) -> Result<Self> {
        let in_h = 4 * SAM2_PROMPT_GRID;
        let in_w = in_h;
        let md = mask_weights(w);
        Ok(Self(PromptMaskCompiled::compile_with_profile(
            md, in_h, in_w, device, profile,
        )?))
    }

    pub fn run(&mut self, mask: &[f32]) -> Result<Vec<f32>> {
        let grid = SAM2_PROMPT_GRID;
        self.0.run(mask, 4 * grid, 4 * grid)
    }
}

fn mask_weights<'a>(w: &'a Sam2PromptEncoderWeights) -> MaskDownscaleWeights<'a> {
    MaskDownscaleWeights {
        mask_in_chans: w.mask_in_chans,
        embed_dim: w.embed_dim,
        mask_conv1_w: &w.mask_conv1_w,
        mask_conv1_b: &w.mask_conv1_b,
        mask_ln1_g: &w.mask_ln1_g,
        mask_ln1_b: &w.mask_ln1_b,
        mask_conv2_w: &w.mask_conv2_w,
        mask_conv2_b: &w.mask_conv2_b,
        mask_ln2_g: &w.mask_ln2_g,
        mask_ln2_b: &w.mask_ln2_b,
        mask_conv3_w: &w.mask_conv3_w,
        mask_conv3_b: &w.mask_conv3_b,
    }
}
