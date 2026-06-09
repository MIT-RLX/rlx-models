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

//! ModelFlow wrappers for encoder + projector.

use crate::config::{VoxtralAudioConfig, VoxtralConfig};
use crate::encoder::build_voxtral_encoder_built;
use crate::projector::build_voxtral_projector_built;
use anyhow::Result;
use rlx_core::weight_map::WeightMap;
use rlx_flow::{BuiltModel, CompileProfile, ModelFlow};
use rlx_ir::{DType, Shape};

pub fn build_voxtral_audio_stack_built(
    cfg: &VoxtralConfig,
    weights: &mut WeightMap,
    batch: usize,
    mel_frames: usize,
) -> Result<(BuiltModel, BuiltModel, usize)> {
    let enc_seq = cfg.audio_config.encoder_seq_len(mel_frames);
    let enc = build_voxtral_encoder_built(&cfg.audio_config, weights, batch, mel_frames)?;
    let proj = build_voxtral_projector_built(cfg, weights, batch, enc_seq)?;
    Ok((enc, proj, enc_seq))
}

pub fn build_voxtral_encoder_flow_built(
    audio_cfg: &VoxtralAudioConfig,
    weights: &mut WeightMap,
    batch: usize,
    mel_frames: usize,
) -> Result<BuiltModel> {
    let enc_seq = audio_cfg.encoder_seq_len(mel_frames);
    let f = DType::F32;
    let h = audio_cfg.d_model;
    let cfg = audio_cfg.clone();

    ModelFlow::new("voxtral_encoder_flow")
        .with_profile(CompileProfile::encoder())
        .input("mel", Shape::new(&[batch, audio_cfg.num_mel_bins, mel_frames], f))
        .plugin_named("voxtral.encoder", move |emit, _| {
            let mel = emit.flow_input("mel")?.hir_id();
            let hir = emit
                .module
                .as_hir_mut()
                .expect("voxtral encoder flow requires HIR stage");
            let mut b = crate::encoder::AudioEncoderBuilder {
                hir,
                params: emit.params,
                weights: emit.weights,
                batch,
                f: DType::F32,
            };
            let hidden = b.emit_encoder_inner(&cfg, mel, mel_frames, enc_seq)?;
            Ok(Some(emit.wrap(
                hidden,
                Shape::new(&[batch, enc_seq, h], f),
            )))
        })
        .output("encoder_hidden")
        .build(&mut rlx_core::flow_util::WeightMapSource(weights))
}
